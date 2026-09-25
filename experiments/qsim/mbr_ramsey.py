# -*- coding: utf-8 -*-
"""MBRRamseyProgram: the pulse sequence every MBR job program shares.

qubit half-pi (preparation phase) -> encode an occupation string into the
cavities -> Floquet evolution -> decode a (possibly different) occupation ->
qubit half-pi (analyzer phase) -> readout. The job programs
(``MBRTimeTraceProgram``, ``MBRStarkCalProgram``, ...) subclass it and set
what they sweep.

The bases, and what each gives:

- ``QsimBaseProgram``: the M1-Sx swap parameters and Floquet waveforms
  (``retrieve_swap_parameters``, ``_initialize_floquet_pulses``), on top of
  ``MMAveragerProgram`` (reset, pulse creator, readout);
- ``FloquetTrain``: the Floquet playback and its phase bookkeeping;
- ``man_reset`` from ``ManipulateModePulses``, set explicitly (see there).

Until step 8A1 this class was built on the dark-mode chain
(``SidebandScrambleProgram``, ``DarkBaseProgram``). It used none of the
dark-mode pulses; ``tests/test_asm_golden.py`` checks that the compiled
programs did not change.

``MBRJobExperiment`` is the shared base of the job experiments: it owns
``acquire``.
"""
from copy import deepcopy

import numpy as np
from slab import AttrDict

from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.floquet_train import FloquetTrain
from experiments.qsim.manipulate_mode_pulses import ManipulateModePulses
from experiments.qsim.qsim_base import (
    QsimBaseExperiment,
    QsimBaseProgram,
    readout_lane_count,
)
from experiments.qsim.utils import ensure_list_in_cfg
from fitting.fit_display_classes import GeneralFitting


class MBRRamseyProgram(FloquetTrain, QsimBaseProgram):
    """Many-body Ramsey sequence: encode, evolve, decode, analyze.

    The shared base of the MBR job programs (docs/qsim/mbr_redesign.md,
    section 3).

    ``spectroscopy_prep_phase`` is theta on the first qubit half-pi pulse, and
    ``spectroscopy_analyzer_phase`` is phi on the final qubit half-pi pulse.
    The measured return uses theta = 0, 180 degrees and phi = 0, 90 degrees.
    With QICK's raw DDS-phase convention, ``Q_phi = Re[A exp(+i phi)]``;
    therefore the complex return is ``A = Q_0 - i Q_90``.
    A complete fixed-N basis is summed in the notebook.

    The AC Stark phase that the Floquet train builds up is removed on the
    final qubit half-pi: its phase is shifted by
    ``-floquet_cycle * final_analyzer_phase_per_cycle_deg``, the per-cycle
    phase of the decoded occupation from the StarkCal calibration set
    (``spectroscopy_phase_correction_mode='final_analyzer'``, the only mode).

    Until step 8A4 there was also a ``'decoder'`` mode: a per-pulse phase
    correction of the decoder from a measured ``decoder_phase_matrix``, and a
    ``storage_phase_matrix`` for the encoder/decoder storage swaps. Kerr
    makes that per-pulse frame differ from the final-half-pi one (a gauge
    issue), and the final half-pi method replaced it (guan, 2026-09-25). Old
    jobs taken in ``'decoder'`` mode still load and analyze; the analysis
    reads the mode from the saved config.
    """

    # The active reset plays this man reset, not MM_base's. The MBR jobs
    # always did, through DarkBaseProgram; the other two ManipulateModePulses
    # methods are dark-mode readout and are not needed here.
    man_reset = ManipulateModePulses.man_reset

    @staticmethod
    def _storage_swap_pulse_name(storage_mode,
                                 photon_number,
                                 use_multiphoton_swap=False):
        """
        Resolve one M1-storage full-swap row without changing N=1.
        If ``use_multiphoton_swap`` is False, it returns ``M1-S{storage_mode}``.
        Otherwise, it returns ``M1-S{storage_mode}@{photon_number}``
        """
        storage_mode = int(storage_mode)
        photon_number = int(photon_number)
        base_name = f"M1-S{storage_mode}"
        if not use_multiphoton_swap or photon_number == 1:
            return base_name
        return f"{base_name}@N{photon_number}"

    @staticmethod
    def _storage_mode_from_pulse_name(pulse_name):
        """
        Extract the physical storage number from M1-Sj[@Nn].
        """
        suffix = str(pulse_name).split("-S", 1)[1] # Returns j@Nn
        return int(suffix.split("@", 1)[0]) #Returns j

    @staticmethod
    def _validate_storage_swap_rows(storage_dataset,
                                    occupations,
                                    swap_stors,
                                    use_multiphoton_swap=False):
        """
        Validate every ds_storage row used by one access path.
        Added by Codex just to check whether each entry exists.
        """
        for stor, occupation in zip(swap_stors, occupations[1:]):
            if occupation == 0:
                continue
            stor_name = (MBRRamseyProgram._storage_swap_pulse_name(stor, occupation, use_multiphoton_swap,))
            if hasattr(storage_dataset, "has_row") and not storage_dataset.has_row(stor_name):
                raise RuntimeError(f"{stor_name} is missing from ds_storage; run the N-photon M1-storage calibration first")
            try:
                frequency = float(storage_dataset.get_value(stor_name, "freq (MHz)"))
                gain = float(storage_dataset.get_value(stor_name, "gain (DAC units)"))
                pi_length = float(storage_dataset.get_value( stor_name, "pi (mus)"))
            except (IndexError, KeyError) as error:
                raise RuntimeError(f"{stor_name} is missing from ds_storage; run the N-photon M1-storage calibration first") from error
            if not np.isfinite(frequency):
                raise RuntimeError(f"{stor_name} needs a finite ds_storage frequency")
            if not np.isfinite(gain) or gain <= 0:
                raise RuntimeError(f"{stor_name} needs a calibrated nonzero ds_storage gain")
            if not np.isfinite(pi_length) or pi_length <= 0:
                raise RuntimeError(f"{stor_name} needs a calibrated positive ds_storage pi length")

    @staticmethod
    def _get_encoder_pulses(occupations,
                            swap_stors,
                            use_multiphoton_swap=False):
        """
        Returning encoding pulse sequence as per `prepulse_creator2`.
        So each element in the list follows
            - output = [pulse1, pulse2, ...]
            - pulse1 = ['transition_name', 'which_transition', 'pi or hpi', relative_phase]
        The input of occupation should be n-string in the order of [n_M1] + [n_S for S in swap_stors].
        The shelving is done except for the final step, which is when last_mode and last_photon is true.
        
        If ``use_multiphoton_swap`` is True, 
        """

        occupied_modes = [(stor, occupation) for stor, occupation in zip(swap_stors, occupations[1:]) if occupation > 0] 
        if occupations[0] > 0:
            # M1 is loaded last because storage loading passes through M1.
            occupied_modes.append((0, occupations[0]))

        encoder_pulses = []
        for mode_index, (mode, photon_number) in enumerate(occupied_modes):
            last_mode = mode_index == len(occupied_modes) - 1

            for n in range(photon_number):
                last_photon = n == photon_number - 1
                last_ladder_step = last_mode and last_photon

                encoder_pulses.append(["multiphoton", f"e{n}-f{n}", "pi", 0.0,])
                if not last_ladder_step:
                    encoder_pulses.append(["qubit", "ge_broadband", "pi", 0.0,]) 
                encoder_pulses.append(["multiphoton", f"f{n}-g{n + 1}", "pi", 0.0,])
                if mode > 0 and last_photon:
                    storage_pulse_name = MBRRamseyProgram._storage_swap_pulse_name(mode,
                                                                                                         photon_number,
                                                                                                         use_multiphoton_swap,) #i don't see any point setting this as a static method tbh. 
                    encoder_pulses.append(["storage", storage_pulse_name, "pi", 0.0])
                if not last_ladder_step:
                    encoder_pulses.append(["qubit", "ge_broadband", "pi", 0.0,])

        return encoder_pulses

    @staticmethod
    def _get_inverse_pulses(encoder_pulses): 
        """
        The input is list of the pulse that follows the syntax of `prepulse_creator2`.
            - input = [pulse1, pulse2, ...]
            - pulse1 = ['transition_name', 'which_transition', 'pi or hpi', relative_phase]
        The function gives the inverse of the input pulse list, by reversing the order and 
        adding relative phase of 180 deg to each pulse.
        Note that the phase is changed by modifying pulse[3] by the aforementioned syntax
            
        """
        
        inverse_pulses = []
        for pulse in reversed(encoder_pulses):
            pulse = list(pulse)
            pulse[3] = (float(pulse[3]) + 180.0) % 360.0
            inverse_pulses.append(pulse)
        return inverse_pulses

    def _add_wait_after_storage_pulses(self, pulses):
        """
        This is to add wait after manipulate storage swap pulses, and 
        this follows the reasoning of `man_stor_swap` method in the MMbase.
        The default is 0.2 us, but this can be modified by specifying `storage_pulse_wait_us` in the 
        experimental config when the program is executed.
        """
        
        wait_us = float(self.cfg.expt.get(
            "storage_pulse_wait_us", 0.2))
        if wait_us <= 0.0:
            return pulses

        pulses_with_wait = []
        for pulse in pulses:
            pulses_with_wait.append(pulse)
            if pulse[0] == "storage":
                pulses_with_wait.append(["wait", wait_us])

        return pulses_with_wait

    def initialize(self):
        """Check the job settings, build the encoder and decoder pulse lists,
        and set up the Floquet pulses."""
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        if len(set(swap_stors)) != len(swap_stors):
            raise ValueError(f"swap_stors must be distinct; got {swap_stors}")
        if any(stor < 1 or stor > 7 for stor in swap_stors):
            raise ValueError(f"swap_stors entries must be in 1..7; got {swap_stors}")
        for key in ("decoder_phase_matrix", "storage_phase_matrix"):
            if ecfg.get(key, None) is not None:
                raise ValueError(f"{key} is no longer used: the AC Stark phase is removed on the final half-pi "
                                 "(final_analyzer_phase_per_cycle_deg)")

        if "spectroscopy_occupations" in ecfg:
            occupations = list(ecfg.spectroscopy_occupations)
        else:
            initial_mode = int(ecfg.spectroscopy_initial_mode)
            photon_number = int(ecfg.get("spectroscopy_photon_number", 1))
            if initial_mode not in [0] + swap_stors:
                raise ValueError(f"spectroscopy_initial_mode must be M1 (0) or one of swap_stors={swap_stors}; got {initial_mode}")
            occupations = [0] * (len(swap_stors) + 1)
            occupation_index = (0 if initial_mode == 0 else swap_stors.index(initial_mode) + 1)
            occupations[occupation_index] = photon_number

        if len(occupations) != len(swap_stors) + 1:
            raise ValueError(f"spectroscopy_occupations must be [n_M1] followed by the occupations of swap_stors={swap_stors}; got {occupations}")
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) for n in occupations):
            raise TypeError("spectroscopy_occupations entries must be non-negative integers")
        occupations = [int(n) for n in occupations]
        if any(n < 0 for n in occupations):
            raise ValueError("spectroscopy_occupations entries must be non-negative")

        photon_number = sum(occupations)
        if photon_number < 1:
            raise ValueError("spectroscopy_occupations must contain photons")

        use_multiphoton_swap = bool( ecfg.get("use_multiphoton_swap", False))
        ecfg.use_multiphoton_swap = use_multiphoton_swap
        final_occupations = ecfg.get("spectroscopy_final_occupations", occupations)
        if len(final_occupations) != len(occupations):
            raise ValueError("spectroscopy_final_occupations has the wrong mode count")
        if any(not isinstance(n, (int, np.integer)) for n in final_occupations):
            raise TypeError("spectroscopy_final_occupations entries must be non-negative integers")
        final_occupations = [int(n) for n in final_occupations]

        if sum(final_occupations) != photon_number:
            print("[WARNING] encoder and decoder occupations have different total photon numbers")

        self.encoder_pulses = self._get_encoder_pulses(occupations, swap_stors, use_multiphoton_swap,)
        self.decoder_encoder_pulses = self._get_encoder_pulses(final_occupations, swap_stors, use_multiphoton_swap,)

        access_pulses = (self.encoder_pulses + self.decoder_encoder_pulses)
        if any(pulse[1] == "ge_broadband" for pulse in access_pulses): #This is also Codex added stupid unnecessary safety check if the experimentalist is sober.
            pulse_key = "pi_ge_broadband"
            if pulse_key not in self.cfg.device.qubit.pulses:
                raise KeyError("This occupation-string encoder requires device.qubit.pulses.pi_ge_broadband")

            broadband_cfg = self.cfg.device.qubit.pulses[pulse_key]
            for field in ("frequency", "gain", "sigma", "length", "type"):
                if field not in broadband_cfg or np.asarray(broadband_cfg[field]).size == 0:
                    raise RuntimeError(f"device.qubit.pulses.pi_ge_broadband.{field} must contain one value")
            gain = np.asarray(broadband_cfg.get("gain", []), dtype=float).reshape(-1)
            
            if gain.size == 0 or gain[0] <= 0:
                raise RuntimeError("device.qubit.pulses.pi_ge_broadband.gain must be a configured nonzero value")

        max_local_occupation = max(max(occupations), max(final_occupations))
        multiphoton_pi = self.cfg.device.multiphoton.pi
        for transition in ("en-fn", "fn-gn+1"):
            if transition not in multiphoton_pi:
                raise KeyError(f"device.multiphoton.pi.{transition} is missing")
            
            for field in ("frequency", "gain", "length", "type", "sigma"):
                values = np.asarray(multiphoton_pi[transition].get(field, [])).reshape(-1)
                if len(values) < max_local_occupation:
                    raise RuntimeError(f"device.multiphoton.pi.{transition}.{field} needs at least {max_local_occupation} entries for spectroscopy_occupations={occupations}")

        storage_dataset = self.cfg.device.storage._ds_storage
        for path_occupations in (occupations, final_occupations):
            self._validate_storage_swap_rows(storage_dataset,path_occupations,swap_stors,use_multiphoton_swap,)

        if ecfg.get("palindrome_scramble", False) and int(ecfg.floquet_cycle) % 2:
            raise ValueError("palindrome spectroscopy uses an even number of nominal cycles; one symmetric sample is a forward/reverse pair")
        
        for flag in ("load_man_dark", "swap_man_dark", "swap_man_large_dark", "perform_wigner",
                     "init_alpha", "parity_readout", "multiparity_readout"):
            if ecfg.get(flag, False):
                raise ValueError(f"{flag}=True is incompatible with vacuum-referenced Hamiltonian spectroscopy")

        prep_phase = float(ecfg.get("spectroscopy_prep_phase", 0.0))
        analyzer_phase = float(ecfg.get("spectroscopy_analyzer_phase", 0.0))
        phase_correction_mode = str(ecfg.get("spectroscopy_phase_correction_mode", "final_analyzer"))
        if phase_correction_mode != "final_analyzer":
            raise ValueError("spectroscopy_phase_correction_mode must be 'final_analyzer'; "
                             "the 'decoder' mode was removed in MBR redesign step 8A4")

        ecfg.spectroscopy_occupations = occupations
        ecfg.spectroscopy_final_occupations = final_occupations
        ecfg.spectroscopy_prep_phase = prep_phase % 360.0
        ecfg.spectroscopy_analyzer_phase = analyzer_phase % 360.0
        ecfg.spectroscopy_phase_correction_mode = phase_correction_mode
        ecfg.final_analyzer_phase_per_cycle_deg = float(ecfg.get("final_analyzer_phase_per_cycle_deg", 0.0))
        ecfg.final_analyzer_phase_application_sign = -1.
        ecfg.spectroscopy_photon_number = photon_number
        ecfg.init_stor = 0
        ecfg.ro_stor = 0

        self.MM_base_initialize()
        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()
        self.man_mode_idx = ecfg.get("man_mode_no", 1) - 1
        self._initialize_floquet_pulses()
        self.sync_all(200)

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        phase_offsets = [0.0] * len(swap_stors)
        disorder_phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        # (|g,0> + exp(i theta)|e,0>) / sqrt(2) ->
        # (|g,0> + exp(i theta)|g,n>) / sqrt(2)
        prepulse_cfg = [["qubit", "ge", "hpi", ecfg.spectroscopy_prep_phase],] + deepcopy(self.encoder_pulses)
        prepulse_cfg = self._add_wait_after_storage_pulses(prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(cfg, prepulse.pulse, prefix="floquet_spec_pre_")
        self.sync_all()

        # U(t)|n>. The Floquet train tracks its own pulse phases in
        # phase_offsets; the detunings leave their phase in
        # disorder_phase_offsets.
        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
        )

        # Decode |n> to |e,0>, then interfere it with |g,0>. Each inverse
        # storage swap carries the phase its mode's detuning built up.
        postpulse_cfg = self._get_inverse_pulses(self.decoder_encoder_pulses)
        for pulse in postpulse_cfg:
            if pulse[0] == "storage":
                stor_index = swap_stors.index(self._storage_mode_from_pulse_name(pulse[1]))
                pulse[3] = self._mod360(pulse[3] + disorder_phase_offsets[stor_index])

        # Q_phi = Re[A exp(+i phi)], so a measured +Gamma phase is
        # removed by shifting the final analyzer by -Gamma.
        analyzer_phase = (float(ecfg.spectroscopy_analyzer_phase)
                          - int(ecfg.floquet_cycle) * float(ecfg.final_analyzer_phase_per_cycle_deg))
        postpulse_cfg.append(["qubit", "ge", "hpi", self._mod360(analyzer_phase),])
        postpulse_cfg = self._add_wait_after_storage_pulses(postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="floquet_spec_post_")
        self.sync_all()
        self.measure_wrapper()


class MBRJobExperiment(QsimBaseExperiment):
    """Base of the MBR job experiments: one job, a 2D sweep, raw shots kept.

    ``cfg.expt.swept_params`` is ``[outer, "ramsey_phase"]``: the job's own
    sweep (cycles, cycle pairs, or decoders) and, inside it, the four
    [preparation, analyzer] phase pairs. Each point compiles and runs its own
    program. ``data`` has ``xpts`` (inner values), ``ypts`` (outer values),
    ``avgi``/``avgq``/``amps``/``phases`` of shape (outer, inner), and the
    raw ``idata``/``qdata`` per point, with ``cfg.read_num`` readouts per shot.

    Subclasses set ``default_program``.
    """

    default_program = None

    def __init__(self, soccfg=None, path='', prefix=None, config_file=None,
                 expt_params=None, program=None, progress=None, **kwargs):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, expt_params=expt_params,
                         program=program or self.default_program,
                         progress=progress, **kwargs)

    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)
        ecfg = self.cfg.expt
        read_num = readout_lane_count(self.cfg)
        self.cfg.read_num = read_num

        self.outer_param, self.inner_param = ecfg.swept_params
        outer_values = ecfg[self.outer_param + "s"]
        inner_values = ecfg[self.inner_param + "s"]

        # With pre-selection, a point's average keeps only the shots whose
        # herald readout found the qubit in g.
        pre_select = ecfg.get("active_reset", False) and ecfg.get("pre_selection_reset", False)

        avgi_points, avgq_points, idata, qdata = [], [], [], []
        for outer in tqdm(outer_values, disable=not progress):
            ecfg[self.outer_param] = outer
            for inner in inner_values:
                ecfg[self.inner_param] = inner
                self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = self.prog.acquire(self.im[self.cfg.aliases.soc],
                                               threshold=None,
                                               load_pulses=True,
                                               progress=False,
                                               debug=debug,
                                               readouts_per_experiment=read_num)
                point_i, point_q = self.prog.collect_shots()
                idata.append(point_i)
                qdata.append(point_q)
                if pre_select:
                    avgi, avgq = GeneralFitting.filter_shots_per_point(
                        point_i, point_q, read_num,
                        threshold=self.cfg.device.readout.threshold[ecfg.qubits[0]],
                        pre_selection=True)
                else:
                    # The science readout is the last of the shot.
                    avgi, avgq = avgi[0][-1], avgq[0][-1]
                avgi_points.append(avgi)
                avgq_points.append(avgq)

        shape = (len(outer_values), len(inner_values))
        avgi = np.reshape(np.array(avgi_points), shape)
        avgq = np.reshape(np.array(avgq_points), shape)
        self.data = dict(
            avgi=avgi, avgq=avgq,
            amps=np.abs(avgi + 1j * avgq),
            phases=np.angle(avgi + 1j * avgq),
            idata=idata, qdata=qdata,
            xpts=inner_values, ypts=outer_values,
        )
        return self.data

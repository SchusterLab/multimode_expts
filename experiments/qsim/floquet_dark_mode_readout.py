# -*- coding: utf-8 -*-
import importlib

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.signal import find_peaks
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.qsim import matrix_pencil as matrix_pencil_analysis
from fitting.qsim import level_statistics as level_statistics_analysis
from fitting.qsim import mbr_spectrum as mbr_spectrum_analysis
from fitting.qsim import mbr_phase as mbr_phase_analysis
from fitting.fit_display_classes import (
    GeneralFitting,
    RamseyFitting,
)
from experiments.MM_base import *
from experiments.characterization_runner import CharacterizationRunner
from experiments.floquet_timing import FLUX_HIGH_THRESHOLD_MHZ
from experiments.qsim.qsim_base import *
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display import *

from experiments.qsim.kerr import *

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.sideband_scramble import SidebandScrambleProgram
from experiments.qsim.dark_base import (
    DarkBaseExperiment,
    DarkBaseProgram,
    DarkBaseRProgram,
    classify_two_parity_readouts,
)
from experiments.qsim.floquet_phase_frame import (
    advance_floquet_offsets,
    advance_matrix_offsets,
    detuning_phase_deg,
    mod360,
)
from experiments.qsim.floquet_register_bank import (
    _play_preloaded_floquet_register_bank_entry,
    _prepare_preloaded_floquet_register_bank,
)
from experiments.qsim.utils import flatten_exp_lists

from copy import copy, deepcopy
from itertools import product

from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view

class SidebandScrambleDarkProgramNewNew(SidebandScrambleProgram, DarkBaseProgram):
    # MRO: this -> SidebandScrambleProgram -> DarkBaseProgram -> QsimBaseProgram
    # Dark load, scramble, and readout share one mutable phase_offsets list.

    def _prepare_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        """
        Dispatch dark-mode load without modifying existing dark helpers.
        """
        if self.cfg.expt.get("swap_man_large_dark", False):
            self._prepare_large_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )
        else:
            self._prepare_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )

    def _read_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        """
        Dispatch dark-mode readout without modifying existing dark helpers.
        """
        if self.cfg.expt.get("swap_man_large_dark", False):
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with four supports")
            self._read_large_dark(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )
        else:
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with two supports")
            self._read_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )

    def core_pulses(self):
        swap_stors = list(self.cfg.expt.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)
        disorder_phase_offsets = [0.0] * len(swap_stors)

        # 1. Optional load:
        # M1/man excitation -> selected dark/normal mode.
        #
        # This mutates phase_offsets.  Those offsets must be the initial
        # frame for the following scramble.
        if self.cfg.expt.get("load_man_dark", False):
            self._prepare_selected_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )

        # 2. Scramble with the same live phase tracker.
        #
        # Do NOT call super().core_pulses() here.  That method creates a
        # fresh local swap_stor_phases = [0, 0, ...] and physically emits
        # the scramble with the wrong initial frame after dark load.
        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
        )

        if not self.cfg.expt.get("swap_man_dark", False):
            return

        # 3. Readout with the final tracker.
        #
        # Do NOT call _accumulate_scramble_phases() here.  The actual scramble
        # above already updated phase_offsets pulse-by-pulse.  Calling
        # _accumulate_scramble_phases() again would double-count.
        self._read_selected_dark_mode(
            phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )


class NPhotonHamiltonianSpectroscopyProgram(
        SidebandScrambleDarkProgramNewNew):
    """Measure the complex diagonal return for one occupation string.

    ``spectroscopy_prep_phase`` is theta on the first qubit half-pi pulse, and
    ``spectroscopy_analyzer_phase`` is phi on the final qubit half-pi pulse.
    The measured return uses theta = 0, 180 degrees and phi = 0, 90 degrees.
    With QICK's raw DDS-phase convention, ``Q_phi = Re[A exp(+i phi)]``;
    therefore the complex return is ``A = Q_0 - i Q_90``.
    A complete fixed-N basis is summed in the notebook.

    ``decoder_phase_matrix[row, column]`` tracks decoder control axes. Row 0
    is the common f_n-g_(n+1) axis. The remaining rows are the M1-storage
    decoder axes in ``swap_stors`` order. Columns are the physical Floquet
    pulses, also in ``swap_stors`` order.

    Every played Floquet pulse advances all measured decoder-axis slopes.
    During decoding the accumulated M1 slope is subtracted from every inverse
    f_n-g_(n+1) pulse.  The storage rows are defined as
    ``mode_path_storage - mode_path_M1`` and are added to the inverse
    M1-storage pulses because those pulse phases enter the recovered return
    amplitude with the opposite sign.

    ``spectroscopy_phase_correction_mode='decoder'`` keeps this original
    decoder-pulse correction.  ``'final_analyzer'`` skips the decoder matrix
    and subtracts
    ``floquet_cycle * final_analyzer_phase_per_cycle_deg`` from the final
    qubit half-pi instead.
    """

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
            stor_name = (NPhotonHamiltonianSpectroscopyProgram._storage_swap_pulse_name(stor, occupation, use_multiphoton_swap,))
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
                    storage_pulse_name = NPhotonHamiltonianSpectroscopyProgram._storage_swap_pulse_name(mode,
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
        """
        The primary purpose of overriding `initialize` method is for the initialization of phase calibration matrix.
        For now, `storage_phase_matrix` is not really being used, as the phase accumulating during floquet cycle matters.
        There are two mode of calibration:  `decoder` and `final_analyzer`
        1. `decoder`
        - uses `decoder_phase_matrix` for the calibration of ac stark shift on every sideband transition including fn_gn+1.
        2. `final_analyzer`
        - uses `expt.cfg.final_analyzer_phase_per_cycle_deg` at the final hpi to calibrate ac stark shift out collectively.
        """
        
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        if len(set(swap_stors)) != len(swap_stors):
            raise ValueError(f"swap_stors must be distinct; got {swap_stors}")
        if any(stor < 1 or stor > 7 for stor in swap_stors):
            raise ValueError(f"swap_stors entries must be in 1..7; got {swap_stors}")

        matrix_shape = (len(swap_stors), len(swap_stors))
        storage_phase_matrix = ecfg.get("storage_phase_matrix", None)
        if storage_phase_matrix is not None:
            storage_phase_matrix = np.asarray(
                storage_phase_matrix, dtype=float)
            if storage_phase_matrix.shape != matrix_shape:
                raise ValueError(f"storage_phase_matrix must have shape {matrix_shape}; got {storage_phase_matrix.shape}")
            ecfg.storage_phase_matrix = storage_phase_matrix

        decoder_matrix_shape = (len(swap_stors) + 1,len(swap_stors),)
        
        decoder_phase_matrix = ecfg.get("decoder_phase_matrix", None)
        self.decoder_phase_matrix_is_calibrated = decoder_phase_matrix is not None

        if decoder_phase_matrix is None:
            self.decoder_phase_matrix = np.zeros(decoder_matrix_shape)
        else:
            decoder_phase_matrix = np.asarray(decoder_phase_matrix,dtype=float,)
            if decoder_phase_matrix.shape != decoder_matrix_shape:
                raise ValueError(f"decoder_phase_matrix must have shape {decoder_matrix_shape}; got {decoder_phase_matrix.shape}")
            self.decoder_phase_matrix = decoder_phase_matrix.copy()

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
        
        for flag in ("load_man_dark", "swap_man_dark", "swap_man_large_dark","perform_wigner", "parity_readout", "multiparity_readout"):
            if ecfg.get(flag, False):
                raise ValueError(f"{flag}=True is incompatible with vacuum-referenced Hamiltonian spectroscopy")

        prep_phase = float(ecfg.get("spectroscopy_prep_phase", 0.0))
        analyzer_phase = float(ecfg.get("spectroscopy_analyzer_phase", 0.0))
        phase_correction_mode = str(ecfg.get("spectroscopy_phase_correction_mode", "decoder"))
        if phase_correction_mode not in ("decoder", "final_analyzer"):
            raise ValueError("spectroscopy_phase_correction_mode must be 'decoder' or 'final_analyzer'")

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
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        phase_correction_mode = ecfg.spectroscopy_phase_correction_mode
        update_decoder_phases = (ecfg.get("update_phases", True) and phase_correction_mode == "decoder")
        storage_phase_offsets = [0.0] * len(swap_stors)
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
        encoder_pulses = deepcopy(self.encoder_pulses)
        for pulse in encoder_pulses:
            if pulse[0] != "storage":
                continue

            stor = self._storage_mode_from_pulse_name(pulse[1])
            stor_index = swap_stors.index(stor)
            pulse[3] = self._mod360(pulse[3] + storage_phase_offsets[stor_index])
            self._advance_storage_phase_offsets(phase_offsets=storage_phase_offsets,swap_stors=swap_stors,pulsed_stor=stor)

        prepulse_cfg = [["qubit", "ge", "hpi", ecfg.spectroscopy_prep_phase],] + encoder_pulses
        prepulse_cfg = self._add_wait_after_storage_pulses(prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(cfg, prepulse.pulse, prefix="floquet_spec_pre_")
        self.sync_all()

        decoder_phase_deg = [0.0] * (len(swap_stors) + 1)
        if update_decoder_phases and not self.decoder_phase_matrix_is_calibrated:
            raise RuntimeError("decoder_phase_matrix is missing. Run the exact-path Floquet phase calibration before spectroscopy.")

        # U(t)|n>. Each physical Floquet pulse advances the measured decoder
        # phase slopes in decoder_phase_deg. The inverse decoder later uses
        # the opposite sign to cancel those slopes.
        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
            decoder_phase_offsets=(decoder_phase_deg if update_decoder_phases else None),
        )

        # Decode |n> to |e,0>, then interfere it with |g,0>.
        postpulse_cfg = self._get_inverse_pulses(self.decoder_encoder_pulses)
        for pulse in postpulse_cfg:
            # Every f_n-g_(n+1) pulse transfers one M1 photon, so all n use
            # the same M1-frame correction. N ladder steps then give N times
            # that phase without an explicit photon-number multiplier.
            if pulse[0] == "multiphoton" and pulse[1].startswith("f") and "-g" in pulse[1]:
                pulse[3] = self._mod360(pulse[3] - decoder_phase_deg[0])

            elif pulse[0] == "storage":
                stor = self._storage_mode_from_pulse_name(pulse[1])
                stor_index = swap_stors.index(stor)
                pulse[3] = self._mod360(
                    pulse[3]
                    + storage_phase_offsets[stor_index]
                    + decoder_phase_deg[stor_index + 1]
                    + disorder_phase_offsets[stor_index]
                )
                self._advance_storage_phase_offsets(
                    phase_offsets=storage_phase_offsets,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )

        analyzer_phase = float(ecfg.spectroscopy_analyzer_phase)
        if phase_correction_mode == "final_analyzer":
            # Q_phi = Re[A exp(+i phi)], so a measured +Gamma phase is
            # removed by shifting the final analyzer by -Gamma.
            analyzer_phase -= (int(ecfg.floquet_cycle)* float(ecfg.final_analyzer_phase_per_cycle_deg))

        postpulse_cfg.append(["qubit", "ge", "hpi",self._mod360(analyzer_phase),])
        postpulse_cfg = self._add_wait_after_storage_pulses(postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="floquet_spec_post_")
        self.sync_all()
        self.measure_wrapper()



class EncodingOrthogonalityProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure coherent cross-return amplitudes between encoder paths.

    One job fixes ``spectroscopy_occupations`` (the encoded column). Its outer
    software sweep packs decoder occupation and analyzer phase into
    ``decoder_analyzer_row``; the inner sweep is the usual preparation phase
    ``0/180``. Only zero Floquet cycles are accepted, so this probes the access
    paths rather than Floquet time evolution.
    """
    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        decoder_occupations = [list(occupation) for occupation in ecfg.orthogonality_decoder_occupations]
        analyzer_phases = ecfg.orthogonality_analyzer_phases
        decoder_analyzer_row = int(ecfg.decoder_analyzer_row)
        decoder_index = decoder_analyzer_row // 2
        analyzer_phase_index = decoder_analyzer_row % 2
        decoder_occupation = decoder_occupations[decoder_index]
        ecfg.spectroscopy_analyzer_phase = float(
            analyzer_phases[analyzer_phase_index])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.
        ecfg.floquet_cycle = 0
        ecfg.spectroscopy_final_occupations = decoder_occupation

        super().initialize()

    def _get_inverse_pulses(self, _):
        """
        Overrides ``_get_inverse_pulses`` in the parent ``NPhotonHamiltonianSpectroscopyProgram``
        for a given decoder_occupation

        """

        # Parent body asks for inverse(encoder); use the selected decoder here.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)


class EncodingPropagatorProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure one raw column of the short-time propagator."""

    def initialize(self):
        ecfg = self.cfg.expt
        cycle_decoder_analyzer = list(ecfg.cycle_decoder_analyzer)
        decoder_occupation = list(cycle_decoder_analyzer[1:-1])

        ecfg.floquet_cycle = int(cycle_decoder_analyzer[0])
        ecfg.spectroscopy_analyzer_phase = float(
            cycle_decoder_analyzer[-1])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.
        ecfg.spectroscopy_final_occupations = decoder_occupation

        super().initialize()

    def _get_inverse_pulses(self, _):
        # The parent body requests inverse(encoder); decode the selected row.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)


class EncodingStarkShiftCalibrationProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure the raw Floquet-pulse phase of one occupation basis state.

    ``spectroscopy_occupations`` selects the encoded occupation string and
    ``stor_B`` selects the physical M1-storage Floquet pulse.  An even number
    of pulses is played with alternating 0/180-degree phase, so the intended
    exchange closes while any repeatable diagonal phase remains measurable.

    No previously measured decoder or ds_storage phase correction is applied
    here.  The fixed encoder/decoder access phase is removed in the notebook
    by dividing the complex return by its zero-pulse value.  A nonzero
    ``final_analyzer_phase_per_pulse_deg`` is multiplied by ``n_pulse`` and
    subtracted only from the final qubit half-pi for an end-to-end sign check.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        n_pulse = int(ecfg.n_pulse)

        if stor_B not in swap_stors:
            raise ValueError(
                f"stor_B must be one of {swap_stors}; got {stor_B}"
            )
        if n_pulse < 0 or n_pulse % 2:
            raise ValueError(
                "n_pulse must be a non-negative even integer so the "
                "alternating Floquet train closes before decoding"
            )
        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(
                "EncodingStarkShiftCalibrationProgram requires "
                "spectroscopy_occupations"
            )

        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        ecfg.final_analyzer_phase_per_pulse_deg = float(ecfg.get(
            "final_analyzer_phase_per_pulse_deg", 0.0
        ))
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.update_phases = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        prepulse_cfg = [[
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_prep_phase),
        ]] + deepcopy(self.encoder_pulses)
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="encoding_stark_pre_")
        self.sync_all()

        self._play_m1s_frac_train(
            stor=stor_B,
            n_frac=int(ecfg.n_pulse),
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            logical_phase_deg=0.0,
            logical_phase_step_deg=180.0,
            update_phases=False,
            label="occupation-resolved encoding Stark calibration",
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        analyzer_phase = (
            float(ecfg.spectroscopy_analyzer_phase)
            # The same Q_phi convention as spectroscopy is used here.
            - int(ecfg.n_pulse)
            * float(ecfg.final_analyzer_phase_per_pulse_deg)
        )

        postpulse_cfg.append([
            "qubit", "ge", "hpi",
            self._mod360(analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="encoding_stark_post_")
        self.sync_all()
        self.measure_wrapper()


class EntireFloquetCyclePhaseCalibrationProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure one occupation's phase from the entire Floquet cycle.

    One closed pair is

        ordered Floquet cycle -> reverse-order inverse Floquet cycle.

    The inverse cycle uses the same tracked logical axes as spectroscopy and
    adds 180 degrees to every beam-splitter pulse.  It therefore cancels the
    intended exchange motion while repeatable diagonal phase can accumulate.
    ``n_cycle_pair`` pairs contain ``2 * n_cycle_pair`` physical entire
    cycles.  The notebook fits against that physical-cycle count, so its
    slope is directly in degrees per entire Floquet cycle.

    ``final_analyzer_phase_per_cycle_deg`` is used only for the end-to-end
    sign check.  It is multiplied by ``2 * n_cycle_pair`` and subtracted from
    the final qubit half-pi.

    ``n_physical_cycle`` also permits odd guide points.  An odd point plays
    all complete forward/inverse pairs followed by one forward cycle.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        if ecfg.get("phase_unwrap_mode", "pair") == "odd_guide":
            n_physical_cycle = int(ecfg.n_physical_cycle)
        else:
            n_physical_cycle = 2 * int(ecfg.n_cycle_pair)

        if n_physical_cycle < 0:
            raise ValueError("n_physical_cycle must be non-negative")
        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(
                "EntireFloquetCyclePhaseCalibrationProgram requires "
                "spectroscopy_occupations"
            )
        if ecfg.get("floquet_hardware_loop", False):
            raise ValueError(
                "The exact multi-mode forward/inverse calibration currently "
                "uses the software-emitted pulse sequence; set "
                "floquet_hardware_loop=False"
            )

        detunings = ecfg.get("detunings", None)
        if detunings is not None and detunings is not False \
                and np.asarray(detunings).size > 0 \
                and not np.allclose(detunings, 0.0):
            raise ValueError(
                "Entire-cycle phase calibration uses zero detuning.  "
                "Disorder is part of the target Hamiltonian and must not be "
                "calibrated out."
            )

        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        ecfg.final_analyzer_phase_per_cycle_deg = float(ecfg.get(
            "final_analyzer_phase_per_cycle_deg", 0.0
        ))
        ecfg.zero_floquet_gain = bool(ecfg.get(
            "zero_floquet_gain", False
        ))
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.n_physical_cycle = n_physical_cycle
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        n_physical_cycle = int(ecfg.n_physical_cycle)
        n_cycle_pair = n_physical_cycle // 2
        phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        prepulse_cfg = [[
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_prep_phase),
        ]] + deepcopy(self.encoder_pulses)
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="entire_cycle_phase_pre_")
        self.sync_all()

        self._play_closed_floquet_cycle_pairs(
            n_cycle_pair=n_cycle_pair,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            extra_forward=n_physical_cycle % 2,
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        analyzer_phase = (
            float(ecfg.spectroscopy_analyzer_phase)
            - n_physical_cycle
            * float(ecfg.final_analyzer_phase_per_cycle_deg)
        )
        postpulse_cfg.append([
            "qubit", "ge", "hpi", self._mod360(analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="entire_cycle_phase_post_")
        self.sync_all()
        self.measure_wrapper()


class SinglePhotonFloquetSpectroscopyProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Backward-compatible wrapper for the original N=1 program name."""

    def initialize(self):
        ecfg = self.cfg.expt
        photon_number = int(
            ecfg.get("spectroscopy_photon_number", 1))
        if photon_number != 1:
            raise ValueError(
                "SinglePhotonFloquetSpectroscopyProgram only supports N=1; "
                "use NPhotonHamiltonianSpectroscopyProgram for arbitrary N"
            )
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        initial_mode = int(ecfg.get("spectroscopy_initial_mode", 0))
        if initial_mode not in [0] + swap_stors:
            raise ValueError(
                f"spectroscopy_initial_mode={initial_mode} is not in "
                f"[0] + swap_stors={swap_stors}"
            )
        occupations = [0] * (len(swap_stors) + 1)
        occupation_index = (
            0 if initial_mode == 0 else swap_stors.index(initial_mode) + 1
        )
        occupations[occupation_index] = 1
        ecfg.spectroscopy_occupations = occupations
        ecfg.spectroscopy_photon_number = 1
        super().initialize()


class FloquetPhaseAccumulationProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure the phase of one closed N=1 access path.

    ``spectroscopy_occupations`` selects M1 or one storage access path and
    ``stor_B`` selects one physical Floquet pulse column. ``n_pulse`` copies
    of that pulse are played with alternating 0/180-degree drive phase.
    ``spectroscopy_prep_phase`` is theta on the initial qubit half-pi and
    ``spectroscopy_analyzer_phase`` is phi on the final qubit half-pi, exactly
    as in active spectroscopy. Their phase cycle reconstructs the complex
    return.

    The M1 path gives the common f_n-g_(n+1) row. A storage path contains that
    common row plus its M1-storage row, so the notebook subtracts the measured
    M1 path before constructing ``decoder_phase_matrix``.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        if stor_B not in swap_stors:
            raise ValueError(
                f"stor_B must be one of {swap_stors}; got {stor_B}"
            )

        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(
                "FloquetPhaseAccumulationProgram requires the exact "
                "spectroscopy_occupations row"
            )
        if int(ecfg.n_pulse) % 2:
            raise ValueError(
                "n_pulse must be even so the alternating Floquet train "
                "closes before decoding"
            )

        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        phase_offsets = [0.0] * len(swap_stors)
        storage_phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        encoder_pulses = deepcopy(self.encoder_pulses)
        for pulse in encoder_pulses:
            if pulse[0] != "storage":
                continue

            stor = self._storage_mode_from_pulse_name(pulse[1])
            stor_index = swap_stors.index(stor)
            pulse[3] = self._mod360(
                pulse[3] + storage_phase_offsets[stor_index]
            )
            self._advance_storage_phase_offsets(
                phase_offsets=storage_phase_offsets,
                swap_stors=swap_stors,
                pulsed_stor=stor,
            )

        prepulse_cfg = [
            [
                "qubit", "ge", "hpi",
                float(ecfg.spectroscopy_prep_phase),
            ],
        ] + encoder_pulses
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="floquet_phase_pre_")
        self.sync_all()

        # The notebook uses even pulse counts so every 0/180 pair closes the
        # intended beam-splitter transfer before its complex phase is read.
        self._play_m1s_frac_train(
            stor=stor_B,
            n_frac=int(ecfg.n_pulse),
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            logical_phase_deg=0.0,
            logical_phase_step_deg=180.0,
            update_phases=False,
            label="exact-path Floquet phase calibration",
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        for pulse in postpulse_cfg:
            if pulse[0] != "storage":
                continue

            stor = self._storage_mode_from_pulse_name(pulse[1])
            stor_index = swap_stors.index(stor)
            pulse[3] = self._mod360(
                pulse[3] + storage_phase_offsets[stor_index]
            )
            self._advance_storage_phase_offsets(
                phase_offsets=storage_phase_offsets,
                swap_stors=swap_stors,
                pulsed_stor=stor,
            )

        postpulse_cfg.append([
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="floquet_phase_post_")
        self.sync_all()
        self.measure_wrapper()


class SidebandScrambleDarkProgramNew(SidebandScrambleProgram, DarkBaseProgram):
    # MRO: this -> SidebandScrambleProgram -> DarkBaseProgram -> QsimBaseProgram
    # so super().core_pulses() plays the scrambling pulses, while the dark-mode
    # helpers (_read_dark_mode, _accumulate_scramble_phases, man_reset, ...) are
    # inherited from DarkBaseProgram.

    def core_pulses(self):
        swap_stors = list(self.cfg.expt.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)

        if self.cfg.expt.get("load_man_dark", False):
            self._prepare_large_dark_mode(phase_offsets)

        super().core_pulses()  # SidebandScrambleProgram.core_pulses(): plays scrambling

        if not self.cfg.expt.get("swap_man_dark", False):
            return

        # Replay the phase bookkeeping in the calibrated frame to match what
        # the (just-played) scrambling left behind. SidebandScrambleProgram
        # keeps its phase tracker local, so we reconstruct it here.
        self._accumulate_scramble_phases(phase_offsets, swap_stors)
        if self.cfg.expt.get("swap_man_dark", False) and not self.cfg.expt.get("swap_man_large_dark", False):
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with two supports")
            # Map the selected dark/normal mode back into M1.
            self._read_dark_mode(phase_offsets)
        elif self.cfg.expt.get("swap_man_large_dark", False):
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with four supports")
            self._read_large_dark(phase_offsets)


class ManStorScrambleProgram(SidebandScrambleProgram, DarkBaseProgram):
    # MRO: this -> SidebandScrambleProgram -> DarkBaseProgram -> QsimBaseProgram
    # so super().core_pulses() plays the scrambling pulses, while the dark-mode
    # helpers (_read_dark_mode, _accumulate_scramble_phases, man_reset, ...) are
    # inherited from DarkBaseProgram.

    def core_pulses(self):
        self.sync_all()
        swap_stor = self.cfg.expt.swap_stor
        _pulse_cfg = [ ['storage', f'M1-S{swap_stor}', 'pi', 0,] ] 
        _pulse_creator = self.get_prepulse_creator(_pulse_cfg)
        _pulse = _pulse_creator.pulse
        _pulse[2][0] = self.cfg.expt.length
        if self.cfg.expt.get("custom_scramble_gain", None) is not None:
            _pulse[1][0] = self.cfg.expt.custom_scramble_gain
        if self.cfg.expt.get("custom_scramble_freq", None) is not None:
            _pulse[0][0] = self.cfg.expt.custom_scramble_freq
        if self.cfg.expt.get("custom_scramble_phase", None) is not None:
            _pulse[3][0] = self.cfg.expt.custom_scramble_phase
        
        self.custom_pulse(self.cfg, _pulse, prefix = 'swap_pulse___manstorscram')
        self.sync_all(0.2)



class SidebandScrambleDarkProgramDebug(SidebandScrambleDarkProgramNewNew):
    # Debug variant for repeated load/readout checks.

    def _prepare_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        return super()._prepare_selected_dark_mode(
            phase_offsets=phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def _read_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        return super()._read_selected_dark_mode(
            phase_offsets=phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def core_pulses(self):
        swap_stors = list(self.cfg.expt.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)

        # 1. Optional load:
        # M1/man excitation -> selected dark/normal mode.
        #
        # This mutates phase_offsets.  Those offsets must be the initial
        # frame for the following scramble.
        for _ in range(self.cfg.expt.get("number_of_load_unload")):
            if self.cfg.expt.get("debug", False):
                print(
                    "doing debugging with "
                    f"{self.cfg.expt.get('number_of_load_unload')}"
                )
            self._prepare_selected_dark_mode(phase_offsets)
            self._read_selected_dark_mode(phase_offsets)
        # if self.cfg.expt.get("load_man_dark", False):
        #     self._prepare_selected_dark_mode(phase_offsets)

        # # 2. Scramble with the same live phase tracker.
        # #
        # # Do NOT call super().core_pulses() here.  That method creates a
        # # fresh local swap_stor_phases = [0, 0, ...] and physically emits
        # # the scramble with the wrong initial frame after dark load.
        # self._play_scramble_with_phase_offsets(
        #     phase_offsets=phase_offsets,
        #     swap_stors=swap_stors,
        # )

        # if not self.cfg.expt.get("swap_man_dark", False):
        #     return

        # # 3. Readout with the final tracker.
        # #
        # # Do NOT call _accumulate_scramble_phases() here.  The actual scramble
        # # above already updated phase_offsets pulse-by-pulse.  Calling
        # # _accumulate_scramble_phases() again would double-count.
        # self._read_selected_dark_mode(phase_offsets)



class KerrWaitProgramDark(DarkBaseProgram):
    def core_pulses(self):
        # print("Adding man-dump pulse")
        # self.man_reset(man_idx=1, dump_mode_idx=2, chi_dressed=True)

        self.sync_all(self.us2cycles(self.cfg.expt.wait_us_time))

'''
============================================================================
============================================================================
============================================================================
========   OLD PROGRAMS: TO BE DELETED AFTER SOME TIME======================
============================================================================
============================================================================
============================================================================
'''
class SidebandScrambleDarkProgram(SidebandScrambleProgram):
    
    def man_reset(self, man_idx=1, dump_mode_idx=2, chi_dressed=True):
        '''
        Reset manipulate mode by swapping it to lossy mode

        chi_dressed: if man freq shifted due to pop in qubit e, f states.
        using_qubit: if True, we do g1-f0/ef/qubit reset instead of using the dump, which is not indeal since it remove only the fock 1 population but can be usefull if dump cannot be found 
        '''
        if self.cfg.expt.get("debug", False):
            print("overrided man reset is called")
        qTest = 0
        cfg=AttrDict(self.cfg)

        MiDj_freq = self.dataset.get_freq(f'M{man_idx}-D{dump_mode_idx}')
        MiDj_gain = self.dataset.get_gain(f'M{man_idx}-D{dump_mode_idx}')
        MiDj_length = self.dataset.get_pi(f'M{man_idx}-D{dump_mode_idx}')
        N = 2 if chi_dressed else 0
        chi_ge = cfg.device.manipulate.chi_ge[qTest]
        chi_ef = cfg.device.manipulate.chi_ef[qTest]

        self.sideband_sigma_high = self.us2cycles(self.cfg.device.storage.ramp_sigma, gen_ch=self.flux_high_ch[qTest])
        self.add_gauss(ch=self.flux_high_ch[qTest],
                    name="ramp_high",# + str(man_idx),
                    sigma=self.sideband_sigma_high,
                    length=self.sideband_sigma_high*6) # M1-x flat tops use 6 sigma
        # self.wait_all(self.us2cycles(0.1))
        self.sync_all(self.us2cycles(0.1))

        chis = [chi_ge, chi_ge+chi_ef] if chi_dressed else [0]
        ch = self.flux_high_ch[qTest]
        iter_num = self.cfg.expt.get("dump_reset_iter_num", 1)
        for n in range(0, N+1): # works when MiDj freq goes down (chi<0, bare freq+chi*n)
            for chi in chis:
                for _ in range(iter_num):
                    freq_chi_shifted = MiDj_freq + (n * chi)
                    # if cfg.expt.get("man_reset_print", True):
                    #     print(ch, freq_chi_shifted, MiDj_length, MiDj_gain)
                    self.set_pulse_registers(
                        ch=ch,
                        freq=self.freq2reg(freq_chi_shifted, gen_ch=ch),
                        style="flat_top",
                        phase=self.deg2reg(0),
                        length=self.us2cycles(MiDj_length, gen_ch=ch),
                        gain=MiDj_gain,
                        waveform="ramp_high"
                        )
                    self.pulse(ch=ch)
                    self.sync_all()
                # self.sync_all(self.us2cycles(0.025))
        # self.wait_all(self.us2cycles(0.25))
        self.sync_all(self.us2cycles(2))
    
    
    def core_pulses(self):
        super().core_pulses() #already has sync_all at the last
        if self.cfg.expt.get("swap_man_dark", False):
            swap_stors = self.cfg.expt.swap_stors
            swap_stor_phases = [0.0] * len(swap_stors)

            if self.cfg.expt.update_phases:
                for _ in range(self.cfg.expt.floquet_cycle):
                    for i_stor, stor in enumerate(swap_stors):
                        for j_stor, stor_B in enumerate(swap_stors):
                            if stor_B != stor:
                                stor_B_name = f"M1-S{stor_B}"
                                stor_name = f"M1-S{stor}"
                                swap_stor_phases[j_stor] += self.swap_ds.get_phase_from(stor_B_name, stor_name)
                                swap_stor_phases[j_stor] = swap_stor_phases[j_stor] % 360
            
            stor_first, stor_last = self.cfg.expt.dark_swap_order
            list_index_start = swap_stors.index(stor_first)
            list_index_last = swap_stors.index(stor_last)

            n_first = self.m1s_pi_fracs[stor_first - 1] 
            n_last = self.m1s_pi_fracs[stor_last - 1] // 2
            first_stor_name = f"M1-S{stor_first}"
            last_stor_name = f"M1-S{stor_last}"

            if self.cfg.expt.get("second_rel_phase", 0) != 0:
                swap_stor_phases[list_index_last] += self.cfg.expt.second_rel_phase
                swap_stor_phases[list_index_last] = swap_stor_phases[list_index_last] % 360

            first_pulse_args = deepcopy(self.m1s_kwargs[stor_first - 1])
            second_pulse_args = deepcopy(self.m1s_kwargs[stor_last - 1])

            for _ in range(n_first): # full swap and phase update
                first_pulse_args['phase'] = self.deg2reg(swap_stor_phases[list_index_start], gen_ch=first_pulse_args['ch'])
                self.setup_and_pulse(**first_pulse_args)
                swap_stor_phases[list_index_last] += self.swap_ds.get_phase_from(last_stor_name, first_stor_name)
                swap_stor_phases[list_index_last] = swap_stor_phases[list_index_last] % 360
                self.sync_all(10)

            for _ in range(n_last):
                second_pulse_args['phase'] = self.deg2reg(swap_stor_phases[list_index_last], gen_ch=second_pulse_args['ch'])
                self.setup_and_pulse(**second_pulse_args)
                self.sync_all(10)
            
            self.sync_all()
            
            


class BatchRunner(CharacterizationRunner):
    """CharacterizationRunner with bounded parallel queue submission."""

    @staticmethod
    def _plain(obj):
        """
        Convert config values only at the queue's JSON boundary.
        Non JSON compatible objects are converted into compatible ones.
        """
        if isinstance(obj, dict):
            return {key: BatchRunner._plain(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BatchRunner._plain(value) for value in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def execute(self,
                  configs, 
                  batch_size=10, 
                  postprocess=True, 
                  priority=0,
                  poll_interval=2., 
                  timeout=None, 
                  log=None,
                  show=None):
        """Submit at most batch_size jobs, then collect them in config order."""
        if self.job_client is None:
            raise ValueError("job_client is required")
        if (isinstance(batch_size, (bool, np.bool_))
                or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")

        configs = list(configs) #list of config dictionary that is overrided in the submitted job
        if not configs:
            raise ValueError("configs cannot be empty")
        expts = []
        self.last_job_ids = []
        program_module = None
        program_class = None
        if self.program is not None:
            program_module = self.program.__module__
            program_class = self.program.__name__

        for start in range(0, len(configs), batch_size):
            pending = []
            batch_configs = [self.preprocessor(self.station, self.default_expt_cfg, **overrides) for overrides in configs[start:start + batch_size]]
            station_config = self._serialize_station_config()
            print(f"batch {start // batch_size + 1}: {len(batch_configs)} jobs")
            try:
                for cfg in batch_configs:
                    job_id = self.job_client.submit_job(
                        experiment_class=self.ExptClass.__name__,
                        experiment_module=self.ExptClass.__module__,
                        expt_config=self._plain(dict(cfg)), 
                        station_config=station_config,
                        user=self.station.user, 
                        priority=priority,
                        program_class=program_class, 
                        program_module=program_module,
                    )
                    pending.append(job_id)
                    self.last_job_ids.append(job_id)

                for job_id in list(pending):
                    result = self.job_client.wait_for_completion(
                        job_id, poll_interval=poll_interval, timeout=timeout, verbose=False)
                    pending.pop(0)
                    self.last_job_result = result
                    if not result.is_successful():
                        raise RuntimeError(
                            f"Job {job_id} {result.status}: {result.error_message or 'No details'}")
                    expt = result.load_expt()
                    if postprocess:
                        self.postprocessor(self.station, expt)
                    self._render_log_show(expt, show=show, log=log, display_kwargs=None)
                    expts.append(expt)
            except BaseException: #BaseException is inherited by Exception, KeyboardInterrupt, SystemExit, GeneratorExit
                # Below is to cancel the running job when there is KeyboardInterrupt
                for job_id in pending:
                    try:
                        self.job_client.cancel_job(job_id)
                    except Exception:
                        pass
                raise
        if self.program is not None:
            batch_expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=f"{self.ExptClass.__name__}_batch",
                config_file=self.station.hardware_config_file,
                program=self.program,
            )
        else:
            batch_expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=f"{self.ExptClass.__name__}_batch",
                config_file=self.station.hardware_config_file,
            )
        batch_expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))
        batch_expt.cfg.expt = deepcopy(self.default_expt_cfg)
        batch_expt.data = AttrDict()
        batch_expt.batch_expts = expts
        batch_expt.batch_job_ids = list(self.last_job_ids)
        batch_expt._analysis_station = self.station
        return batch_expt


class EncodingHamiltonianSpectroscopyExperiment(DarkBaseExperiment):
    """Per-job and aggregate analysis for encoding-calibrated spectroscopy."""

    @classmethod
    def _from_expts(cls, expts, job_ids=None, station=None):
        """Collect saved jobs for analysis; station supplies missing hardware data."""
        
        expts = list(flatten_exp_lists(expts))
        if not expts:
            raise ValueError("experiment jobs cannot be empty")
        aggregate = cls.__new__(cls)
        aggregate.cfg = expts[0].cfg
        aggregate.data = AttrDict()
        aggregate.batch_expts = expts
        aggregate.batch_job_ids = list(job_ids or [])
        aggregate._analysis_station = station
        return aggregate

    @classmethod
    def from_job_files(cls, job_files, station=None):
        """
        Load child experiment pickle/H5 files into one analysis object.
        """
        import pickle
        from pathlib import Path

        if isinstance(job_files, (str, Path)):
            job_files = [job_files]
        expts = []
        for job_file in flatten_exp_lists(job_files):
            if not isinstance(job_file, (str, Path)):
                expts.append(job_file)
                continue
            path = Path(job_file)
            if path.suffix.lower() in (".h5", ".hdf5"):
                expts.append(cls.from_h5file(str(path)))
            else:
                with path.open("rb") as handle:
                    expts.append(pickle.load(handle))
        return cls._from_expts(expts, station=station)

    @classmethod
    def from_job_ids(cls, job_ids, client=None, station=None):
        """
        Load completed queue jobs without rebuilding a BatchRunner.
        If station is properly specified (along with project name and directory (which is hardcoded)),
        a list of job_ids is okay. Otherwise, a complete directory is necessary.
        """
        if isinstance(job_ids, (str, int, np.integer)):
            job_ids = [job_ids]
        job_ids = [str(job_id) for job_id in flatten_exp_lists(job_ids)]
        if not job_ids:
            raise ValueError("job_ids cannot be empty")
        if client is None:
            if station is None:
                raise ValueError("client or station is required to resolve job IDs")
            job_files = [station.expt_objs_path / f"{job_id}_expt.pkl" for job_id in job_ids]
            aggregate = cls.from_job_files(job_files, station=station)
            aggregate.batch_job_ids = job_ids
            return aggregate
        expts = []
        for job_id in job_ids:
            result = client.get_status(job_id)
            if not result.is_successful():
                raise RuntimeError(f"Job {job_id} {result.status}: {result.error_message or 'No details'}")
            expts.append(result.load_expt())
        return cls._from_expts(expts, job_ids=job_ids, station=station)

    @staticmethod
    def _first_scalar(value):
        """
        This is to avoid a conflict due to different data type of 
        self Kerr. Sometimes it is stored as a list, sometimes as a float...
        """
        
        values = np.asarray(value).reshape(-1)
        if len(values) == 0:
            raise ValueError("saved scalar config is empty")
        return float(values[0])

    @staticmethod
    def _saved_detunings(ecfg, mode_count):
        """
        This is to avoid a conflict due to different data type of 
        detuning. Sometimes it is stored as a list, sometimes as an np array...
        """
        
        detunings = ecfg.get("detunings", None)
        if detunings is None or detunings is False or np.asarray(detunings).size == 0:
            detunings = [0.] * mode_count
        return np.asarray(detunings, dtype=float)

    @classmethod
    def _saved_parameters(cls, expts, station=None):
        """
        Check whether all the sister expts have the same params,
        and return the params as an AttrDict.
        If the class is called for post processing using hdf5 files,
        one should specify station with proper config as well.
        
        Returning params are:
            - `swap_stors`
            - `detunings`
            - `mode_labels`
            - `hardware_parameters`
                - `floquet_cycles_us`
                - `couplings_MHz`
                - `physical_kerr_MHz`
        """
        
        first_cfg = expts[0].cfg
        first_expt_cfg = first_cfg.expt
        swap_stors = [int(stor) for stor in first_expt_cfg.swap_stors]
        detunings = cls._saved_detunings(first_expt_cfg, len(swap_stors))
        physical_kerr_MHz = -abs(cls._first_scalar(first_cfg.device.manipulate.kerr))
        if len(detunings) != len(swap_stors) or not np.all(np.isfinite(detunings)):
            raise ValueError("saved detunings do not match swap_stors")

        program_hardware = []
        sync_cycles = int(first_expt_cfg.get("scramble_sync_cycles", 10))
        floquet_gauss_sigma = first_expt_cfg.get("floquet_gauss_sigma", None)
        floquet_waveform = first_expt_cfg.get("floquet_waveform", None)
        for expt in expts:
            cfg = expt.cfg
            ecfg = cfg.expt
            prog = getattr(expt, "prog", None)
            if prog is not None and hasattr(prog, "calculate_floquet_cycle_us") and hasattr(prog, "m1s_pi_fracs"):
                floquet_cycle_us = float(prog.calculate_floquet_cycle_us())
                pi_fracs = np.asarray([prog.m1s_pi_fracs[stor - 1] for stor in swap_stors], dtype=float)
                couplings_MHz = 1. / (4. * pi_fracs * floquet_cycle_us)
                program_hardware.append((floquet_cycle_us, couplings_MHz))
                break

        if program_hardware:
            floquet_cycle_us, couplings_MHz = program_hardware[0]
            hardware_source = "saved program"
        elif station is not None:
            hardware = cls.hardware_parameters(station, 
                                               swap_stors, 
                                               sync_cycles, 
                                               floquet_gauss_sigma, 
                                               floquet_waveform)
            floquet_cycle_us, couplings_MHz = hardware.floquet_cycle_us, hardware.couplings_MHz
            hardware_source = "current station (H5 fallback)"
        else:
            raise RuntimeError("Floquet cycle time and couplings need the job pickle or station when loading H5 files")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0. or not np.all(np.isfinite(couplings_MHz)) or np.min(couplings_MHz) <= 0.:
            raise ValueError("saved Floquet hardware parameters must be finite and positive")
        hardware = AttrDict(dict(floquet_cycle_us=float(floquet_cycle_us), couplings_MHz=np.asarray(couplings_MHz), physical_kerr_MHz=physical_kerr_MHz, source=hardware_source))
        return AttrDict(dict(swap_stors=swap_stors, 
                             detunings=detunings, 
                             mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors], 
                             hardware=hardware))

    def analyze(self, data=None, **kwargs):
        """Convert one saved job's two preparation phases to a real quadrature.

        This is the per-job analysis the worker runs after ``acquire``. It
        produces ``Q_phi`` at that job's analyzer phase and does *not* combine
        the ``phi=0`` and ``phi=90`` jobs into a complex return -- that is an
        aggregate step and belongs to a stage Experiment.

        The four aggregate stages that used to hide behind ``stage=`` are now
        separate classes; see :data:`STAGE_CLASSES` and the migration table in
        ``analysis_notebooks/guan/MBR_analysis.py``.
        """
        if "stage" in kwargs:
            raise TypeError(_stage_migration_message(kwargs["stage"]))
        if data is not None:
            self.data = data
        self._quadrature(self)
        return self.data

    @staticmethod
    def _quadrature(expt):
        if "return_quadrature" in expt.data:
            return np.asarray(expt.data["return_quadrature"])
        cycles = expt.data["ypts"]
        theta = expt.data["xpts"]
        signal = np.asarray(expt.data["avgi"]).reshape(len(cycles), len(theta))
        q = expt.cfg.expt.qubits[0]
        Ig = expt.cfg.device.readout.Ig[q]
        Ie = expt.cfg.device.readout.Ie[q]
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        expt.data["Pe"] = (signal - Ig) / (Ie - Ig)
        expt.data["return_quadrature"] = expt.data["Pe"][:, 0] - expt.data["Pe"][:, 1]
        return expt.data["return_quadrature"]

    # Analyzer-phase numerics live in fitting/qsim/mbr_phase.py (spec 7.5).
    # Wrappers keep the historical call sites and notebook usage working.
    _cycle_branches = staticmethod(mbr_phase_analysis.cycle_branches)
    build_phase_correction = staticmethod(mbr_phase_analysis.build_phase_correction)
    _unwrap_cycle_phase = staticmethod(mbr_phase_analysis.unwrap_cycle_phase)
    _saved_correction = staticmethod(mbr_phase_analysis.saved_correction)


    # Spectrum and Hamiltonian numerics live in fitting/qsim/mbr_spectrum.py
    # (spec 7.5). Wrapper keeps the historical call sites working.
    analyze_spectrum = staticmethod(mbr_spectrum_analysis.analyze_spectrum)


    # Matrix-Pencil numerics live in fitting/qsim/matrix_pencil.py (spec 7.5).
    # These wrappers keep the historical call sites and notebook usage working.
    analyze_matrix_pencil = staticmethod(matrix_pencil_analysis.analyze_matrix_pencil)
    analyze_matrix_pencil_trace = staticmethod(matrix_pencil_analysis.analyze_matrix_pencil_trace)

    # Spectrum merging, level statistics and SFF numerics live in
    # fitting/qsim/level_statistics.py (spec 7.5). Wrappers keep the historical
    # call sites, notebook usage and self.data defaulting.
    merge_spectra = staticmethod(level_statistics_analysis.merge_spectra)

    def display(self, data=None, **kwargs):
        """Per-job display, inherited. Aggregate plots live on the stage classes."""
        if data is not None:
            self.data = data
        for key, stage in (("matrix", "orthogonality"),
                           ("spectrum", "spectrum"),
                           ("phase_mod180", "calibration")):
            if key in self.data:
                raise TypeError(_stage_migration_message(stage))
        return super().display(data=self.data, **kwargs)

    @staticmethod
    def hardware_parameters(station, 
                            swap_stors, 
                            sync_cycles, 
                            floquet_gauss_sigma=None,
                            floquet_waveform=None):
        """
        Returns hardware related physical paramters such as
            - floquet_cycle_us: time for a single floquet cycle in a microsecond
            - couplings_MHz: an array of effective BS coupling between man and stor
            - physical_kerr_MHz: self Kerr on a central mode (manipulate)
        All the values are calculated from the config/expt_cfg input
        """
        
        if (isinstance(sync_cycles, (bool, np.bool_))
                or not isinstance(sync_cycles, (int, np.integer)) or sync_cycles < 0):
            raise ValueError("sync_cycles must be a nonnegative integer")
        ramp_sigma = station.hardware_cfg.device.manipulate.ramp_sigma
        if isinstance(ramp_sigma, (list, tuple, np.ndarray)):
            ramp_sigma = ramp_sigma[0]
        pulse_us = []
        pi_fracs = []
        cycle_tproc_cycles = 0
        for stor in swap_stors:
            pulse_name = f"M1-S{stor}"
            if station.ds_floquet.get_freq(pulse_name) < FLUX_HIGH_THRESHOLD_MHZ:
                gen_ch = station.hardware_cfg.hw.soc.dacs.flux_low.ch[0]
            else:
                gen_ch = station.hardware_cfg.hw.soc.dacs.flux_high.ch[0]
            waveform = floquet_waveform if floquet_waveform is not None else station.ds_floquet.get_waveform(pulse_name)
            # Match calculate_floquet_cycle_us: round each envelope segment
            # to its generator clock before adding the tProc sync interval.
            if waveform in ("gauss", "gaussian", "arb"):
                sigma = floquet_gauss_sigma
                if sigma is None:
                    sigma = station.ds_floquet.get_gauss_sigma(pulse_name)
                sigma_cycles = station.soccfg.us2cycles(sigma, gen_ch=gen_ch)
                pulse_cycles = sigma_cycles * station.ds_floquet.get_gauss_n_sigma(pulse_name)
            elif waveform == "preload_flattop":
                flat_cycles = station.soccfg.us2cycles(station.ds_floquet.get_len(pulse_name), gen_ch=gen_ch)
                ramp_cycles = station.soccfg.us2cycles(station.ds_floquet.get_ramp_sigma(pulse_name), gen_ch=gen_ch)
                pulse_cycles = flat_cycles + 6 * ramp_cycles
            else:
                flat_cycles = station.soccfg.us2cycles(station.ds_floquet.get_len(pulse_name), gen_ch=gen_ch)
                ramp_cycles = station.soccfg.us2cycles(ramp_sigma, gen_ch=gen_ch)
                pulse_cycles = flat_cycles + 6 * ramp_cycles
            pulse_us.append(station.soccfg.cycles2us(pulse_cycles, gen_ch=gen_ch))
            clock_ratio = float(station.soccfg["tprocs"][0]["f_time"]) / float(station.soccfg["gens"][gen_ch]["f_fabric"])
            cycle_tproc_cycles += int(pulse_cycles * clock_ratio + sync_cycles)
            pi_fracs.append(station.ds_floquet.get_pi_frac(pulse_name))

        # Match the integer synci advances, not the unquantized pulse+gap sum.
        floquet_cycle_us = station.soccfg.cycles2us(cycle_tproc_cycles)
        if not np.all(np.isfinite(pulse_us + pi_fracs)) or min(pulse_us + pi_fracs) <= 0.:
            raise ValueError("Floquet pulse lengths and pi fractions must be finite and positive")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0.:
            raise ValueError("Floquet cycle duration must be finite and positive")

        # 2 * pi * g * t_swap = pi / 2 -> g = 1/ 4/ t_swap
        # pi_frac repetitions of each pulse+sync block make a full swap:
        # g_{bare} = 1/ 4 / n_frac / (t_pulse + t_sync)
        # len(swap_stors) -> g_{eff} * T_F = g_{bare} * (t_pulse+t_sync) = 1/4/n_frac
        # So g_{eff} = 1/4/n_frac/T_F
        couplings_MHz = [1. / (4. * pi_frac * floquet_cycle_us) for pi_frac in pi_fracs]
        physical_kerr_MHz = station.hardware_cfg.device.manipulate.kerr
        if isinstance(physical_kerr_MHz, (list, tuple, np.ndarray)):
            physical_kerr_MHz = physical_kerr_MHz[0]
        physical_kerr_MHz = -abs(physical_kerr_MHz)
        if not np.isfinite(physical_kerr_MHz):
            raise ValueError("physical Kerr must be finite")
        return AttrDict(dict(floquet_cycle_us=floquet_cycle_us,
                             couplings_MHz=np.asarray(couplings_MHz),
                             physical_kerr_MHz=physical_kerr_MHz))


# ---------------------------------------------------------------------------
# Where the four aggregate stages went.
#
# `analyze(stage=...)` is gone. It selected between four unrelated analyses on
# one class; each is now its own Experiment with its own analyze and display.
# This map exists so the error message can name the replacement, and so
# consumers can resolve a stage programmatically. It is not a forwarding shim:
# nothing here re-exports a method under its old address.
STAGE_CLASSES = {
    "calibration": "experiments.qsim.mbr_phase_correction"
                   ".MBRPhaseCorrectionExperiment",
    "orthogonality": "experiments.qsim.mbr_orthogonality"
                     ".MBROrthogonalityExperiment",
    "propagator": "experiments.qsim.mbr_propagator.MBRPropagatorExperiment",
    "spectrum": "experiments.qsim.mbr_spectrum.MBRSpectrumExperiment",
}


def _stage_migration_message(stage):
    """Name the replacement class, so the traceback is the migration note."""
    target = STAGE_CLASSES.get(stage)
    if target is None:
        return (f"unknown stage {stage!r}; the aggregate analyses are now "
                f"separate Experiments: {sorted(STAGE_CLASSES)}")
    module, _, name = target.rpartition(".")
    return (
        f"stage={stage!r} is gone. Use {name} instead:\n"
        f"    from {module} import {name}\n"
        f"    expt = {name}.from_job_files(paths)   # or .from_job_ids(...)\n"
        f"    expt.analyze()\n"
        f"    expt.display()\n"
        f"See analysis_notebooks/guan/MBR_analysis.py for a worked example.")


# ---------------------------------------------------------------------------
# Compatibility: measurement families that moved to their own modules.
#
# The acquisition notebooks address programs as
# ``meas.qsim.floquet_dark_mode_readout.<Name>``, so the old attribute has to
# keep resolving. A module-level ``__getattr__`` (PEP 562) does that lazily,
# which matters: every new module imports the still-resident base classes from
# here, so a top-level re-import would be circular.
#
# Note: ``__dir__`` advertises these names, so the flattening exporter in
# ``experiments/__init__.py`` does re-export them to the ``experiments``
# namespace. That is harmless -- it resolves to the very same class object the
# defining module exports, so the second write is idempotent.
_MOVED_TO = {
    "BroadbandGeValidationProgram": "dark_mode_broadband_ge_validation",
    "DarkT1Experiment": "dark_mode_t1",
    "DarkT1Program": "dark_mode_t1",
    "FloquetDisplacementKerrExperiment": "floquet_displacement_kerr",
    "FloquetDisplacementKerrProgram": "floquet_displacement_kerr",
    "ManStorMultiparityChevronRExperiment": "dark_mode_multiparity_chevron",
    "ManStorMultiparityChevronRProgram": "dark_mode_multiparity_chevron",
    "SidebandStarkAmplificationModifiedProgram": "sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_newold": "sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_old": "sideband_stark_shift_cal",
    "StorageSwapPhaseAccumulationProgram": "storage_swap_phase_cal",
}


def __getattr__(name):
    module = _MOVED_TO.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(
        importlib.import_module(f"experiments.qsim.{module}"), name)


def __dir__():
    return sorted(list(globals()) + list(_MOVED_TO))

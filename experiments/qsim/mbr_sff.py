# -*- coding: utf-8 -*-
"""Disorder-averaged spectral form factor for the many-body Ramsey Floquet map.

The SFF, ``|Tr U|^2 / D^2``, is the standard diagnostic for whether a
many-body spectrum shows level repulsion: its dip-ramp-plateau shape separates
chaotic from integrable dynamics, and it is what the level-statistics analysis
in ``fitting/qsim/level_statistics.py`` estimates from a resolved spectrum.
This module measures it directly instead, which is what makes it worth having:
no peak fitting, no resolution limit set by the longest depth reached.

How the measurement works
-------------------------
``U`` is the Floquet propagator for one cycle and ``D`` the dimension of the
fixed-photon-number sector. ``z = Tr(U)/D`` is obtained by summing the
diagonal returns ``<b|U^q|b>`` over a complete basis of that sector, one
occupation string per basis state. Each return is read out as a complex
amplitude from four phase settings, ``(theta, phi) = (0,0), (180,0), (0,90),
(180,90)``:

    Q_phi = Pe(theta=0, phi) - Pe(theta=180, phi)      and      A = Q_0 - 1j*Q_90

Disorder is a set of random per-mode detunings, one draw per realization, and
the ensemble average runs over those draws.

Two design choices are worth understanding before changing anything here.

**Two independent shot replicas, never one squared.** ``|z|^2`` computed from a
single noisy estimate of ``z`` carries a positive shot-noise bias that survives
averaging, which would masquerade as a plateau. Every diagonal return is
therefore saved as two independent raw-shot replicas and the estimator is
``Re[z_A z_B*]``, whose noise is mean-zero because the replicas are
independent.

**Visibility is measured separately.** A disorder-free depth-zero job measures
the complex access-path gain of every occupation at high statistics, and each
occupation is divided by its own visibility. Folding this into the disorder
jobs would put the normalization and the signal in the same shot noise.

``sff_connected`` is ``sff_full`` minus a disconnected part built from
*different* disorder realizations, so it isolates correlations within a
realization from the ensemble mean.

Layout
------
Three classes with genuinely separate jobs, which is why they are worth keeping
apart rather than merging into one program:

* :class:`HardwareFloquetDepthSweepMixin` -- tProc register allocation and the
  counted Floquet loop. No access path, no estimator.
* :class:`DisorderSFFSequenceMixin` -- the four access-path quadratures,
  borrowing the pulse-list builders from the fixed-depth spectroscopy program.
* :class:`DisorderSFFDepthSweepProgram` -- connects those to the RAverager
  lifecycle, and nothing else.

Status
------
Jonginn marked the two mixins and the program "NOT PERUSED AND MAYBE DELETED IN
A NEAR FUTURE"; those markers are kept verbatim below. Treat this module as
provisional and check with him before building on it.

On the base class
-----------------
``DisorderSFFExperiment`` subclassed ``EncodingHamiltonianSpectroscopyExperiment``
on main. It uses nothing from it: no ``super()`` calls, no reference to any of
its members, its own ``acquire`` and ``analyze``, and callers only reach its own
``batch`` and ``analyze_ensemble`` classmethods. In particular it does not use
the ``from_job_ids``/``from_job_files`` loading layer, which is the one thing
that class offers its subclasses. Since that class has been split four ways,
inheriting from it would force an arbitrary choice among the four stages for no
benefit, so this subclasses ``QsimBaseExperiment`` -- the level that actually
supplies what it uses (``ProgramClass``, ``__init__``, ``save_data``,
``display``). Flagged for jonginn: if the intent was to reuse the loading
layer, say so and it can inherit a stage class instead.
"""
from copy import deepcopy
from itertools import product
from math import comb

import numpy as np
from slab import AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import MMRAveragerProgram
from experiments.qsim.qsim_base import QsimBaseExperiment
from experiments.qsim.utils import ensure_list_in_cfg
from experiments.qsim.floquet_dark_mode_readout import (
    DarkBaseProgram,
    DarkBaseRProgram,
    NPhotonHamiltonianSpectroscopyProgram,
    _play_preloaded_floquet_register_bank_entry,
    _prepare_preloaded_floquet_register_bank,
    flatten_exp_lists,
)


class HardwareFloquetDepthSweepMixin:
    """
    Low-level tProc machinery for a variable-depth Floquet sweep.

    This mixin owns register allocation, pulse phases updated by depth, and
    the counted Floquet loop.  It deliberately contains no spectroscopy
    access path or SFF estimator logic.
    """

    def _allocate_custom_register(self,
                                  ch,
                                  purpose,
                                  page):
        """
        Assign one named SFF custom register on a tProc register page for a
        hardware loop.

        Requires ``_sff_register_map``, which follows the same convention as
        ``_gen_regmap`` and ``_ro_regmap``; e.g.
            - _sff_register_map[(ch, "purpose")] = (page, regnum)
            
        It stores allocated custom resiger to ``self._sff_register_map`` as 
        ``(page, register)`` with the key ``(ch, purpose)``
        and also returns allocated ``register``
        """
        page = int(page)
        register_key = (ch, purpose)
        if register_key in self._sff_register_map:
            raise RuntimeError(
                f"SFF register {register_key!r} was already assigned"
            )

        _gen_register_maps = self._gen_regmap.values() #format: (page, reg_num)
        _ro_register_maps  = self._ro_regmap.values() #format: (page, reg_num)

        register_maps = list(_gen_register_maps) + list(_ro_register_maps) #list of all sregs (page, reg_num)

        special_registers = [
            special_register for register_page, special_register in register_maps
            if register_page == page and special_register > 0
        ] # picks reg_num of regs in designated ``page``

        if page == 0:
            # MM active reset hard-codes page-zero registers 8~11 and the
            # RAverager uses high page-zero bookkeeping registers.  Only 1--7
            # are treated as custom register space here.
            special_registers.append(8)
            special_registers.append(9)
            special_registers.append(10)
            special_registers.append(11)
            # Also, RAverager tacitly uses regnum of 13~15 at page 0 for the internal looping..
            # dunno why it does not really give the wrapper that returns those valuses
            # adding the reg nums for page 0 as well
            special_registers.append(13)
            special_registers.append(14)
            special_registers.append(15)
            special_registers.append(16)

        used_registers = {
            register
            for register_page, register in self._sff_register_map.values()
            if register_page == page
        }

        tproc_register_count = 32
        register = None
        for candidate in range(1, tproc_register_count):
            if candidate not in special_registers and candidate not in used_registers:
                register = candidate
                break

        if register is None:
            raise RuntimeError(f"Not have enough custom registers on page {page}")

        self._sff_register_map[register_key] = (page, register)

        return register

    def _prepare_pulse_with_phase_updated_by_depth(self,
                                                   pulse_description,
                                                   phase_at_first_depth_deg,
                                                   phase_change_per_depth_point_deg,
                                                   pulse_name):
        """
        Compile one pulse whose phase must change at every depth point.
        The primary purpose of this method is to prepare and store the 
        decoding pulses to do real time phase update.

        ``initialize()`` runs only once for the whole RAverager sweep, so
        Python cannot recalculate the pulse phase at every depth.  Instead,
        this method stores the phase in a custom tProc register.  ``update()``
        increments that register once per depth point, and the play method
        copies its current value into the generator phase register.

        This method only prepares the waveform and phase register.  It does
        not play the pulse.
        
        Note: ``prepulse_creator2`` contains ``self.pulse`` as 
            - [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]]
        
        Args:
            - pulse_description: one of our ununified pulse convention: e.g. ["storage", "M1-S5", "pi", 180.0]
            - phase_at_first_depth_deg: phase at first (phi_first)
            - phase_change_per_depth_point_deg: phase per each step (phi_step); 
                - per each step, the phase is phi_first + n_step * phi_step
        
        Return: {"channel": channel, # Channel for the pulse
                "phase_accumulator": (register_page, phase_accumulator_register,), #Register page and regnum for phase accumulation
                "phase_increment_reg_val": phase_increment_reg_val,
                "qick_pulse_registers": qick_pulse_registers,}
        
            
        """
        # The phase in the ordinary pulse description is deliberately zero.
        # The custom phase accumulator supplies the actual phase at run time.
        pulse_description_without_fixed_phase = list(pulse_description)
        pulse_description_without_fixed_phase[3] = 0.0
        pulse_data = self.get_prepulse_creator([pulse_description_without_fixed_phase]).pulse
        (pulse_frequencies_mhz,
         pulse_gains,
         pulse_lengths_us,
         _unused_fixed_phases_deg,
         pulse_channels,
         pulse_shapes,
         pulse_sigmas_us,) = pulse_data
        
        if len(pulse_frequencies_mhz) != 1:
            raise RuntimeError(f"SFF pulse {pulse_name!r} did not compile to one pulse")

        channel = pulse_channels[0]
        if isinstance(channel, list):
            channel = channel[0]
        channel = int(channel)
        register_page = self.ch_page(channel)

        if register_page == 0: #This is just a conservative safety check to avoid using page 0 for a pulse register.
            raise RuntimeError("Pulse register page must not be 0")
        
        generator_manager = self._gen_mgrs[channel]
        generator_manager_name = generator_manager.__class__.__name__
        if generator_manager_name != "FullSpeedGenManager":
            raise RuntimeError(f"depth-dependent SFF phases require a full-speed generator; channel {channel} uses {generator_manager_name}")

        phase_accumulator_register = self._allocate_custom_register(ch=channel,
                                                                    purpose=f"phase_updated_by_depth_{pulse_name}",
                                                                    page=register_page,)
        # deg2reg returns the wrapped hardware phase register value. Therefore a
        # negative phase change is represented by its equivalent positive
        # phase register value, and repeated mathi additions still give the intended
        # phase evolution modulo the generator's phase width.
        phase_increment_reg_val = self.deg2reg(phase_change_per_depth_point_deg,gen_ch=channel,)
        if not 0 <= phase_increment_reg_val < 2 ** 31:
            raise RuntimeError(f"phase increment register value for {pulse_name} does not fit mathi")

        phase_at_first_depth_reg_val = self.deg2reg(self._mod360(phase_at_first_depth_deg), gen_ch=channel,)
        self.safe_regwi(register_page,
                        phase_accumulator_register,
                        phase_at_first_depth_reg_val,)

        pulse_shape = pulse_shapes[0]
        pulse_frequency_mhz = pulse_frequencies_mhz[0]
        pulse_gain = pulse_gains[0]
        pulse_length_us = pulse_lengths_us[0]
        pulse_sigma_us = pulse_sigmas_us[0]
        
        qick_pulse_registers = dict(ch=channel,
                                    freq=self.freq2reg(pulse_frequency_mhz, gen_ch=channel),
                                    phase=0,
                                    gain=pulse_gain)

        if pulse_shape in ("gaussian", "gauss", "g"):
            sigma_cycles = self.us2cycles(pulse_sigma_us, gen_ch=channel,)
            waveform_name = (f"sff_depth_phase_g_{pulse_name}_{channel}_{sigma_cycles}")
            self.add_gauss(ch=channel,name=waveform_name,sigma=sigma_cycles,length=4 * sigma_cycles,)
            qick_pulse_registers.update(style="arb", waveform=waveform_name,)
        elif pulse_shape in ("flat_top", "f"):
            sigma_cycles = self.us2cycles(pulse_sigma_us,gen_ch=channel,)
            gaussian_length_multiplier = (6 if channel in (0, 1, 3) else 4) # This needs elaboration. I think the current program is really messed up with this gaussian length
            
            waveform_name = (f"sff_depth_phase_ft_{pulse_name}_{channel}_{sigma_cycles}")
            self.add_gauss(ch=channel,
                           name=waveform_name,
                           sigma=sigma_cycles,
                           length=gaussian_length_multiplier * sigma_cycles,)
            qick_pulse_registers.update(style="flat_top",
                                        waveform=waveform_name, 
                                        length=self.us2cycles(pulse_length_us,gen_ch=channel))

        elif isinstance(pulse_shape, list):
            raise ValueError(f"optimal-control pulses are not supported as depth-dependent SFF phase pulses ({pulse_name})")

        else:
            qick_pulse_registers.update(style="const",length=self.us2cycles(pulse_length_us,gen_ch=channel))

        return {"channel": channel,
                "phase_accumulator": (register_page, phase_accumulator_register,),
                "phase_increment_reg_val": phase_increment_reg_val,
                "qick_pulse_registers": qick_pulse_registers,}

    def _play_pulse_with_current_depth_phase(self, depth_phase_pulse):
        """
        Copy the current phase accumulator into the generator and pulse.
        """
        channel = depth_phase_pulse["channel"]
        register_page, phase_accumulator_register = depth_phase_pulse["phase_accumulator"]

        # set_pulse_registers writes phase=0, so the phase accumulator must be
        # copied into the generator phase register after this call.
        self.set_pulse_registers(**depth_phase_pulse["qick_pulse_registers"])
        self.mathi(register_page,
                   self.sreg(channel, "phase"),
                   phase_accumulator_register,
                   "+",
                   0)
        self.pulse(channel)
        self.sync_all(self.us2cycles(0.01)) # same as self.custome_pulse(). it needs to be refined soon.

    def _prepare_floquet_loop(self, 
                              swap_stors, 
                              detunings):
        
        
        ecfg = self.cfg.expt
        phase_offsets = [0.0] * len(swap_stors)
        update_phases = bool(ecfg.get("update_phases", True))

        self._sff_floquet_pulse_args = []
        for index, stor in enumerate(swap_stors):
            pulse_args = deepcopy(self.m1s_kwargs[stor - 1])
            pulse_args["freq"] += self.freq2reg(detunings[index], gen_ch=pulse_args["ch"])
            self._sff_floquet_pulse_args.append(pulse_args)

        first_cycle_phases = []
        for stor in swap_stors:
            index = swap_stors.index(stor)
            first_cycle_phases.append(self._mod360(phase_offsets[index]))
            if update_phases:
                self._advance_phase_offsets(phase_offsets=phase_offsets,
                                            swap_stors=swap_stors,
                                            pulsed_stor=stor)
        phase_steps = [self._mod360(phase) for phase in phase_offsets]

        self._sff_use_preloaded_register_bank = all(
            self.m1s_waveform_mode[stor - 1] == "preload_flattop"
            for stor in swap_stors
        )
        if self._sff_use_preloaded_register_bank:
            result = _prepare_preloaded_floquet_register_bank(
                self,
                self._sff_floquet_pulse_args,
                first_cycle_phases,
                phase_steps,
            )
            self._sff_floquet_register_bank, _, _ = result
        else:
            self._sff_floquet_phase_registers = []
            for stor, pulse_args, phase_step in zip(
                    swap_stors,
                    self._sff_floquet_pulse_args,
                    phase_steps):
                ch = int(pulse_args["ch"])
                generator_manager_name = (
                    self._gen_mgrs[ch].__class__.__name__
                )
                if generator_manager_name != "FullSpeedGenManager":
                    raise RuntimeError(
                        "the SFF Floquet loop requires full-speed "
                        f"generators; channel {ch} uses "
                        f"{generator_manager_name}"
                    )
                page = self.ch_page(ch)
                if page == 0:
                    raise RuntimeError("Pulse register page must not be 0")
                phase_register = self._allocate_custom_register(
                    ch=ch,
                    purpose=f"floquet_phase_S{stor}",
                    page=page,
                )
                step_register_value = int(
                    self.deg2reg(phase_step, gen_ch=ch)
                )
                if not -(2 ** 30) <= step_register_value < 2 ** 31:
                    raise RuntimeError("Floquet phase step does not fit mathi")
                self._sff_floquet_phase_registers.append((
                    page,
                    phase_register,
                    step_register_value,
                ))

        # Page zero has ample ordinary registers and avoids competing with the
        # flux/qubit phase accumulators on the shared generator page.
        self._sff_depth_page = 0
        self._sff_depth_register = self._allocate_custom_register(ch=None,
                                                                  purpose="floquet_depth",
                                                                  page=self._sff_depth_page)
        self._sff_loop_register = self._allocate_custom_register(ch=None,
                                                                 purpose="floquet_loop_counter",
                                                                 page=self._sff_depth_page,)
        self.safe_regwi(self._sff_depth_page,
                        self._sff_depth_register,
                        self._sff_cycle_start)

        self._sff_first_cycle_phases = first_cycle_phases
        self._sff_loop_label = "DISORDER_SFF_FLOQUET_LOOP"

    def _play_floquet_depth(self):
        """Play the current depth register as a counted Floquet loop."""
        if self._sff_use_preloaded_register_bank:
            for entry in self._sff_floquet_register_bank:
                self.safe_regwi(
                    entry["register_page"],
                    entry["phase_register"],
                    entry["first_phase_register_value"],
                )
        else:
            for pulse_args, first_phase, registers in zip(
                    self._sff_floquet_pulse_args,
                    self._sff_first_cycle_phases,
                    self._sff_floquet_phase_registers):
                page, phase_register, _ = registers
                self.safe_regwi(
                    page,
                    phase_register,
                    self.deg2reg(
                        first_phase, gen_ch=pulse_args["ch"]),
                )

        self.mathi(self._sff_depth_page,
                   self._sff_loop_register,
                   self._sff_depth_register,
                   "+",
                   0)
        self.mathi(self._sff_depth_page,
                   self._sff_loop_register,
                   self._sff_loop_register,
                   "-",
                   1) #subtract -1 to repeat n_depth of floquet pulses

        if not self._sff_use_preloaded_register_bank:
            first_pulse = self._sff_floquet_pulse_args[0]
            first_pulse["phase"] = 0
            self.set_pulse_registers(**first_pulse)
        loop_index = getattr(self, "_sff_loop_index", 0)
        self._sff_loop_index = loop_index + 1
        loop_label = f"{self._sff_loop_label}_{loop_index}"
        
        
        #--------------- FloquetHardware Loop Starts
        self.label(loop_label) 
        sync_cycles = int(self.cfg.expt.get("scramble_sync_cycles", 10))

        for index, pulse_args in enumerate(self._sff_floquet_pulse_args):
            if self._sff_use_preloaded_register_bank:
                entry = self._sff_floquet_register_bank[index]
                _play_preloaded_floquet_register_bank_entry(self, entry)
                self.math(
                    entry["register_page"],
                    entry["phase_register"],
                    entry["phase_register"],
                    "+",
                    entry["phase_step_register"],
                )
            else:
                ch = int(pulse_args["ch"])
                page, phase_register, step_register_value = (
                    self._sff_floquet_phase_registers[index]
                )
                self.mathi(
                    page,
                    self.sreg(ch, "phase"),
                    phase_register,
                    "+",
                    0,
                )
                self.pulse(ch)
                self.mathi(
                    page,
                    phase_register,
                    phase_register,
                    "+",
                    step_register_value,
                )

                next_index = (index + 1) % len(self._sff_floquet_pulse_args)
                next_pulse = self._sff_floquet_pulse_args[next_index]
                next_pulse["phase"] = 0
                self.set_pulse_registers(**next_pulse)
            self.sync_all(sync_cycles)

        self.loopnz(self._sff_depth_page, self._sff_loop_register, loop_label)
        
        #--------------- FloquetHardware Loop Ends
        self.sync_all()

    def _advance_sff_depth(self):
        """Advance depth and every phase that is linear in depth."""
        self.mathi(self._sff_depth_page,
                   self._sff_depth_register,
                   self._sff_depth_register,
                   "+",
                   self._sff_cycle_step)

        for depth_phase_pulse in self._sff_pulses_with_phase_updated_by_depth:
            register_page, phase_accumulator_register = depth_phase_pulse["phase_accumulator"]
            
            
            self.mathi(register_page,
                       phase_accumulator_register,
                       phase_accumulator_register,
                       "+",
                       depth_phase_pulse["phase_increment_reg_val"])


class DisorderSFFSequenceMixin:
    """
    NOT PERUSED AND MAYBE DELETED IN A NEAR FUTURE
    
    Configure and play the four access-path quadratures for SFF.

    The established fixed-depth spectroscopy program remains unchanged.  The
    SFF path only borrows its pure pulse-list builders and otherwise keeps a
    separate RAverager implementation.
    
    NOT PERUSED AND MAYBE DELETED IN A NEAR FUTURE
    """

    _get_encoder_pulses = staticmethod(NPhotonHamiltonianSpectroscopyProgram._get_encoder_pulses)
    _storage_mode_from_pulse_name = staticmethod(NPhotonHamiltonianSpectroscopyProgram._storage_mode_from_pulse_name)
    _validate_storage_swap_rows = staticmethod(NPhotonHamiltonianSpectroscopyProgram._validate_storage_swap_rows)
    _get_inverse_pulses = staticmethod(NPhotonHamiltonianSpectroscopyProgram._get_inverse_pulses)
    _add_wait_after_storage_pulses = (NPhotonHamiltonianSpectroscopyProgram._add_wait_after_storage_pulses)
    _mod360 = DarkBaseProgram._mod360
    _advance_phase_offsets = DarkBaseProgram._advance_phase_offsets
    _advance_storage_phase_offsets = DarkBaseProgram._advance_storage_phase_offsets
    calculate_floquet_cycle_us = DarkBaseProgram.calculate_floquet_cycle_us

    def _configure_sff_experiment(self):
        ecfg = self.cfg.expt
        ecfg.perform_wigner = False
        ecfg.init_stor = 0
        ecfg.ro_stor = 0
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.floquet_hardware_loop = False

        raw_swap_stors = list(ecfg.swap_stors)
        if any(isinstance(stor, (bool, np.bool_)) or not isinstance(
                stor, (int, np.integer)) for stor in raw_swap_stors):
            raise TypeError("swap_stors entries must be integers in 1..7")
        swap_stors = [int(stor) for stor in raw_swap_stors]
        raw_occupations = list(ecfg.spectroscopy_occupations)
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)) for n in raw_occupations):
            raise TypeError(
                "spectroscopy_occupations entries must be integers"
            )
        occupations = [int(n) for n in raw_occupations]
        if len(swap_stors) < 1 or len(set(swap_stors)) != len(swap_stors):
            raise ValueError("swap_stors must be a nonempty list of distinct modes")
        if any(stor < 1 or stor > 7 for stor in swap_stors):
            raise ValueError("swap_stors entries must be in 1..7")
        if len(occupations) != len(swap_stors) + 1:
            raise ValueError(
                "spectroscopy_occupations must contain M1 followed by one "
                "entry for every swap_stor"
            )
        if any(n < 0 for n in occupations) or sum(occupations) < 1:
            raise ValueError(
                "spectroscopy_occupations must be non-negative and non-vacuum"
            )
        if max(occupations) > 9:
            raise ValueError("The current multiphoton transition parser supports local occupations only through n=9")
        if ecfg.get("palindrome_scramble", False):
            raise ValueError("the SFF hardware depth sweep does not support palindrome_scramble")
        if ecfg.get("pre_selection_reset", False):
            raise ValueError("SFF A/B replicas do not support pre_selection_reset")
        for flag in ("load_man_dark", "swap_man_dark", "swap_man_large_dark",
                     "perform_wigner", "parity_check", "pre_selection_parity",
                     "parity_readout", "multiparity_readout"):
            if ecfg.get(flag, False):
                raise ValueError(f"{flag}=True is incompatible with disorder SFF")
        if int(ecfg.get("scramble_sync_cycles", 10)) < 10:
            raise ValueError("scramble_sync_cycles must be at least 10 for register setup")

        self._sff_cycle_start = int(ecfg.start)
        self._sff_cycle_step = int(ecfg.step)
        self._sff_cycle_count = int(ecfg.expts)
        if self._sff_cycle_start < 1:
            raise ValueError("SFF RAverager start must be at least one cycle")
        if self._sff_cycle_step < 1 or self._sff_cycle_count < 1:
            raise ValueError("SFF RAverager step and expts must be positive")
        reps = int(ecfg.reps)
        if reps < 2 or reps % 2:
            raise ValueError("SFF reps must be an even integer of at least two")
        if int(ecfg.get("rounds", 1)) != 1:
            raise ValueError("SFF raw A/B pairing requires rounds=1")

        detunings = np.asarray(
            ecfg.get("detunings", [0.0] * len(swap_stors)), dtype=float)
        if detunings.shape != (len(swap_stors),) or not np.all(
                np.isfinite(detunings)):
            raise ValueError(
                f"detunings must have shape ({len(swap_stors)},) and be finite"
            )

        ecfg.spectroscopy_occupations = occupations
        ecfg.spectroscopy_final_occupations = occupations
        phase_settings = [
            tuple(map(float, setting))
            for setting in ecfg.get("sff_phase_settings", [
                [0.0, 0.0],
                [180.0, 0.0],
                [0.0, 90.0],
                [180.0, 90.0],
            ])
        ]
        expected_settings = [
            (0.0, 0.0), (180.0, 0.0),
            (0.0, 90.0), (180.0, 90.0),
        ]
        canonical_settings = [
            (theta % 360.0, phi % 360.0)
            for theta, phi in phase_settings
        ]
        if canonical_settings != expected_settings:
            raise ValueError(
                "sff_phase_settings must be ordered exactly as (theta, phi) "
                "= (0,0), (180,0), (0,90), (180,90)"
            )
        self._sff_phase_settings = canonical_settings
        ecfg.sff_phase_settings = [list(setting)
                                   for setting in canonical_settings]
        ecfg.spectroscopy_prep_phase = 0.0
        ecfg.spectroscopy_analyzer_phase = 0.0
        ecfg.final_analyzer_phase_per_cycle_deg = float(
            ecfg.final_analyzer_phase_per_cycle_deg)
        ecfg.final_analyzer_phase_application_sign = -1.0
        ecfg.spectroscopy_photon_number = sum(occupations)
        ecfg.dedupe_waveforms = True

    def _compile_sff_sequences(self):
        """Compile access paths and bind their depth-dependent phases."""
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        occupations = [int(n) for n in ecfg.spectroscopy_occupations]
        detunings = np.asarray(
            ecfg.get("detunings", [0.0] * len(swap_stors)), dtype=float)

        self._sff_register_map = {}
        if self.storage_phase_matrix is not None and \
                self.storage_phase_matrix.shape != (
                    len(swap_stors), len(swap_stors)):
            raise ValueError(
                "storage_phase_matrix must have shape "
                f"({len(swap_stors)}, {len(swap_stors)})"
            )
        self.floquet_cycle_us = float(self.calculate_floquet_cycle_us())
        ecfg.floquet_cycle_us = self.floquet_cycle_us

        use_multiphoton_swap = bool(
            ecfg.get("use_multiphoton_swap", False))
        ecfg.use_multiphoton_swap = use_multiphoton_swap
        self._validate_storage_swap_rows(
            self.cfg.device.storage._ds_storage,
            occupations,
            swap_stors,
            use_multiphoton_swap,
        )

        encoder_pulses = self._get_encoder_pulses(
            occupations,
            swap_stors,
            use_multiphoton_swap,
        )
        if any(pulse[1] == "ge_broadband" for pulse in encoder_pulses):
            pulse_key = "pi_ge_broadband"
            if pulse_key not in self.cfg.device.qubit.pulses:
                raise KeyError(
                    "This occupation-string encoder requires "
                    "device.qubit.pulses.pi_ge_broadband"
                )
            broadband_cfg = self.cfg.device.qubit.pulses[pulse_key]
            for field in ("frequency", "gain", "sigma", "length", "type"):
                if field not in broadband_cfg or np.asarray(
                        broadband_cfg[field]).size == 0:
                    raise RuntimeError(
                        "device.qubit.pulses.pi_ge_broadband."
                        f"{field} must contain one value"
                    )
            gain = np.asarray(
                broadband_cfg.get("gain", []), dtype=float).reshape(-1)
            if gain.size == 0 or gain[0] <= 0:
                raise RuntimeError(
                    "device.qubit.pulses.pi_ge_broadband.gain must be a "
                    "configured nonzero value"
                )

        max_local_occupation = max(occupations)
        multiphoton_pi = self.cfg.device.multiphoton.pi
        for transition in ("en-fn", "fn-gn+1"):
            if transition not in multiphoton_pi:
                raise KeyError(
                    f"device.multiphoton.pi.{transition} is missing")
            for field in ("frequency", "gain", "length", "type", "sigma"):
                values = np.asarray(
                    multiphoton_pi[transition].get(field, [])).reshape(-1)
                if len(values) < max_local_occupation:
                    raise RuntimeError(
                        f"device.multiphoton.pi.{transition}.{field} needs "
                        f"at least {max_local_occupation} entries for "
                        f"spectroscopy_occupations={occupations}"
                    )
        storage_phase_offsets = [0.0] * len(swap_stors)
        encoder_pulses = deepcopy(encoder_pulses)
        for pulse in encoder_pulses:
            if pulse[0] != "storage":
                continue
            stor = self._storage_mode_from_pulse_name(pulse[1])
            index = swap_stors.index(stor)
            pulse[3] = self._mod360(
                pulse[3] + storage_phase_offsets[index])
            self._advance_storage_phase_offsets(
                storage_phase_offsets, swap_stors, stor)

        self._sff_preparation_pulses = {}
        for preparation_phase_deg in sorted({
                setting[0] for setting in self._sff_phase_settings}):
            prepulse_cfg = [[
                "qubit", "ge", "hpi", preparation_phase_deg
            ]] + encoder_pulses
            prepulse_cfg = self._add_wait_after_storage_pulses(
                prepulse_cfg)
            self._sff_preparation_pulses[preparation_phase_deg] = (
                self.get_prepulse_creator(prepulse_cfg).pulse
            )

        self._prepare_floquet_loop(swap_stors, detunings)

        self._sff_decoder_operations = []
        storage_wait_us = float(ecfg.get("storage_pulse_wait_us", 0.2))
        decoder_encoder_pulses = self._get_encoder_pulses(
            occupations,
            swap_stors,
            use_multiphoton_swap,
        )
        postpulse_cfg = self._get_inverse_pulses(
            decoder_encoder_pulses)
        for operation_index, pulse in enumerate(postpulse_cfg):
            if pulse[0] != "storage":
                self._sff_decoder_operations.append({
                    "phase_behavior": "fixed",
                    "pulse": self.get_prepulse_creator([pulse]).pulse,
                    "wait_after_us": 0.0,
                })
                continue

            stor = self._storage_mode_from_pulse_name(pulse[1])
            index = swap_stors.index(stor)
            base_phase = self._mod360(
                pulse[3] + storage_phase_offsets[index])
            disorder_phase_per_cycle = (
                360.0 * float(detunings[index]) * self.floquet_cycle_us
            )
            depth_phase_pulse = (
                self._prepare_pulse_with_phase_updated_by_depth(
                    pulse_description=pulse,
                    phase_at_first_depth_deg=(
                        base_phase
                        + self._sff_cycle_start * disorder_phase_per_cycle
                    ),
                    phase_change_per_depth_point_deg=(
                        self._sff_cycle_step * disorder_phase_per_cycle
                    ),
                    pulse_name=f"storage_{operation_index}",
                )
            )
            self._sff_decoder_operations.append({
                "phase_behavior": "updated_by_depth",
                "pulse": depth_phase_pulse,
                "wait_after_us": storage_wait_us,
            })
            self._advance_storage_phase_offsets(
                storage_phase_offsets, swap_stors, stor)

        analyzer_phase_per_cycle = -float(
            ecfg.final_analyzer_phase_per_cycle_deg)
        self._sff_analyzer_pulses = {}
        for analyzer_phase_deg in sorted({
                setting[1] for setting in self._sff_phase_settings}):
            self._sff_analyzer_pulses[analyzer_phase_deg] = (
                self._prepare_pulse_with_phase_updated_by_depth(
                    pulse_description=["qubit", "ge", "hpi", 0.0],
                    phase_at_first_depth_deg=(
                        analyzer_phase_deg
                        + self._sff_cycle_start
                        * analyzer_phase_per_cycle
                    ),
                    phase_change_per_depth_point_deg=(
                        self._sff_cycle_step
                        * analyzer_phase_per_cycle
                    ),
                    pulse_name=f"analyzer_{int(analyzer_phase_deg)}",
                )
            )

        self._sff_pulses_with_phase_updated_by_depth = [
            operation["pulse"]
            for operation in self._sff_decoder_operations
            if operation["phase_behavior"] == "updated_by_depth"
        ] + list(self._sff_analyzer_pulses.values())
        self.sync_all(200)

    def _body_one_phase_setting(
            self,
            preparation_phase_deg,
            analyzer_phase_deg):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMRAveragerProgram.get_active_reset_params(self.cfg)
            params["prefix"] = (
                f"sff_{int(preparation_phase_deg)}_"
                f"{int(analyzer_phase_deg)}_"
            )
            self.active_reset(**params)
            pre_relax_delay = float(ecfg.get("pre_relax_delay", 0.0))
            if pre_relax_delay > 0.0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        self.sync_all()
        self.custom_pulse(
            cfg,
            deepcopy(self._sff_preparation_pulses[preparation_phase_deg]),
            prefix=f"sff_pre_{int(preparation_phase_deg)}_",
        )
        self.sync_all()
        self._play_floquet_depth()

        for post_operation in self._sff_decoder_operations:
            phase_behavior = post_operation["phase_behavior"]
            pulse = post_operation["pulse"]
            wait_after_us = post_operation["wait_after_us"]
            if phase_behavior == "fixed":
                self.custom_pulse(
                    cfg, deepcopy(pulse), prefix="sff_post_")
            elif phase_behavior == "updated_by_depth":
                self._play_pulse_with_current_depth_phase(pulse)
                if wait_after_us > 0.0:
                    self.sync_all(self.us2cycles(wait_after_us))
            else:
                raise RuntimeError(
                    "unknown SFF post-pulse phase behavior: "
                    f"{phase_behavior!r}"
                )

        self._play_pulse_with_current_depth_phase(
            self._sff_analyzer_pulses[analyzer_phase_deg]
        )
        self.sync_all()
        self.measure_wrapper()


class DisorderSFFDepthSweepProgram(
        DisorderSFFSequenceMixin,
        HardwareFloquetDepthSweepMixin,
        DarkBaseRProgram):
    """
    NOT PERUSED AND MAYBE DELETED IN A NEAR FUTURE
    
    Hardware depth sweep for one occupation and disorder realization.

    At every depth the program executes four access-path quadratures:

        reset -> encode -> U**q -> decode -> analyze -> measure

    The sequence mixin owns the spectroscopy access paths, the hardware mixin
    owns tProc registers and the counted Floquet loop, and this concrete class
    only connects those pieces to the RAverager lifecycle.
    NOT PERUSED AND MAYBE DELETED IN A NEAR FUTURE
    """

    def initialize(self):
        # Configuration must be normalized before MM/QICK creates channels.
        self._configure_sff_experiment()
        super().initialize()

        # Pulse compilation and register allocation require initialized QICK
        # generators and the calibrated Floquet pulse dictionaries.
        self._compile_sff_sequences()

    def body(self):
        # Four fixed readout lanes per RAverager point.  Packing only these
        # short phase settings (not all depths) cuts program loads by four
        # without unrolling the 100-depth evolution into tProc memory.
        for preparation_phase_deg, analyzer_phase_deg in (
                self._sff_phase_settings):
            self._body_one_phase_setting(
                preparation_phase_deg,
                analyzer_phase_deg,
            )

    def update(self):
        self._advance_sff_depth()


class DisorderSFFExperiment(QsimBaseExperiment):
    """Direct disorder-ensemble spectral-form-factor acquisition.

    The production job has two deliberately separate parts:

    * one high-statistics, disorder-free depth-zero visibility job measuring
      the complex access-path gain of every fixed-N occupation;
    * chunked disorder jobs whose program hardware-sweeps positive Floquet
      depths and saves two independent raw-shot replicas of every diagonal
      return.

    Four phase settings are packed into each RAverager program in the order
    ``(theta, phi) = (0,0), (180,0), (0,90), (180,90)``.  For each replica,

    ``Q_phi = Pe(theta=0, phi) - Pe(theta=180, phi)`` and
    ``A = Q_0 - 1j*Q_90``.

    ``analyze`` divides each occupation by the independent visibility,
    constructs ``z=Tr(U)/D``, and uses the cross-replica estimator
    ``Re[z_A z_B*]``.  It never squares a noisy trace against itself.
    """

    SFF_PHASE_SETTINGS = np.asarray([
        [0.0, 0.0],
        [180.0, 0.0],
        [0.0, 90.0],
        [180.0, 90.0],
    ])

    @staticmethod
    def _positive_integer(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @classmethod
    def _validate_complete_basis(cls, occupations, swap_stors):
        occupations = np.asarray(occupations, dtype=object)
        mode_count = len(swap_stors) + 1
        if occupations.ndim != 2 or occupations.shape[1] != mode_count:
            raise ValueError(
                f"occupations must have shape (D, {mode_count})"
            )
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)) for n in occupations.reshape(-1)):
            raise TypeError("occupation entries must be integers")
        occupations = occupations.astype(int)
        if np.any(occupations < 0):
            raise ValueError("occupation entries must be non-negative")
        photon_numbers = occupations.sum(axis=1)
        if len(occupations) == 0 or np.any(
                photon_numbers != photon_numbers[0]):
            raise ValueError("all occupations must have one common photon number")
        photon_number = int(photon_numbers[0])
        if photon_number < 1:
            raise ValueError("the disorder SFF basis must be non-vacuum")
        rows = [tuple(row) for row in occupations]
        if len(set(rows)) != len(rows):
            raise ValueError("occupations contain duplicate rows")
        expected_dimension = comb(
            photon_number + mode_count - 1, photon_number)
        if len(rows) != expected_dimension:
            raise ValueError(
                "disorder SFF requires the complete fixed-N diagonal basis: "
                f"expected {expected_dimension} rows for N={photon_number}, "
                f"got {len(rows)}"
            )
        expected_rows = set(product(
            range(photon_number + 1), repeat=mode_count))
        expected_rows = {row for row in expected_rows
                         if sum(row) == photon_number}
        missing = expected_rows.difference(rows)
        extra = set(rows).difference(expected_rows)
        if missing or extra:
            raise ValueError(
                "occupations are not the complete fixed-N basis; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return occupations, photon_number

    @staticmethod
    def _phase_corrections(occupations, phase_by_occupation):
        corrections = []
        for occupation in occupations:
            key = tuple(int(n) for n in occupation)
            if key not in phase_by_occupation:
                raise ValueError(f"missing phase correction for {key}")
            phase = float(phase_by_occupation[key])
            if not np.isfinite(phase):
                raise ValueError(f"phase correction for {key} is not finite")
            corrections.append(phase)
        return np.asarray(corrections, dtype=float)

    @classmethod
    def batch(cls,
              default_expt_cfg,
              swap_stors,
              occupations,
              cycles,
              phase_by_occupation,
              realization_detunings_MHz,
              realization_seeds=None,
              realization_ids=None,
              sync_cycles=10,
              shots_per_replica=1,
              visibility_reps=1000,
              realizations_per_job=5,
              include_visibility=True):
        """Build queue configs for a direct full-basis disorder SFF campaign.

        ``cycles`` may contain zero, but its strictly positive entries must be
        a uniform integer grid.  Zero is represented by the independent
        visibility job and is not sent through the destructive RAverager loop.
        ``realization_detunings_MHz`` contains the physical detuning vector of
        every disorder realization in ``swap_stors`` order.
        """
        raw_swap_stors = list(swap_stors)
        if any(isinstance(stor, (bool, np.bool_)) or not isinstance(
                stor, (int, np.integer)) for stor in raw_swap_stors):
            raise TypeError("swap_stors entries must be integers in 1..7")
        swap_stors = [int(stor) for stor in raw_swap_stors]
        if len(swap_stors) < 1 or len(set(swap_stors)) != len(swap_stors):
            raise ValueError("swap_stors must be a nonempty list of distinct modes")
        if any(stor < 1 or stor > 7 for stor in swap_stors):
            raise ValueError("swap_stors entries must be in 1..7")
        occupations, photon_number = cls._validate_complete_basis(
            occupations, swap_stors)
        phase_corrections = cls._phase_corrections(
            occupations, phase_by_occupation)

        cycles_array = np.asarray(cycles, dtype=object)
        if cycles_array.ndim != 1 or len(cycles_array) == 0:
            raise ValueError("cycles must be a nonempty one-dimensional array")
        if any(isinstance(cycle, (bool, np.bool_)) or not isinstance(
                cycle, (int, np.integer)) for cycle in cycles_array):
            raise TypeError("cycles must contain integers")
        cycles_array = cycles_array.astype(int)
        if np.any(cycles_array < 0):
            raise ValueError("cycles must be non-negative")
        if len(np.unique(cycles_array)) != len(cycles_array) or np.any(
                np.diff(cycles_array) <= 0):
            raise ValueError("cycles must be unique and strictly increasing")
        positive_cycles = cycles_array[cycles_array > 0]
        if len(positive_cycles) == 0:
            raise ValueError("cycles must contain at least one positive depth")
        if len(positive_cycles) == 1:
            cycle_step = 1
        else:
            cycle_steps = np.diff(positive_cycles)
            if np.any(cycle_steps != cycle_steps[0]):
                raise ValueError(
                    "positive cycles must be uniformly spaced for the "
                    "RAverager depth sweep"
                )
            cycle_step = int(cycle_steps[0])

        detunings = np.asarray(realization_detunings_MHz, dtype=float)
        if detunings.ndim != 2 or detunings.shape[1] != len(swap_stors):
            raise ValueError(
                "realization_detunings_MHz must have shape "
                f"(R, {len(swap_stors)})"
            )
        if len(detunings) == 0 or not np.all(np.isfinite(detunings)):
            raise ValueError(
                "realization_detunings_MHz must be nonempty and finite")
        realization_count = len(detunings)

        if realization_ids is None:
            realization_ids = np.arange(realization_count, dtype=np.int64)
        realization_ids = np.asarray(realization_ids)
        if realization_ids.shape != (realization_count,) or not np.issubdtype(
                realization_ids.dtype, np.integer):
            raise ValueError(
                "realization_ids must be an integer vector with one entry "
                "per detuning row"
            )
        realization_ids = realization_ids.astype(np.int64)
        if len(np.unique(realization_ids)) != realization_count:
            raise ValueError("realization_ids must be unique")

        if realization_seeds is None:
            realization_seeds = realization_ids
        realization_seeds = np.asarray(realization_seeds)
        if realization_seeds.shape != (realization_count,) or not np.issubdtype(
                realization_seeds.dtype, np.integer):
            raise ValueError(
                "realization_seeds must be an integer vector with one entry "
                "per detuning row"
            )
        realization_seeds = realization_seeds.astype(np.int64)

        shots_per_replica = cls._positive_integer(
            shots_per_replica, "shots_per_replica")
        visibility_reps = cls._positive_integer(
            visibility_reps, "visibility_reps")
        realizations_per_job = cls._positive_integer(
            realizations_per_job, "realizations_per_job")
        sync_cycles = int(sync_cycles)
        if sync_cycles < 10:
            raise ValueError(
                "sync_cycles must be at least 10 for Floquet register setup"
            )

        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=2 * shots_per_replica,
            rounds=1,
            expts=len(positive_cycles),
            start=int(positive_cycles[0]),
            step=cycle_step,
            storage_reset=list(swap_stors),
            swap_stors=list(swap_stors),
            scramble_sync_cycles=sync_cycles,
            floquet_hardware_loop=False,
            update_phases=True,
            palindrome_scramble=False,
            parity_check=False,
            pre_selection_parity=False,
            pre_selection_reset=False,
            parity_readout=False,
            multiparity_readout=False,
            perform_wigner=False,
            spectroscopy_phase_correction_mode="final_analyzer",
            spectroscopy_prep_phase=0.0,
            spectroscopy_analyzer_phase=0.0,
            final_analyzer_phase_per_cycle_deg=0.0,
            sff_phase_settings=cls.SFF_PHASE_SETTINGS.tolist(),
            sff_occupations=occupations.tolist(),
            sff_phase_corrections_deg=phase_corrections.tolist(),
            sff_requested_cycles=cycles_array.tolist(),
            sff_positive_cycles=positive_cycles.tolist(),
            sff_photon_number=photon_number,
            sff_shots_per_replica=shots_per_replica,
            sff_visibility_reps=visibility_reps,
            sff_schema_version=1,
            dedupe_waveforms=True,
        ))

        configs = []
        if include_visibility:
            configs.append(dict(
                sff_job_kind="visibility",
                sff_realization_ids=[],
                sff_disorder_seeds=[],
                sff_detunings_MHz=[],
            ))
        for start in range(0, realization_count, realizations_per_job):
            stop = min(start + realizations_per_job, realization_count)
            configs.append(dict(
                sff_job_kind="disorder",
                sff_realization_ids=realization_ids[start:stop].tolist(),
                sff_disorder_seeds=realization_seeds[start:stop].tolist(),
                sff_detunings_MHz=detunings[start:stop].tolist(),
            ))

        dimension = len(occupations)
        setting_count = len(cls.SFF_PHASE_SETTINGS)
        disorder_elementary_shots = (
            realization_count * dimension * len(positive_cycles)
            * setting_count * 2 * shots_per_replica
        )
        visibility_elementary_shots = (
            dimension * setting_count * visibility_reps
            if include_visibility else 0
        )
        return AttrDict(dict(
            default_expt_cfg=defaults,
            configs=configs,
            program=DisorderSFFDepthSweepProgram,
            experiment=cls,
            occupations=occupations.copy(),
            cycles=cycles_array.copy(),
            positive_cycles=positive_cycles.copy(),
            realization_count=realization_count,
            hilbert_dimension=dimension,
            program_loads_per_realization=dimension,
            disorder_elementary_shots=disorder_elementary_shots,
            visibility_elementary_shots=visibility_elementary_shots,
            total_elementary_shots=(
                disorder_elementary_shots
                + visibility_elementary_shots
            ),
        ))

    @staticmethod
    def _read_num(cfg):
        read_num = 1
        if cfg.expt.get("parity_check", False):
            read_num += 1
        if cfg.expt.get("active_reset", False):
            params = MMRAveragerProgram.get_active_reset_params(cfg)
            read_num += MMRAveragerProgram.active_reset_read_num(**params)
        if cfg.expt.get("multiparity_readout", False):
            read_num += 1
        return read_num

    @classmethod
    def _replica_returns_from_raw(cls, prog, avgi, avgq,
                                  cycle_count, reps, read_num, cfg):
        setting_count = len(cls.SFF_PHASE_SETTINGS)
        idata, qdata = prog.collect_shots()
        expected_values = cycle_count * reps * setting_count * read_num
        if np.asarray(idata).size != expected_values:
            raise RuntimeError(
                "unexpected SFF I-buffer size: "
                f"got {np.asarray(idata).size}, expected {expected_values}"
            )
        if np.asarray(qdata).size != expected_values:
            raise RuntimeError(
                "unexpected SFF Q-buffer size: "
                f"got {np.asarray(qdata).size}, expected {expected_values}"
            )

        final_i = np.asarray(idata, dtype=float).reshape(
            1, cycle_count, reps, setting_count, read_num
        )[0, ..., -1]
        np.asarray(qdata, dtype=float).reshape(
            1, cycle_count, reps, setting_count, read_num
        )

        saved_i = np.empty((cycle_count, setting_count), dtype=float)
        for setting_index in range(setting_count):
            readout_index = (setting_index + 1) * read_num - 1
            values = np.asarray(
                avgi[0][readout_index], dtype=float).reshape(-1)
            if len(values) != cycle_count:
                raise RuntimeError(
                    "unexpected SFF averaged-I shape for readout lane "
                    f"{readout_index}: got {values.shape}, expected "
                    f"({cycle_count},)"
                )
            saved_i[:, setting_index] = values

        # collect_shots and firmware averages can differ by a fixed ADC offset.
        # Preserve shot fluctuations while forcing their complete mean back to
        # the saved averaged value, exactly as subsample_spectroscopy_shots.
        raw_mean = np.mean(final_i, axis=1)
        corrected_i = final_i + (
            saved_i - raw_mean)[:, np.newaxis, :]

        q = int(cfg.expt.qubits[0])
        Ig = float(np.asarray(cfg.device.readout.Ig[q]).reshape(-1)[0])
        Ie = float(np.asarray(cfg.device.readout.Ie[q]).reshape(-1)[0])
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        Pe = (corrected_i - Ig) / (Ie - Ig)

        replica_pe = np.asarray([
            np.mean(Pe[:, 0::2, :], axis=1),
            np.mean(Pe[:, 1::2, :], axis=1),
        ])
        settings = [tuple(setting) for setting in cls.SFF_PHASE_SETTINGS]
        setting_index = {setting: index
                         for index, setting in enumerate(settings)}
        q0 = (
            replica_pe[..., setting_index[(0.0, 0.0)]]
            - replica_pe[..., setting_index[(180.0, 0.0)]]
        )
        q90 = (
            replica_pe[..., setting_index[(0.0, 90.0)]]
            - replica_pe[..., setting_index[(180.0, 90.0)]]
        )
        return q0 - 1j * q90

    def _acquire_visibility(self, progress=False, debug=False):
        ecfg = self.cfg.expt
        occupations = np.asarray(ecfg.sff_occupations, dtype=int)
        corrections = np.asarray(
            ecfg.sff_phase_corrections_deg, dtype=float)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        read_num = self._read_num(self.cfg)
        self.cfg.read_num = read_num
        visibility = np.empty(len(occupations), dtype=complex)
        q = int(ecfg.qubits[0])
        Ig = float(np.asarray(self.cfg.device.readout.Ig[q]).reshape(-1)[0])
        Ie = float(np.asarray(self.cfg.device.readout.Ie[q]).reshape(-1)[0])
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")

        ecfg.reps = int(ecfg.sff_visibility_reps)
        ecfg.rounds = 1
        self.cfg.reps = int(ecfg.reps)
        self.cfg.rounds = 1
        ecfg.floquet_cycle = 0
        ecfg.detunings = [0.0] * len(swap_stors)
        for occupation_index in tqdm(
                range(len(occupations)), disable=not progress):
            ecfg.spectroscopy_occupations = occupations[
                occupation_index].tolist()
            ecfg.spectroscopy_final_occupations = occupations[
                occupation_index].tolist()
            ecfg.final_analyzer_phase_per_cycle_deg = float(
                corrections[occupation_index])
            Pe = {}
            for theta, phi in self.SFF_PHASE_SETTINGS:
                ecfg.spectroscopy_prep_phase = float(theta)
                ecfg.spectroscopy_analyzer_phase = float(phi)
                prog = NPhotonHamiltonianSpectroscopyProgram(
                    soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = prog.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                    debug=debug,
                    readouts_per_experiment=read_num,
                )
                signal = float(np.asarray(avgi[0][-1]).reshape(-1)[0])
                Pe[(float(theta), float(phi))] = (
                    signal - Ig) / (Ie - Ig)
                self.prog = prog
            q0 = Pe[(0.0, 0.0)] - Pe[(180.0, 0.0)]
            q90 = Pe[(0.0, 90.0)] - Pe[(180.0, 90.0)]
            visibility[occupation_index] = q0 - 1j * q90

        data = AttrDict(dict(
            sff_schema_version=np.asarray([1], dtype=int),
            sff_job_kind_code=np.asarray([0], dtype=np.int8),
            occupations=occupations,
            swap_stors=np.asarray(swap_stors, dtype=int),
            phase_corrections_deg=corrections,
            phase_settings=self.SFF_PHASE_SETTINGS.copy(),
            cycles=np.asarray([0], dtype=int),
            visibility_real=np.real(visibility),
            visibility_imag=np.imag(visibility),
            visibility_reps=np.asarray(
                [int(ecfg.sff_visibility_reps)], dtype=int),
        ))
        self.data = data
        return data

    def _acquire_disorder(self, progress=False, debug=False):
        ecfg = self.cfg.expt
        occupations = np.asarray(ecfg.sff_occupations, dtype=int)
        corrections = np.asarray(
            ecfg.sff_phase_corrections_deg, dtype=float)
        cycles = np.asarray(ecfg.sff_positive_cycles, dtype=int)
        realization_ids = np.asarray(ecfg.sff_realization_ids, dtype=np.int64)
        seeds = np.asarray(ecfg.sff_disorder_seeds, dtype=np.int64)
        detunings = np.asarray(ecfg.sff_detunings_MHz, dtype=float)
        if detunings.shape != (len(realization_ids), len(ecfg.swap_stors)):
            raise ValueError(
                "this disorder job has inconsistent realization IDs and "
                "detuning rows"
            )
        if seeds.shape != realization_ids.shape:
            raise ValueError(
                "this disorder job has inconsistent realization IDs and seeds"
            )

        read_num = self._read_num(self.cfg)
        self.cfg.read_num = read_num
        reps = int(ecfg.reps)
        shots_per_replica = int(ecfg.sff_shots_per_replica)
        self.cfg.reps = reps
        self.cfg.rounds = 1
        self.cfg.expts = len(cycles)
        self.cfg.start = int(cycles[0])
        self.cfg.step = int(cycles[1] - cycles[0]) \
            if len(cycles) > 1 else int(ecfg.step)
        if reps != 2 * shots_per_replica:
            raise ValueError(
                "reps must equal 2*sff_shots_per_replica for A/B pairing"
            )
        if ecfg.get("pre_selection_reset", False):
            raise ValueError("SFF A/B replicas do not support pre_selection_reset")

        returns = np.empty(
            (len(realization_ids), 2, len(occupations), len(cycles)),
            dtype=complex,
        )
        floquet_cycle_us = None
        for realization_index in tqdm(
                range(len(realization_ids)), disable=not progress):
            ecfg.detunings = detunings[realization_index].tolist()
            for occupation_index, occupation in enumerate(occupations):
                ecfg.spectroscopy_occupations = occupation.tolist()
                ecfg.spectroscopy_final_occupations = occupation.tolist()
                ecfg.final_analyzer_phase_per_cycle_deg = float(
                    corrections[occupation_index])
                self.prog = self.ProgramClass(
                    soccfg=self.soccfg, cfg=self.cfg)
                xpts, avgi, avgq = self.prog.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                    debug=debug,
                    readouts_per_experiment=(
                        len(self.SFF_PHASE_SETTINGS) * read_num),
                )
                if not np.array_equal(
                        np.asarray(xpts, dtype=int), cycles):
                    raise RuntimeError(
                        "SFF RAverager returned the wrong cycle grid: "
                        f"got {np.asarray(xpts)}, expected {cycles}"
                    )
                returns[realization_index, :, occupation_index, :] = (
                    self._replica_returns_from_raw(
                        self.prog, avgi, avgq,
                        cycle_count=len(cycles),
                        reps=reps,
                        read_num=read_num,
                        cfg=self.cfg,
                    )
                )
                this_cycle_us = float(self.prog.floquet_cycle_us)
                if floquet_cycle_us is None:
                    floquet_cycle_us = this_cycle_us
                elif not np.isclose(floquet_cycle_us, this_cycle_us):
                    raise RuntimeError(
                        "Floquet cycle duration changed within one SFF job"
                    )

        data = AttrDict(dict(
            sff_schema_version=np.asarray([1], dtype=int),
            sff_job_kind_code=np.asarray([1], dtype=np.int8),
            realization_ids=realization_ids,
            disorder_seeds=seeds,
            detunings_MHz=detunings,
            occupations=occupations,
            swap_stors=np.asarray(ecfg.swap_stors, dtype=int),
            phase_corrections_deg=corrections,
            phase_settings=self.SFF_PHASE_SETTINGS.copy(),
            cycles=cycles,
            replica_returns_real=np.real(returns),
            replica_returns_imag=np.imag(returns),
            shots_per_replica=np.asarray(
                [shots_per_replica], dtype=int),
            floquet_cycle_us=np.asarray(
                [floquet_cycle_us], dtype=float),
        ))
        self.data = data
        return data

    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)
        kind = str(self.cfg.expt.sff_job_kind).lower()
        if kind == "visibility":
            return self._acquire_visibility(
                progress=progress, debug=debug)
        if kind == "disorder":
            return self._acquire_disorder(
                progress=progress, debug=debug)
        raise ValueError(
            "sff_job_kind must be 'visibility' or 'disorder'"
        )

    @staticmethod
    def _visibility_vector(visibility):
        if visibility is None:
            return None
        if not isinstance(visibility, dict) and hasattr(visibility, "data"):
            visibility = visibility.data
        if isinstance(visibility, dict):
            if "visibility_real" not in visibility or \
                    "visibility_imag" not in visibility:
                raise ValueError(
                    "visibility data needs visibility_real and visibility_imag"
                )
            return (
                np.asarray(visibility["visibility_real"], dtype=float)
                + 1j * np.asarray(
                    visibility["visibility_imag"], dtype=float)
            )
        return np.asarray(visibility, dtype=complex)

    @staticmethod
    def _visibility_occupation_order(visibility):
        if visibility is None:
            return None
        if not isinstance(visibility, dict) and hasattr(visibility, "data"):
            visibility = visibility.data
        if isinstance(visibility, dict) and "occupations" in visibility:
            return np.asarray(visibility["occupations"], dtype=int)
        return None

    @classmethod
    def analyze_ensemble(cls, expts, visibility=None,
                         bootstrap_samples=0, bootstrap_seed=None,
                         include_zero=True):
        """Analyze completed child jobs with an unbiased cross-replica SFF.

        The reported ``sff_full`` is normalized by ``D**2``.  The unnormalized
        convention is also stored.  ``sff_disconnected`` uses different
        disorder realizations in its cross product, so ``sff_connected`` is
        not contaminated by the same-realization term.  Connected quantities
        require a common phase convention across realizations; ``sff_full``
        itself is invariant under a realization-wide phase.
        """
        if hasattr(expts, "batch_expts"):
            expts = expts.batch_expts
        elif not isinstance(expts, (list, tuple, set)):
            expts = [expts]
        expts = list(flatten_exp_lists(expts))
        if not expts:
            raise ValueError("SFF experiment jobs cannot be empty")
        visibility_candidates = []
        disorder_expts = []
        for expt in expts:
            schema_version = int(np.asarray(
                expt.data.get("sff_schema_version", [-1])
            ).reshape(-1)[0])
            if schema_version != 1:
                raise ValueError(
                    f"unsupported SFF schema version {schema_version}"
                )
            code = int(np.asarray(
                expt.data.get("sff_job_kind_code", [-1])).reshape(-1)[0])
            if code == 0:
                visibility_candidates.append((
                    cls._visibility_vector(expt.data),
                    cls._visibility_occupation_order(expt.data),
                ))
            elif code == 1:
                disorder_expts.append(expt)
            else:
                raise ValueError("an input job is not a disorder SFF job")
        if not disorder_expts:
            raise ValueError("no disorder SFF child jobs were supplied")

        visibility_vector = cls._visibility_vector(visibility)
        visibility_occupations = cls._visibility_occupation_order(
            visibility)
        if visibility_vector is None:
            if not visibility_candidates:
                raise ValueError(
                    "an independent depth-zero visibility job is required"
                )
            visibility_vector, visibility_occupations = (
                visibility_candidates[0])
            for candidate, candidate_occupations in visibility_candidates[1:]:
                if not np.allclose(candidate, visibility_vector) or not \
                        np.array_equal(
                            candidate_occupations, visibility_occupations):
                    raise ValueError(
                        "multiple visibility jobs disagree; select one "
                        "explicitly"
                    )

        reference_occupations = np.asarray(
            disorder_expts[0].data["occupations"], dtype=int)
        reference_cycles = np.asarray(
            disorder_expts[0].data["cycles"], dtype=int)
        reference_swap_stors = np.asarray(
            disorder_expts[0].data["swap_stors"], dtype=int)
        reference_phase_corrections = np.asarray(
            disorder_expts[0].data["phase_corrections_deg"], dtype=float)
        reference_phase_settings = np.asarray(
            disorder_expts[0].data["phase_settings"], dtype=float)
        if reference_phase_corrections.shape != (
                len(reference_occupations),) or not np.all(
                    np.isfinite(reference_phase_corrections)):
            raise ValueError(
                "phase_corrections_deg must contain one finite value per "
                "occupation"
            )
        if not np.array_equal(
                reference_phase_settings, cls.SFF_PHASE_SETTINGS):
            raise ValueError("saved SFF phase-setting order is not supported")
        if visibility_occupations is not None and not np.array_equal(
                visibility_occupations, reference_occupations):
            raise ValueError(
                "visibility occupation order does not match disorder jobs"
            )
        all_ids = []
        all_seeds = []
        all_detunings = []
        all_returns = []
        cycle_us_values = []
        for expt in disorder_expts:
            data = expt.data
            occupations = np.asarray(data["occupations"], dtype=int)
            cycles = np.asarray(data["cycles"], dtype=int)
            swap_stors = np.asarray(data["swap_stors"], dtype=int)
            phase_corrections = np.asarray(
                data["phase_corrections_deg"], dtype=float)
            phase_settings = np.asarray(
                data["phase_settings"], dtype=float)
            if not np.array_equal(occupations, reference_occupations):
                raise ValueError("SFF child jobs use different occupation bases")
            if not np.array_equal(cycles, reference_cycles):
                raise ValueError("SFF child jobs use different cycle grids")
            if not np.array_equal(swap_stors, reference_swap_stors):
                raise ValueError("SFF child jobs use different swap_stors")
            if not np.allclose(
                    phase_corrections, reference_phase_corrections):
                raise ValueError(
                    "SFF child jobs use different phase corrections"
                )
            if not np.array_equal(
                    phase_settings, reference_phase_settings):
                raise ValueError("SFF child jobs use different phase settings")
            returns = (
                np.asarray(data["replica_returns_real"], dtype=float)
                + 1j * np.asarray(
                    data["replica_returns_imag"], dtype=float)
            )
            if returns.ndim != 4 or returns.shape[1:] != (
                    2, len(reference_occupations), len(reference_cycles)):
                raise ValueError(
                    "replica returns must have shape (R, 2, D, Q)"
                )
            realization_ids_job = np.asarray(
                data["realization_ids"], dtype=np.int64)
            disorder_seeds_job = np.asarray(
                data["disorder_seeds"], dtype=np.int64)
            detunings_job = np.asarray(
                data["detunings_MHz"], dtype=float)
            if realization_ids_job.shape != (len(returns),) or \
                    disorder_seeds_job.shape != (len(returns),):
                raise ValueError(
                    "realization IDs and seeds must match the return count"
                )
            if detunings_job.shape != (
                    len(returns), len(reference_swap_stors)) or not np.all(
                        np.isfinite(detunings_job)):
                raise ValueError(
                    "detunings_MHz has the wrong shape or non-finite values"
                )
            all_returns.append(returns)
            all_ids.append(realization_ids_job)
            all_seeds.append(disorder_seeds_job)
            all_detunings.append(detunings_job)
            cycle_us_values.append(float(np.asarray(
                data["floquet_cycle_us"]).reshape(-1)[0]))

        returns = np.concatenate(all_returns, axis=0)
        realization_ids = np.concatenate(all_ids)
        seeds = np.concatenate(all_seeds)
        detunings = np.concatenate(all_detunings, axis=0)
        if len(np.unique(realization_ids)) != len(realization_ids):
            raise ValueError("duplicate realization IDs appear across SFF jobs")
        if not np.all(np.isfinite(returns)):
            raise ValueError("SFF replica returns contain non-finite values")
        if not np.allclose(cycle_us_values, cycle_us_values[0]):
            raise ValueError("SFF child jobs use different Floquet cycle times")
        if returns.shape[0] < 2:
            raise ValueError(
                "at least two disorder realizations are required for the "
                "connected/disconnected SFF decomposition"
            )

        visibility_vector = np.asarray(
            visibility_vector, dtype=complex).reshape(-1)
        if visibility_vector.shape != (len(reference_occupations),):
            raise ValueError(
                "visibility must contain one complex gain per occupation"
            )
        if not np.all(np.isfinite(visibility_vector)) or np.any(
                np.isclose(np.abs(visibility_vector), 0.0)):
            raise ValueError("visibility gains must be finite and nonzero")

        normalized_returns = (
            returns / visibility_vector[np.newaxis, np.newaxis, :, np.newaxis]
        )
        dimension = len(reference_occupations)
        trace_replicas = np.sum(normalized_returns, axis=2) / dimension
        zA = trace_replicas[:, 0, :]
        zB = trace_replicas[:, 1, :]
        realization_count = len(zA)
        same_realization_pair = zA * np.conj(zB)
        full_complex = np.mean(same_realization_pair, axis=0)
        disconnected_complex = (
            np.sum(zA, axis=0) * np.conj(np.sum(zB, axis=0))
            - np.sum(same_realization_pair, axis=0)
        ) / (realization_count * (realization_count - 1))
        connected_complex = full_complex - disconnected_complex
        naive = 0.5 * np.mean(
            np.abs(zA) ** 2 + np.abs(zB) ** 2, axis=0)
        full_standard_error = np.std(
            np.real(same_realization_pair), axis=0, ddof=1
        ) / np.sqrt(realization_count)

        analysis_cycles = reference_cycles.copy()
        full = np.real(full_complex)
        disconnected = np.real(disconnected_complex)
        connected = np.real(connected_complex)
        pair_imaginary = np.imag(full_complex)
        disconnected_imaginary = np.imag(disconnected_complex)
        noise_bias = naive - full
        bootstrap_low = np.asarray([], dtype=float)
        bootstrap_high = np.asarray([], dtype=float)
        bootstrap_connected_low = np.asarray([], dtype=float)
        bootstrap_connected_high = np.asarray([], dtype=float)
        bootstrap_samples = int(bootstrap_samples)
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative")
        if bootstrap_samples:
            rng = np.random.default_rng(bootstrap_seed)
            bootstrap_full = np.empty(
                (bootstrap_samples, len(reference_cycles)), dtype=float)
            bootstrap_connected = np.empty_like(bootstrap_full)
            for sample_index in range(bootstrap_samples):
                indices = rng.integers(
                    0, realization_count, size=realization_count)
                sample_A = zA[indices]
                sample_B = zB[indices]
                sample_pair = sample_A * np.conj(sample_B)
                sample_full = np.mean(sample_pair, axis=0)
                sample_disconnected = (
                    np.sum(sample_A, axis=0)
                    * np.conj(np.sum(sample_B, axis=0))
                    - np.sum(sample_pair, axis=0)
                ) / (realization_count * (realization_count - 1))
                bootstrap_full[sample_index] = np.real(sample_full)
                bootstrap_connected[sample_index] = np.real(
                    sample_full - sample_disconnected)
            bootstrap_low, bootstrap_high = np.percentile(
                bootstrap_full, [2.5, 97.5], axis=0)
            bootstrap_connected_low, bootstrap_connected_high = np.percentile(
                bootstrap_connected, [2.5, 97.5], axis=0)

        if include_zero and (len(analysis_cycles) == 0
                             or analysis_cycles[0] != 0):
            analysis_cycles = np.concatenate(([0], analysis_cycles))
            full = np.concatenate(([1.0], full))
            disconnected = np.concatenate(([1.0], disconnected))
            connected = np.concatenate(([0.0], connected))
            naive = np.concatenate(([1.0], naive))
            noise_bias = np.concatenate(([0.0], noise_bias))
            pair_imaginary = np.concatenate(([0.0], pair_imaginary))
            disconnected_imaginary = np.concatenate(
                ([0.0], disconnected_imaginary))
            full_standard_error = np.concatenate(
                ([0.0], full_standard_error))
            trace_replicas = np.concatenate((
                np.ones(
                    (realization_count, 2, 1), dtype=complex),
                trace_replicas,
            ), axis=2)
            if bootstrap_samples:
                bootstrap_low = np.concatenate(([1.0], bootstrap_low))
                bootstrap_high = np.concatenate(([1.0], bootstrap_high))
                bootstrap_connected_low = np.concatenate(
                    ([0.0], bootstrap_connected_low))
                bootstrap_connected_high = np.concatenate(
                    ([0.0], bootstrap_connected_high))

        result = AttrDict(dict(
            sff_schema_version=np.asarray([1], dtype=int),
            cycles=analysis_cycles,
            time_us=analysis_cycles * cycle_us_values[0],
            occupations=reference_occupations,
            swap_stors=reference_swap_stors,
            phase_corrections_deg=reference_phase_corrections,
            phase_settings=reference_phase_settings,
            realization_ids=realization_ids,
            disorder_seeds=seeds,
            detunings_MHz=detunings,
            visibility_real=np.real(visibility_vector),
            visibility_imag=np.imag(visibility_vector),
            trace_replica_real=np.real(trace_replicas),
            trace_replica_imag=np.imag(trace_replicas),
            sff_full=full,
            sff_disconnected=disconnected,
            sff_connected=connected,
            sff_naive=naive,
            sff_noise_bias=noise_bias,
            sff_pair_imaginary=pair_imaginary,
            sff_disconnected_imaginary=disconnected_imaginary,
            sff_full_standard_error=full_standard_error,
            sff_full_unnormalized=full * dimension ** 2,
            sff_disconnected_unnormalized=(
                disconnected * dimension ** 2),
            sff_connected_unnormalized=connected * dimension ** 2,
            bootstrap_95_low=bootstrap_low,
            bootstrap_95_high=bootstrap_high,
            bootstrap_connected_95_low=bootstrap_connected_low,
            bootstrap_connected_95_high=bootstrap_connected_high,
            bootstrap_samples=np.asarray([bootstrap_samples], dtype=int),
            hilbert_dimension=np.asarray([dimension], dtype=int),
            realization_count=np.asarray(
                [realization_count], dtype=int),
            floquet_cycle_us=np.asarray(
                [cycle_us_values[0]], dtype=float),
        ))
        return result

    def analyze(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        if hasattr(self, "batch_expts"):
            self.data = self.analyze_ensemble(
                self.batch_expts,
                visibility=kwargs.get("visibility", None),
                bootstrap_samples=kwargs.get("bootstrap_samples", 0),
                bootstrap_seed=kwargs.get("bootstrap_seed", None),
                include_zero=kwargs.get("include_zero", True),
            )
        return self.data

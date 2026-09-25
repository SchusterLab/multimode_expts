# -*- coding: utf-8 -*-
"""Playing calibrated M1-Sx swap trains, and the frame they leave behind.

This is the Floquet drive itself: the repeated, ordered train of calibrated
beamsplitter pulses between the manipulate mode and the storage modes, plus
the bookkeeping that makes a train composable with the load and read
sequences around it.

The contract every method here honours
--------------------------------------
A train is played *against* a caller-owned phase ledger (see
``floquet_phase_frame``). It emits each pulse at the ledger's current phase
for the mode being driven, and advances the ledger after every pulse. Load,
scramble and read therefore share one mutable list, in time order, and the
read is only correct because the scramble left the ledger where the read
expects it. Nothing here may reset, copy-and-discard, or reorder that ledger.

Three ways to emit the same train
---------------------------------
* the software loop -- Python emits every pulse, one ``setup_and_pulse`` each;
* ``floquet_hardware_loop`` -- one ``loopnz`` over one cycle's worth of
  instructions, with the per-cycle phase step held in a tProc register, for
  depths the instruction memory cannot hold unrolled;
* preloaded register banks (``floquet_register_bank``) -- pulse settings
  written once, one raw ``set`` per repeat.

They must agree. The hardware loop computes cycle 0's phases and the
per-cycle step in Python, then advances the ledger by ``cycles * step`` at
the end, which is only equal to the unrolled result because the step is
constant -- which is why it refuses ``palindrome_scramble``.

Requirements on the host program: ``m1s_kwargs``, ``m1s_waveform_mode``,
``m1s_ch``, ``m1s_length``, ``m1s_is_low_freq``, ``swap_ds``,
``storage_phase_matrix``, ``decoder_phase_matrix``, and the qick program
methods.
"""
from copy import deepcopy

from experiments.floquet_timing import floquet_cycle_us
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


class FloquetTrain:
    """Mixin: see the module docstring."""

    def calculate_floquet_cycle_us(self, swap_stors=None):
        """Scheduled cycle duration, including QICK v1 sync_all quantization.

        Thin adapter over ``floquet_timing.floquet_cycle_us``: it supplies
        this program's channels, envelopes and firmware conversions. The
        arithmetic is shared with the offline resolver on purpose -- the
        cycle time divides into every coupling rate, so the two drifting
        apart would read as a wrong Hamiltonian, not as an error.
        """
        ecfg = self.cfg.expt
        if swap_stors is None:
            swap_stors = ecfg.swap_stors
        return floquet_cycle_us(
            swap_stors,
            swap_ds=self.swap_ds,
            waveform_modes=self.m1s_waveform_mode,
            channels=self.m1s_ch,
            lengths=self.m1s_length,
            ramp_cycles=[self.pi_m1_sigma_low if is_low
                         else self.pi_m1_sigma_high
                         for is_low in self.m1s_is_low_freq],
            sync_cycles=int(ecfg.get("scramble_sync_cycles", 10)),
            us2cycles=lambda us, ch: self.us2cycles(us, gen_ch=ch),
            cycles2us=self.cycles2us,
            clock_ratio=lambda ch: (float(self.tproccfg["f_time"])
                                    / float(self.soccfg["gens"][ch]["f_fabric"])),
            gauss_sigma_override=ecfg.get("floquet_gauss_sigma", None),
        )

    def _mod360(self, phase_deg):
        return mod360(phase_deg)

    def _advance_phase_offsets(self, phase_offsets, swap_stors, pulsed_stor):
        """
        Preserve the existing phase tracking between Floquet pulses.

        This uses only the Floquet-to-Floquet matrix in ds_floquet:

            phase[stor_B] += get_phase_from("M1-S{stor_B}", "M1-S{pulsed_stor}")

        The separate decoder_phase_matrix is directional and must never be
        used here. It is applied only to the decoder storage swaps and f0g1.
        """
        advance_floquet_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            pulsed_stor=pulsed_stor,
            swap_ds=self.swap_ds,
        )

    def _play_closed_floquet_cycle_pairs(
            self, n_cycle_pair, phase_offsets, swap_stors,
            extra_forward=False):
        update_phases = bool(self.cfg.expt.get("update_phases", True))
        sync_cycles = int(
            self.cfg.expt.get("scramble_sync_cycles", 10))
        pulse_args_by_stor = {
            stor: deepcopy(self.m1s_kwargs[stor - 1])
            for stor in swap_stors
        }
        if self.cfg.expt.zero_floquet_gain:
            for pulse_args in pulse_args_by_stor.values():
                pulse_args["gain"] = 0

        register_bank = {}
        preloaded_stors = [stor for stor in swap_stors
                           if self.m1s_waveform_mode[stor - 1] == "preload_flattop"]
        if preloaded_stors and (n_cycle_pair > 0 or extra_forward):
            bank, _, _ = _prepare_preloaded_floquet_register_bank(
                self, [pulse_args_by_stor[stor] for stor in preloaded_stors],
                [0.0] * len(preloaded_stors), [0.0] * len(preloaded_stors))
            register_bank = dict(zip(preloaded_stors, bank))

        self.sync_all()

        for pair_index in range(n_cycle_pair):
            forward_phases = []
            for stor in swap_stors:
                stor_index = swap_stors.index(stor)
                phase_deg = self._mod360(
                    phase_offsets[stor_index])
                forward_phases.append((stor, phase_deg))

                pulse_args = pulse_args_by_stor[stor]
                pulse_args["phase"] = self.deg2reg(
                    phase_deg, gen_ch=pulse_args["ch"])
                if stor in register_bank:
                    entry = register_bank[stor]
                    self.safe_regwi(entry["register_page"], entry["phase_register"], pulse_args["phase"])
                    _play_preloaded_floquet_register_bank_entry(self, entry)
                else:
                    self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)

                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

            # Replay the actual forward control phases in reverse order and
            # add 180 degrees.  Recomputing the inverse phases from the live
            # tracker would not be the adjoint of the emitted forward cycle.
            for stor, forward_phase_deg in reversed(forward_phases):
                inverse_phase_deg = self._mod360(
                    forward_phase_deg + 180.0
                )
                pulse_args = pulse_args_by_stor[stor]
                pulse_args["phase"] = self.deg2reg(
                    inverse_phase_deg, gen_ch=pulse_args["ch"])
                if stor in register_bank:
                    entry = register_bank[stor]
                    self.safe_regwi(entry["register_page"], entry["phase_register"], pulse_args["phase"])
                    _play_preloaded_floquet_register_bank_entry(self, entry)
                else:
                    self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)

                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

            if self.cfg.expt.get("debug", False):
                print(
                    "[EntireCyclePhase] pair=",
                    pair_index,
                    "zero Floquet gain=",
                    self.cfg.expt.zero_floquet_gain,
                    "forward phases=",
                    forward_phases,
                )

        if extra_forward:
            for stor in swap_stors:
                stor_index = swap_stors.index(stor)
                phase_deg = self._mod360(phase_offsets[stor_index])
                pulse_args = pulse_args_by_stor[stor]
                pulse_args["phase"] = self.deg2reg(
                    phase_deg, gen_ch=pulse_args["ch"])
                if stor in register_bank:
                    entry = register_bank[stor]
                    self.safe_regwi(entry["register_page"], entry["phase_register"], pulse_args["phase"])
                    _play_preloaded_floquet_register_bank_entry(self, entry)
                else:
                    self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)
                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )
        self.sync_all()

    def _advance_storage_phase_offsets(
            self, phase_offsets, swap_stors, pulsed_stor):
        """Advance later ds_storage swap phases after one ds_storage swap.

        Unlike the legacy Floquet matrix, this matrix may have a calibrated
        diagonal: ``matrix[i, i]`` is the active-access phase of mode i.
        """
        if self.storage_phase_matrix is None:
            return

        advance_matrix_offsets(
            offsets=phase_offsets,
            matrix=self.storage_phase_matrix,
            pulsed_column=swap_stors.index(pulsed_stor),
        )

    def _advance_decoder_phase_offsets(
            self, decoder_phase_offsets, swap_stors, pulsed_stor):
        """Advance every decoder axis after one physical Floquet pulse."""
        advance_matrix_offsets(
            offsets=decoder_phase_offsets,
            matrix=self.decoder_phase_matrix,
            pulsed_column=swap_stors.index(pulsed_stor),
        )

    def _play_scramble_with_phase_offsets(
        self,
        phase_offsets,
        swap_stors,
        disorder_phase_offsets=None,
        decoder_phase_offsets=None,
    ):
        """Play ``floquet_cycle`` cycles of the ordered Floquet train.

        ``phase_offsets`` is the caller's live phase ledger (module
        docstring): each pulse plays at the ledger's phase for its mode, and
        with ``update_phases`` the ledger advances after every pulse. Load,
        scramble and read share it, in time order.

        - ``disorder_phase_offsets`` advances by the phase each detuning
          (``cfg.expt.detunings``) builds up over the whole scramble.
        - ``decoder_phase_offsets``, if given, advances by
          ``decoder_phase_matrix`` after every pulse.

        Settings from ``cfg.expt``: ``floquet_cycle``, ``detunings``,
        ``update_phases``, ``scramble_sync_cycles``, and two ways to play:

        - ``floquet_hardware_loop``: one tProc loop over one cycle
          (:meth:`_play_floquet_hardware_loop`);
        - otherwise every pulse is emitted (:meth:`_play_floquet_software_loop`),
          and ``palindrome_scramble`` plays every second cycle in reverse order.
        """
        ecfg = self.cfg.expt
        swap_stors = list(swap_stors)
        if disorder_phase_offsets is None:
            disorder_phase_offsets = [0.0] * len(swap_stors)
        self._check_scramble_ledgers(phase_offsets, swap_stors,
                                     disorder_phase_offsets, decoder_phase_offsets)
        detunings = self._scramble_detunings(swap_stors)

        update_phases = ecfg.get("update_phases", True)
        scramble_sync_cycles = int(ecfg.get("scramble_sync_cycles", 10))
        palindrome_scramble = bool(ecfg.get("palindrome_scramble", False))
        floquet_hardware_loop = bool(ecfg.get("floquet_hardware_loop", False))
        floquet_cycle = int(ecfg.floquet_cycle)
        if floquet_cycle < 0:
            raise ValueError("floquet_cycle must be non-negative")
        if floquet_hardware_loop and palindrome_scramble:
            raise ValueError(
                "floquet_hardware_loop does not support palindrome_scramble"
            )

        all_pulse_args = self._detuned_floquet_pulse_args(swap_stors, detunings)
        scramble_elapsed_us = floquet_cycle * self.calculate_floquet_cycle_us(swap_stors)

        self.sync_all()
        if ecfg.get("debug", False):
            print("[DarkScramble] using shared phase_offsets for scramble")
            print("[DarkScramble] initial phase_offsets:", phase_offsets)
            print("[DarkScramble] initial disorder_phase_offsets:", disorder_phase_offsets)
            print("[DarkScramble] detunings MHz:", detunings)
            print("[DarkScramble] scramble sync cycles:", scramble_sync_cycles)
            print("[DarkScramble] palindrome scramble:", palindrome_scramble)
            print("[DarkScramble] hardware loop:", floquet_hardware_loop)
            print("[DarkScramble] forward sequence:", swap_stors)
            if palindrome_scramble:
                print("[DarkScramble] reverse sequence:", list(reversed(swap_stors)))
                if floquet_cycle % 2:
                    print(
                        "[DarkScramble] odd floquet_cycle leaves one unpaired "
                        "forward cycle"
                    )
            print("[DarkScramble] scramble elapsed us:", scramble_elapsed_us)
            print("[DarkScramble] pulse args:", all_pulse_args)

        if floquet_hardware_loop and floquet_cycle > 0 and swap_stors:
            self._play_floquet_hardware_loop(
                all_pulse_args, phase_offsets, swap_stors, floquet_cycle,
                update_phases, scramble_sync_cycles, decoder_phase_offsets)
        else:
            self._play_floquet_software_loop(
                all_pulse_args, phase_offsets, swap_stors, floquet_cycle,
                update_phases, scramble_sync_cycles, palindrome_scramble,
                decoder_phase_offsets)

        for j_stor, detuning_MHz in enumerate(detunings):
            disorder_phase_offsets[j_stor] = self._mod360(
                disorder_phase_offsets[j_stor]
                + detuning_phase_deg(detuning_MHz, scramble_elapsed_us)
            )

        if ecfg.get("debug", False):
            print("[DarkScramble] final phase_offsets:", phase_offsets)
            print("[DarkScramble] final disorder_phase_offsets:", disorder_phase_offsets)

        self.sync_all()

    @staticmethod
    def _check_scramble_ledgers(phase_offsets, swap_stors,
                                disorder_phase_offsets, decoder_phase_offsets):
        """Each ledger has one entry per mode (decoder: one more, for M1)."""
        if len(phase_offsets) != len(swap_stors):
            raise ValueError(
                f"phase_offsets length {len(phase_offsets)} does not match "
                f"swap_stors length {len(swap_stors)}"
            )
        if decoder_phase_offsets is not None \
                and len(decoder_phase_offsets) != len(swap_stors) + 1:
            raise ValueError(
                "decoder_phase_offsets must be ordered as "
                "[M1 photon lowering, M1-S4, ...]"
            )
        if len(disorder_phase_offsets) != len(swap_stors):
            raise ValueError(
                f"disorder_phase_offsets length {len(disorder_phase_offsets)} "
                f"does not match swap_stors length {len(swap_stors)}"
            )

    def _scramble_detunings(self, swap_stors):
        """-> ``cfg.expt.detunings`` in MHz, one per mode; zeros if unset."""
        raw_detunings = self.cfg.expt.get("detunings", None)
        if raw_detunings is None or raw_detunings is False \
                or len(raw_detunings) == 0:
            return [0.0] * len(swap_stors)
        if len(raw_detunings) != len(swap_stors):
            raise AssertionError(
                "length of detunings doesn't match that of swap_stors"
            )
        return [float(d) for d in raw_detunings]

    def _detuned_floquet_pulse_args(self, swap_stors, detunings):
        """-> copies of the calibrated pulse arguments, with the detunings added."""
        all_pulse_args = []
        for stor, detuning_MHz in zip(swap_stors, detunings):
            pulse_args = deepcopy(self.m1s_kwargs[stor - 1])
            pulse_args["freq"] += self.freq2reg(
                detuning_MHz,
                gen_ch=pulse_args["ch"],
            )
            all_pulse_args.append(pulse_args)
        return all_pulse_args

    def _play_floquet_software_loop(
            self, all_pulse_args, phase_offsets, swap_stors, floquet_cycle,
            update_phases, scramble_sync_cycles, palindrome_scramble,
            decoder_phase_offsets):
        """Emit every pulse of every cycle, each at the ledger's phase.

        Preloaded modes keep their pulse settings in a register bank, so each
        pulse writes only its phase; the other modes use ``setup_and_pulse``.
        """
        ecfg = self.cfg.expt
        forward_sequence = list(range(len(swap_stors)))
        reverse_sequence = list(reversed(forward_sequence))

        register_bank = {}
        preloaded_indices = [i for i, stor in enumerate(swap_stors)
                             if self.m1s_waveform_mode[stor - 1] == "preload_flattop"]
        if floquet_cycle > 0 and preloaded_indices:
            bank, _, _ = _prepare_preloaded_floquet_register_bank(
                self, [all_pulse_args[i] for i in preloaded_indices],
                [0.0] * len(preloaded_indices), [0.0] * len(preloaded_indices))
            register_bank = dict(zip(preloaded_indices, bank))

        for kk in range(floquet_cycle):
            if palindrome_scramble and kk % 2:
                cycle_sequence = reverse_sequence
            else:
                cycle_sequence = forward_sequence

            for step_idx, i_stor in enumerate(cycle_sequence):
                stor = swap_stors[i_stor]
                pulse_args = all_pulse_args[i_stor]

                phase_deg = self._mod360(phase_offsets[i_stor])
                pulse_args["phase"] = self.deg2reg(
                    phase_deg,
                    gen_ch=pulse_args["ch"],
                )

                if ecfg.get("debug", False) and kk == 0:
                    print(
                        f"[DarkScramble] cycle={kk}, step={step_idx}, "
                        f"stor={stor}, phase_deg={phase_deg:.3f}, "
                        f"stark_phase={phase_offsets[i_stor]:.3f}"
                    )

                if i_stor in register_bank:
                    entry = register_bank[i_stor]
                    self.safe_regwi(entry["register_page"], entry["phase_register"], pulse_args["phase"])
                    _play_preloaded_floquet_register_bank_entry(self, entry)
                else:
                    self.setup_and_pulse(**pulse_args)
                self.sync_all(scramble_sync_cycles)

                if decoder_phase_offsets is not None:
                    self._advance_decoder_phase_offsets(
                        decoder_phase_offsets=decoder_phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

    def _floquet_cycle_phases(self, phase_offsets, swap_stors, update_phases):
        """-> (phase of each pulse in cycle 0, phase step per cycle), in degrees.

        Later pulses of cycle 0 include the shifts from earlier pulses. The
        ledger itself is not changed. The step is the same for every cycle,
        which is what lets one tProc loop play them all.
        """
        first_cycle_phases = [0.0] * len(swap_stors)
        phase_offsets_after_cycle = list(phase_offsets)
        for i_stor, stor in enumerate(swap_stors):
            first_cycle_phases[i_stor] = self._mod360(
                phase_offsets_after_cycle[i_stor]
            )
            if update_phases:
                self._advance_phase_offsets(
                    phase_offsets=phase_offsets_after_cycle,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )

        phase_step_per_cycle = [
            self._mod360(phase_after - phase_before)
            for phase_before, phase_after in zip(
                phase_offsets,
                phase_offsets_after_cycle,
            )
        ]
        return first_cycle_phases, phase_step_per_cycle

    def _write_floquet_phase_registers(self, all_pulse_args, first_cycle_phases,
                                       phase_step_per_cycle):
        """Give each pulse a phase and a phase-step register, and write them.

        For modes that are not preloaded. Registers are taken from 1 up on
        each channel's page, below the page's first register that qick uses.
        -> (``[(page, phase_register, phase_step_register)]`` per pulse,
        loop page, loop register).
        """
        phase_registers = []
        next_register_by_page = {}
        for pulse_args in all_pulse_args:
            ch = pulse_args["ch"]
            page = self.ch_page(ch)
            if page == 0:
                raise RuntimeError(
                    "floquet_hardware_loop cannot use page 0 scratch "
                    "registers"
                )

            phase_register = next_register_by_page.get(page, 1)
            phase_step_register = phase_register + 1
            next_register_by_page[page] = phase_step_register + 1
            phase_registers.append(
                (page, phase_register, phase_step_register)
            )

        loop_page = phase_registers[0][0]
        loop_register = next_register_by_page.get(loop_page, 1)
        next_register_by_page[loop_page] = loop_register + 1

        register_maps = list(self._gen_regmap.values()) + list(
            self._ro_regmap.values()
        )
        for page, next_register in next_register_by_page.items():
            first_special_register = min(
                register
                for register_page, register in register_maps
                if register_page == page and register > 0
            )
            if next_register > first_special_register:
                raise RuntimeError(
                    "floquet_hardware_loop does not have enough "
                    f"scratch registers on page {page}"
                )

        for i_stor, pulse_args in enumerate(all_pulse_args):
            ch = pulse_args["ch"]
            gen_manager_name = self._gen_mgrs[ch].__class__.__name__
            if gen_manager_name != "FullSpeedGenManager":
                raise RuntimeError(
                    "floquet_hardware_loop requires a full-speed "
                    f"generator; channel {ch} uses {gen_manager_name}"
                )

            page, phase_register, phase_step_register = \
                phase_registers[i_stor]
            self.safe_regwi(
                page,
                phase_register,
                self.deg2reg(
                    first_cycle_phases[i_stor], gen_ch=ch),
            )
            self.safe_regwi(
                page,
                phase_step_register,
                self.deg2reg(
                    phase_step_per_cycle[i_stor], gen_ch=ch),
            )
        return phase_registers, loop_page, loop_register

    def _play_floquet_hardware_loop(
            self, all_pulse_args, phase_offsets, swap_stors, floquet_cycle,
            update_phases, scramble_sync_cycles, decoder_phase_offsets):
        """Play one cycle inside a tProc ``loopnz``, for depths the
        instruction memory cannot hold unrolled.

        The phases live in tProc registers: cycle 0's phase per pulse, plus
        a constant step added after each pulse. After the loop the Python
        ledgers advance by ``floquet_cycle`` steps, so they end where the
        software loop would leave them.
        """
        ecfg = self.cfg.expt
        first_cycle_phases, phase_step_per_cycle = self._floquet_cycle_phases(
            phase_offsets, swap_stors, update_phases)

        use_preloaded_register_bank = all(
            self.m1s_waveform_mode[stor - 1] == "preload_flattop"
            for stor in swap_stors
        )
        if use_preloaded_register_bank:
            register_bank, loop_page, loop_register = \
                _prepare_preloaded_floquet_register_bank(
                    self,
                    all_pulse_args,
                    first_cycle_phases,
                    phase_step_per_cycle,
                    reserved_registers=1,
                )
        else:
            phase_registers, loop_page, loop_register = \
                self._write_floquet_phase_registers(
                    all_pulse_args, first_cycle_phases, phase_step_per_cycle)

        self.safe_regwi(loop_page, loop_register, floquet_cycle - 1)

        floquet_loop_number = getattr(self, "_floquet_loop_number", 0)
        self._floquet_loop_number = floquet_loop_number + 1
        floquet_loop_label = f"FLOQUET_LOOP_{floquet_loop_number}"

        if not use_preloaded_register_bank:
            # Configure the next legacy waveform while the current pulse
            # is playing.  The setup margin remains unchanged.
            first_pulse_args = all_pulse_args[0]
            first_pulse_args["phase"] = 0
            self.set_pulse_registers(**first_pulse_args)
        self.label(floquet_loop_label)

        for i_stor, stor in enumerate(swap_stors):
            if ecfg.get("debug", False):
                print(
                    f"[DarkScramble] hardware step={i_stor}, "
                    f"stor={stor}, "
                    f"first_phase_deg={first_cycle_phases[i_stor]:.3f}, "
                    f"phase_step_deg={phase_step_per_cycle[i_stor]:.3f}"
                )

            if use_preloaded_register_bank:
                entry = register_bank[i_stor]
                _play_preloaded_floquet_register_bank_entry(self, entry)
                self.math(
                    entry["register_page"],
                    entry["phase_register"],
                    entry["phase_register"],
                    "+",
                    entry["phase_step_register"],
                )
            else:
                pulse_args = all_pulse_args[i_stor]
                ch = pulse_args["ch"]
                page, phase_register, phase_step_register = \
                    phase_registers[i_stor]
                self.mathi(
                    page,
                    self.sreg(ch, "phase"),
                    phase_register,
                    "+",
                    0,
                )
                self.pulse(ch)
                self.math(
                    page,
                    phase_register,
                    phase_register,
                    "+",
                    phase_step_register,
                )

                next_pulse_args = all_pulse_args[(i_stor + 1) % len(swap_stors)]
                next_pulse_args["phase"] = 0
                self.set_pulse_registers(**next_pulse_args)
            self.sync_all(scramble_sync_cycles)

        self.loopnz(loop_page, loop_register, floquet_loop_label)

        for i_stor in range(len(swap_stors)):
            phase_offsets[i_stor] = self._mod360(
                phase_offsets[i_stor]
                + floquet_cycle * phase_step_per_cycle[i_stor]
            )

        if decoder_phase_offsets is not None:
            for _ in range(floquet_cycle):
                for stor in swap_stors:
                    self._advance_decoder_phase_offsets(
                        decoder_phase_offsets=decoder_phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

    def _play_m1s_frac_train(
        self,
        stor,
        n_frac,
        phase_offsets,
        swap_stors,
        disorder_phase_offsets=None,
        logical_phase_deg=0.0,
        logical_phase_step_deg=0.0,
        inverse=False,
        update_phases=True,
        label="",
    ):
        r"""
        Play n_frac copies of the calibrated M1-S{stor} fractional pulse.

        logical_phase_deg:
            Desired logical phase of this beam-splitter pulse.

        logical_phase_step_deg:
            Phase added after every physical fractional pulse.  A value of
            180 degrees emits the sequence +, -, +, -, ... in one train.

        inverse:
            If True, implements U^\dagger by adding 180 degrees to the pulse phase.
            This uses U(-theta, phi) = U(theta, phi + 180 deg).

        phase_offsets:
            Mutable list tracking calibrated Stark/off-resonant frame
            corrections from previous pulses.

        disorder_phase_offsets:
            Optional synthetic-disorder rotating-frame phases. These are added
            to the dark load/readout pulse axes but are not advanced inside the
            load/readout sequence; they represent the frame accumulated before
            the analyzer starts.
        """
        n_frac = int(n_frac)
        if n_frac <= 0:
            return

        idx = swap_stors.index(stor)
        pulse_args = deepcopy(self.m1s_kwargs[stor - 1])

        disorder_phase_deg = 0.0
        if disorder_phase_offsets is not None:
            disorder_phase_deg = disorder_phase_offsets[idx]

        inverse_phase = 180.0 if inverse else 0.0
        first_phase_deg = self._mod360(
            phase_offsets[idx]
            + disorder_phase_deg
            + logical_phase_deg
            + inverse_phase
        )
        phase_step_deg = self._mod360(logical_phase_step_deg)
        sync_cycles = int(
            self.cfg.expt.get("scramble_sync_cycles", 10))
        hardware_loop = bool(
            self.cfg.expt.get("floquet_hardware_loop", False))

        if self.cfg.expt.get("debug", False):
            direction = "inverse" if inverse else "forward"
            print(
                f"[DarkT1] {label}: stor={stor}, {direction}, "
                f"n_frac={n_frac}, phase_deg={first_phase_deg:.3f}, "
                f"phase_step_deg={phase_step_deg:.3f}, "
                f"phase_offset={phase_offsets[idx]:.3f}, "
                f"disorder_phase={disorder_phase_deg:.3f}, "
                f"logical_phase={logical_phase_deg:.3f}, "
                f"hardware_loop={hardware_loop}"
            )

        if hardware_loop:
            ch = pulse_args["ch"]
            page = self.ch_page(ch)
            if page == 0: #<--- This part should be reviewed; not sure if page 0 is fully forbidden, but seems this is added to avoid
                          #the collision with other registers assigned for RAverager or hardware loop
                raise RuntimeError(
                    "floquet_hardware_loop cannot use page 0 scratch registers"
                )
            if self._gen_mgrs[ch].__class__.__name__ != "FullSpeedGenManager": #This is done because only full speed generator has separate phase sreg.
                raise RuntimeError(
                    "floquet_hardware_loop requires a full-speed generator; "
                    f"channel {ch} uses "
                    f"{self._gen_mgrs[ch].__class__.__name__}"
                )

            phase_register = 1
            phase_step_register = 2
            loop_register = 3
            register_maps = list(self._gen_regmap.values()) + list(
                self._ro_regmap.values()) #In general, _gen_regmap[(ch, "freg")] = (page, register), so values() collapses those into
                                          #list, which is used to get the lowest value of sreg for the page. 
            first_special_register = min(
                register
                for register_page, register in register_maps
                if register_page == page and register > 0
            ) #This is done as QICK allocates sregs from the highest numbers
            if loop_register >= first_special_register:
                raise RuntimeError(
                    "floquet_hardware_loop does not have enough scratch "
                    f"registers on page {page}"
                )

            self.safe_regwi(
                page,
                phase_register,
                self.deg2reg(first_phase_deg, gen_ch=ch),
            )
            self.safe_regwi(
                page,
                phase_step_register,
                self.deg2reg(phase_step_deg, gen_ch=ch),
            )
            self.safe_regwi(page, loop_register, n_frac - 1)

            pulse_args["phase"] = 0
            self.set_pulse_registers(**pulse_args)

            floquet_loop_number = getattr(
                self, "_floquet_loop_number", 0)
            self._floquet_loop_number = floquet_loop_number + 1
            loop_label = f"FLOQUET_FRAC_LOOP_{floquet_loop_number}"
            self.label(loop_label)

            self.mathi(
                page,
                self.sreg(ch, "phase"),
                phase_register,
                "+",
                0,
            )
            self.pulse(ch)
            self.math(
                page,
                phase_register,
                phase_register,
                "+",
                phase_step_register,
            )
            self.sync_all(sync_cycles)
            self.loopnz(page, loop_register, loop_label)
        else:
            entry = None
            if self.m1s_waveform_mode[stor - 1] == "preload_flattop":
                bank, _, _ = _prepare_preloaded_floquet_register_bank(
                    self, [pulse_args], [first_phase_deg], [0.0])
                entry = bank[0]

            for kk in range(n_frac):
                phase_deg = self._mod360(
                    first_phase_deg + kk * phase_step_deg)
                pulse_args["phase"] = self.deg2reg(
                    phase_deg,
                    gen_ch=pulse_args["ch"],
                )
                if entry is not None:
                    self.safe_regwi(entry["register_page"], entry["phase_register"], pulse_args["phase"])
                    _play_preloaded_floquet_register_bank_entry(self, entry)
                else:
                    self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)

        if update_phases:
            for _ in range(n_frac):
                self._advance_phase_offsets(
                    phase_offsets=phase_offsets,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )
    
    def _accumulate_scramble_phases(self, phase_offsets, swap_stors):
        if not self.cfg.expt.get("update_phases", True):
            return
        for _ in range(self.cfg.expt.floquet_cycle):
            for stor in swap_stors:
                self._advance_phase_offsets(
                    phase_offsets=phase_offsets,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )


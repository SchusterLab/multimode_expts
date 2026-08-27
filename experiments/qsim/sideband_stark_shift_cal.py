# -*- coding: utf-8 -*-
"""Sideband Stark-shift phase calibration: the three surviving variants.

Split out of ``floquet_dark_mode_readout.py`` unchanged. Grouping the variants
in one file does not decide their fate: ``_old`` is live, and the other two are
still pending the variant comparison (spec appendix A). The names violate
section 6 and are renamed in the naming pass, not here.
"""
from copy import deepcopy

from experiments.qsim.qsim_base import QsimBaseProgram
from experiments.qsim.floquet_dark_mode_readout import DarkBaseProgram

class SidebandStarkAmplificationModifiedProgram_old(QsimBaseProgram):
    """
    Original phase-accumulation calibration sequence.

    Kept for comparing against the DarkBaseProgram/setup_and_pulse-matched
    version below.
    
    THIS IS STILL BEING USED; MUST BE KEPT IN THE REFACTORING.
    """

    def core_pulses(self):
        _scramble_sync_cycles = self.cfg.expt.get("scramble_sync_cycles", 10)
        i_storA = self.cfg.expt.stor_A - 1
        i_storB = self.cfg.expt.stor_B - 1
        m1s_kwarg_A = self.m1s_kwargs[i_storA]
        m1s_kwarg_B = self.m1s_kwargs[i_storB]

        n_pulse_B = self.cfg.expt.n_pulse
        pi_frac_A = self.m1s_pi_fracs[i_storA]
        pi_frac_B = self.m1s_pi_fracs[i_storB]

        ch_A = m1s_kwarg_A['ch']
        ch_B = m1s_kwarg_B['ch']
        channel_page_B = self.ch_page(ch_B)
        r_phase_B= self.sreg(ch_B, "phase")

        # Apply pi/2 pulse on stor_A
        self.set_pulse_registers(**m1s_kwarg_A)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)
            if self.cfg.expt.get("include_10cycles_buffer", False) and self.cfg.expt.get("include_10cycles_buffer_in_pi_half", False):
                self.sync_all(_scramble_sync_cycles)
        self.sync_all()

        # # Apply a 2pi * n_pulse gate on stor_B
        # self.set_pulse_registers(**m1s_kwarg_B)
        # for i in range(n_pulse_B * 2 * pi_frac_B):
        #     self.pulse(ch_B)
        # advance_phase_A = self.deg2reg(n_pulse_B * pi_frac * self.cfg.expt.advance_phase)
        # self.sync_all()

        # Apply a (pi/12, -pi/12) * n_pulse gate on stor_B
        phase = 0
        self.set_pulse_registers(**m1s_kwarg_B)
        for i in range(n_pulse_B):
            for j in range(2):
                self.pulse(ch_B)
                # update the phase modulo 360
                phase += 180
                phase = phase % 360
                _phase_reg = self.deg2reg(phase, gen_ch=ch_B)
                self.safe_regwi(channel_page_B, r_phase_B, _phase_reg)
                if self.cfg.expt.get("include_10cycles_buffer", False):
                    self.sync_all(_scramble_sync_cycles)
        advance_phase_A = self.deg2reg(2 * n_pulse_B * self.cfg.expt.advance_phase)
        self.sync_all()

        # Apply -pi/2 pulse on stor_A with advanced phase
        m1s_kwarg_A_advanced = deepcopy(m1s_kwarg_A)
        m1s_kwarg_A_advanced['phase'] = advance_phase_A
        self.set_pulse_registers(**m1s_kwarg_A_advanced)
        for i in range(pi_frac_A // 2):
            self.pulse(m1s_kwarg_A_advanced['ch'])
            if self.cfg.expt.get("include_10cycles_buffer", False) and self.cfg.expt.get("include_10cycles_buffer_in_pi_half", False):
                self.sync_all(_scramble_sync_cycles)
        self.sync_all()


class SidebandStarkAmplificationModifiedProgram(DarkBaseProgram):
    """
    Measure how a Floquet pulse on B shifts the later ds_storage swap on A.

    The Ramsey preparation/readout pulses on A use exactly half of the
    calibrated ds_storage full-swap length. Between them, the program applies
    (+Floquet B, -Floquet B) pairs. The final A half-swap phase is advanced by
    ``2 * n_pulse * advance_phase``, so the fitted ``advance_phase`` is the
    compensation per physical Floquet B pulse.
    """

    def _play_storage_half_swap(self, stor, phase_deg):
        storage_ds = self.cfg.device.storage._ds_storage
        stor_name = f"M1-S{stor}"
        freq = storage_ds.get_freq(stor_name)

        if freq < 1800:
            ch = self.flux_low_ch[0]
            waveform = "pi_m1si_low"
        else:
            ch = self.flux_high_ch[0]
            waveform = "pi_m1si_high"

        self.setup_and_pulse(
            ch=ch,
            style="flat_top",
            freq=self.freq2reg(freq, gen_ch=ch),
            phase=self.deg2reg(self._mod360(phase_deg), gen_ch=ch),
            gain=storage_ds.get_gain(stor_name),
            length=self.us2cycles(
                storage_ds.get_pi(stor_name) / 2.0,
                gen_ch=ch,
            ),
            waveform=waveform,
        )
        self.sync_all(self.us2cycles(0.01))

    def core_pulses(self):
        stor_A = int(self.cfg.expt.stor_A)
        stor_B = int(self.cfg.expt.stor_B)
        swap_stors = [stor_A, stor_B]
        phase_offsets = [0.0, 0.0]

        n_pulse_B = int(self.cfg.expt.n_pulse)
        advance_phase = float(self.cfg.expt.advance_phase)

        self._play_storage_half_swap(stor=stor_A, phase_deg=0.0)

        # This is the same weak physical Floquet pulse and sync gap used by
        # spectroscopy. The 180-degree alternation closes population transfer.
        self._play_m1s_frac_train(
            stor=stor_B,
            n_frac=2 * n_pulse_B,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            logical_phase_deg=0.0,
            logical_phase_step_deg=180.0,
            update_phases=False,
            label="phase calibration: alternating B train",
        )

        self._play_storage_half_swap(
            stor=stor_A,
            phase_deg=2.0 * n_pulse_B * advance_phase,
        )
        self.sync_all()


class SidebandStarkAmplificationModifiedProgram_newold(DarkBaseProgram):
    """
    1. Apply pi/2 swap pulse made of floquet pulses on stor_A
    2. Apply another floquet 2pi pulse on stor_B to calibrate the matrix element for. Do this xN times for error amplification
    3. Apply a -pi/2 swap pulse of floquet pulses on stor_A, with advanced phase
    
    Parameters in cfg.expt (sweepable):
    stor_A
    stor_B
    n_pulse: Nx pulses on stor B 
    advance_phase: phase of the last pulse on stor_A
    """

    def core_pulses(self):
        i_storA = self.cfg.expt.stor_A - 1
        i_storB = self.cfg.expt.stor_B - 1
        m1s_kwarg_A = self.m1s_kwargs[i_storA]
        m1s_kwarg_B = self.m1s_kwargs[i_storB]

        n_pulse_B = self.cfg.expt.n_pulse
        pi_frac_A = self.m1s_pi_fracs[i_storA]

        ch_A = m1s_kwarg_A['ch']
        ch_B = m1s_kwarg_B['ch']

        # Apply pi/2 pulse on stor_A
        self.set_pulse_registers(**m1s_kwarg_A)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)
        self.sync_all()

        # Apply a (pi/12, -pi/12) * n_pulse gate on stor_B
        m1s_kwarg_B = deepcopy(m1s_kwarg_B)
        for i in range(n_pulse_B):
            for phase in (0, 180):
                m1s_kwarg_B['phase'] = self.deg2reg(phase, gen_ch=ch_B)
                self.setup_and_pulse(**m1s_kwarg_B)
                self.sync_all(10)
        advance_phase_A = self.deg2reg(
            2 * n_pulse_B * self.cfg.expt.advance_phase,
            gen_ch=ch_A,
        )
        
        # Apply -pi/2 pulse on stor_A with advanced phase
        m1s_kwarg_A_advanced = deepcopy(m1s_kwarg_A)
        m1s_kwarg_A_advanced['phase'] = advance_phase_A
        self.set_pulse_registers(**m1s_kwarg_A_advanced)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)

'''
Dual Rail Ramsey Experiment

Measures T2 Ramsey coherence of the dual-rail qubit encoded in two storage modes,
with optional joint parity (JP) checks distributed as equal waits.

Uses the dual rail helper methods from MM_base:
- prep_dual_rail_logical_state() for state preparation (|+>, |-> etc.)
- dual_rail_gate_sequence() for echo gates
- joint_parity_active_reset() for JP checks
- Gate decomposition pattern (swap_in + middle_op + swap_out) for the
  second pi/2 with firmware-swept Ramsey phase.

Pulse sequence (unified distributed waits):
  All waits are equal = total_variable_time / n_wait_segments,
  where n_wait_segments = n_checks + 1 + num_echoes.

  echo=0, n_checks=0:  prep -> wait -> x/2 -> measure
  echo=0, n_checks>0:  prep -> [wait -> JP]xn -> wait -> x/2 -> measure
  echo=1, n_checks=0:  prep -> wait -> echo -> wait -> x/2 -> measure
  echo=1, n_checks=2N: prep -> [wait -> JP]xN -> wait -> echo
                             -> [wait -> JP]xN -> wait -> x/2 -> measure

JP phase correction:
- echo=0: n_checks * jp_phase_per_check (all checks same sign)
- echo=1: 0 (symmetric N before + N after echo = cancellation)

AC Stark correction added to effective_ramsey_freq only when echo=0
(echo refocuses the AC Stark shift).

Config naming follows dual_rail_sandbox_v2 convention:
- storage_swap: storage mode index for logical |1> (photon here = |1_L>)
- storage_parity: storage mode index for logical |0> (photon here = |0_L>)

Seb 02/2026
'''

import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import *


class DualRailRamseyProgram(MMRAveragerProgram):
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        man = cfg.expt.manipulate

        # --- Build storage mode names (sandbox_v2 convention) ---
        s_swap = cfg.expt.storage_swap
        s_parity = cfg.expt.storage_parity
        self.stor_name_swap = f'M{man}-S{s_swap}'
        self.stor_name_parity = f'M{man}-S{s_parity}'
        self.stor_pair_name = f'M{man}-S{s_parity}'

        _ds = cfg.device.storage._ds_storage

        # --- Joint parity check configuration ---
        self.n_active_checks = int(cfg.expt.get('n_active_checks', 0))

        # --- Echo + JP: symmetric distributed structure ---
        # When echo=1 with JP checks, split checks equally around the echo.
        # All waits are equal (CPMG-like). n_checks must be even.
        self.num_echoes = cfg.expt.get('echoes', 0)
        self.n_checks_before = 0
        self.n_checks_after = 0
        if self.num_echoes == 1 and self.n_active_checks > 0:
            assert self.n_active_checks % 2 == 0, \
                "n_checks must be even when echo=1"
            N = self.n_active_checks // 2
            self.n_checks_before = N
            self.n_checks_after = N

        # --- Phase tracking setup ---
        self.pair_name = _ds.pair_name(s_swap, s_parity)
        self.phase_tracking = cfg.expt.get('phase_tracking', True)
        self.ac_stark_rate = 0.0
        if self.phase_tracking:
            rates = MM_base.load_dr_ac_stark_rates(
                _ds, [self.pair_name],
                cfg_override=([cfg.expt.ac_stark_rate]
                              if 'ac_stark_rate' in cfg.expt else None))
            self.ac_stark_rate = rates[0]

        # --- JP-induced phase correction ---
        self.jp_phase_per_check = 0.0
        if self.n_active_checks > 0 and self.phase_tracking:
            jp_matrix = MM_base.load_dr_jp_phase_matrix(
                _ds, [self.pair_name],
                cfg_override=cfg.expt.get('jp_phase_matrix'))
            self.jp_phase_per_check = jp_matrix.get(
                self.pair_name, {}).get(self.pair_name, 0.0)

        # --- State preparation via prep_dual_rail_logical_state ---
        state_start = cfg.expt.get('state_start', '+')
        self.state_start = state_start

        state_seq = self.prep_dual_rail_logical_state(
            state_start, self.stor_name_swap, self.stor_name_parity)
        if state_seq is not None:
            self.state_prep_pulse = self.get_prepulse_creator(state_seq).pulse.tolist()
        else:
            self.state_prep_pulse = None

        # --- Second pi/2 gate components (gate decomposition pattern) ---
        # Swap in/out: pi on stor_swap (static, compiled once)
        swap_seq = [['storage', self.stor_name_swap, 'pi', 0]]
        self.swap_pulse = self.get_prepulse_creator(swap_seq).pulse.tolist()

        # Middle hpi on stor_parity (register-controlled for Ramsey phase)
        self.hpi_freq = _ds.get_freq(self.stor_name_parity)
        self.hpi_gain = _ds.get_gain(self.stor_name_parity)
        hpi_length_us = _ds.get_h_pi(self.stor_name_parity)

        flux_low_ch = cfg.hw.soc.dacs.flux_low.ch[0]
        flux_high_ch = cfg.hw.soc.dacs.flux_high.ch[0]
        self.hpi_flux_ch = flux_low_ch if self.hpi_freq < 1800 else flux_high_ch

        ramp_sigma_us = cfg.device.storage.ramp_sigma
        sigma_cycles = self.us2cycles(ramp_sigma_us, gen_ch=self.hpi_flux_ch)
        self.add_gauss(ch=self.hpi_flux_ch, name="hpi_stor_ramp",
                       sigma=sigma_cycles, length=sigma_cycles * 6)

        self.hpi_freq_reg = self.freq2reg(self.hpi_freq, gen_ch=self.hpi_flux_ch)
        self.hpi_length_cycles = self.us2cycles(hpi_length_us, gen_ch=self.hpi_flux_ch)

        # --- Echo configuration (self.num_echoes set above in split detection) ---
        if self.num_echoes > 0:
            echo_gate = cfg.expt.get('echo_gate', 'Y')
            echo_seq = self.dual_rail_gate_sequence(
                echo_gate, self.stor_name_swap, self.stor_name_parity)
            self.echo_pulse = self.get_prepulse_creator(echo_seq).pulse.tolist()

        # --- Sweep registers on hpi flux channel's register page ---
        self.flux_rp = self.ch_page(self.hpi_flux_ch)
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_flux_phase = self.sreg(self.hpi_flux_ch, "phase")

        # Initialize registers (us2cycles WITHOUT gen_ch, matching sideband_ramsey)
        # Universal formula: all cases use distributed equal waits
        n_wait_segments = self.n_active_checks + 1 + self.num_echoes
        self.n_wait_segments = n_wait_segments
        self.safe_regwi(self.flux_rp, self.r_wait,
                        self.us2cycles(cfg.expt.start / n_wait_segments))

        initial_ramsey_phase = cfg.expt.get('initial_ramsey_phase', 0)
        self.safe_regwi(self.flux_rp, self.r_phase2,
                        self.deg2reg(initial_ramsey_phase, gen_ch=self.hpi_flux_ch))

        # JP phase correction register (accumulated JP-induced shift on this pair)
        if self.n_active_checks > 0:
            jp_phase = self.jp_phase_per_check
            if self.num_echoes == 1:
                # Symmetric CPMG: N checks before echo (negated) + N after = 0
                total_jp_phase_corr = 0
                print(f"  Echo+JP symmetric: {self.n_checks_before} before + "
                      f"{self.n_checks_after} after echo, "
                      f"jp_phase={jp_phase:.2f} deg/check, correction = 0")
            else:
                total_jp_phase_corr = self.n_active_checks * jp_phase
            self.r_jp_phase_corr = 6
            self.safe_regwi(self.flux_rp, self.r_jp_phase_corr,
                            self.deg2reg(total_jp_phase_corr, gen_ch=self.hpi_flux_ch))

        # Phase step register: use safe_regwi + math (register-register)
        # to avoid mathi's 31-bit immediate limit (deg2reg can exceed 2^31
        # when abs(ramsey_freq) * step > 0.5)
        # When phase tracking is on and no echoes, the AC Stark idle rotation
        # is absorbed into the effective Ramsey frequency.
        # With echoes, the echo refocuses the AC Stark shift so we skip it.
        self.effective_ramsey_freq = cfg.expt.ramsey_freq
        if self.phase_tracking and self.num_echoes == 0:
            self.effective_ramsey_freq += self.ac_stark_rate
        # Note: JP phase correction is still applied regardless of echoes

        self.r_phase_step = 5
        phase_step_val = self.deg2reg(
            360 * abs(self.effective_ramsey_freq) * cfg.expt.step,
            gen_ch=self.hpi_flux_ch)
        self.safe_regwi(self.flux_rp, self.r_phase_step, phase_step_val)
        self.ramsey_freq_sign = 1 if self.effective_ramsey_freq >= 0 else -1

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)

        # 1. Phase reset
        self.reset_and_sync()

        # 2. Active reset (if enabled)
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # 3. State preparation (e.g. |+> via prep_dual_rail_logical_state)
        if self.state_prep_pulse is not None:
            self.custom_pulse(cfg, self.state_prep_pulse, prefix='state_prep_')

        # 4. State-prep post-selection
        if cfg.expt.get('state_prep_postselect', False):
            self.post_selection_measure(
                parity=cfg.expt.get('state_prep_ps_parity', False),
                man_idx=cfg.expt.manipulate,
                parity_fast=cfg.expt.get('parity_fast', False),
                prefix='state_prep_ps'
            )

        if self.num_echoes == 1 and self.n_active_checks > 0:
            # === CPMG distributed: (wait→JP)×N → wait → echo → (wait→JP)×N → wait ===

            # Pre-echo checks
            for k in range(self.n_checks_before):
                self.sync_all()
                self.sync(self.flux_rp, self.r_wait)
                self.sync_all()
                self.custom_pulse(cfg, self.swap_pulse, prefix=f'swap_in_{k}_')
                self.joint_parity_active_reset(
                    stor_pair_name=self.stor_pair_name,
                    name=f'jp_{k}',
                    register_label=f'jp_label_{k}',
                    second_phase=0,
                    fast=cfg.expt.get('parity_fast', False))
                self.custom_pulse(cfg, self.swap_pulse, prefix=f'swap_out_{k}_')

            # Wait before echo
            self.sync_all()
            self.sync(self.flux_rp, self.r_wait)
            self.sync_all()

            # Echo pulse
            self.custom_pulse(cfg, self.echo_pulse, prefix='echo_')
            self.sync_all(self.us2cycles(0.01))

            # Post-echo checks
            for k in range(self.n_checks_after):
                self.sync_all()
                self.sync(self.flux_rp, self.r_wait)
                self.sync_all()
                idx = self.n_checks_before + k
                self.custom_pulse(cfg, self.swap_pulse, prefix=f'swap_in_{idx}_')
                self.joint_parity_active_reset(
                    stor_pair_name=self.stor_pair_name,
                    name=f'jp_{idx}',
                    register_label=f'jp_label_{idx}',
                    second_phase=0,
                    fast=cfg.expt.get('parity_fast', False))
                self.custom_pulse(cfg, self.swap_pulse, prefix=f'swap_out_{idx}_')

            # Final wait
            self.sync_all()
            self.sync(self.flux_rp, self.r_wait)
            self.sync_all()

        elif self.n_active_checks > 0:
            # === Distributed echo=0+JP: (wait → JP)×n → wait ===
            for k in range(self.n_active_checks):
                self.sync_all()
                self.sync(self.flux_rp, self.r_wait)
                self.sync_all()
                self.custom_pulse(cfg, self.swap_pulse, prefix=f'swap_in_{k}_')
                self.joint_parity_active_reset(
                    stor_pair_name=self.stor_pair_name,
                    name=f'jp_{k}',
                    register_label=f'jp_label_{k}',
                    second_phase=0,
                    fast=cfg.expt.get('parity_fast', False))
                self.custom_pulse(cfg, self.swap_pulse, prefix=f'swap_out_{k}_')

            # Final wait
            self.sync_all()
            self.sync(self.flux_rp, self.r_wait)
            self.sync_all()

        else:
            # === Pure Ramsey or echo without JP ===
            self.sync_all()
            self.sync(self.flux_rp, self.r_wait)
            self.sync_all()

            if self.num_echoes > 0:
                for _ in range(self.num_echoes):
                    self.custom_pulse(cfg, self.echo_pulse, prefix='echo_')
                    self.sync_all(self.us2cycles(0.01))
                    self.sync_all()
                    self.sync(self.flux_rp, self.r_wait)
                    self.sync_all()

        # 6. Second pi/2 gate with Ramsey phase
        #    Gate decomposition: swap_in + hpi(phase) + swap_out
        #    (same structure as dual_rail_gate_sequence but with dynamic phase)

        # 6a. Swap in: pi on stor_swap (M <-> S_swap)
        self.custom_pulse(cfg, self.swap_pulse, prefix='gate_swap_in_')

        # 6b. Middle hpi on stor_parity with accumulated Ramsey phase
        self.set_pulse_registers(
            ch=self.hpi_flux_ch, style="flat_top",
            freq=self.hpi_freq_reg,
            phase=self.deg2reg(0, gen_ch=self.hpi_flux_ch),
            gain=self.hpi_gain,
            length=self.hpi_length_cycles,
            waveform="hpi_stor_ramp")
        self.sync_all(self.us2cycles(0.01))
        # Copy accumulated Ramsey phase + JP correction to the flux channel's phase register
        if self.n_active_checks > 0:
            self.math(self.flux_rp, self.r_flux_phase, self.r_phase2,
                      '+', self.r_jp_phase_corr)
        else:
            self.mathi(self.flux_rp, self.r_flux_phase, self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))
        self.pulse(ch=self.hpi_flux_ch)
        self.sync_all(self.us2cycles(0.01))

        # 6c. Swap out: pi on stor_swap (M <-> S_swap)
        self.custom_pulse(cfg, self.swap_pulse, prefix='gate_swap_out_')

        # 7. Final dual rail measurement
        self.measure_dual_rail(
            storage_idx=(cfg.expt.storage_swap, cfg.expt.storage_parity),
            measure_parity=cfg.expt.get('measure_parity', True),
            reset_before=cfg.expt.get('reset_before_dual_rail', False),
            reset_after=cfg.expt.get('reset_after_dual_rail', False),
            man_idx=cfg.expt.manipulate,
            final_sync=True)

    def update(self):
        # Increment wait time (us2cycles WITHOUT gen_ch, matching sideband_ramsey)
        step_cycles = self.us2cycles(self.cfg.expt.step / self.n_wait_segments)
        self.mathi(self.flux_rp, self.r_wait, self.r_wait, '+', step_cycles)
        self.sync_all(self.us2cycles(0.01))
        # Increment Ramsey phase using register-register math (no 31-bit limit)
        op = '+' if self.ramsey_freq_sign >= 0 else '-'
        self.math(self.flux_rp, self.r_phase2, self.r_phase2,
                  op, self.r_phase_step)
        self.sync_all(self.us2cycles(0.01))

    def _calculate_read_num(self):
        """Calculate total number of readouts per rep"""
        cfg = self.cfg
        read_num = 0

        # Active reset readouts
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        # State-prep post-selection
        if cfg.expt.get('state_prep_postselect', False):
            read_num += 1

        # JP readouts (one per active check)
        read_num += self.n_active_checks

        # measure_dual_rail: 2 parity measurements (one per storage)
        read_num += 2
        if cfg.expt.get('reset_before_dual_rail', False):
            read_num += 1
        if cfg.expt.get('reset_after_dual_rail', False):
            read_num += 1

        print('read num', read_num)

        return read_num

    def collect_shots(self):
        """Collect single-shot data reshaped for per-sweep-point analysis.

        QICK RAveragerProgram buffer layout (C-order):
            (rounds, expts, reps, read_num)
        We transpose rounds into the reps axis and return shape:
            (expts, total_reps * read_num)
        where total_reps = rounds * reps. Each row's data, when reshaped
        to (total_reps, read_num), gives columns corresponding to each
        readout trigger within body() (e.g. col 0 = swap, col 1 = parity).
        """
        read_num = self._calculate_read_num()
        expts = self.cfg.expt.expts
        reps = self.cfg.expt.reps
        rounds = self.cfg.expt.get('rounds', 1)
        total_reps = rounds * reps

        buf_i = self.di_buf[0] / self.readout_lengths_adc[0]
        buf_q = self.dq_buf[0] / self.readout_lengths_adc[0]

        # Reshape following QICK buffer layout, then group rounds into reps
        shots_i0 = buf_i.reshape((rounds, expts, reps, read_num))
        shots_i0 = shots_i0.transpose(1, 0, 2, 3).reshape((expts, total_reps * read_num))
        shots_q0 = buf_q.reshape((rounds, expts, reps, read_num))
        shots_q0 = shots_q0.transpose(1, 0, 2, 3).reshape((expts, total_reps * read_num))

        return shots_i0, shots_q0, read_num


class DualRailRamseyExperiment(Experiment):
    """
    Dual Rail Ramsey Experiment

    Measures T2 Ramsey coherence of the dual-rail qubit by sweeping the
    wait time between state preparation and a final pi/2 gate in firmware.

    Uses dual rail helper methods from MM_base for state preparation and gates.

    Experimental Config:
    expt = dict(
        qubits: [0],
        reps: number of averages per sweep point,
        rounds: number of rounds (default 1),
        start: wait time start [us],
        step: wait time step [us] (Nyquist: 1/step > 2*ramsey_freq),
        expts: number of sweep points,
        ramsey_freq: virtual detuning frequency [MHz],
        storage_swap: storage mode index for logical |1> (e.g. 1),
        storage_parity: storage mode index for logical |0> (e.g. 2),
        manipulate: manipulate mode index (e.g. 1),
        state_start: logical state to prepare (default '+'),
            supported: '+', '-', '+i', '-i', '0', '1'
            can also be a list e.g. ['+', '-'] to run both
        echoes: number of echo pulses (0 = Ramsey, N>0 = N echoes),
        echo_gate: gate for echo pulse (default 'Y'),
            supported: 'X', 'Y', 'X/2', '-X/2', 'Y/2', '-Y/2'
        active_reset: if True, perform active reset,
        state_prep_postselect: if True, measure qubit after state prep for post-selection,
        state_prep_ps_parity: if True, play parity pulse before state-prep PS measurement,
        reset_before_dual_rail: if True, reset before dual rail measurement,
        reset_after_dual_rail: if True, reset after dual rail measurement,
        measure_parity: if True use parity, if False use slow pi,
        n_checks: number of distributed JP checks (0 = pure Ramsey, default 0),
            must be even when echoes=1,
        parity_fast: True/False (use fast multiphoton hpi for JP, default False),
    )
    """

    STATE_COLORS = {
        '00': 'tab:blue',
        '10': 'tab:orange',
        '01': 'tab:green',
        '11': 'tab:red',
    }

    def __init__(self, soccfg=None, path='', prefix='DualRailRamsey', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        # Handle state_start as list or single string
        state_list = self.cfg.expt.get('state_start', '+')
        if isinstance(state_list, str):
            state_list = [state_list]

        n_checks = int(self.cfg.expt.get('n_checks', 0))

        # Extract sweep parameters
        start = self.cfg.expt.start
        step = self.cfg.expt.step
        expts = self.cfg.expt.expts
        reps = self.cfg.expt.reps
        rounds = self.cfg.expt.get('rounds', 1)

        # Compute full time array
        times = np.array([start + i * step for i in range(expts)])

        data = {
            'states': state_list,
            'threshold': self.cfg.device.readout.threshold[0],
            'xpts': times,
            'n_checks': n_checks,
            'reps': reps,
            'rounds': rounds,
            'step': step,
        }

        num_echoes = int(self.cfg.expt.get('echoes', 0))

        # Set n_active_checks for JP mode
        if n_checks > 0:
            if num_echoes == 1:
                assert n_checks % 2 == 0, "n_checks must be even for echo=1"
            self.cfg.expt.n_active_checks = n_checks
            n_wait_segments = n_checks + 1 + num_echoes
            print(f"Distributed JP mode: {n_checks} checks, "
                  f"n_wait_segments={n_wait_segments}")

        for state in state_list:
            print(f"\n=== Dual Rail Ramsey: state='{state}' ===")
            self.cfg.expt.state_start = state

            prog = DualRailRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
            read_num = prog._calculate_read_num()

            if prog.phase_tracking:
                print(f"Phase tracking ON: AC Stark rate={prog.ac_stark_rate} MHz, "
                      f"effective Ramsey freq={prog.effective_ramsey_freq:.6g} MHz")

            x_pts, avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num)

            i0, q0, _ = prog.collect_shots()

            data[f'i0_{state}'] = i0
            data[f'q0_{state}'] = q0
            data[f'avgi_{state}'] = np.array(avgi[0])
            data[f'avgq_{state}'] = np.array(avgq[0])
            data[f'read_num_{state}'] = read_num
            data[f'ac_stark_rate_{state}'] = prog.ac_stark_rate
            data[f'effective_ramsey_freq_{state}'] = prog.effective_ramsey_freq

            # Compute overheads (once, from first program)
            if n_checks > 0 and 'jp_overhead' not in data:
                swap_dur = prog.get_total_time(
                    np.array(prog.swap_pulse, dtype=object))
                jp_meas_dur = prog.get_jp_measurement_duration(
                    prog.stor_pair_name,
                    fast=self.cfg.expt.get('parity_fast', False))
                data['jp_overhead'] = 2 * swap_dur + jp_meas_dur
                print(f"  JP overhead: {data['jp_overhead']:.2f} us")
            if num_echoes > 0 and 'echo_overhead' not in data:
                echo_dur = prog.get_total_time(
                    np.array(prog.echo_pulse, dtype=object))
                data['echo_overhead'] = num_echoes * echo_dur
                print(f"  Echo overhead: {data['echo_overhead']:.2f} us")

        # Clean up temporary cfg params
        if n_checks > 0:
            self.cfg.expt.pop('n_active_checks', None)

        self.data = data
        return data

    def _get_shot_indices(self, n_active_checks=0):
        """Return dict mapping measurement type to column indices within a single rep.

        Args:
            n_active_checks: number of active JP checks (default 0 for backward compat)
        """
        cfg = self.cfg
        idx = 0
        indices = {}

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            ar_read_num = MMAveragerProgram.active_reset_read_num(**params)
            if params.get('pre_selection_reset', False):
                indices['ar_pre_selection'] = idx + ar_read_num - 1
            idx += ar_read_num

        if cfg.expt.get('state_prep_postselect', False):
            indices['state_prep_ps'] = idx
            idx += 1

        if n_active_checks > 0:
            indices['jp'] = list(range(idx, idx + n_active_checks))
            idx += n_active_checks

        if cfg.expt.get('reset_before_dual_rail', False):
            indices['dr_reset_before'] = idx
            idx += 1

        indices['dr_stor_swap'] = idx
        indices['dr_stor_parity'] = idx + 1
        idx += 2

        if cfg.expt.get('reset_after_dual_rail', False):
            indices['dr_reset_after'] = idx

        return indices

    def _bin_dual_rail_shots(self, i0_filtered, indices, threshold, measure_parity=True):
        """Bin dual rail shots into 00, 10, 01, 11 populations."""
        n_shots = len(i0_filtered)
        if n_shots == 0:
            return {'00': 0, '01': 0, '10': 0, '11': 0}

        swap_shots = i0_filtered[:, indices['dr_stor_swap']]
        parity_shots = i0_filtered[:, indices['dr_stor_parity']]

        # Threshold: > threshold = qubit |e> = odd parity = 1 photon
        if measure_parity:
            swap_state = (swap_shots > threshold).astype(int)
            parity_state = (parity_shots > threshold).astype(int)
        else:
            swap_state = (swap_shots < threshold).astype(int)
            parity_state = (parity_shots < threshold).astype(int)

        counts = {'00': 0, '01': 0, '10': 0, '11': 0}
        for s1, s2 in zip(swap_state, parity_state):
            key = f'{s1}{s2}'
            counts[key] += 1

        pops = {k: v / n_shots for k, v in counts.items()}
        return pops

    def _fit_decaysin(self, xpts_fit, p_0, p_1, state, data,
                       fitparams=None, prefix=''):
        """Fit decaysin1 to p_0 and p_1 and store results in data dict.

        Args:
            prefix: key prefix for storing results (e.g. 'ps_' for
                    post-selected fits -> 'fit_p0_ps_{state}')
        """
        if len(xpts_fit) <= 5:
            return

        p0_fit = None
        try:
            pOpt, pCov = fitter.fitdecaysin(
                xpts_fit, p_0, fitparams=fitparams, use_x0=False)
            data[f'fit_p0_{prefix}{state}'] = pOpt
            data[f'fit_err_p0_{prefix}{state}'] = pCov
            p0_fit = pOpt
        except Exception as e:
            print(f"Fit failed for state {state} ({prefix or 'raw'}) p_0: {e}")
            data[f'fit_p0_{prefix}{state}'] = None
            data[f'fit_err_p0_{prefix}{state}'] = None

        # Seed p_1 from p_0: flip phase by 180, flip y0
        p1_initparams = fitparams
        if p0_fit is not None:
            p1_initparams = [
                p0_fit[0], p0_fit[1], p0_fit[2] + 180,
                p0_fit[3], 1 - p0_fit[4],
            ]
        try:
            pOpt, pCov = fitter.fitdecaysin(
                xpts_fit, p_1, fitparams=p1_initparams, use_x0=False)
            data[f'fit_p1_{prefix}{state}'] = pOpt
            data[f'fit_err_p1_{prefix}{state}'] = pCov
        except Exception as e:
            print(f"Fit failed for state {state} ({prefix or 'raw'}) p_1: {e}")
            data[f'fit_p1_{prefix}{state}'] = None
            data[f'fit_err_p1_{prefix}{state}'] = None

    def analyze(self, data=None, fit=True, fitparams=None, post_select=True):
        """
        Analyze dual rail Ramsey data.

        Pass 1: Raw populations (pre-selection only, no JP post-selection).
        Pass 2: Post-selected populations (JP even parity filter).
                 Only runs when post_select=True and n_checks > 0.
        """
        if data is None:
            data = self.data

        state_list = data.get('states', ['+'])
        threshold = data.get('threshold')
        xpts = data.get('xpts')
        n_checks = int(data.get('n_checks', 0))
        reps = int(data.get('reps', self.cfg.expt.reps))
        rounds = int(data.get('rounds', self.cfg.expt.get('rounds', 1)))
        total_reps = reps * rounds
        expts = self.cfg.expt.expts
        measure_parity = self.cfg.expt.get('measure_parity', True)

        bar_labels = ['00', '10', '01', '11']
        indices = self._get_shot_indices(n_active_checks=n_checks)

        # Compute true_times (constant overhead for all points)
        jp_overhead = data.get('jp_overhead', 0)
        echo_overhead = data.get('echo_overhead', 0)
        if n_checks > 0 or echo_overhead > 0:
            true_times = xpts + n_checks * jp_overhead + echo_overhead
            data['true_times'] = true_times
        else:
            true_times = None

        for state in state_list:
            read_num = data.get(f'read_num_{state}')
            i0 = data.get(f'i0_{state}')
            if i0 is None or read_num is None:
                continue

            # === Pass 1: Raw populations ===
            pop_arrays = {label: np.zeros(expts) for label in bar_labels}
            raw_counts = np.zeros(expts)
            jp_even_frac = (np.full((expts, n_checks), np.nan)
                            if n_checks > 0 else None)

            # === Pass 2: Post-selected populations ===
            do_ps = post_select and n_checks > 0
            ps_pop_arrays = ({label: np.zeros(expts) for label in bar_labels}
                             if do_ps else None)
            ps_counts = np.zeros(expts) if do_ps else None

            for expt_idx in range(expts):
                i0_row = i0[expt_idx]
                i0_at_point = i0_row.reshape(total_reps, read_num)

                if 'ar_pre_selection' in indices:
                    mask = i0_at_point[:, indices['ar_pre_selection']] < threshold
                    i0_at_point = i0_at_point[mask]

                # State-prep post-selection
                if 'state_prep_ps' in indices:
                    sp_mask = i0_at_point[:, indices['state_prep_ps']] < threshold
                    i0_at_point = i0_at_point[sp_mask]

                # JP even fraction (computed but NOT used for Pass 1 filtering)
                if n_checks > 0 and 'jp' in indices:
                    jp_idx = indices['jp']
                    jp_shots = i0_at_point[:, jp_idx]
                    jp_ef = np.mean(jp_shots < threshold, axis=0)
                    jp_even_frac[expt_idx, :] = jp_ef

                raw_counts[expt_idx] = len(i0_at_point)

                # Raw populations (no JP post-selection)
                pops = self._bin_dual_rail_shots(
                    i0_at_point, indices, threshold, measure_parity)

                for label in bar_labels:
                    pop_arrays[label][expt_idx] = pops[label]

                # Post-selected populations
                if do_ps and 'jp' in indices:
                    jp_idx = indices['jp']
                    jp_shots = i0_at_point[:, jp_idx]
                    ps_mask = np.all(jp_shots < threshold, axis=1)
                    i0_ps = i0_at_point[ps_mask]
                    ps_counts[expt_idx] = len(i0_ps)

                    ps_pops = self._bin_dual_rail_shots(
                        i0_ps, indices, threshold, measure_parity)
                    for label in bar_labels:
                        ps_pop_arrays[label][expt_idx] = ps_pops.get(
                            label, 0)

            # Store raw populations
            for label in bar_labels:
                data[f'pop_{label}_{state}'] = pop_arrays[label]
            data[f'raw_counts_{state}'] = raw_counts
            if jp_even_frac is not None:
                data[f'jp_even_frac_{state}'] = jp_even_frac

            pop_10 = pop_arrays['10']
            pop_01 = pop_arrays['01']
            logical_total = pop_10 + pop_01
            with np.errstate(divide='ignore', invalid='ignore'):
                p_1 = np.where(logical_total > 0, pop_10 / logical_total, 0.5)
                p_0 = np.where(logical_total > 0, pop_01 / logical_total, 0.5)

            data[f'p_0_{state}'] = p_0
            data[f'p_1_{state}'] = p_1
            data[f'logical_total_{state}'] = logical_total

            xpts_fit = true_times[:expts] if true_times is not None else xpts[:expts]
            if fit:
                self._fit_decaysin(xpts_fit, p_0, p_1, state, data, fitparams)

            # Store post-selected data
            if do_ps:
                for label in bar_labels:
                    data[f'pop_ps_{label}_{state}'] = ps_pop_arrays[label]
                data[f'ps_counts_{state}'] = ps_counts

                # Post-selected logical populations
                ps_10 = ps_pop_arrays['10']
                ps_01 = ps_pop_arrays['01']
                ps_logical = ps_10 + ps_01
                with np.errstate(divide='ignore', invalid='ignore'):
                    ps_p_1 = np.where(ps_logical > 0,
                                      ps_10 / ps_logical, 0.5)
                    ps_p_0 = np.where(ps_logical > 0,
                                      ps_01 / ps_logical, 0.5)

                data[f'p_0_ps_{state}'] = ps_p_0
                data[f'p_1_ps_{state}'] = ps_p_1
                data[f'logical_total_ps_{state}'] = ps_logical

                if fit:
                    self._fit_decaysin(
                        xpts_fit, ps_p_0, ps_p_1, state, data,
                        fitparams, prefix='ps_')

        return data

    @staticmethod
    def _get_pops(data, state, prefix='pop'):
        """Reconstruct pops dict from individual arrays in data."""
        pops = {}
        for label in ['00', '10', '01', '11']:
            key = f'{prefix}_{label}_{state}'
            if key in data:
                pops[label] = data[key]
        return pops if pops else None

    def _display_fit_overlay(self, ax, xpts_plot, state, data, title_parts,
                              fit_prefix=''):
        """Add decaysin fit overlays and T2 info to title_parts."""
        xfit = np.linspace(xpts_plot[0], xpts_plot[-1], 200)
        for p_label, color_key, name in [('p1', '10', '$p_1$'),
                                          ('p0', '01', '$p_0$')]:
            fit_key = f'fit_{p_label}_{fit_prefix}{state}'
            fit_p = data.get(fit_key)
            if fit_p is not None and isinstance(fit_p, (list, np.ndarray)):
                fit_err = data.get(f'fit_err_{p_label}_{fit_prefix}{state}')
                ax.plot(xfit, fitter.decaysin1(xfit, *fit_p), '--',
                        color=self.STATE_COLORS[color_key], linewidth=2)
                T2 = fit_p[3]
                try:
                    T2_err = np.sqrt(fit_err[3][3])
                except (TypeError, IndexError):
                    T2_err = 0
                fit_freq = fit_p[1]
                title_parts.append(
                    f'{name}: $T_2$ = {T2:.3g} $\\pm$ {T2_err:.2g} us, '
                    f'f = {fit_freq:.4g} MHz')
                print(f"State {state} ({fit_prefix or 'raw'}), {name}: "
                      f"T2 = {T2:.4g} +/- {T2_err:.3g} us, "
                      f"fit freq = {fit_freq:.6g} MHz")

    def display(self, data=None, fit=True, show_iq=False, n_iq_panels=8, **kwargs):
        if data is None:
            data = self.data

        state_list = data.get('states', ['+'])
        xpts = data.get('xpts')
        bar_labels = ['00', '10', '01', '11']
        stor_pair = f'S{self.cfg.expt.storage_swap}-S{self.cfg.expt.storage_parity}'
        n_checks = int(data.get('n_checks', 0))

        n_states = len(state_list)
        num_echoes = self.cfg.expt.get('echoes', 0)
        if num_echoes > 0:
            echo_gate = self.cfg.expt.get('echo_gate', 'Y')
            expt_type = f'Echo (N={num_echoes}, {echo_gate})'
        else:
            expt_type = 'Ramsey'

        expts = self.cfg.expt.expts

        # Determine x-axis: use true_times if available (JP mode), else xpts
        true_times = data.get('true_times')
        if true_times is not None:
            xpts_plot = true_times[:expts]
            x_label = 'Elapsed time [us]'
        else:
            xpts_plot = xpts[:expts]
            x_label = 'Wait time [us]'

        def _freq_str(state):
            eff_freq = data.get(f'effective_ramsey_freq_{state}')
            if eff_freq is not None and eff_freq != self.cfg.expt.ramsey_freq:
                return (f'Ramsey freq: {self.cfg.expt.ramsey_freq} MHz '
                        f'(eff: {eff_freq:.6g} MHz)')
            return f'Ramsey freq: {self.cfg.expt.ramsey_freq} MHz'

        # === Plot 1: Population plot ===
        fig_pop, axes_pop = plt.subplots(1, n_states,
                                          figsize=(8 * n_states, 5),
                                          squeeze=False)
        axes_pop = axes_pop.flatten()

        for state_idx, state in enumerate(state_list):
            ax = axes_pop[state_idx]
            for label in bar_labels:
                pop = data.get(f'pop_{label}_{state}')
                if pop is not None:
                    ax.plot(xpts_plot, pop, 'o-',
                            label=r'$|%s\rangle$' % label,
                            color=self.STATE_COLORS[label],
                            markersize=4)

            ax.set_xlabel(x_label)
            ax.set_ylabel('Population')
            title = f'Dual Rail {expt_type} [{stor_pair}]\n'
            title += f'Prepared: |{state}>, {_freq_str(state)}'
            if n_checks > 0:
                title += f' ({n_checks} JP checks)'
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.show()

        # === Plot 2: Logical subspace ===
        fig_log, axes_log = plt.subplots(1, n_states,
                                          figsize=(8 * n_states, 5),
                                          squeeze=False)
        axes_log = axes_log.flatten()

        for state_idx, state in enumerate(state_list):
            ax = axes_log[state_idx]

            p_0 = data.get(f'p_0_{state}')
            p_1 = data.get(f'p_1_{state}')
            logical_total = data.get(f'logical_total_{state}')

            if p_1 is not None:
                ax.plot(xpts_plot, p_1, 'o',
                        label=r'$p_1$', color=self.STATE_COLORS['10'],
                        markersize=5)
            if p_0 is not None:
                ax.plot(xpts_plot, p_0, 's',
                        label=r'$p_0$', color=self.STATE_COLORS['01'],
                        markersize=5)

            title_parts = [f'Dual Rail {expt_type} [{stor_pair}] (Logical Subspace)',
                           f'Prepared: |{state}>, {_freq_str(state)}'
                           + (f' ({n_checks} JP checks)' if n_checks > 0 else '')]

            if fit:
                self._display_fit_overlay(
                    ax, xpts_plot, state, data, title_parts)

            ax.set_xlabel(x_label)
            ax.set_ylabel('Logical Population')
            ax.set_title('\n'.join(title_parts))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

            if logical_total is not None:
                ax2 = ax.twinx()
                ax2.plot(xpts_plot, 1 - logical_total, 'x-',
                         color='gray', alpha=0.5, markersize=3, label='Leakage')
                ax2.set_ylabel('Leakage', color='gray')
                ax2.set_ylim([-0.05, 1.05])
                ax2.tick_params(axis='y', labelcolor='gray')

        plt.tight_layout()
        plt.show()

        # === Plot 3: Post-selected population ===
        has_ps = (n_checks > 0
                  and f'pop_ps_00_{state_list[0]}' in data)

        if has_ps:
            fig_ps, axes_ps = plt.subplots(
                1, n_states, figsize=(8 * n_states, 5), squeeze=False)
            axes_ps = axes_ps.flatten()

            for state_idx, state in enumerate(state_list):
                ax = axes_ps[state_idx]

                # Background: raw population as thin lines
                for label in bar_labels:
                    raw_pop = data.get(f'pop_{label}_{state}')
                    if raw_pop is not None:
                        ax.plot(xpts_plot, raw_pop, '-',
                                color=self.STATE_COLORS[label],
                                alpha=0.3, linewidth=1)

                # Post-selected population
                for label in bar_labels:
                    ps_pop = data.get(f'pop_ps_{label}_{state}')
                    if ps_pop is not None:
                        ax.plot(xpts_plot, ps_pop, 'o-',
                                color=self.STATE_COLORS[label],
                                markersize=4,
                                label=r'PS $|%s\rangle$' % label)

                # Annotations: ps_count / raw_count
                ps_counts_arr = data.get(f'ps_counts_{state}')
                raw_counts_arr = data.get(f'raw_counts_{state}')
                if ps_counts_arr is not None and raw_counts_arr is not None:
                    n_annot = min(10, len(xpts_plot))
                    annot_indices = np.linspace(
                        0, len(xpts_plot) - 1, n_annot, dtype=int)
                    for ai in annot_indices:
                        rc = int(raw_counts_arr[ai])
                        pc = int(ps_counts_arr[ai])
                        ax.annotate(
                            f'{pc}/{rc}',
                            (xpts_plot[ai], 0.02),
                            fontsize=7, ha='center', alpha=0.7, rotation=90)

                ax.set_xlabel(x_label)
                ax.set_ylabel('Population')
                title = f'Post-selected: Dual Rail {expt_type} [{stor_pair}]\n'
                title += f'Prepared: |{state}>, {_freq_str(state)}'
                title += f' ({n_checks} JP checks)'
                ax.set_title(title)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.05, 1.05])

            plt.tight_layout()
            plt.show()

            # === Plot 4: Post-selected logical subspace ===
            fig_ps_log, axes_ps_log = plt.subplots(
                1, n_states, figsize=(8 * n_states, 5), squeeze=False)
            axes_ps_log = axes_ps_log.flatten()

            for state_idx, state in enumerate(state_list):
                ax = axes_ps_log[state_idx]

                # Background: raw logical as thin lines
                p_0_raw = data.get(f'p_0_{state}')
                p_1_raw = data.get(f'p_1_{state}')
                if p_1_raw is not None:
                    ax.plot(xpts_plot, p_1_raw, '-',
                            color=self.STATE_COLORS['10'],
                            alpha=0.3, linewidth=1, label=r'Raw $p_1$')
                if p_0_raw is not None:
                    ax.plot(xpts_plot, p_0_raw, '-',
                            color=self.STATE_COLORS['01'],
                            alpha=0.3, linewidth=1, label=r'Raw $p_0$')

                # Post-selected logical
                ps_p_1 = data.get(f'p_1_ps_{state}')
                ps_p_0 = data.get(f'p_0_ps_{state}')
                if ps_p_1 is not None:
                    ax.plot(xpts_plot, ps_p_1, 'o',
                            label=r'PS $p_1$',
                            color=self.STATE_COLORS['10'], markersize=5)
                if ps_p_0 is not None:
                    ax.plot(xpts_plot, ps_p_0, 's',
                            label=r'PS $p_0$',
                            color=self.STATE_COLORS['01'], markersize=5)

                title_parts = [
                    f'PS Logical: Dual Rail {expt_type} [{stor_pair}]',
                    f'Prepared: |{state}>, {_freq_str(state)}'
                    + f' ({n_checks} JP checks)']

                if fit:
                    self._display_fit_overlay(
                        ax, xpts_plot, state, data, title_parts,
                        fit_prefix='ps_')

                ax.set_xlabel(x_label)
                ax.set_ylabel('Logical Population')
                ax.set_title('\n'.join(title_parts))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.05, 1.05])

                # Leakage on twin axis
                ps_logical = data.get(f'logical_total_ps_{state}')
                if ps_logical is not None:
                    ax2 = ax.twinx()
                    ax2.plot(xpts_plot, 1 - ps_logical, 'x-',
                             color='gray', alpha=0.5, markersize=3,
                             label='PS Leakage')
                    ax2.set_ylabel('Leakage', color='gray')
                    ax2.set_ylim([-0.05, 1.05])
                    ax2.tick_params(axis='y', labelcolor='gray')

            plt.tight_layout()
            plt.show()

        # === Plot 5: JP even fraction ===
        if n_checks > 0:
            fig_jp, axes_jp = plt.subplots(
                1, n_states, figsize=(8 * n_states, 5), squeeze=False)
            axes_jp = axes_jp.flatten()

            for state_idx, state in enumerate(state_list):
                ax = axes_jp[state_idx]
                jp_ef = data.get(f'jp_even_frac_{state}')
                if jp_ef is None:
                    continue

                for k in range(n_checks):
                    col = jp_ef[:, k]
                    valid_mask = ~np.isnan(col)
                    if np.any(valid_mask):
                        ax.plot(xpts_plot[valid_mask], col[valid_mask], 'o-',
                                label=f'Check {k+1}',
                                markersize=4)

                ax.set_xlabel(x_label)
                ax.set_ylabel('JP Even Fraction')
                ax.set_title(r'JP Results: $|%s\rangle$ [%s] (%d JP checks)'
                             % (state, stor_pair, n_checks))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.05, 1.05])

            plt.tight_layout()
            plt.show()

        # === IQ scatter plot (non-JP mode only) ===
        figs_iq = []
        if show_iq and n_checks == 0:
            threshold = data.get('threshold')
            indices = self._get_shot_indices()
            total_reps = self.cfg.expt.reps * self.cfg.expt.get('rounds', 1)

            for state in state_list:
                i0 = data.get(f'i0_{state}')
                q0 = data.get(f'q0_{state}')
                read_num = data.get(f'read_num_{state}')
                if i0 is None or q0 is None or read_num is None:
                    continue

                n_panels = min(n_iq_panels, expts)
                panel_indices = np.linspace(0, expts - 1, n_panels, dtype=int)

                ncols = min(4, n_panels)
                nrows = int(np.ceil(n_panels / ncols))
                fig_iq, axes_iq = plt.subplots(nrows, ncols,
                                                figsize=(4 * ncols, 3.5 * nrows),
                                                squeeze=False)
                axes_iq_flat = axes_iq.flatten()

                for panel_idx, expt_idx in enumerate(panel_indices):
                    ax = axes_iq_flat[panel_idx]
                    wait_time = xpts_plot[expt_idx]

                    i0_at_point = i0[expt_idx].reshape(total_reps, read_num)
                    q0_at_point = q0[expt_idx].reshape(total_reps, read_num)

                    i_swap = i0_at_point[:, indices['dr_stor_swap']]
                    q_swap = q0_at_point[:, indices['dr_stor_swap']]
                    i_parity = i0_at_point[:, indices['dr_stor_parity']]
                    q_parity = q0_at_point[:, indices['dr_stor_parity']]

                    ax.scatter(i_swap, q_swap, alpha=0.3, s=10,
                               label=f'S{self.cfg.expt.storage_swap}')
                    ax.scatter(i_parity, q_parity, alpha=0.3, s=10,
                               label=f'S{self.cfg.expt.storage_parity}')

                    if threshold is not None:
                        ax.axvline(x=threshold, color='red', linestyle='--',
                                   linewidth=1, alpha=0.7)

                    ax.set_title(f't={wait_time:.2f} us', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    if panel_idx == 0:
                        ax.legend(fontsize=7, loc='upper right')

                for idx in range(n_panels, len(axes_iq_flat)):
                    axes_iq_flat[idx].set_visible(False)

                fig_iq.suptitle(f'IQ Shots [{stor_pair}] - Prepared: |{state}>',
                                fontsize=12)
                fig_iq.supxlabel('I [ADC]')
                fig_iq.supylabel('Q [ADC]')
                plt.tight_layout()
                plt.show()
                figs_iq.append(fig_iq)

        return fig_pop, fig_log, figs_iq

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data

        save_data = {}
        for key, value in data.items():
            if key == 'states':
                if isinstance(value, list):
                    save_data[key] = np.array(value, dtype='S')
                else:
                    save_data[key] = value
            elif value is None:
                continue
            elif isinstance(value, dict):
                print(f"Warning: Skipping dict field '{key}' for HDF5 save")
                continue
            elif isinstance(value, np.ndarray) and value.dtype == object:
                try:
                    save_data[key] = np.array(value.tolist())
                except (ValueError, TypeError):
                    print(f"Warning: Skipping object-dtype field '{key}' for HDF5 save")
                    continue
            else:
                save_data[key] = value

        super().save_data(data=save_data)
        return self.fname

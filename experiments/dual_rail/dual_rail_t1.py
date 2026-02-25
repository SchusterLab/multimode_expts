'''
Dual Rail T1 Experiment

Measures dual-rail T1 (population decay vs wait time) with optional
joint parity checks distributed as equal waits.

Pulse sequence (distributed equal waits, like dual_rail_ramsey):
  n_checks=0: prep -> wait -> measure
  n_checks>0: prep -> [wait -> JP]*n -> wait -> measure

  All waits are equal = total_variable_time / (n_checks + 1),
  swept in firmware via a single RAverager program.

Seb 02/2026
'''

import matplotlib.pyplot as plt
import numpy as np

from slab import Experiment, AttrDict
from fitting.fitting import expfunc1
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import *


class DualRailT1Program(MMRAveragerProgram):
    """RAverager program for dual-rail T1 measurement.

    Sweeps total wait time on the FPGA with distributed equal waits
    between joint parity checks (like DualRailRamseyProgram).

    Config params:
        n_active_checks: int, number of JP checks
        start: float, total wait start [us]
        step: float, total wait step [us]
        expts: int, number of sweep points
    """
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)

        # Build storage mode names
        man = cfg.expt.manipulate
        s_swap = cfg.expt.storage_swap
        s_parity = cfg.expt.storage_parity
        self.stor_name_swap = f'M{man}-S{s_swap}'
        self.stor_name_parity = f'M{man}-S{s_parity}'
        self.stor_pair_name = f'M{man}-S{s_parity}'

        # Build swap pulse (storage_swap <-> manipulate)
        self.swap_pulse = self.get_prepulse_creator(
            [['storage', self.stor_name_swap, 'pi', 0]]
        ).pulse.tolist()

        # Build state preparation pulse
        state_start = cfg.expt.get('state_start', '00')
        self.state_start = state_start
        state_prep_seq = self.prep_dual_rail_state(
            state_start, self.stor_name_swap, self.stor_name_parity)
        if state_prep_seq is not None:
            self.state_prep_pulse = self.get_prepulse_creator(
                state_prep_seq).pulse.tolist()
        else:
            self.state_prep_pulse = None

        # Joint parity check configuration
        self.n_active_checks = int(cfg.expt.get('n_active_checks', 0))

        # Distributed equal waits (like Ramsey)
        n_wait_segments = self.n_active_checks + 1
        self.n_wait_segments = n_wait_segments

        # Setup swept wait register on a flux channel page
        # (matches Ramsey pattern: keeps wait register separate from qubit
        # channel operations during JP checks)
        self.r_wait = 3
        flux_low_ch = cfg.hw.soc.dacs.flux_low.ch[0]
        self.wait_page = self.ch_page(flux_low_ch)
        self.safe_regwi(self.wait_page, self.r_wait,
                        self.us2cycles(cfg.expt.start / n_wait_segments))

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)

        # 1. Phase reset
        self.reset_and_sync()

        # 2. Active reset
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # 3. State preparation
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

        # 5. Distributed waits with JP checks: [wait -> JP]*n -> wait
        if self.n_active_checks > 0:
            for k in range(self.n_active_checks):
                self.sync_all()
                self.sync(self.wait_page, self.r_wait)
                self.sync_all()
                # Use shared prefixes to avoid duplicate waveform allocations
                # (all swap pulses are identical; unique prefixes waste generator memory)
                self.custom_pulse(cfg, self.swap_pulse,
                                  prefix='jp_swap_in_')
                self.joint_parity_active_reset(
                    stor_pair_name=self.stor_pair_name,
                    name=f'jp_{k}',
                    register_label=f'jp_label_{k}',
                    second_phase=0,
                    fast=cfg.expt.get('parity_fast', False))
                self.custom_pulse(cfg, self.swap_pulse,
                                  prefix='jp_swap_out_')

        # 5. Final wait
        self.sync_all()
        self.sync(self.wait_page, self.r_wait)
        self.sync_all()

        # 6. Final dual rail measurement
        self.measure_dual_rail(
            storage_idx=(cfg.expt.storage_swap, cfg.expt.storage_parity),
            measure_parity=cfg.expt.get('measure_parity', True),
            reset_before=cfg.expt.get('reset_before_dual_rail', False),
            reset_after=cfg.expt.get('reset_after_dual_rail', False),
            man_idx=cfg.expt.manipulate,
            final_sync=True)

    def update(self):
        step_cycles = self.us2cycles(
            self.cfg.expt.step / self.n_wait_segments)
        self.mathi(self.wait_page, self.r_wait, self.r_wait, '+',
                   step_cycles)

    def _calculate_read_num(self):
        """Calculate total number of readouts per rep."""
        cfg = self.cfg
        read_num = 0

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        if cfg.expt.get('state_prep_postselect', False):
            read_num += 1

        read_num += self.n_active_checks  # one JP readout per check

        read_num += 2  # dual-rail measurements (one per storage)
        if cfg.expt.get('reset_before_dual_rail', False):
            read_num += 1
        if cfg.expt.get('reset_after_dual_rail', False):
            read_num += 1

        return read_num

    def collect_shots(self):
        """Collect single-shot data (same layout as DualRailRamseyProgram).

        Returns (i0, q0, read_num) with shape (expts, total_reps * read_num).
        """
        read_num = self._calculate_read_num()
        expts = self.cfg.expt.expts
        reps = self.cfg.expt.reps
        rounds = self.cfg.expt.get('rounds', 1)
        total_reps = rounds * reps

        buf_i = self.di_buf[0] / self.readout_lengths_adc[0]
        buf_q = self.dq_buf[0] / self.readout_lengths_adc[0]

        shots_i0 = buf_i.reshape((rounds, expts, reps, read_num))
        shots_i0 = shots_i0.transpose(1, 0, 2, 3).reshape(
            (expts, total_reps * read_num))
        shots_q0 = buf_q.reshape((rounds, expts, reps, read_num))
        shots_q0 = shots_q0.transpose(1, 0, 2, 3).reshape(
            (expts, total_reps * read_num))

        return shots_i0, shots_q0, read_num


class DualRailT1Experiment(Experiment):
    """
    Dual Rail T1 Experiment

    Sweeps total wait time and measures dual-rail populations, with optional
    joint parity checks distributed as equal waits (like DualRailRamsey).

    Experimental Config:
    expt = dict(
        start: total time sweep start [us],
        step: total time sweep step [us],
        expts: number of time points,
        reps: averages per point,
        rounds: number of rounds (default 1),
        n_checks: number of distributed JP checks (0 = pure T1),
        state_start: str or list of str ('00', '10', '01', '11'),
        storage_swap: storage mode swapped to/from manipulate (e.g., 1 for S1),
        storage_parity: storage mode probed by joint parity pulse (e.g., 3 for S3),
        manipulate: manipulate mode number (e.g., 1 for M1),
        active_reset: True/False,
        state_prep_postselect: True/False (measure qubit after state prep for post-selection),
        state_prep_ps_parity: True/False (play parity pulse before state-prep PS measurement),
        parity_fast: True/False (use fast multiphoton hpi for JP),
        measure_parity: True/False (True = parity measurement, False = slow pi),
        reset_before_dual_rail: True/False,
        reset_after_dual_rail: True/False,
    )
    """

    STATE_COLORS = {
        '00': 'tab:blue',
        '10': 'tab:orange',
        '01': 'tab:green',
        '11': 'tab:red',
    }

    def __init__(self, soccfg=None, path='', prefix='DualRailT1',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        # Handle state_start as list or single string
        state_list = self.cfg.expt.get('state_start', '00')
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

        threshold = self.cfg.device.readout.threshold[0]

        data = {
            'xpts': times,
            'states': np.array(state_list, dtype='S4'),
            'n_checks': n_checks,
            'threshold': threshold,
            'reps': reps,
            'rounds': rounds,
            'step': step,
        }

        n_wait_segments = n_checks + 1
        print(f"Dual Rail T1: {expts} time points from {start} to "
              f"{times[-1]} us, step={step} us, {n_checks} JP checks")
        if n_checks > 0:
            print(f"  Distributed equal waits: {n_wait_segments} segments")

        # Set n_active_checks for JP mode
        if n_checks > 0:
            self.cfg.expt.n_active_checks = n_checks

        for state in tqdm(state_list, desc="States", disable=not progress):
            print(f"\n=== Running state_start='{state}' ===")
            self.cfg.expt.state_start = state

            prog = DualRailT1Program(soccfg=self.soccfg, cfg=self.cfg)
            read_num = prog._calculate_read_num()

            # Compute JP overhead (once, from first program)
            if n_checks > 0 and 'jp_overhead' not in data:
                swap_duration = prog.get_total_time(
                    np.array(prog.swap_pulse, dtype=object))
                jp_meas_duration = prog.get_jp_measurement_duration(
                    prog.stor_pair_name,
                    fast=self.cfg.expt.get('parity_fast', False))
                data['jp_overhead'] = 2 * swap_duration + jp_meas_duration
                print(f"  JP check overhead: {data['jp_overhead']:.2f} us "
                      f"(swap={swap_duration:.2f}, "
                      f"jp_meas={jp_meas_duration:.2f})")

            x_pts, avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num)

            i0, q0, _ = prog.collect_shots()

            data[f'i0_{state}'] = i0
            data[f'q0_{state}'] = q0
            data[f'read_num_{state}'] = read_num

        # Clean up temporary cfg params
        if n_checks > 0:
            self.cfg.expt.pop('n_active_checks', None)

        self.data = data
        return data

    def analyze(self, data=None, post_select=True):
        """
        Analyze dual rail T1 data.

        Pass 1: Raw populations (pre-selection only, no JP post-selection).
        Pass 2: Post-selected populations (JP even parity filter).
                 Only runs when post_select=True and n_checks > 0.
        """
        if data is None:
            data = self.data

        state_list = data.get('states', ['00'])
        if isinstance(state_list, np.ndarray):
            state_list = [s.decode() if isinstance(s, bytes) else s
                          for s in state_list]
        n_checks = int(data.get('n_checks', 0))
        threshold = data.get('threshold')
        reps = int(data.get('reps'))
        rounds = int(data.get('rounds', 1))
        total_reps = reps * rounds
        times = data.get('xpts')
        expts = len(times)
        measure_parity = self.cfg.expt.get('measure_parity', True)

        # Compute true elapsed times including JP overhead
        jp_overhead = data.get('jp_overhead', 0)
        if jp_overhead > 0 and n_checks > 0:
            true_times = times + n_checks * jp_overhead
        else:
            true_times = times.copy()
        data['true_times'] = true_times

        indices = self._get_shot_indices(n_checks)
        bar_labels = ['00', '10', '01', '11']

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
                per_shot = i0_row.reshape(total_reps, read_num)

                # Pre-selection (active reset)
                if 'ar_pre_selection' in indices:
                    mask = per_shot[:, indices['ar_pre_selection']] < threshold
                    per_shot = per_shot[mask]

                # State-prep post-selection
                if 'state_prep_ps' in indices:
                    sp_mask = per_shot[:, indices['state_prep_ps']] < threshold
                    per_shot = per_shot[sp_mask]

                # JP even fraction (computed but NOT used for Pass 1 filtering)
                if n_checks > 0 and 'jp' in indices:
                    jp_idx = indices['jp']
                    jp_shots = per_shot[:, jp_idx]
                    jp_ef = np.mean(jp_shots < threshold, axis=0)
                    jp_even_frac[expt_idx, :] = jp_ef

                raw_counts[expt_idx] = len(per_shot)

                # Raw populations (no JP post-selection)
                point_pops = self._bin_dual_rail_shots(
                    per_shot, indices, threshold, measure_parity)
                for label in bar_labels:
                    pop_arrays[label][expt_idx] = point_pops.get(label, 0)

                # Post-selected populations
                if do_ps and 'jp' in indices:
                    jp_idx = indices['jp']
                    jp_shots = per_shot[:, jp_idx]
                    ps_mask = np.all(jp_shots < threshold, axis=1)
                    per_shot_ps = per_shot[ps_mask]
                    ps_counts[expt_idx] = len(per_shot_ps)

                    ps_pops = self._bin_dual_rail_shots(
                        per_shot_ps, indices, threshold, measure_parity)
                    for label in bar_labels:
                        ps_pop_arrays[label][expt_idx] = ps_pops.get(
                            label, 0)

            # Shot-level debug dump: compare blip vs non-blip points
            if do_ps and n_checks > 0:
                # Detect blip points: raw01 significantly above baseline
                r01_arr = pop_arrays['01']
                r01_median = np.median(r01_arr)
                blip_mask = r01_arr > max(r01_median * 3, 0.06)
                blip_pts = np.where(blip_mask)[0]
                nonblip_pts = np.where(~blip_mask)[0]
                if len(blip_pts) > 0 and len(nonblip_pts) > 0:
                    bp = blip_pts[0]
                    nbp = nonblip_pts[0]
                    col_names = []
                    if 'ar_pre_selection' in indices:
                        ar_n = indices['ar_pre_selection'] + 1
                        col_names += [f'ar{j}' for j in range(ar_n)]
                    if 'state_prep_ps' in indices:
                        col_names.append('sp_ps')
                    if 'jp' in indices:
                        col_names += [f'jp{j}' for j in range(n_checks)]
                    col_names += ['dr_swap', 'dr_par']
                    hdr = ' '.join(f'{c:>8s}' for c in col_names)
                    for label, idx in [('BLIP', bp), ('NORMAL', nbp)]:
                        raw = i0[idx].reshape(total_reps, read_num)
                        print(f"\n  [{label}] pt={idx} t={times[idx]:.1f}us "
                              f"raw01={r01_arr[idx]:.3f} "
                              f"(threshold={threshold:.2f})")
                        print(f"  Mean: {hdr}")
                        means = raw.mean(axis=0)
                        print(f"        {' '.join(f'{m:8.2f}' for m in means)}")
                        print(f"  Std:  {' '.join(f'{s:8.2f}' for s in raw.std(axis=0))}")
                        print(f"  First 5 shots:")
                        for s in range(min(5, len(raw))):
                            vals = ' '.join(f'{raw[s,c]:8.2f}'
                                            for c in range(read_num))
                            print(f"    {vals}")

            # Store raw populations
            for label in bar_labels:
                data[f'pops_{state}_{label}'] = pop_arrays[label]
            data[f'raw_counts_{state}'] = raw_counts
            if jp_even_frac is not None:
                data[f'jp_even_frac_{state}'] = jp_even_frac

            # Compute raw logical populations
            pop_10, pop_01 = pop_arrays['10'], pop_arrays['01']
            logical_total = pop_10 + pop_01
            valid = logical_total > 0
            data[f'p_0_{state}'] = np.where(valid, pop_01 / logical_total, 0)
            data[f'p_1_{state}'] = np.where(valid, pop_10 / logical_total, 0)

            # Fit raw exponential decay (using true elapsed times)
            if state in ['10', '01']:
                try:
                    times_ms = true_times / 1000
                    y = pop_arrays[state]
                    p0 = [0.5, y[0] - 0.5,
                          (times_ms[-1] - times_ms[0]) / 5]
                    bounds = ([0, 0, 0], [1, 1.5, np.inf])
                    pOpt, pCov = curve_fit(expfunc1, times_ms, y, p0=p0,
                                           bounds=bounds, maxfev=200000)
                    T1 = pOpt[2]
                    T1_err = (np.sqrt(pCov[2, 2])
                              if pCov[2, 2] < np.inf else 0)
                    data[f'fit_{state}'] = pOpt
                    data[f'fit_err_{state}'] = np.sqrt(np.diag(pCov))
                    print(f"State |{state}> (raw): "
                          f"T1 = {T1:.4g} +/- {T1_err:.4g} ms")
                except Exception as e:
                    print(f"Fit failed for state |{state}> (raw): {e}")

            # Store post-selected data
            if do_ps:
                for label in bar_labels:
                    data[f'pops_ps_{state}_{label}'] = ps_pop_arrays[label]
                data[f'ps_counts_{state}'] = ps_counts

                # Post-selected logical populations
                ps_10, ps_01 = ps_pop_arrays['10'], ps_pop_arrays['01']
                ps_logical = ps_10 + ps_01
                ps_valid = ps_logical > 0
                data[f'p_0_ps_{state}'] = np.where(
                    ps_valid, ps_01 / ps_logical, 0)
                data[f'p_1_ps_{state}'] = np.where(
                    ps_valid, ps_10 / ps_logical, 0)

                # Fit post-selected decay
                if state in ['10', '01']:
                    try:
                        times_ms = true_times / 1000
                        y_ps = ps_pop_arrays[state]
                        p0 = [0.5, y_ps[0] - 0.5,
                              (times_ms[-1] - times_ms[0]) / 3]
                        bounds = ([0, 0, 0], [1, 1.5, np.inf])
                        pOpt_ps, pCov_ps = curve_fit(
                            expfunc1, times_ms, y_ps, p0=p0,
                            bounds=bounds, maxfev=200000)
                        T1_ps = pOpt_ps[2]
                        T1_ps_err = (np.sqrt(pCov_ps[2, 2])
                                     if pCov_ps[2, 2] < np.inf else 0)
                        data[f'fit_ps_{state}'] = pOpt_ps
                        data[f'fit_err_ps_{state}'] = np.sqrt(
                            np.diag(pCov_ps))
                        print(f"State |{state}> (post-selected): "
                              f"T1 = {T1_ps:.4g} +/- {T1_ps_err:.4g} ms")
                    except Exception as e:
                        print(f"PS fit failed for state |{state}>: {e}")

            # Diagnostic summary table
            if do_ps:
                # Build per-JP-check header
                jp_hdr = ' '.join([f'jp{j:d}' for j in range(n_checks)])
                print(f"\n--- Diagnostic: state='{state}', n_checks={n_checks} ---")
                print(f"{'pt':>3} {'time_us':>8} {'raw_ct':>7} {'ps_ct':>6} "
                      f"{'ps_surv%':>8} {'jp_avg':>7} "
                      f"{'raw10':>6} {'raw01':>6} {'ps10':>6} {'ps01':>6}"
                      f"  {jp_hdr}")
                for ei in range(expts):
                    t = times[ei]
                    rc = int(raw_counts[ei])
                    pc = int(ps_counts[ei])
                    surv = 100 * pc / rc if rc > 0 else 0
                    jp_avg = np.mean(jp_even_frac[ei]) if jp_even_frac is not None else 0
                    r10 = pop_arrays['10'][ei]
                    r01 = pop_arrays['01'][ei]
                    p10 = ps_pop_arrays['10'][ei]
                    p01 = ps_pop_arrays['01'][ei]
                    # Per-JP-check even fraction
                    if jp_even_frac is not None:
                        jp_per = ' '.join([f'{jp_even_frac[ei, j]:.2f}'
                                           for j in range(n_checks)])
                    else:
                        jp_per = ''
                    # Flag points where raw01 is anomalously high
                    flag = ' <--' if r01 > 0.06 else ''
                    print(f"{ei:3d} {t:8.1f} {rc:7d} {pc:6d} "
                          f"{surv:7.1f}% {jp_avg:7.3f} "
                          f"{r10:6.3f} {r01:6.3f} {p10:6.3f} {p01:6.3f}"
                          f"  {jp_per}{flag}")
                print(f"--- End diagnostic ---\n")

        return data

    def _get_shot_indices(self, n_active_checks=0):
        """Return dict mapping measurement type to column indices."""
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

    def _bin_dual_rail_shots(self, i0_filtered, indices, threshold,
                             measure_parity=True):
        """Bin dual rail shots into 00, 10, 01, 11 populations."""
        n_shots = len(i0_filtered)
        if n_shots == 0:
            return {'00': 0, '01': 0, '10': 0, '11': 0}

        swap_shots = i0_filtered[:, indices['dr_stor_swap']]
        parity_shots = i0_filtered[:, indices['dr_stor_parity']]

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

    @staticmethod
    def _get_pops(data, state, prefix='pops'):
        """Reconstruct pops dict from individual arrays in data."""
        pops = {}
        for label in ['00', '10', '01', '11']:
            key = f'{prefix}_{state}_{label}'
            if key in data:
                pops[label] = data[key]
        return pops if pops else None

    def display(self, data=None, fit=True, **kwargs):
        """Display dual rail T1 results.

        Plot 1: Raw population vs time
        Plot 2: Raw logical subspace (p_0, p_1)
        Plot 3 (if JP): Post-selected population
        Plot 4 (if JP): Post-selected logical
        Plot 5 (if JP): JP even fraction
        """
        if data is None:
            data = self.data

        state_list = data.get('states', ['00'])
        times = data.get('xpts')
        true_times = data.get('true_times', times)
        n_checks = int(data.get('n_checks', 0))
        jp_overhead = data.get('jp_overhead', 0)
        times_ms = true_times / 1000

        if jp_overhead > 0 and n_checks > 0:
            x_label = 'Elapsed time (ms)'
        else:
            x_label = 'Total time (ms)'

        if isinstance(state_list, np.ndarray):
            state_list = [s.decode() if isinstance(s, bytes) else s
                          for s in state_list]

        stor_pair = (f'S{self.cfg.expt.storage_swap}'
                     f'-S{self.cfg.expt.storage_parity}')
        bar_labels = ['00', '10', '01', '11']
        n_states = len(state_list)
        ncols = min(2, n_states)
        nrows = (n_states + ncols - 1) // ncols

        has_ps = (n_checks > 0
                  and f'pops_ps_{state_list[0]}_00' in data)

        # === Plot 1: Raw population vs total time ===
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(8 * ncols, 5 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for state_idx, prepared_state in enumerate(state_list):
            ax = axes_flat[state_idx]
            pops = self._get_pops(data, prepared_state)
            if pops is None:
                continue

            for measured_state in bar_labels:
                y = pops.get(measured_state)
                if y is None:
                    continue
                ax.plot(times_ms, y, 'o-',
                        label=r'$|%s\rangle$' % measured_state,
                        color=self.STATE_COLORS[measured_state],
                        markersize=4)

                if (fit and prepared_state == measured_state
                        and f'fit_{prepared_state}' in data):
                    pOpt = data[f'fit_{prepared_state}']
                    t_fit = np.linspace(times_ms[0], times_ms[-1], 200)
                    ax.plot(t_fit, expfunc1(t_fit, *pOpt), '--',
                            color=self.STATE_COLORS[measured_state],
                            linewidth=2,
                            label=f'T1={pOpt[2]:.3g} ms (raw)')

            ax.set_xlabel(x_label)
            ax.set_ylabel('Population')
            title = r'Raw: $|%s\rangle$ [%s]' % (prepared_state, stor_pair)
            if n_checks > 0:
                title += f' ({n_checks} JP checks)'
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

        for idx in range(n_states, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.tight_layout()
        plt.show()

        # === Plot 2: Raw logical population ===
        fig_log, axes_log = plt.subplots(nrows, ncols,
                                          figsize=(8 * ncols, 5 * nrows),
                                          squeeze=False)
        axes_log_flat = axes_log.flatten()

        for state_idx, prepared_state in enumerate(state_list):
            ax = axes_log_flat[state_idx]

            p_0 = data.get(f'p_0_{prepared_state}')
            p_1 = data.get(f'p_1_{prepared_state}')
            if p_0 is None or p_1 is None:
                pops = self._get_pops(data, prepared_state)
                if pops is None:
                    continue
                pop_10 = pops.get('10', np.zeros_like(times))
                pop_01 = pops.get('01', np.zeros_like(times))
                logical_total = pop_10 + pop_01
                valid = logical_total > 0
                p_0 = np.where(valid, pop_01 / logical_total, 0)
                p_1 = np.where(valid, pop_10 / logical_total, 0)

            ax.plot(times_ms, p_0, 'o', label=r'$p_0$',
                    color=self.STATE_COLORS['01'], markersize=4)
            ax.plot(times_ms, p_1, 's', label=r'$p_1$',
                    color=self.STATE_COLORS['10'], markersize=4)

            fit_text = []
            for label, y_data, color in [
                    ('p_0', p_0, self.STATE_COLORS['01']),
                    ('p_1', p_1, self.STATE_COLORS['10'])]:
                if fit and len(times_ms) >= 3 and np.max(times_ms) > 0:
                    try:
                        p0 = [0.5, y_data[0] - 0.5,
                              (times_ms[-1] - times_ms[0]) / 5]
                        bounds = ([0, -2, 0], [1, 2, np.inf])
                        pOpt, pCov = curve_fit(expfunc1, times_ms, y_data,
                                               p0=p0, bounds=bounds,
                                               maxfev=200000)
                        T = pOpt[2]
                        T_err = (np.sqrt(pCov[2, 2])
                                 if pCov[2, 2] < np.inf else 0)
                        t_fit = np.linspace(times_ms[0], times_ms[-1], 200)
                        ax.plot(t_fit, expfunc1(t_fit, *pOpt), '--',
                                color=color, linewidth=2)
                        fit_text.append(
                            r'$T_{%s}=%.3g \pm %.3g$ ms'
                            % (label, T, T_err))
                    except Exception:
                        pass

            title = (r'Raw Logical: $|%s\rangle$ [%s]'
                     % (prepared_state, stor_pair))
            if fit_text:
                title += '\n' + ', '.join(fit_text)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Logical Population')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

        for idx in range(n_states, len(axes_log_flat)):
            axes_log_flat[idx].set_visible(False)
        plt.tight_layout()
        plt.show()

        # === Plot 3: Post-selected population ===
        if has_ps:
            fig_ps, axes_ps = plt.subplots(
                nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
            axes_ps_flat = axes_ps.flatten()

            for state_idx, prepared_state in enumerate(state_list):
                ax = axes_ps_flat[state_idx]

                # Background: raw population
                pops_raw = self._get_pops(data, prepared_state)
                if pops_raw is not None and prepared_state in pops_raw:
                    ax.plot(times_ms, pops_raw[prepared_state], '-',
                            color=self.STATE_COLORS[prepared_state],
                            alpha=0.4, linewidth=1,
                            label=r'Raw $|%s\rangle$' % prepared_state)

                # Post-selected population
                for ms in bar_labels:
                    ps_key = f'pops_ps_{prepared_state}_{ms}'
                    if ps_key not in data:
                        continue
                    y_ps = data[ps_key]
                    ax.plot(times_ms, y_ps, 'o-',
                            color=self.STATE_COLORS[ms], markersize=4,
                            label=r'PS $|%s\rangle$' % ms)

                # Post-selected fit overlay
                if (fit and prepared_state in ['10', '01']
                        and f'fit_ps_{prepared_state}' in data):
                    pOpt_ps = data[f'fit_ps_{prepared_state}']
                    t_fit = np.linspace(times_ms[0], times_ms[-1], 200)
                    ax.plot(t_fit, expfunc1(t_fit, *pOpt_ps), '--',
                            color=self.STATE_COLORS[prepared_state],
                            linewidth=2,
                            label=f'PS T1={pOpt_ps[2]:.3g} ms')

                # Annotations: ps_count / raw_count
                ps_counts = data.get(f'ps_counts_{prepared_state}')
                raw_counts = data.get(f'raw_counts_{prepared_state}')
                if ps_counts is not None and raw_counts is not None:
                    # Annotate a subset of points to avoid clutter
                    n_annot = min(10, len(times_ms))
                    annot_indices = np.linspace(
                        0, len(times_ms) - 1, n_annot, dtype=int)
                    for ai in annot_indices:
                        rc = int(raw_counts[ai])
                        pc = int(ps_counts[ai])
                        ax.annotate(
                            f'{pc}/{rc}',
                            (times_ms[ai], 0.02),
                            fontsize=7, ha='center', alpha=0.7, rotation=90)

                ax.set_xlabel(x_label)
                ax.set_ylabel('Population')
                ax.set_title(r'Post-selected: $|%s\rangle$ [%s] (%d JP)'
                             % (prepared_state, stor_pair, n_checks))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])

            for idx in range(n_states, len(axes_ps_flat)):
                axes_ps_flat[idx].set_visible(False)
            plt.tight_layout()
            plt.show()

            # === Plot 4: Post-selected logical ===
            fig_psl, axes_psl = plt.subplots(
                nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
            axes_psl_flat = axes_psl.flatten()

            for state_idx, prepared_state in enumerate(state_list):
                ax = axes_psl_flat[state_idx]

                # Background: raw logical
                p_0_raw = data.get(f'p_0_{prepared_state}')
                p_1_raw = data.get(f'p_1_{prepared_state}')
                if p_0_raw is not None:
                    ax.plot(times_ms, p_0_raw, '-',
                            color=self.STATE_COLORS['01'],
                            alpha=0.4, linewidth=1, label=r'Raw $p_0$')
                if p_1_raw is not None:
                    ax.plot(times_ms, p_1_raw, '-',
                            color=self.STATE_COLORS['10'],
                            alpha=0.4, linewidth=1, label=r'Raw $p_1$')

                # Post-selected logical
                p_0_ps = data.get(f'p_0_ps_{prepared_state}')
                p_1_ps = data.get(f'p_1_ps_{prepared_state}')
                if p_0_ps is not None:
                    ax.plot(times_ms, p_0_ps, 'o',
                            color=self.STATE_COLORS['01'], markersize=4,
                            label=r'PS $p_0$')
                if p_1_ps is not None:
                    ax.plot(times_ms, p_1_ps, 's',
                            color=self.STATE_COLORS['10'], markersize=4,
                            label=r'PS $p_1$')

                # Fit post-selected logical
                fit_text = []
                if fit and len(times_ms) >= 3:
                    for label, y_ps, color in [
                            ('p_0', p_0_ps, self.STATE_COLORS['01']),
                            ('p_1', p_1_ps, self.STATE_COLORS['10'])]:
                        if y_ps is None:
                            continue
                        try:
                            p0 = [0.5, y_ps[0] - 0.5,
                                  (times_ms[-1] - times_ms[0]) / 3]
                            bounds = ([0, -2, 0], [1, 2, np.inf])
                            pOpt, pCov = curve_fit(
                                expfunc1, times_ms, y_ps, p0=p0,
                                bounds=bounds, maxfev=200000)
                            T = pOpt[2]
                            T_err = (np.sqrt(pCov[2, 2])
                                     if pCov[2, 2] < np.inf else 0)
                            t_fit = np.linspace(
                                times_ms[0], times_ms[-1], 200)
                            ax.plot(t_fit, expfunc1(t_fit, *pOpt), '--',
                                    color=color, linewidth=2)
                            fit_text.append(
                                r'$T_{%s}=%.3g \pm %.3g$ ms'
                                % (label, T, T_err))
                        except Exception:
                            pass

                title = (r'PS Logical: $|%s\rangle$ [%s]'
                         % (prepared_state, stor_pair))
                if fit_text:
                    title += '\n' + ', '.join(fit_text)
                ax.set_title(title)
                ax.set_xlabel(x_label)
                ax.set_ylabel('Logical Population')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])

            for idx in range(n_states, len(axes_psl_flat)):
                axes_psl_flat[idx].set_visible(False)
            plt.tight_layout()
            plt.show()

        # === Plot 5: JP even fraction ===
        if n_checks > 0:
            fig_jp, axes_jp = plt.subplots(nrows, ncols,
                                            figsize=(8 * ncols, 5 * nrows),
                                            squeeze=False)
            axes_jp_flat = axes_jp.flatten()

            for state_idx, prepared_state in enumerate(state_list):
                ax = axes_jp_flat[state_idx]
                jp_ef = data.get(f'jp_even_frac_{prepared_state}')
                if jp_ef is None:
                    continue

                for k in range(n_checks):
                    col = jp_ef[:, k]
                    valid_mask = ~np.isnan(col)
                    if np.any(valid_mask):
                        ax.plot(times_ms[valid_mask], col[valid_mask], 'o-',
                                label=f'Check {k+1}', markersize=4)

                ax.set_xlabel(x_label)
                ax.set_ylabel('JP Even Fraction')
                ax.set_title(r'JP Results: $|%s\rangle$ [%s]'
                             % (prepared_state, stor_pair))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])

            for idx in range(n_states, len(axes_jp_flat)):
                axes_jp_flat[idx].set_visible(False)

            plt.tight_layout()
            plt.show()

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
                    print(f"Warning: Skipping object-dtype field "
                          f"'{key}' for HDF5 save")
                    continue
            else:
                save_data[key] = value

        super().save_data(data=save_data)
        return self.fname

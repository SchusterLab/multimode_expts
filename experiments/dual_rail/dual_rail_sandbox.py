'''
Dual Rail Sandbox Experiment
Prepares dual rail qubit states (00, 10, 01, 11), optionally waits and measures
joint parity, then performs final dual rail measurement.

State encoding:
- First digit: photon in storage_1 (0 or 1)
- Second digit: photon in storage_2 (0 or 1)

Pulse sequence:
1. (Optional) Active reset
2. State preparation (if state_start != '00')
3. Repeat loop N times:
   - Wait for wait_time (if > 0)
   - (Optional) Joint parity measurement + conditional reset
4. Final dual rail measurement via measure_dual_rail()

Seb 02/2026
'''

import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from fitting.fitting import expfunc, fitexp
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import *


class DualRailSandboxProgram(MMAveragerProgram):
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

        # Build storage mode names
        man = cfg.expt.manipulate
        s1 = cfg.expt.storage_1
        s2 = cfg.expt.storage_2
        self.stor_name_1 = f'M{man}-S{s1}'
        self.stor_name_2 = f'M{man}-S{s2}'
        self.stor_pair_name = f'S{s1}-S{s2}'  # For joint parity lookup

        # Build state preparation pulse
        state_start = cfg.expt.get('state_start', '00')
        self.state_start = state_start

        state_prep_seq = self.prep_dual_rail_state(state_start, self.stor_name_1, self.stor_name_2)
        if state_prep_seq is not None:
            creator = self.get_prepulse_creator(state_prep_seq)
            self.state_prep_pulse = creator.pulse.tolist()
            # print(f"State preparation sequence for state_start='{state_start}': {state_prep_seq}")
        else:
            self.state_prep_pulse = None

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        # print(f"\n--- Dual Rail Sandbox Program: state_start='{self.state_start}' ---")

        # 1. Phase reset
        self.reset_and_sync()

        # 2. Active reset (if enabled)
        if cfg.expt.get('active_reset', False):
            # print("Performing active reset at start")
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # 3. State preparation
        if self.state_prep_pulse is not None:
            # print(f"Preparing dual rail state |{self.state_start}>")
            self.custom_pulse(cfg, self.state_prep_pulse, prefix='state_prep_')

        # 4. Repeat loop: (wait + optional joint_parity)
        repeat_count = int(cfg.expt.get('repeat_count', 1))
        wait_time = cfg.expt.get('wait_time', 0) or 0

        for rep_idx in range(repeat_count):
            # print(f"--- Repeat loop {rep_idx + 1}/{repeat_count} ---")
            # Wait (skip if wait_time <= 0)
            if wait_time > 0:
                # print(f"Waiting for {wait_time} us")
                self.sync_all(self.us2cycles(wait_time))

            # Joint parity measurement (if enabled)
            if cfg.expt.get('parity_flag', False):
                # print("Performing joint parity measurement")
                self.joint_parity_active_reset(
                    stor_pair_name=self.stor_pair_name,
                    name=f'jp_rep{rep_idx}',
                    register_label=f'jp_label_{rep_idx}',
                    second_phase=0,
                    fast=cfg.expt.get('parity_fast', False)
                )

        # 5. Final dual rail measurement
        # print("Performing final dual rail measurement")
        self.measure_dual_rail(
            storage_idx=(cfg.expt.storage_1, cfg.expt.storage_2),
            measure_parity=True,
            reset_before=cfg.expt.get('reset_before_dual_rail', False),
            reset_after=cfg.expt.get('reset_after_dual_rail', False),
            man_idx=cfg.expt.manipulate, 
            final_sync=True)


    def collect_shots(self):
        cfg = self.cfg
        read_num = self._calculate_read_num()

        shots_i0 = self.di_buf[0].reshape((1, read_num * self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape((1, read_num * self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]

        return shots_i0, shots_q0, read_num

    def _calculate_read_num(self):
        """Calculate total number of readouts per rep"""
        cfg = self.cfg
        read_num = 0

        # Active reset adds measurements based on active_reset configuration
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        # Joint parity measurements during repeat loop
        if cfg.expt.get('parity_flag', False):
            read_num += int(cfg.expt.get('repeat_count', 1))

        # measure_dual_rail measurements
        # Base: 2 parity measurements (one per storage)
        # + 1 if reset_before (from active_reset call inside measure_dual_rail)
        # + 1 if reset_after
        read_num += 2
        if cfg.expt.get('reset_before_dual_rail', False):
            read_num += 1
        if cfg.expt.get('reset_after_dual_rail', False):
            read_num += 1

        return read_num


class DualRailSandboxExperiment(Experiment):
    """
    Dual Rail Sandbox Experiment

    Prepares dual rail states, optionally waits and measures joint parity,
    then performs final dual rail measurement.

    Experimental Config:
    expt = dict(
        qubits: [0],
        reps: number of averages per state,
        rounds: number of rounds (default 1),
        storage_1: first storage mode number (e.g., 1 for S1),
        storage_2: second storage mode number (e.g., 2 for S2),
        manipulate: manipulate mode number (e.g., 1 for M1),
        state_start: str or list of str, initial state(s): '00', '10', '01', '11',
        wait_time: wait time in us (0 = skip wait),
        repeat_count: number of (wait + joint_parity) loops,
        parity_flag: if True, measure joint parity during repeat loop,
        parity_fast: if True, use fast multiphoton hpi pulses,
        active_reset: if True, perform active reset at start,
        reset_before_dual_rail: if True, reset before dual rail measurement,
        reset_after_dual_rail: if True, reset after dual rail measurement,
    )
    """

    # Color mapping for each state
    STATE_COLORS = {
        '00': 'tab:blue',
        '10': 'tab:orange',
        '01': 'tab:green',
        '11': 'tab:red',
    }

    def __init__(self, soccfg=None, path='', prefix='DualRailSandbox', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        # Set default values
        if 'parity_flag' not in self.cfg.expt:
            self.cfg.expt.parity_flag = False
        if 'parity_fast' not in self.cfg.expt:
            self.cfg.expt.parity_fast = False
        if 'wait_time' not in self.cfg.expt:
            self.cfg.expt.wait_time = 0
        if 'repeat_count' not in self.cfg.expt:
            self.cfg.expt.repeat_count = 1

        # Handle state_start as list or single string
        state_list = self.cfg.expt.get('state_start', '00')
        if isinstance(state_list, str):
            state_list = [state_list]

        # Handle repeat_count as int or list
        repeat_list = self.cfg.expt.get('repeat_count', 1)
        if isinstance(repeat_list, (int, float)):
            repeat_list = [int(repeat_list)]
        repeat_list = list(repeat_list)

        # Handle wait_time as number or list
        wait_list = self.cfg.expt.get('wait_time', 0)
        if isinstance(wait_list, (int, float)):
            wait_list = [wait_list]
        wait_list = list(wait_list)

        data = {
            'states': state_list,
            'repeat_counts': repeat_list,
            'wait_times': wait_list,
            'parity_flag': self.cfg.expt.get('parity_flag', False),
            'threshold': self.cfg.device.readout.threshold[0],
        }

        # Combinatorial loop over states, repeat_counts, and wait_times
        total_combos = len(state_list) * len(repeat_list) * len(wait_list)
        combo_idx = 0

        for state in state_list:
            for repeat in repeat_list:
                for wait in wait_list:
                    combo_idx += 1
                    print(f"\n=== [{combo_idx}/{total_combos}] state='{state}', repeat={repeat}, wait={wait} ===")

                    # Update config for this combination
                    self.cfg.expt.state_start = state
                    self.cfg.expt.repeat_count = repeat
                    self.cfg.expt.wait_time = wait

                    prog = DualRailSandboxProgram(soccfg=self.soccfg, cfg=self.cfg)
                    read_num = prog._calculate_read_num()

                    # MMAveragerProgram.acquire returns (avgi, avgq)
                    avgi, avgq = prog.acquire(
                        self.im[self.cfg.aliases.soc],
                        threshold=None,
                        load_pulses=True,
                        progress=progress,
                        readouts_per_experiment=read_num
                    )

                    # Collect single shot data
                    i0, q0, _ = prog.collect_shots()

                    # Store with extended key
                    key = f'{state}_r{repeat}_w{wait}'
                    data[f'i0_{key}'] = i0
                    data[f'q0_{key}'] = q0
                    data[f'avgi_{key}'] = avgi[0][0]
                    data[f'avgq_{key}'] = avgq[0][0]
                    data[f'read_num_{key}'] = read_num

        self.data = data
        return data

    def analyze(self, data=None, post_select=True):
        """
        Analyze dual rail data: bin into populations and optionally post-select.

        Args:
            data: Data dict (default: self.data)
            post_select: If True, only keep trajectories where joint parity
                         measurements returned g (stayed in 10/01 subspace)
        """
        if data is None:
            data = self.data

        state_list = data.get('states', ['00'])
        repeat_list = data.get('repeat_counts', [1])
        wait_list = data.get('wait_times', [0])
        threshold = data.get('threshold')
        parity_flag = data.get('parity_flag', False)
        reps = self.cfg.expt.reps

        # Iterate over all combinations
        for state in state_list:
            for repeat in repeat_list:
                for wait in wait_list:
                    key = f'{state}_r{repeat}_w{wait}'
                    read_num = data.get(f'read_num_{key}')

                    if read_num is None:
                        print(f"Warning: read_num not found for {key}, skipping")
                        continue

                    i0 = data.get(f'i0_{key}')
                    if i0 is None:
                        print(f"Warning: i0 data not found for {key}, skipping")
                        continue

                    i0_reshaped = i0.reshape(reps, read_num)

                    # Update config for correct shot indices
                    self.cfg.expt.repeat_count = repeat
                    self.cfg.expt.wait_time = wait
                    indices = self._get_shot_indices()

                    # Pre-selection mask: discard shots where active_reset pre_selection failed
                    if 'ar_pre_selection' in indices:
                        pre_sel_idx = indices['ar_pre_selection']
                        pre_sel_mask = i0_reshaped[:, pre_sel_idx] < threshold
                        i0_reshaped = i0_reshaped[pre_sel_mask]
                        data[f'pre_select_count_{key}'] = np.sum(pre_sel_mask)

                    # Post-selection mask: keep trajectories where all JP shots are g (< threshold)
                    if post_select and parity_flag and 'jp' in indices:
                        jp_indices = indices['jp']
                        jp_shots = i0_reshaped[:, jp_indices]
                        # g = below threshold (stayed in 10/01 subspace)
                        post_select_mask = np.all(jp_shots < threshold, axis=1)
                        i0_filtered = i0_reshaped[post_select_mask]
                        data[f'post_select_count_{key}'] = np.sum(post_select_mask)
                    else:
                        i0_filtered = i0_reshaped
                        data[f'post_select_count_{key}'] = len(i0_reshaped)

                    # Bin final dual rail measurements
                    pops, counts = self._bin_dual_rail_shots(i0_filtered, indices, threshold)
                    data[f'pop_{key}'] = pops
                    data[f'counts_{key}'] = counts

        return data

    def _get_shot_indices(self):
        """Return dict mapping measurement type to column indices"""
        cfg = self.cfg
        idx = 0
        indices = {}

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            ar_read_num = MMAveragerProgram.active_reset_read_num(**params)
            if params.get('pre_selection_reset', False):
                indices['ar_pre_selection'] = idx + ar_read_num - 1
            idx += ar_read_num

        if cfg.expt.get('parity_flag', False):
            repeat_count = int(cfg.expt.get('repeat_count', 1))
            indices['jp'] = list(range(idx, idx + repeat_count))
            idx += repeat_count

        if cfg.expt.get('reset_before_dual_rail', False):
            indices['dr_reset_before'] = idx
            idx += 1

        indices['dr_stor1'] = idx
        indices['dr_stor2'] = idx + 1
        idx += 2

        if cfg.expt.get('reset_after_dual_rail', False):
            indices['dr_reset_after'] = idx

        return indices

    def _bin_dual_rail_shots(self, i0_filtered, indices, threshold):
        """
        Bin dual rail shots into 00, 10, 01, 11 populations.
        """
        n_shots = len(i0_filtered)
        if n_shots == 0:
            return {'00': 0, '01': 0, '10': 0, '11': 0}, {'00': 0, '01': 0, '10': 0, '11': 0}

        stor1_shots = i0_filtered[:, indices['dr_stor1']]
        stor2_shots = i0_filtered[:, indices['dr_stor2']]  

        # Threshold: > threshold = qubit in |e> = odd parity = 1 photon
        stor1_state = (stor1_shots > threshold).astype(int)
        stor2_state = (stor2_shots > threshold).astype(int)  

        counts = {'00': 0, '01': 0, '10': 0, '11': 0}
        for s1, s2 in zip(stor1_state, stor2_state):  
            key = f'{s1}{s2}'
            counts[key] += 1

        pops = {k: v / n_shots for k, v in counts.items()}
        return pops, counts

    def display(self, data=None, show_iq=False, show_histograms=True, log_scale=False, **kwargs):
        """
        Display dual rail results.

        Args:
            data: Data dict (default: self.data)
            show_iq: If True, also show I/Q scatter plots with threshold lines
            show_histograms: If True (default), show bar chart grid for each combo

        Creates:
            - Grid of bar charts (+ IQ plots if show_iq) for each combo (if show_histograms)
            - Summary plot showing population decay vs total idle time
            - Logical subspace population plot (p_0 = 01/(10+01), p_1 = 10/(10+01))
        """
        if data is None:
            data = self.data

        state_list = data.get('states', ['00'])
        repeat_list = data.get('repeat_counts', [1])
        wait_list = data.get('wait_times', [0])

        # Convert byte strings to regular strings if needed (for loaded data)
        if isinstance(state_list, np.ndarray):
            state_list = [s.decode() if isinstance(s, bytes) else s for s in state_list]
        elif isinstance(state_list, list):
            state_list = [s.decode() if isinstance(s, bytes) else s for s in state_list]

        # Convert to lists if needed (for loaded data)
        if isinstance(repeat_list, np.ndarray):
            repeat_list = list(repeat_list)
        if isinstance(wait_list, np.ndarray):
            wait_list = list(wait_list)

        threshold = data.get('threshold', 0)
        bar_labels = ['00', '10', '01', '11']
        total_reps = self.cfg.expt.reps

        # Build list of all combinations
        combos = []
        for state in state_list:
            for repeat in repeat_list:
                for wait in wait_list:
                    combos.append((state, repeat, wait))

        fig = None
        if show_histograms:
            # Create subplot grid for bar charts (+ IQ if requested)
            n_combos = len(combos)
            ncols = min(4, n_combos)
            nrows = (n_combos + ncols - 1) // ncols
            if show_iq:
                nrows *= 2

            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
            axes = axes.flatten()

            for idx, (prepared_state, repeat, wait) in enumerate(combos):
                key = f'{prepared_state}_r{repeat}_w{wait}'

                # Population bar chart
                ax_idx = idx * 2 if show_iq else idx
                ax = axes[ax_idx]

                pops_raw = data.get(f'pop_{key}', {})
                post_count = data.get(f'post_select_count_{key}', total_reps)

                # Handle both dict format (in-memory) and array format (loaded from HDF5)
                if isinstance(pops_raw, dict):
                    pop_values = [pops_raw.get(label, 0) for label in bar_labels]
                elif isinstance(pops_raw, np.ndarray):
                    pop_values = list(pops_raw)
                else:
                    pop_values = [0, 0, 0, 0]
                bar_colors = [self.STATE_COLORS[label] for label in bar_labels]

                bars = ax.bar(bar_labels, pop_values, color=bar_colors, alpha=0.7)
                ax.set_ylabel('Population')
                ax.set_xlabel('Measured State')
                total_time = repeat * wait
                total_time_ms = total_time / 1000  # convert us to ms
                ax.set_title(f'|{prepared_state}> r={repeat} w={wait} (t={total_time_ms:.3g}ms)\n'
                            f'post-sel: {post_count}/{total_reps}')
                ax.set_ylim([0, 1])
                if log_scale:
                    ax.set_yscale('log')
                    ax.set_ylim([5e-4, 1.5])  # Set lower limit for log scale

                # Add probability counts on top of each bar
                for bar, pop in zip(bars, pop_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{pop:.3f}', ha='center', va='bottom', fontsize=9)

                # Highlight expected state
                if prepared_state in bar_labels:
                    expected_idx = bar_labels.index(prepared_state)
                    bars[expected_idx].set_edgecolor(color='black')
                    bars[expected_idx].set_linewidth(1)

                # I/Q scatter plot (if show_iq)
                if show_iq:
                    ax_iq = axes[ax_idx + 1]
                    i0 = data.get(f'i0_{key}')
                    q0 = data.get(f'q0_{key}')
                    read_num = data.get(f'read_num_{key}', 1)

                    if i0 is not None and q0 is not None:
                        reps = self.cfg.expt.reps
                        i0_reshaped = i0.reshape(reps, read_num)
                        q0_reshaped = q0.reshape(reps, read_num)

                        # Update config for correct shot indices
                        self.cfg.expt.repeat_count = repeat
                        self.cfg.expt.wait_time = wait
                        indices = self._get_shot_indices()

                        # Plot storage 1 and storage 2 final measurements
                        i_stor1 = i0_reshaped[:, indices['dr_stor1']]
                        q_stor1 = q0_reshaped[:, indices['dr_stor1']]
                        i_stor2 = i0_reshaped[:, indices['dr_stor2']]
                        q_stor2 = q0_reshaped[:, indices['dr_stor2']]

                        ax_iq.scatter(i_stor1, q_stor1, alpha=0.3, s=10, label='Stor1')
                        ax_iq.scatter(i_stor2, q_stor2, alpha=0.3, s=10, label='Stor2')

                        # Plot threshold line
                        ax_iq.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                                      label=f'Thresh={threshold:.2f}')

                        ax_iq.set_xlabel('I [ADC]')
                        ax_iq.set_ylabel('Q [ADC]')
                        ax_iq.set_title(f'I/Q: |{prepared_state}> r={repeat} w={wait}')
                        ax_iq.legend(loc='upper right', fontsize=7)
                        ax_iq.grid(True, alpha=0.3)

            # Hide unused subplots
            total_used = n_combos * (2 if show_iq else 1)
            for idx in range(total_used, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.show()

        # Summary plot: population vs total idle time
        # Only show if there are multiple (repeat, wait) combinations
        if len(repeat_list) > 1 or len(wait_list) > 1:
            # One subplot per prepared state, showing all 4 measured populations
            n_states = len(state_list)
            ncols_sum = min(2, n_states)
            nrows_sum = (n_states + ncols_sum - 1) // ncols_sum
            fig_summary, axes_sum = plt.subplots(nrows_sum, ncols_sum,
                                                  figsize=(8*ncols_sum, 5*nrows_sum),
                                                  squeeze=False)
            axes_sum = axes_sum.flatten()

            for state_idx, prepared_state in enumerate(state_list):
                ax = axes_sum[state_idx]

                # For each measured state (00, 10, 01, 11), collect data points
                for measured_state in bar_labels:
                    data_points = []  # (total_time_ms, population)
                    for repeat in repeat_list:
                        for wait in wait_list:
                            total_time_ms = (repeat * wait) / 1000  # convert us to ms
                            key = f'{prepared_state}_r{repeat}_w{wait}'
                            pops_raw = data.get(f'pop_{key}', {})

                            # Get population of this measured state
                            if isinstance(pops_raw, dict):
                                pop = pops_raw.get(measured_state, 0)
                            elif isinstance(pops_raw, np.ndarray):
                                # Array order: [pop_00, pop_10, pop_01, pop_11]
                                meas_idx = bar_labels.index(measured_state)
                                pop = pops_raw[meas_idx]
                            else:
                                pop = 0
                            data_points.append((total_time_ms, pop))

                    # Sort by time for clean line plot
                    data_points.sort(key=lambda x: x[0])
                    if data_points:
                        times, pops = zip(*data_points)
                        ax.plot(times, pops, 'o-',
                               label=r'$|%s\rangle$' % measured_state,
                               color=self.STATE_COLORS[measured_state],
                               markersize=6)

                ax.set_xlabel('Total idle time (ms)')
                ax.set_ylabel('Population')
                ax.set_title(r'Prepared: $|%s\rangle$' % prepared_state)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])

            # Hide unused subplots
            for idx in range(n_states, len(axes_sum)):
                axes_sum[idx].set_visible(False)

            plt.tight_layout()
            plt.show()

            # Logical subspace population plot
            # p_0 = pop_01 / (pop_10 + pop_01), p_1 = pop_10 / (pop_10 + pop_01)
            fig_logical, axes_log = plt.subplots(nrows_sum, ncols_sum,
                                                  figsize=(8*ncols_sum, 5*nrows_sum),
                                                  squeeze=False)
            axes_log = axes_log.flatten()

            for state_idx, prepared_state in enumerate(state_list):
                ax = axes_log[state_idx]

                # Collect logical population data points
                p0_data = []  # (total_time_ms, p_0)
                p1_data = []  # (total_time_ms, p_1)

                for repeat in repeat_list:
                    for wait in wait_list:
                        total_time_ms = (repeat * wait) / 1000  # convert us to ms
                        key = f'{prepared_state}_r{repeat}_w{wait}'
                        pops_raw = data.get(f'pop_{key}', {})

                        # Get pop_10 and pop_01
                        if isinstance(pops_raw, dict):
                            pop_10 = pops_raw.get('10', 0)
                            pop_01 = pops_raw.get('01', 0)
                        elif isinstance(pops_raw, np.ndarray):
                            # Array order: [pop_00, pop_10, pop_01, pop_11]
                            pop_10 = pops_raw[1]
                            pop_01 = pops_raw[2]
                        else:
                            pop_10 = 0
                            pop_01 = 0

                        # Compute logical subspace populations
                        logical_total = pop_10 + pop_01
                        if logical_total > 0:
                            p_0 = pop_01 / logical_total  # |01> is logical |0>
                            p_1 = pop_10 / logical_total  # |10> is logical |1>
                        else:
                            p_0 = 0
                            p_1 = 0

                        p0_data.append((total_time_ms, p_0))
                        p1_data.append((total_time_ms, p_1))

                # Sort by time
                p0_data.sort(key=lambda x: x[0])
                p1_data.sort(key=lambda x: x[0])

                fit_text = []

                if p0_data:
                    times, p0_vals = zip(*p0_data)
                    times = np.array(times)
                    p0_vals = np.array(p0_vals)
                    ax.plot(times, p0_vals, 'o', label=r'$p_0$',
                           color=self.STATE_COLORS['01'], markersize=6)

                    # Exponential fit for p0 (times in ms)
                    if len(times) >= 3 and np.max(times) > 0:
                        try:
                            pOpt_p0, pCov_p0 = fitexp(times, p0_vals)
                            T_p0 = pOpt_p0[3]  # decay parameter (lifetime in ms)
                            T_p0_err = np.sqrt(pCov_p0[3, 3]) if pCov_p0[3, 3] < np.inf else 0
                            # Plot fit curve
                            t_fit = np.linspace(np.min(times), np.max(times), 100)
                            ax.plot(t_fit, expfunc(t_fit, *pOpt_p0), '--',
                                   color=self.STATE_COLORS['01'], linewidth=2)
                            fit_text.append(r'$T_{p_0}=%.3g \pm %.3g$ ms' % (T_p0, T_p0_err))
                        except Exception as e:
                            print(f"p0 fit failed for {prepared_state}: {e}")

                if p1_data:
                    times, p1_vals = zip(*p1_data)
                    times = np.array(times)
                    p1_vals = np.array(p1_vals)
                    ax.plot(times, p1_vals, 's', label=r'$p_1$',
                           color=self.STATE_COLORS['10'], markersize=6)

                    # Exponential fit for p1 (times in ms)
                    if len(times) >= 3 and np.max(times) > 0:
                        try:
                            pOpt_p1, pCov_p1 = fitexp(times, p1_vals)
                            print(f"p1 fit params for {prepared_state}: {pOpt_p1}")
                            print(f"p1 fit cov for {prepared_state}: {pCov_p1}")
                            T_p1 = pOpt_p1[3]  # decay parameter (lifetime in ms)
                            T_p1_err = np.sqrt(pCov_p1[3, 3]) if pCov_p1[3, 3] < np.inf else 0
                            # Plot fit curve
                            t_fit = np.linspace(np.min(times), np.max(times), 100)
                            ax.plot(t_fit, expfunc(t_fit, *pOpt_p1), '--',
                                   color=self.STATE_COLORS['10'], linewidth=2)
                            fit_text.append(r'$T_{p_1}=%.3g \pm %.3g$ ms' % (T_p1, T_p1_err))
                        except Exception as e:
                            print(f"p1 fit failed for {prepared_state}: {e}")

                ax.set_xlabel('Total idle time (ms)')
                ax.set_ylabel('Logical Population')
                title = r'Prepared: $|%s\rangle$ (Logical Subspace)' % prepared_state
                if fit_text:
                    title += '\n' + ', '.join(fit_text)
                ax.set_title(title)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])

            # Hide unused subplots
            for idx in range(n_states, len(axes_log)):
                axes_log[idx].set_visible(False)

            plt.tight_layout()
            plt.show()

            return fig, fig_summary, fig_logical

        return fig

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data

        # Create a copy to avoid modifying original data
        save_data = {}

        for key, value in data.items():
            # Convert string list 'states' to bytes for HDF5 compatibility
            if key == 'states':
                if isinstance(value, list):
                    save_data[key] = np.array(value, dtype='S')
                else:
                    save_data[key] = value
            # Convert numeric lists to numpy arrays
            elif key in ('repeat_counts', 'wait_times'):
                if isinstance(value, list):
                    save_data[key] = np.array(value)
                else:
                    save_data[key] = value
            # Convert population dicts to arrays (pop_00, pop_10, etc.)
            elif key.startswith('pop_') and isinstance(value, dict):
                # Store as array: [pop_00, pop_10, pop_01, pop_11]
                save_data[key] = np.array([value.get('00', 0), value.get('10', 0),
                                           value.get('01', 0), value.get('11', 0)])
            # Convert counts dicts to arrays
            elif key.startswith('counts_') and isinstance(value, dict):
                save_data[key] = np.array([value.get('00', 0), value.get('10', 0),
                                           value.get('01', 0), value.get('11', 0)])
            # Skip any other dict or object types that HDF5 can't handle
            elif isinstance(value, dict):
                print(f"Warning: Skipping dict field '{key}' for HDF5 save")
                continue
            else:
                save_data[key] = value

        super().save_data(data=save_data)
        return self.fname

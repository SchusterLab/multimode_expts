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
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import *


class DualRailSandboxProgram(MMAveragerProgram):
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
            print(f"State preparation sequence for state_start='{state_start}': {state_prep_seq}")
        else:
            self.state_prep_pulse = None

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        print(f"\n--- Dual Rail Sandbox Program: state_start='{self.state_start}' ---")

        # 1. Phase reset
        self.reset_and_sync()

        # 2. Active reset (if enabled)
        if cfg.expt.get('active_reset', False):
            print("Performing active reset at start")
            self.active_reset(man_reset=False, storage_reset=False,
                              ef_reset=False, pre_selection_reset=False)

        # 3. State preparation
        if self.state_prep_pulse is not None:
            print(f"Preparing dual rail state |{self.state_start}>")
            self.custom_pulse(cfg, self.state_prep_pulse, prefix='state_prep_')

        # 4. Repeat loop: (wait + optional joint_parity)
        repeat_count = int(cfg.expt.get('repeat_count', 1))
        wait_time = cfg.expt.get('wait_time', 0) or 0

        for rep_idx in range(repeat_count):
            print(f"--- Repeat loop {rep_idx + 1}/{repeat_count} ---")
            # Wait (skip if wait_time <= 0)
            if wait_time > 0:
                print(f"Waiting for {wait_time} us")
                self.sync_all(self.us2cycles(wait_time))

            # Joint parity measurement (if enabled)
            if cfg.expt.get('parity_flag', False):
                print("Performing joint parity measurement")
                self.joint_parity_active_reset(
                    stor_pair_name=self.stor_pair_name,
                    name=f'jp_rep{rep_idx}',
                    register_label=f'jp_label_{rep_idx}',
                    second_phase=0,
                    fast=cfg.expt.get('parity_fast', False)
                )

        # 5. Final dual rail measurement
        print("Performing final dual rail measurement")
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

        # Active reset adds 1 measurement (ge level only since ef_reset=False)
        if cfg.expt.get('active_reset', False):
            read_num += 1

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

        # Create a temporary program to get read_num
        self.cfg.expt.state_start = state_list[0]
        temp_prog = DualRailSandboxProgram(soccfg=self.soccfg, cfg=self.cfg)
        read_num = temp_prog._calculate_read_num()

        data = {
            'states': state_list,
            'wait_time': self.cfg.expt.get('wait_time', 0),
            'repeat_count': self.cfg.expt.get('repeat_count', 1),
            'parity_flag': self.cfg.expt.get('parity_flag', False),
            'threshold': self.cfg.device.readout.threshold[0],
            'read_num': read_num,
        }

        for state in tqdm(state_list, desc="States", disable=not progress):
            print(f"\n=== Running state_start='{state}' ===")

            # Update config for this state
            self.cfg.expt.state_start = state

            prog = DualRailSandboxProgram(soccfg=self.soccfg, cfg=self.cfg)

            # MMAveragerProgram.acquire returns (avgi, avgq), not (x_pts, avgi, avgq)
            avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num
            )

            # Collect single shot data
            i0, q0, _ = prog.collect_shots()

            data[f'i0_{state}'] = i0
            data[f'q0_{state}'] = q0
            data[f'avgi_{state}'] = avgi[0][0]
            data[f'avgq_{state}'] = avgq[0][0]

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
        threshold = data.get('threshold')
        read_num = data.get('read_num')
        parity_flag = data.get('parity_flag', False)
        reps = self.cfg.expt.reps

        indices = self._get_shot_indices()

        for state in state_list:
            i0 = data[f'i0_{state}']
            i0_reshaped = i0.reshape(reps, read_num)

            # Post-selection mask: keep trajectories where all JP shots are g (< threshold)
            if post_select and parity_flag:
                jp_indices = indices['jp']
                jp_shots = i0_reshaped[:, jp_indices]
                # g = below threshold (stayed in 10/01 subspace)
                post_select_mask = np.all(jp_shots < threshold, axis=1)
                i0_filtered = i0_reshaped[post_select_mask]
                data[f'post_select_count_{state}'] = np.sum(post_select_mask)
            else:
                i0_filtered = i0_reshaped
                data[f'post_select_count_{state}'] = reps

            # Bin final dual rail measurements
            pops, counts = self._bin_dual_rail_shots(i0_filtered, indices, threshold)
            data[f'pop_{state}'] = pops
            data[f'counts_{state}'] = counts

        return data

    def _get_shot_indices(self):
        """Return dict mapping measurement type to column indices"""
        cfg = self.cfg
        idx = 0
        indices = {}

        if cfg.expt.get('active_reset', False):
            indices['ar_ge'] = idx
            idx += 1

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

    def display(self, data=None, show_iq=False, **kwargs):
        """
        Display dual rail results.

        Args:
            data: Data dict (default: self.data)
            show_iq: If True, also show I/Q scatter plots with threshold lines
        """
        if data is None:
            data = self.data

        state_list = data.get('states', ['00'])
        # Convert byte strings to regular strings if needed (for loaded data)
        if isinstance(state_list, np.ndarray):
            state_list = [s.decode() if isinstance(s, bytes) else s for s in state_list]
        elif isinstance(state_list, list):
            state_list = [s.decode() if isinstance(s, bytes) else s for s in state_list]

        threshold = data.get('threshold', 0)

        # Create subplot grid
        n_states = len(state_list)
        ncols = min(2, n_states)
        nrows = (n_states + ncols - 1) // ncols

        # If show_iq, double the rows (population + IQ for each state)
        if show_iq:
            nrows *= 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
        axes = axes.flatten()

        bar_labels = ['00', '10', '01', '11']

        for idx, prepared_state in enumerate(state_list):
            # Population bar chart
            ax_idx = idx * 2 if show_iq else idx
            ax = axes[ax_idx]

            pops_raw = data.get(f'pop_{prepared_state}', {})
            post_count = data.get(f'post_select_count_{prepared_state}', self.cfg.expt.reps)
            total_reps = self.cfg.expt.reps

            # Handle both dict format (in-memory) and array format (loaded from HDF5)
            if isinstance(pops_raw, dict):
                pop_values = [pops_raw.get(label, 0) for label in bar_labels]
            elif isinstance(pops_raw, np.ndarray):
                # Array format: [pop_00, pop_10, pop_01, pop_11]
                pop_values = list(pops_raw)
            else:
                pop_values = [0, 0, 0, 0]
            bar_colors = [self.STATE_COLORS[label] for label in bar_labels]

            bars = ax.bar(bar_labels, pop_values, color=bar_colors)
            ax.set_ylabel('Population')
            ax.set_xlabel('Measured State')
            ax.set_title(f'Prepared: |{prepared_state}> (post-sel: {post_count}/{total_reps})')
            ax.set_ylim([0, 1])

            # Add probability counts on top of each bar
            for bar, pop in zip(bars, pop_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{pop:.3f}', ha='center', va='bottom', fontsize=10)

            # Highlight expected state
            expected_idx = bar_labels.index(prepared_state)
            bars[expected_idx].set_edgecolor('black')
            bars[expected_idx].set_linewidth(2)

            # I/Q scatter plot (if show_iq)
            if show_iq:
                ax_iq = axes[ax_idx + 1]
                i0 = data.get(f'i0_{prepared_state}')
                q0 = data.get(f'q0_{prepared_state}')
                read_num = data.get('read_num', 1)

                if i0 is not None and q0 is not None:
                    reps = self.cfg.expt.reps
                    i0_reshaped = i0.reshape(reps, read_num)
                    q0_reshaped = q0.reshape(reps, read_num)

                    # Get indices for dual rail measurements
                    indices = self._get_shot_indices()

                    # Plot storage 1 and storage 2 final measurements
                    i_stor1 = i0_reshaped[:, indices['dr_stor1']]
                    q_stor1 = q0_reshaped[:, indices['dr_stor1']]
                    i_stor2 = i0_reshaped[:, indices['dr_stor2']]  # COMMENTED OUT FOR DEBUGGING
                    q_stor2 = q0_reshaped[:, indices['dr_stor2']]  # COMMENTED OUT FOR DEBUGGING

                    ax_iq.scatter(i_stor1, q_stor1, alpha=0.3, s=10, label='Stor1')
                    ax_iq.scatter(i_stor2, q_stor2, alpha=0.3, s=10, label='Stor2')  # COMMENTED OUT FOR DEBUGGING

                    # Plot threshold line (vertical at I = threshold)
                    ax_iq.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')

                    # Plot expected g/e localization markers
                    Ig = self.cfg.device.readout.get('Ig', [None])[0]
                    Ie = self.cfg.device.readout.get('Ie', [None])[0]
                    Qg = self.cfg.device.readout.get('Qg', [0])[0]  # Often 0 if not calibrated
                    Qe = self.cfg.device.readout.get('Qe', [0])[0]

                    if Ig is not None:
                        ax_iq.scatter([Ig], [Qg], marker='o', s=200, c='green', edgecolors='black',
                                      linewidths=2, zorder=10, label=f'|g> ({Ig:.2f})')
                    if Ie is not None:
                        ax_iq.scatter([Ie], [Qe], marker='o', s=200, c='orange', edgecolors='black',
                                      linewidths=2, zorder=10, label=f'|e> ({Ie:.2f})')

                    # If f state calibration exists, plot it too
                    If = self.cfg.device.readout.get('If', [None])[0]
                    Qf = self.cfg.device.readout.get('Qf', [0])[0]
                    if If is not None:
                        ax_iq.scatter([If], [Qf], marker='o', s=200, c='red', edgecolors='black',
                                      linewidths=2, zorder=10, label=f'|f> ({If:.2f})')

                    ax_iq.set_xlabel('I [ADC]')
                    ax_iq.set_ylabel('Q [ADC]')
                    ax_iq.set_title(f'I/Q for |{prepared_state}>')
                    ax_iq.legend(loc='upper right', fontsize=8)
                    ax_iq.grid(True, alpha=0.3)

        # Hide unused subplots
        total_used = n_states * (2 if show_iq else 1)
        for idx in range(total_used, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()
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

'''
Flexible Multi-Pair Dual Rail Sandbox Experiment

Supports multiple dual rail pairs sharing one manipulate mode,
logical state preparation (including superpositions), a user-defined
pulse sequence of gate operations / joint parity / waits, and
per-pair final measurement.

Config format:
    expt = dict(
        qubits=[0],
        reps=1000,
        manipulate=1,
        storage_swap=[1, 3, 7],       # S_swap per pair
        storage_parity=[2, 4, 6],     # S_parity per pair
        state_start=['1', '+', '0'],  # logical state per pair
        pulse_sequence=[              # list of (op, arg) tuples
            ('wait', 10),
            ('joint_parity', 0),
            ('X', 1),
            ('X/2', 0),
        ],
        measure_pairs=[0, 1, 2],      # which pairs to measure
        active_reset=True,
        parity_fast=False,
        measure_parity=True,
        reset_before_dual_rail=False,
        reset_after_dual_rail=False,
    )

State encoding per pair (S_swap, S_parity):
    '0'  -> |0_L> = |01> (photon in S_parity)
    '1'  -> |1_L> = |10> (photon in S_swap)
    '+'  -> (|0> + |1>)/sqrt(2)
    '-'  -> (|0> - |1>)/sqrt(2)
    '+i' -> (|0> + i|1>)/sqrt(2)
    '-i' -> (|0> - i|1>)/sqrt(2)
    '00' -> |00> (vacuum, no photons — leakage state)
    '11' -> |11> (one photon in each mode — leakage state)

Seb 02/2026
'''

import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import *


class DualRailSandboxV2Program(MMAveragerProgram):
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

        # --- Validate config ---
        storage_swap = cfg.expt.storage_swap
        storage_parity = cfg.expt.storage_parity
        self.n_pairs = len(storage_swap)
        assert len(storage_parity) == self.n_pairs, \
            f"storage_swap ({len(storage_swap)}) and storage_parity ({len(storage_parity)}) must have same length"

        state_list = cfg.expt.get('state_start', ['0'] * self.n_pairs)
        if isinstance(state_list, str):
            state_list = [state_list]
        assert len(state_list) == self.n_pairs, \
            f"state_start ({len(state_list)}) must match number of pairs ({self.n_pairs})"

        # --- Build storage mode names ---
        self.stor_names_swap = [f'M{man}-S{s}' for s in storage_swap]
        self.stor_names_parity = [f'M{man}-S{s}' for s in storage_parity]

        # --- Phase tracking setup ---
        _ds = cfg.device.storage._ds_storage
        self.pair_names = [_ds.pair_name(s, p)
                           for s, p in zip(storage_swap, storage_parity)]
        self.phase_tracking = cfg.expt.get('phase_tracking', True)

        if self.phase_tracking:
            self.ac_stark_rates = MM_base.load_dr_ac_stark_rates(
                _ds, self.pair_names,
                cfg_override=cfg.expt.get('ac_stark_rates'))
            self.jp_phase_matrix = MM_base.load_dr_jp_phase_matrix(
                _ds, self.pair_names,
                cfg_override=cfg.expt.get('jp_phase_matrix'))
            self.phase_corrections = MM_base.compute_dr_phase_corrections(
                cfg.expt.get('pulse_sequence', []),
                self.n_pairs,
                self.pair_names,
                self.ac_stark_rates,
                self.jp_phase_matrix,
            )

        # --- Compile state prep pulses (one per pair) ---
        self.state_prep_pulses = []
        for i in range(self.n_pairs):
            seq = self.prep_dual_rail_logical_state(
                state_list[i], self.stor_names_swap[i], self.stor_names_parity[i])
            if seq is not None:
                pulse = self.get_prepulse_creator(seq).pulse.tolist()
            else:
                pulse = None
            self.state_prep_pulses.append(pulse)

        # --- Compile pulse sequence operations ---
        self._swap_pulse_cache = {}
        self.compiled_ops = []

        for seq_idx, (op_name, arg) in enumerate(cfg.expt.get('pulse_sequence', [])):
            if op_name == 'wait':
                self.compiled_ops.append({
                    'type': 'wait',
                    'cycles': self.us2cycles(arg),
                })

            elif op_name == 'joint_parity':
                pair_idx = int(arg)
                assert 0 <= pair_idx < self.n_pairs, \
                    f"joint_parity pair_idx {pair_idx} out of range [0, {self.n_pairs})"
                self.compiled_ops.append({
                    'type': 'joint_parity',
                    'pair_idx': pair_idx,
                    'swap_pulse': self._get_swap_pulse(pair_idx),
                    'stor_pair_name': f'M{man}-S{storage_parity[pair_idx]}',
                })

            elif op_name in MM_base.DUAL_RAIL_GATE_MAP:
                pair_idx = int(arg)
                assert 0 <= pair_idx < self.n_pairs, \
                    f"Gate {op_name} pair_idx {pair_idx} out of range [0, {self.n_pairs})"

                # Phase correction from accumulated idle rotation + JP shifts
                phase_offset = 0
                if self.phase_tracking:
                    phase_offset = self.phase_corrections[seq_idx].get(pair_idx, 0)

                gate_seq = self.dual_rail_gate_sequence(
                    op_name, self.stor_names_swap[pair_idx],
                    self.stor_names_parity[pair_idx],
                    phase_offset=phase_offset)
                gate_pulse = self.get_prepulse_creator(gate_seq).pulse.tolist()
                self.compiled_ops.append({
                    'type': 'gate',
                    'pair_idx': pair_idx,
                    'gate_name': op_name,
                    'gate_pulse': gate_pulse,
                    'phase_offset': phase_offset,
                })

            else:
                raise ValueError(
                    f"Unknown sequence operation '{op_name}'. "
                    f"Use: 'wait', 'joint_parity', or one of {list(MM_base.DUAL_RAIL_GATE_MAP.keys())}")

        self.sync_all(200)

    def _get_swap_pulse(self, pair_idx):
        """Get compiled pi swap pulse for a pair (cached)."""
        if pair_idx not in self._swap_pulse_cache:
            seq = [['storage', self.stor_names_swap[pair_idx], 'pi', 0]]
            self._swap_pulse_cache[pair_idx] = \
                self.get_prepulse_creator(seq).pulse.tolist()
        return self._swap_pulse_cache[pair_idx]

    def body(self):
        cfg = AttrDict(self.cfg)

        # 1. Phase reset
        self.reset_and_sync()

        # 2. Active reset (if enabled)
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # 3. State preparation (sequential across pairs)
        for i, pulse in enumerate(self.state_prep_pulses):
            if pulse is not None:
                self.custom_pulse(cfg, pulse, prefix=f'sprep_p{i}_')

        # 4. State-prep post-selection
        if cfg.expt.get('state_prep_postselect', False):
            self.post_selection_measure(
                parity=cfg.expt.get('state_prep_ps_parity', False),
                man_idx=cfg.expt.manipulate,
                parity_fast=cfg.expt.get('parity_fast', False),
                prefix='state_prep_ps'
            )

        # 5. Execute pulse sequence
        jp_counter = 0  # unique index for JP register labels
        for op_idx, op in enumerate(self.compiled_ops):
            if op['type'] == 'wait':
                self.sync_all(op['cycles'])

            elif op['type'] == 'joint_parity':
                self.dual_rail_joint_parity(
                    cfg=cfg,
                    swap_pulse=op['swap_pulse'],
                    stor_pair_name=op['stor_pair_name'],
                    parity_fast=cfg.expt.get('parity_fast', False),
                    op_idx=jp_counter,
                )
                jp_counter += 1

            elif op['type'] == 'gate':
                self.custom_pulse(cfg, op['gate_pulse'], prefix=f'gate{op_idx}_')

        # 6. Final measurement
        measure_pairs = cfg.expt.get('measure_pairs', list(range(self.n_pairs)))
        for m_idx, pair_idx in enumerate(measure_pairs):
            is_last = (m_idx == len(measure_pairs) - 1)
            self.measure_dual_rail(
                storage_idx=(cfg.expt.storage_swap[pair_idx],
                             cfg.expt.storage_parity[pair_idx]),
                measure_parity=cfg.expt.get('measure_parity', True),
                reset_before=(cfg.expt.get('reset_before_dual_rail', False) and m_idx == 0),
                reset_after=(cfg.expt.get('reset_after_dual_rail', False) and is_last),
                man_idx=cfg.expt.manipulate,
                final_sync=is_last,
            )

    def collect_shots(self):
        read_num = self._calculate_read_num()
        shots_i0 = self.di_buf[0].reshape(
            (1, read_num * self.cfg["reps"]), order='F'
        ) / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape(
            (1, read_num * self.cfg["reps"]), order='F'
        ) / self.readout_lengths_adc[0]
        return shots_i0, shots_q0, read_num

    def _calculate_read_num(self):
        """Calculate total number of readouts per rep."""
        cfg = self.cfg
        read_num = 0

        # Active reset readouts
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        # State-prep post-selection
        if cfg.expt.get('state_prep_postselect', False):
            read_num += 1

        # Pulse sequence: joint parity operations
        for op_name, arg in cfg.expt.get('pulse_sequence', []):
            if op_name == 'joint_parity':
                read_num += 1

        # Final measurements: 2 readouts per measured pair
        measure_pairs = cfg.expt.get('measure_pairs',
                                      list(range(len(cfg.expt.storage_swap))))
        read_num += 2 * len(measure_pairs)

        # Optional resets around dual rail measurement
        if cfg.expt.get('reset_before_dual_rail', False):
            read_num += 1
        if cfg.expt.get('reset_after_dual_rail', False):
            read_num += 1

        return read_num


class DualRailSandboxV2Experiment(Experiment):
    """
    Flexible Multi-Pair Dual Rail Sandbox Experiment

    Experimental Config:
    expt = dict(
        qubits: [0],
        reps: number of averages,
        rounds: number of rounds (default 1),
        manipulate: manipulate mode index (e.g. 1),
        storage_swap: list of storage mode indices for S_swap per pair,
        storage_parity: list of storage mode indices for S_parity per pair,
        state_start: list of logical states per pair ('0','1','+','-','+i','-i','00','11'),
        pulse_sequence: list of (op_name, arg) tuples,
        measure_pairs: list of pair indices to measure (default: all),
        active_reset: bool,
        parity_fast: bool,
        measure_parity: bool,
        reset_before_dual_rail: bool,
        reset_after_dual_rail: bool,
    )
    """

    STATE_COLORS = {
        '00': 'tab:blue',
        '10': 'tab:orange',
        '01': 'tab:green',
        '11': 'tab:red',
    }

    def __init__(self, soccfg=None, path='', prefix='DualRailSandboxV2',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        cfg = self.cfg
        n_pairs = len(cfg.expt.storage_swap)

        # Defaults
        if 'parity_fast' not in cfg.expt:
            cfg.expt.parity_fast = False
        if 'pulse_sequence' not in cfg.expt:
            cfg.expt.pulse_sequence = []
        if 'measure_pairs' not in cfg.expt:
            cfg.expt.measure_pairs = list(range(n_pairs))
        if 'state_start' not in cfg.expt:
            cfg.expt.state_start = ['0'] * n_pairs

        state_list = cfg.expt.state_start
        if isinstance(state_list, str):
            state_list = [state_list]
            cfg.expt.state_start = state_list

        measure_pairs = cfg.expt.measure_pairs

        print(f"Dual Rail Sandbox V2: {n_pairs} pairs, "
              f"states={state_list}, "
              f"sequence={len(cfg.expt.pulse_sequence)} ops, "
              f"measuring pairs {measure_pairs}")

        prog = DualRailSandboxV2Program(soccfg=self.soccfg, cfg=cfg)

        # Print phase tracking info
        if prog.phase_tracking:
            print(f"Phase tracking ON: "
                  f"AC Stark rates={prog.ac_stark_rates} MHz")
            for op in prog.compiled_ops:
                if op['type'] == 'gate':
                    print(f"  Gate {op['gate_name']} on pair {op['pair_idx']}: "
                          f"phase_offset={op.get('phase_offset', 0):.2f} deg")
        read_num = prog._calculate_read_num()

        avgi, avgq = prog.acquire(
            self.im[cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            readouts_per_experiment=read_num,
        )

        i0, q0, _ = prog.collect_shots()

        data = {
            'n_pairs': n_pairs,
            'storage_swap': list(cfg.expt.storage_swap),
            'storage_parity': list(cfg.expt.storage_parity),
            'state_start': state_list,
            'pulse_sequence': cfg.expt.pulse_sequence,
            'measure_pairs': measure_pairs,
            'threshold': cfg.device.readout.threshold[0],
            'i0': i0,
            'q0': q0,
            'avgi': avgi[0][0],
            'avgq': avgq[0][0],
            'read_num': read_num,
        }

        self.data = data
        return data

    def _get_shot_indices(self):
        """Return dict mapping measurement type to column indices within a single rep."""
        cfg = self.cfg
        idx = 0
        indices = {}

        # Active reset
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            ar_read_num = MMAveragerProgram.active_reset_read_num(**params)
            if params.get('pre_selection_reset', False):
                indices['ar_pre_selection'] = idx + ar_read_num - 1
            idx += ar_read_num

        # State-prep post-selection
        if cfg.expt.get('state_prep_postselect', False):
            indices['state_prep_ps'] = idx
            idx += 1

        # Pulse sequence: track JP readout indices
        jp_indices = []
        jp_pair_map = []
        for op_name, arg in cfg.expt.get('pulse_sequence', []):
            if op_name == 'joint_parity':
                jp_indices.append(idx)
                jp_pair_map.append(int(arg))
                idx += 1
        if jp_indices:
            indices['jp'] = jp_indices
            indices['jp_pair_map'] = jp_pair_map

        # Final dual-rail measurements
        measure_pairs = cfg.expt.get('measure_pairs',
                                      list(range(len(cfg.expt.storage_swap))))

        if cfg.expt.get('reset_before_dual_rail', False):
            indices['dr_reset_before'] = idx
            idx += 1

        for pair_idx in measure_pairs:
            indices[f'dr_swap_p{pair_idx}'] = idx
            indices[f'dr_parity_p{pair_idx}'] = idx + 1
            idx += 2

        if cfg.expt.get('reset_after_dual_rail', False):
            indices['dr_reset_after'] = idx

        return indices

    def _bin_pair_shots(self, i0_filtered, indices, pair_idx, threshold, measure_parity=True):
        """
        Bin dual rail shots for a single pair into 00, 10, 01, 11 populations.

        Args:
            i0_filtered: array of shape (n_shots, read_num) after filtering
            indices: dict from _get_shot_indices()
            pair_idx: which pair to bin
            threshold: readout threshold
            measure_parity: if False, invert threshold logic (slow pi)
        """
        n_shots = len(i0_filtered)
        if n_shots == 0:
            return {'00': 0, '01': 0, '10': 0, '11': 0}, \
                   {'00': 0, '01': 0, '10': 0, '11': 0}

        swap_shots = i0_filtered[:, indices[f'dr_swap_p{pair_idx}']]
        parity_shots = i0_filtered[:, indices[f'dr_parity_p{pair_idx}']]

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
        return pops, counts

    def analyze(self, data=None, post_select=True):
        """
        Analyze dual rail data: bin into populations per pair.

        Args:
            data: Data dict (default: self.data)
            post_select: If True, only keep trajectories where joint parity
                         measurements returned g (even parity)
        """
        if data is None:
            data = self.data

        threshold = data['threshold']
        reps = self.cfg.expt.reps
        read_num = data['read_num']
        measure_pairs = data['measure_pairs']
        measure_parity = self.cfg.expt.get('measure_parity', True)
        indices = self._get_shot_indices()

        i0 = data['i0']
        i0_reshaped = i0.reshape(reps, read_num)

        # Pre-selection: discard shots where active_reset pre_selection failed
        if 'ar_pre_selection' in indices:
            pre_sel_idx = indices['ar_pre_selection']
            pre_sel_mask = i0_reshaped[:, pre_sel_idx] < threshold
            i0_reshaped = i0_reshaped[pre_sel_mask]
            data['pre_select_count'] = int(np.sum(pre_sel_mask))

        # State-prep post-selection
        if post_select and 'state_prep_ps' in indices:
            sp_ps_idx = indices['state_prep_ps']
            sp_ps_mask = i0_reshaped[:, sp_ps_idx] < threshold
            data['state_prep_ps_count'] = int(np.sum(sp_ps_mask))
            i0_reshaped = i0_reshaped[sp_ps_mask]

        # JP statistics (before post-selection filtering)
        if 'jp' in indices:
            jp_idx = indices['jp']
            jp_shots = i0_reshaped[:, jp_idx]
            jp_even_frac = np.mean(jp_shots < threshold, axis=0)
            data['jp_even_frac'] = jp_even_frac
            data['jp_pair_map'] = indices.get('jp_pair_map', [])

        # JP post-selection
        if post_select and 'jp' in indices:
            jp_shots = i0_reshaped[:, indices['jp']]
            ps_mask = np.all(jp_shots < threshold, axis=1)
            i0_filtered = i0_reshaped[ps_mask]
            data['post_select_count'] = int(np.sum(ps_mask))
        else:
            i0_filtered = i0_reshaped
            data['post_select_count'] = len(i0_reshaped)

        # Bin per pair
        for pair_idx in measure_pairs:
            pops, counts = self._bin_pair_shots(
                i0_filtered, indices, pair_idx, threshold, measure_parity)
            data[f'pop_p{pair_idx}'] = pops
            data[f'counts_p{pair_idx}'] = counts

        return data

    def display(self, data=None, show_iq=False, log_scale=False, **kwargs):
        """
        Display dual rail results.

        Creates bar charts for each measured pair showing 00/10/01/11 populations.
        """
        if data is None:
            data = self.data

        measure_pairs = data['measure_pairs']
        state_list = data['state_start']
        storage_swap = data['storage_swap']
        storage_parity = data['storage_parity']
        threshold = data.get('threshold', 0)
        total_reps = self.cfg.expt.reps
        post_count = data.get('post_select_count', total_reps)

        bar_labels = ['00', '10', '01', '11']
        n_pairs = len(measure_pairs)

        # --- Bar charts: one per measured pair ---
        ncols = min(4, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols
        if show_iq:
            nrows *= 2

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4 * nrows),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for plot_idx, pair_idx in enumerate(measure_pairs):
            ax_idx = plot_idx * 2 if show_iq else plot_idx
            ax = axes_flat[ax_idx]

            pops_raw = data.get(f'pop_p{pair_idx}', {})

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

            pair_label = f'S{storage_swap[pair_idx]}-S{storage_parity[pair_idx]}'
            prep_state = state_list[pair_idx] if pair_idx < len(state_list) else '?'

            ax.set_title(f'Pair {pair_idx} [{pair_label}]\n'
                         f'Prep: |{prep_state}>, PS: {post_count}/{total_reps}')
            ax.set_ylim([0, 1])
            if log_scale:
                ax.set_yscale('log')
                ax.set_ylim([5e-4, 1.5])

            for bar, pop in zip(bars, pop_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{pop:.3f}', ha='center', va='bottom', fontsize=9)

            # I/Q scatter plot
            if show_iq:
                ax_iq = axes_flat[ax_idx + 1]
                i0 = data.get('i0')
                q0 = data.get('q0')
                read_num = data.get('read_num', 1)
                indices = self._get_shot_indices()

                if i0 is not None and q0 is not None:
                    i0_reshaped = i0.reshape(total_reps, read_num)
                    q0_reshaped = q0.reshape(total_reps, read_num)

                    swap_key = f'dr_swap_p{pair_idx}'
                    parity_key = f'dr_parity_p{pair_idx}'

                    if swap_key in indices and parity_key in indices:
                        i_swap = i0_reshaped[:, indices[swap_key]]
                        q_swap = q0_reshaped[:, indices[swap_key]]
                        i_parity = i0_reshaped[:, indices[parity_key]]
                        q_parity = q0_reshaped[:, indices[parity_key]]

                        ax_iq.scatter(i_swap, q_swap, alpha=0.3, s=10,
                                      label=f'S{storage_swap[pair_idx]}')
                        ax_iq.scatter(i_parity, q_parity, alpha=0.3, s=10,
                                      label=f'S{storage_parity[pair_idx]}')
                        ax_iq.axvline(x=threshold, color='red', linestyle='--',
                                      linewidth=2, label=f'Thresh={threshold:.2f}')
                        ax_iq.set_xlabel('I [ADC]')
                        ax_iq.set_ylabel('Q [ADC]')
                        ax_iq.set_title(f'I/Q: Pair {pair_idx} [{pair_label}]')
                        ax_iq.legend(loc='upper right', fontsize=7)
                        ax_iq.grid(True, alpha=0.3)

        # Hide unused subplots
        total_used = n_pairs * (2 if show_iq else 1)
        for idx in range(total_used, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

        # --- Joint parity summary ---
        jp_even_frac = data.get('jp_even_frac')
        jp_pair_map = data.get('jp_pair_map', [])

        if jp_even_frac is not None and len(jp_even_frac) > 0:
            fig_jp, ax_jp = plt.subplots(1, 1, figsize=(6, 4))
            x = np.arange(len(jp_even_frac))
            even_vals = jp_even_frac
            odd_vals = 1 - jp_even_frac

            width = 0.35
            bars_e = ax_jp.bar(x - width / 2, even_vals, width,
                               label='Even (g)', color='tab:blue', alpha=0.7)
            bars_o = ax_jp.bar(x + width / 2, odd_vals, width,
                               label='Odd (e)', color='tab:red', alpha=0.7)

            # Label x-axis with JP operation info
            x_labels = []
            for i, pair_idx in enumerate(jp_pair_map):
                s_s = storage_swap[pair_idx]
                s_p = storage_parity[pair_idx]
                x_labels.append(f'JP{i}\nP{pair_idx} (S{s_s}-S{s_p})')
            ax_jp.set_xticks(x)
            ax_jp.set_xticklabels(x_labels, fontsize=8)

            ax_jp.set_ylabel('Fraction')
            ax_jp.set_title('Joint Parity Results')
            ax_jp.set_ylim([0, 1.15])
            ax_jp.legend()

            for bar, val in zip(list(bars_e) + list(bars_o),
                                list(even_vals) + list(odd_vals)):
                ax_jp.text(bar.get_x() + bar.get_width() / 2.,
                           bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.show()

        return fig

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data

        save_data = {}
        for key, value in data.items():
            # Convert string lists to bytes for HDF5
            if key in ('state_start',):
                if isinstance(value, list):
                    save_data[key] = np.array(value, dtype='S')
                else:
                    save_data[key] = value
            # Convert numeric lists to numpy arrays
            elif key in ('storage_swap', 'storage_parity', 'measure_pairs',
                         'jp_pair_map'):
                if isinstance(value, list):
                    save_data[key] = np.array(value)
                else:
                    save_data[key] = value
            # Convert population dicts to arrays: [pop_00, pop_10, pop_01, pop_11]
            elif key.startswith('pop_') and isinstance(value, dict):
                save_data[key] = np.array([
                    value.get('00', 0), value.get('10', 0),
                    value.get('01', 0), value.get('11', 0)])
            elif key.startswith('counts_') and isinstance(value, dict):
                save_data[key] = np.array([
                    value.get('00', 0), value.get('10', 0),
                    value.get('01', 0), value.get('11', 0)])
            # Skip pulse_sequence (list of tuples, not HDF5 compatible)
            elif key == 'pulse_sequence':
                continue
            # Skip other dicts
            elif isinstance(value, dict):
                print(f"Warning: Skipping dict field '{key}' for HDF5 save")
                continue
            elif value is None:
                continue
            else:
                save_data[key] = value

        super().save_data(data=save_data)
        return self.fname

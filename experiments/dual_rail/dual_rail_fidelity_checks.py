'''
Dual Rail Fidelity vs JP Checks Experiment

Measures how well joint parity (JP) checks preserve quantum information
in the dual rail encoding. Prepares all 6 cardinal Bloch sphere states,
applies N distributed JP checks with an echo, and measures survival
fidelity. Compares against an idle baseline (same time, no JP) to isolate
JP-induced errors. Extracts Pauli error rates from the 6 fidelities.

Pulse sequence per (state, n_checks):
    prep(state) -> [JP]x(N/2) -> echo -> [JP]x(N/2) -> basis_rotation -> measure_Z

Gate convention (Bloch sphere, basis [+Z,+X,+Y,-Z,-X,-Y]):
    - X/2: +Z->-Y, +X fixed  =>  X/2|0> = (|0> - i|1>)/sqrt(2) = |-i>
    - Y/2: +Z->+X, +Y fixed  =>  Y/2|0> = (|0> + |1>)/sqrt(2)  = |+>
    - Rotation axis is 90 deg shifted from the state projected to.
    - Gate decomp (swap_in+hpi+swap_out) shifts effective Clifford:
        Y/2  (hpi@90)  -> Clifford -X/2: maps -Y->+Z  (use for Y-quad meas)
        -X/2 (hpi@180) -> Clifford  Y/2: maps -X->+Z  (use for X-quad meas)
    - For +/- (X quad):   echo = Y, meas = -X/2
    - For +i/-i (Y quad): echo = X, meas = Y/2
    - For 0/1 (Z basis): no meas rotation, echo = Y

Config:
    expt = dict(
        qubits=[0],
        reps=2000,
        rounds=1,
        manipulate=1,
        storage_swap=1,
        storage_parity=3,
        n_checks_list=[0, 2, 4, 6, 8, 10],
        states=['0', '1', '+', '-', '+i', '-i'],
        wait_between_checks=0,
        active_reset=True,
        state_prep_postselect=True,
        parity_fast=False,
        measure_parity=True,
        phase_tracking=True,
        run_idle_baseline=True,
        # TEMPORARY DIAGNOSTIC: override (meas_gate, echo_gate, expected_pop)
        # per state — use to find correct gates for pairs with coupling phase offset
        # state_config_override={'+': ('X/2', 'X', '01'), '-': ('X/2', 'X', '10')},
    )

Seb 02/2026
'''

import matplotlib.pyplot as plt
import numpy as np

from slab import Experiment, AttrDict

from experiments.MM_base import MM_base, MMAveragerProgram
from experiments.dual_rail.dual_rail_sandbox_v2 import (
    DualRailSandboxV2Program,
)


# State -> (measurement_gate, echo_gate, expected_pop_key_with_echo)
# Expected outcome verified via beamsplitter algebra tracing through
# gate decomposition: swap_in + middle_op + swap_out.
STATE_CONFIG = {
    '0':  (None,  'Y', '10'),   # Z basis, Y echo flips |0>->|1>
    '1':  (None,  'Y', '01'),   # Z basis, Y echo flips |1>->|0>
    '+':  ('-X/2', 'Y', '01'),   # X quad, Y echo: |+>->|->  -> -X/2 -> |0_L>  (-X->+Z)
    '-':  ('-X/2', 'Y', '10'),   # X quad, Y echo: |->->|+>  -> -X/2 -> |1_L>  (+X->-Z)
    '+i': ('Y/2',  'X', '01'),   # Y quad, X echo: |+i>->|-i>->  Y/2 -> |0_L>  (-Y->+Z)
    '-i': ('Y/2',  'X', '10'),   # Y quad, X echo: |-i>->|+i>->  Y/2 -> |1_L>  (+Y->-Z)
}

# Colors for plotting
AXIS_COLORS = {
    'Z': 'tab:green',
    'X': 'tab:blue',
    'Y': 'tab:red',
}
STATE_AXIS = {
    '0': 'Z', '1': 'Z',
    '+': 'X', '-': 'X',
    '+i': 'Y', '-i': 'Y',
}


def build_fidelity_pulse_sequence(n_checks, state, wait_between=0,
                                  idle_mode=False, jp_overhead=0,
                                  state_config=None):
    """Build pulse_sequence for DualRailSandboxV2Program.

    Args:
        n_checks: Total number of JP checks (must be even or 0).
        state: One of '0', '1', '+', '-', '+i', '-i'.
        wait_between: Wait time (us) between consecutive JP checks.
        idle_mode: If True, replace JP with equivalent wait time.
        jp_overhead: JP check duration (us), used when idle_mode=True.
        state_config: Optional dict overriding STATE_CONFIG (temporary diagnostic).

    Returns:
        List of (op_name, arg) tuples for sandbox_v2 pulse_sequence config.
    """
    if n_checks > 0:
        assert n_checks % 2 == 0, f"n_checks must be even, got {n_checks}"

    cfg = state_config if state_config is not None else STATE_CONFIG
    meas_gate, echo_gate, _ = cfg[state]
    n_half = n_checks // 2

    seq = []

    # First half: JP checks (or equivalent wait)
    for k in range(n_half):
        if k > 0 and wait_between > 0:
            seq.append(('wait', wait_between))
        if idle_mode:
            seq.append(('wait', jp_overhead))
        else:
            seq.append(('joint_parity', 0))

    # Echo gate
    seq.append((echo_gate, 0))

    # Second half: JP checks (or equivalent wait)
    for k in range(n_half):
        if k > 0 and wait_between > 0:
            seq.append(('wait', wait_between))
        if idle_mode:
            seq.append(('wait', jp_overhead))
        else:
            seq.append(('joint_parity', 0))

    # Basis rotation (if needed)
    if meas_gate is not None:
        seq.append((meas_gate, 0))

    return seq


class DualRailFidelityChecksExperiment(Experiment):
    """
    Dual Rail Fidelity vs JP Checks Experiment.

    Sweeps number of JP checks for all 6 cardinal states.
    Optionally runs idle baseline for comparison.
    Extracts Pauli error rates from fidelity measurements.
    """

    def __init__(self, soccfg=None, path='', prefix='DualRailFidelityChecks',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        cfg = self.cfg
        s_swap = cfg.expt.storage_swap
        s_parity = cfg.expt.storage_parity
        man = cfg.expt.manipulate
        reps = cfg.expt.reps
        rounds = cfg.expt.get('rounds', 1)

        n_checks_list = list(cfg.expt.n_checks_list)
        states = list(cfg.expt.get('states', ['0', '1', '+', '-', '+i', '-i']))
        wait_between = cfg.expt.get('wait_between_checks', 0)
        run_idle = cfg.expt.get('run_idle_baseline', True)

        # --- Per-pair gate override (TEMPORARY DIAGNOSTIC) ---
        state_config_override = cfg.expt.get('state_config_override', None)
        if state_config_override is not None:
            # Merge: override only the keys provided, fall back to STATE_CONFIG
            state_config = {**STATE_CONFIG, **state_config_override}
            print(f"\n{'!'*60}")
            print(f"WARNING: state_config_override is active (TEMPORARY DIAGNOSTIC)")
            print(f"  Overriding gates for states: {list(state_config_override.keys())}")
            for s, (mg, eg, ep) in state_config_override.items():
                orig = STATE_CONFIG.get(s, ('?', '?', '?'))
                print(f"  |{s}>: meas={orig[0]}->{mg}, echo={orig[1]}->{eg}, "
                      f"expected={orig[2]}->{ep}")
            print(f"{'!'*60}\n")
        else:
            state_config = STATE_CONFIG

        # Validate
        for state in states:
            assert state in state_config, f"Unknown state '{state}'"
        for n in n_checks_list:
            if n > 0:
                assert n % 2 == 0, f"n_checks must be even, got {n}"

        data = {
            'n_checks_list': np.array(n_checks_list),
            'states': states,
            'storage_swap': s_swap,
            'storage_parity': s_parity,
            'threshold': cfg.device.readout.threshold[0],
            'reps': reps,
            'rounds': rounds,
            'wait_between': wait_between,
            'state_config': state_config,
        }

        # --- Compute JP overhead from a dummy program ---
        jp_overhead = self._compute_jp_overhead(cfg, s_swap, s_parity)
        data['jp_overhead'] = jp_overhead
        print(f"JP overhead: {jp_overhead:.2f} us")

        # --- Run JP mode ---
        print(f"\n{'='*60}")
        print(f"Fidelity Checks: {len(states)} states x "
              f"{len(n_checks_list)} n_checks = "
              f"{len(states) * len(n_checks_list)} points")
        print(f"{'='*60}")

        for n_idx, n_checks in enumerate(n_checks_list):
            for state in states:
                label = f"{state}_n{n_checks}"
                print(f"\n--- JP mode: state='{state}', n_checks={n_checks} ---")

                pulse_seq = build_fidelity_pulse_sequence(
                    n_checks, state, wait_between=wait_between,
                    state_config=state_config)

                i0, q0, read_num = self._run_single_point(
                    cfg, s_swap, s_parity, man, state, pulse_seq,
                    progress=progress)

                data[f'i0_{label}'] = i0
                data[f'q0_{label}'] = q0
                data[f'read_num_{label}'] = read_num

        # --- Run idle baseline ---
        if run_idle:
            print(f"\n{'='*60}")
            print(f"Idle baseline (replacing JP with {jp_overhead:.1f} us wait)")
            print(f"{'='*60}")

            for n_idx, n_checks in enumerate(n_checks_list):
                for state in states:
                    label = f"idle_{state}_n{n_checks}"
                    print(f"\n--- Idle mode: state='{state}', "
                          f"n_checks={n_checks} ---")

                    pulse_seq = build_fidelity_pulse_sequence(
                        n_checks, state, wait_between=wait_between,
                        idle_mode=True, jp_overhead=jp_overhead,
                        state_config=state_config)

                    i0, q0, read_num = self._run_single_point(
                        cfg, s_swap, s_parity, man, state, pulse_seq,
                        progress=progress)

                    data[f'i0_{label}'] = i0
                    data[f'q0_{label}'] = q0
                    data[f'read_num_{label}'] = read_num

        self.data = data
        return data

    def _run_single_point(self, cfg, s_swap, s_parity, man, state, pulse_seq,
                          progress=False):
        """Run a single (state, pulse_sequence) point via sandbox_v2."""
        # Build sandbox_v2 config
        sandbox_expt = dict(
            qubits=cfg.expt.qubits,
            reps=cfg.expt.reps,
            rounds=cfg.expt.get('rounds', 1),
            manipulate=man,
            storage_swap=[s_swap],
            storage_parity=[s_parity],
            state_start=[state],
            pulse_sequence=pulse_seq,
            measure_pairs=[0],
            # Active reset + sub-params (must match get_active_reset_params defaults)
            active_reset=cfg.expt.get('active_reset', False),
            ef_reset=cfg.expt.get('ef_reset', True),
            man_reset=cfg.expt.get('man_reset', True),
            storage_reset=cfg.expt.get('storage_reset', True),
            coupler_reset=cfg.expt.get('coupler_reset', False),
            pre_selection_reset=cfg.expt.get('pre_selection_reset', False),
            use_qubit_man_reset=cfg.expt.get('use_qubit_man_reset', False),
            pre_selection_parity=cfg.expt.get('pre_selection_parity', False),
            # Experiment options
            state_prep_postselect=cfg.expt.get('state_prep_postselect', False),
            state_prep_ps_parity=cfg.expt.get('state_prep_ps_parity', False),
            parity_fast=cfg.expt.get('parity_fast', False),
            measure_parity=cfg.expt.get('measure_parity', True),
            phase_tracking=cfg.expt.get('phase_tracking', True),
            reset_before_dual_rail=cfg.expt.get('reset_before_dual_rail', False),
            reset_after_dual_rail=cfg.expt.get('reset_after_dual_rail', False),
        )

        # Override expt config temporarily
        orig_expt = cfg.expt
        cfg.expt = AttrDict(sandbox_expt)

        try:
            prog = DualRailSandboxV2Program(soccfg=self.soccfg, cfg=cfg)
            read_num = prog._calculate_read_num()

            avgi, avgq = prog.acquire(
                self.im[cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num,
            )

            i0, q0, _ = prog.collect_shots()
        finally:
            cfg.expt = orig_expt

        return i0, q0, read_num

    def _compute_jp_overhead(self, cfg, s_swap, s_parity):
        """Compute JP check overhead (us) by instantiating a dummy program."""
        man = cfg.expt.manipulate

        # Build a minimal config with 2 JP checks to get overhead
        dummy_expt = dict(
            qubits=cfg.expt.qubits,
            reps=1,
            rounds=1,
            manipulate=man,
            storage_swap=[s_swap],
            storage_parity=[s_parity],
            state_start=['0'],
            pulse_sequence=[('joint_parity', 0), ('joint_parity', 0)],
            measure_pairs=[0],
            active_reset=False,
            parity_fast=cfg.expt.get('parity_fast', False),
            measure_parity=cfg.expt.get('measure_parity', True),
            phase_tracking=cfg.expt.get('phase_tracking', True),
        )

        orig_expt = cfg.expt
        cfg.expt = AttrDict(dummy_expt)
        try:
            prog = DualRailSandboxV2Program(soccfg=self.soccfg, cfg=cfg)
            # Swap pulse duration
            swap_dur = prog.get_total_time(
                np.array(prog._get_swap_pulse(0), dtype=object))
            # JP measurement duration
            stor_pair_name = prog.compiled_ops[0]['stor_pair_name']
            jp_meas_dur = prog.get_jp_measurement_duration(
                stor_pair_name,
                fast=cfg.expt.get('parity_fast', False))
        finally:
            cfg.expt = orig_expt

        return 2 * swap_dur + jp_meas_dur

    def _get_shot_indices(self, pulse_sequence):
        """Compute shot column indices for a given pulse_sequence.

        Replicates the logic from DualRailSandboxV2Experiment._get_shot_indices()
        but takes pulse_sequence as argument rather than reading from cfg.
        """
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

        # JP readouts from pulse sequence
        jp_indices = []
        for op_name, arg in pulse_sequence:
            if op_name == 'joint_parity':
                jp_indices.append(idx)
                idx += 1
        if jp_indices:
            indices['jp'] = jp_indices

        # Dual-rail measurement (single pair at index 0)
        if cfg.expt.get('reset_before_dual_rail', False):
            idx += 1
        indices['dr_swap_p0'] = idx
        indices['dr_parity_p0'] = idx + 1
        idx += 2
        if cfg.expt.get('reset_after_dual_rail', False):
            idx += 1

        return indices

    @staticmethod
    def _bin_pair_shots(i0_filtered, indices, threshold, measure_parity=True):
        """Bin dual rail shots into 00, 10, 01, 11 populations."""
        n_shots = len(i0_filtered)
        if n_shots == 0:
            return {'00': 0, '01': 0, '10': 0, '11': 0}

        swap_shots = i0_filtered[:, indices['dr_swap_p0']]
        parity_shots = i0_filtered[:, indices['dr_parity_p0']]

        if measure_parity:
            swap_state = (swap_shots > threshold).astype(int)
            parity_state = (parity_shots > threshold).astype(int)
        else:
            swap_state = (swap_shots < threshold).astype(int)
            parity_state = (parity_shots < threshold).astype(int)

        combined = swap_state * 10 + parity_state
        counts = {
            '00': int(np.sum(combined == 0)),
            '01': int(np.sum(combined == 1)),
            '10': int(np.sum(combined == 10)),
            '11': int(np.sum(combined == 11)),
        }
        pops = {k: v / n_shots for k, v in counts.items()}
        return pops

    def _analyze_single_point(self, data, label, pulse_sequence, state,
                              state_config=None):
        """Analyze a single (state, n_checks) data point.

        Returns dict with populations, fidelity, JP statistics.
        """
        threshold = data['threshold']
        reps = data['reps']
        measure_parity = self.cfg.expt.get('measure_parity', True)

        i0 = data[f'i0_{label}']
        read_num = data[f'read_num_{label}']
        indices = self._get_shot_indices(pulse_sequence)

        i0_reshaped = i0.reshape(reps, read_num)

        # Pre-selection filtering
        if 'ar_pre_selection' in indices:
            mask = i0_reshaped[:, indices['ar_pre_selection']] < threshold
            i0_reshaped = i0_reshaped[mask]

        # State-prep post-selection
        if 'state_prep_ps' in indices:
            mask = i0_reshaped[:, indices['state_prep_ps']] < threshold
            i0_reshaped = i0_reshaped[mask]

        result = {'raw_count': len(i0_reshaped)}

        # JP statistics
        n_jp = 0
        if 'jp' in indices:
            jp_idx = indices['jp']
            n_jp = len(jp_idx)
            jp_shots = i0_reshaped[:, jp_idx]
            jp_even_frac = np.mean(jp_shots < threshold, axis=0)
            result['jp_even_frac'] = jp_even_frac
            # Fraction passing ALL checks
            all_pass = np.all(jp_shots < threshold, axis=1)
            result['jp_all_pass_frac'] = float(np.mean(all_pass))
            result['jp_pass_count'] = int(np.sum(all_pass))

        # Raw populations
        pops_raw = self._bin_pair_shots(
            i0_reshaped, indices, threshold, measure_parity)
        result['pops_raw'] = pops_raw

        # Post-selected populations (filter on JP even parity)
        if n_jp > 0:
            jp_shots = i0_reshaped[:, indices['jp']]
            ps_mask = np.all(jp_shots < threshold, axis=1)
            i0_ps = i0_reshaped[ps_mask]
            pops_ps = self._bin_pair_shots(
                i0_ps, indices, threshold, measure_parity)
            result['pops_ps'] = pops_ps
            result['ps_count'] = len(i0_ps)
        else:
            result['pops_ps'] = pops_raw
            result['ps_count'] = len(i0_reshaped)

        # Fidelity: probability of expected outcome in logical subspace
        cfg = state_config if state_config is not None else STATE_CONFIG
        _, _, expected_key = cfg[state]
        for pop_type in ['raw', 'ps']:
            pops = result[f'pops_{pop_type}']
            logical_total = pops['10'] + pops['01']
            if logical_total > 0:
                fidelity = pops[expected_key] / logical_total
            else:
                fidelity = 0.0
            result[f'fidelity_{pop_type}'] = fidelity
            result[f'leakage_{pop_type}'] = pops['00'] + pops['11']

        return result

    def analyze(self, data=None):
        if data is None:
            data = self.data

        n_checks_list = data['n_checks_list']
        states = data['states']
        wait_between = data.get('wait_between', 0)
        jp_overhead = data.get('jp_overhead', 0)
        run_idle = self.cfg.expt.get('run_idle_baseline', True)
        state_config = data.get('state_config', STATE_CONFIG)

        n_states = len(states)
        n_sweep = len(n_checks_list)

        # Initialize result arrays
        fidelity_jp_raw = np.zeros((n_states, n_sweep))
        fidelity_jp_ps = np.zeros((n_states, n_sweep))
        leakage_jp_raw = np.zeros((n_states, n_sweep))
        jp_all_pass_frac = np.zeros((n_states, n_sweep))

        fidelity_idle = np.zeros((n_states, n_sweep))
        leakage_idle = np.zeros((n_states, n_sweep))

        # Analyze JP mode
        for s_idx, state in enumerate(states):
            for n_idx, n_checks in enumerate(n_checks_list):
                label = f"{state}_n{n_checks}"
                pulse_seq = build_fidelity_pulse_sequence(
                    n_checks, state, wait_between=wait_between,
                    state_config=state_config)

                result = self._analyze_single_point(
                    data, label, pulse_seq, state, state_config=state_config)

                fidelity_jp_raw[s_idx, n_idx] = result['fidelity_raw']
                fidelity_jp_ps[s_idx, n_idx] = result['fidelity_ps']
                leakage_jp_raw[s_idx, n_idx] = result['leakage_raw']

                if 'jp_all_pass_frac' in result:
                    jp_all_pass_frac[s_idx, n_idx] = result['jp_all_pass_frac']
                else:
                    jp_all_pass_frac[s_idx, n_idx] = 1.0

                # Store per-check JP fractions
                if 'jp_even_frac' in result:
                    data[f'jp_even_frac_{label}'] = result['jp_even_frac']

        data['fidelity_jp_raw'] = fidelity_jp_raw
        data['fidelity_jp_ps'] = fidelity_jp_ps
        data['leakage_jp_raw'] = leakage_jp_raw
        data['jp_all_pass_frac'] = jp_all_pass_frac

        # Analyze idle baseline
        if run_idle:
            for s_idx, state in enumerate(states):
                for n_idx, n_checks in enumerate(n_checks_list):
                    label = f"idle_{state}_n{n_checks}"
                    pulse_seq = build_fidelity_pulse_sequence(
                        n_checks, state, wait_between=wait_between,
                        idle_mode=True, jp_overhead=jp_overhead,
                        state_config=state_config)

                    result = self._analyze_single_point(
                        data, label, pulse_seq, state, state_config=state_config)

                    fidelity_idle[s_idx, n_idx] = result['fidelity_raw']
                    leakage_idle[s_idx, n_idx] = result['leakage_raw']

            data['fidelity_idle'] = fidelity_idle
            data['leakage_idle'] = leakage_idle

        # Pauli error rates (from raw JP fidelity)
        self._compute_pauli_rates(data, fidelity_jp_raw, 'jp_raw')
        self._compute_pauli_rates(data, fidelity_jp_ps, 'jp_ps')
        if run_idle:
            self._compute_pauli_rates(data, fidelity_idle, 'idle')

        return data

    def _compute_pauli_rates(self, data, fidelity_arr, prefix):
        """Extract Pauli error rates from 6-state fidelity array.

        fidelity_arr: shape (n_states, n_sweep), ordered as data['states'].
        """
        states = data['states']
        n_sweep = fidelity_arr.shape[1]

        # Average fidelity per axis
        f_x = np.zeros(n_sweep)
        f_y = np.zeros(n_sweep)
        f_z = np.zeros(n_sweep)
        count_x, count_y, count_z = 0, 0, 0

        for s_idx, state in enumerate(states):
            axis = STATE_AXIS[state]
            if axis == 'X':
                f_x += fidelity_arr[s_idx]
                count_x += 1
            elif axis == 'Y':
                f_y += fidelity_arr[s_idx]
                count_y += 1
            elif axis == 'Z':
                f_z += fidelity_arr[s_idx]
                count_z += 1

        if count_x > 0:
            f_x /= count_x
        if count_y > 0:
            f_y /= count_y
        if count_z > 0:
            f_z /= count_z

        data[f'F_X_{prefix}'] = f_x
        data[f'F_Y_{prefix}'] = f_y
        data[f'F_Z_{prefix}'] = f_z

        # Pauli rates: p_I + p_X + p_Y + p_Z = 1
        # F_X = p_I + p_X, F_Y = p_I + p_Y, F_Z = p_I + p_Z
        p_x = (1 + f_x - f_y - f_z) / 2
        p_y = (1 - f_x + f_y - f_z) / 2
        p_z = (1 - f_x - f_y + f_z) / 2
        p_i = (f_x + f_y + f_z - 1) / 2

        data[f'p_X_{prefix}'] = p_x
        data[f'p_Y_{prefix}'] = p_y
        data[f'p_Z_{prefix}'] = p_z
        data[f'p_I_{prefix}'] = p_i

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data

        n_checks_list = data['n_checks_list']
        states = data['states']
        run_idle = 'fidelity_idle' in data

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Plot 1: Fidelity vs N, all 6 states (raw) ---
        ax = axes[0, 0]
        for s_idx, state in enumerate(states):
            axis = STATE_AXIS[state]
            color = AXIS_COLORS[axis]
            linestyle = '-' if state in ('0', '+', '+i') else '--'
            ax.plot(n_checks_list, data['fidelity_jp_raw'][s_idx],
                    'o-', color=color, linestyle=linestyle,
                    label=f"|{state}> ({axis})", markersize=5)
        ax.set_xlabel('Number of JP checks')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity vs JP checks (raw)')
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim([0.4, 1.02])
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Fidelity vs N, JP (raw + PS) vs idle, per axis ---
        ax = axes[0, 1]
        for axis_label, color in AXIS_COLORS.items():
            f_key = f'F_{axis_label}'
            # JP raw
            if f'{f_key}_jp_raw' in data:
                ax.plot(n_checks_list, data[f'{f_key}_jp_raw'],
                        'o-', color=color, label=f'{axis_label} (JP raw)')
            # JP post-selected
            if f'{f_key}_jp_ps' in data:
                ax.plot(n_checks_list, data[f'{f_key}_jp_ps'],
                        's--', color=color, alpha=0.6,
                        label=f'{axis_label} (JP PS)')
            # Idle
            if run_idle and f'{f_key}_idle' in data:
                ax.plot(n_checks_list, data[f'{f_key}_idle'],
                        '^:', color=color, alpha=0.4,
                        label=f'{axis_label} (idle)')
        ax.set_xlabel('Number of JP checks')
        ax.set_ylabel('Axis-averaged fidelity')
        ax.set_title('JP vs idle comparison')
        ax.legend(fontsize=7, ncol=3)
        ax.set_ylim([0.4, 1.02])
        ax.grid(True, alpha=0.3)

        # --- Plot 3: Pauli error rates vs N ---
        ax = axes[1, 0]
        for pauli, color in [('X', 'tab:blue'), ('Y', 'tab:red'),
                              ('Z', 'tab:green')]:
            key_jp = f'p_{pauli}_jp_raw'
            if key_jp in data:
                ax.plot(n_checks_list, data[key_jp],
                        'o-', color=color, label=f'p_{pauli} (JP)')
            if run_idle:
                key_idle = f'p_{pauli}_idle'
                if key_idle in data:
                    ax.plot(n_checks_list, data[key_idle],
                            '^:', color=color, alpha=0.5,
                            label=f'p_{pauli} (idle)')
        ax.set_xlabel('Number of JP checks')
        ax.set_ylabel('Pauli error rate')
        ax.set_title('Pauli error rates')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Plot 4: JP passing fraction vs N ---
        ax = axes[1, 1]
        # Average across states
        jp_frac_avg = np.mean(data['jp_all_pass_frac'], axis=0)
        ax.plot(n_checks_list, jp_frac_avg, 'ko-',
                label='Average (all states)', linewidth=2)
        # Per state
        for s_idx, state in enumerate(states):
            axis = STATE_AXIS[state]
            color = AXIS_COLORS[axis]
            linestyle = '-' if state in ('0', '+', '+i') else '--'
            ax.plot(n_checks_list, data['jp_all_pass_frac'][s_idx],
                    'o', color=color, linestyle=linestyle,
                    alpha=0.5, markersize=4, label=f"|{state}>")
        ax.set_xlabel('Number of JP checks')
        ax.set_ylabel('Fraction passing all checks')
        ax.set_title('JP passing fraction')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        """Save data, converting non-HDF5 types."""
        if data is None:
            data = self.data

        # Convert lists/tuples to arrays for HDF5; skip non-serializable objects
        _skip = {'state_config'}  # dicts of tuples have no HDF5 equivalent
        save_data = {}
        for k, v in data.items():
            if k in _skip:
                continue
            if isinstance(v, (list, tuple)):
                if v and isinstance(v[0], str):
                    save_data[k] = np.array(v, dtype='S')
                else:
                    save_data[k] = np.array(v)
            else:
                save_data[k] = v

        super().save_data(save_data)

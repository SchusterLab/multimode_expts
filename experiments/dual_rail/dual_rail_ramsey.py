'''
Dual Rail Ramsey Experiment

Measures T2 Ramsey coherence of the dual-rail qubit encoded in two storage modes.

Pulse sequence:
1. (Optional) Active reset
2. State preparation (|10> or |01>)
3. Pi/2 gate: M1-Si hpi (half swap) + M1-Sj pi (full swap) -> superposition
4. Variable wait time (swept in firmware via MMRAveragerProgram)
5. Pi/2 gate with Ramsey phase: M1-Si hpi (phase from virtual Ramsey freq) + M1-Sj pi
6. Final dual rail measurement via measure_dual_rail()

The pi/2 gate automatically assigns hpi to the storage containing the photon:
- State '10': hpi on M1-S1, pi on M1-S2
- State '01': hpi on M1-S2, pi on M1-S1

Seb 02/2026
'''

import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

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
        qTest = self.qubits[0]

        # Build storage mode names
        man = cfg.expt.manipulate
        s1 = cfg.expt.storage_1
        s2 = cfg.expt.storage_2
        self.stor_name_1 = f'M{man}-S{s1}'
        self.stor_name_2 = f'M{man}-S{s2}'
        self.man_idx = man

        # Get dataset for pulse parameters
        _ds = cfg.device.storage._ds_storage

        # Determine hpi/pi storage assignment based on state_start
        state_start = cfg.expt.get('state_start', '10')
        self.state_start = state_start

        if state_start == '10':
            self.hpi_stor = self.stor_name_1  # hpi on storage with photon
            self.pi_stor = self.stor_name_2
        elif state_start == '01':
            self.hpi_stor = self.stor_name_2  # hpi on storage with photon
            self.pi_stor = self.stor_name_1
        else:
            raise ValueError(f"state_start must be '10' or '01' for Ramsey, got '{state_start}'")

        # --- HPI storage pulse parameters (register-based for phase sweep) ---
        self.hpi_freq = _ds.get_freq(self.hpi_stor)
        self.hpi_gain = _ds.get_gain(self.hpi_stor)
        hpi_length_us = _ds.get_h_pi(self.hpi_stor)
        # print('carefull playing PI only')
        # hpi_length_us = _ds.get_pi(self.hpi_stor)

        # Determine flux channel based on frequency (matches prepulse_creator2.storage)
        flux_low_ch = cfg.hw.soc.dacs.flux_low.ch[0]
        flux_high_ch = cfg.hw.soc.dacs.flux_high.ch[0]
        self.hpi_flux_ch = flux_low_ch if self.hpi_freq < 1800 else flux_high_ch

        # Add gaussian ramp waveform for hpi flat_top pulse
        ramp_sigma_us = cfg.device.storage.ramp_sigma
        sigma_cycles = self.us2cycles(ramp_sigma_us, gen_ch=self.hpi_flux_ch)
        self.add_gauss(ch=self.hpi_flux_ch, name="hpi_stor_ramp",
                       sigma=sigma_cycles, length=sigma_cycles * 6)

        # Pre-compute register values for hpi
        self.hpi_freq_reg = self.freq2reg(self.hpi_freq, gen_ch=self.hpi_flux_ch)
        self.hpi_length_cycles = self.us2cycles(hpi_length_us, gen_ch=self.hpi_flux_ch)

        # --- PI storage pulse (via prepulse_creator, no phase sweep needed) ---
        pi_seq = [['storage', self.pi_stor, 'pi', 0]]
        creator = self.get_prepulse_creator(pi_seq)
        self.pi_stor_pulse = creator.pulse.tolist()

        # --- Sweep registers on hpi flux channel's register page ---
        self.flux_rp = self.ch_page(self.hpi_flux_ch)
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_flux_phase = self.sreg(self.hpi_flux_ch, "phase")

        # Initialize registers (us2cycles WITHOUT gen_ch, matching sideband_ramsey)
        self.safe_regwi(self.flux_rp, self.r_wait,
                        self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.flux_rp, self.r_phase2, 0)

        # Phase step register: use safe_regwi + math (register-register)
        # to avoid mathi's 31-bit immediate limit (deg2reg can exceed 2^31
        # when abs(ramsey_freq) * step > 0.5)
        self.r_phase_step = 5
        phase_step_val = self.deg2reg(
            360 * abs(cfg.expt.ramsey_freq) * cfg.expt.step,
            gen_ch=self.hpi_flux_ch)
        self.safe_regwi(self.flux_rp, self.r_phase_step, phase_step_val)
        self.ramsey_freq_sign = 1 if cfg.expt.ramsey_freq >= 0 else -1

        # --- State preparation pulse ---
        state_prep_seq = self.prep_dual_rail_state(state_start, self.stor_name_1, self.stor_name_2)
        if state_prep_seq is not None:
            creator = self.get_prepulse_creator(state_prep_seq)
            self.state_prep_pulse = creator.pulse.tolist()
        else:
            self.state_prep_pulse = None

        # --- check if doing an echo 
        self.echo = cfg.expt.get('echo', False)
        if self.echo:
            pi_echo_seq = [
                ['storage', self.pi_stor, 'pi', 0],
                ['storage', self.hpi_stor, 'pi', 0],
                ['storage', self.pi_stor, 'pi', 0],
                ]
            creator = self.get_prepulse_creator(pi_echo_seq)
            self.pi_echo_pulse = creator.pulse.tolist()

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)

        # 1. Phase reset
        self.reset_and_sync()

        # 2. Active reset (if enabled)
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # 3. State preparation
        if self.state_prep_pulse is not None:
            self.custom_pulse(cfg, self.state_prep_pulse, prefix='state_prep_')

        # 4. First pi/2 gate: hpi (phase=0) + pi (phase=0)
        self.setup_and_pulse(
            ch=self.hpi_flux_ch, style="flat_top",
            freq=self.hpi_freq_reg,
            phase=self.deg2reg(0, gen_ch=self.hpi_flux_ch),
            gain=self.hpi_gain,
            length=self.hpi_length_cycles,
            waveform="hpi_stor_ramp")
        self.sync_all(self.us2cycles(0.01))

        self.custom_pulse(cfg, self.pi_stor_pulse, prefix='pi1_')

        # # 5. Variable wait
        self.sync_all()
        self.sync(self.flux_rp, self.r_wait)
        self.sync_all()

        if self.echo:
            # Echo option: add an extra pi pulse in the middle of the wait time
            self.custom_pulse(cfg, self.pi_echo_pulse, prefix='pi_echo_')
            self.sync_all(self.us2cycles(0.01))
            # add an extra wait after the echo pulse
            self.sync_all()
            self.sync(self.flux_rp, self.r_wait)
            self.sync_all()


        # 6. Second pi/2 gate with Ramsey phase:  pi (phase=0) + hpi (phase from r_phase2) + an extra pi to empty the manipulate

        self.custom_pulse(cfg, self.pi_stor_pulse, prefix='pi2_')

        self.set_pulse_registers(
            ch=self.hpi_flux_ch, style="flat_top",
            freq=self.hpi_freq_reg,
            phase=self.deg2reg(0, gen_ch=self.hpi_flux_ch),
            gain=self.hpi_gain,
            length=self.hpi_length_cycles,
            waveform="hpi_stor_ramp")
        self.sync_all(self.us2cycles(0.01))
        # # Copy accumulated Ramsey phase to the flux channel's phase register
        self.mathi(self.flux_rp, self.r_flux_phase, self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))
        self.pulse(ch=self.hpi_flux_ch)
        self.sync_all(self.us2cycles(0.01))

        self.custom_pulse(cfg, self.pi_stor_pulse, prefix='pi3_')

        # 7. Final dual rail measurement
        self.measure_dual_rail(
            storage_idx=(cfg.expt.storage_1, cfg.expt.storage_2),
            measure_parity=cfg.expt.get('measure_parity', True),
            reset_before=cfg.expt.get('reset_before_dual_rail', False),
            reset_after=cfg.expt.get('reset_after_dual_rail', False),
            man_idx=self.man_idx,
            final_sync=True)

    def update(self):
        # Increment wait time (us2cycles WITHOUT gen_ch, matching sideband_ramsey)
        step_cycles = self.us2cycles(self.cfg.expt.step)
        if self.echo:
            # divide step by 2 for echo, since we have two waits
            step_cycles = self.us2cycles(self.cfg.expt.step / 2)
        self.mathi(self.flux_rp, self.r_wait, self.r_wait, '+', step_cycles)
        self.sync_all(self.us2cycles(0.01))
        # Increment Ramsey phase using register-register math (no 31-bit limit)
        op = '+' if self.ramsey_freq_sign >= 0 else '-'
        self.math(self.flux_rp, self.r_phase2, self.r_phase2,
                  op, self.r_phase_step)
        self.sync_all(self.us2cycles(0.01))
        # print('update doing nothing')

    def _calculate_read_num(self):
        """Calculate total number of readouts per rep"""
        cfg = self.cfg
        read_num = 0

        # Active reset readouts
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

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
        readout trigger within body() (e.g. col 0 = stor1, col 1 = stor2).
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
    wait time between two pi/2 beamsplitter gates in firmware.

    Experimental Config:
    expt = dict(
        qubits: [0],
        reps: number of averages per sweep point,
        rounds: number of rounds (default 1),
        start: wait time start [us],
        step: wait time step [us] (Nyquist: 1/step > 2*ramsey_freq),
        expts: number of sweep points,
        ramsey_freq: virtual detuning frequency [MHz],
        storage_1: first storage mode index (e.g., 1),
        storage_2: second storage mode index (e.g., 2),
        manipulate: manipulate mode index (e.g., 1),
        state_start: '10', '01', or ['10', '01'] for both,
        active_reset: if True, perform active reset,
        reset_before_dual_rail: if True, reset before dual rail measurement,
        reset_after_dual_rail: if True, reset after dual rail measurement,
        measure_parity: if True use parity, if False use slow pi,
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
        state_list = self.cfg.expt.get('state_start', '10')
        if isinstance(state_list, str):
            state_list = [state_list]

        data = {
            'states': state_list,
            'threshold': self.cfg.device.readout.threshold[0],
        }

        for state in state_list:
            print(f"\n=== Dual Rail Ramsey: state='{state}' ===")
            self.cfg.expt.state_start = state

            prog = DualRailRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
            read_num = prog._calculate_read_num()

            x_pts, avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num)

            # Collect single-shot data for dual rail binning
            i0, q0, _ = prog.collect_shots()

            data[f'i0_{state}'] = i0
            data[f'q0_{state}'] = q0
            # avgi/avgq from RAveragerProgram are nested lists [adc_ch][readout][expts]
            # Convert to numpy arrays for HDF5 compatibility
            data[f'avgi_{state}'] = np.array(avgi[0])
            data[f'avgq_{state}'] = np.array(avgq[0])
            data[f'read_num_{state}'] = read_num
            data['xpts'] = x_pts

        self.data = data
        return data

    def _get_shot_indices(self):
        """Return dict mapping measurement type to column indices within a single rep."""
        cfg = self.cfg
        idx = 0
        indices = {}

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            ar_read_num = MMAveragerProgram.active_reset_read_num(**params)
            if params.get('pre_selection_reset', False):
                indices['ar_pre_selection'] = idx + ar_read_num - 1
            idx += ar_read_num

        if cfg.expt.get('reset_before_dual_rail', False):
            indices['dr_reset_before'] = idx
            idx += 1

        indices['dr_stor1'] = idx
        indices['dr_stor2'] = idx + 1
        idx += 2

        if cfg.expt.get('reset_after_dual_rail', False):
            indices['dr_reset_after'] = idx

        return indices

    def _bin_dual_rail_shots(self, i0_at_point, indices, threshold, measure_parity=True):
        """
        Bin dual rail shots into 00, 10, 01, 11 populations for a single sweep point.

        Args:
            i0_at_point: array of shape (n_shots, read_num) for one sweep point
            indices: dict from _get_shot_indices()
            threshold: readout threshold
            measure_parity: if False, invert threshold logic (slow pi)
        """
        n_shots = len(i0_at_point)
        if n_shots == 0:
            return {'00': 0, '01': 0, '10': 0, '11': 0}, {'00': 0, '01': 0, '10': 0, '11': 0}

        stor1_shots = i0_at_point[:, indices['dr_stor1']]
        stor2_shots = i0_at_point[:, indices['dr_stor2']]

        # Threshold: > threshold = qubit |e> = odd parity = 1 photon
        if measure_parity:
            stor1_state = (stor1_shots > threshold).astype(int)
            stor2_state = (stor2_shots > threshold).astype(int)
        else:
            stor1_state = (stor1_shots < threshold).astype(int)
            stor2_state = (stor2_shots < threshold).astype(int)

        counts = {'00': 0, '01': 0, '10': 0, '11': 0}
        for s1, s2 in zip(stor1_state, stor2_state):
            key = f'{s1}{s2}'
            counts[key] += 1

        pops = {k: v / n_shots for k, v in counts.items()}
        return pops, counts

    def analyze(self, data=None, fit=True, fitparams=None):
        if data is None:
            data = self.data

        state_list = data.get('states', ['10'])
        threshold = data.get('threshold')
        xpts = data.get('xpts')
        reps = self.cfg.expt.reps
        rounds = self.cfg.expt.get('rounds', 1)
        total_reps = reps * rounds
        expts = self.cfg.expt.expts
        measure_parity = self.cfg.expt.get('measure_parity', True)
        indices = self._get_shot_indices()

        bar_labels = ['00', '10', '01', '11']

        for state in state_list:
            read_num = data.get(f'read_num_{state}')
            i0 = data.get(f'i0_{state}')
            if i0 is None or read_num is None:
                continue

            # Population arrays for each measured state vs sweep point
            pop_arrays = {label: np.zeros(expts) for label in bar_labels}

            for expt_idx in range(expts):
                # Reshape this sweep point's data: (total_reps, read_num)
                # Each column = one readout trigger in body()
                i0_row = i0[expt_idx]  # shape: (total_reps * read_num,)
                i0_at_point = i0_row.reshape(total_reps, read_num)

                # Pre-selection filter (if active_reset with pre_selection)
                if 'ar_pre_selection' in indices:
                    pre_sel_idx = indices['ar_pre_selection']
                    mask = i0_at_point[:, pre_sel_idx] < threshold
                    i0_at_point = i0_at_point[mask]

                pops, _ = self._bin_dual_rail_shots(
                    i0_at_point, indices, threshold, measure_parity)

                for label in bar_labels:
                    pop_arrays[label][expt_idx] = pops[label]

            # Store population arrays
            for label in bar_labels:
                data[f'pop_{label}_{state}'] = pop_arrays[label]

            # Compute logical populations
            pop_10 = pop_arrays['10']
            pop_01 = pop_arrays['01']
            logical_total = pop_10 + pop_01
            with np.errstate(divide='ignore', invalid='ignore'):
                p_1 = np.where(logical_total > 0, pop_10 / logical_total, 0.5)
                p_0 = np.where(logical_total > 0, pop_01 / logical_total, 0.5)

            data[f'p_0_{state}'] = p_0
            data[f'p_1_{state}'] = p_1
            data[f'logical_total_{state}'] = logical_total

            # Fit decaysin1 (5 params: yscale, freq, phase_deg, decay, y0)
            # xpts may have expts or expts+1 elements depending on QICK version
            xpts_fit = xpts[:expts]  # ensure same length as pop arrays
            if fit and len(xpts_fit) > 5:
                # Fit p_0 first (more stable initial guess from FFT),
                # then seed p_1 fit from p_0 result (since p_1 = 1 - p_0,
                # same freq/decay but phase shifted by 180°)
                p0_fit = None
                try:
                    pOpt, pCov = fitter.fitdecaysin(
                        xpts_fit, p_0, fitparams=fitparams, use_x0=False)
                    data[f'fit_p0_{state}'] = pOpt
                    data[f'fit_err_p0_{state}'] = pCov
                    p0_fit = pOpt
                except Exception as e:
                    print(f"Fit failed for state {state} p_0: {e}")
                    data[f'fit_p0_{state}'] = None
                    data[f'fit_err_p0_{state}'] = None

                # Seed p_1 from p_0: flip phase by 180°, flip y0
                p1_initparams = fitparams
                if p0_fit is not None:
                    # decaysin1 params: [yscale, freq, phase_deg, decay, y0]
                    p1_initparams = [
                        p0_fit[0],              # same amplitude
                        p0_fit[1],              # same frequency
                        p0_fit[2] + 180,        # phase + 180°
                        p0_fit[3],              # same decay
                        1 - p0_fit[4],          # y0 -> 1 - y0
                    ]
                try:
                    pOpt, pCov = fitter.fitdecaysin(
                        xpts_fit, p_1, fitparams=p1_initparams, use_x0=False)
                    data[f'fit_p1_{state}'] = pOpt
                    data[f'fit_err_p1_{state}'] = pCov
                except Exception as e:
                    print(f"Fit failed for state {state} p_1: {e}")
                    data[f'fit_p1_{state}'] = None
                    data[f'fit_err_p1_{state}'] = None

        return data

    def display(self, data=None, fit=True, show_iq=False, n_iq_panels=8, **kwargs):
        if data is None:
            data = self.data

        state_list = data.get('states', ['10'])
        xpts = data.get('xpts')
        bar_labels = ['00', '10', '01', '11']
        stor_pair = f'S{self.cfg.expt.storage_1}-S{self.cfg.expt.storage_2}'

        n_states = len(state_list)
        expt_type = 'Echo' if self.cfg.expt.get('echo', False) else 'Ramsey'

        # --- Population plot: all 4 states vs wait time ---
        fig_pop, axes_pop = plt.subplots(1, n_states,
                                          figsize=(8 * n_states, 5),
                                          squeeze=False)
        axes_pop = axes_pop.flatten()

        # xpts may have expts or expts+1 elements depending on QICK version
        expts = self.cfg.expt.expts
        xpts_plot = xpts[:expts]

        for state_idx, state in enumerate(state_list):
            ax = axes_pop[state_idx]
            for label in bar_labels:
                pop = data.get(f'pop_{label}_{state}')
                if pop is not None:
                    ax.plot(xpts_plot, pop, 'o-',
                            label=r'$|%s\rangle$' % label,
                            color=self.STATE_COLORS[label],
                            markersize=4)

            ax.set_xlabel('Wait time [us]')
            ax.set_ylabel('Population')
            ax.set_title(f'Dual Rail {expt_type} [{stor_pair}]\n'
                         f'Prepared: |{state}>, '
                         f'Ramsey freq: {self.cfg.expt.ramsey_freq} MHz')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.show()

        # --- Logical subspace plot: p_0, p_1 vs wait time ---
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
                           f'Prepared: |{state}>, Ramsey freq: {self.cfg.expt.ramsey_freq} MHz']

            if fit:
                xfit = np.linspace(xpts_plot[0], xpts_plot[-1], 200)
                for p_label, color_key, name in [('p1', '10', '$p_1$'), ('p0', '01', '$p_0$')]:
                    fit_p = data.get(f'fit_{p_label}_{state}')
                    if fit_p is not None and isinstance(fit_p, (list, np.ndarray)):
                        fit_err = data.get(f'fit_err_{p_label}_{state}')
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
                        print(f"State {state}, {name}: "
                              f"T2 = {T2:.4g} +/- {T2_err:.3g} us, "
                              f"fit freq = {fit_freq:.6g} MHz")

            ax.set_xlabel('Wait time [us]')
            ax.set_ylabel('Logical Population')
            ax.set_title('\n'.join(title_parts))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

            # Secondary axis: show leakage (1 - logical_total)
            if logical_total is not None:
                ax2 = ax.twinx()
                ax2.plot(xpts_plot, 1 - logical_total, 'x-',
                         color='gray', alpha=0.5, markersize=3, label='Leakage')
                ax2.set_ylabel('Leakage', color='gray')
                ax2.set_ylim([-0.05, 1.05])
                ax2.tick_params(axis='y', labelcolor='gray')

        plt.tight_layout()
        plt.show()

        # --- IQ scatter plot: individual shots at selected sweep points ---
        figs_iq = []
        if show_iq:
            threshold = data.get('threshold')
            indices = self._get_shot_indices()
            total_reps = self.cfg.expt.reps * self.cfg.expt.get('rounds', 1)

            for state in state_list:
                i0 = data.get(f'i0_{state}')
                q0 = data.get(f'q0_{state}')
                read_num = data.get(f'read_num_{state}')
                if i0 is None or q0 is None or read_num is None:
                    continue

                # Select evenly spaced sweep points to display
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

                    # Reshape shots for this sweep point: (total_reps, read_num)
                    i0_at_point = i0[expt_idx].reshape(total_reps, read_num)
                    q0_at_point = q0[expt_idx].reshape(total_reps, read_num)

                    # Extract storage 1 and storage 2 IQ
                    i_stor1 = i0_at_point[:, indices['dr_stor1']]
                    q_stor1 = q0_at_point[:, indices['dr_stor1']]
                    i_stor2 = i0_at_point[:, indices['dr_stor2']]
                    q_stor2 = q0_at_point[:, indices['dr_stor2']]

                    ax.scatter(i_stor1, q_stor1, alpha=0.3, s=10,
                               label=f'S{self.cfg.expt.storage_1}')
                    ax.scatter(i_stor2, q_stor2, alpha=0.3, s=10,
                               label=f'S{self.cfg.expt.storage_2}')

                    # Threshold line
                    if threshold is not None:
                        ax.axvline(x=threshold, color='red', linestyle='--',
                                   linewidth=1, alpha=0.7)

                    ax.set_title(f't={wait_time:.2f} us', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    if panel_idx == 0:
                        ax.legend(fontsize=7, loc='upper right')

                # Hide unused subplots
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

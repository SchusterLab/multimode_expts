'''
Parity Phase Experiment
Sweeps the phase of the second pi/2 pulse in a joint parity measurement sequence.

The experiment involves two storage modes:
- storage_swap: swapped to/from manipulate cavity
- storage_parity: probed by joint parity pulse (NOT swapped)

State preparation (state_start parameter):
- '00': both storage modes in vacuum (no preparation)
- '10': storage_swap has 1 photon, storage_parity has 0
- '01': storage_swap has 0 photons, storage_parity has 1
- '11': both storage modes have 1 photon

State prep uses prep_man_photon() to prepare photon in manipulate via
multiphoton pulses (g0-e0, e0-f0, f0-g1), then swaps to the target storage.

Pulse sequence:
0. State preparation (if state_start != '00')
1. Swap storage_swap → manipulate
2. pi/2 on qubit (phase=0)
3. Joint parity pulse for storage_parity
4. Wait time
5. pi/2 on qubit (phase=sweep variable)
6. Swap back manipulate → storage_swap
7. Readout

Eesh 01/2026
'''

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.fit_display_classes import GeneralFitting
from experiments.MM_base import *


class ParityPhaseProgram(MMRAveragerProgram):
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # Build storage mode names
        stor_swap_name = f'M{cfg.expt.manipulate}-S{cfg.expt.storage_swap}'
        stor_parity_name = f'M{cfg.expt.manipulate}-S{cfg.expt.storage_parity}'
        self.stor_swap_name = stor_swap_name

        # Get joint parity params for the NON-swapped storage
        # Support both direct ds_storage in expt (local) and via device.storage (queue worker)
        ds_storage = cfg.expt.get('ds_storage') or cfg.device.storage.get('_ds_storage')
        if ds_storage is None:
            raise ValueError("ds_storage not found in cfg.expt or cfg.device.storage._ds_storage")
        jp_params = ds_storage.get_joint_parity(stor_parity_name)
        if jp_params is None:
            raise ValueError(f"Joint parity parameters not found for {stor_parity_name}")
        freq_bs, gain, length, wait_time = jp_params
        print('CAREFULL - setting gain to 0 for testing purposes')

        # Auto-select flux channel based on frequency
        flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        flux_ch = flux_low_ch if freq_bs < 1000 else flux_high_ch

        # Build joint parity pulse for custom_pulse
        ramp_sigma = cfg.device.storage.ramp_sigma
        self.joint_parity_pulse = [
            [freq_bs], [int(gain)], [length], [0], [flux_ch], ['flat_top'], [ramp_sigma]
        ]
        # self.joint_parity_pulse = [
            # [freq_bs], [int(0)], [length], [0], [flux_ch], ['flat_top'], [ramp_sigma]
        # ]
        # print('CAREFULL SETTING GAIN TO ZERO FOR JP')

        # Build wait pulse
        # self.wait_pulse = self.get_prepulse_creator([['wait', wait_time]]).pulse.tolist()
        self.wait_time = wait_time
        # print('CAREFULL REDUCING WAIT TIME BY 10 PERCENT')

        # Build swap pulse
        self.swap_pulse = self.get_prepulse_creator(
            [['storage', stor_swap_name, 'pi', 0]]
        ).pulse.tolist()

        # Build state preparation pulse based on state_start
        # '00': no preparation, '10': 1 photon in storage_swap, '01': 1 photon in storage_parity, '11': 1 photon in both
        # State prep uses prep_man_photon to prepare photon in manipulate, then swap to storage
        state_start = cfg.expt.get('state_start', '00')
        self.state_start = state_start

        if state_start == '00':
            self.state_prep_pulse = None
        else:
            state_prep_sequence = []
            # First digit is for storage_swap, second digit is for storage_parity

            if state_start[1] == '1':
                # Prepare 1 photon in storage_parity: prep_man → swap to storage_parity
                state_prep_sequence += self.prep_man_photon(1)
                state_prep_sequence.append(['storage', stor_parity_name, 'pi', 0])
            if state_start[0] == '1':
                # Prepare 1 photon in storage_swap: prep_man → swap to storage_swap
                state_prep_sequence += self.prep_man_photon(1)
                # print('CAREFULL I AM NOT SWAPPING THE RIGHT ONE')
                state_prep_sequence.append(['storage', stor_swap_name, 'pi', 0])

            print(f"State preparation sequence for state_start='{state_start}': {state_prep_sequence}")


            if state_prep_sequence:
                self.state_prep_pulse = self.get_prepulse_creator(state_prep_sequence).pulse.tolist()
            else:
                self.state_prep_pulse = None

        # For fast mode: define the hpi waveform with multiphoton sigma
        # Also compute AC Stark phase offset
        self.ac_stark_offset = 0  # Default: no AC Stark correction
        if cfg.expt.parity_fast:
            sigma = cfg.device.multiphoton.hpi['gn-en']['sigma'][0]
            _sigma = self.us2cycles(sigma, gen_ch=self.qubit_chs[qTest])
            self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit_ge_fast",
                           sigma=_sigma, length=_sigma*4)
            self.f_ge_fast = cfg.device.multiphoton.hpi['gn-en']['frequency'][0]
            self.f_ge_fast_reg = self.freq2reg(self.f_ge_fast, gen_ch=self.qubit_chs[qTest])
            self.gain_fast = cfg.device.multiphoton.hpi['gn-en']['gain'][0]
            print('f_ge_fast:', self.f_ge_fast)
            print('gain_fast:', self.gain_fast)
            print('sigma_fast:', sigma)

            # AC Stark shift correction: phase accumulates during wait_time
            # theta_ac = 360 * revival_stark_shift * wait_time (in degrees)
            print('no AC STARK CORRECTION FOR FAST PARITY YET')
            # man_mode_no = cfg.expt.manipulate - 1  # 0-indexed for array access
            # freq_AC = cfg.device.manipulate.revival_stark_shift[man_mode_no]
            # self.ac_stark_offset = (360 * freq_AC * wait_time) % 360
            # print(f"AC Stark phase offset: {self.ac_stark_offset:.2f} deg (freq_AC={freq_AC} MHz, wait={wait_time} us)")

        # Setup phase register for sweeping second pi/2
        self.q_rp = self.ch_page(self.qubit_chs[qTest])
        self.r_phase2 = 4  # accumulating register for sweep phase
        self.reg_phase = self.sreg(self.qubit_chs[qTest], "phase")  # actual phase register

        # Initialize sweep phase register
        self.safe_regwi(self.q_rp, self.r_phase2, self.deg2reg(cfg.expt.start, gen_ch=self.qubit_chs[qTest]))
        # Store AC Stark offset as immediate value (used in mathi)
        self.ac_stark_offset_reg = self.deg2reg(self.ac_stark_offset, gen_ch=self.qubit_chs[qTest])

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # Phase reset
        self.reset_and_sync()

        # Active reset
        if cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        self.sync_all(self.us2cycles(0.2))

        # Optional prepulse
        if cfg.expt.prepulse:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='pre_')
            else:
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        # 0. State preparation (if state_start != '00')
        if self.state_prep_pulse is not None:
            self.custom_pulse(cfg, self.state_prep_pulse, prefix='state_prep_')

        # 1. Swap storage_swap → manipulate
        self.custom_pulse(cfg, self.swap_pulse, prefix='swap_in_')

        # 2. First pi/2 (phase=0)
        if cfg.expt.parity_fast:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb",
                freq=self.freq2reg(self.f_ge_fast, gen_ch=self.qubit_chs[qTest]),
                phase=self.deg2reg(0), gain=self.gain_fast,
                waveform="hpi_qubit_ge_fast")
        else:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb",
                freq=self.f_ge_reg[qTest], phase=self.deg2reg(0),
                gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")

        self.sync_all()

        # 3. Joint parity pulse
        self.custom_pulse(cfg, self.joint_parity_pulse, prefix='parity_')

        # 4. Wait
        # self.custom_pulse(cfg, self.wait_pulse, prefix='wait_')
        self.sync_all(self.us2cycles(self.wait_time))

        # 5. Second pi/2 (phase from register + AC Stark offset)
        # Pattern: set_pulse_registers -> mathi to set phase -> pulse()
        if cfg.expt.parity_fast:
            self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb",
                freq=self.f_ge_fast_reg, phase=0, gain=self.gain_fast,
                waveform="hpi_qubit_ge_fast")
        else:
            self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb",
                freq=self.f_ge_reg[qTest], phase=0,
                gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")

        # Copy sweep phase (r_phase2) + AC Stark offset to actual phase register
        # mathi: reg_phase = r_phase2 + ac_stark_offset_reg (immediate value)
        self.mathi(self.q_rp, self.reg_phase, self.r_phase2, "+", self.ac_stark_offset_reg)
        self.sync_all(self.us2cycles(0.01))  # Wait for mathi to complete

        # Play the pulse using the register-set parameters
        self.pulse(ch=self.qubit_chs[qTest])

        self.sync_all()

        # 6. Swap back
        self.custom_pulse(cfg, self.swap_pulse, prefix='swap_out_')

        self.measure_wrapper()

    def update(self):
        qTest = self.qubits[0]
        # Increment phase register
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+',
                   self.deg2reg(self.cfg.expt.step, gen_ch=self.qubit_chs[qTest]))


class ParityPhaseExperiment(Experiment):
    """
    Parity Phase Experiment

    Sweeps the phase of the second pi/2 pulse in a joint parity measurement.

    Experimental Config:
    expt = dict(
        start: phase sweep start [degrees]
        step: phase sweep step [degrees]
        expts: number of steps in sweep
        reps: number of averages per experiment
        rounds: number of rounds to repeat experiment sweep
        storage_swap: storage mode to swap to/from manipulate (e.g., 1 for M1-S1)
        storage_parity: storage mode for joint parity (e.g., 2 for M1-S2)
        manipulate: manipulate mode number (e.g., 1 for M1)
        ds_storage: StorageManSwapDataset object
        parity_fast: bool, whether to use fast multiphoton hpi pulses (default: False)
        state_start: str or list of str, initial state(s) of the two storage modes
                     '00' = both vacuum, '10' = 1 photon in storage_swap,
                     '01' = 1 photon in storage_parity, '11' = 1 photon in both
                     Can be a list like ['00', '10', '01', '11'] to run multiple states
    )
    """

    # Color mapping for each state
    STATE_COLORS = {
        '00': 'tab:blue',
        '10': 'tab:orange',
        '01': 'tab:green',
        '11': 'tab:red',
    }

    def __init__(self, soccfg=None, path='', prefix='ParityPhase', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        # Set default values
        if 'parity_fast' not in self.cfg.expt:
            self.cfg.expt.parity_fast = False
        if 'gate_based' not in self.cfg.expt:
            self.cfg.expt.gate_based = False
        if 'prepulse' not in self.cfg.expt:
            self.cfg.expt.prepulse = False
        if 'state_start' not in self.cfg.expt:
            self.cfg.expt.state_start = '00'

        # Handle state_start as list or single string
        state_list = self.cfg.expt.state_start
        if isinstance(state_list, str):
            state_list = [state_list]

        # Calculate read_num to account for active_reset measurements
        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        # Store Ig, Ie for E/G scaling (from config)
        self.Ig = self.cfg.device.readout.Ig[0] if hasattr(self.cfg.device.readout, 'Ig') else None
        self.Ie = self.cfg.device.readout.Ie[0] if hasattr(self.cfg.device.readout, 'Ie') else None

        # Collect data for each state
        data = {
            'states': state_list,
            'xpts': None,
            'Ig': self.Ig,
            'Ie': self.Ie,
        }

        pre_selection = (self.cfg.expt.active_reset
                         and self.cfg.expt.get('pre_selection_reset', False))
        if pre_selection:
            threshold = self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]]

        for state in tqdm(state_list, desc="States", disable=not progress):
            print(f"\n=== Running state_start='{state}' ===")
            # Update config for this state
            self.cfg.expt.state_start = state

            prog = ParityPhaseProgram(soccfg=self.soccfg, cfg=self.cfg)

            x_pts, avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num
            )

            # Collect single shot data
            try:
                idata, qdata = prog.collect_shots()
                data[f'idata_{state}'] = idata
                data[f'qdata_{state}'] = qdata
            except:
                idata = None

            if pre_selection and idata is not None:
                # Reshape raw data to (expts, rounds * reps * read_num) for per-point filtering
                rounds = self.cfg.expt.rounds
                reps = self.cfg.expt.reps
                expts = self.cfg.expt.expts
                I_reshaped = np.reshape(np.transpose(
                    np.reshape(idata, (rounds, expts, reps, read_num)), (1, 0, 2, 3)),
                    (expts, rounds * reps * read_num))
                Q_reshaped = np.reshape(np.transpose(
                    np.reshape(qdata, (rounds, expts, reps, read_num)), (1, 0, 2, 3)),
                    (expts, rounds * reps * read_num))
                avgi_list, avgq_list = [], []
                for ii in range(expts):
                    mi, mq = GeneralFitting.filter_shots_per_point(
                        I_reshaped[ii], Q_reshaped[ii], read_num,
                        threshold=threshold, pre_selection=True)
                    avgi_list.append(mi)
                    avgq_list.append(mq)
                avgi = np.array(avgi_list)
                avgq = np.array(avgq_list)
            else:
                avgi = avgi[0][-1]
                avgq = avgq[0][-1]

            amps = np.abs(avgi + 1j*avgq)
            phases = np.angle(avgi + 1j*avgq)

            # Store xpts (same for all states)
            if data['xpts'] is None:
                data['xpts'] = x_pts

            # Store per-state data
            data[f'avgi_{state}'] = avgi
            data[f'avgq_{state}'] = avgq
            data[f'amps_{state}'] = amps
            data[f'phases_{state}'] = phases

        self.data = data
        return data

    def _rescale_to_pe(self, i_data, Ig=None, Ie=None):
        """
        Rescale I data to excited state probability P_e using linear scaling.
        P_e = (I - Ig) / (Ie - Ig)

        Args:
            i_data: I quadrature data (ADC units)
            Ig: Ground state I value (default: from config)
            Ie: Excited state I value (default: from config)

        Returns:
            P_e: Excited state probability (0 to 1, or beyond if outside calibration range)
        """
        if Ig is None:
            Ig = self.Ig
        if Ie is None:
            Ie = self.Ie

        if Ig is None or Ie is None:
            print("Warning: Ig/Ie not available, returning raw I data")
            return i_data

        return (i_data - Ig) / (Ie - Ig)

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        state_list = data.get('states', ['00'])

        # Fit sinusoidal oscillation for each state
        for state in state_list:
            avgi_key = f'avgi_{state}'
            avgq_key = f'avgq_{state}'

            if avgi_key not in data:
                continue

            try:
                data[f'fit_avgi_{state}'], data[f'fit_err_avgi_{state}'] = fitter.fitsin(
                    data['xpts'][:-1], data[avgi_key][:-1], fitparams=None)
                data[f'fit_avgq_{state}'], data[f'fit_err_avgq_{state}'] = fitter.fitsin(
                    data['xpts'][:-1], data[avgq_key][:-1], fitparams=None)

                # Also fit rescaled P_e data
                pe_data = self._rescale_to_pe(data[avgi_key])
                data[f'pe_{state}'] = pe_data
                data[f'fit_pe_{state}'], data[f'fit_err_pe_{state}'] = fitter.fitsin(
                    data['xpts'][:-1], pe_data[:-1], fitparams=None)

                # Extract fit parameters
                p = data[f'fit_pe_{state}']
                print(f"State '{state}': period={1/p[1]:.1f} deg, phase={p[2]:.1f} deg, amplitude={p[0]:.3f}, offset={p[3]:.3f}")

            except Exception as e:
                print(f"Fitting failed for state '{state}': {e}")

        return data

    def display(self, data=None, fit=True, rescale=True, **kwargs):
        """
        Display parity phase sweep results.

        Args:
            data: Data dict (default: self.data)
            fit: Whether to show fits (default: True)
            rescale: Whether to rescale I to P_e (default: True)
                     When True and Ig/Ie available: shows single P_e plot
                     When False or Ig/Ie not available: shows I and Q plots
        """
        if data is None:
            data = self.data

        # Get state list and convert byte strings to regular strings if needed
        state_list = data.get('states', ['00'])
        if isinstance(state_list, np.ndarray):
            state_list = [s.decode() if isinstance(s, bytes) else s for s in state_list]
        elif isinstance(state_list, list):
            state_list = [s.decode() if isinstance(s, bytes) else s for s in state_list]

        xpts = data['xpts']

        # Determine if we can rescale to P_e
        can_rescale = rescale and data.get('Ig') is not None and data.get('Ie') is not None

        if can_rescale:
            # Single plot for P_e (no Q plot needed)
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax1.set_ylabel(r'$P_e$ (Excited State Probability)')
            ax1.set_title('Parity Phase Sweep - Rescaled to $P_e$')
            ax1.set_xlabel('Phase [deg]')

            for state in state_list:
                color = self.STATE_COLORS.get(state, 'tab:gray')

                # Get P_e data
                if f'pe_{state}' in data:
                    y_data = data[f'pe_{state}']
                elif f'avgi_{state}' in data:
                    # Compute P_e on the fly
                    y_data = self._rescale_to_pe(data[f'avgi_{state}'], data.get('Ig'), data.get('Ie'))
                else:
                    print(f"Warning: No data found for state '{state}'")
                    continue

                # Plot data points
                ax1.plot(xpts[:-1], y_data[:-1], 'o', color=color, label=f"State '{state}'", markersize=5)

                # Plot fit
                fit_key = f'fit_pe_{state}'
                if fit and fit_key in data:
                    p = data[fit_key]
                    x_fit = np.linspace(xpts[0], xpts[-2], 200)
                    y_fit = fitter.sinfunc(x_fit, *p)
                    ax1.plot(x_fit, y_fit, '-', color=color, linewidth=2)

            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

        else:
            # Two plots: I and Q
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Top plot: I
            ax1 = axes[0]
            ax1.set_ylabel('I [ADC units]')
            ax1.set_title('Parity Phase Sweep')

            for state in state_list:
                color = self.STATE_COLORS.get(state, 'tab:gray')

                y_data = data.get(f'avgi_{state}')
                if y_data is None:
                    print(f"Warning: No I data found for state '{state}'")
                    continue

                # Plot data points
                ax1.plot(xpts[:-1], y_data[:-1], 'o', color=color, label=f"State '{state}'", markersize=5)

                # Plot fit
                fit_key = f'fit_avgi_{state}'
                if fit and fit_key in data:
                    p = data[fit_key]
                    x_fit = np.linspace(xpts[0], xpts[-2], 200)
                    y_fit = fitter.sinfunc(x_fit, *p)
                    ax1.plot(x_fit, y_fit, '-', color=color, linewidth=2)

            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Bottom plot: Q
            ax2 = axes[1]
            ax2.set_xlabel('Phase [deg]')
            ax2.set_ylabel('Q [ADC units]')

            for state in state_list:
                color = self.STATE_COLORS.get(state, 'tab:gray')

                y_data = data.get(f'avgq_{state}')
                if y_data is None:
                    continue

                # Plot data points
                ax2.plot(xpts[:-1], y_data[:-1], 'o', color=color, label=f"State '{state}'", markersize=5)

                # Plot fit
                fit_key = f'fit_avgq_{state}'
                if fit and fit_key in data:
                    p = data[fit_key]
                    x_fit = np.linspace(xpts[0], xpts[-2], 200)
                    y_fit = fitter.sinfunc(x_fit, *p)
                    ax2.plot(x_fit, y_fit, '-', color=color, linewidth=2)

            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n=== Fit Summary ===")
        for state in state_list:
            fit_key = 'fit_pe_' if can_rescale else 'fit_avgi_'
            if f'{fit_key}{state}' in data:
                p = data[f'{fit_key}{state}']
                print(f"State '{state}': period={1/p[1]:.1f} deg, phase={p[2]:.1f} deg, "
                      f"amplitude={p[0]:.3f}, offset={p[3]:.3f}")

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data

        # Convert string list 'states' to bytes for HDF5 compatibility
        if 'states' in data and isinstance(data['states'], list):
            data['states'] = np.array(data['states'], dtype='S')  # Convert to byte strings

        super().save_data(data=data)
        return self.fname

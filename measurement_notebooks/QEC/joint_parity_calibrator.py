"""
Joint Parity Calibrator

Automated calibration for joint parity measurement.

Two-stage calibration:
1. ALICE: Sweep beam splitter rate WITH swap to storage until pi phase shift
2. BOB: Sweep wait time WITHOUT swap to storage (at fixed rate) until pi phase shift
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from slab import AttrDict
from experiments import CharacterizationRunner
import experiments as meas
from experiments.MM_dual_rail_base import MM_dual_rail_base
import fitting.fitting as fitter


def gaussian(x, a, x0, sigma):
    """Gaussian function for fitting Wigner phase."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


class JointParityCalibrator:
    """
    Automated calibration for joint parity measurement.

    Two-stage workflow:
    - ALICE: Calibrate beam splitter rate (with storage swap)
    - BOB: Calibrate wait time (without storage swap, fixed rate)

    Parameters
    ----------
    station : MultimodeStation
        The station object containing hardware config and datasets
    client : JobClient
        The job client for submitting experiments
    use_queue : bool
        Whether to use the job queue (default: True)
    verbose : bool
        Whether to print progress messages (default: True)
    """

    def __init__(self, station, client, use_queue=True, verbose=True, debug=False):
        self.station = station
        self.client = client
        self.use_queue = use_queue
        self.verbose = verbose
        self.debug = debug
        self.mm_base = MM_dual_rail_base(station.hardware_cfg, station.soc)

        # Default experiment configs (can be overridden via kwargs)
        self.error_amp_defaults = AttrDict(dict(
            reps=50, rounds=1, qubit=0, qubits=[0],
            n_start=0, n_step=3, n_pulses=15, expts=100,
            active_reset=False, man_reset=True, storage_reset=True,
            relax_delay=2500, qubit_state_start='g',
        ))

        self.sideband_defaults = AttrDict(dict(
            start=0.01, expts=100, reps=200,
            rounds=1, qubit=0, qubits=[0],
            man_mode_no=1, stor_mode_no=1,
            prepulse=True, postpulse=True,
            pre_sweep_pulse=None, post_sweep_pulse=None,
            gate_based=True, active_reset=False,
            man_reset=True, storage_reset=True,
            update_post_pulse_phase=[False, 0],
            relax_delay=2500,
        ))

        self.wigner_defaults = AttrDict(dict(
            displace_length=0.05, reps=200,
            qubits=[0], gate_based=False,
            pulse_correction=False, relax_delay=2500,
            active_reset=False, parity_fast=False,
            prepulse=True,
        ))

        self.parity_phase_defaults = AttrDict(dict(
            start=0, step=3.6, expts=100,
            reps=200, rounds=1, manipulate=1,
            state_start='00', parity_fast=False,
            active_reset=False, man_reset=False,
            storage_reset=False, prepulse=False,
            gate_based=False, qubits=[0], relax_delay=2500,
        ))

    def _log(self, msg):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[JointParityCalibrator] {msg}")

    @staticmethod
    def _angular_diff_deg(phase1_deg, phase2_deg):
        """
        Compute signed angular difference (phase1 - phase2) in [-180, 180] degrees.

        Handles wrapping for phases returned by fitsin (which are in [-360, 360]).
        """
        diff = (phase1_deg - phase2_deg) % 360
        if diff > 180:
            diff -= 360
        return diff

    def _plot_rabi_fit(self, xpts, avgi, fit_params, state_label):
        """
        Plot Rabi oscillation data with decaysin fit.

        Parameters
        ----------
        xpts : array
            X-axis data (pulse lengths)
        avgi : array
            Y-axis data (I quadrature)
        fit_params : array
            Fit parameters [yscale, freq, phase_deg, decay, y0, x0]
        state_label : str
            Label for the state ('g' or 'e')
        """
        color = 'tab:blue' if state_label == 'g' else 'tab:red'
        pi_length = self._extract_pi_length_from_fit(fit_params)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        # Plot data points
        ax.plot(xpts, avgi, 'o', color=color, label=f'Qubit in |{state_label}>')

        # Generate and plot fit curve on denser grid
        x_fit = np.linspace(xpts[0], xpts[-2], 200)
        y_fit = fitter.decaysin(x_fit, *fit_params)
        ax.plot(x_fit, y_fit, '-', color=color, label=f'Fit: pi={pi_length:.4f} us')

        ax.set_xlabel('Pulse Length (us)')
        ax.set_ylabel('I [a.u.]')
        ax.set_title(f'Rabi - Qubit in |{state_label}>')
        ax.legend()
        plt.show()

    def _plot_wigner_fit(self, phases, parity, theta_opt, state_label=''):
        """
        Plot Wigner parity vs phase with Gaussian fit.

        Parameters
        ----------
        phases : array
            Displacement phases (radians)
        parity : array
            Measured parity values
        theta_opt : float
            Optimal phase angle (radians)
        state_label : str
            Label for the qubit state ('g' or 'e')
        """
        color = 'tab:blue' if state_label == 'g' else 'tab:red'
        phases_deg = phases * 180 / np.pi
        theta_opt_deg = theta_opt * 180 / np.pi

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        # Plot data
        ax.plot(phases_deg, parity, 'o', color=color, label=f'Qubit in |{state_label}>')

        # Fit and plot Gaussian
        try:
            p0 = [np.max(parity), phases_deg[np.argmax(parity)], 30]
            popt, _ = curve_fit(gaussian, phases_deg, parity, p0=p0)
            x_fit = np.linspace(min(phases_deg), max(phases_deg), 200)
            y_fit = gaussian(x_fit, *popt)
            ax.plot(x_fit, y_fit, '-', color='black', label=f'angle opt: {theta_opt_deg:.2f} deg')
        except Exception:
            pass

        ax.set_xlabel('Displacement Phase (deg)')
        ax.set_ylabel('Measured Parity')
        title = 'Wigner Function Cut'
        if state_label:
            title += f' (Qubit in |{state_label}>)'
        ax.set_title(title)
        ax.legend()
        plt.show()

    def _get_storage_mode_parameters(self, man_mode_no, stor_mode_no):
        """Get pulse parameters for a given storage mode."""
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'
        freq = self.station.ds_storage.get_freq(stor_name)
        gain = self.station.ds_storage.get_gain(stor_name)
        pi_len = self.station.ds_storage.get_pi(stor_name)
        h_pi_len = self.station.ds_storage.get_h_pi(stor_name)

        cfg = self.station.hardware_cfg
        flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        ch = flux_low_ch if freq < 1800 else flux_high_ch

        return freq, gain, pi_len, h_pi_len, ch

    def _build_rabi_prepulses(self, man_mode_no, stor_mode_no):
        """
        Build prepulse and postpulse sequences for Rabi experiments.

        Returns
        -------
        dict with keys: prepulse_g, prepulse_e, postpulse_g, postpulse_e
        """
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'

        # Build prepulse sequence
        prepulse = self.mm_base.prep_man_photon(man_mode_no)
        prepulse.append(['storage', stor_name, 'pi', 0])

        prepulse_g = deepcopy(prepulse)
        prepulse_e = prepulse + [['qubit', 'ge', 'pi', 0]]

        # Build postpulse (reverse of prepulse without the storage swap)
        postpulse_g = prepulse_g[-2::-1]  # reverse, skip last element
        postpulse_e = [['qubit', 'ge', 'pi', 0]] + postpulse_g

        # Convert to pulse format
        prepulse_g = self.mm_base.get_prepulse_creator(prepulse_g).pulse.tolist()
        prepulse_e = self.mm_base.get_prepulse_creator(prepulse_e).pulse.tolist()
        postpulse_g = self.mm_base.get_prepulse_creator(postpulse_g).pulse.tolist()
        postpulse_e = self.mm_base.get_prepulse_creator(postpulse_e).pulse.tolist()

        return {
            'prepulse_g': prepulse_g,
            'prepulse_e': prepulse_e,
            'postpulse_g': postpulse_g,
            'postpulse_e': postpulse_e,
        }

    def _generate_alpha_circle(self, alpha_amplitude, n_points=100):
        """
        Generate displacement points on a circle of given amplitude.

        Parameters
        ----------
        alpha_amplitude : float
            Radius of the circle in phase space
        n_points : int
            Number of points on the circle

        Returns
        -------
        alpha_circle_2d : ndarray
            Array of shape (n_points, 2) with [Re(alpha), Im(alpha)]
        """
        theta_list = np.linspace(0, 2*np.pi, n_points)
        alpha_circle = alpha_amplitude * np.exp(1j * theta_list)
        alpha_circle_2d = np.zeros((n_points, 2))
        alpha_circle_2d[:, 0] = np.real(alpha_circle)
        alpha_circle_2d[:, 1] = np.imag(alpha_circle)
        return alpha_circle_2d

    def _build_displacement_pulse(self, alpha_amplitude, man_mode_no=1):
        """
        Build a gaussian displacement pulse for the cavity.

        Parameters
        ----------
        alpha_amplitude : float
            Target displacement amplitude
        man_mode_no : int
            Manipulate mode index (1-indexed)

        Returns
        -------
        displace_pulse : list
            Pulse in standard format for prepulse_creator
        """
        cfg = self.station.hardware_cfg
        man_idx = man_mode_no - 1  # Convert to 0-indexed

        # Get displacement parameters from config
        sigma_cav = cfg.device.manipulate.displace_sigma[man_idx]
        gain_to_alpha = cfg.device.manipulate.gain_to_alpha[man_idx]
        man_ch = cfg.hw.soc.dacs.manipulate_in.ch[man_idx]
        f_cav = cfg.device.manipulate.f_ge[man_idx]

        # Calculate gain for desired alpha
        # alpha = gain * gain_to_alpha, so gain = alpha / gain_to_alpha
        gain = int(alpha_amplitude / gain_to_alpha)

        # Build gaussian displacement pulse
        # Format: [[freq], [gain], [length], [phase], [ch], [shape], [sigma]]
        displace_pulse = [
            [f_cav],           # frequency
            [gain],            # gain
            [sigma_cav * 4],   # length (4 sigma for gaussian)
            [0],               # phase (will be swept by Wigner)
            [man_ch],          # channel
            ['gauss'],         # pulse type
            [sigma_cav]        # sigma
        ]

        return displace_pulse

    def _build_joint_parity_pulse(self, freq_bs, gain, pi_length, man_mode_no=1, stor_mode_no=1):
        """
        Build the joint parity pulse.

        Parameters
        ----------
        freq_bs : float
            Beam splitter frequency
        gain : int
            Beam splitter gain
        pi_length : float
            Pi pulse length (will be doubled for parity)

        Returns
        -------
        joint_parity_pulse : list
            Pulse in standard format
        """
        _, _, _, _, ch = self._get_storage_mode_parameters(man_mode_no, stor_mode_no)
        ramp_sigma = self.station.hardware_cfg.device.storage.ramp_sigma


        joint_parity_pulse = [
            [freq_bs],           # freq (MHz)
            [gain],              # gain
            [pi_length * 2],     # length (us) - 2x pi for parity
            [0],                 # phase
            [ch[0]],             # channel
            ["flat_top"],        # pulse type
            [ramp_sigma]         # ramp sigma
        ]

        return joint_parity_pulse

    def _build_wait_pulse(self, wait_time):
        """
        Build a wait/idle pulse.

        Uses the 'wait' pulse type which creates a zero-amplitude
        const pulse at the manipulate frequency.

        Parameters
        ----------
        wait_time : float
            Wait time in microseconds

        Returns
        -------
        wait_pulse : list
            Pulse in standard format (via prepulse_creator)
        """
        wait_pulse = [['wait', wait_time]]
        return self.mm_base.get_prepulse_creator(wait_pulse).pulse.tolist()

    def _build_storage_swap_pulse(self, man_mode_no, stor_mode_no):
        """
        Build storage swap pulse (M1-S1 pi pulse).

        Returns
        -------
        swap_pulse : list
            Pulse in standard format (via prepulse_creator)
        """
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'
        swap = [['storage', stor_name, 'pi', 0]]
        return self.mm_base.get_prepulse_creator(swap).pulse.tolist()

    def _build_qubit_ge_pulse(self):
        """Build qubit ge pi pulse."""
        qubit_ge = [['qubit', 'ge', 'pi', 0]]
        return self.mm_base.get_prepulse_creator(qubit_ge).pulse.tolist()

    def _run_error_amp(self, stor_name, freq_guess, gain, length_bs, qubit_state,
                       man_mode_no=1, stor_mode_no=1, reps=50, n_pulses=15, span=0.05):
        """
        Run error amplification for one qubit state.

        Returns
        -------
        optimal_freq : float
            Optimized frequency from Gaussian fit
        expt : Experiment
            The experiment object
        """
        self._log(f"  Running error amp for qubit in |{qubit_state}>...")

        print(f"    Temporarily setting storage parameters: freq={freq_guess:.4f} MHz, gain={gain}, pi_length={length_bs:.4f} us")
        print("storage is: ", stor_name)

        # Temporarily update storage parameters
        original_freq = self.station.ds_storage.get_freq(stor_name)
        original_gain = self.station.ds_storage.get_gain(stor_name)
        original_pi = self.station.ds_storage.get_pi(stor_name)

        self.station.ds_storage.update_pi(stor_name, length_bs)
        self.station.ds_storage.update_gain(stor_name, gain)
        self.station.ds_storage.update_freq(stor_name, freq_guess)

        # Setup preprocessor
        def error_amp_preproc(station, default_expt_cfg, **kwargs):
            expt_cfg = deepcopy(default_expt_cfg)
            expt_cfg.update(kwargs)
            expt_cfg.pulse_type = ['storage', stor_name, 'pi', 0]
            expt_cfg.parameter_to_test = 'frequency'
            expt_cfg.start = freq_guess - span / 2
            expt_cfg.step = span / (expt_cfg.expts - 1)
            return expt_cfg

        runner = CharacterizationRunner(
            station=self.station,
            ExptClass=meas.ErrorAmplificationExperiment,
            default_expt_cfg=self.error_amp_defaults,
            preprocessor=error_amp_preproc,
            job_client=self.client,
            use_queue=self.use_queue,
        )

        expt = runner.execute(
            man_mode_no=man_mode_no,
            stor_mode_no=stor_mode_no,
            parameter_to_test='frequency',
            span=span,
            n_pulses=n_pulses,
            reps=reps,
            qubit_state_start=qubit_state,
            analyze=False,
            display=False,
        )

        # Analyze and extract optimal frequency
        expt.analyze(state_fin='e')
        if self.debug:
            expt.display()
        optimal_freq = expt.data['fit_avgi'][2]

        # Restore original parameters
        self.station.ds_storage.update_pi(stor_name, original_pi)
        self.station.ds_storage.update_gain(stor_name, original_gain)
        self.station.ds_storage.update_freq(stor_name, original_freq)

        self._log(f"    Optimal frequency: {optimal_freq:.4f} MHz")
        return optimal_freq, expt

    def _run_error_amp_general(self, stor_name, freq, gain, length_bs, qubit_state,
                               parameter_to_test='frequency',
                               man_mode_no=1, stor_mode_no=1,
                               reps=100, n_pulses=10, n_step=3,
                               span=None, restore_params=False):
        """
        Run error amplification for frequency or gain.

        Sets dataset params before running so the preprocessor reads them.
        Does NOT restore dataset params unless restore_params=True.

        Parameters
        ----------
        stor_name : str
            Storage mode name (e.g., 'M1-S1')
        freq : float
            Frequency to set in dataset before running
        gain : float
            Gain to set in dataset before running
        length_bs : float
            Pi length to set in dataset before running
        qubit_state : str
            'g' or 'e'
        parameter_to_test : str
            'frequency' or 'gain'
        span : float or None
            Sweep span. Defaults: 0.2 MHz for frequency, 35% of gain for gain
        restore_params : bool
            If True, save and restore original dataset params around the run

        Returns
        -------
        optimal_value : float
            Optimized frequency or gain
        expt : Experiment
            The experiment object
        """
        self._log(f"  Running error amp ({parameter_to_test}) for qubit in |{qubit_state}>...")

        # Optionally save original params
        if restore_params:
            original_freq = self.station.ds_storage.get_freq(stor_name)
            original_gain = self.station.ds_storage.get_gain(stor_name)
            original_pi = self.station.ds_storage.get_pi(stor_name)

        # Set dataset params so the preprocessor reads them
        self.station.ds_storage.update_pi(stor_name, length_bs)
        self.station.ds_storage.update_gain(stor_name, int(gain))
        self.station.ds_storage.update_freq(stor_name, freq)

        # Compute span and sweep params
        if parameter_to_test == 'frequency':
            if span is None:
                span = 0.2
            start = freq - span / 2
        elif parameter_to_test == 'gain':
            if span is None:
                span = int(gain * 0.35)
            start = int(gain - span / 2)
        else:
            raise ValueError(f"Unknown parameter_to_test: {parameter_to_test}")

        def error_amp_preproc(station, default_expt_cfg, **kwargs):
            expt_cfg = deepcopy(default_expt_cfg)
            expt_cfg.update(kwargs)
            expt_cfg.pulse_type = ['storage', stor_name, 'pi', 0]
            expt_cfg.parameter_to_test = parameter_to_test
            expt_cfg.start = start
            expt_cfg.step = (span / (expt_cfg.expts - 1) if parameter_to_test == 'frequency'
                             else int(span / (expt_cfg.expts - 1)))
            return expt_cfg

        runner = CharacterizationRunner(
            station=self.station,
            ExptClass=meas.ErrorAmplificationExperiment,
            default_expt_cfg=self.error_amp_defaults,
            preprocessor=error_amp_preproc,
            job_client=self.client,
            use_queue=self.use_queue,
        )

        expt = runner.execute(
            man_mode_no=man_mode_no,
            stor_mode_no=stor_mode_no,
            parameter_to_test=parameter_to_test,
            span=span,
            n_step=n_step,
            n_pulses=n_pulses,
            reps=reps,
            qubit_state_start=qubit_state,
            analyze=False,
            display=False,
        )

        expt.analyze(state_fin='e')
        if self.debug:
            expt.display()
        optimal_value = expt.data['fit_avgi'][2]

        # Optionally restore original params
        if restore_params:
            self.station.ds_storage.update_pi(stor_name, original_pi)
            self.station.ds_storage.update_gain(stor_name, original_gain)
            self.station.ds_storage.update_freq(stor_name, original_freq)

        self._log(f"    Optimal {parameter_to_test}: {optimal_value:.4f}")
        return optimal_value, expt

    def _build_gain_sweep_vectors(self, man_mode_no=1, stor_mode_no=1,
                                  num_pts=20, length_range_frac=(0.3, 1.2)):
        """
        Build sweep vectors for the BS rate lookup table calibration.

        Computes gain/length/frequency arrays from current dataset parameters.
        Length and gain are inversely proportional (constant gain*length product).

        Parameters
        ----------
        man_mode_no : int
            Manipulate mode number
        stor_mode_no : int
            Storage mode number
        num_pts : int
            Number of sweep points
        length_range_frac : tuple of float
            (min, max) fraction of pi_length for the sweep range

        Returns
        -------
        dict with keys:
            'gain_vectors': dict {'g': array, 'e': array}
            'length_vectors': dict {'g': array, 'e': array}
            'freq_vectors': dict {'g': array, 'e': array}
            'original_freq': float
            'original_gain': float
            'original_pi_len': float
            'stor_name': str
        """
        freq, gain, pi_len, _, _ = self._get_storage_mode_parameters(man_mode_no, stor_mode_no)
        chi_ge = self.station.hardware_cfg.device.manipulate.chi_ge[man_mode_no - 1]

        len_min = pi_len * length_range_frac[0]
        len_max = pi_len * length_range_frac[1]
        gain_min = gain * pi_len / len_max
        gain_max = gain * pi_len / len_min

        gain_vectors = {}
        length_vectors = {}
        freq_vectors = {}

        for qs in ['g', 'e']:
            gain_vectors[qs] = np.linspace(gain_min, gain_max, num=num_pts)
            length_vectors[qs] = pi_len * gain / gain_vectors[qs]
            freq_vectors[qs] = np.full(num_pts, freq)
            if qs == 'e':
                freq_vectors[qs] = freq_vectors[qs] + chi_ge

        stor_name = f'M{man_mode_no}-S{stor_mode_no}'

        self._log(f"  Gain range: {gain_min:.1f} to {gain_max:.1f}")
        self._log(f"  Length range: {len_min:.4f} to {len_max:.4f} us")

        return {
            'gain_vectors': gain_vectors,
            'length_vectors': length_vectors,
            'freq_vectors': freq_vectors,
            'original_freq': freq,
            'original_gain': gain,
            'original_pi_len': pi_len,
            'stor_name': stor_name,
        }

    def _calibrate_at_gain_point(self, stor_name, freq_guess, gain, length_bs, qubit_state,
                                 man_mode_no=1, stor_mode_no=1,
                                 reps=100, n_pulses=10, n_step=3,
                                 freq_span_initial=0.2, gain_span_frac=0.35,
                                 freq_span_final=0.1):
        """
        Run the 3-step error amp sequence at one gain/length point.

        Sequence: freq error amp -> gain error amp -> freq error amp (refined).
        Does NOT save/restore dataset params; the caller is responsible.

        Parameters
        ----------
        stor_name : str
            Storage mode name
        freq_guess : float
            Initial frequency guess
        gain : float
            Gain at this sweep point
        length_bs : float
            Pi length at this sweep point
        qubit_state : str
            'g' or 'e'
        freq_span_initial : float
            Frequency span in MHz for the first freq sweep (default: 0.2)
        gain_span_frac : float
            Gain span as fraction of current gain (default: 0.35)
        freq_span_final : float
            Frequency span for the final refined freq sweep (default: 0.1)

        Returns
        -------
        dict with keys: 'freq', 'gain', 'length', 'experiments'
        """
        experiments = {}

        # Step 1: Frequency optimization
        self._log(f"    Step 1/3: Frequency error amp")
        optimal_freq, experiments['error_amp_freq_1'] = self._run_error_amp_general(
            stor_name, freq_guess, gain, length_bs, qubit_state,
            parameter_to_test='frequency',
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
            reps=reps, n_pulses=n_pulses, n_step=n_step,
            span=freq_span_initial, restore_params=False,
        )
        self.station.ds_storage.update_freq(stor_name, optimal_freq)

        # Step 2: Gain optimization
        self._log(f"    Step 2/3: Gain error amp")
        optimal_gain, experiments['error_amp_gain'] = self._run_error_amp_general(
            stor_name, optimal_freq, gain, length_bs, qubit_state,
            parameter_to_test='gain',
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
            reps=reps, n_pulses=n_pulses, n_step=n_step,
            span=int(gain * gain_span_frac), restore_params=False,
        )
        self.station.ds_storage.update_gain(stor_name, int(optimal_gain))

        # Step 3: Final frequency optimization (refined, narrower span)
        self._log(f"    Step 3/3: Frequency error amp (refined)")
        optimal_freq_final, experiments['error_amp_freq_2'] = self._run_error_amp_general(
            stor_name, optimal_freq, int(optimal_gain), length_bs, qubit_state,
            parameter_to_test='frequency',
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
            reps=reps, n_pulses=n_pulses, n_step=n_step,
            span=freq_span_final, restore_params=False,
        )
        self.station.ds_storage.update_freq(stor_name, optimal_freq_final)

        return {
            'freq': optimal_freq_final,
            'gain': optimal_gain,
            'length': length_bs,
            'experiments': experiments,
        }

    def _extract_pi_length_from_fit(self, fit_params):
        """
        Extract pi_length from decaysin fit parameters.

        Parameters
        ----------
        fit_params : array
            Fit parameters [yscale, freq, phase_deg, decay, y0, x0]

        Returns
        -------
        pi_length : float
            Extracted pi pulse length
        """
        # Normalize phase to [-180, 180]
        phase = fit_params[2]
        if phase > 180:
            phase = phase - 360
        elif phase < -180:
            phase = phase + 360

        # Calculate pi_length from phase
        freq = fit_params[1]
        if phase < 0:
            pi_length = (1/2 - phase/180) / 2 / freq
        else:
            pi_length = (3/2 - phase/180) / 2 / freq

        return pi_length

    def _run_rabi(self, freq_bs, gain, length_bs, man_mode_no, stor_mode_no,
                  reps=100, run_both_states=False,
                  start=None, stop=None, expts=None):
        """
        Run Rabi experiment and extract pi_length.

        Parameters
        ----------
        freq_bs : float
            Beam splitter frequency
        gain : int
            Beam splitter gain
        length_bs : float
            Expected beam splitter pulse length (for setting sweep range)
        man_mode_no, stor_mode_no : int
            Mode indices
        reps : int
            Number of repetitions
        run_both_states : bool
            If True, run for both g and e states and average pi_length.
            If False (default), run only for g state.
        start, stop, expts : float, float, int
            Sweep parameters. If None, defaults based on length_bs.

        Returns
        -------
        pi_length : float
            Extracted pi pulse length from fit
        expt_g : Experiment
            The g-state experiment object
        expt_e : Experiment or None
            The e-state experiment object (if run_both_states=True)
        """
        self._log(f"  Running Rabi confirmation (run_both_states={run_both_states})...")

        pulses = self._build_rabi_prepulses(man_mode_no, stor_mode_no)
        _, _, _, _, ch = self._get_storage_mode_parameters(man_mode_no, stor_mode_no)

        def sideband_preproc(station, default_expt_cfg, **kwargs):
            expt_cfg = deepcopy(default_expt_cfg)
            expt_cfg.update(kwargs)
            expt_cfg.flux_drive = ['low', freq_bs, gain, 0]
            return expt_cfg

        runner = CharacterizationRunner(
            station=self.station,
            ExptClass=meas.SidebandGeneralExperiment,
            default_expt_cfg=self.sideband_defaults,
            preprocessor=sideband_preproc,
            job_client=self.client,
            use_queue=self.use_queue,
        )

        # Set defaults if not provided
        if start is None:
            start = 0.01
        if stop is None:
            stop = length_bs * 2
        if expts is None:
            expts = 100
        step = (stop - start) / (expts - 1)

        print('reps: ', reps)

        # Run for g state
        expt_g = runner.execute(
            freq=freq_bs, gain=gain,
            stor_mode_no=stor_mode_no,
            pre_sweep_pulse=pulses['prepulse_g'],
            post_sweep_pulse=pulses['postpulse_g'],
            reps=reps, expts=expts, start=start, step=step,
            analyze=False, display=False,
        )

        # Fit manually using fitdecaysin
        p_g, pCov_g = fitter.fitdecaysin(expt_g.data['xpts'][:-1], expt_g.data['avgi'][:-1])
        if self.debug:
            self._plot_rabi_fit(expt_g.data['xpts'], expt_g.data['avgi'], p_g, 'g')
        pi_length_g = self._extract_pi_length_from_fit(p_g)

        expt_e = None
        if run_both_states:
            # Run for e state
            expt_e = runner.execute(
                freq=freq_bs, gain=gain,
                stor_mode_no=stor_mode_no,
                pre_sweep_pulse=pulses['prepulse_e'],
                post_sweep_pulse=pulses['postpulse_e'],
                reps=reps, expts=expts, start=start, step=step,
                analyze=False, display=False,
            )

            # Fit manually
            p_e, pCov_e = fitter.fitdecaysin(expt_e.data['xpts'][:-1], expt_e.data['avgi'][:-1])
            if self.debug:
                self._plot_rabi_fit(expt_e.data['xpts'], expt_e.data['avgi'], p_e, 'e')
            pi_length_e = self._extract_pi_length_from_fit(p_e)
            pi_length = (pi_length_g + pi_length_e) / 2
            self._log(f"    pi_length_g={pi_length_g:.4f}, pi_length_e={pi_length_e:.4f}, avg={pi_length:.4f} us")
        else:
            pi_length = pi_length_g
            self._log(f"    Extracted pi_length: {pi_length:.4f} us")

        return pi_length, expt_g, expt_e

    def _run_wigner_alice(self, freq_bs, gain, pi_length, qubit_state,
                          alpha_amplitude, man_mode_no=1, stor_mode_no=1, reps=100,
                          n_points=100):
        """
        Run Wigner tomography for ALICE mode (WITH storage swap).

        Sequence: displace → swap_to_storage → (qubit_pi if e) → joint_parity → (qubit_pi if e) → swap_from_storage

        Parameters
        ----------
        n_points : int
            Number of points on the alpha circle (default: 100)

        Returns
        -------
        theta : float
            Optimal displacement angle (radians)
        expt : Experiment
            The experiment object
        """
        self._log(f"  Running ALICE Wigner for qubit in |{qubit_state}>...")

        # Build pulses
        displace = self._build_displacement_pulse(alpha_amplitude, man_mode_no)
        m1_s1 = self._build_storage_swap_pulse(man_mode_no, stor_mode_no)
        qubit_ge = self._build_qubit_ge_pulse()
        joint_parity = self._build_joint_parity_pulse(freq_bs, gain, pi_length, man_mode_no, stor_mode_no)

        # Build ALICE prepulse: displace + swap + (qubit_pi) + parity + (qubit_pi) + swap
        if qubit_state == 'g':
            prepulse = [displace[i] + m1_s1[i] + joint_parity[i] + m1_s1[i]
                       for i in range(len(displace))]
        else:  # 'e'
            prepulse = [displace[i] + m1_s1[i] + qubit_ge[i] + joint_parity[i] + qubit_ge[i] + m1_s1[i]
                       for i in range(len(displace))]

        return self._run_wigner_common(prepulse, alpha_amplitude, reps, n_points=n_points, qubit_state=qubit_state)

    def _run_wigner_bob(self, freq_bs, gain, pi_length, qubit_state,
                        alpha_amplitude, man_mode_no=1, stor_mode_no=1,
                        reps=100, wait_time=0.0, n_points=100):
        """
        Run Wigner tomography for BOB mode (WITHOUT storage swap).

        Sequence: displace → (qubit_pi if e) → joint_parity → (wait) → (qubit_pi if e)

        Parameters
        ----------
        n_points : int
            Number of points on the alpha circle (default: 100)

        Returns
        -------
        theta : float
            Optimal displacement angle (radians)
        expt : Experiment
            The experiment object
        """
        self._log(f"  Running BOB Wigner for qubit in |{qubit_state}> (wait={wait_time:.3f} us)...")

        # Build pulses
        displace = self._build_displacement_pulse(alpha_amplitude, man_mode_no)
        qubit_ge = self._build_qubit_ge_pulse()
        joint_parity = self._build_joint_parity_pulse(freq_bs, gain, pi_length, man_mode_no, stor_mode_no)

        # Build BOB prepulse
        if wait_time > 0:
            wait_pulse = self._build_wait_pulse(wait_time)
            if qubit_state == 'g':
                prepulse = [displace[i] + joint_parity[i] + wait_pulse[i]
                           for i in range(len(displace))]
            else:  # 'e'
                prepulse = [displace[i] + qubit_ge[i] + joint_parity[i] + wait_pulse[i] + qubit_ge[i]
                           for i in range(len(displace))]
        else:
            if qubit_state == 'g':
                prepulse = [displace[i] + joint_parity[i]
                           for i in range(len(displace))]
            else:  # 'e'
                prepulse = [displace[i] + qubit_ge[i] + joint_parity[i] + qubit_ge[i]
                           for i in range(len(displace))]

        return self._run_wigner_common(prepulse, alpha_amplitude, reps, n_points=n_points, qubit_state=qubit_state)

    def _run_wigner_common(self, prepulse, alpha_amplitude, reps, n_points=100, qubit_state=''):
        """
        Common Wigner execution and analysis.

        Parameters
        ----------
        prepulse : list
            Prepulse sequence
        alpha_amplitude : float
            Displacement amplitude
        reps : int
            Number of repetitions
        n_points : int
            Number of points on the alpha circle (default: 100)
        qubit_state : str
            Qubit state label ('g' or 'e') for plotting

        Returns
        -------
        theta : float
            Optimal displacement angle (radians)
        expt : Experiment
            The experiment object
        """
        # Generate alpha circle
        alpha_circle = self._generate_alpha_circle(alpha_amplitude, n_points=n_points)

        # Scale cutoff with alpha_amplitude
        cutoff = max(4, int(np.ceil(alpha_amplitude**2 + 2 * alpha_amplitude + 2)))

        runner = CharacterizationRunner(
            station=self.station,
            ExptClass=meas.WignerTomography1ModeExperiment,
            default_expt_cfg=self.wigner_defaults,
            job_client=self.client,
            use_queue=self.use_queue,
        )

        expt = runner.execute(
            prepulse=True,
            pre_sweep_pulse=prepulse,
            alpha_list=alpha_circle,
            reps=reps,
            go_kwargs=dict(analyze=False, display=False),
        )

        # Analyze and extract theta
        expt.analyze(cutoff=cutoff, debug=False)

        # Fit parity vs phase to gaussian to find optimal angle
        phases = expt.data['phases'] * 180 / np.pi  # convert to degrees
        parity = expt.data['parity']

        try:
            # Initial guess: amplitude at max, center at phase of max, width ~30 deg
            p0 = [np.max(parity), phases[np.argmax(parity)], 30]
            popt, _ = curve_fit(gaussian, phases, parity, p0=p0)
            theta = popt[1] * np.pi / 180  # convert back to radians
        except Exception:
            # Fallback: use theta_opt from analyze
            theta = expt.data.get('theta_opt', 0)

        if self.debug:
            self._plot_wigner_fit(expt.data['phases'], expt.data['parity'], theta, qubit_state)
        self._log(f"    Theta: {theta * 180 / np.pi:.2f} deg")
        return theta, expt

    def _run_parity_phase(self, storage_swap, storage_parity, state_start,
                          man_mode_no=1, reps=200, parity_fast=False):
        """
        Run ParityPhaseExperiment and extract fitted phases.

        Joint parity params must be pre-set in ds_storage for the storage_parity
        mode before calling this method.

        Parameters
        ----------
        storage_swap : int
            Storage mode to swap to/from manipulate (e.g., 3 for M1-S3)
        storage_parity : int
            Storage mode for joint parity (e.g., 1 for M1-S1)
        state_start : str or list of str
            States to prepare, e.g. '00', '01', '10', ['00', '01']
        man_mode_no : int
            Manipulate mode number (default: 1)
        reps : int
            Number of repetitions (default: 200)
        parity_fast : bool
            Whether to use fast multiphoton hpi pulses (default: False)

        Returns
        -------
        phases : dict
            Mapping state -> fitted phase in degrees, e.g. {'00': 45.2, '01': -120.5}
        expt : ParityPhaseExperiment
            The experiment object
        """
        self._log(f"  Running ParityPhaseExperiment (states={state_start})...")

        runner = CharacterizationRunner(
            station=self.station,
            ExptClass=meas.ParityPhaseExperiment,
            default_expt_cfg=self.parity_phase_defaults,
            job_client=self.client,
            use_queue=self.use_queue,
        )

        expt = runner.execute(
            storage_swap=storage_swap,
            storage_parity=storage_parity,
            manipulate=man_mode_no,
            state_start=state_start,
            reps=reps,
            parity_fast=parity_fast,
            go_kwargs=dict(analyze=True, display=False),
        )

        # Analyze locally if not already done
        state_list = state_start if isinstance(state_start, list) else [state_start]
        first_state = state_list[0]
        if f'fit_pe_{first_state}' not in expt.data and f'fit_avgi_{first_state}' not in expt.data:
            expt.analyze()

        # Display locally (go_kwargs display runs on worker, not in notebook)
        if self.debug:
            expt.display()

        # Extract phases for each state
        phases = {}
        for state in state_list:
            # Prefer P_e fit, fall back to avgi fit
            fit_key = f'fit_pe_{state}'
            if fit_key not in expt.data:
                fit_key = f'fit_avgi_{state}'
            if fit_key in expt.data:
                phases[state] = expt.data[fit_key][2]  # phase in degrees
                self._log(f"    State '{state}': phase = {phases[state]:.1f} deg")
            else:
                self._log(f"    WARNING: No fit found for state '{state}'")

        return phases, expt

    def fit_bs_rate_lookup_table(self, lookup_data, degree_rate=2, degree_freq=10,
                                store=True, skip_indices=None):
        """
        Fit polynomial lookup tables from calibration data and optionally store to dataset.

        Can be called independently to refit with different polynomial degrees
        without re-running experiments.

        Parameters
        ----------
        lookup_data : dict
            Calibration data returned by calibrate_bs_rate_lookup_table().
            Must contain: 'gain_vectors', 'length_vectors', 'freq_vectors', 'stor_name'
        degree_rate : int
            Polynomial degree for bs_rate vs gain fit (default: 2)
        degree_freq : int
            Polynomial degree for freq vs gain fit (default: 10)
        store : bool
            If True, store coefficients and gain ranges to dataset (default: True)
        skip_indices : list of int or dict, optional
            Indices of points to exclude from the fit.
            If a list, the same indices are skipped for all qubit states.
            If a dict, keys are qubit states ('g', 'e') mapping to lists of indices.

        Returns
        -------
        dict with keys:
            'bs_coeffs': dict {'g': array, 'e': array}
            'freq_coeffs': dict {'g': array, 'e': array}
            'bs_rates': dict {'g': array, 'e': array}
        """
        gain_vectors = lookup_data['gain_vectors']
        length_vectors = lookup_data['length_vectors']
        freq_vectors = lookup_data['freq_vectors']
        stor_name = lookup_data['stor_name']

        # Compute beam splitter rates from pi lengths
        bs_rates = {}
        bs_coeffs = {}
        freq_coeffs = {}

        for qs in gain_vectors.keys():
            bs_rates[qs] = 0.25 / length_vectors[qs]

            # Determine which indices to skip for this qubit state
            if skip_indices is None:
                mask = np.ones(len(gain_vectors[qs]), dtype=bool)
            elif isinstance(skip_indices, dict):
                mask = np.ones(len(gain_vectors[qs]), dtype=bool)
                mask[skip_indices.get(qs, [])] = False
            else:
                mask = np.ones(len(gain_vectors[qs]), dtype=bool)
                mask[skip_indices] = False

            n_skipped = np.sum(~mask)
            if n_skipped > 0:
                self._log(f"  Skipping {n_skipped} points for |{qs}>: indices {list(np.where(~mask)[0])}")

            bs_coeffs[qs] = np.polyfit(gain_vectors[qs][mask], bs_rates[qs][mask], deg=degree_rate)
            freq_coeffs[qs] = np.polyfit(gain_vectors[qs][mask], freq_vectors[qs][mask], deg=degree_freq)

        # Store to dataset
        if store:
            for qs in gain_vectors.keys():
                self.station.ds_storage.update_bs_rate_coeffs(stor_name, bs_coeffs[qs], qubit_state=qs)
                self.station.ds_storage.update_freq_coeffs(stor_name, freq_coeffs[qs], qubit_state=qs)
                gains = gain_vectors[qs]
                self.station.ds_storage.update_gain_range(stor_name, gains.min(), gains.max(), qubit_state=qs)
            self._log(f"  Stored polynomial coefficients and gain ranges for {stor_name}")

        # Plot results
        for qs in gain_vectors.keys():
            if skip_indices is None:
                mask = np.ones(len(gain_vectors[qs]), dtype=bool)
            elif isinstance(skip_indices, dict):
                mask = np.ones(len(gain_vectors[qs]), dtype=bool)
                mask[skip_indices.get(qs, [])] = False
            else:
                mask = np.ones(len(gain_vectors[qs]), dtype=bool)
                mask[skip_indices] = False

            gain_fit_range = np.linspace(gain_vectors[qs].min(), gain_vectors[qs].max(), 100)
            bs_fit = np.polyval(bs_coeffs[qs], gain_fit_range)
            freq_fit = np.polyval(freq_coeffs[qs], gain_fit_range)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            color = 'tab:blue' if qs == 'g' else 'tab:red'

            axes[0].plot(gain_vectors[qs][mask], bs_rates[qs][mask], 'o', color=color,
                        label=f'Data (|{qs}>)')
            axes[0].plot(gain_vectors[qs][~mask], bs_rates[qs][~mask], 'x', color='gray',
                        markersize=10, label='Skipped')
            axes[0].plot(gain_fit_range, bs_fit, '--', color='black',
                        label=f'Fit (deg {degree_rate})')
            axes[0].set_xlabel('Gain')
            axes[0].set_ylabel('Beam Splitter Rate (MHz)')
            axes[0].set_title(f'BS Rate vs Gain — qubit in |{qs}>')
            axes[0].legend()

            axes[1].plot(gain_vectors[qs][mask], freq_vectors[qs][mask], 'o', color=color,
                        label=f'Data (|{qs}>)')
            axes[1].plot(gain_vectors[qs][~mask], freq_vectors[qs][~mask], 'x', color='gray',
                        markersize=10, label='Skipped')
            axes[1].plot(gain_fit_range, freq_fit, '--', color='black',
                        label=f'Fit (deg {degree_freq})')
            axes[1].set_xlabel('Gain')
            axes[1].set_ylabel('Frequency (MHz)')
            axes[1].set_title(f'Frequency vs Gain — qubit in |{qs}>')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        self._log(f"  BS rate fit (degree {degree_rate}), freq fit (degree {degree_freq})")

        return {
            'bs_coeffs': bs_coeffs,
            'freq_coeffs': freq_coeffs,
            'bs_rates': bs_rates,
        }

    def calibrate_bs_rate_lookup_table(self, man_mode_no=1, stor_mode_no=1,
                                       qubit_states=('g', 'e'), num_pts=20,
                                       length_range_frac=(0.3, 1.2),
                                       reps=100, n_pulses=10, n_step=3,
                                       freq_span_initial=0.2, gain_span_frac=0.35,
                                       freq_span_final=0.1,
                                       degree_rate=2, degree_freq=10):
        """
        Build beam splitter rate lookup table by sweeping gain points.

        For each qubit state, sweeps over gain values (derived from a range of
        pi_lengths), runs a 3-step error amplification (freq -> gain -> freq) at
        each point, then fits polynomial models for bs_rate(gain) and freq(gain).

        The resulting polynomials are stored in the dataset and used by
        calibrate_beam_splitter_rate() to look up gain and frequency for a
        desired beam splitter rate.

        To refit with different polynomial degrees without re-running experiments:
            calibrator.fit_bs_rate_lookup_table(result, degree_rate=3, degree_freq=5)

        Parameters
        ----------
        man_mode_no : int
            Manipulate mode number (default: 1)
        stor_mode_no : int
            Storage mode number (default: 1)
        qubit_states : tuple of str
            Which qubit states to calibrate, e.g. ('g', 'e') or ('e',)
        num_pts : int
            Number of gain points to sweep (default: 20)
        length_range_frac : tuple of float
            (min, max) fraction of pi_length for sweep range (default: (0.3, 1.2))
        reps : int
            Repetitions per error amp measurement (default: 100)
        n_pulses : int
            Number of pulses in error amplification (default: 10)
        n_step : int
            Step size for n_pulses in error amp (default: 3)
        freq_span_initial : float
            Frequency span in MHz for the first freq error amp (default: 0.2)
        gain_span_frac : float
            Gain span as fraction of current gain (default: 0.35)
        freq_span_final : float
            Frequency span for final freq error amp (default: 0.1)
        degree_rate : int
            Polynomial degree for bs_rate vs gain fit (default: 2)
        degree_freq : int
            Polynomial degree for freq vs gain fit (default: 10)

        Returns
        -------
        dict with keys:
            'gain_vectors', 'length_vectors', 'freq_vectors': dict by qubit state
            'stor_name': str
            'bs_coeffs', 'freq_coeffs', 'bs_rates': from fit_bs_rate_lookup_table
            'experiments': nested dict of all experiment objects
        """
        self._log(f"=== Building BS rate lookup table ===")

        # Build sweep vectors
        vectors = self._build_gain_sweep_vectors(
            man_mode_no, stor_mode_no, num_pts, length_range_frac
        )
        stor_name = vectors['stor_name']
        gain_vectors = vectors['gain_vectors']
        length_vectors = vectors['length_vectors']
        freq_vectors = vectors['freq_vectors']

        # Save original dataset params
        original_freq = vectors['original_freq']
        original_gain = vectors['original_gain']
        original_pi_len = vectors['original_pi_len']

        all_experiments = {}

        try:
            for qs in qubit_states:
                self._log(f"\n--- Calibrating for qubit in |{qs}> ({num_pts} points) ---")
                all_experiments[qs] = {}

                for idx in range(num_pts):
                    gain_pt = gain_vectors[qs][idx]
                    len_pt = length_vectors[qs][idx]

                    # Warm-start: use previous point's freq for guess (except first point)
                    if idx == 0:
                        freq_guess = freq_vectors[qs][0]
                    else:
                        freq_guess = freq_vectors[qs][idx - 1]

                    self._log(f"\n  Point {idx+1}/{num_pts}: gain={gain_pt:.1f}, "
                              f"length={len_pt:.4f} us, freq_guess={freq_guess:.4f} MHz")

                    try:
                        result = self._calibrate_at_gain_point(
                            stor_name, freq_guess, gain_pt, len_pt, qs,
                            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                            reps=reps, n_pulses=n_pulses, n_step=n_step,
                            freq_span_initial=freq_span_initial,
                            gain_span_frac=gain_span_frac,
                            freq_span_final=freq_span_final,
                        )

                        # Update vectors with calibrated values
                        freq_vectors[qs][idx] = result['freq']
                        gain_vectors[qs][idx] = result['gain']
                        all_experiments[qs][idx] = result['experiments']

                        self._log(f"  -> freq={result['freq']:.4f}, gain={result['gain']:.1f}")

                    except Exception as e:
                        self._log(f"  ERROR at point {idx+1}: {e}")
                        # Keep the initial guess values for this point
                        all_experiments[qs][idx] = None

        finally:
            # Always restore original dataset params
            self.station.ds_storage.update_pi(stor_name, original_pi_len)
            self.station.ds_storage.update_gain(stor_name, int(original_gain))
            self.station.ds_storage.update_freq(stor_name, original_freq)
            self._log(f"\n  Restored original dataset params for {stor_name}")

        # Build the lookup data dict
        lookup_data = {
            'gain_vectors': gain_vectors,
            'length_vectors': length_vectors,
            'freq_vectors': freq_vectors,
            'stor_name': stor_name,
            'experiments': all_experiments,
        }

        # Store for later refitting
        self.last_lookup_data = lookup_data

        # Fit and store polynomials
        fit_result = self.fit_bs_rate_lookup_table(
            lookup_data, degree_rate=degree_rate, degree_freq=degree_freq
        )

        # Merge fit results into the return dict
        lookup_data.update(fit_result)

        self._log(f"\n=== BS rate lookup table complete ===")
        return lookup_data

    def calibrate_beam_splitter_rate(self, g_bs, man_mode_no=1, stor_mode_no=1,
                                      alpha_amplitude=1.0,
                                      reps_error_amp=50, reps_rabi=100, reps_wigner=100,
                                      error_amp_span=0.05, n_pulses=15,
                                      run_both_rabi_states=False,
                                      rabi_start=None, rabi_stop=None, rabi_expts=None,
                                      wigner_n_points=100,
                                      f_bs_g_override=None, f_bs_e_override=None,
                                      gain_override=None):
        """
        ALICE: Calibrate at a given beam splitter rate (WITH storage swap).

        Parameters
        ----------
        g_bs : float
            Beam splitter rate in MHz (used for length_bs = 0.25 / g_bs)
        alpha_amplitude : float
            Displacement amplitude for Wigner circle
        run_both_rabi_states : bool
            If True, run Rabi for both g and e states and average pi_length
        rabi_start, rabi_stop, rabi_expts : float, float, int
            Rabi sweep parameters (optional)
        wigner_n_points : int
            Number of points on the Wigner alpha circle (default: 100)
        gain_override : int, optional
            If provided, use this gain directly instead of looking it up from g_bs.

        Returns
        -------
        dict with calibration results
        """
        self._log(f"=== ALICE: Calibrating at g_bs={g_bs:.4f} MHz ===")

        stor_name = f'M{man_mode_no}-S{stor_mode_no}'

        # Step 1: Setup - get gain/freq from lookup tables
        if gain_override is not None:
            gain = gain_override
        else:
            gain_e = self.station.ds_storage.get_gain_at_bs_rate(stor_name, g_bs, 'e')
            gain_g = self.station.ds_storage.get_gain_at_bs_rate(stor_name, g_bs, 'g')
            gain = int((gain_e + gain_g) / 2)
        f_bs_g_guess = f_bs_g_override if f_bs_g_override is not None else self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'g')
        f_bs_e_guess = f_bs_e_override if f_bs_e_override is not None else self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'e')
        length_bs = 0.25 / g_bs

        self._log(f"  Initial: gain={gain}, length_bs={length_bs:.4f} us")
        self._log(f"  Freq guesses: f_bs_g={f_bs_g_guess:.4f}, f_bs_e={f_bs_e_guess:.4f} (override={'yes' if f_bs_g_override is not None else 'no'})")

        experiments = {}

        # Step 2: Error amplification for both states
        f_bs_g, experiments['error_amp_g'] = self._run_error_amp(
            stor_name, f_bs_g_guess, gain, length_bs, 'g',
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
            reps=reps_error_amp, n_pulses=n_pulses, span=error_amp_span
        )
        f_bs_e, experiments['error_amp_e'] = self._run_error_amp(
            stor_name, f_bs_e_guess, gain, length_bs, 'e',
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
            reps=reps_error_amp, n_pulses=n_pulses, span=error_amp_span
        )
        freq_bs = (f_bs_g + f_bs_e) / 2
        self._log(f"  Calibrated freq_bs={freq_bs:.4f} MHz")

        # Step 3: Rabi confirmation
        pi_length, experiments['rabi_g'], experiments['rabi_e'] = self._run_rabi(
            freq_bs, gain, length_bs, man_mode_no, stor_mode_no,
            reps=reps_rabi, run_both_states=run_both_rabi_states,
            start=rabi_start, stop=rabi_stop, expts=rabi_expts
        )
        self._log(f"  Calibrated pi_length={pi_length:.4f} us")

        # Step 4: ALICE Wigner tomography for both states
        theta_g, experiments['wigner_g'] = self._run_wigner_alice(
            freq_bs, gain, pi_length, 'g', alpha_amplitude,
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no, reps=reps_wigner,
            n_points=wigner_n_points
        )
        theta_e, experiments['wigner_e'] = self._run_wigner_alice(
            freq_bs, gain, pi_length, 'e', alpha_amplitude,
            man_mode_no=man_mode_no, stor_mode_no=stor_mode_no, reps=reps_wigner,
            n_points=wigner_n_points
        )

        phase_diff_pi = np.abs(theta_e - theta_g) / np.pi
        self._log(f"  theta_g={theta_g*180/np.pi:.2f} deg, theta_e={theta_e*180/np.pi:.2f} deg")
        self._log(f"  Phase difference: {phase_diff_pi:.4f} pi")

        return {
            'g_bs': g_bs,
            'freq_bs': freq_bs,
            'f_bs_g': f_bs_g,
            'f_bs_e': f_bs_e,
            'gain': gain,
            'length_bs': length_bs,
            'pi_length': pi_length,
            'theta_g': theta_g,
            'theta_e': theta_e,
            'phase_diff_pi': phase_diff_pi,
            'alpha_amplitude': alpha_amplitude,
            'man_mode_no': man_mode_no,
            'stor_mode_no': stor_mode_no,
            'success': True,
            'experiments': experiments,
        }

    def fit_gain_sweep(self, sweep_data, fit_degree=1, target_phase_diff=1.0,
                       target_phase_tol=0.1, skip_indices=None):
        """
        Fit phase difference vs gain from sweep data.

        Can be called independently to refit with different polynomial degrees
        or skip outlier points without re-running experiments.

        Parameters
        ----------
        sweep_data : dict
            Sweep data returned by sweep_beam_splitter_rate().
            Must contain 'results' key with list of per-gain calibration dicts,
            and 'bs_coeffs' for the secondary rate axis on the plot.
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)
        target_phase_diff : float
            Target phase difference in units of pi (default: 1.0)
        target_phase_tol : float
            Tolerance on phase difference in units of pi (default: 0.1)
        skip_indices : list of int, optional
            Indices of successful points to exclude from the fit.

        Returns
        -------
        dict with keys:
            'fit_params': polynomial coefficients from np.polyfit (highest degree first)
            'fit_degree': degree of polynomial used
            'fitted_optimal_gain': gain estimated from polynomial fit (int)
            'fit_figure': matplotlib figure of the fit
            'optimal_idx': index of best measured gain in full results list
            'optimal_gain': best measured gain
            'optimal_result': calibration result for best measured gain
            'converged': whether best measured gain is within tolerance
        """
        results = sweep_data['results']
        successful = [r for r in results if r.get('success', False)]

        if not successful:
            self._log("No successful calibrations to fit!")
            return {
                'fit_params': None,
                'fit_degree': fit_degree,
                'fitted_optimal_gain': None,
                'fit_figure': None,
                'optimal_idx': None,
                'optimal_gain': None,
                'optimal_result': None,
                'converged': False,
            }

        # Extract data — fit in gain space
        gains = np.array([r['gain'] for r in successful], dtype=float)
        phase_diffs = np.array([r['phase_diff_pi'] for r in successful])

        # Apply skip mask
        if skip_indices is None:
            mask = np.ones(len(successful), dtype=bool)
        else:
            mask = np.ones(len(successful), dtype=bool)
            mask[skip_indices] = False

        n_skipped = np.sum(~mask)
        if n_skipped > 0:
            self._log(f"  Skipping {n_skipped} points: indices {list(np.where(~mask)[0])}")

        # Find best measured result (among non-skipped points)
        masked_errors = np.full(len(successful), np.inf)
        masked_errors[mask] = np.abs(phase_diffs[mask] - target_phase_diff)
        optimal_idx = np.argmin(masked_errors)
        converged = masked_errors[optimal_idx] <= target_phase_tol
        original_idx = results.index(successful[optimal_idx])

        self._log(f"Best measured gain: {gains[optimal_idx]:.0f} "
                  f"(phase_diff={phase_diffs[optimal_idx]:.4f} pi)")

        # Polynomial fit in gain space
        fit_params = None
        fitted_optimal_gain = None
        fit_figure = None

        min_points_needed = fit_degree + 1
        n_fit_points = np.sum(mask)
        if n_fit_points >= min_points_needed:
            try:
                fit_params = np.polyfit(gains[mask], phase_diffs[mask], fit_degree)

                # Find gain where phase_diff = target_phase_diff
                root_coeffs = fit_params.copy()
                root_coeffs[-1] -= target_phase_diff

                roots = np.roots(root_coeffs)

                # Filter for real roots within a reasonable range of the data
                gain_min, gain_max = min(gains[mask]), max(gains[mask])
                gain_margin = (gain_max - gain_min) * 0.5
                valid_roots = []
                for root in roots:
                    if np.isreal(root):
                        root_real = np.real(root)
                        if (gain_min - gain_margin) <= root_real <= (gain_max + gain_margin):
                            valid_roots.append(root_real)

                if valid_roots:
                    gain_center = (gain_min + gain_max) / 2
                    fitted_optimal_gain = int(round(
                        min(valid_roots, key=lambda x: abs(x - gain_center))
                    ))

                    self._log(f"Polynomial fit (degree {fit_degree}): coeffs = {fit_params}")
                    self._log(f"Fitted optimal gain: {fitted_optimal_gain}")

                    bs_coeffs = sweep_data.get('bs_coeffs')
                    fit_figure = self._plot_gain_sweep_fit(
                        gains, phase_diffs, fit_params, fitted_optimal_gain,
                        target_phase_diff, target_phase_tol, fit_degree,
                        bs_coeffs=bs_coeffs
                    )
                else:
                    self._log("WARNING: No valid root found in data range, cannot estimate optimal gain")

            except Exception as e:
                self._log(f"ERROR during polynomial fit: {e}")

        else:
            self._log(f"WARNING: Not enough non-skipped points for degree-{fit_degree} fit "
                      f"(have {n_fit_points}, need >= {min_points_needed})")

        return {
            'fit_params': fit_params,
            'fit_degree': fit_degree,
            'fitted_optimal_gain': fitted_optimal_gain,
            'fit_figure': fit_figure,
            'optimal_idx': original_idx,
            'optimal_gain': int(gains[optimal_idx]),
            'optimal_result': successful[optimal_idx],
            'converged': converged,
        }

    def fit_gain_sweep_parity(self, sweep_data, fit_degree=1,
                              target_phase_diff_deg=180.0,
                              target_phase_tol_deg=18.0,
                              skip_indices=None):
        """
        Fit unwrapped phase difference vs gain from parity sweep data.

        Unlike fit_gain_sweep() which works with wrapped phase_diff_pi in [0,1],
        this method unwraps the raw phase differences so they are monotonic,
        then fits a polynomial and finds where it crosses the target (180 deg).

        Can be called independently to refit with different polynomial degrees
        or skip outlier points without re-running experiments.

        Parameters
        ----------
        sweep_data : dict
            From sweep_beam_splitter_rate_parity(). Must have 'results' and
            'phase_00_ref'.
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)
        target_phase_diff_deg : float
            Target phase difference in degrees (default: 180.0)
        target_phase_tol_deg : float
            Tolerance in degrees (default: 18.0, i.e. 0.1 pi)
        skip_indices : list of int, optional
            Indices of successful points to exclude from the fit.

        Returns
        -------
        dict with keys matching fit_gain_sweep() output:
            'fit_params', 'fit_degree', 'fitted_optimal_gain', 'fit_figure',
            'optimal_idx', 'optimal_gain', 'optimal_result', 'converged',
            'phase_01_optimal'
        Also includes 'phase_01_optimal' (float, degrees): the predicted
        phase_01 at the fitted optimal gain, computed from the polynomial fit
        as phase_00_ref + polyval(fit_params, fitted_optimal_gain).
        """
        results = sweep_data['results']
        phase_00_ref = sweep_data['phase_00_ref']
        successful = [r for r in results if r.get('success', False)]

        if not successful:
            self._log("No successful calibrations to fit!")
            return {
                'fit_params': None, 'fit_degree': fit_degree,
                'fitted_optimal_gain': None, 'fit_figure': None,
                'optimal_idx': None, 'optimal_gain': None,
                'optimal_result': None, 'converged': False,
            }

        # Extract gains and raw phase_01 values
        gains = np.array([r['gain'] for r in successful], dtype=float)
        phases_01_raw = np.array([r['phase_01'] for r in successful])

        # Compute raw differences and unwrap to make monotonic
        raw_diffs = phases_01_raw - phase_00_ref
        unwrapped_diffs = np.rad2deg(np.unwrap(np.deg2rad(raw_diffs)))

        # Determine sign: the unwrapped diff should approach +180 or -180
        # Pick the target sign that matches the data trend
        mean_diff = np.mean(unwrapped_diffs)
        if mean_diff < 0:
            target = -target_phase_diff_deg
        else:
            target = target_phase_diff_deg
        # Also allow the target to be shifted by 360*k to match the data range
        k = round((mean_diff - target) / 360)
        target = target + 360 * k

        self._log(f"  Unwrapped diffs range: [{unwrapped_diffs.min():.1f}, "
                  f"{unwrapped_diffs.max():.1f}] deg, target={target:.1f} deg")

        # Apply skip mask
        if skip_indices is None:
            mask = np.ones(len(successful), dtype=bool)
        else:
            mask = np.ones(len(successful), dtype=bool)
            mask[skip_indices] = False

        n_skipped = np.sum(~mask)
        if n_skipped > 0:
            self._log(f"  Skipping {n_skipped} points: indices {list(np.where(~mask)[0])}")

        # Find best measured result (closest to target among non-skipped)
        masked_errors = np.full(len(successful), np.inf)
        masked_errors[mask] = np.abs(unwrapped_diffs[mask] - target)
        optimal_idx = np.argmin(masked_errors)
        converged = masked_errors[optimal_idx] <= target_phase_tol_deg
        original_idx = results.index(successful[optimal_idx])

        self._log(f"Best measured gain: {gains[optimal_idx]:.0f} "
                  f"(unwrapped_diff={unwrapped_diffs[optimal_idx]:.1f} deg)")

        # Polynomial fit on unwrapped differences
        fit_params = None
        fitted_optimal_gain = None
        phase_01_optimal = None
        fit_figure = None

        min_points_needed = fit_degree + 1
        n_fit_points = np.sum(mask)
        if n_fit_points >= min_points_needed:
            try:
                fit_params = np.polyfit(gains[mask], unwrapped_diffs[mask], fit_degree)

                # Find gain where unwrapped_diff = target
                root_coeffs = fit_params.copy()
                root_coeffs[-1] -= target

                roots = np.roots(root_coeffs)

                # Filter for real roots within a reasonable range
                gain_min, gain_max = min(gains[mask]), max(gains[mask])
                gain_margin = (gain_max - gain_min) * 0.5
                valid_roots = []
                for root in roots:
                    if np.isreal(root):
                        root_real = np.real(root)
                        if (gain_min - gain_margin) <= root_real <= (gain_max + gain_margin):
                            valid_roots.append(root_real)

                if valid_roots:
                    gain_center = (gain_min + gain_max) / 2
                    fitted_optimal_gain = int(round(
                        min(valid_roots, key=lambda x: abs(x - gain_center))
                    ))

                    # Compute predicted phase_01 at optimal gain from the fit
                    fitted_phase_diff = np.polyval(fit_params, fitted_optimal_gain)
                    phase_01_optimal = phase_00_ref + fitted_phase_diff
                    # Wrap to [-180, 180]
                    phase_01_optimal = ((phase_01_optimal + 180) % 360) - 180

                    self._log(f"Polynomial fit (degree {fit_degree}): coeffs = {fit_params}")
                    self._log(f"Fitted optimal gain: {fitted_optimal_gain}")
                    self._log(f"Predicted phase_01 at optimal: {phase_01_optimal:.1f} deg "
                              f"(phase_00_ref={phase_00_ref:.1f}, fitted_diff={fitted_phase_diff:.1f})")

                    bs_coeffs = sweep_data.get('bs_coeffs')
                    fit_figure = self._plot_gain_sweep_parity_fit(
                        gains, unwrapped_diffs, fit_params, fitted_optimal_gain,
                        target, target_phase_tol_deg, fit_degree,
                        bs_coeffs=bs_coeffs
                    )
                else:
                    self._log("WARNING: No valid root found in data range")

            except Exception as e:
                self._log(f"ERROR during polynomial fit: {e}")
        else:
            self._log(f"WARNING: Not enough non-skipped points for degree-{fit_degree} fit "
                      f"(have {n_fit_points}, need >= {min_points_needed})")

        return {
            'fit_params': fit_params,
            'fit_degree': fit_degree,
            'fitted_optimal_gain': fitted_optimal_gain,
            'fit_figure': fit_figure,
            'optimal_idx': original_idx,
            'optimal_gain': int(gains[optimal_idx]),
            'optimal_result': successful[optimal_idx],
            'converged': converged,
            'phase_01_optimal': phase_01_optimal,
        }

    def _plot_gain_sweep_parity_fit(self, gains, unwrapped_diffs, fit_params,
                                     fitted_optimal_gain, target_deg,
                                     target_tol_deg, fit_degree=1,
                                     bs_coeffs=None):
        """
        Plot unwrapped phase difference vs gain with polynomial fit.

        Parameters
        ----------
        gains : array
            Beam splitter gains
        unwrapped_diffs : array
            Unwrapped phase differences (degrees)
        fit_params : array
            Polynomial coefficients from np.polyfit
        fitted_optimal_gain : int
            Estimated optimal gain from fit
        target_deg : float
            Target phase difference (degrees, with sign)
        target_tol_deg : float
            Tolerance (degrees)
        fit_degree : int
            Degree of polynomial fit
        bs_coeffs : array, optional
            Polynomial coefficients for gain→rate lookup (secondary axis)

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.plot(gains, unwrapped_diffs, 'o', markersize=10, label='Measured', zorder=5)

        # Plot fit curve
        gain_range = np.linspace(min(gains), max(gains), 100)
        fit_line = np.polyval(fit_params, gain_range)
        fit_label = f'Polynomial fit (degree {fit_degree})'
        ax.plot(gain_range, fit_line, '-', color='blue', alpha=0.7, label=fit_label)

        # Plot target line and tolerance band
        ax.axhline(target_deg, color='r', linestyle='--',
                   label=f'Target ({target_deg:.0f} deg)')
        ax.axhspan(target_deg - target_tol_deg, target_deg + target_tol_deg,
                   color='r', alpha=0.1, label=f'+/-{target_tol_deg:.0f} deg tolerance')

        # Mark fitted optimal gain
        if fitted_optimal_gain is not None:
            fitted_phase = np.polyval(fit_params, fitted_optimal_gain)
            ax.scatter([fitted_optimal_gain], [fitted_phase], color='green', s=200,
                       marker='*', zorder=10,
                       label=f'Fitted optimal gain: {fitted_optimal_gain}')
            ax.axvline(fitted_optimal_gain, color='green', linestyle=':', alpha=0.5)

        ax.set_xlabel('Gain', fontsize=12)
        ax.set_ylabel('Phase Difference (degrees, unwrapped)', fontsize=12)
        title = f'ALICE (parity): Gain Sweep (Degree-{fit_degree} Fit)'
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add secondary x-axis showing theoretical beam splitter rate
        if bs_coeffs is not None:
            gains_sorted = np.sort(gains)
            rates_sorted = np.polyval(bs_coeffs, gains_sorted)

            def gain_to_rate(g):
                return np.interp(g, gains_sorted, rates_sorted)

            def rate_to_gain(r):
                return np.interp(r, rates_sorted, gains_sorted)

            ax2 = ax.secondary_xaxis('top', functions=(gain_to_rate, rate_to_gain))
            ax2.set_xlabel('Theoretical Rate (MHz)', fontsize=12)

        plt.tight_layout()
        plt.show()

        return fig

    def calibrate_at_fitted_gain(self, sweep_data, fitted_optimal_gain,
                                man_mode_no=1, stor_mode_no=1,
                                reps_error_amp=50, n_pulses=15, error_amp_span=0.05,
                                reps_rabi=100, run_both_rabi_states=False,
                                rabi_start=None, rabi_stop=None, rabi_expts=None):
        """
        Run error amplification and Rabi at a fitted optimal gain.

        Can be called independently after fit_gain_sweep() to calibrate at a
        new fitted gain without re-running the full sweep.

        Parameters
        ----------
        sweep_data : dict
            Sweep data from sweep_beam_splitter_rate(), must contain 'results'.
        fitted_optimal_gain : int
            Gain at which to run calibration.
        man_mode_no : int
            Manipulate mode number (default: 1)
        stor_mode_no : int
            Storage mode number (default: 1)
        reps_error_amp : int
            Repetitions for error amplification (default: 50)
        n_pulses : int
            Number of pulses in error amplification (default: 15)
        error_amp_span : float
            Frequency span for error amplification (default: 0.05)
        reps_rabi : int
            Repetitions for Rabi measurement (default: 100)
        run_both_rabi_states : bool
            Whether to run Rabi for both qubit states (default: False)
        rabi_start : float, optional
            Start of Rabi sweep range
        rabi_stop : float, optional
            End of Rabi sweep range
        rabi_expts : int, optional
            Number of Rabi experiments

        Returns
        -------
        dict with calibration results:
            'gain', 'freq_bs', 'f_bs_g', 'f_bs_e', 'g_bs', 'length_bs',
            'pi_length', 'man_mode_no', 'stor_mode_no', 'experiments'
        Returns None if calibration fails.
        """
        self._log(f"\n=== Calibrating at fitted optimal gain: {fitted_optimal_gain} ===")
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'

        try:
            gain = fitted_optimal_gain

            # Get theoretical rate and length from lookup table
            rate_g = self.station.ds_storage.get_bs_rate_at_gain(stor_name, gain, 'g')
            rate_e = self.station.ds_storage.get_bs_rate_at_gain(stor_name, gain, 'e')
            g_bs = (rate_g + rate_e) / 2
            length_bs = 0.25 / g_bs
            self._log(f"  Theoretical rate: {g_bs:.4f} MHz, length_bs: {length_bs:.4f} us")

            # Use frequencies from the closest swept point as initial guess
            results = sweep_data['results']
            successful = [r for r in results if r.get('success', False)]
            gains = np.array([r['gain'] for r in successful], dtype=float)
            closest_idx = np.argmin(np.abs(gains - fitted_optimal_gain))
            f_bs_g_guess = successful[closest_idx]['f_bs_g']
            f_bs_e_guess = successful[closest_idx]['f_bs_e']
            self._log(f"  Using freq guess from closest sweep point (gain={gains[closest_idx]:.0f}): "
                      f"f_bs_g={f_bs_g_guess:.4f}, f_bs_e={f_bs_e_guess:.4f}")

            fitted_experiments = {}

            # Error amplification for both states
            f_bs_g, fitted_experiments['error_amp_g'] = self._run_error_amp(
                stor_name, f_bs_g_guess, gain, length_bs, 'g',
                man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                reps=reps_error_amp, n_pulses=n_pulses, span=error_amp_span
            )
            f_bs_e, fitted_experiments['error_amp_e'] = self._run_error_amp(
                stor_name, f_bs_e_guess, gain, length_bs, 'e',
                man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                reps=reps_error_amp, n_pulses=n_pulses, span=error_amp_span
            )
            freq_bs = (f_bs_g + f_bs_e) / 2
            self._log(f"  Calibrated freq_bs={freq_bs:.4f} MHz")

            # Rabi confirmation
            pi_length, fitted_experiments['rabi_g'], fitted_experiments['rabi_e'] = self._run_rabi(
                freq_bs, gain, length_bs, man_mode_no, stor_mode_no,
                reps=reps_rabi, run_both_states=run_both_rabi_states,
                start=rabi_start, stop=rabi_stop, expts=rabi_expts
            )
            self._log(f"  Calibrated pi_length={pi_length:.4f} us")

            return {
                'gain': gain,
                'g_bs': g_bs,
                'freq_bs': freq_bs,
                'f_bs_g': f_bs_g,
                'f_bs_e': f_bs_e,
                'length_bs': length_bs,
                'pi_length': pi_length,
                'man_mode_no': man_mode_no,
                'stor_mode_no': stor_mode_no,
                'experiments': fitted_experiments,
            }

        except Exception as e:
            self._log(f"ERROR during fitted calibration: {e}")
            return None

    def sweep_beam_splitter_rate(self, gain_list, target_phase_diff=1.0,
                                  target_phase_tol=0.1, fit_degree=1,
                                  f_bs_g_guess=None, f_bs_e_guess=None, **kwargs):
        """
        ALICE: Sweep beam splitter gains to find optimal one.

        Performs a sweep over gains, fits phase difference vs gain to a
        polynomial, estimates the optimal gain from the fit, and runs a final
        calibration (error amp + Rabi) at the fitted optimal gain.

        Parameters
        ----------
        gain_list : array-like
            List of gains to test (integer DAC values)
        target_phase_diff : float
            Target phase difference in units of pi (default: 1.0)
        target_phase_tol : float
            Tolerance on phase difference in units of pi (default: 0.1)
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)
        f_bs_g_guess : float, optional
            Initial frequency guess for |g> state for the first sweep point.
            If None, uses the lookup table.
        f_bs_e_guess : float, optional
            Initial frequency guess for |e> state for the first sweep point.
            If None, uses the lookup table.

        Returns
        -------
        dict with sweep results including:
            - results: list of calibration results for each gain
            - optimal_idx: index of best measured gain
            - optimal_gain: best measured gain
            - optimal_result: calibration result for best measured gain
            - converged: whether best measured gain is within tolerance
            - fit_params: polynomial coefficients from np.polyfit (highest degree first)
            - fit_degree: degree of polynomial used
            - fitted_optimal_gain: gain estimated from polynomial fit
            - fitted_calibration: error amp + Rabi results at fitted gain
            - fit_figure: matplotlib figure of the fit
            - bs_coeffs: lookup table coefficients for gain→rate mapping
        """
        self._log(f"=== ALICE: Sweeping {len(gain_list)} gains ===")

        # Filter out class-level settings that shouldn't be passed to calibrate_beam_splitter_rate
        kwargs.pop('debug', None)

        # Extract kwargs for later use in fitted calibration
        man_mode_no = kwargs.get('man_mode_no', 1)
        stor_mode_no = kwargs.get('stor_mode_no', 1)
        reps_error_amp = kwargs.get('reps_error_amp', 50)
        reps_rabi = kwargs.get('reps_rabi', 100)
        error_amp_span = kwargs.get('error_amp_span', 0.05)
        n_pulses = kwargs.get('n_pulses', 15)
        run_both_rabi_states = kwargs.get('run_both_rabi_states', False)
        rabi_start = kwargs.get('rabi_start', None)
        rabi_stop = kwargs.get('rabi_stop', None)
        rabi_expts = kwargs.get('rabi_expts', None)

        stor_name = f'M{man_mode_no}-S{stor_mode_no}'

        # Get bs_coeffs for gain→rate mapping (used in plot secondary axis)
        bs_coeffs = self.station.ds_storage.get_bs_rate_coeffs(stor_name, qubit_state='g')

        results = []
        prev_f_bs_g = f_bs_g_guess
        prev_f_bs_e = f_bs_e_guess
        for i, gain in enumerate(gain_list):
            gain = int(gain)

            # Get theoretical rate from lookup table
            rate_g = self.station.ds_storage.get_bs_rate_at_gain(stor_name, gain, 'g')
            rate_e = self.station.ds_storage.get_bs_rate_at_gain(stor_name, gain, 'e')
            g_bs = (rate_g + rate_e) / 2

            self._log(f"\nGain {i+1}/{len(gain_list)}: gain={gain}, "
                      f"theoretical rate={g_bs:.4f} MHz")
            try:
                result = self.calibrate_beam_splitter_rate(
                    g_bs, gain_override=gain,
                    f_bs_g_override=prev_f_bs_g, f_bs_e_override=prev_f_bs_e,
                    **kwargs
                )
                results.append(result)
                # Use fitted frequencies as guess for the next point
                if result.get('success', False):
                    prev_f_bs_g = result['f_bs_g']
                    prev_f_bs_e = result['f_bs_e']
            except Exception as e:
                self._log(f"  ERROR: {e}")
                results.append({'gain': gain, 'g_bs': g_bs, 'success': False, 'error': str(e)})

        # Build sweep data and store for later refitting
        sweep_data = {'results': results, 'bs_coeffs': bs_coeffs}
        self.last_rate_sweep_data = sweep_data

        # Fit phase_diff vs gain using the refittable method
        fit_result = self.fit_gain_sweep(
            sweep_data, fit_degree=fit_degree,
            target_phase_diff=target_phase_diff,
            target_phase_tol=target_phase_tol,
        )

        # Merge fit results into return dict
        sweep_data.update(fit_result)

        # Run error amplification and Rabi at the fitted optimal gain (if found)
        fitted_calibration = None
        fitted_optimal_gain = fit_result.get('fitted_optimal_gain')
        if fitted_optimal_gain is not None:
            fitted_calibration = self.calibrate_at_fitted_gain(
                sweep_data, fitted_optimal_gain,
                man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                reps_error_amp=reps_error_amp, n_pulses=n_pulses,
                error_amp_span=error_amp_span, reps_rabi=reps_rabi,
                run_both_rabi_states=run_both_rabi_states,
                rabi_start=rabi_start, rabi_stop=rabi_stop, rabi_expts=rabi_expts,
            )

        sweep_data['fitted_calibration'] = fitted_calibration
        return sweep_data

    def sweep_beam_splitter_rate_parity(self, gain_list, storage_swap,
                                        storage_parity=None,
                                        target_phase_diff_deg=180.0,
                                        target_phase_tol_deg=18.0,
                                        fit_degree=1,
                                        f_bs_g_guess=None, f_bs_e_guess=None,
                                        **kwargs):
        """
        ALICE (parity): Sweep beam splitter gains using ParityPhaseExperiment.

        Alternative to sweep_beam_splitter_rate() that uses ParityPhaseExperiment
        instead of Wigner tomography. Runs state '00' once as a fixed reference,
        then sweeps '01' at each gain point to measure the phase difference.

        The raw phase differences are unwrapped to produce a monotonic curve,
        then fitted with a polynomial to find the gain where the phase difference
        crosses 180 degrees (pi).

        Parameters
        ----------
        gain_list : array-like
            List of gains to test (integer DAC values)
        storage_swap : int
            Storage mode to swap to/from manipulate in ParityPhaseExperiment
        storage_parity : int, optional
            Storage mode for joint parity. Defaults to stor_mode_no.
        target_phase_diff_deg : float
            Target phase difference in degrees (default: 180.0)
        target_phase_tol_deg : float
            Tolerance in degrees (default: 18.0, i.e. 0.1 pi)
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)
        f_bs_g_guess : float, optional
            Initial frequency guess for |g> state for the first sweep point.
        f_bs_e_guess : float, optional
            Initial frequency guess for |e> state for the first sweep point.

        Keyword Arguments
        -----------------
        man_mode_no : int
            Manipulate mode number (default: 1)
        stor_mode_no : int
            Storage mode number for BS calibration (default: 1)
        reps_error_amp : int
            Repetitions for error amplification (default: 50)
        reps_rabi : int
            Repetitions for Rabi (default: 100)
        reps_parity : int
            Repetitions for ParityPhaseExperiment (default: 200)
        error_amp_span : float
            Frequency sweep span for error amp (default: 0.05)
        n_pulses : int
            Number of pulses in error amp (default: 15)
        run_both_rabi_states : bool
            Run Rabi for both g and e states (default: False)
        rabi_start, rabi_stop, rabi_expts : float, float, int
            Rabi sweep parameters (optional)
        parity_fast : bool
            Use fast multiphoton hpi pulses (default: False)
        parity_wait_time : float
            Wait time in ParityPhaseExperiment (default: 0.0)

        Returns
        -------
        dict with sweep results, same structure as sweep_beam_splitter_rate()
        plus 'phase_01_optimal' (float, degrees) for use by calibrate_wait_time_parity()
        """
        self._log(f"=== ALICE (parity): Sweeping {len(gain_list)} gains ===")

        # Extract kwargs
        kwargs.pop('debug', None)
        man_mode_no = kwargs.get('man_mode_no', 1)
        stor_mode_no = kwargs.get('stor_mode_no', 1)
        reps_error_amp = kwargs.get('reps_error_amp', 50)
        reps_rabi = kwargs.get('reps_rabi', 100)
        reps_parity = kwargs.get('reps_parity', 200)
        error_amp_span = kwargs.get('error_amp_span', 0.05)
        n_pulses = kwargs.get('n_pulses', 15)
        run_both_rabi_states = kwargs.get('run_both_rabi_states', False)
        rabi_start = kwargs.get('rabi_start', None)
        rabi_stop = kwargs.get('rabi_stop', None)
        rabi_expts = kwargs.get('rabi_expts', None)
        parity_fast = kwargs.get('parity_fast', False)
        parity_wait_time = kwargs.get('parity_wait_time', 0.0)

        if storage_parity is None:
            storage_parity = stor_mode_no

        stor_name = f'M{man_mode_no}-S{stor_mode_no}'
        stor_parity_name = f'M{man_mode_no}-S{storage_parity}'

        # Get bs_coeffs for gain→rate mapping (used in plot secondary axis)
        bs_coeffs = self.station.ds_storage.get_bs_rate_coeffs(stor_name, qubit_state='g')

        # Save original joint parity params for restore
        original_jp = self.station.ds_storage.get_joint_parity(stor_parity_name)

        # If no joint parity params exist yet, initialize from the first gain point
        if original_jp is None:
            first_gain = int(gain_list[0])
            rate_g = self.station.ds_storage.get_bs_rate_at_gain(stor_name, first_gain, 'g')
            rate_e = self.station.ds_storage.get_bs_rate_at_gain(stor_name, first_gain, 'e')
            g_bs_init = (rate_g + rate_e) / 2
            length_init = 0.5 / g_bs_init  # 2 * pi_length
            freq_init = self.station.ds_storage.get_freq_at_gain(stor_name, first_gain, 'g')
            self.station.ds_storage.update_joint_parity(
                stor_parity_name, freq_init, first_gain, length_init, parity_wait_time
            )
            self._log(f"  Initialized joint parity params for {stor_parity_name} "
                      f"(freq={freq_init:.4f}, gain={first_gain}, length={length_init:.4f})")

        results = []
        phase_00_ref = None

        try:
            # --- Run '00' reference once (gain-independent) ---
            # Use current ds_storage params for the reference measurement
            self._log("Running '00' reference measurement...")
            ref_phases, _ = self._run_parity_phase(
                storage_swap=storage_swap, storage_parity=storage_parity,
                state_start='00', man_mode_no=man_mode_no,
                reps=reps_parity, parity_fast=parity_fast,
            )
            if '00' in ref_phases:
                phase_00_ref = ref_phases['00']
                self._log(f"  Reference phase_00 = {phase_00_ref:.1f} deg")
            else:
                raise RuntimeError("Failed to extract phase for '00' reference")

            # --- Sweep gains, running only '01' at each point ---
            prev_f_bs_g = f_bs_g_guess
            prev_f_bs_e = f_bs_e_guess
            for i, gain in enumerate(gain_list):
                gain = int(gain)

                # Get theoretical rate from lookup table
                rate_g = self.station.ds_storage.get_bs_rate_at_gain(stor_name, gain, 'g')
                rate_e = self.station.ds_storage.get_bs_rate_at_gain(stor_name, gain, 'e')
                g_bs = (rate_g + rate_e) / 2
                length_bs = 0.25 / g_bs

                self._log(f"\nGain {i+1}/{len(gain_list)}: gain={gain}, "
                          f"theoretical rate={g_bs:.4f} MHz")

                try:
                    # Step 1: Error amplification for both states
                    f_bs_g_init = prev_f_bs_g if prev_f_bs_g is not None else \
                        self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'g')
                    f_bs_e_init = prev_f_bs_e if prev_f_bs_e is not None else \
                        self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'e')

                    f_bs_g, _ = self._run_error_amp(
                        stor_name, f_bs_g_init, gain, length_bs, 'g',
                        man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                        reps=reps_error_amp, n_pulses=n_pulses, span=error_amp_span
                    )
                    f_bs_e, _ = self._run_error_amp(
                        stor_name, f_bs_e_init, gain, length_bs, 'e',
                        man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                        reps=reps_error_amp, n_pulses=n_pulses, span=error_amp_span
                    )
                    freq_bs = (f_bs_g + f_bs_e) / 2
                    self._log(f"  Calibrated freq_bs={freq_bs:.4f} MHz")

                    # Step 2: Rabi confirmation
                    pi_length, _, _ = self._run_rabi(
                        freq_bs, gain, length_bs, man_mode_no, stor_mode_no,
                        reps=reps_rabi, run_both_states=run_both_rabi_states,
                        start=rabi_start, stop=rabi_stop, expts=rabi_expts
                    )
                    self._log(f"  Calibrated pi_length={pi_length:.4f} us")

                    # Step 3: Update joint parity params and run parity phase with '01'
                    self.station.ds_storage.update_joint_parity(
                        stor_parity_name, freq_bs, gain,
                        pi_length * 2, parity_wait_time
                    )

                    phases, _ = self._run_parity_phase(
                        storage_swap=storage_swap, storage_parity=storage_parity,
                        state_start='01', man_mode_no=man_mode_no,
                        reps=reps_parity, parity_fast=parity_fast,
                    )
                    phase_01 = phases['01']

                    # Compute phase difference
                    phase_diff_deg = self._angular_diff_deg(phase_01, phase_00_ref)
                    phase_diff_pi = abs(phase_diff_deg) / 180.0
                    self._log(f"  phase_01={phase_01:.1f} deg, "
                              f"phase_diff={phase_diff_deg:.1f} deg = {phase_diff_pi:.4f} pi")

                    results.append({
                        'gain': gain,
                        'g_bs': g_bs,
                        'freq_bs': freq_bs,
                        'f_bs_g': f_bs_g,
                        'f_bs_e': f_bs_e,
                        'pi_length': pi_length,
                        'length_bs': length_bs,
                        'phase_00_ref': phase_00_ref,
                        'phase_01': phase_01,
                        'phase_diff_deg': phase_diff_deg,
                        'phase_diff_pi': phase_diff_pi,
                        'alpha_amplitude': None,
                        'man_mode_no': man_mode_no,
                        'stor_mode_no': stor_mode_no,
                        'success': True,
                    })

                    # Warm-start next point
                    prev_f_bs_g = f_bs_g
                    prev_f_bs_e = f_bs_e

                except Exception as e:
                    self._log(f"  ERROR: {e}")
                    results.append({
                        'gain': gain, 'g_bs': g_bs,
                        'success': False, 'error': str(e),
                    })

        finally:
            # Restore original joint parity params
            if original_jp is not None:
                self.station.ds_storage.update_joint_parity(
                    stor_parity_name, *original_jp
                )
                self._log(f"  Restored original joint parity params for {stor_parity_name}")

        # Build sweep data and store for later refitting
        sweep_data = {
            'results': results,
            'bs_coeffs': bs_coeffs,
            'storage_swap': storage_swap,
            'phase_00_ref': phase_00_ref,
        }
        self.last_rate_sweep_parity_data = sweep_data

        # Fit unwrapped phase diff vs gain
        fit_result = self.fit_gain_sweep_parity(
            sweep_data, fit_degree=fit_degree,
            target_phase_diff_deg=target_phase_diff_deg,
            target_phase_tol_deg=target_phase_tol_deg,
        )
        sweep_data.update(fit_result)

        # Use phase_01_optimal from the fit (predicted from polynomial)
        phase_01_optimal = fit_result.get('phase_01_optimal')

        # Run error amplification and Rabi at the fitted optimal gain (if found)
        fitted_calibration = None
        fitted_optimal_gain = fit_result.get('fitted_optimal_gain')
        if fitted_optimal_gain is not None:
            fitted_calibration = self.calibrate_at_fitted_gain(
                sweep_data, fitted_optimal_gain,
                man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                reps_error_amp=reps_error_amp, n_pulses=n_pulses,
                error_amp_span=error_amp_span, reps_rabi=reps_rabi,
                run_both_rabi_states=run_both_rabi_states,
                rabi_start=rabi_start, rabi_stop=rabi_stop, rabi_expts=rabi_expts,
            )

            if fitted_calibration is not None:
                fitted_calibration['phase_01_optimal'] = phase_01_optimal
                fitted_calibration['stor_mode_no'] = stor_mode_no

        sweep_data['fitted_calibration'] = fitted_calibration
        sweep_data['phase_01_optimal'] = phase_01_optimal
        return sweep_data

    def _plot_gain_sweep_fit(self, gains, phase_diffs, fit_params, fitted_optimal_gain,
                              target_phase_diff, target_phase_tol, fit_degree=1,
                              bs_coeffs=None):
        """
        Plot phase difference vs gain with polynomial fit and secondary rate axis.

        Parameters
        ----------
        gains : array
            Beam splitter gains
        phase_diffs : array
            Measured phase differences (units of pi)
        fit_params : array
            Polynomial coefficients from np.polyfit (highest degree first)
        fitted_optimal_gain : int
            Estimated optimal gain from fit
        target_phase_diff : float
            Target phase difference (units of pi)
        target_phase_tol : float
            Tolerance on phase difference (units of pi)
        fit_degree : int
            Degree of polynomial fit
        bs_coeffs : array, optional
            Polynomial coefficients for gain→rate lookup (for secondary axis)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.plot(gains, phase_diffs, 'o', markersize=10, label='Measured', zorder=5)

        # Plot fit curve
        gain_range = np.linspace(min(gains), max(gains), 100)
        fit_line = np.polyval(fit_params, gain_range)

        fit_label = f'Polynomial fit (degree {fit_degree})'
        ax.plot(gain_range, fit_line, '-', color='blue', alpha=0.7, label=fit_label)

        # Plot target line and tolerance band
        ax.axhline(target_phase_diff, color='r', linestyle='--', label=f'Target ({target_phase_diff} pi)')
        ax.axhspan(target_phase_diff - target_phase_tol, target_phase_diff + target_phase_tol,
                   color='r', alpha=0.1, label=f'+/-{target_phase_tol} pi tolerance')

        # Mark fitted optimal gain
        fitted_phase = np.polyval(fit_params, fitted_optimal_gain)
        ax.scatter([fitted_optimal_gain], [fitted_phase], color='green', s=200,
                   marker='*', zorder=10, label=f'Fitted optimal gain: {fitted_optimal_gain}')
        ax.axvline(fitted_optimal_gain, color='green', linestyle=':', alpha=0.5)

        ax.set_xlabel('Gain', fontsize=12)
        ax.set_ylabel('Phase Difference (pi)', fontsize=12)
        title = f'ALICE: Gain Sweep (Degree-{fit_degree} Fit)'
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add secondary x-axis showing theoretical beam splitter rate
        if bs_coeffs is not None:
            gains_sorted = np.sort(gains)
            rates_sorted = np.polyval(bs_coeffs, gains_sorted)

            def gain_to_rate(g):
                return np.interp(g, gains_sorted, rates_sorted)

            def rate_to_gain(r):
                return np.interp(r, rates_sorted, gains_sorted)

            ax2 = ax.secondary_xaxis('top', functions=(gain_to_rate, rate_to_gain))
            ax2.set_xlabel('Theoretical Rate (MHz)', fontsize=12)

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_wait_sweep_fit(self, wait_times, phase_diffs, fit_params, fitted_optimal_wait,
                              target_phase_diff, target_phase_tol, fit_degree=1):
        """
        Plot phase difference vs wait time with polynomial fit.

        Parameters
        ----------
        wait_times : array
            Wait times (us)
        phase_diffs : array
            Measured phase differences (units of pi)
        fit_params : array
            Polynomial coefficients from np.polyfit (highest degree first)
        fitted_optimal_wait : float
            Estimated optimal wait time from fit
        target_phase_diff : float
            Target phase difference (units of pi)
        target_phase_tol : float
            Tolerance on phase difference (units of pi)
        fit_degree : int
            Degree of polynomial fit

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.plot(wait_times, phase_diffs, 'o', markersize=10, label='Measured', zorder=5)

        # Plot fit curve
        wait_range = np.linspace(min(wait_times), max(wait_times), 100)
        fit_line = np.polyval(fit_params, wait_range)

        # Build fit label
        if fit_degree == 1:
            fit_label = f'Fit: y = {fit_params[0]:.4f}x + {fit_params[1]:.4f}'
        else:
            fit_label = f'Polynomial fit (degree {fit_degree})'

        ax.plot(wait_range, fit_line, '-', color='blue', alpha=0.7, label=fit_label)

        # Plot target line and tolerance band
        ax.axhline(target_phase_diff, color='r', linestyle='--', label=f'Target ({target_phase_diff} pi)')
        ax.axhspan(target_phase_diff - target_phase_tol, target_phase_diff + target_phase_tol,
                   color='r', alpha=0.1, label=f'+/-{target_phase_tol} pi tolerance')

        # Mark fitted optimal wait time
        fitted_phase = np.polyval(fit_params, fitted_optimal_wait)
        ax.scatter([fitted_optimal_wait], [fitted_phase], color='green', s=200,
                   marker='*', zorder=10, label=f'Fitted optimal: {fitted_optimal_wait:.4f} us')
        ax.axvline(fitted_optimal_wait, color='green', linestyle=':', alpha=0.5)

        ax.set_xlabel('Wait Time (us)', fontsize=12)
        ax.set_ylabel('Phase Difference (pi)', fontsize=12)
        title = 'BOB: Wait Time Sweep'
        if fit_degree == 1:
            title += ' (Linear Fit)'
        else:
            title += f' (Degree-{fit_degree} Polynomial Fit)'
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_wait_sweep_parity_fit(self, wait_times, phases_deg, fit_params,
                                     fitted_optimal_wait, target_phase,
                                     target_phase_tol, fit_degree=1):
        """
        Plot phase_10 (degrees) vs wait time with polynomial fit.

        Parameters
        ----------
        wait_times : array
            Wait times (us)
        phases_deg : array
            Measured phase_10 values (degrees, unwrapped)
        fit_params : array
            Polynomial coefficients from np.polyfit
        fitted_optimal_wait : float
            Estimated optimal wait time from fit
        target_phase : float
            Target phase (degrees)
        target_phase_tol : float
            Tolerance on phase (degrees)
        fit_degree : int
            Degree of polynomial fit

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.plot(wait_times, phases_deg, 'o', markersize=10, label='Measured (state 10)', zorder=5)

        # Plot fit curve
        wait_range = np.linspace(min(wait_times), max(wait_times), 100)
        fit_line = np.polyval(fit_params, wait_range)
        fit_label = f'Polynomial fit (degree {fit_degree})'
        ax.plot(wait_range, fit_line, '-', color='blue', alpha=0.7, label=fit_label)

        # Plot target line and tolerance band
        ax.axhline(target_phase, color='r', linestyle='--',
                   label=f'Target phase ({target_phase:.1f} deg)')
        ax.axhspan(target_phase - target_phase_tol, target_phase + target_phase_tol,
                   color='r', alpha=0.1, label=f'+/-{target_phase_tol:.1f} deg tolerance')

        # Mark fitted optimal wait time
        if fitted_optimal_wait is not None:
            fitted_phase = np.polyval(fit_params, fitted_optimal_wait)
            ax.scatter([fitted_optimal_wait], [fitted_phase], color='green', s=200,
                       marker='*', zorder=10,
                       label=f'Fitted optimal: {fitted_optimal_wait:.4f} us')
            ax.axvline(fitted_optimal_wait, color='green', linestyle=':', alpha=0.5)

        ax.set_xlabel('Wait Time (us)', fontsize=12)
        ax.set_ylabel('Phase (degrees)', fontsize=12)
        title = 'BOB (parity): Wait Time Sweep'
        if fit_degree == 1:
            title += ' (Linear Fit)'
        else:
            title += f' (Degree-{fit_degree} Polynomial Fit)'
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def fit_wait_time(self, sweep_data, fit_degree=1, target_phase_diff=1.0,
                      target_phase_tol=0.1, skip_indices=None):
        """
        Fit phase difference vs wait time from sweep data.

        Can be called independently to refit with different polynomial degrees
        or skip outlier points without re-running experiments.

        Parameters
        ----------
        sweep_data : dict
            Sweep data returned by calibrate_wait_time().
            Must contain 'results' key with list of per-wait-time dicts.
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)
        target_phase_diff : float
            Target phase difference in units of pi (default: 1.0)
        target_phase_tol : float
            Tolerance on phase difference in units of pi (default: 0.1)
        skip_indices : list of int, optional
            Indices of points to exclude from the fit.

        Returns
        -------
        dict with keys:
            'fit_params': polynomial coefficients from np.polyfit (highest degree first)
            'fit_degree': degree of polynomial used
            'fitted_optimal_wait': wait time estimated from polynomial fit
            'fit_figure': matplotlib figure of the fit
            'optimal_idx': index of best measured wait time
            'optimal_wait': best measured wait time
            'optimal_phase_diff': phase difference at best measured wait time
            'converged': whether best measured wait time is within tolerance
        """
        results = sweep_data['results']

        # Extract data
        wait_times = np.array([r['wait_time'] for r in results])
        phase_diffs = np.array([r['phase_diff_pi'] for r in results])

        # Apply skip mask
        if skip_indices is None:
            mask = np.ones(len(results), dtype=bool)
        else:
            mask = np.ones(len(results), dtype=bool)
            mask[skip_indices] = False

        n_skipped = np.sum(~mask)
        if n_skipped > 0:
            self._log(f"  Skipping {n_skipped} points: indices {list(np.where(~mask)[0])}")

        # Find best measured result (among non-skipped points)
        masked_errors = np.full(len(results), np.inf)
        masked_errors[mask] = np.abs(phase_diffs[mask] - target_phase_diff)
        optimal_idx = int(np.argmin(masked_errors))
        converged = masked_errors[optimal_idx] <= target_phase_tol

        self._log(f"Best measured wait: {wait_times[optimal_idx]:.3f} us "
                  f"(phase_diff={phase_diffs[optimal_idx]:.4f} pi)")

        # Polynomial fit
        fit_params = None
        fitted_optimal_wait = None
        fit_figure = None

        min_points_needed = fit_degree + 1
        n_fit_points = np.sum(mask)
        if n_fit_points >= min_points_needed:
            try:
                fit_params = np.polyfit(wait_times[mask], phase_diffs[mask], fit_degree)

                # Find wait_time where phase_diff = target_phase_diff
                root_coeffs = fit_params.copy()
                root_coeffs[-1] -= target_phase_diff

                roots = np.roots(root_coeffs)

                # Filter for real roots within a reasonable range of the data
                wait_min, wait_max = min(wait_times[mask]), max(wait_times[mask])
                wait_margin = (wait_max - wait_min) * 0.5
                valid_roots = []
                for root in roots:
                    if np.isreal(root):
                        root_real = np.real(root)
                        if (wait_min - wait_margin) <= root_real <= (wait_max + wait_margin):
                            valid_roots.append(root_real)

                if valid_roots:
                    wait_center = (wait_min + wait_max) / 2
                    fitted_optimal_wait = min(valid_roots, key=lambda x: abs(x - wait_center))

                    if fit_degree == 1:
                        self._log(f"Linear fit: phase_diff = {fit_params[0]:.4f} * wait + {fit_params[1]:.4f}")
                    else:
                        self._log(f"Polynomial fit (degree {fit_degree}): coeffs = {fit_params}")
                    self._log(f"Fitted optimal wait: {fitted_optimal_wait:.4f} us")

                    fit_figure = self._plot_wait_sweep_fit(
                        wait_times, phase_diffs, fit_params, fitted_optimal_wait,
                        target_phase_diff, target_phase_tol, fit_degree
                    )
                else:
                    self._log("WARNING: No valid root found in data range, cannot estimate optimal wait time")

            except Exception as e:
                self._log(f"ERROR during polynomial fit: {e}")

        else:
            self._log(f"WARNING: Not enough non-skipped points for degree-{fit_degree} fit "
                      f"(have {n_fit_points}, need >= {min_points_needed})")

        return {
            'fit_params': fit_params,
            'fit_degree': fit_degree,
            'fitted_optimal_wait': fitted_optimal_wait,
            'fit_figure': fit_figure,
            'optimal_idx': optimal_idx,
            'optimal_wait': wait_times[optimal_idx],
            'optimal_phase_diff': phase_diffs[optimal_idx],
            'converged': converged,
        }

    def fit_wait_time_parity(self, sweep_data, fit_degree=1, target_phase=None,
                             target_phase_tol=10.0, skip_indices=None):
        """
        Fit phase_10 (degrees) vs wait time from parity sweep data.

        Can be called independently to refit with different polynomial degrees
        or skip outlier points without re-running experiments.

        Parameters
        ----------
        sweep_data : dict
            From calibrate_wait_time_parity(). Must have 'results' and 'target_phase'.
        fit_degree : int
            Degree of polynomial fit (default: 1)
        target_phase : float or None
            Target phase in degrees. If None, uses sweep_data['target_phase'].
        target_phase_tol : float
            Tolerance in degrees (default: 10.0)
        skip_indices : list of int, optional
            Indices to exclude from fit.

        Returns
        -------
        dict with keys:
            'fit_params': polynomial coefficients
            'fit_degree': degree used
            'fitted_optimal_wait': wait time from fit
            'fit_figure': matplotlib figure
            'optimal_idx': index of best measured point
            'optimal_wait': best measured wait time
            'converged': whether within tolerance
        """
        results = sweep_data['results']
        if target_phase is None:
            target_phase = sweep_data['target_phase']

        # Extract data
        wait_times = np.array([r['wait_time'] for r in results])
        phases_raw = np.array([r['phase_10_deg'] for r in results])

        # Apply skip mask
        if skip_indices is None:
            mask = np.ones(len(results), dtype=bool)
        else:
            mask = np.ones(len(results), dtype=bool)
            mask[skip_indices] = False

        n_skipped = np.sum(~mask)
        if n_skipped > 0:
            self._log(f"  Skipping {n_skipped} points: indices {list(np.where(~mask)[0])}")

        # Unwrap phases to handle wrapping
        phases_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phases_raw)))

        # Unwrap target phase to match the data range
        data_mean = np.mean(phases_unwrapped[mask])
        target_unwrapped = target_phase
        # Find closest unwrapped equivalent: target + 360*k
        k = round((data_mean - target_phase) / 360)
        target_unwrapped = target_phase + 360 * k

        self._log(f"  Target phase: {target_phase:.1f} deg (unwrapped: {target_unwrapped:.1f} deg)")

        # Find best measured result (among non-skipped points)
        masked_errors = np.full(len(results), np.inf)
        # Use angular difference for finding the best point
        for idx in range(len(results)):
            if mask[idx]:
                masked_errors[idx] = abs(self._angular_diff_deg(
                    phases_raw[idx], target_phase
                ))
        optimal_idx = int(np.argmin(masked_errors))
        converged = masked_errors[optimal_idx] <= target_phase_tol

        self._log(f"Best measured wait: {wait_times[optimal_idx]:.3f} us "
                  f"(phase_10={phases_raw[optimal_idx]:.1f} deg, "
                  f"error={masked_errors[optimal_idx]:.1f} deg)")

        # Polynomial fit
        fit_params = None
        fitted_optimal_wait = None
        fit_figure = None

        min_points_needed = fit_degree + 1
        n_fit_points = np.sum(mask)
        if n_fit_points >= min_points_needed:
            try:
                fit_params = np.polyfit(
                    wait_times[mask], phases_unwrapped[mask], fit_degree
                )

                # Find wait_time where phase = target_unwrapped
                root_coeffs = fit_params.copy()
                root_coeffs[-1] -= target_unwrapped

                roots = np.roots(root_coeffs)

                # Filter for real roots within a reasonable range
                wait_min, wait_max = min(wait_times[mask]), max(wait_times[mask])
                wait_margin = (wait_max - wait_min) * 0.5
                valid_roots = []
                for root in roots:
                    if np.isreal(root):
                        root_real = np.real(root)
                        if (wait_min - wait_margin) <= root_real <= (wait_max + wait_margin):
                            valid_roots.append(root_real)

                if valid_roots:
                    wait_center = (wait_min + wait_max) / 2
                    fitted_optimal_wait = min(
                        valid_roots, key=lambda x: abs(x - wait_center)
                    )
                    self._log(f"Fitted optimal wait: {fitted_optimal_wait:.4f} us")

                    fit_figure = self._plot_wait_sweep_parity_fit(
                        wait_times, phases_unwrapped, fit_params,
                        fitted_optimal_wait, target_unwrapped,
                        target_phase_tol, fit_degree
                    )
                else:
                    self._log("WARNING: No valid root found in data range")

            except Exception as e:
                self._log(f"ERROR during polynomial fit: {e}")

        else:
            self._log(f"WARNING: Not enough non-skipped points for degree-{fit_degree} fit "
                      f"(have {n_fit_points}, need >= {min_points_needed})")

        return {
            'fit_params': fit_params,
            'fit_degree': fit_degree,
            'fitted_optimal_wait': fitted_optimal_wait,
            'fit_figure': fit_figure,
            'optimal_idx': optimal_idx,
            'optimal_wait': wait_times[optimal_idx],
            'optimal_phase_10': phases_raw[optimal_idx],
            'converged': converged,
        }

    def calibrate_wait_time(self, wait_time_list, fixed_params,
                            target_phase_diff=1.0, target_phase_tol=0.1,
                            fit_degree=1, alpha_amplitude=None, reps=100,
                            wigner_n_points=100):
        """
        BOB: Sweep wait time at fixed beam splitter rate (WITHOUT storage swap).

        Performs a sweep over wait times, fits phase difference vs wait time
        to a polynomial, and estimates the optimal wait time from the fit.

        Parameters
        ----------
        wait_time_list : array-like
            List of wait times to test (microseconds)
        fixed_params : dict
            Must contain: freq_bs, gain, pi_length (from calibrate_beam_splitter_rate)
        target_phase_diff : float
            Target phase difference in units of pi (default: 1.0)
        target_phase_tol : float
            Tolerance on phase difference in units of pi (default: 0.1)
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)
        alpha_amplitude : float
            Displacement amplitude (if None, uses fixed_params['alpha_amplitude'])
        reps : int
            Number of repetitions per Wigner measurement (default: 100)
        wigner_n_points : int
            Number of points on the Wigner alpha circle (default: 100)

        Returns
        -------
        dict with wait time sweep results including:
            - results: list of results for each wait time
            - optimal_idx: index of best measured wait time
            - optimal_wait: best measured wait time
            - optimal_phase_diff: phase difference at best measured wait time
            - converged: whether best measured wait time is within tolerance
            - fit_params: polynomial coefficients from np.polyfit (highest degree first)
            - fit_degree: degree of polynomial used
            - fitted_optimal_wait: wait time estimated from polynomial fit
            - fit_figure: matplotlib figure of the fit
        """
        self._log(f"=== BOB: Sweeping {len(wait_time_list)} wait times ===")

        freq_bs = fixed_params['freq_bs']
        gain = fixed_params['gain']
        pi_length = fixed_params['pi_length']
        man_mode_no = fixed_params.get('man_mode_no', 1)
        stor_mode_no = fixed_params.get('stor_mode_no', 1)

        if alpha_amplitude is None:
            alpha_amplitude = fixed_params.get('alpha_amplitude', 1.0)

        results = []
        for i, wait_time in enumerate(wait_time_list):
            self._log(f"\nWait time {i+1}/{len(wait_time_list)}: {wait_time:.3f} us")

            theta_g, _ = self._run_wigner_bob(
                freq_bs, gain, pi_length, 'g', alpha_amplitude,
                man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                reps=reps, wait_time=wait_time, n_points=wigner_n_points
            )
            theta_e, _ = self._run_wigner_bob(
                freq_bs, gain, pi_length, 'e', alpha_amplitude,
                man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                reps=reps, wait_time=wait_time, n_points=wigner_n_points
            )

            phase_diff = np.abs(theta_e - theta_g) / np.pi
            self._log(f"  Phase difference: {phase_diff:.4f} pi")

            results.append({
                'wait_time': wait_time,
                'theta_g': theta_g,
                'theta_e': theta_e,
                'phase_diff_pi': phase_diff,
            })

        # Build sweep data and store for later refitting
        sweep_data = {'results': results}
        self.last_wait_time_data = sweep_data

        # Fit phase_diff vs wait_time using the refittable method
        fit_result = self.fit_wait_time(
            sweep_data, fit_degree=fit_degree,
            target_phase_diff=target_phase_diff,
            target_phase_tol=target_phase_tol,
        )

        # Merge fit results into return dict
        sweep_data.update(fit_result)

        self._log(f"\nOptimal wait: {fit_result['optimal_wait']:.3f} us "
                  f"(phase_diff={fit_result['optimal_phase_diff']:.4f} pi, "
                  f"converged={fit_result['converged']})")

        return sweep_data

    def calibrate_wait_time_parity(self, wait_time_list, fixed_params,
                                    storage_swap, target_phase=None,
                                    target_phase_tol=10.0, fit_degree=1,
                                    reps=200, parity_fast=False):
        """
        BOB (parity): Sweep wait time using ParityPhaseExperiment with state '10'.

        Alternative to calibrate_wait_time() that uses ParityPhaseExperiment
        instead of Wigner tomography. Prepares state '10' (1 photon in
        storage_swap) and sweeps wait time until the phase matches the
        optimal '01' phase from sweep_beam_splitter_rate_parity().

        Parameters
        ----------
        wait_time_list : array-like
            List of wait times to test (microseconds)
        fixed_params : dict
            Must contain: freq_bs, gain, pi_length, stor_mode_no.
            Typically from sweep_beam_splitter_rate_parity()['fitted_calibration'].
        storage_swap : int
            Storage mode to swap to/from manipulate
        target_phase : float, optional
            Target phase for state '10' in degrees.
            If None, uses fixed_params['phase_01_optimal'].
        target_phase_tol : float
            Tolerance in degrees (default: 10.0)
        fit_degree : int
            Degree of polynomial fit (default: 1)
        reps : int
            Repetitions per ParityPhaseExperiment (default: 200)
        parity_fast : bool
            Use fast multiphoton hpi pulses (default: False)

        Returns
        -------
        dict with wait time sweep results including:
            - results: list of results for each wait time
            - target_phase: target phase in degrees
            - fit_params, fit_degree, fitted_optimal_wait, fit_figure
            - optimal_idx, optimal_wait, converged
        """
        self._log(f"=== BOB (parity): Sweeping {len(wait_time_list)} wait times ===")

        freq_bs = fixed_params['freq_bs']
        gain = fixed_params['gain']
        pi_length = fixed_params['pi_length']
        man_mode_no = fixed_params.get('man_mode_no', 1)
        stor_mode_no = fixed_params.get('stor_mode_no', 1)
        stor_parity_name = f'M{man_mode_no}-S{stor_mode_no}'

        if target_phase is None:
            target_phase = fixed_params.get('phase_01_optimal')
            if target_phase is None:
                raise ValueError(
                    "target_phase must be provided or present in "
                    "fixed_params['phase_01_optimal']"
                )

        self._log(f"  Target phase: {target_phase:.1f} deg")

        # Save original joint parity params for restore
        original_jp = self.station.ds_storage.get_joint_parity(stor_parity_name)

        results = []
        try:
            for i, wait_time in enumerate(wait_time_list):
                self._log(f"\nWait time {i+1}/{len(wait_time_list)}: {wait_time:.3f} us")

                # Update joint parity params with this wait time
                self.station.ds_storage.update_joint_parity(
                    stor_parity_name, freq_bs, gain,
                    pi_length * 2, wait_time
                )

                # Run parity phase with state '10'
                phases, _ = self._run_parity_phase(
                    storage_swap=storage_swap, storage_parity=stor_mode_no,
                    state_start='10', man_mode_no=man_mode_no,
                    reps=reps, parity_fast=parity_fast,
                )

                phase_10 = phases.get('10')
                if phase_10 is not None:
                    phase_error = self._angular_diff_deg(phase_10, target_phase)
                    self._log(f"  phase_10={phase_10:.1f} deg, "
                              f"error={phase_error:.1f} deg")
                else:
                    self._log(f"  WARNING: Failed to extract phase for state '10'")
                    phase_10 = np.nan

                results.append({
                    'wait_time': wait_time,
                    'phase_10_deg': phase_10,
                })

        finally:
            # Restore original joint parity params
            if original_jp is not None:
                self.station.ds_storage.update_joint_parity(
                    stor_parity_name, *original_jp
                )
                self._log(f"  Restored original joint parity params for {stor_parity_name}")

        # Build sweep data and store for later refitting
        sweep_data = {
            'results': results,
            'target_phase': target_phase,
        }
        self.last_wait_time_parity_data = sweep_data

        # Fit phase_10 vs wait_time
        fit_result = self.fit_wait_time_parity(
            sweep_data, fit_degree=fit_degree,
            target_phase=target_phase,
            target_phase_tol=target_phase_tol,
        )

        # Merge fit results into return dict
        sweep_data.update(fit_result)

        self._log(f"\nOptimal wait: {fit_result['optimal_wait']:.3f} us "
                  f"(phase_10={fit_result['optimal_phase_10']:.1f} deg, "
                  f"converged={fit_result['converged']})")

        return sweep_data

    def plot_rate_sweep(self, sweep_result, title="ALICE: Gain Sweep"):
        """Plot phase difference vs gain."""
        successful = [r for r in sweep_result['results'] if r.get('success', False)]
        gains = [r['gain'] for r in successful]
        phase_diffs = [r['phase_diff_pi'] for r in successful]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(gains, phase_diffs, 'o-', markersize=8)
        ax.axhline(1.0, color='r', linestyle='--', label='Target (pi)')
        ax.axhline(0.9, color='r', linestyle=':', alpha=0.5)
        ax.axhline(1.1, color='r', linestyle=':', alpha=0.5)

        if sweep_result['optimal_idx'] is not None:
            opt_gain = sweep_result['optimal_gain']
            opt_phase = sweep_result['optimal_result']['phase_diff_pi']
            ax.scatter([opt_gain], [opt_phase], color='green', s=150,
                      zorder=5, label=f'Optimal (gain={opt_gain})')

        ax.set_xlabel('Gain')
        ax.set_ylabel('Phase Difference (pi)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_wait_sweep(self, wait_result, title="BOB: Wait Time Sweep"):
        """Plot phase difference vs wait time."""
        wait_times = [r['wait_time'] for r in wait_result['results']]
        phase_diffs = [r['phase_diff_pi'] for r in wait_result['results']]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wait_times, phase_diffs, 'o-', markersize=8)
        ax.axhline(1.0, color='r', linestyle='--', label='Target (pi)')
        ax.axhline(0.9, color='r', linestyle=':', alpha=0.5)
        ax.axhline(1.1, color='r', linestyle=':', alpha=0.5)

        opt_wait = wait_result['optimal_wait']
        opt_phase = wait_result['optimal_phase_diff']
        ax.scatter([opt_wait], [opt_phase], color='green', s=150,
                  zorder=5, label=f'Optimal ({opt_wait:.3f} us)')

        ax.set_xlabel('Wait Time (us)')
        ax.set_ylabel('Phase Difference (pi)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

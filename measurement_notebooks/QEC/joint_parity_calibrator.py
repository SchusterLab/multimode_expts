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
            n_start=0, n_step=3, n_pulses=15, expts=50,
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

    def _log(self, msg):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[JointParityCalibrator] {msg}")

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
        ch = flux_low_ch if freq < 1000 else flux_high_ch

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
            expt_cfg.flux_drive = [ch[0], freq_bs, gain, 0]
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

    def calibrate_beam_splitter_rate(self, g_bs, man_mode_no=1, stor_mode_no=1,
                                      alpha_amplitude=1.0,
                                      reps_error_amp=50, reps_rabi=100, reps_wigner=100,
                                      error_amp_span=0.05, n_pulses=15,
                                      run_both_rabi_states=False,
                                      rabi_start=None, rabi_stop=None, rabi_expts=None,
                                      wigner_n_points=100):
        """
        ALICE: Calibrate at a given beam splitter rate (WITH storage swap).

        Parameters
        ----------
        g_bs : float
            Beam splitter rate in MHz
        alpha_amplitude : float
            Displacement amplitude for Wigner circle
        run_both_rabi_states : bool
            If True, run Rabi for both g and e states and average pi_length
        rabi_start, rabi_stop, rabi_expts : float, float, int
            Rabi sweep parameters (optional)
        wigner_n_points : int
            Number of points on the Wigner alpha circle (default: 100)

        Returns
        -------
        dict with calibration results
        """
        self._log(f"=== ALICE: Calibrating at g_bs={g_bs:.4f} MHz ===")

        stor_name = f'M{man_mode_no}-S{stor_mode_no}'

        # Step 1: Setup - get gain/freq from lookup tables
        gain_e = self.station.ds_storage.get_gain_at_bs_rate(stor_name, g_bs, 'e')
        gain_g = self.station.ds_storage.get_gain_at_bs_rate(stor_name, g_bs, 'g')
        gain = int((gain_e + gain_g) / 2)
        f_bs_g_guess = self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'g')
        f_bs_e_guess = self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'e')
        length_bs = 0.25 / g_bs

        self._log(f"  Initial: gain={gain}, length_bs={length_bs:.4f} us")
        self._log(f"  Freq guesses: f_bs_g={f_bs_g_guess:.4f}, f_bs_e={f_bs_e_guess:.4f}")

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

    def sweep_beam_splitter_rate(self, rate_list, target_phase_diff=1.0,
                                  target_phase_tol=0.1, fit_degree=1, **kwargs):
        """
        ALICE: Sweep beam splitter rates to find optimal one.

        Performs a sweep over beam splitter rates, fits phase difference vs rate
        to a polynomial, estimates the optimal rate from the fit, and runs
        a final calibration (error amp + Rabi) at the fitted optimal rate.

        Parameters
        ----------
        rate_list : array-like
            List of beam splitter rates to test (MHz)
        target_phase_diff : float
            Target phase difference in units of pi (default: 1.0)
        target_phase_tol : float
            Tolerance on phase difference in units of pi (default: 0.1)
        fit_degree : int
            Degree of polynomial fit (default: 1 for linear)

        Returns
        -------
        dict with sweep results including:
            - results: list of calibration results for each rate
            - optimal_idx: index of best measured rate
            - optimal_rate: best measured rate
            - optimal_result: calibration result for best measured rate
            - converged: whether best measured rate is within tolerance
            - fit_params: polynomial coefficients from np.polyfit (highest degree first)
            - fit_degree: degree of polynomial used
            - fitted_optimal_rate: rate estimated from polynomial fit
            - fitted_calibration: error amp + Rabi results at fitted rate
            - fit_figure: matplotlib figure of the fit
        """
        self._log(f"=== ALICE: Sweeping {len(rate_list)} beam splitter rates ===")

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
        alpha_amplitude = kwargs.get('alpha_amplitude', 1.0)
        reps_wigner = kwargs.get('reps_wigner', 100)
        wigner_n_points = kwargs.get('wigner_n_points', 100)

        results = []
        for i, g_bs in enumerate(rate_list):
            self._log(f"\nRate {i+1}/{len(rate_list)}: g_bs={g_bs:.4f} MHz")
            try:
                result = self.calibrate_beam_splitter_rate(g_bs, **kwargs)
                results.append(result)
            except Exception as e:
                self._log(f"  ERROR: {e}")
                results.append({'g_bs': g_bs, 'success': False, 'error': str(e)})

        # Find best result among successful ones
        successful = [r for r in results if r.get('success', False)]
        if not successful:
            self._log("No successful calibrations!")
            return {
                'results': results,
                'optimal_idx': None,
                'optimal_rate': None,
                'optimal_result': None,
                'converged': False,
                'fit_params': None,
                'fit_degree': fit_degree,
                'fitted_optimal_rate': None,
                'fitted_calibration': None,
                'fit_figure': None,
            }

        # Extract data for fitting
        rates = np.array([r['g_bs'] for r in successful])
        phase_diffs = np.array([r['phase_diff_pi'] for r in successful])

        # Find best measured result
        errors = [abs(pd - target_phase_diff) for pd in phase_diffs]
        optimal_idx = np.argmin(errors)
        converged = errors[optimal_idx] <= target_phase_tol
        original_idx = results.index(successful[optimal_idx])

        self._log(f"\nBest measured rate: {rates[optimal_idx]:.4f} MHz "
                  f"(phase_diff={phase_diffs[optimal_idx]:.4f} pi)")

        # Fit phase_diff vs rate to polynomial
        fit_params = None
        fitted_optimal_rate = None
        fitted_calibration = None
        fit_figure = None

        min_points_needed = fit_degree + 1
        if len(successful) >= min_points_needed:
            try:
                # Polynomial fit: phase_diff = poly(rate)
                fit_params = np.polyfit(rates, phase_diffs, fit_degree)

                # Find rate where phase_diff = target_phase_diff
                # Solve: poly(rate) - target = 0
                root_coeffs = fit_params.copy()
                root_coeffs[-1] -= target_phase_diff  # Subtract target from constant term

                # Find roots
                roots = np.roots(root_coeffs)

                # Filter for real roots within a reasonable range of the data
                rate_min, rate_max = min(rates), max(rates)
                rate_margin = (rate_max - rate_min) * 0.5
                valid_roots = []
                for root in roots:
                    if np.isreal(root):
                        root_real = np.real(root)
                        if (rate_min - rate_margin) <= root_real <= (rate_max + rate_margin):
                            valid_roots.append(root_real)

                if valid_roots:
                    # Pick the root closest to the center of the data range
                    rate_center = (rate_min + rate_max) / 2
                    fitted_optimal_rate = min(valid_roots, key=lambda x: abs(x - rate_center))

                    # Log fit info
                    if fit_degree == 1:
                        self._log(f"\nLinear fit: phase_diff = {fit_params[0]:.4f} * rate + {fit_params[1]:.4f}")
                    else:
                        self._log(f"\nPolynomial fit (degree {fit_degree}): coeffs = {fit_params}")
                    self._log(f"Fitted optimal rate: {fitted_optimal_rate:.4f} MHz")

                    # Create fit plot
                    fit_figure = self._plot_rate_sweep_fit(
                        rates, phase_diffs, fit_params, fitted_optimal_rate,
                        target_phase_diff, target_phase_tol, fit_degree
                    )

                    # Run error amplification and Rabi at the fitted optimal rate
                    self._log(f"\n=== Calibrating at fitted optimal rate: {fitted_optimal_rate:.4f} MHz ===")
                    stor_name = f'M{man_mode_no}-S{stor_mode_no}'

                    # Get gain and length for the fitted rate
                    gain_e = self.station.ds_storage.get_gain_at_bs_rate(stor_name, fitted_optimal_rate, 'e')
                    gain_g = self.station.ds_storage.get_gain_at_bs_rate(stor_name, fitted_optimal_rate, 'g')
                    gain = int((gain_e + gain_g) / 2)
                    f_bs_g_guess = self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'g')
                    f_bs_e_guess = self.station.ds_storage.get_freq_at_gain(stor_name, gain, 'e')
                    length_bs = 0.25 / fitted_optimal_rate

                    self._log(f"  Initial: gain={gain}, length_bs={length_bs:.4f} us")

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

                    # check phase difference at fitted rate
                    # theta_g, fitted_experiments['wigner_g'] = self._run_wigner_alice(
                    #     freq_bs, gain, pi_length, 'g', alpha_amplitude=alpha_amplitude,
                    #     man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                    #     reps=reps_wigner, n_points=wigner_n_points,
                    # )
                    # theta_e, fitted_experiments['wigner_e'] = self._run_wigner_alice(
                    #     freq_bs, gain, pi_length, 'e', alpha_amplitude=alpha_amplitude,
                    #     man_mode_no=man_mode_no, stor_mode_no=stor_mode_no,
                    #     reps=reps_wigner, n_points=wigner_n_points,
                    # )
                    # phase_diff_pi = np.abs(theta_e - theta_g) / np.pi
                    # self._log(f"  theta_g={theta_g*180/np.pi:.2f} deg, theta_e={theta_e*180/np.pi:.2f} deg")
                    # self._log(f"  Phase difference at fitted rate: {phase_diff_pi:.4f} pi")


                    fitted_calibration = {
                        'g_bs': fitted_optimal_rate,
                        'freq_bs': freq_bs,
                        'f_bs_g': f_bs_g,
                        'f_bs_e': f_bs_e,
                        'gain': gain,
                        'length_bs': length_bs,
                        'pi_length': pi_length,
                        'man_mode_no': man_mode_no,
                        'stor_mode_no': stor_mode_no,
                        'experiments': fitted_experiments,
                    }
                else:
                    self._log("WARNING: No valid root found in data range, cannot estimate optimal rate")

            except Exception as e:
                self._log(f"ERROR during polynomial fit or fitted calibration: {e}")

        else:
            self._log(f"WARNING: Not enough successful points for degree-{fit_degree} fit (need >= {min_points_needed})")

        return {
            'results': results,
            'optimal_idx': original_idx,
            'optimal_rate': rates[optimal_idx],
            'optimal_result': successful[optimal_idx],
            'converged': converged,
            'fit_params': fit_params,
            'fit_degree': fit_degree,
            'fitted_optimal_rate': fitted_optimal_rate,
            'fitted_calibration': fitted_calibration,
            'fit_figure': fit_figure,
        }

    def _plot_rate_sweep_fit(self, rates, phase_diffs, fit_params, fitted_optimal_rate,
                              target_phase_diff, target_phase_tol, fit_degree=1):
        """
        Plot phase difference vs rate with polynomial fit.

        Parameters
        ----------
        rates : array
            Beam splitter rates (MHz)
        phase_diffs : array
            Measured phase differences (units of pi)
        fit_params : array
            Polynomial coefficients from np.polyfit (highest degree first)
        fitted_optimal_rate : float
            Estimated optimal rate from fit
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
        ax.plot(rates, phase_diffs, 'o', markersize=10, label='Measured', zorder=5)

        # Plot fit curve
        rate_range = np.linspace(min(rates), max(rates), 100)
        fit_line = np.polyval(fit_params, rate_range)

        # Build fit label
        if fit_degree == 1:
            fit_label = f'Fit: y = {fit_params[0]:.4f}x + {fit_params[1]:.4f}'
        else:
            fit_label = f'Polynomial fit (degree {fit_degree})'

        ax.plot(rate_range, fit_line, '-', color='blue', alpha=0.7, label=fit_label)

        # Plot target line and tolerance band
        ax.axhline(target_phase_diff, color='r', linestyle='--', label=f'Target ({target_phase_diff} pi)')
        ax.axhspan(target_phase_diff - target_phase_tol, target_phase_diff + target_phase_tol,
                   color='r', alpha=0.1, label=f'+/-{target_phase_tol} pi tolerance')

        # Mark fitted optimal rate
        fitted_phase = np.polyval(fit_params, fitted_optimal_rate)
        ax.scatter([fitted_optimal_rate], [fitted_phase], color='green', s=200,
                   marker='*', zorder=10, label=f'Fitted optimal: {fitted_optimal_rate:.4f} MHz')
        ax.axvline(fitted_optimal_rate, color='green', linestyle=':', alpha=0.5)

        ax.set_xlabel('Beam Splitter Rate (MHz)', fontsize=12)
        ax.set_ylabel('Phase Difference (pi)', fontsize=12)
        title = 'ALICE: Beam Splitter Rate Sweep'
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

        # Extract data for fitting
        wait_times = np.array([r['wait_time'] for r in results])
        phase_diffs = np.array([r['phase_diff_pi'] for r in results])

        # Find best measured result
        errors = [abs(pd - target_phase_diff) for pd in phase_diffs]
        optimal_idx = np.argmin(errors)
        converged = errors[optimal_idx] <= target_phase_tol

        self._log(f"\nBest measured wait: {wait_times[optimal_idx]:.3f} us "
                  f"(phase_diff={phase_diffs[optimal_idx]:.4f} pi)")

        # Fit phase_diff vs wait_time to polynomial
        fit_params = None
        fitted_optimal_wait = None
        fit_figure = None

        min_points_needed = fit_degree + 1
        if len(results) >= min_points_needed:
            try:
                # Polynomial fit: phase_diff = poly(wait_time)
                fit_params = np.polyfit(wait_times, phase_diffs, fit_degree)

                # Find wait_time where phase_diff = target_phase_diff
                # Solve: poly(wait_time) - target = 0
                root_coeffs = fit_params.copy()
                root_coeffs[-1] -= target_phase_diff  # Subtract target from constant term

                # Find roots
                roots = np.roots(root_coeffs)

                # Filter for real roots within a reasonable range of the data
                wait_min, wait_max = min(wait_times), max(wait_times)
                wait_margin = (wait_max - wait_min) * 0.5
                valid_roots = []
                for root in roots:
                    if np.isreal(root):
                        root_real = np.real(root)
                        if (wait_min - wait_margin) <= root_real <= (wait_max + wait_margin):
                            valid_roots.append(root_real)

                if valid_roots:
                    # Pick the root closest to the center of the data range
                    wait_center = (wait_min + wait_max) / 2
                    fitted_optimal_wait = min(valid_roots, key=lambda x: abs(x - wait_center))

                    # Log fit info
                    if fit_degree == 1:
                        self._log(f"\nLinear fit: phase_diff = {fit_params[0]:.4f} * wait + {fit_params[1]:.4f}")
                    else:
                        self._log(f"\nPolynomial fit (degree {fit_degree}): coeffs = {fit_params}")
                    self._log(f"Fitted optimal wait: {fitted_optimal_wait:.4f} us")

                    # Create fit plot
                    fit_figure = self._plot_wait_sweep_fit(
                        wait_times, phase_diffs, fit_params, fitted_optimal_wait,
                        target_phase_diff, target_phase_tol, fit_degree
                    )
                else:
                    self._log("WARNING: No valid root found in data range, cannot estimate optimal wait time")

            except Exception as e:
                self._log(f"ERROR during polynomial fit: {e}")

        else:
            self._log(f"WARNING: Not enough points for degree-{fit_degree} fit (need >= {min_points_needed})")

        self._log(f"\nOptimal wait: {wait_times[optimal_idx]:.3f} us "
                  f"(phase_diff={results[optimal_idx]['phase_diff_pi']:.4f} pi, converged={converged})")

        return {
            'results': results,
            'optimal_idx': optimal_idx,
            'optimal_wait': wait_times[optimal_idx],
            'optimal_phase_diff': results[optimal_idx]['phase_diff_pi'],
            'converged': converged,
            'fit_params': fit_params,
            'fit_degree': fit_degree,
            'fitted_optimal_wait': fitted_optimal_wait,
            'fit_figure': fit_figure,
        }

    def plot_rate_sweep(self, sweep_result, title="ALICE: Beam Splitter Rate Sweep"):
        """Plot phase difference vs beam splitter rate."""
        successful = [r for r in sweep_result['results'] if r.get('success', False)]
        rates = [r['g_bs'] for r in successful]
        phase_diffs = [r['phase_diff_pi'] for r in successful]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rates, phase_diffs, 'o-', markersize=8)
        ax.axhline(1.0, color='r', linestyle='--', label='Target (pi)')
        ax.axhline(0.9, color='r', linestyle=':', alpha=0.5)
        ax.axhline(1.1, color='r', linestyle=':', alpha=0.5)

        if sweep_result['optimal_idx'] is not None:
            opt_rate = sweep_result['optimal_rate']
            opt_phase = sweep_result['optimal_result']['phase_diff_pi']
            ax.scatter([opt_rate], [opt_phase], color='green', s=150,
                      zorder=5, label=f'Optimal ({opt_rate:.4f} MHz)')

        ax.set_xlabel('Beam Splitter Rate (MHz)')
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

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from scipy.ndimage import gaussian_filter1d
from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from MM_dual_rail_base import MM_dual_rail_base

"""
In this file, each program looks at the effect of a particular combo of
detuning, amplitude and duration of the Kerr engineering pulse on the qubit.
We need to know the following effects of such a tone:
- Qubit heating test: apply pump and check qubit state again to check heat up
- Qubit AC Stark shift: find the phase correction for the second qubit pi/2 pulse
  Q: do we need this for qsim if we never use non-pi gates on QB?
- Cavity Ramsey: extract the effective Kerr under this drive
"""

class KerrEngBaseProgram(QsimBaseProgram):
    def initialize(self):
        super().initialize()
        cfg = self.cfg

        # for kerr engineering, drive a tone near the qubit
        if "qubit_drive_pulse" in cfg.expt and cfg.expt.qubit_drive_pulse[0]:
            self.qTest = self.qubits[self.qTest]
            self.qubit_drive_freq = self.freq2reg(cfg.expt.qubit_drive_pulse[1], gen_ch=self.qubit_chs[self.qTest])
            self.qubit_drive_gain = cfg.expt.qubit_drive_pulse[2]
            self.qubit_drive_sigma = self.us2cycles(cfg.expt.qubit_drive_pulse[3], gen_ch=self.qubit_chs[self.qTest])
            self.qubit_drive_length = self.us2cycles(cfg.expt.qubit_drive_pulse[4], gen_ch=self.qubit_chs[self.qTest])
            # Flat top pulse
            if self.qubit_drive_length == 0:
                self.add_gauss(ch=self.qubit_chs[self.qTest],
                               name="test_qubit_drive",
                               sigma=self.qubit_drive_sigma,
                               length=self.qubit_drive_sigma*4)

    def core_pulses(self):
        qTest = 0
        ecfg = self.cfg.expt
        kerr_pulse = [
            [self.cfg.device.qubit.f_ge[qTest] + ecfg.kerr_detune],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.qubit_chs[qTest]],
            ['flat_top'],
            [self.cfg.device.qubit.ramp_sigma[qTest]],
        ]
        self.custom_pulse(self.cfg, kerr_pulse, prefix='kerr_')
        # [[frequency], [gain], [length (us)], [phases],
        # [drive channel], [shape], [ramp sigma]]


class KerrStarkProgram(KerrEngBaseProgram):
    def body(self):
        pass


class KerrCavityRamseyProgram(KerrEngBaseProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)


    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = 0 # only one qubit for now

        self.swap_ds = self.cfg.expt.ds_thisrun
        # choose the channel on which ramsey will run
        if cfg.expt.user_defined_pulse[5] == 1:
            self.cavity_ch = self.flux_low_ch
            self.cavity_ch_types = self.flux_low_ch_type
        elif cfg.expt.user_defined_pulse[5] == 2:
            self.cavity_ch= self.qubit_chs
            self.cavity_ch_types = self.qubit_ch_types
        elif cfg.expt.user_defined_pulse[5] == 3:
            self.cavity_ch = self.flux_high_ch
            self.cavity_ch_types = self.flux_high_ch_type
        elif cfg.expt.user_defined_pulse[5] == 6:
            self.cavity_ch = self.storage_ch
            self.cavity_ch_types = self.storage_ch_type
        elif cfg.expt.user_defined_pulse[5] == 0:
            self.cavity_ch = self.f0g1_ch
            self.cavity_ch_types = self.f0g1_ch_type
        elif cfg.expt.user_defined_pulse[5] == 4:
            self.cavity_ch = self.man_ch
            self.cavity_ch_types = self.man_ch_type

        if self.cfg.expt.storage_ramsey[0]:
            # decide which channel do we flux drive on
            sweep_pulse = [
                ['storage', 'M'+ str(self.cfg.expt.man_mode_no) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi', 0],
            ]
            self.creator = self.get_prepulse_creator(sweep_pulse)
            freq = self.creator.pulse[0][0]
            self.flux_ch = self.flux_low_ch if freq < 1000 else self.flux_high_ch

        if self.cfg.expt.man_ramsey[0]:
            sweep_pulse = [
                ['man', 'M'+ str(self.cfg.expt.man_ramsey[1]) , 'pi', 0],
            ]
            self.creator = self.get_prepulse_creator(sweep_pulse)

        if self.cfg.expt.coupler_ramsey:
            # decide which channel do we flux drive on
            pulse_str = self.cfg.expt.custom_coupler_pulse
            freq = pulse_str[0][0]
            self.flux_ch = self.flux_low_ch if freq < 1000 else self.flux_high_ch

        if self.cfg.expt.echoes[0]:
            mm_base_dummy = MM_dual_rail_base(self.cfg, self.soccfg)
            if self.cfg.expt.storage_ramsey[0]:
                prep_stor = mm_base_dummy.prep_random_state_mode(3, self.cfg.expt.storage_ramsey[1])  # prepare the storage state +
            elif self.cfg.expt.man_ramsey[0]:
                prep_stor = mm_base_dummy.prep_man_photon(man_no=self.cfg.expt.man_ramsey[1], hpi = True)
            get_stor = prep_stor[::-1] # get the storage state
            self.echo_pulse_str = get_stor + prep_stor # echo pulse is the sum of the two pulse sequences
            self.echo_pulse = self.get_prepulse_creator(self.echo_pulse_str).pulse.tolist()
            # print(self.echo_pulse)

        # declare registers for phase incrementing
        # self.r_wait = 3
        # self.r_wait_flux = 3
        # self.r_phase2 = 4
        # self.r_phase3 = 0
        # self.r_phase4 = 6
        if (self.cfg.expt.storage_ramsey[0] and self.cfg.expt.storage_ramsey[2]) or self.cfg.expt.coupler_ramsey:
            self.phase_update_channel = self.flux_ch
        elif self.cfg.expt.man_ramsey[0]:
            self.phase_update_channel = self.cavity_ch

        elif self.cfg.expt.user_defined_pulse[0] and self.cfg.expt.storage_ramsey[0]:
            # print('Running Kerr; will update phase ch')
            self.phase_update_channel = self.cavity_ch
        elif self.cfg.expt.user_defined_pulse[0] :
            # print('Running f0g1 ramsey')
            self.phase_update_channel = self.cavity_ch
        # print(f'phase update channel: {self.phase_update_channel}')
        # self.phase_update_page = [self.ch_page(self.phase_update_channel[qTest])]
        # self.r_phase = self.sreg(self.phase_update_channel[qTest], "phase")

        self.current_phase = 0   # in degree

        #for user defined
        if cfg.expt.user_defined_pulse[0]:
            # print('This is designed for displacing manipulate mode, not for swapping pi/2 into man')
            self.user_freq = self.freq2reg(cfg.expt.user_defined_pulse[1], gen_ch=self.cavity_ch[qTest])
            self.user_gain = cfg.expt.user_defined_pulse[2]
            # dirty patch...
            if 'displace_gain' in cfg.expt.keys():
                self.user_gain = cfg.expt.displace_gain
            self.user_sigma = self.us2cycles(cfg.expt.user_defined_pulse[3], gen_ch=self.cavity_ch[qTest])
            self.user_length  = self.us2cycles(cfg.expt.user_defined_pulse[4], gen_ch=self.cavity_ch[qTest])
            # print(f"if user length is 0, then it is a gaussian pulse with sigma {self.user_sigma} cycles")
            # print('user length:', self.user_length)
            self.add_gauss(ch=self.cavity_ch[qTest], name="user_test",
                       sigma=self.user_sigma, length=self.user_sigma*4)

        # load the slow pulse waveform
        _sigma = cfg.device.qubit.pulses.slow_pi_ge.sigma[qTest]
        sigma_2_cycles = self.us2cycles(_sigma, gen_ch=self.qubit_chs[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="slow_pi_ge",
                       sigma=sigma_2_cycles, length=sigma_2_cycles*4)

        # initialize wait registers
        # self.safe_regwi(self.phase_update_page[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        # #self.safe_regwi(self.flux_rps, self.r_wait_flux, self.us2cycles(cfg.expt.start))
        # self.safe_regwi(self.phase_update_page[qTest], self.r_phase2, self.deg2reg(0))
        # self.safe_regwi(self.phase_update_page[qTest], self.r_phase3, 0)
        # self.safe_regwi(self.phase_update_page[qTest], self.r_phase4 , 0)

        self.sync_all(200)
        self.parity_meas_pulse = self.get_parity_str(self.cfg.expt.man_mode_no, return_pulse=True, second_phase=180, fast = False)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # reset and sync all channels
        self.reset_and_sync()

        # active reset
        if self.cfg.expt.active_reset:
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        # pre pulse
        if cfg.expt.prepulse:
            print('pre pulse')
            # print(cfg.expt.pre_sweep_pulse)
            if cfg.expt.gate_based:
                print('gate based prepulse')
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else:
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')

        # play the prepulse for kerr experiment (displacement of manipulate)
        if self.cfg.user_defined_pulse[0]:
            if "prep_e_first" in self.cfg.expt.keys() and self.cfg.expt.prep_e_first:
                print('prep e first')
                _prepulse = [['qubit', 'ge', 'pi', 0]]
                creator = self.get_prepulse_creator(_prepulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre')

            if self.user_length == 0: # its a gaussian pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest],
                                     style="arb",
                                     freq=self.user_freq,
                                     phase=self.deg2reg(0, gen_ch=self.cavity_ch[qTest]),
                                     gain=self.user_gain,
                                     waveform="user_test")
            else: # its a flat top pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest],
                                     style="flat_top",
                                     freq=self.user_freq,
                                     phase=0,
                                     gain=self.user_gain,
                                     length=self.user_length,
                                     waveform="user_test")
            self.sync_all(self.us2cycles(0.01))

        if cfg.expt.storage_ramsey[0]:
            # sweep_pulse = [['storage', 'M'+ str(self.cfg.expt.man_idx) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi'], ]
            # creator = self.get_prepulse_creator(sweep_pulse)
            self.custom_pulse(self.cfg, self.creator.pulse, prefix='Storage' + str(cfg.expt.storage_ramsey[1]))
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.coupler_ramsey:
            self.custom_pulse(cfg, cfg.expt.custom_coupler_pulse, prefix='CustomCoupler')
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.man_ramsey[0]:
            # man ramsey should be true if you are swapping in a 0+1 into manipulate instead of doing displacements;
            # if displacements, then do user defined pulse
            self.custom_pulse(self.cfg, self.creator.pulse, prefix='Manipulate' + str(cfg.expt.man_ramsey[1]))
            self.sync_all(self.us2cycles(0.01))


        # wait advanced wait time
        self.sync_all(self.us2cycles(0.01))
        # self.sync(self.phase_update_page[qTest], self.r_wait)
        ecfg = self.cfg.expt
        if ecfg.drive_coupler:
            m1c_freq = self.swap_ds.get_freq('M1-C')
            kerr_pulse = [
                [m1c_freq + ecfg.kerr_detune],
                [ecfg.kerr_gain],
                [ecfg.kerr_length],
                [0],
                [self.cfg.hw.soc.dacs.flux_low.ch[0]],
                ['flat_top'],
                [0.005],
            ]
            self.cfg.expt.m1c_freq = m1c_freq # just so we can double check in the h5 file
        else:
            kerr_pulse = [
                [self.cfg.device.qubit.f_ge[qTest] + ecfg.kerr_detune],
                [ecfg.kerr_gain],
                [ecfg.kerr_length],
                [0],
                [self.qubit_chs[qTest]],
                ['flat_top'],
                [self.cfg.device.qubit.ramp_sigma[qTest]],
            ]
        self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
        self.sync_all(self.us2cycles(0.01))

        # echoes
        if cfg.expt.echoes[0]:
            for i in range(cfg.expt.echoes[1]):
                if cfg.expt.storage_ramsey[0] or self.cfg.expt.man_ramsey[0] :
                    self.custom_pulse(cfg, self.echo_pulse, prefix='Echo')
                else:
                    # print('echoes not supported for coupler or user defined pulses')
                    self.sync_all()
                    self.sync(self.phase_update_page[qTest], self.r_wait)
                    self.sync_all()

        # self.mathi(self.phase_update_page[qTest], self.r_phase, self.r_phase2, "+", 0)
        # self.sync_all(self.us2cycles(0.01))

        if cfg.expt.storage_ramsey[0] or self.cfg.expt.coupler_ramsey:
            self.pulse(ch=self.flux_ch[qTest])
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.man_ramsey[0]:
            self.pulse(ch=self.cavity_ch[qTest])
            self.sync_all(self.us2cycles(0.01))

        if self.cfg.user_defined_pulse[0]:
            phase_adv = cfg.expt.ramsey_freq * cfg.expt.kerr_length *360 # in degree
            if self.user_length == 0: # its a gaussian pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest],
                                     style="arb",
                                     freq=self.user_freq,
                                     phase=self.deg2reg(phase_adv, gen_ch=self.cavity_ch[qTest]),
                                     gain=self.user_gain,
                                     waveform="user_test")
            else: # its a flat top pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest],
                                     style="flat_top",
                                     freq=self.user_freq,
                                     phase=self.deg2reg(phase_adv, gen_ch=self.cavity_ch[qTest]),
                                     gain=self.user_gain,
                                     length=self.user_length,
                                     waveform="user_test")
            self.sync_all(self.us2cycles(0.01))

        # postpulse
        self.sync_all()
        if cfg.expt.postpulse:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.post_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'post_')
            else:
                self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix = 'post_')

        if not self.cfg.user_defined_pulse[0]:
            # parity measurement
            if self.cfg.expt.parity_meas:
                self.custom_pulse(self.cfg, self.parity_meas_pulse, prefix='ParityMeas')
        else:
            _freq = cfg.device.qubit.f_ge[qTest]
            _phase = 0
            _gain = cfg.device.qubit.pulses.slow_pi_ge.gain[qTest]
            _sigma = cfg.device.qubit.pulses.slow_pi_ge.sigma[qTest]
            _length = cfg.device.qubit.pulses.slow_pi_ge.length[qTest]
            _style = cfg.device.qubit.pulses.slow_pi_ge.type[qTest]
            freq_2_reg = self.freq2reg(_freq, gen_ch=self.qubit_chs[qTest])
            _sigma_2_cycles = self.us2cycles(_sigma, gen_ch=self.qubit_chs[qTest])
            _length_2_cycles = self.us2cycles(_length, gen_ch=self.qubit_chs[qTest])
            phase_2_reg = self.deg2reg(_phase, gen_ch=self.qubit_chs[qTest])
            # print(f'_freq: {_freq}, _phase: {_phase}, _gain: {_gain}, _length: {_length}, _style: {_style}')

            self.setup_and_pulse(ch=self.qubit_chs[qTest],
                                 style=_style,
                                 freq=freq_2_reg,
                                 phase=phase_2_reg,
                                 gain=_gain,
                                 length=_length_2_cycles,
                                 waveform="slow_pi_ge") # slow pi pulse for readout

        self.measure_wrapper()


class KerrCavityRamseyExperiment(QsimBaseExperiment):

    def analyze(self, data=None, **kwargs):


        def estimate_periodicity(y, sampling_rate=1.0):
            # Compute FFT
            fft = np.fft.fft(y - np.mean(y))  # remove DC offset
            freqs = np.fft.fftfreq(len(y), d=1/sampling_rate)

            # Only take the positive frequencies
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            power = np.abs(fft[pos_mask])

            # Find the dominant frequency
            dominant_freq = freqs[np.argmax(power)]

            # Convert frequency to period
            estimated_period = 1 / dominant_freq if dominant_freq != 0 else np.inf
            return estimated_period

        def fit_model(x, alpha2, f, scale, offset):
            """Fitting model: exp(2*alpha2*(-cos(2*pi*f*x)-1))"""
            return scale * np.exp(2 * alpha2 * (-np.cos(2 * np.pi * f * x) - 1)) + offset

        def normalize(z):
            Ig = self.cfg.device.readout.Ig[0]
            Ie = self.cfg.device.readout.Ie[0]
            return (z - Ig) / (Ie - Ig)

        x, y, z = self.data['xpts'], self.data['ypts'], normalize(self.data['avgi'])

        # Lists to collect fit results
        alpha2_fits = []
        f_fits = []
        fit_results = []
        z_fits = []
        z_smooths = []

        period_estimate = estimate_periodicity(z[0], sampling_rate=1/(x[1] - x[0]))
        f_initial = 1.0 / period_estimate if period_estimate != 0 and np.isfinite(period_estimate) else 0.1

        # Fit each line
        for lid, line in enumerate(z):
            signal_smooth = gaussian_filter1d(line, sigma=1.5)
            z_smooths.append(signal_smooth)

            alpha_guess = y[lid]*self.cfg.device.manipulate.gain_to_alpha[0]

            # Create lmfit Model
            model = Model(fit_model)

            # Set initial parameters
            params = model.make_params(
                # alpha2 = dict(value=alpha_guess**2, min=alpha_guess**2/2, max=alpha_guess**2*2),
                alpha2 = dict(value=alpha_guess**2, vary=False),
                f=f_initial,
                scale=dict(value=1, min=0.1, max=1.2),
                offset=dict(value=0, min=-0.1, max=0.5))

            # Perform fit
            result = model.fit(line, params, x=x)

            # Collect best-fit parameters
            alpha2_fits.append(result.params['alpha2'].value)
            f_fits.append(result.params['f'].value)
            fit_results.append(result)
            z_fits.append(result.best_fit)

        f_fits = np.array(f_fits)
        alpha2_fits = np.array(alpha2_fits)
        fit_rsq_threshold = kwargs.get('fit_rsq_threshold', 0.2)
        fit_good = [res.rsquared > fit_rsq_threshold for res in fit_results]

        filtered_alpha2 = alpha2_fits[fit_good]
        filtered_f = f_fits[fit_good]

        # here we deduct the virtual ramsey from fitted f
        # self.cfg.expt got erased during initialization... so extracting it another way
        cfg = json.loads(self.data['attrs']['config'])
        virtual_freq = cfg['expt']['ramsey_freq']
        kerr_gain = cfg['expt']['kerr_gain']

        alpha2_array = np.array(filtered_alpha2)
        f_array = np.array(filtered_f) - virtual_freq
        z_smooths = np.array(z_smooths)

        # Create linear model: w = kc * alpha2 + delta
        linear_model = Model(lambda x, kc, delta: kc * x + delta, independent_vars=['x'])
        linear_params = linear_model.make_params(kc=1.0, delta=0.0)
        linear_result = linear_model.fit(f_array, linear_params, x=alpha2_array)

        # Store results
        self.fit_results = {
            'alpha2': alpha2_array,
            'f': f_array,
            'results': fit_results,
            'z_fits': np.array(z_fits),
            'kc': linear_result.params['kc'].value,
            'delta': linear_result.params['delta'].value,
            'linear_fit_result': linear_result,
            'z_smooths': z_smooths,
            'fit_good': fit_good,
            'kerr_gain': kerr_gain,
        }



    def display(self, data=None, fit=True, **kwargs):
        if not data:
            data = self.data

        if not fit:
            return super().display(data, fit, **kwargs)

        # Create figure with 3 panels
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # Panel 1: Original data heatmap
        ax1 = axes[0]
        im1 = ax1.pcolormesh(self.data['xpts'], self.data['ypts'], self.data['avgi'],
                             shading='auto', cmap='viridis')
        ax1.set_xlabel('duration (us)')
        ax1.set_ylabel('displacement gain')
        ax1.set_title(f'{self.fname.split(os.path.sep)[-1]} (avgi)')
        plt.colorbar(im1, ax=ax1)

        # Panel 2: Best fit heatmap
        ax2 = axes[1]
        im2 = ax2.pcolormesh(self.data['xpts'], self.data['ypts'], self.fit_results['z_fits'],
                             shading='auto', cmap='viridis')
        ax2.set_xlabel('duration (us)')
        ax2.set_ylabel('displacement gain')
        ax2.set_title('Best Fit')
        plt.colorbar(im2, ax=ax2)
        for lid, good in enumerate(self.fit_results['fit_good']):
            if not good:
                ax2.axhline(self.data['ypts'][lid], color= 'r', ls='--', linewidth=1)

        # Panel 3: Linear fit of f vs alpha2
        ax3 = axes[2]
        alpha2_array = self.fit_results['alpha2']
        f_array = self.fit_results['f']
        kc = self.fit_results['kc']
        delta = self.fit_results['delta']

        # Scatter plot of data points
        ax3.scatter(alpha2_array, f_array*1e3, alpha=0.6, label='Data')

        # Plot best fit line
        alpha2_fit_line = np.linspace(alpha2_array.min(), alpha2_array.max(), 3)
        f_fit_line = (kc * alpha2_fit_line + delta)
        ax3.plot(alpha2_fit_line, f_fit_line*1e3, 'r-', linewidth=2, label='Linear Fit')

        ax3.set_xlabel(r'$|\alpha|^2$')
        ax3.set_ylabel(r'$f$ (kHz)')
        ax3.set_title('freq vs displacement')
        ax3.legend()

        # Add annotation with fit parameters
        textstr = f'$K_c$ = {kc*1e3:.3f} kHz\n$\delta$ = {delta*1e3:.3f} kHz'
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.2))

        plt.tight_layout()
        return fig


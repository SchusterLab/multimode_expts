# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.fit_display_classes import (
    CavityRamseyGainSweepFitting,
    GeneralFitting,
    RamseyFitting,
)
from experiments.MM_base import *
from experiments.qsim.qsim_base import *
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display import *


from experiments.qsim.kerr import *


############################################################
############################################################
############################################################
class KerrCavityRamseyExperimentMod(KerrCavityRamseyExperiment, 
                                    QsimBaseExperiment):
    @staticmethod
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
    # @staticmethod
    # def fit_model(x, alpha2, f, scale, offset):
    #     """Fitting model: exp(2*alpha2*(-cos(2*pi*f*x)-1))"""
    #     return scale * np.exp(2 * alpha2 * (-np.cos(2 * np.pi   * f * x) - 1)) + offset
    @staticmethod
    def fit_model(x, alpha2, f, tphi, scale, offset):
        return scale * np.exp(-2.0*alpha2*(1.0 + np.cos(2.0*np.pi*f*x))) + offset
        # return scale * np.exp(-2.0*alpha2*(1.0 + np.exp(-x/tphi)*np.cos(2.0*np.pi*f*x))) + offset

    @staticmethod
    def fit_func(x, kc, delta):
        return  kc * x + delta
    
    def normalize(self, z):
        Ig = self.cfg.device.readout.Ig[0]
        Ie = self.cfg.device.readout.Ie[0]
        return (z - Ig) / (Ie - Ig)
    def test_if_replaced(self):
        return None
    def analyze(self, data=None, **kwargs):

        x, y, z = self.data['xpts'], self.data['ypts'], self.normalize(self.data['avgi'])

        # Lists to collect fit results
        alpha2_fits = []
        f_fits = []
        fit_results = []
        z_fits = []
        z_smooths = []

        period_estimate = self.estimate_periodicity(z[0], sampling_rate=1/(x[1] - x[0]))
        f_initial = 1.0 / period_estimate if period_estimate != 0 and np.isfinite(period_estimate) else 0.1

        # Fit each line
        for lid, line in enumerate(z):
            signal_smooth = gaussian_filter1d(line, sigma=1.5)
            z_smooths.append(signal_smooth)

            alpha_guess = y[lid]*self.cfg.device.manipulate.gain_to_alpha[0]

            # Create lmfit Model
            model = Model(self.fit_model)

            # Set initial parameters
            # params = model.make_params(
            #     # alpha2 = dict(value=alpha_guess**2, min=alpha_guess**2/2, max=alpha_guess**2*2),
            #     alpha2 = dict(value=alpha_guess**2, vary=False),
            #     f=f_initial,
            #     scale=dict(value=1, min=0.1, max=1.2),
            #     offset=dict(value=0, min=-0.1, max=0.5))
            params = model.make_params(
                alpha2=dict(value=alpha_guess**2, vary=False),
                f=f_initial,
                tphi=dict(value=100*(x[-1]-x[0]), min=10*(x[-1]-x[0]), max=1e6),  # units are the same as x
                scale=dict(value=1, min=0.1, max=1.2),
                offset=dict(value=0, min=-0.1, max=0.5),
                )
            try:
                # Perform fit
                result = model.fit(line, params, x=x)

                # Collect best-fit parameters
                alpha2_fits.append(result.params['alpha2'].value)
                f_fits.append(result.params['f'].value)
                fit_results.append(result)
                z_fits.append(result.best_fit)
            except:
                None

        f_fits = np.array(f_fits)
        alpha2_fits = np.array(alpha2_fits)
        fit_rsq_threshold = kwargs.get('fit_rsq_threshold', 0.2)
        fit_good = [res.rsquared > fit_rsq_threshold for res in fit_results]

        filtered_alpha2 = alpha2_fits[fit_good]
        filtered_f = f_fits[fit_good]

        # here we deduct the virtual ramsey from fitted f
        # self.cfg.expt got erased during initialization... so extracting it another way
        virtual_freq = self.cfg.expt.ramsey_freq
        kerr_gain = self.cfg.expt.kerr_gain

        alpha2_array = np.array(filtered_alpha2)
        f_array = np.array(filtered_f) - virtual_freq
        z_smooths = np.array(z_smooths)
        try:
            # Create linear model: w = kc * alpha2 + delta
            linear_model = Model(self.fit_func, independent_vars=['x'])
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
        except:
            self.fit_results = {
                'alpha2': alpha2_array,
                'f': f_array,
                'kc': np.nan,
                'delta': np.nan,
                'linear_fit_result': None,
            }



############################################################
############################################################
############################################################


###ADDED AFTER KERR.PY
class KerrCavityRamseyExcursionProgram(KerrEngBaseProgram):
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

        self.swap_ds = self.cfg.device.storage._ds_storage
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

        self.phase_update_channel = self.cavity_ch
        
        self.phase_update_page = [self.ch_page(self.phase_update_channel[qTest])]
        self.r_phase = self.sreg(self.phase_update_channel[qTest], "phase")

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

        self.sync_all(200)
        self.parity_meas_pulse = self.get_parity_str(self.cfg.expt.man_mode_no, return_pulse=True, second_phase=180, fast = False)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # reset and sync all channels
        self.reset_and_sync()

        # active reset
        if self.cfg.expt.get('active_reset', False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)


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


        # wait advanced wait time
        # self.sync_all(self.us2cycles(0.01))
        # self.sync(self.phase_update_page[qTest], self.r_wait)
        ecfg = self.cfg.expt
        kerr_pulse = [
            [ecfg.kerr_freq],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.cfg.hw.soc.dacs.flux_low.ch[0]],
            ['flat_top'],
            [0.005],
        ]
        _ch_kerr = kerr_pulse[4][0]
        self.sync_all()
        self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
        # self.setup_and_pulse(ch = _ch_kerr,
        #                      style = "const",
        #                      freq = self.freq2reg(kerr_pulse[0][0], gen_ch = _ch_kerr),
        #                      phase = self.deg2reg(kerr_pulse[3][0], gen_ch = _ch_kerr),
        #                      gain = kerr_pulse[1][0],
        #                      length = self.us2cycles(kerr_pulse[2][0]),
        #                      phrst = 1
        #                      )
        self.sync_all()
        self.sync_all(self.us2cycles(0.01))


        if self.cfg.user_defined_pulse[0]:
            if cfg.expt.storage_ramsey[0] and cfg.expt.storage_ramsey[2]:
                phase_adv = 0 
            else:
                phase_adv = (cfg.expt.ramsey_freq * cfg.expt.kerr_length * 360) % 360
            # phase_adv = cfg.expt.ramsey_freq * cfg.expt.kerr_length *360 # in degree
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
            # _freq = cfg.device.qubit.f_ge[qTest]
            # _phase = 0
            # _gain = cfg.device.qubit.pulses.slow_pi_ge.gain[qTest]
            # _sigma = cfg.device.qubit.pulses.slow_pi_ge.sigma[qTest]
            # _length = cfg.device.qubit.pulses.slow_pi_ge.length[qTest]
            # _style = cfg.device.qubit.pulses.slow_pi_ge.type[qTest]
            # freq_2_reg = self.freq2reg(_freq, gen_ch=self.qubit_chs[qTest])
            # _sigma_2_cycles = self.us2cycles(_sigma, gen_ch=self.qubit_chs[qTest])
            # _length_2_cycles = self.us2cycles(_length, gen_ch=self.qubit_chs[qTest])
            # phase_2_reg = self.deg2reg(_phase, gen_ch=self.qubit_chs[qTest])
            # print(f'_freq: {_freq}, _phase: {_phase}, _gain: {_gain}, _length: {_length}, _style: {_style}')
            gain = cfg.device.qubit.pulses.very_slow_pi_ge.gain[qTest]
            length = cfg.device.qubit.pulses.slow_pi_ge.length[qTest]
            style = cfg.device.qubit.pulses.slow_pi_ge.type[qTest]
            sigma = cfg.device.qubit.pulses.very_slow_pi_ge.sigma[qTest]
            pulse_data = [
                [cfg.device.qubit.f_ge[qTest]],     # frequency
                [gain],                             # gain
                [length],                           # length (us)
                [0],                                # phase
                [self.qubit_chs[qTest]],            # drive channel
                [style],                            # shape
                [sigma]
            ]
            self.custom_pulse(cfg, pulse_data, prefix='slow_ge_rabi')
            self.sync_all()
            # self.setup_and_pulse(ch=self.qubit_chs[qTest],
            #                      style=_style,
            #                      freq=freq_2_reg,
            #                      phase=phase_2_reg,
            #                      gain=_gain,
            #                      length=_length_2_cycles,
            #                      waveform="slow_pi_ge") # slow pi pulse for readout

        self.measure_wrapper()


class KerrCavityRamseyExcursionExperiment(KerrCavityRamseyExperimentMod):
# class KerrCavityRamseyExcursionExperiment(KerrCavityRamseyExperiment):
    def __init__(self, *args, **kwargs):
        kwargs["program"] = KerrCavityRamseyExcursionProgram
        super().__init__(*args, **kwargs)
        
    def fitter_compatibility(self):
        data = self.data
        data['gain_list'] = np.array(self.cfg.expt.displace_gains)
        data['g_avgi'] = np.array(data['avgi'])
        data['e_avgi'] = np.zeros(np.shape(data['g_avgi']))
        # _dummy = np.zeros((2, len(data['xpts']))) 
        # _dummy[0] = data['xpts']
        # data['xpts'] = _dummy
        data['xpts'] = np.atleast_2d(data['xpts'])
        self.cfg.expt.do_g_and_e = False
        
        
    def analyze_peak(self, data=None, fit=True, **kwargs):
        self.fitter_compatibility()
        if data is None:
            data = self.data
        print(f"data['gain_list']")
        
        if fit: 
            cavity_ramsey_analysis = CavityRamseyGainSweepFitting(
                data, config=self.cfg, 
            )
            # forward any selection/debug kwargs to the fitter
            cavity_ramsey_analysis.analyze(fit=fit, **kwargs)

        return cavity_ramsey_analysis.data


    def display_peak(self, data=None, **kwargs):

        if data is None:
            data=self.data

        cavity_ramsey_analysis = CavityRamseyGainSweepFitting(
            data, config=self.cfg,
        )

        if "save_fig" in kwargs and kwargs["save_fig"]:
            save_fit = True
        else:
            save_fit = False

        # forward any extra kwargs to display as well
        cavity_ramsey_analysis.display(
            save_fig=save_fit,
            **{k: v for k, v in kwargs.items() if k != 'save_fig'}
        )
    # def analyze(self, data=None, debug=False, **kwargs):


    #     def estimate_periodicity(y, sampling_rate=1.0):
    #         # Compute FFT
    #         fft = np.fft.fft(y - np.mean(y))  # remove DC offset
    #         freqs = np.fft.fftfreq(len(y), d=1/sampling_rate)

    #         # Only take the positive frequencies
    #         pos_mask = freqs > 0
    #         freqs = freqs[pos_mask]
    #         power = np.abs(fft[pos_mask])

    #         # Find the dominant frequency
    #         dominant_freq = freqs[np.argmax(power)]

    #         # Convert frequency to period
    #         estimated_period = 1 / dominant_freq if dominant_freq != 0 else np.inf
    #         return estimated_period
        
    #     def estimate_phase(y):
    #         print("y[0]", y[0], "min", np.min(y), "max", np.max(y))
    #         return np.abs(y[0] - np.min(y)) / np.abs(np.max(y) - np.min(y)) * np.pi/2
    #         # if np.abs(y[0] - np.min(y)) < np.abs(y[0] - np.max(y)):
    #         #     return 0
    #         # return np.pi/2

    #     def fit_model(x, alpha2, f, scale, offset, phase):
    #         """Fitting model: exp(2*alpha2*(-cos(2*pi*f*x)-1))"""
    #         return scale * np.exp(2 * alpha2 * (-np.cos(2 * np.pi * f * x - phase) - 1)) + offset

    #     def normalize(z):
    #         Ig = self.cfg.device.readout.Ig[0]
    #         Ie = self.cfg.device.readout.Ie[0]
    #         return (z - Ig) / (Ie - Ig)

    #     x, y, z = self.data['xpts'], self.data['ypts'], normalize(self.data['avgi'])

    #     # Lists to collect fit results
    #     alpha2_fits = []
    #     f_fits = []
    #     fit_results = []
    #     z_fits = []
    #     z_smooths = []

    #     period_estimate = estimate_periodicity(z[0], sampling_rate=1/(x[1] - x[0]))
    #     f_initial = 1.0 / period_estimate if period_estimate != 0 and np.isfinite(period_estimate) else 0.1

    #     # Fit each line
    #     for lid, line in enumerate(z):
    #         signal_smooth = gaussian_filter1d(line, sigma=1.5)
    #         z_smooths.append(signal_smooth)

    #         alpha_guess = y[lid]*self.cfg.device.manipulate.gain_to_alpha[0]
    #         phase_guess = estimate_phase(line)
    #         scale_guess = np.max(line) - np.min(line)

    #         # Create lmfit Model
    #         model = Model(fit_model)

    #         # Set initial parameters
    #         params = model.make_params(
    #             # alpha2 = dict(value=alpha_guess**2, min=alpha_guess**2/2, max=alpha_guess**2*2),
    #             alpha2 = dict(value=alpha_guess**2, vary=False),
    #             f=f_initial,
    #             scale=dict(value=scale_guess, min=0.75*scale_guess, max=1.25*scale_guess),
    #             offset=dict(value=0, min=-0.1, max=0.5),
    #             phase=dict(value=phase_guess, min=0, max=np.pi)
    #             )

    #         # Perform fit
    #         result = model.fit(line, params, x=x)

    #         # Collect best-fit parameters
    #         alpha2_fits.append(result.params['alpha2'].value)
    #         f_fits.append(result.params['f'].value)
    #         fit_results.append(result)
    #         z_fits.append(result.best_fit)

    #         if debug:
    #             fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    #             ax.plot(x, line, '.-', markersize=3, label='data')
    #             ax.plot(x, result.best_fit, 'r-', label='fit')
    #             ax.set_title(f'line {lid}: gain={y[lid]:.0f}, '
    #                          f'f={result.params["f"].value:.4f}, '
    #                          f'R²={result.rsquared:.3f}')
    #             ax.set_xlabel('duration (us)')
    #             ax.legend(fontsize=8)
    #             plt.tight_layout()
    #             plt.show()

    #     f_fits = np.array(f_fits)
    #     alpha2_fits = np.array(alpha2_fits)
    #     fit_rsq_threshold = kwargs.get('fit_rsq_threshold', 0.2)
    #     fit_good = [res.rsquared > fit_rsq_threshold for res in fit_results]

    #     filtered_alpha2 = alpha2_fits[fit_good]
    #     filtered_f = f_fits[fit_good]

    #     # here we deduct the virtual ramsey from fitted f
    #     # self.cfg.expt got erased during initialization... so extracting it another way
    #     cfg = self.cfg
    #     virtual_freq = cfg['expt']['ramsey_freq']
    #     kerr_gain = cfg['expt']['kerr_gain']

    #     alpha2_array = np.array(filtered_alpha2)
    #     f_array = np.array(filtered_f) - virtual_freq
    #     z_smooths = np.array(z_smooths)

    #     # Create linear model: w = kc * alpha2 + delta
    #     linear_model = Model(lambda x, kc, delta: kc * x + delta, independent_vars=['x'])
    #     linear_params = linear_model.make_params(kc=1.0, delta=0.0)
    #     linear_result = linear_model.fit(f_array, linear_params, x=alpha2_array)

    #     # Store results
    #     self.fit_results = {
    #         'alpha2': alpha2_array,
    #         'f': f_array,
    #         'results': fit_results,
    #         'z_fits': np.array(z_fits),
    #         'kc': linear_result.params['kc'].value,
    #         'delta': linear_result.params['delta'].value,
    #         'linear_fit_result': linear_result,
    #         'z_smooths': z_smooths,
    #         'fit_good': fit_good,
    #         'kerr_gain': kerr_gain,
    #     }

        


##################################################################################
##########Slow-pi-ge-calibration##################################################
##################################################################################
import fitting.fitting as fitter
from fitting.fit_display_classes import AmplitudeRabiFitting




class SlowLengthRabiProgram(QsimBaseProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def core_pulses(self):
        cfg=AttrDict(self.cfg)

        qTest = self.qubits[0]

        gain = cfg.device.qubit.pulses.very_slow_pi_ge.gain[qTest]
        # length = cfg.device.qubit.pulses.very_slow_pi_ge.length[qTest]
        # length = self.us2cycles(, gen_ch=self.qubit_chs[qTest])
        length = self.cfg.length_to_sweep
        style = cfg.device.qubit.pulses.slow_pi_ge.type[qTest]
        sigma = cfg.device.qubit.pulses.very_slow_pi_ge.sigma[qTest]
        print(f"slow ge pulse params: gain {gain}, length {length} us, sigma {sigma} us")

        pulse_data = [
            [cfg.device.qubit.f_ge[qTest]],     # frequency
            [gain],                             # gain
            [length],                           # length (us)
            [0],                                # phase
            [self.qubit_chs[qTest]],            # drive channel
            [style],                            # shape
            [sigma],                            # ramp sigma
        ]
        self.custom_pulse(cfg, pulse_data, prefix='slow_ge_rabi')
        self.sync_all()

class SlowLengthRabiQsimExperiment(QsimBaseExperiment):
    def __init__(self, *args, **kwargs):
        kwargs["program"] = SlowLengthRabiProgram
        super().__init__(*args, **kwargs)


############################################################################################
############    Debugging Programs            ##############################################
############################################################################################

class ParityDebuggingProgram(QsimBaseProgram):
    def core_pulses(self):
        pulse_cfg = [
            ['qubit', 'ge', 'pi', 0,],
            ['man', 'M1', 'pi', 0]
        ]
        pulse = self.get_prepulse_creator(pulse_cfg)
        self.sync_all()
        cfg = AttrDict(self.cfg)
        self.custom_pulse(cfg, pulse.pulse, prefix = 'test_')
        self.sync_all()
        # self.sync_all(self.us2cycles(20))
        
class ActiveResetVerificationProgram(QsimBaseProgram):
    def initialize(self):
        """
        MM_base_init to pull basic info 
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulses
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        #TODO: this should use a config key to determine whether
        # to use floquet or gate (pi or pi/2) datasets
        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1  # using first manipulate channel index needs to be fixed at some point

        self.m1s_kwargs = [{
                'ch': self.m1s_ch[stor],
                'style': 'flat_top',
                'freq': self.m1s_freq[stor],
                'phase': 0,
                'gain': self.m1s_gain[stor],
                'length': self.m1s_length[stor],
                'waveform': self.m1s_wf_name[stor],
        } for stor in range(7)]

        if self.cfg.expt.perform_wigner:
            self.displace_man(setup=True, play=False)

        self.sync_all(200)
    
    def core_pulses(self):
        pulse_cfg = self.prep_man_photon(1)
        # pulse_cfg2 = [
        #     ['man', 'M1', 'pi', 0]
        # ]
        pulse = self.get_prepulse_creator(pulse_cfg)
        # pulse2 = self.get_prepulse_creator(pulse_cfg2)
        self.sync_all()
        self.custom_pulse(AttrDict(self.cfg), pulse.pulse, prefix = 'prep_man_1')
        self.sync_all()
        # self.custom_pulse(AttrDict(self.cfg), pulse2.pulse, prefix = 'prep_man_1_f0g1')
        # self.sync_all()
        
class MActiveResetVerificationProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)
        
    def initialize(self):
        """
        MM_base_init to pull basic info 
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulses
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1  

        self.sync_all(200)
        
    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.cfg.expt.qubits[0]

        # phase reset
        self.reset_and_sync()
        pulse_cfg = self.prep_man_photon(1)
        pulse = self.get_prepulse_creator(pulse_cfg)
        self.custom_pulse(AttrDict(self.cfg), pulse.pulse, prefix = 'prep_man_1_initializer')
        self.sync_all()
        #do the active reset
        if cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        #  prepulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        self.sync_all()  # align channels

        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post')
        self.sync_all(self.us2cycles(0.05))
        self.measure_wrapper()
    

class ExcursionTransitionDebuggingProgram(QsimBaseProgram):

    def core_pulses(self):
        cfg=AttrDict(self.cfg)
        ecfg = self.cfg.expt
        qTest = self.qubits[0]

        kerr_pulse = [
            [ecfg.kerr_freq],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.cfg.hw.soc.dacs.flux_low.ch[0]],
            ['flat_top'],
            [0.005],
        ]
        _ch_kerr = kerr_pulse[4][0]
        
        pulse_cfg = self.prep_man_photon(1)
        prep_man_pulse = self.get_prepulse_creator(pulse_cfg)
        
        if not ecfg.get("skip_all", False):
            if not ecfg.get("prepulse", False):
                if ecfg.get("debug", False):
                    print("performing prep manipulate photon pulse")
                self.sync_all()
                self.custom_pulse(AttrDict(self.cfg), prep_man_pulse.pulse.tolist(), prefix = 'prep_man_1')
            self.sync_all()
            
            if not ecfg.get("skip_kerr_pulse", False):
                if ecfg.get("debug", False):
                    print("performing kerr pulse")
                if ecfg.get("use_flat_top", True):
                    self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
                else:
                    if ecfg.get("debug", False):
                        print("pulsing const")
                    self.setup_and_pulse(ch = _ch_kerr,
                                    style = "const",
                                    freq = self.freq2reg(kerr_pulse[0][0], gen_ch = _ch_kerr),
                                    phase = self.deg2reg(kerr_pulse[3][0], gen_ch = _ch_kerr),
                                    gain = kerr_pulse[1][0],
                                    length = self.us2cycles(kerr_pulse[2][0]),
                                    phrst = 1
                                    )
                self.sync_all()
            if not ecfg.get("perform_wigner", False):
                if ecfg.get("debug", False):
                    print("performing man pi and qubit ef")
                pulse_cfg2 = [
                    ['man', 'M1', 'pi', 0],
                    ['qubit', 'ef', 'pi', 0]
                ]
                pulse_f0g1 = self.get_prepulse_creator(pulse_cfg2)
                self.custom_pulse(AttrDict(self.cfg), pulse_f0g1.pulse.tolist(), prefix = 'man_to_f0g1')
                self.sync_all()
        else:
            if ecfg.get("debug", False):
                print("skipping all pulses, only measuring")
            self.sync_all()

        

class Qsimf0g1Sepctroscopy(QsimBaseProgram):
    def initialize(self):
        super().initialize()
        # flux line modulation
        if self.cfg.expt.get("modulate_flux", False):
            qTest = self.qubits[0]
            _flux_ch = self.flux_low_ch[qTest] #from the parse_config method of MM base
            self.setup_and_pulse(ch=_flux_ch,
                                style="const",
                                freq=self.freq2reg(self.cfg.expt.flux_freq, gen_ch = _flux_ch), 
                                phase=0,
                                gain=self.cfg.expt.flux_drive_gain,
                                length=self.us2cycles(self.cfg.expt.length),
                                mode = "periodic")
        self.sync_all(10)

    def core_pulses(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]
        self.sync_all()
        _f_load_pulse = self.prep_man_photon(1)[0:2:1]
        _load_f_state = self.get_prepulse_creator(_f_load_pulse).pulse.tolist()
        # _man_unload_pulse = _man_load_pulse[-1:-3:-1] #reverse until ge
        # _unload_manipulate = self.get_prepulse_creator(_man_unload_pulse).pulse.tolist()
        self.custom_pulse(cfg, _load_f_state, prefix = "Load_Manipulate")
        self.sync_all()
        # f0g1 spectroscopy
        self.setup_and_pulse(
            ch=self.f0g1_ch[qTest],
            style="const",
            freq= self.freq2reg(self.cfg.expt.freq, gen_ch = self.f0g1_ch[qTest]), #sweep_params should be 'freqs'
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.f0g1_ch[qTest]))
        
        self.sync_all()  # align channels

        # post pulse
        # self.custom_pulse(cfg, _unload_manipulate, prefix = 'Unload_Manipulate')
        self.sync_all(self.us2cycles(0.05))
        
class Manf0g1RamseyProgram(QsimBaseProgram):

    def core_pulses(self):
        cfg = AttrDict(self.cfg)
        ecfg = self.cfg.expt
        
        qTest = self.qubits[0]
        
        self.sync_all()
        
        #----configuring pulse sequence
        prep_and_man_hpi_seq = [
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ef', 'pi', 0],
            ['man', 'M1', 'hpi', 0]
        ]
        man_hpi_seq = [
            ['man', 'M1', 'hpi', 0],
            ['qubit', 'ef', 'pi', 0],
        ]
        if ecfg.get("do_virtual_ramsey", False):
            virtual_ramsey_freq = ecfg.virtual_ramsey_freq
            if not ecfg.get("use_clock", False):
                kerr_length = ecfg.kerr_length
            else:
                kerr_length = self.cycles2us(ecfg.kerr_length) # convert us to clock cycle
                if ecfg.get("debug", False):
                    print(f"using clock for kerr length. the time is {kerr_length}")
            virtual_ramsey_phase = (virtual_ramsey_freq * kerr_length * 360) % 360
            man_hpi_seq[0][3] = virtual_ramsey_phase
            if ecfg.get("debug", False):
                print(f"Applying virtual ramsey phase shift of {virtual_ramsey_phase} degrees to compensate for frequency detuning during kerr pulse")
                print(man_hpi_seq)
        kerr_pulse = [
            [ecfg.kerr_freq],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.cfg.hw.soc.dacs.flux_low.ch[0]],
            ['flat_top'],
            [0.005],
        ]
        
        prep_and_man_hpi_pulse = self.get_prepulse_creator(prep_and_man_hpi_seq)
        self.custom_pulse(cfg, prep_and_man_hpi_pulse.pulse.tolist(), prefix = "Prep_and_Manipulate_half_pi")
        self.sync_all()
        
        _ch_kerr = kerr_pulse[4][0]
        self.sync_all()
        #--------- Ramsey wait is now controlled by kerr excursion length.
        if ecfg.get("use_flat_top", True):
            self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
        else:
            if not ecfg.get("use_clock", False):
                _length_of_kerr = self.us2cycles(kerr_pulse[2][0])
            else:
                _length_of_kerr = max(kerr_pulse[2][0], 3) # minimal clock cycles required for RFSoC
                if ecfg.get("debug", False):
                    print("using clock for kerr length. the time is ", self.cycles2us(_length_of_kerr))
            self.setup_and_pulse(ch = _ch_kerr,
                                style = "const",
                                freq = self.freq2reg(kerr_pulse[0][0], gen_ch = _ch_kerr),
                                phase = self.deg2reg(kerr_pulse[3][0], gen_ch = _ch_kerr),
                                gain = kerr_pulse[1][0],
                                length = _length_of_kerr,
                                phrst = 1
                                )
        self.sync_all()
        if not self.cfg.expt.get("postpulse", True):
            if self.cfg.expt.get("debug", False):
                print("playing man_hpi_seq")
            man_hpi_pulse = self.get_prepulse_creator(man_hpi_seq).pulse.tolist()
            self.custom_pulse(cfg, man_hpi_pulse, prefix = "Manipulate_half_pi")
            self.sync_all()

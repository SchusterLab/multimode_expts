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
############################################################
############################################################
############################################################


###ADDED AFTER KERR.PY
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




############################################################################################
############    Debugging Programs            ##############################################
############################################################################################

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
        
####################################################################################################################################################
######################## Codes for Darkmode Search. To be moved to somewhere########################################################################
####################################################################################################################################################

import fitting.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram


from experiments.qsim.sideband_scramble import SidebandScrambleProgram

class SidebandScrambleDarkProgram(SidebandScrambleProgram):
    
    def man_reset(self, man_idx=1, dump_mode_idx=2, chi_dressed=True):
        '''
        Reset manipulate mode by swapping it to lossy mode

        chi_dressed: if man freq shifted due to pop in qubit e, f states.
        using_qubit: if True, we do g1-f0/ef/qubit reset instead of using the dump, which is not indeal since it remove only the fock 1 population but can be usefull if dump cannot be found 
        '''
        if self.cfg.expt.get("debug", False):
            print("overrided man reset is called")
        qTest = 0
        cfg=AttrDict(self.cfg)

        MiDj_freq = self.dataset.get_freq(f'M{man_idx}-D{dump_mode_idx}')
        MiDj_gain = self.dataset.get_gain(f'M{man_idx}-D{dump_mode_idx}')
        MiDj_length = self.dataset.get_pi(f'M{man_idx}-D{dump_mode_idx}')
        N = 2 if chi_dressed else 0
        chi_ge = cfg.device.manipulate.chi_ge[qTest]
        chi_ef = cfg.device.manipulate.chi_ef[qTest]

        self.sideband_sigma_high = self.us2cycles(self.cfg.device.storage.ramp_sigma, gen_ch=self.flux_high_ch[qTest])
        self.add_gauss(ch=self.flux_high_ch[qTest],
                    name="ramp_high",# + str(man_idx),
                    sigma=self.sideband_sigma_high,
                    length=self.sideband_sigma_high*6) # M1-x flat tops use 6 sigma
        # self.wait_all(self.us2cycles(0.1))
        self.sync_all(self.us2cycles(0.1))

        chis = [chi_ge, chi_ge+chi_ef] if chi_dressed else [0]
        ch = self.flux_high_ch[qTest]
        iter_num = self.cfg.expt.get("dump_reset_iter_num", 1)
        for n in range(0, N+1): # works when MiDj freq goes down (chi<0, bare freq+chi*n)
            for chi in chis:
                for _ in range(iter_num):
                    freq_chi_shifted = MiDj_freq + (n * chi)
                    # if cfg.expt.get("man_reset_print", True):
                    #     print(ch, freq_chi_shifted, MiDj_length, MiDj_gain)
                    self.set_pulse_registers(
                        ch=ch,
                        freq=self.freq2reg(freq_chi_shifted, gen_ch=ch),
                        style="flat_top",
                        phase=self.deg2reg(0),
                        length=self.us2cycles(MiDj_length, gen_ch=ch),
                        gain=MiDj_gain,
                        waveform="ramp_high"
                        )
                    self.pulse(ch=ch)
                    self.sync_all()
                # self.sync_all(self.us2cycles(0.025))
        # self.wait_all(self.us2cycles(0.25))
        self.sync_all(self.us2cycles(2))
    
    
    def core_pulses(self):
        super().core_pulses() #already has sync_all at the last
        if self.cfg.expt.get("swap_man_dark", False):
            swap_stors = self.cfg.expt.swap_stors
            swap_stor_phases = [0.0] * len(swap_stors)

            if self.cfg.expt.update_phases:
                for _ in range(self.cfg.expt.floquet_cycle):
                    for i_stor, stor in enumerate(swap_stors):
                        for j_stor, stor_B in enumerate(swap_stors):
                            if stor_B != stor:
                                stor_B_name = f"M1-S{stor_B}"
                                stor_name = f"M1-S{stor}"
                                swap_stor_phases[j_stor] += self.swap_ds.get_phase_from(stor_B_name, stor_name)
                                swap_stor_phases[j_stor] = swap_stor_phases[j_stor] % 360
            
            stor_first, stor_last = self.cfg.expt.dark_swap_order
            list_index_start = swap_stors.index(stor_first)
            list_index_last = swap_stors.index(stor_last)

            n_first = self.m1s_pi_fracs[stor_first - 1] 
            n_last = self.m1s_pi_fracs[stor_last - 1] // 2
            first_stor_name = f"M1-S{stor_first}"
            last_stor_name = f"M1-S{stor_last}"

            if self.cfg.expt.get("second_rel_phase", 0) != 0:
                swap_stor_phases[list_index_last] += self.cfg.expt.second_rel_phase
                swap_stor_phases[list_index_last] = swap_stor_phases[list_index_last] % 360

            first_pulse_args = deepcopy(self.m1s_kwargs[stor_first - 1])
            second_pulse_args = deepcopy(self.m1s_kwargs[stor_last - 1])

            for _ in range(n_first): # full swap and phase update
                first_pulse_args['phase'] = self.deg2reg(swap_stor_phases[list_index_start], gen_ch=first_pulse_args['ch'])
                self.setup_and_pulse(**first_pulse_args)
                swap_stor_phases[list_index_last] += self.swap_ds.get_phase_from(last_stor_name, first_stor_name)
                swap_stor_phases[list_index_last] = swap_stor_phases[list_index_last] % 360
                self.sync_all(10)

            for _ in range(n_last):
                second_pulse_args['phase'] = self.deg2reg(swap_stor_phases[list_index_last], gen_ch=second_pulse_args['ch'])
                self.setup_and_pulse(**second_pulse_args)
                self.sync_all(10)
            
            self.sync_all()

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import *
from experiments.MM_dual_rail_base import MM_dual_rail_base

class CavityRamseyProgram(MMRAveragerProgram):
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
        qTest = self.qubits[0] # get the qubit we are testing

        if cfg.expt.user_defined_pulse[5] == 1:
            self.cavity_ch = self.flux_low_ch
            self.cavity_ch_types = self.flux_low_ch_type
        elif cfg.expt.user_defined_pulse[5] == 2:
            self.cavity_ch= self.qubit_chs
            self.cavity_ch_types = self.qubit_ch_type
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
        
        
        self.q_rps = [self.ch_page(ch) for ch in self.cavity_ch] # get register page for f0g1 channel
        self.stor_rps = 0 # get register page for storage channel
        if self.cfg.expt.storage_ramsey[0]: 
            # decide which channel do we flux drive on 
            sweep_pulse = [['storage', 'M'+ str(self.cfg.expt.man_idx) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi', 0], 
                       ] 
            self.creator = self.get_prepulse_creator(sweep_pulse)
            freq = self.creator.pulse[0][0]
            self.flux_ch = self. flux_low_ch 
            if freq > 1000: self.flux_ch = self.flux_high_ch

            # get register page for that channel 
            self.flux_rps = [self.ch_page(self.flux_ch[qTest])]
        if self.cfg.expt.man_ramsey[0]: 
            sweep_pulse = [['man', 'M'+ str(self.cfg.expt.man_ramsey[1]) , 'pi', 0], 
                       ] 
            self.creator = self.get_prepulse_creator(sweep_pulse)

        
        if self.cfg.expt.coupler_ramsey: 
            # decide which channel do we flux drive on 
            pulse_str = self.cfg.expt.custom_coupler_pulse
            freq = pulse_str[0][0]
            self.flux_ch = self. flux_low_ch 
            if freq > 1000: self.flux_ch = self.flux_high_ch

            # get register page for that channel 
            self.flux_rps = [self.ch_page(self.flux_ch[qTest])]
        # if self.cfg.expt.custom_coupler_pulse[0]:
        #     self.ramse

        if self.cfg.expt.echoes[0]: 
            mm_base_dummy = MM_dual_rail_base(self.cfg)
            if self.cfg.expt.storage_ramsey[0]:
                prep_stor = mm_base_dummy.prep_random_state_mode(3, self.cfg.expt.storage_ramsey[1])  # prepare the storage state + 
            elif self.cfg.expt.man_ramsey[0]:
                prep_stor = mm_base_dummy.prep_man_photon(man_no=self.cfg.expt.man_ramsey[1], hpi = True)
            get_stor = prep_stor[::-1] # get the storage state
            self.echo_pulse_str = get_stor + prep_stor # echo pulse is the sum of the two pulse sequences
            self.echo_pulse = self.get_prepulse_creator(self.echo_pulse_str).pulse.tolist()
            print(self.echo_pulse)

        

        # declare registers for phase incrementing
        self.r_wait = 3
        self.r_wait_flux = 3
        self.r_phase2 = 4
        self.r_phase3 = 0
        self.r_phase4 = 6

        if (self.cfg.expt.storage_ramsey[0] and self.cfg.expt.storage_ramsey[2]) or self.cfg.expt.coupler_ramsey:
            self.phase_update_channel = self.flux_ch
            # self.q_rps = self.flux_rps
        elif self.cfg.expt.man_ramsey[0]:
            self.phase_update_channel = self.cavity_ch

        elif self.cfg.expt.user_defined_pulse[0] and self.cfg.expt.storage_ramsey[0]:
            print('Running Kerr; will update phase ch')
            self.phase_update_channel = self.cavity_ch
            # if 
        elif self.cfg.expt.user_defined_pulse[0] :
            print('Running f0g1 ramsey')
            self.phase_update_channel = self.cavity_ch
            # if 
        print(f'phase update channel: {self.phase_update_channel}')
        self.phase_update_page = [self.ch_page(self.phase_update_channel[qTest])]
        self.r_phase = self.sreg(self.phase_update_channel[qTest], "phase")

        self.current_phase = 0   # in degree



        #for user defined 
        if cfg.expt.user_defined_pulse[0]:
            self.user_freq = self.freq2reg(cfg.expt.user_defined_pulse[1], gen_ch=self.cavity_ch[qTest])
            self.user_gain = cfg.expt.user_defined_pulse[2]
            self.user_sigma = self.us2cycles(cfg.expt.user_defined_pulse[3], gen_ch=self.cavity_ch[qTest])
            self.user_length  = self.us2cycles(cfg.expt.user_defined_pulse[4], gen_ch=self.cavity_ch[qTest])
            self.add_gauss(ch=self.cavity_ch[qTest], name="user_test",
                       sigma=self.user_sigma, length=self.user_sigma*4)
        
        # qubit pi and hpi pulse 
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)


        # initialize wait registers
        self.safe_regwi(self.phase_update_page[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        #self.safe_regwi(self.flux_rps, self.r_wait_flux, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase2, self.deg2reg(0)) 
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase3, 0) 
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase4 , 0) 

        

        self.sync_all(200)

    
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
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')

        # play the prepulse for kerr experimenty (dispalcement of manipulate)
        if self.cfg.user_defined_pulse[0]:
            if self.user_length == 0: # its a gaussian pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest], style="arb", freq=self.user_freq, phase=self.deg2reg(0), gain=self.user_gain,waveform="user_test")
            else: # its a flat top pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest], style="flat_top", freq=self.user_freq, phase=0, gain=self.user_gain, length=self.user_length, waveform="user_test")
            self.sync_all(self.us2cycles(0.01))

        if cfg.expt.storage_ramsey[0]:
            # sweep_pulse = [['storage', 'M'+ str(self.cfg.expt.man_idx) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi'], 
            #            ]
            # creator = self.get_prepulse_creator(sweep_pulse)
            self.custom_pulse(self.cfg, self.creator.pulse, prefix='Storage' + str(cfg.expt.storage_ramsey[1]))
            self.sync_all(self.us2cycles(0.01))
            print(self.creator.pulse)
            print(self.flux_ch)
        elif self.cfg.expt.coupler_ramsey:
            self.custom_pulse(cfg, cfg.expt.custom_coupler_pulse, prefix='CustomCoupler')
            self.sync_all(self.us2cycles(0.01))
            print(cfg.expt.custom_coupler_pulse)
            print(self.flux_ch)
        elif self.cfg.expt.man_ramsey[0]:
            self.custom_pulse(self.cfg, self.creator.pulse, prefix='Manipulate' + str(cfg.expt.man_ramsey[1]))
            self.sync_all(self.us2cycles(0.01))
            print(self.creator.pulse)
            print(self.cavity_ch)
        


        # wait advanced wait time
        self.sync_all()
        self.sync(self.phase_update_page[qTest], self.r_wait)
        self.sync_all()

        # echoes 
        if cfg.expt.echoes[0]:
            for i in range(cfg.expt.echoes[1]):
                if cfg.expt.storage_ramsey[0] or self.cfg.expt.man_ramsey[0] :
                    self.custom_pulse(cfg, self.echo_pulse, prefix='Echo')
                else:
                    print('echoes not supported for coupler or user defined pulses')
                self.sync_all()
                self.sync(self.phase_update_page[qTest], self.r_wait)
                self.sync_all()
        
        
        self.mathi(self.phase_update_page[qTest], self.r_phase, self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))


        if cfg.expt.storage_ramsey[0] or self.cfg.expt.coupler_ramsey:
            self.pulse(ch=self.flux_ch[qTest])
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.man_ramsey[0]:   
            self.pulse(ch=self.cavity_ch[qTest])
            self.sync_all(self.us2cycles(0.01))
        
        


        if self.cfg.user_defined_pulse[0]:
            self.pulse(ch=self.cavity_ch[qTest])
            self.sync_all(self.us2cycles(0.01))

        # postpulse 
        self.sync_all()
        if cfg.expt.postpulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'post_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'post_')

        # parity measurement
        if self.cfg.expt.parity_meas: 
            parity_meas_str = [['qubit', 'ge', 'hpi'], # Starting parity meas
                       ['qubit', 'ge', 'parity_M' + str(self.cfg.expt.man_idx)], 
                       ['qubit', 'ge', 'hpi']]
            creator = self.get_prepulse_creator(parity_meas_str)
            print(creator.pulse)
            self.custom_pulse(self.cfg, creator.pulse, prefix='ParityMeas', sync_zero_const=True)
        
        # align channels and measure
        self.measure_wrapper()

    def update(self):
        '''
        Math i does not like values above 180 for the last argument 
        '''
        qTest = self.qubits[0]

        # update the phase of the LO for the second π/2 pulse
        phase_step_deg = 360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step 
        phase_step_deg = phase_step_deg % 360 # make sure it is between 0 and 360
        if phase_step_deg < 0: # given the wrapping statement above, this should never be true
            if phase_step_deg < -180:  # between -360 and -180
                phase_step_deg += 360
                logic = '+'
            else:                      # between -180 and 0
                phase_step_deg = abs(phase_step_deg)
                logic = '-'
        else:
            if phase_step_deg < 180: # between 0 and 180
                phase_step_deg = phase_step_deg 
                logic = '+'
            else:                     # between 180 and 360
                phase_step_deg = 360 - phase_step_deg
                logic = '-'
        print(f'phase step deg: {phase_step_deg}')
        print(f'phase step logic: {logic}')
        phase_step = self.deg2reg(phase_step_deg -85, gen_ch=self.phase_update_channel[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        

 
        self.mathi(self.phase_update_page[qTest], self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update the time between two π/2 pulses
        self.sync_all(self.us2cycles(0.01))
        # if self.cfg.expt.storage_ramsey[0]:
        #     self.mathi(self.flux_rps, self.r_wait_flux, self.r_wait_flux, '+', self.us2cycles(self.cfg.expt.step))
        #     self.sync_all(self.us2cycles(0.01))

        # Note that mathi only likes the last argument to be between 0 and 90!!!
        remaining_phase = phase_step_deg
        while remaining_phase != 0:
            if remaining_phase > 85: 
                phase_step = self.deg2reg(85, gen_ch=self.phase_update_channel[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
                remaining_phase -= 85
            else:
                phase_step = self.deg2reg(remaining_phase, gen_ch=self.phase_update_channel[qTest])
                remaining_phase = 0
            self.mathi(self.phase_update_page[qTest], self.r_phase2, self.r_phase2, logic, phase_step) # advance the phase of the LO for the second π/2 pulse
            
        
        self.sync_all(self.us2cycles(0.01))
   

class CavityRamseyExperiment(Experiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Ramsey', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        
        ramsey = CavityRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
        print('inide t2 cavity acquire')

        print(self.cfg.expt.expts)
        
        x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug,
                                            readouts_per_experiment=read_num)        
 
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases} 
        data['idata'], data['qdata'] = ramsey.collect_shots()      
        #print(ramsey) 
        
        # if self.cfg.expt.normalize:
        #     from experiments.single_qubit.normalize import normalize_calib
        #     g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)
            
        #     data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
        #     data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
        #     data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]
        self.data = data
        return data

    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        if data is None:
            data=self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = None
            # fitparams=[8, 0.5, 0, 20, None, None]
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            if isinstance(p_avgi, (list, np.ndarray)): data['f_adjust_ramsey_avgi'] = sorted((self.cfg.expt.ramsey_freq - p_avgi[1], self.cfg.expt.ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)): data['f_adjust_ramsey_avgq'] = sorted((self.cfg.expt.ramsey_freq - p_avgq[1], self.cfg.expt.ramsey_freq + p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)): data['f_adjust_ramsey_amps'] = sorted((self.cfg.expt.ramsey_freq - p_amps[1], self.cfg.expt.ramsey_freq + p_amps[1]), key=abs)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        self.qubits = self.cfg.expt.qubits
        self.checkEF = self.cfg.expt.checkEF

        q = self.qubits[0]

        f_pi_test = self.cfg.device.qubit.f_ge[q]
        if self.checkEF: f_pi_test = self.cfg.device.qubit.f_ef[q]
        if self.cfg.expt.f0g1_cavity > 0:
            ii = 0
            jj = 0
            if self.cfg.expt.f0g1_cavity==1: 
                ii=1
                jj=0
            if self.cfg.expt.f0g1_cavity==2: 
                ii=0
                jj=1
            # systematic way of adding qubit pulse under chi shift
            f_pi_test = self.cfg.device.QM.chi_shift_matrix[0][self.cfg.expt.f0g1_cavity]+self.cfg.device.qubit.f_ge[0] # freq we are trying to calibrate

        title = ('EF' if self.checkEF else '') + 'Ramsey' 

        # plt.figure(figsize=(10, 6))
        # plt.subplot(111,title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
        #             xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        # plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     if isinstance(p, (list, np.ndarray)): 
        #         pCov = data['fit_err_amps']
        #         captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        #         plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
        #         plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
        #         plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
        #         plt.legend()
        #         print(f'Current pi pulse frequency: {f_pi_test}')
        #         print(f"Fit frequency from amps [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}")
        #         if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
        #         print(f'Suggested new pi pulse frequencies from fit amps [MHz]:\n',
        #               f'\t{f_pi_test + data["f_adjust_ramsey_amps"][0]}\n',
        #               f'\t{f_pi_test + data["f_adjust_ramsey_amps"][1]}')
        #         print(f'T2 Ramsey from fit amps [us]: {p[3]}')

        plt.figure(figsize=(10,9))
        plt.subplot(211, 
            title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
            ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgi']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgq']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    

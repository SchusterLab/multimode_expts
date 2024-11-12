import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from MM_base import *

class CavityRamseyProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.checkEF = self.cfg.expt.checkEF

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        
        qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.f0g1_chs = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_types = cfg.hw.soc.dacs.sideband.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])

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
        elif cfg.expt.user_defined_pulse[5] == 5:
            self.cavity_ch = self.f0g1_chs
            self.cavity_ch_types = self.f0g1_ch_types
        elif cfg.expt.user_defined_pulse[5] == 4:
            self.cavity_ch = self.man_ch
            self.cavity_ch_types = self.man_ch_type
        
        
        self.q_rps = [self.ch_page(ch) for ch in self.cavity_ch] # get register page for f0g1 channel
        self.stor_rps = 0 # get register page for storage channel
        if self.cfg.expt.storage_ramsey[0]: 
            # decide which channel do we flux drive on 
            sweep_pulse = [['storage', 'M'+ str(self.cfg.expt.man_idx) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi'], 
                       ]
            self.creator = self.get_prepulse_creator(sweep_pulse)
            freq = self.creator.pulse[0][0]
            self.flux_ch = self. flux_low_ch 
            if freq > 1000: self.flux_ch = self.flux_high_ch

            # get register page for that channel 
            self.flux_rps = [self.ch_page(self.flux_ch[qTest])]
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

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]


        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_chs[qTest]
        # if self.res_ch_types[qTest] == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        # elif self.res_ch_types[qTest] == 'mux4':
        #     assert self.res_chs[qTest] == 6
        #     mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        #     mux_freqs = [0]*4
        #     mux_freqs[qTest] = cfg.device.readout.frequency[qTest]
        #     mux_gains = [0]*4
        #     mux_gains[qTest] = cfg.device.readout.gain[qTest]
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # declare registers for phase incrementing
        self.r_wait = 3
        self.r_wait_flux = 3
        self.r_phase2 = 4
        self.r_phase3 = 0
        self.r_phase4 = 6
        # if self.cavity_ch_types[qTest] == 'int4':
        #     self.r_phase = self.sreg(self.cavity_ch[qTest], "freq")
        #     self.r_phase3 = 5 # for storing the left shifted value
        # else:
        if self.cfg.expt.storage_ramsey[2] or self.cfg.expt.coupler_ramsey:
            self.phase_update_channel = self.flux_ch
        else:
            self.phase_update_channel = self.cavity_ch
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


        # add readout pulses to respective channels
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # initialize wait registers
        self.safe_regwi(self.phase_update_page[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        #self.safe_regwi(self.flux_rps, self.r_wait_flux, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase2, self.deg2reg(0)) 
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase3, 0) 
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase4 , 0) 
        # reg = self.deg2reg(-5)
        # print(f"-5 phase: {reg}")
        # reg = self.deg2reg(355)
        # print(f"355 phase: {reg}")
        # reg = self.deg2reg(5)
        # print(f"5 phase: {reg}")
        # reg = self.deg2reg(365)
        # print(f"365 phase: {reg}")
        

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
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='Prepulse')

        # play pi f0g1 pulse with the freq that we want to calibrate
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
        if self.cfg.expt.coupler_ramsey:
            self.custom_pulse(cfg, cfg.expt.custom_coupler_pulse, prefix='CustomCoupler')
            self.sync_all(self.us2cycles(0.01))
            print(cfg.expt.custom_coupler_pulse)
            print(self.flux_ch)

        # wait advanced wait time
        self.sync_all()
        # if cfg.expt.storage_ramsey[0]:
        #     #print('waiting for storage ramsey')
        #     #self.sync(self.flux_rps, self.r_wait_flux)
        #     self.sync(self.q_rps[qTest], self.r_wait)
        #     self.sync_all()
        # else:
        #     self.sync(self.q_rps[qTest], self.r_wait)
        #     self.sync_all()
        self.sync(self.phase_update_page[qTest], self.r_wait)
        self.sync_all()
        
        # self.sync_all(self.r_wait)

        # swap from storage to man 
        # if cfg.expt.storage_ramsey[0]:
        #     # sweep_pulse = [['storage', 'M'+ str(self.cfg.expt.man_idx) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi'], 
        #     #            ]
        #     # creator = self.get_prepulse_creator(sweep_pulse)
        #     self.custom_pulse(self.cfg, self.creator.pulse, prefix='Storage' + str(cfg.expt.storage_ramsey[1]) + 'dump', advance_qubit_phase=self.current_phase)
        #     self.sync_all(self.us2cycles(0.01))

        # play pi/2 pulse with advanced phase (all regs except phase are already set by previous pulse)
        # if self.cavity_ch_types[qTest] == 'int4':
        #     self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase2, '<<', 16)
        #     self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase3, '|', self.user_freq)
        #     self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase3, "+", 0)
        #     self.sync_all(self.us2cycles(0.01))
        # else:
        self.mathi(self.phase_update_page[qTest], self.r_phase, self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))


        if cfg.expt.storage_ramsey[0] or self.cfg.expt.coupler_ramsey:
            self.pulse(ch=self.flux_ch[qTest])
            self.sync_all(self.us2cycles(0.01))
        
        


        if self.cfg.user_defined_pulse[0]:
            self.pulse(ch=self.cavity_ch[qTest])
            self.sync_all(self.us2cycles(0.01))

        # postpulse 
        self.sync_all()
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='Postpulse')

        # parity measurement
        if self.cfg.expt.parity_meas: 
            parity_meas_str = [['qubit', 'ge', 'hpi'], # Starting parity meas
                       ['qubit', 'ge', 'parity_M' + str(self.cfg.expt.man_idx)], 
                       ['qubit', 'ge', 'hpi']]
            creator = self.get_prepulse_creator(parity_meas_str)
            print(creator.pulse)
            self.custom_pulse(self.cfg, creator.pulse, prefix='ParityMeas', sync_zero_const=True)
        # if self.cfg.expt.parity_meas: 
        #     self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge, phase=self.deg2reg(0), gain=cfg.device.qubit.pulses.hpi_ge.gain[0], waveform="hpi_qubit")
        #     self.sync_all() # align channels
        #     #self.sync_all(self.us2cycles(np.abs(1 / self.cfg.device.QM.chi_shift_matrix[0][self.cfg.expt.manipulate] / 2))) # wait for pi/chi (noe chi in config is in MHz)
        #     self.sync_all(self.us2cycles(np.abs(self.cfg.device.manipulate.revival_time[self.cfg.expt.manipulate]))) # wait for parity revival time
        #     self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge, phase=self.deg2reg(0), gain=cfg.device.qubit.pulses.hpi_ge.gain[0], waveform="hpi_qubit")
        #     self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns


        # align channels and measure
        self.sync_all(5)
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

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
        
        #self.safe_regwi(self.q_rps[qTest], self.r_phase3, phase_step) 
        # self.current_phase += 360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step
        # print(self.current_phase)
        # self.current_phase = self.current_phase % 360
        # if self.current_phase > 180: self.current_phase -= 360
        # if self.current_phase < -180: self.current_phase += 360
 
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
            
        # if phase_step_deg > 0:
        #     self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', phase_step)
        # else: 
        #     self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '-', phase_step) # advance the phase of the LO for the second π/2 pulse
        self.sync_all(self.us2cycles(0.01))
        # self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase4, '+', self.deg2reg(self.current_phase, gen_ch=self.cavity_ch[qTest])) # advance the phase of the LO for the second π/2 pulse
        # self.sync_all(self.us2cycles(0.01))

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
    
# def reset_and_sync(self, cfg):
#         # Phase reset all channels except readout DACs 

#         # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
#         # self.setup_and_pulse(ch=self.qubit_chs[qTest]s[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[qTest]s[0]), phase=0, gain=5, length=10, phrst=1)
#         # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
#         # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
#         # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
#         # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
#         # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)

#         self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
#         self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
#         # for prepulse 
#         self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
#         self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
#         self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
#         self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
#         self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
#         self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
#         self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
#         self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
#         self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
#         self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

#         # some dummy variables 
#         qTest = 0
#         self.f_q = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
#         self.f_cav = self.freq2reg(5000, gen_ch=self.man_ch[0])

#         #initialize the phase to be 0
#         self.set_pulse_registers(ch=self.qubit_ch[0], freq=self.f_q,
#                                  phase=0, gain=0, length=10, style="const", phrst=1)
#         self.pulse(ch=self.qubit_ch[0])
#         self.set_pulse_registers(ch=self.man_ch[0], freq=self.f_cav,
#                                  phase=0, gain=0, length=10, style="const", phrst=1)
#         self.pulse(ch=self.man_ch[0])
#         # self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cav,
#         #                          phase=0, gain=0, length=10, style="const", phrst=1)
#         # self.pulse(ch=self.storage_ch[0])
#         self.set_pulse_registers(ch=self.flux_low_ch[0], freq=self.f_cav,
#                                  phase=0, gain=0, length=10, style="const", phrst=1)
#         self.pulse(ch=self.flux_low_ch[0])
#         self.set_pulse_registers(ch=self.flux_high_ch[0], freq=self.f_cav,
#                                  phase=0, gain=0, length=10, style="const", phrst=1)
#         self.pulse(ch=self.flux_high_ch[0])
#         self.set_pulse_registers(ch=self.f0g1_ch[0], freq=self.f_q,
#                                  phase=0, gain=0, length=10, style="const", phrst=1)
#         self.pulse(ch=self.f0g1_ch[0])

#         self.sync_all(10)

#---------------------------------------------------------
# old code 
#---------------------------------------------------------
class CavityRamseyProgram_old(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.checkEF = self.cfg.expt.checkEF

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        
        qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.f0g1_chs = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_types = cfg.hw.soc.dacs.sideband.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])

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
        elif cfg.expt.user_defined_pulse[5] == 5:
            self.cavity_ch = self.f0g1_chs
            self.cavity_ch_types = self.f0g1_ch_types
        elif cfg.expt.user_defined_pulse[5] == 4:
            self.cavity_ch = self.man_ch
            self.cavity_ch_types = self.man_ch_type
        
        
        self.q_rps = [self.ch_page(ch) for ch in self.cavity_ch] # get register page for qubit_chs
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]


        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_chs[qTest]
        # if self.res_ch_types[qTest] == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        # elif self.res_ch_types[qTest] == 'mux4':
        #     assert self.res_chs[qTest] == 6
        #     mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        #     mux_freqs = [0]*4
        #     mux_freqs[qTest] = cfg.device.readout.frequency[qTest]
        #     mux_gains = [0]*4
        #     mux_gains[qTest] = cfg.device.readout.gain[qTest]
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # declare registers for phase incrementing
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase3 = 0
        if self.cavity_ch_types[qTest] == 'int4':
            self.r_phase = self.sreg(self.cavity_ch[qTest], "freq")
            self.r_phase3 = 5 # for storing the left shifted value
        else: self.r_phase = self.sreg(self.cavity_ch[qTest], "phase")



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


        # add readout pulses to respective channels
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # initialize wait registers
        self.safe_regwi(self.q_rps[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.q_rps[qTest], self.r_phase2, 0) 
        self.safe_regwi(self.q_rps[qTest], self.r_phase3, 0) 

        self.sync_all(200)

    def reset_and_sync(self, cfg):
        # Phase reset all channels except readout DACs 

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[qTest]s[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[qTest]s[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)

        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        # for prepulse 
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        # some dummy variables 
        qTest = 0
        self.f_q = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_cav = self.freq2reg(5000, gen_ch=self.man_ch[0])

        #initialize the phase to be 0
        self.set_pulse_registers(ch=self.qubit_ch[0], freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_ch[0])
        self.set_pulse_registers(ch=self.man_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_ch[0])
        # self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cav,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.storage_ch[0])
        self.set_pulse_registers(ch=self.flux_low_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_low_ch[0])
        self.set_pulse_registers(ch=self.flux_high_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_high_ch[0])
        self.set_pulse_registers(ch=self.f0g1_ch[0], freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.f0g1_ch[0])

        self.sync_all(10)
    def custom_pulse(self, cfg, pulse_data): 
        '''
        Executes prepulse or postpulse or middling pulse
        '''
        # self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        # self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        # # for prepulse 
        # self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        # self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        # self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        # self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        # self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        # self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        # self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        # self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        # self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        # self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        for jj in range(len(pulse_data[0])):
                # translate ch id to ch ( we don't need this )
                # if pulse_data[4][jj] == 1:
                #     self.tempch = self.flux_low_ch
                # elif pulse_data[4][jj] == 2:
                #     self.tempch = self.qubit_ch
                # elif pulse_data[4][jj] == 3:
                #     self.tempch = self.flux_high_ch
                # elif pulse_data[4][jj] == 4:
                #     self.tempch = self.storage_ch
                # elif pulse_data[4][jj] == 5:
                #     self.tempch = self.f0g1_ch
                # elif pulse_data[4][jj] == 6:
                #     self.tempch = self.man_ch

                self.tempch = [pulse_data[4][jj]]
                # print(self.tempch)
                # determine the pulse shape
                if pulse_data[5][jj] == "gaussian":
                    # print('gaussian')
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch[0])
                    self.add_gauss(ch=self.tempch[0], name="temp_gaussian" + str(jj),
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch[0], style="arb", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     waveform="temp_gaussian" + str(jj))
                elif pulse_data[5][jj] == "flat_top":
                    # print('flat_top')
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch[0])
                    self.add_gauss(ch=self.tempch[0], name="temp_gaussian" + str(jj),
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch[0], style="flat_top", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     length=self.us2cycles(pulse_data[2][jj], 
                                                           gen_ch=self.tempch[0]),
                                    waveform="temp_gaussian" + str(jj))
                else:
                    self.setup_and_pulse(ch=self.tempch[0], style="const", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     length=self.us2cycles(pulse_data[2][jj], 
                                                           gen_ch=self.tempch[0]))
                self.sync_all()  

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0] 
        
        # reset and sync all channels
        self.reset_and_sync(cfg)

        # pre pulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse)

        # play pi f0g1 pulse with the freq that we want to calibrate
        if self.user_length == 0: # its a gaussian pulse
            self.setup_and_pulse(ch=self.cavity_ch[qTest], style="arb", freq=self.user_freq, phase=self.deg2reg(0), gain=self.user_gain,waveform="user_test")
        else: # its a flat top pulse
            self.setup_and_pulse(ch=self.cavity_ch[qTest], style="flat_top", freq=self.user_freq, phase=0, gain=self.user_gain, length=self.user_length, waveform="user_test")

        # wait advanced wait time
        self.sync_all()
        self.sync(self.q_rps[qTest], self.r_wait)

        # play pi/2 pulse with advanced phase (all regs except phase are already set by previous pulse)
        if self.cavity_ch_types[qTest] == 'int4':
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase2, '<<', 16)
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase3, '|', self.user_freq)
            self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase3, "+", 0)
        else: self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.cavity_ch[qTest])

        # postpulse 
        self.sync_all()
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse)

        # parity measurement
        if self.cfg.expt.parity_meas: 
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge, phase=self.deg2reg(0), gain=cfg.device.qubit.pulses.hpi_ge.gain[0], waveform="hpi_qubit")
            self.sync_all() # align channels
            #self.sync_all(self.us2cycles(np.abs(1 / self.cfg.device.QM.chi_shift_matrix[0][self.cfg.expt.manipulate] / 2))) # wait for pi/chi (noe chi in config is in MHz)
            self.sync_all(self.us2cycles(np.abs(self.cfg.device.manipulate.revival_time[self.cfg.expt.manipulate]))) # wait for parity revival time
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge, phase=self.deg2reg(180), gain=cfg.device.qubit.pulses.hpi_ge.gain[0], waveform="hpi_qubit")
            self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns

        
        

        # align channels and measure
        self.sync_all(5)
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

    def update(self):
        qTest = self.qubits[0]
        phase_step = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step, gen_ch=self.cavity_ch[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        #self.safe_regwi(self.q_rps[qTest], self.r_phase3, phase_step) 
        self.mathi(self.q_rps[qTest], self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update the time between two π/2 pulses
        self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', phase_step) # advance the phase of the LO for the second π/2 pulse


class CavityRamseyExperiment_old(Experiment):
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

        ramsey = CavityRamseyProgram_old(soccfg=self.soccfg, cfg=self.cfg)
        
        x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
 
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}        
        self.data=data
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
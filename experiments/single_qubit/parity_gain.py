import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from copy import deepcopy # single shot dictionary cfg copy

import experiments.fitting as fitter
from MM_base import *
from MM_dual_rail_base import *

class ParityGainProgram(MMRAveragerProgram):
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
        qTest=0
        # self.cfg.update(cfg.expt) 
        
        # self.adc_ch = cfg.hw.soc.adcs.readout.ch
        # self.res_ch = cfg.hw.soc.dacs.readout.ch
        # self.res_ch_type = cfg.hw.soc.dacs.readout.type
        # self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        # self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        # self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        # self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        # self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        # self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        # self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        # self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        # self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        # self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        # self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        # self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        # self.man_chs = cfg.hw.soc.dacs.manipulate_in.ch
        # self.man_ch_types = cfg.hw.soc.dacs.manipulate_in.type

        # # declare register pages (gain)
        self.man_rp = [self.ch_page(self.man_ch[qTest])] # get register page for qubit_ch
        self.r_gain = self.sreg(self.man_ch[qTest], "gain") # get gain register for qubit_ch
        self.r_gain2 = 4 # dummy register for gain  (since multiple qubit pulses)
        self.safe_regwi(self.man_rp[qTest], self.r_gain2, self.cfg.expt.start) # set dummygain register to start value

        # # declare qubit dacs
        
        # self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        # self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        # self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        # self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        # mask = None
        # mixer_freq = 0 # MHz
        # mux_freqs = None # MHz
        # mux_gains = None
        # ro_ch = self.adc_ch
        # if self.res_ch_type == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        # elif self.res_ch_type == 'mux4':
        #     assert self.res_ch == 6
        #     mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        #     mux_freqs = [0]*4
        #     mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
        #     mux_gains = [0]*4
        #     mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
        # self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # cavity pulse param
        self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.manipulate - 1], gen_ch = self.man_ch[qTest])
        print(self.man_ch)
        print(self.cfg.expt.manipulate)
        if cfg.expt.displace[0]:
            self.displace_sigma = self.us2cycles(cfg.expt.displace[1], gen_ch=self.man_ch[qTest])
            self.add_gauss(ch=self.man_ch[qTest], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)

        #f0g1 sideband
        self.f0g1 = self.freq2reg(cfg.device.QM.pulses.f0g1.freq[cfg.expt.f0g1_cavity-1], gen_ch=self.qubit_ch[qTest])
        self.f0g1_length = self.us2cycles(cfg.device.QM.pulses.f0g1.length[cfg.expt.f0g1_cavity-1], gen_ch=self.qubit_ch[qTest])
        self.pif0g1_gain = cfg.device.QM.pulses.f0g1.gain[cfg.expt.f0g1_cavity-1]
        
        # # declare qubit dacs
        # mixer_freq = 0
        # if self.qubit_ch_type == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        # self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # # declare adcs
        # self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        # self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma, gen_ch=self.qubit_ch)
        # self.hpi_sigma_fast = self.us2cycles(cfg.device.qubit.pulses.hpi_ge_fast.sigma, gen_ch=self.qubit_ch)
        # self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        # self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        # self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain

        # # add qubit and readout pulses to respective channels
        # if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            
        #     self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        #     self.add_gauss(ch=self.qubit_ch, name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)
        #     self.add_gauss(ch=self.qubit_ch, name="hpi_qubit_fast", sigma=self.hpi_sigma_fast, length=self.hpi_sigma_fast*4)
        #     self.add_gauss(ch=self.qubit_ch, name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
        #     self.add_gauss(ch=self.f0g1_ch, name="f0g1",
        #                sigma=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma), length=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma)*4)
        #     #self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        # else:
        #     self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=cfg.expt.start, length=self.pi_sigma)


        # if self.res_ch_type == 'mux4':
        #     self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        # self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        # load ECD file data
        if cfg.expt.ECD_pulse:
            with open(cfg.expt.pulse_fname + '.npy', 'rb') as f:
                self.cavity_dac_gauss= np.load(f)   
                self.qubit_dac_gauss = np.load(f)

        self.sync_all(200)

    

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest=0

        # add cavity reset
        self.reset_and_sync()
        # active reset 
        if self.cfg.expt.active_reset: 
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        # pre pulse
        if cfg.expt.prepulse:
            print('Inside parity gain code')
            print(cfg.expt.pre_sweep_pulse)
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='Prepulse')

        
        

        # --------------------------------------------------------------------------------
        # Iterate over ECD pulses
        

        if cfg.expt.ECD_pulse:
            # too lazy to change variable names
            self.f_q = self.f_ge
            self.f_cav = self.f_cavity

            

            # add appropriate delay for cavity pulse 
            #qubit man channel delay
            self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_cav, phase=0, gain=0, length=self.cfg.expt.man_delay)
          

            # iterate over cavity and qubit pulses
            for idx, cav_arr in enumerate(self.cavity_dac_gauss): 
                qub_arr = self.qubit_dac_gauss[idx]

                amp_c = cav_arr[0]
                sigma_c = self.us2cycles(cav_arr[1].real * 1e-3) 

                amp_q = qub_arr[0]
                sigma_q = self.us2cycles(qub_arr[1].real * 1e-3)

                name = 'gauss' + str(idx)

                #Pathological Case 1 0 length pulses
                if np.abs(sigma_c) < 1:
                    continue 

                # Case 1: qubit off, cavity off  (** replace with sync command)
                elif int(np.abs(amp_c)) == 0 and int(np.abs(amp_q)) == 0 : 
                
                    self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_q, phase=0, gain=0, length=sigma_q)
                    self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_cav, phase=0, gain=0, length=sigma_c)
                
                # Case 2: qubit on, cavity off
                elif int(np.abs(amp_c)) == 0 and int(np.abs(amp_q)) != 0  : 
                    # self.add_gauss_ecd_specific1(ch = self.qubit_ch, name = name, sigma = sigma_q,
                    #                         length = 4*sigma_q)
                    self.add_gauss(ch = self.qubit_ch, name = name, sigma = sigma_q,
                                            length = 4*sigma_q)
                    self.setup_and_pulse(ch = self.qubit_ch, style = "arb", freq=self.f_q, 
                                        phase=self.deg2reg(np.angle(amp_q)/np.pi*180), gain = int(np.abs(amp_q)), waveform = name)

                    self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_cav, phase=0, gain=0, length=sigma_c)
                
                # Case 3: qubit off, cavity on
                elif int(np.abs(amp_c)) != 0 and int(np.abs(amp_q)) == 0  :
                    
                
                    self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_q, phase=0, gain=0, length=sigma_q)

                    # self.add_gauss_ecd_specific1(ch = self.man_ch, name = name, sigma = sigma_c,
                    #                         length = 4*sigma_c)
                    self.add_gauss(ch = self.man_ch, name = name, sigma = sigma_c,
                                            length = 4*sigma_c)
                    self.setup_and_pulse(ch = self.man_ch, style = "arb",  freq=self.f_cav, 
                                        phase=self.deg2reg(np.angle(amp_c)/np.pi*180), gain = int(np.abs(amp_c)),waveform = name)

                    # print('cavity on')
                    print('amp is ' + str(amp_c))
                    print('sigma is ' + str(sigma_c))
                
                # self.sync_all()
                # #qubit man channel delay
                # self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_cav, phase=0, gain=0, length=self.cfg.expt.man_delay)
          
        #------------------------------------------------------------------------------------
        #  Now Parity Gain 
        #  Setup cavity pulse form
        if self.cfg.expt.displace[0]:
            self.set_pulse_registers(
                    ch=self.man_ch[qTest],
                    style="arb",
                    freq=self.f_cavity,
                    phase=self.deg2reg(0), 
                    gain=self.cfg.expt.start, # placeholder
                    waveform="displace")
            
        
        if self.cfg.expt.const_pulse[0]:
            self.set_pulse_registers(ch=self.man_ch[qTest], 
                                 style="const", 
                                 freq=self.f_cavity, 
                                 phase=self.deg2reg(0),
                                gain=self.cfg.expt.start, # placeholder
                                length=self.us2cycles(self.cfg.expt.const_pulse[1]))
        # Update gain and pulse  
        self.mathi(self.man_rp[qTest], self.r_gain, self.r_gain2, "+", 0) # update gain register
        self.pulse(ch = self.man_ch[qTest])
        self.sync_all() # align channels

        # Parity Measurement
        self.setup_and_pulse(ch=self.qubit_ch[qTest], style="arb", freq=self.f_ge, phase=self.deg2reg(0), gain=cfg.device.qubit.pulses.hpi_ge.gain[qTest], waveform="hpi_qubit_ge")
        self.sync_all() # align channels
        # self.sync_all(self.us2cycles(np.abs(1 / self.cfg.device.QM.chi_shift_matrix[0][1] / 2))) # wait for pi/chi (noe chi in config is in MHz)
        # self.sync_all(self.us2cycles(np.abs(self.cfg.device.manipulate.revival_time[self.cfg.expt.manipulate-1]))) # wait for parity revival time
        self.setup_and_pulse(ch=self.qubit_ch[qTest], style="const", freq=self.f_ge, phase=self.deg2reg(0), gain=0, length=self.us2cycles(np.abs(self.cfg.device.manipulate.revival_time[self.cfg.expt.manipulate-1])))
        self.sync_all() # align channels
        self.setup_and_pulse(ch=self.qubit_ch[qTest], style="arb", freq=self.f_ge, phase=self.deg2reg(180), gain=cfg.device.qubit.pulses.hpi_ge.gain[qTest], waveform="hpi_qubit_ge")
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

    def update(self):
        qTest=0
        self.mathi(self.man_rp[qTest], self.r_gain2, self.r_gain2, '+', self.cfg.expt.step) # update gain register

    # def collect_shots(self):
    #     # collect shots for the relevant adc and I and Q channels
    #     # print(np.average(self.di_buf[0]))
    #     shots_i0 = self.di_buf[0] / self.readout_length_adc
    #     shots_q0 = self.dq_buf[0] / self.readout_length_adc
    #     return shots_i0, shots_q0
    #     # return shots_i0[:5000], shots_q0[:5000]

class ParityGainExperiment(Experiment):
    """
    ParityGain Experiment
    Experimental Config:
    expt = dict(
        start: gain sweep start [us]
        step: gain sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubits[0]
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update(
                                    {key3: [value3]*num_qubits_sample})
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
                                     
        if not self.cfg.expt.single_shot:
            read_num = 1
            if self.cfg.expt.active_reset: read_num = 4
            
            prog = ParityGainProgram(soccfg=self.soccfg, cfg=self.cfg)
            
            x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug,
                                                readouts_per_experiment=read_num)        
    
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase

            data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases} 
            data['idata'], data['qdata'] = prog.collect_shots()

            
        else:

            # ----------------- Single Shot Calibration -----------------
            data = dict()
            mm_dr_base = MM_dual_rail_base(cfg=self.cfg)
            data = mm_dr_base.run_single_shot(self_expt=self, data = data, progress=progress, debug=debug)

            fids = data['fids']
            thresholds = data['thresholds']
            angle = data['angle']
            confusion_matrix = data['confusion_matrix']

            print(f'ge fidelity (%): {100*fids[0]}')
            print(f'rotation angle (deg): {angle}')
            print(f'threshold ge: {thresholds[0]}')
            print('Confusion matrix [Pgg, Pge, Peg, Pee]: ',confusion_matrix)


            # ------------------- Experiment -------------------
            read_num = 1
            if self.cfg.expt.active_reset: read_num = 4

            data['I_data']= []
            data['Q_data']= []
            data['avgi'] = [] # for debugging
            data['avgq'] = []
            # Do single round experiments since collecting shots for all rounds is not supported
            rounds = self.cfg.expt.rounds

            for round in range(rounds): 
                print(f'Round {round}')
                rcfg = AttrDict(deepcopy(self.cfg))
                rcfg.expt.rounds = 1

                prog = ParityGainProgram(soccfg=self.soccfg, cfg=rcfg)
                x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug,
                                                 readouts_per_experiment=read_num)
                II, QQ = prog.collect_shots()
                # save data for each round
                data['I_data'].append(II)
                data['Q_data'].append(QQ)
                data['avgi'].append(avgi) # for debugging
                data['avgq'].append(avgq)
                data['xpts'] = x_pts # same for all rounds
            
            fids = data['fids']
            thresholds = data['thresholds']
            angle = data['angle']
            confusion_matrix = data['confusion_matrix']

            print(f'ge fidelity (%): {100*fids[0]}')
            print(f'rotation angle (deg): {angle}')
            print(f'threshold ge: {thresholds[0]}')
            print('Confusion matrix [Pgg, Pge, Peg, Pee]: ',confusion_matrix)
            
        self.data=data
        return data

    def single_shot_analysis(self, data=None, **kwargs):
        '''
        Bin shots in g and e state s
        '''
        threshold = self.cfg.device.readout.threshold # for i data
        theta = self.cfg.device.readout.phase * np.pi / 180 # degrees to rad
        I = data['I']
        Q = data['Q']

        # """Rotate the IQ data"""
        # I_new = I*np.cos(theta) - Q*np.sin(theta)
        # Q_new = I*np.sin(theta) + Q*np.cos(theta) 
        I_new = I
        Q_new = Q

        # """Threshold the data"""
        shots = np.zeros(len(I_new))
        #shots[I_new < threshold] = 0 # ground state
        shots[I_new > threshold] = 1 # excited state

        # Reshape data into 2D array: expts x reps 
        shots = shots.reshape(self.cfg.expt.expts, self.cfg.expt.reps)

        # Average over reps
        probs_ge = np.mean(shots, axis=1)

        data['probs_ge'] = probs_ge
        return data



    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        # plt.figure(figsize=(12, 8))
        # plt.subplot(111,title="$T_1$", xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        # plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     pCov = data['fit_err_amps']
        #     captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        #     plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_amps"]), label=captionStr)
        #     plt.legend()
        #     print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter

class FastResetProgram(RAveragerProgram):
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
        # self.checkZZ = self.cfg.expt.checkZZ
        self.checkEF = self.cfg.expt.checkEF
        self.pulse_ge = self.cfg.expt.pulse_ge
        if self.checkEF:
            if 'pulse_ge' not in self.cfg.expt: self.pulse_ge = True
            else: self.pulse_ge = self.cfg.expt.pulse_ge

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits

        
        qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        # if self.checkZZ: self.f_Q1_ZZ_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q1_ZZ]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qTest] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        elif self.res_ch_types[qTest] == 'mux4':
            assert self.res_chs[qTest] == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
            mux_freqs = [0]*4
            mux_freqs[qTest] = cfg.device.readout.frequency[qTest]
            mux_gains = [0]*4
            mux_gains[qTest] = cfg.device.readout.gain[qTest]
            ro_ch=self.adc_chs[qTest]
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
        
        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.pi_test_sigma = self.us2cycles(cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest])
        self.f_pi_test_reg = self.freq2reg(self.cfg.expt.freq, gen_ch=self.qubit_chs[qTest]) # freq we are trying to calibrate

        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain[qTest]
        

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])

        self.add_gauss(ch=self.qubit_chs[qTest], name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        # if self.checkZZ:
        #     self.pisigma_ge_qA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA], gen_ch=self.qubit_chs[qA])
        #     self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qA], gen_ch=self.qubit_chs[qTest])
        #     self.f_ge_init_reg = self.f_Q1_ZZ_reg[qA] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
        #     self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qA] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
        #     self.f_pi_test_reg = self.f_Q1_ZZ_reg[qA] # freq we are trying to calibrate
        if self.checkEF:
            self.f_pi_test_reg = self.f_ef_reg[qTest] # freq we are trying to calibrate
        if cfg.expt.f0g1_cavity > 0:
            ii = 0
            jj = 0
            if cfg.expt.f0g1_cavity==1: 
                ii=1
                jj=0
            if cfg.expt.f0g1_cavity==2: 
                ii=0
                jj=1
            # systematic way of adding qubit pulse under chi shift
            self.pif0g1_gain = self.cfg.device.QM.pulses.f0g1.gain[cfg.expt.f0g1_cavity-1]
            # self.f_pi_test_reg = self.freq2reg(self.cfg.device.QM.chi_shift_matrix[0][cfg.expt.f0g1_cavity]+self.cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest]) # freq we are trying to calibrate
            self.gain_pi_test = self.cfg.device.QM.pulses.qubit_pi_ge.gain[ii][jj] # gain of the pulse we are trying to calibrate
            # self.pi2sigma_test = self.cfg.device.QM.pulses.qubit_pi_ge.sigma[ii][jj]
            # self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test", sigma=self.pi2sigma, length=self.pi2sigma*4)
            self.f0g1 = self.freq2reg(cfg.device.QM.pulses.f0g1.freq[cfg.expt.f0g1_cavity-1], gen_ch=self.f0g1_ch[0])
            self.f0g1_length = self.us2cycles(cfg.device.QM.pulses.f0g1.length[cfg.expt.f0g1_cavity-1], gen_ch=self.f0g1_ch[0])
        
        # add qubit and readout pulses to respective channels
        if cfg.expt.pulse_type.lower() == "gauss" and self.pi_test_sigma > 0:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test", sigma=self.pi_test_sigma, length=self.pi_test_sigma*4)
        # if self.checkZZ:
        #     self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pisigma_ge_qA, length=self.pisigma_ge_qA*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge*4)

        # add readout pulses to respective channels
        # if self.res_ch_types[qTest] == 'mux4':
        #     self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])



        # initialize registers
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_gain = self.sreg(self.qubit_chs[qTest], "addr") # get gain register for qubit_ch    
        else: self.r_gain = self.sreg(self.qubit_chs[qTest], "gain") # get gain register for qubit_ch    
        self.r_gain2 = 4
        self.safe_regwi(self.q_rps[qTest], self.r_gain2, self.cfg.expt.start)

        ## ALL ACTIVE RESET REQUIREMENTS
        # read val definition
        self.r_read_q = 3
        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0

        # threshold definition
        self.r_thresh_q = 5  # Define a location to store the threshold info

        # print(np.round(cfg.expt.threshold * self.readout_lengths_dac[qTest]))

        # # multiplication bc the readout is summed, so need common thing to compare to
        self.safe_regwi(0, self.r_thresh_q, int(cfg.expt.threshold * self.readout_lengths_adc[qTest]))
        # print((cfg.expt.threshold *self.cfg.device.readout.readout_length[0]))
        # print((cfg.expt.threshold *self.readout_lengths_adc[qTest]))
        # Define a location to store a counter for how frequently the condj is triggered
        self.r_counter = 7
        self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0

        self.sync_all(200)

    def active_reset(self):
        # Perform measurement of both readouts.
        qTest = 0
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.res_chs[qTest], 
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                     t='auto', wait=True, syncdelay=self.us2cycles(2))#self.cfg["relax_delay"])  # self.us2cycles(1))
        
        self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced

        self.read(0, 0, "lower", self.r_read_q)  # read data from I buffer, QA, and store
        self.wait_all()  # to allow the read to be complete might be reduced
        self.sync_all()

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   "LABEL_1")  # compare the value recorded above to the value stored in threshold.

        #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
        self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ge_init_reg, style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_gain, waveform='pi_qubit')
        self.pulse(ch=self.qubit_chs[qTest])
        self.label("LABEL_1")  # location to be jumped to
        self.wait_all() 

        self.sync_all(self.us2cycles(0.05))

        #perform second measurement for additional postselection
        self.measure(pulse_ch=self.res_chs[qTest], 
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                     t='auto', wait=True, syncdelay=self.us2cycles(2))#self.cfg["relax_delay"])

        self.sync_all(self.us2cycles(0.05))  # not sure if this is needed?

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # initializations as necessary
        self.sync_all()
        

        if self.pi_test_sigma > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="arb",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_test")
            else:
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="const",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.sigma_test)
        self.mathi(self.q_rps[qTest], self.r_gain, self.r_gain2, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])
        #self.pulse(ch=self.qubit_chs[qTest]) # debug 
      
        self.sync_all()

        #do the active reset
        if cfg.expt.active_reset:
            self.active_reset()

        # align channels and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )
 
    def update(self):
        qTest = self.qubits[0]

        step = self.cfg.expt.step
        if self.qubit_ch_types[qTest] == 'int4': step = step << 16
        self.mathi(self.q_rps[qTest], self.r_gain2, self.r_gain2, '+', step) # update test gain

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[0]
        return shots_i0, shots_q0

# ====================================================== #
                      
class FastResetExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='FastReset', config_file=None, progress=None):
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

        qTest = self.cfg.expt.qubits[0]

        if 'sigma_test' not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma[qTest]
            # else: self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qA]
        
        amprabi = FastResetProgram(soccfg=self.soccfg, cfg=self.cfg)

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 3
        # print(read_num)
        
        xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug, readouts_per_experiment=read_num)

        # shots_i = amprabi.di_buf[adc_ch].reshape((self.cfg.expt.expts, self.cfg.expt.reps)) / amprabi.readout_length_adc
        # shots_i = np.average(shots_i, axis=1)
        # print(len(shots_i), self.cfg.expt.expts)
        # shots_q = amprabi.dq_buf[adc_ch] / amprabi.readout_length_adc
        # print(np.std(shots_i), np.std(shots_q))
        
        # avgi = avgi[0][0]
        # avgq = avgq[0][0]
        # amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        # phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        # data={'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq}
        data['I'], data['Q'] = amprabi.collect_shots()
        self.data=data
        return data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data=self.data
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            xdata = data['xpts']

            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
        return data

    def display(self, data=None, fit=True, fitparams=None, vline = None, **kwargs):
        if data is None:
            data=self.data 

        # plt.figure(figsize=(10, 6))
        # plt.subplot(111, title=f"Amplitude Rabi (Pulse Length {self.cfg.expt.sigma_test})", xlabel="Gain [DAC units]", ylabel="Amplitude [ADC units]")
        # plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     plt.plot(data["xpts"][1:-1], fitter.decaysin(data["xpts"][1:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        #     pi2_gain = pi_gain/2
        #     print(f'Pi gain from amps data [dac units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from amps data [dac units]: {int(pi2_gain)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=f"Amplitude Rabi (Pulse Length {self.cfg.expt.sigma_test})", ylabel="I [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
            # print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
            print(f'\tPi/2 gain from avgi data [dac units]: {int(1/4/p[1])}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
            if not(vline==None):
                plt.axvline(vline, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
            # print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
            print(f'\tPi/2 gain from avgq data [dac units]: {int(1/4/p[1])}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

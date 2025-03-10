import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class PhaseSweepAveragerProgram(AveragerProgram):
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

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge_idle)
        self.qubits = self.cfg.expt.qubits
        
        qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        # self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
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


        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge_idle[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef_idle[qTest], gen_ch=self.qubit_chs[qTest])
        

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        # self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs[qTest])]
        self.f_ge_reg = self.freq2reg(cfg.device.qubit.f_ge_idle[qTest], gen_ch=self.qubit_chs[qTest])
        # self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs[qTest])]
        self.f_ef_reg = self.freq2reg(cfg.device.qubit.f_ef_idle[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain[qTest]
        self.hpi_gain = cfg.device.qubit.pulses.hpi_ge.gain[qTest]
        self.hpief_gain = cfg.device.qubit.pulses.hpi_ef.gain[qTest]
        

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.hpief_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])

        self.add_gauss(ch=self.qubit_chs[qTest], name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)

        self.add_gauss(ch=self.qubit_chs[qTest], name="hpief_qubit", sigma=self.hpief_sigma, length=self.hpief_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)

        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_chs[qTest]
       
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
        self.r_phase = self.sreg(self.qubit_chs[qTest], "phase")
        self.r_phase2 = 4

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.f_ge_reg
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest]/2, gen_ch=self.qubit_chs[qTest])
        self.f_pi_test_reg = self.f_ge_reg # freq we are trying to calibrate
        self.gain_pi_test = self.cfg.device.qubit.pulses.pi_ge.gain[qTest] # gain of the pulse we are trying to calibrate
        
        
        # add qubit pulses to respective channels
        # print(f"Calibrating pi/2 pulse on qubit {qTest} with freq {self.f_pi_test_reg} MHz")
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test", sigma=self.pi2sigma, length=self.pi2sigma*4)
        
        # add readout pulses to respective channels
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # initialize wait registers
        #self.safe_regwi(self.q_rps[qTest], self.r_phase2, self.deg2reg(cfg.expt.start, self.qubit_chs[qTest])) 
        self.safe_regwi(self.q_rps[qTest], self.r_phase, self.deg2reg(0, self.qubit_chs[qTest])) 
        #print the phases
        #print(f'Initial phase on rphase: {self.reg2val(self.r_phase,self.qubit_chs[qTest])}')
        #print(f'Initial phase on rphase2: {self.reg2val(self.r_phase2, self.qubit_chs[qTest])}')
        #print(f'Sanity Check: {self.reg2deg(self.deg2reg(180, self.qubit_chs[qTest]), self.qubit_chs[qTest])}')
        

        self.sync_all(200)

    def custom_pulse(self, cfg, pulse_data, advance_qubit_phase=None): 
        '''
        Executes prepulse or postpulse or middling pulse

        # [[frequency], [gain], [length (us)], [phases], [drive channel],
        #  [shape], [ramp sigma]],
        #  drive channel=1 (flux low), 
        # 2 (qubit),3 (flux high),4 (storage),5 (f0g1),6 (manipulate),
        '''
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

        if advance_qubit_phase is not None:
            pulse_data[3] = [x + advance_qubit_phase for x in pulse_data[3]]
            print(f'Advancing qubit phase by {advance_qubit_phase} degrees')

        for jj in range(len(pulse_data[0])):
                # translate ch id to ch
                if pulse_data[4][jj] == 1:
                    self.tempch = self.flux_low_ch
                elif pulse_data[4][jj] == 2:
                    self.tempch = self.qubit_ch
                elif pulse_data[4][jj] == 3:
                    self.tempch = self.flux_high_ch
                elif pulse_data[4][jj] == 4:
                    self.tempch = self.storage_ch
                elif pulse_data[4][jj] == 5:
                    self.tempch = self.f0g1_ch
                elif pulse_data[4][jj] == 6:
                    self.tempch = self.man_ch
                # print(self.tempch)
                # determine the pulse shape
                if pulse_data[5][jj] == "gaussian":
                    # print('gaussian')
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch[0])
                    self.add_gauss(ch=self.tempch[0], name="temp_gaussian",
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch[0], style="arb", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     waveform="temp_gaussian")
                elif pulse_data[5][jj] == "flat_top":
                    # print('flat_top')
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch[0])
                    self.add_gauss(ch=self.tempch[0], name="temp_gaussian",
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch[0], style="flat_top", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     length=self.us2cycles(pulse_data[2][jj], 
                                                           gen_ch=self.tempch[0]),
                                    waveform="temp_gaussian")
                else:
                    self.setup_and_pulse(ch=self.tempch[0], style="const", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     length=self.us2cycles(pulse_data[2][jj], 
                                                           gen_ch=self.tempch[0]))
                self.sync_all()
    def reset_and_sync(self):
        # Phase reset all channels except readout DACs
        #  
        cfg = self.cfg
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

        self.f_q = self.f_ge_reg
        self.f_cav = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_ch[0])

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[qTest]s[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[qTest]s[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)


        #initialize the phase to be 0
        self.set_pulse_registers(ch=self.qubit_ch[0], freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_chs[0])
        self.set_pulse_registers(ch=self.man_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_ch[0])
        self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.storage_ch[0])
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

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0] 

        # phase reset 
        self.reset_and_sync()

        #phase reset 
        
        #prepulse : 
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse)

        if cfg.expt.post_prepulse_reset: 
            self.reset_and_sync()

        # play pi/2 pulse with the freq that we want to calibrate
        self.vz = 0
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg, phase=self.deg2reg(self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
        #pi ef 
        if self.cfg.expt.piEF:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ef_reg, phase=self.deg2reg(0), gain=self.pief_gain, waveform="pief_qubit")

        #middlepulse :
        if cfg.expt.middlepulse:
            for i in range(cfg.expt.reps_middlepulse):
                self.custom_pulse(cfg, cfg.expt.mid_sweep_pulse)
                self.vz+= self.cfg.expt.zz_phase  # advance phase of ge pulse
        
        # play pi/2 pulse with advanced phase (all regs except phase are already set by previous pulse)
        #self.custom_pulse(cfg, cfg.expt.post_hpi_pulse, advance_qubit_phase=self.reg2deg(self.r_phase2))

        ##pi ef 
        if self.cfg.expt.piEF:
            self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ef_reg, phase=self.deg2reg(0), gain=self.pief_gain, waveform="pief_qubit")
            #self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, '+', 0)
            self.pulse(ch=self.qubit_chs[qTest])
            

        
        ## pi/2 ge 
        self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg, 
                                 phase= self.deg2reg(0), #self.deg2reg(cfg.expt.advance_qubit_phase, gen_ch = self.qubit_chs[qTest]),
                                   gain=self.hpi_gain, waveform="hpi_qubit")
        self.safe_regwi(self.q_rps[qTest], self.r_phase,
                          self.deg2reg(180 + cfg.expt.phase_update + self.vz, gen_ch = self.qubit_chs[qTest])) # add offset
        #self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', self.deg2reg(self.cfg.expt.advance_qubit_phase)) # add offset
        #self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', self.deg2reg(cfg.expt.advance_qubit_phase)) # add the step
        # if self.cfg.expt.advance_qubit_phase > 0:
        #     self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, '+' , self.deg2reg(cfg.expt.advance_qubit_phase)) # add the phase from update 
        # else: 
        #     self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, '-' , self.deg2reg(abs(cfg.expt.advance_qubit_phase)))
        self.pulse(ch=self.qubit_chs[qTest])

        #postpulse :
        self.sync_all()
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse)

        # align channels and measure
        self.sync_all(5)
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

        


class PhaseSweepAveragerExperiment(Experiment):
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

    def __init__(self, soccfg=None, path='', prefix='PhaseSweepAverager', config_file=None, progress=None):
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

        phases_test = self.cfg.expt["phase_start"] + self.cfg.expt["phase_step"] * np.arange(self.cfg.expt["expts"])

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for phase_update in tqdm(phases_test, disable=not progress):
            self.cfg.expt.phase_update = float(phase_update)
            lengthrabi = PhaseSweepAveragerProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(phase_update)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k]=np.array(a)

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)
            
            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]
        
        
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
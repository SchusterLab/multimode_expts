import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm
# from scipy.sepcial import erf

import fitting.fitting as fitter

class WignerTomography1ModeCatProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

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

        self.man_chs = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_types = cfg.hw.soc.dacs.manipulate_in.type

        # self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        # self.man_rps = self.ch_page(self.man_chs)  # get register page for man_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
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
        self.hpisigma_ge = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default hpi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_hge_init = self.cfg.device.qubit.pulses.hpi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.f_pi_test_reg = self.f_ge_reg[qTest] # freq we are trying to calibrate

        # define the displace sigma for calibration
        self.displace_sigma = self.us2cycles(cfg.expt.displace_length, gen_ch=self.man_chs[0])       
        self.displace_pre_sigma = self.us2cycles(cfg.expt.displace_pre_sigma, gen_ch=self.man_chs[0])     
        # self.prepulse_sigma = self.us2cycles(cfg.expt.pre_pulse[1], gen_ch=self.man_chs[0])   
        

        
        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit", sigma=self.hpisigma_ge, length=self.hpisigma_ge*4)

        if cfg.expt.cavity_name == 0:
            self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_chs[0])
            # calculate revival time=pi/chi
            # self.revival_time = np.abs(1/cfg.device.QM.chi_shift_matrix[0][1])/2
            self.revival_time = cfg.device.manipulate.revival_time[0]
        else:
            self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[1], gen_ch=self.man_chs[0])
            # calculate revival time=pi/chi
            # self.revival_time = np.abs(1/cfg.device.QM.chi_shift_matrix[0][2])/2
            self.revival_time = cfg.device.manipulate.revival_time[1]

        self.add_gauss(ch=self.man_chs[0], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)
        self.add_gauss(ch=self.man_chs[0], name="displace_pre", sigma=self.displace_pre_sigma, length=self.displace_pre_sigma*4)

        # if cfg.expt.pre_pulse[0]:
        #     self.add_gauss(ch=self.man_chs[0], name="pre_pulse", sigma=self.prepulse_sigma, length=self.prepulse_sigma*4)


        # add readout pulses to respective channels
        # if self.res_ch_types[qTest] == 'mux4':
        #     self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])


        # self.chi_shift = cfg.expt.guessed_chi
        # self.ratio = np.cos(np.pi*2*self.chi_shift/4*(2*self.cycles2us(self.tp)+3*self.cycles2us(self.displace_sigma*4)))/np.cos(np.pi*2*self.chi_shift/4*self.cycles2us(self.displace_sigma*4))
        if cfg.expt.optpulse:
            self.add_opt_pulse(ch=self.qubit_chs[0], name="test_opt_qubit", pulse_location=cfg.expt.opt_file_path[0])
            self.add_opt_pulse(ch=self.man_chs[0], name="test_opt_cavity", pulse_location=cfg.expt.opt_file_path[1])

        self.sync_all(200)

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs 

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)


        #initialize the phase to be 0
        self.set_pulse_registers(ch=self.qubit_chs[0], freq=self.f_ge_init_reg,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_chs[0])
        self.set_pulse_registers(ch=self.man_chs[0], freq=self.f_cavity,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_chs[0])
        self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cavity,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.storage_ch[0])
        self.set_pulse_registers(ch=self.flux_low_ch[0], freq=self.f_cavity,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_low_ch[0])
        self.set_pulse_registers(ch=self.flux_high_ch[0], freq=self.f_cavity,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_high_ch[0])
        self.set_pulse_registers(ch=self.f0g1_ch[0], freq=self.f_ge_init_reg,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.f0g1_ch[0])

        self.sync_all(10)

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # phase reset
        self.reset_and_sync()
            
        #  prepulse 
        if cfg.expt.prepulse:
            for ii in range(len(cfg.expt.pre_sweep_pulse[0])):
                # translate ch id to ch
                if cfg.expt.pre_sweep_pulse[4][ii] == 1:
                    self.tempch = self.flux_low_ch
                elif cfg.expt.pre_sweep_pulse[4][ii] == 2:
                    self.tempch = self.qubit_chs
                elif cfg.expt.pre_sweep_pulse[4][ii] == 3:
                    self.tempch = self.flux_high_ch
                elif cfg.expt.pre_sweep_pulse[4][ii] == 4:
                    self.tempch = self.storage_ch
                elif cfg.expt.pre_sweep_pulse[4][ii] == 5:
                    self.tempch = self.f0g1_ch
                elif cfg.expt.pre_sweep_pulse[4][ii] == 6:
                    self.tempch = self.man_ch
                # print(self.tempch)
                # determine the pulse shape
                if cfg.expt.pre_sweep_pulse[5][ii] == "gaussian":
                    # print('gaussian')
                    self.pisigma_resolved = self.us2cycles(
                        cfg.expt.pre_sweep_pulse[6][ii], gen_ch=self.tempch[0])
                    self.add_gauss(ch=self.tempch[0], name="temp_gaussian",
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch[0], style="arb", 
                                     freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
                                     gain=cfg.expt.pre_sweep_pulse[1][ii], 
                                     waveform="temp_gaussian")
                elif cfg.expt.pre_sweep_pulse[5][ii] == "flat_top":
                    # print('flat_top')
                    self.pisigma_resolved = self.us2cycles(
                        cfg.expt.pre_sweep_pulse[6][ii], gen_ch=self.tempch[0])
                    self.add_gauss(ch=self.tempch[0], name="temp_gaussian",
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch[0], style="flat_top", 
                                     freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
                                     gain=cfg.expt.pre_sweep_pulse[1][ii], 
                                     length=self.us2cycles(cfg.expt.pre_sweep_pulse[2][ii], 
                                                           gen_ch=self.tempch[0]),
                                    waveform="temp_gaussian")
                else:
                    self.setup_and_pulse(ch=self.tempch[0], style="const", 
                                     freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch[0]), 
                                     phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
                                     gain=cfg.expt.pre_sweep_pulse[1][ii], 
                                     length=self.us2cycles(cfg.expt.pre_sweep_pulse[2][ii], 
                                                           gen_ch=self.tempch[0]))
                self.sync_all()
        

        #  optpulse
        # qTest = self.qubits[0]
        if cfg.expt.optpulse:
            if cfg.expt.opt_delay_start[0]>0:
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=self.freq2reg(cfg.expt.opt_freq[0], gen_ch=self.qubit_chs[qTest]), phase=0, 
                                gain=0, length=cfg.expt.opt_delay_start[0])
            if cfg.expt.opt_delay_start[1]>0:
                self.setup_and_pulse(ch=self.man_chs[qTest], style="const", freq=self.freq2reg(cfg.expt.opt_freq[1], gen_ch=self.man_chs[qTest]), phase=0, 
                                gain=0, length=cfg.expt.opt_delay_start[1])
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.freq2reg(cfg.expt.opt_freq[0], gen_ch=self.qubit_chs[qTest]), phase=0, 
                                gain=cfg.expt.opt_gain[0], waveform="test_opt_qubit")
            self.setup_and_pulse(ch=self.man_chs[qTest], style="arb", freq=self.freq2reg(cfg.expt.opt_freq[1], gen_ch=self.man_chs[qTest]), phase=0, 
                                gain=cfg.expt.opt_gain[1], waveform="test_opt_cavity")
            

        #====================================#

        # preparing cats on manipulate

        # displace the cavity
        self.setup_and_pulse(ch=self.man_chs[0], style="arb", freq=self.f_cavity, 
                            phase=self.deg2reg(self.cfg.expt.phase_pre), 
                            gain=self.cfg.expt.amp_pre, waveform="displace_pre")

        # pi/2 pulse to the qubit
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=self.deg2reg(0), gain=self.gain_hge_init, waveform="hpi_qubit")       

        # delay the revival time
        # print(self.revival_ time)
        self.sync_all(self.us2cycles(self.revival_time))

        # pi/2 pulse to the qubit
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=self.deg2reg(0), gain=self.gain_hge_init, waveform="hpi_qubit")

        # align channels and measure
        self.sync_all(self.us2cycles(0.01))
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.expt.photon_releasing)
        )

        self.sync_all()


        #====================================#
                

        # displace the cavity
        self.setup_and_pulse(ch=self.man_chs[0], style="arb", freq=self.f_cavity, 
                            phase=self.deg2reg(self.cfg.expt.phase_placeholder), 
                            gain=self.cfg.expt.amp_placeholder, waveform="displace")

        self.sync_all(self.us2cycles(0.05))

        # pi/2 pulse to the qubit
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=self.deg2reg(0), gain=self.gain_hge_init, waveform="hpi_qubit")       

        # delay the revival time
        # print(self.revival_ time)
        self.sync_all(self.us2cycles(self.revival_time))

        # pi/2 pulse to the qubit
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=self.deg2reg(0), gain=self.gain_hge_init, waveform="hpi_qubit")


        # align channels and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )
    
    def collect_shots(self):
        # collect shots for 1 adc and I and Q channels
        cfg = self.cfg
        # shots_i0 = self.di_buf[0].reshape((1, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]
        # shots_q0 = self.dq_buf[0].reshape((1, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]

        shots_i0 = self.di_buf[0].reshape((2, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]
        # print(shots_i0)
        shots_q0 = self.dq_buf[0].reshape((2, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]

        return shots_i0, shots_q0

# ====================================================== #
                      
class WignerTomography1ModeCatExperiment(Experiment):
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

    def __init__(self, soccfg=None, path='', prefix='WignweTomography1ModeCat', config_file=None, progress=None):
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

        # extract displacement list from file path

        alpha_list = np.load(self.cfg.expt["displacement_path"])

        gain2alpha = self.cfg.expt.gain2alpha

        data={"alpha":[],"avgi":[], "avgq":[], "amps":[], "phases":[], "i0":[], "q0":[]}

        for alpha in tqdm(alpha_list, disable=not progress):
            scale =  1.764162781524843     # np.sqrt(np.pi)*erf(2) = ratio of gaussian/square 
            self.cfg.expt.amp_placeholder =  int(np.abs(alpha)/gain2alpha/scale/self.cfg.expt.displace_length) # scaled, reference is 1us const pulse
            self.cfg.expt.phase_placeholder = np.angle(alpha)/np.pi*180
            lengthrabi = WignerTomography1ModeCatProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], readouts_per_experiment=2, threshold=None, load_pulses=True, progress=False, debug=debug)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(alpha) # Calculating the magnitude
            phase = np.angle(alpha) # Calculating the phase
            data["alpha"].append(alpha)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
            # collect single shots
            i0, q0 = lengthrabi.collect_shots()
            data["i0"].append(i0)
            data["q0"].append(q0)

          
        
        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data
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

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=f"Displace amplitude calibration (Pulse Length {self.cfg.expt.displace_sigma})", ylabel="I [ADC units]")
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
                      
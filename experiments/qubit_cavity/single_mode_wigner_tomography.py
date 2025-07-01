import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm
from MM_base import MMAveragerProgram
# from scipy.sepcial import erf

import experiments.fitting as fitter

class WignerTomography1ModeProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = self.qubits[0]
        # define the displace sigma for calibration     
        self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_ch[0])
        self.displace_sigma = self.us2cycles(cfg.device.manipulate.displace_sigma[0], gen_ch = self.man_ch[0])
       

        self.add_gauss(ch=self.man_ch[0], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest],gen_ch = self.man_ch[0]),
                                  gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])


        self.parity_pulse = self.get_parity_str(1, return_pulse=True, second_phase=180, fast=True)
        # self.chi_shift = cfg.expt.guessed_chi
        # self.ratio = np.cos(np.pi*2*self.chi_shift/4*(2*self.cycles2us(self.tp)+3*self.cycles2us(self.displace_sigma*4)))/np.cos(np.pi*2*self.chi_shift/4*self.cycles2us(self.displace_sigma*4))
        # if cfg.expt.optpulse:
        #     self.add_opt_pulse(ch=self.qubit_chs[0], name="test_opt_qubit", pulse_location=cfg.expt.opt_file_path[0])
        #     self.add_opt_pulse(ch=self.man_chs[0], name="test_opt_cavity", pulse_location=cfg.expt.opt_file_path[1])

        self.sync_all(200)

    

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # phase reset
        self.reset_and_sync()
            
        #  prepulse
        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')


        #  optpulse
        # qTest = self.qubits[0]
        # if cfg.expt.optpulse:
        #     if cfg.expt.opt_delay_start[0]>0:
        #         self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=self.freq2reg(cfg.expt.opt_freq[0], gen_ch=self.qubit_chs[qTest]), phase=0, 
        #                         gain=0, length=cfg.expt.opt_delay_start[0])
        #     if cfg.expt.opt_delay_start[1]>0:
        #         self.setup_and_pulse(ch=self.man_chs[qTest], style="const", freq=self.freq2reg(cfg.expt.opt_freq[1], gen_ch=self.man_chs[qTest]), phase=0, 
        #                         gain=0, length=cfg.expt.opt_delay_start[1])
        #     self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.freq2reg(cfg.expt.opt_freq[0], gen_ch=self.qubit_chs[qTest]), phase=0, 
        #                         gain=cfg.expt.opt_gain[0], waveform="test_opt_qubit")
        #     self.setup_and_pulse(ch=self.man_chs[qTest], style="arb", freq=self.freq2reg(cfg.expt.opt_freq[1], gen_ch=self.man_chs[qTest]), phase=0, 
        #                         gain=cfg.expt.opt_gain[1], waveform="test_opt_cavity")
                

        # displace the cavity
        # phase reset
        # self.set_pulse_registers(ch=self.qubit_ch[0], freq=self.f_ge,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.qubit_ch[0])
        # self.set_pulse_registers(ch=self.man_ch[0], freq=self.f_cav,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.man_ch[0])
        # self.set_pulse_registers(ch=self.f0g1_ch[0], freq=self.f_ge,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.f0g1_ch[0])
        # self.sync_all(10)
        # now displace
        self.setup_and_pulse(ch=self.man_ch[0], style="arb", freq=self.f_cavity, 
                            phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch = self.man_ch[0]), 
                            gain=self.cfg.expt.amp_placeholder, waveform="displace")

        self.sync_all(self.us2cycles(0.05))

        # Parity pulse
        self.custom_pulse(self.cfg, self.parity_pulse, prefix='ParityPulse')

        # align channels and measure
        # self.sync_all(self.us2cycles(0.01))
        self.measure_wrapper()
    
    def collect_shots(self):
        # collect shots for 1 adc and I and Q channels
        cfg = self.cfg
        shots_i0 = self.di_buf[0].reshape((1, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape((1, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]

        return shots_i0, shots_q0

# ====================================================== #
                      
class WignerTomography1ModeExperiment(Experiment):
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

    def __init__(self, soccfg=None, path='', prefix='WignweTomography1Mode', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment( num_qubits_sample) 

        qTest = self.cfg.expt.qubits[0]

        # extract displacement list from file path

        alpha_list = np.load(self.cfg.expt["displacement_path"])
        # alpha_list = [0., 1, 2]
        # alpha_list = np.append(alpha_list, 1j*alpha_list) # add 0 to the end of the list
        # print("alpha_list:", alpha_list)




        man_mode_no = 1
        print(f"man mode no: {man_mode_no}")
        man_mode_idx = man_mode_no -1
        gain2alpha = self.cfg.device.manipulate.gain_to_alpha[man_mode_idx] 
        print(f"gain2alpha: {gain2alpha}")
        displace_sigma = self.cfg.device.manipulate.displace_sigma[man_mode_idx]
        print(f"displace_sigma: {displace_sigma}")

        data={"alpha":[],"avgi":[], "avgq":[], "amps":[], "phases":[], "i0":[], "q0":[]}

        for alpha in tqdm(alpha_list, disable=not progress):
            # scale =  1.764162781524843     # np.sqrt(np.pi)*erf(2) = ratio of gaussian/square 
            scale =  displace_sigma# parity gain calibration Gaussian pulse length here (in unit of us)
            self.cfg.expt.amp_placeholder =  int(np.abs(alpha)/gain2alpha*scale/self.cfg.expt.displace_length) # scaled, reference is a Gaussian pulse
            self.cfg.expt.phase_placeholder = np.angle(alpha)/np.pi*180
            lengthrabi = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                            #  debug=debug
                                             )        
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
        
        self.cfg.expt['expts'] = len(data["alpha"])

          
        
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
                      
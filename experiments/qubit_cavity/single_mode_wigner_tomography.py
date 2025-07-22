import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm
from MM_base import MMAveragerProgram
from fitting_folder.wigner import WignerAnalysis
from qutip import fock  
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

        self.parity_pulse_ = self.get_parity_str(1, return_pulse=True, second_phase=self.cfg.expt.phase_second_pulse, fast=True)
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


        
        self.setup_and_pulse(ch=self.man_ch[0], style="arb", freq=self.f_cavity, 
                            phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch = self.man_ch[0]), 
                            gain=self.cfg.expt.amp_placeholder, waveform="displace")

        self.sync_all(self.us2cycles(0.05))

        # Parity pulse
        self.custom_pulse(self.cfg, self.parity_pulse_, prefix='ParityPulse')

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
        self.format_config_before_experiment(num_qubits_sample) 

        qTest = self.cfg.expt.qubits[0]

        if 'pulse_correction' in self.cfg.expt:
            print("Pulse correction is applied")
            self.pulse_correction = self.cfg.expt.pulse_correction
        else:
            self.pulse_correction = False

        

        # extract displacement list from file path

        alpha_list = np.load(self.cfg.expt["displacement_path"])
        # alpha_list = np.linspace(0, 4, 20)
        # alpha_list = np.append(alpha_list, 1j*alpha_list) # add 0 to the end of the list
        # print("alpha_list:", alpha_list)



        man_mode_no = 1
        # print(f"man mode no: {man_mode_no}")
        man_mode_idx = man_mode_no -1
        gain2alpha = self.cfg.device.manipulate.gain_to_alpha[man_mode_idx] 
        print(f"gain2alpha: {gain2alpha}")
        # print(f"gain2alpha: {gain2alpha}")
        displace_sigma = self.cfg.device.manipulate.displace_sigma[man_mode_idx]
        # print(f"displace_sigma: {displace_sigma}")

        
        ### TRY SOMETHING DIFFERENT
        # revival_time = self.cfg.device.manipulate.revival_time[man_mode_idx]

        data={"alpha":[],"avgi":[], "avgq":[], "amps":[], "phases":[], "i0":[], "q0":[]}

        for alpha in tqdm(alpha_list, disable=not progress):
            self.cfg.expt.phase_second_pulse = 180 # reset the phase of the second pulse
            scale =  displace_sigma# parity gain calibration Gaussian pulse length here (in unit of us)
            self.cfg.expt.amp_placeholder =  int(np.abs(alpha)/gain2alpha*scale/self.cfg.expt.displace_length) # scaled, reference is a Gaussian pulse
            self.cfg.expt.phase_placeholder = np.angle(alpha)/np.pi*180
            wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = wigner
            avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
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
            i0, q0 = wigner.collect_shots()
            data["i0"].append(i0)
            data["q0"].append(q0)

            if self.pulse_correction:
                self.cfg.expt.phase_second_pulse = 0
                wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                                #  debug=debug
                                                )
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                i0, q0 = wigner.collect_shots()
                data["avgi"].append(avgi)
                data["avgq"].append(avgq)
                data["i0"].append(i0)
                data["q0"].append(q0)


            # ### TRY SOMETHING DIFFERENT
            # if self.pulse_correction:
            #     # add i0_calibration and q0_calibration
            #     if 'i0_calibration' not in data:
            #         data['i0_calibration'] = []
            #         data['q0_calibration'] = []
                
            #     # set the revival time to zero, first with second phase = 180
            #     self.cfg.device.manipulate.revival_time[man_mode_idx] = 0
            #     wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
            #     avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
            #                                     #  debug=debug
            #                                     )
            #     i0, q0 = wigner.collect_shots()
            #     data["i0_calibration"].append(i0)
            #     data["q0_calibration"].append(q0)

            #     # now with second phase = 0
            #     self.cfg.expt.phase_second_pulse = 0
            #     wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
            #     avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
            #                                     #  debug=debug
            #                                     )
            #     i0, q0 = wigner.collect_shots()
            #     data["i0_calibration"].append(i0)
            #     data["q0_calibration"].append(q0)

            #     # set the revival time back to the original value
            #     self.cfg.device.manipulate.revival_time[man_mode_idx] = revival_time

        self.cfg.expt['expts'] = len(data["alpha"])

          
        
        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        expt = self.cfg.expt
        if 'pulse_correction' in self.cfg.expt:
            self.pulse_correction = self.cfg.expt.pulse_correction
        else:
            self.pulse_correction = False

        if 'mode_state_num' in kwargs:
            mode_state_num = kwargs['mode_state_num']
        else:
            mode_state_num = 10

        if self.pulse_correction:
            # we need to reshape the data before processing
            data_minus = {}
            data_plus = {}
            data_minus["i0"] = data["i0"][::2, :, :]
            data_minus["q0"] = data["q0"][::2, :, :]
            data_plus["i0"] = data["i0"][1::2, :, :]
            data_plus["q0"] = data["q0"][1::2, :, :]

            wigner_analysis_minus = WignerAnalysis(data=data_minus,
                                                   config=self.cfg, 
                                                    mode_state_num=mode_state_num,
                                                    alphas=data["alpha"])

            wigner_analysis_plus = WignerAnalysis(data=data_plus,
                                                  config=self.cfg,
                                                  mode_state_num=mode_state_num,
                                                  alphas=data["alpha"])
            
            pe_plus = wigner_analysis_plus.bin_ss_data()
            pe_minus = wigner_analysis_minus.bin_ss_data()
            parity_plus = (1 - pe_plus) - pe_plus
            parity_minus = (1 - pe_minus) - pe_minus
            parity = (parity_minus - parity_plus) / 2
            
            data["pe_plus"] = pe_plus
            data["pe_minus"] = pe_minus
            data["parity_plus"] = parity_plus
            data["parity_minus"] = parity_minus
            data["parity"] = parity


            # # TRY SOMETHING DIFFERENT
            # data_calib_plus = {}
            # data_calib_minus = {}
            # data_calib_minus["i0"] = data["i0_calibration"][::2, :, :]
            # data_calib_minus["q0"] = data["q0_calibration"][::2, :, :]
            # data_calib_plus["i0"] = data["i0_calibration"][1::2, :, :]
            # data_calib_plus["q0"] = data["q0_calibration"][1::2, :, :]
            # wigner_analysis_calib_minus = WignerAnalysis(data=data_calib_minus,
            #                                              config=self.cfg, 
            #                                              mode_state_num=mode_state_num,
            #                                              alphas=data["alpha"])
            # wigner_analysis_calib_plus = WignerAnalysis(data=data_calib_plus,
            #                                                 config=self.cfg,
            #                                                 mode_state_num=mode_state_num,
            #                                                 alphas=data["alpha"])
            # pe_calib_plus = wigner_analysis_calib_plus.bin_ss_data()
            # pe_calib_minus = wigner_analysis_calib_minus.bin_ss_data()

            # a = 1 / (pe_calib_plus - pe_calib_minus)
            # b = -pe_calib_minus / (pe_calib_plus - pe_calib_minus)

            # wigner_analysis_calib = WignerAnalysis(data=data,
            #                                        config=self.cfg,
            #                                         mode_state_num=mode_state_num,
            #                                         alphas=data["alpha"])
            # pe = wigner_analysis_calib.bin_ss_data()
            # parity_no_corr = (1 - pe) - pe
            # parity = a*parity_no_corr + b
            # data["pe_calib_plus"] = pe_calib_plus
            # data["pe_calib_minus"] = pe_calib_minus
            # data['a'] = a
            # data['b'] = b
            # data["pe"] = pe
            # data["parity"] = parity
            # data["parity_no_corr"] = parity_no_corr


        else:
            wigner_analysis = WignerAnalysis(data=data,
                                              config=self.cfg, 
                                              mode_state_num=mode_state_num,
                                              alphas=data["alpha"])
            pe = wigner_analysis.bin_ss_data()
            data["pe"] = pe
            data["parity"] = (1 - pe) - pe

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
                      
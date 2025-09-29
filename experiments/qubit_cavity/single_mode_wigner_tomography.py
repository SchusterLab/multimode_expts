import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qick import *
from qick.helpers import gauss
from qutip import fock
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from fit_display_classes import GeneralFitting
from fitting_folder.wigner import WignerAnalysis
from MM_base import MMAveragerProgram

# from scipy.sepcial import erf


class WignerTomography1ModeProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg, loaded_pulses=None):
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


        if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
            waveform_names = self.load_opt_ctrl_pulse(pulse_conf=cfg.expt.opt_pulse, 
                                IQ_table=cfg.expt.IQ_table,
                                ) 
            self.waveforms_opt_ctrl = waveform_names

        

    
    # def body(self):
    #     cfg=AttrDict(self.cfg)
    #     qTest = self.qubits[0]

    #     # phase reset
    #     self.reset_and_sync()

    #     # fire pulses 
    #     self.setup_and_pulse(ch=self.man_ch[0], style="const", freq=self.f_cavity, phase=self.deg2reg(0),
    #                         gain=10000, length=self.us2cycles(5, gen_ch = self.man_ch[0]) )
    #     self.setup_and_pulse(ch=self.qubit_chs[0], style="const", freq=self.f_ge_reg[0], phase=self.deg2reg(0),
    #                         gain=10000, length=self.us2cycles(5, gen_ch = self.qubit_chs[0]) )
    
            
        # #  prepulse
        # if cfg.expt.prepulse:
        #     if cfg.expt.gate_based: 
        #         creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
        #         self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
        #     else: 
        #         print("Using custom pulse for pre-sweep pulse")
        #         print(cfg.expt.pre_sweep_pulse)
        #         self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')


        # if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
        #     creator = self.get_prepulse_creator(cfg.expt.opt_pulse)
        #     self.custom_pulse(cfg, creator.pulse.tolist(),
        #                       waveform_preload=self.waveforms_opt_ctrl)

        # if 'post_select_pre_pulse' in cfg.expt and cfg.expt.post_select_pre_pulse:

        #     # do the eg/ef measurement after the custom pulse, before the tomography
        #     man_reset = False
        #     storage_reset = False
        #     coupler_reset = False
        #     pre_selection_reset = False
        #     ef_reset = False

        #     self.active_reset(man_reset=man_reset, storage_reset=storage_reset,
        #                       coupler_reset=coupler_reset,
        #                       pre_selection_reset=pre_selection_reset,
        #                       ef_reset=ef_reset)

    



        
        # self.setup_and_pulse(ch=self.man_ch[0], style="arb", freq=self.f_cavity, 
        #                     phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch = self.man_ch[0]), 
        #                     gain=self.cfg.expt.amp_placeholder, waveform="displace")

        # self.sync_all(self.us2cycles(0.05))

        # Parity pulse
        # self.custom_pulse(self.cfg, self.parity_pulse_, prefix='ParityPulse')

        # align channels and measure
        # self.sync_all(self.us2cycles(0.01))
        # self.measure_wrapper()
   
   
   
    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # phase reset
        self.reset_and_sync()

        if 'active_reset' in cfg.expt and cfg.expt.active_reset:
            man_reset = False
            storage_reset = False
            coupler_reset = False
            pre_selection_reset = False
            ef_reset = False
            self.active_reset(man_reset=man_reset, storage_reset=storage_reset,
                              coupler_reset=coupler_reset,
                              pre_selection_reset=pre_selection_reset,
                              ef_reset=ef_reset)

        #  prepulse
        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')


        if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
            creator = self.get_prepulse_creator(cfg.expt.opt_pulse)
            self.custom_pulse(cfg, creator.pulse.tolist(),
                              waveform_preload=self.waveforms_opt_ctrl)

        if 'post_select_pre_pulse' in cfg.expt and cfg.expt.post_select_pre_pulse:

            # do the eg/ef measurement after the custom pulse, before the tomography
            man_reset = False
            storage_reset = False
            coupler_reset = False
            pre_selection_reset = False
            ef_reset = False

            self.active_reset(man_reset=man_reset, storage_reset=storage_reset,
                              coupler_reset=coupler_reset,
                              pre_selection_reset=pre_selection_reset,
                              ef_reset=ef_reset)

    



        
        self.setup_and_pulse(ch=self.man_ch[0], style="arb", freq=self.f_cavity, 
                            phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch = self.man_ch[0]), 
                            gain=self.cfg.expt.amp_placeholder, waveform="displace")

        # self.sync_all(self.us2cycles(0.05))
        self.sync_all()

        # Parity pulse
        self.custom_pulse(self.cfg, self.parity_pulse_, prefix='ParityPulse')

        # align channels and measure
        # self.sync_all(self.us2cycles(0.01))
        self.measure_wrapper()
    
    def collect_shots(self):
        # collect shots for 1 adc and I and Q channels
        cfg = self.cfg
        read_num = 1
        if 'active_reset' in cfg.expt and cfg.expt.active_reset:
            read_num += 1
        if 'post_select_pre_pulse' in cfg.expt and cfg.expt.post_select_pre_pulse:
            read_num += 1

        shots_i0 = self.di_buf[0].reshape((1, read_num*self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape((1, read_num*self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]

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
        self._loaded_pulses = set()



    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample) 

        qTest = self.cfg.expt.qubits[0]

        if 'pulse_correction' in self.cfg.expt:
            self.pulse_correction = self.cfg.expt.pulse_correction
        else:
            self.pulse_correction = False

        read_num = 1
        if 'post_select_pre_pulse' in self.cfg.expt and self.cfg.expt.post_select_pre_pulse:
            read_num += 1
        if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
            read_num += 1

        # extract displacement list from file path
        alpha_list = np.load(self.cfg.expt["displacement_path"])

        man_mode_no = 1
        man_mode_idx = man_mode_no -1
        gain2alpha = self.cfg.device.manipulate.gain_to_alpha[man_mode_idx] 
        displace_sigma = self.cfg.device.manipulate.displace_sigma[man_mode_idx]

        data={"alpha":[],"avgi":[], "avgq":[], "amps":[], "phases":[], "i0":[], "q0":[]}

        for alpha in tqdm(alpha_list, disable=not progress):
            self.cfg.expt.phase_second_pulse = 180 # reset the phase of the second pulse
            scale =  displace_sigma# parity gain calibration Gaussian pulse length here (in unit of us)
            _alpha = np.conj(alpha) # convert to conjugate to respect qick convention
            self.cfg.expt.amp_placeholder =  int(np.abs(_alpha)/gain2alpha*scale/self.cfg.expt.displace_length) # scaled, reference is a Gaussian pulse
            self.cfg.expt.phase_placeholder = np.angle(_alpha)/np.pi*180 - 90 # 90 is needed since da/dt = -i*drive
            wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = wigner
            avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                        readouts_per_experiment=read_num,
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
                                            readouts_per_experiment=read_num,
                                                #  debug=debug
                                                )
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                i0, q0 = wigner.collect_shots()
                data["avgi"].append(avgi)
                data["avgq"].append(avgq)
                data["i0"].append(i0)
                data["q0"].append(q0)

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

        read_num = 1
        if 'post_select_pre_pulse' in self.cfg.expt and self.cfg.expt.post_select_pre_pulse:
            read_num += 1
        if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
            read_num += 1

        idx_start = read_num - 1
        idx_step = read_num
        idx_post_select = 0 
        if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
            idx_post_select += 1

        if self.pulse_correction:
            # we need to reshape the data before processing
            # if pulse correction i0 = [i_minus0, i_plus0, i_minus1, i_plus1, ...]
            # if post_select_pre_pulse i0 = [i_gem0, i_efm0, i_minus0, i_gep0, i_efp0, i_plus0, ...]

            data_minus = {}
            data_plus = {}



            data_minus["i0"] = data["i0"][0::2, :, idx_start::idx_step]
            data_minus["q0"] = data["q0"][0::2, :, idx_start::idx_step]
            data_plus["i0"] = data["i0"][1::2, :, idx_start::idx_step]
            data_plus["q0"] = data["q0"][1::2, :, idx_start::idx_step]

            if 'post_select_pre_pulse' in self.cfg.expt and self.cfg.expt.post_select_pre_pulse:
                I_eg = data["i0"][0::2, 0, idx_post_select::idx_step]
                Q_eg = data["q0"][0::2, 0, idx_post_select::idx_step]

                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.plot(I_eg[0, :], Q_eg[0, :], 'o')
                ax.set_title("I_EG vs Q_EG")
                # axis should be equal
                ax.axis('equal')
                fig.tight_layout()

                data["I_postpulse_minus"] = I_eg
                data["Q_postpulse_minus"] = Q_eg

                data_temp = {}
                data_temp["i0"] = data["i0"][0::2, :, idx_post_select::idx_step]
                data_temp["q0"] = data["q0"][0::2, :, idx_post_select::idx_step]
                wigner_analysis = WignerAnalysis(data=data_temp,
                                                    config=self.cfg,
                                                    mode_state_num=mode_state_num,
                                                    alphas=data["alpha"])
                pe_postpulse_minus = 1 - wigner_analysis.bin_ss_data()

                data_temp = {}
                data_temp["i0"] = data["i0"][1::2, :, idx_post_select::idx_step]
                data_temp["q0"] = data["q0"][1::2, :, idx_post_select::idx_step]
                wigner_analysis = WignerAnalysis(data=data_temp,
                                                    config=self.cfg,
                                                    mode_state_num=mode_state_num,
                                                    alphas=data["alpha"])
                pe_postpulse_plus = 1 - wigner_analysis.bin_ss_data()

                pe_postpulse = np.average((pe_postpulse_plus + pe_postpulse_minus) / 2)
                data["pe_postpulse"] = pe_postpulse
                data["pe_postpulse_plus"] = pe_postpulse_plus
                data["pe_postpulse_minus"] = pe_postpulse_minus

                # apply thresholding on I_eg and calibration matrix to get pe


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


        else:
            data_wigner = {}
            idx_start = read_num - 1
            idx_step = read_num
            data_wigner["i0"] = data["i0"][:, :, idx_start::idx_step]
            data_wigner["q0"] = data["q0"][:, :, idx_start::idx_step]

            wigner_analysis = WignerAnalysis(data=data_wigner,
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


class WignerTomographyOptimalPulseExperiment(WignerTomography1ModeExperiment):

    def __init__(self, soccfg=None, path='', prefix='WignerTomographyOptimalPulseExperiment', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, 
                         path=path, 
                         prefix=prefix,
                           config_file=config_file,
                             progress=progress)
        
    def acquire(self, progress=False, debug=False):

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

        ### 

        pulse_to_test = self.cfg.expt.pulse_to_test
        nb_plot = self.cfg.expt.nb_plot


        self.filename = self.cfg.device.optimal_control[pulse_to_test[0][1]][pulse_to_test[0][2]]['filename']
        data = np.load(self.filename, allow_pickle=True)
        times = data['times'] * 1e-3
        Ic = data['I_c']
        Qc = -data['Q_c']
        Iq = data['I_q']
        Qq = -data['Q_q']
        qb_scale = max((np.max(np.abs(Iq)), np.max(np.abs(Qq))))
        cav_scale = max((np.max(np.abs(Ic)), np.max(np.abs(Qc))))

        if qb_scale !=0: 
            Iq /= qb_scale
            Qq /= qb_scale

        if cav_scale !=0: 
            Ic /= cav_scale
            Qc /= cav_scale

        t_length = len(times)
        t_step = (times[-1] - times[0]) / nb_plot
        t_to_plot = np.arange(0, times[-1]+t_step, t_step)
        t_to_plot[0] +=0.05
        self.t_to_plot = t_to_plot

        data = []

        for i in range(len(t_to_plot)):
            idx_t  = np.argmin(np.abs(times - t_to_plot[i]))
            IQ_table = {
                'I_c': Ic[:idx_t],
                'Q_c': Qc[:idx_t],
                'I_q': Iq[:idx_t],
                'Q_q': Qq[:idx_t],
                'times': times[:idx_t]
            }

            wigner = WignerTomography1ModeProgram(soccfg=self.soccfg,
                                                   cfg=self.cfg)
            data_temp = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, progress=False,
                                        # readouts_per_experiment=1,
                                        # debug=debug,
                                        # IQ_table=IQ_table
                                        )
            
            data.append(data_temp)

        return data

    def analyze(self, data=None, cutoff=10,
                state=qt.fock(10, 0),
                rotate=False,
                plot=False,

                  **kwargs):
        if data is None:
            data = self.data
        
        data_analyzed = []

        for i, data_temp in enumerate(data):

            _data_analysed = super().analyze(data=data_temp, **kwargs)
            wigner_analysis = WignerAnalysis(_data_analysed, config=self.cfg,
                                      mode_state_num=cutoff, alphas = data_temp['alpha'])
            
            results = wigner_analysis.wigner_analysis_results(data_temp['parity'],
                                    initial_state=state, rotate=rotate)
            # if plot:
            #     fig = wigner_analysis.plot_wigner_reconstruction_results(results, initial_state=state, state_label=f'')
            
            _data_analysed.append(results)
            data_analyzed.append(_data_analysed)

        self.data_analyzed = data_analyzed

                # Insert the provided analysis code here to calculate photon number distributions and fidelity
        plot_wigner = plot
        n_distr = np.zeros((len(data), cutoff))
        p_vec = np.zeros((len(data), 2))
        
        # For each time slice, use the rho to compute the photon number distribution
        for i in range(len(data)):
            _n_distr = np.diag(data[i]['rho'])
            n_distr[i, :] = _n_distr
            p_vec[i, 1] = data[i]['pe_postpulse']
            p_vec[i, 0] = 1 - p_vec[i, 1]

        filename_th = self.filename.replace('pulse', 'pop_sim/populations')
        data_th = np.load(filename_th, allow_pickle=True)
        n_distr_th = data_th['cavity']
        p_vec_th = data_th['qubit'][1, :]
        times_th = data_th['times'] * 1e-3
        rho_cav_t = data_th['rho_cav_t']
        t_to_plot = self.t_to_plot


        F_vec = np.zeros(len(data))

        for i in range(len(data)):
            rho_exp = qt.Qobj(data[i]['rho'])
            cutoff = rho_exp.shape[0]
            idx_t = np.argmin(np.abs(times_th - t_to_plot[i]))
            rho_th = qt.Qobj(rho_cav_t[:cutoff, :cutoff, idx_t])
            F_vec[i] = qt.fidelity(rho_exp, rho_th)

            if plot_wigner:
                vmin = -2 / np.pi
                vmax = 2 / np.pi
                alpha_list = data[i]['alpha']
                alpha_max = np.max(np.abs(alpha_list))*1.5
                x_vec = np.linspace(-alpha_max, alpha_max, 150)
                W_exp = qt.wigner(rho_exp, x_vec, x_vec)
                W_th = qt.wigner(rho_th, x_vec, x_vec)
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].pcolormesh(x_vec, x_vec, W_exp, vmin=vmin, vmax=vmax, cmap='RdBu_r')
                ax[0].set_xlabel('Re(α)')
                ax[0].set_ylabel('Im(α)')
                ax[0].set_title(f'Exp at t={t_to_plot[i]:.2f} us')

                ax[1].pcolormesh(-x_vec, -x_vec, W_th, vmin=vmin, vmax=vmax, cmap='RdBu_r')
                ax[1].set_xlabel('Re(α)')
                ax[1].set_ylabel('Im(α)')
                ax[1].set_title(f'Th')
                fig.tight_layout()


        # Plot a heatmap of the photon number distribution and qubit probabilities
        fig1, ax1 = plt.subplots(2, 1, figsize=(6, 6))
        cax = ax1[0].imshow(n_distr.T, 
                            aspect='auto', origin='lower', cmap='viridis', 
                            extent=[t_to_plot[0], t_to_plot[-1], 0, n_distr.shape[1]-1])
        ax1[0].set_xlabel('Time (us)')
        ax1[0].set_ylabel('Photon Number')
        ax1[0].set_title('Photon Number Distribution exp/theory')

        cax2 = ax1[1].imshow(n_distr_th[:n_distr.T.shape[0], ],
                             aspect='auto', origin='lower', cmap='viridis',
                             extent=[times_th[0], times_th[-1], 0, n_distr.T.shape[0]-1])
        ax1[1].set_xlabel('Time (us)')
        ax1[1].set_ylabel('Photon Number')
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6))

        ax2[0].plot(t_to_plot, p_vec[:, 0], 'o-', label='e',
                   color='tab:blue', markersize=4, linewidth=1.2)
        ax2[0].plot(t_to_plot, p_vec[:, 1], 'o-', label='g',
                   color='tab:red', markersize=4, linewidth=1.2)
        ax2[0].plot(times_th, p_vec_th, linestyle='--', color='tab:blue')
        ax2[0].plot(times_th, 1-p_vec_th, linestyle='--', color='tab:red')
        ax2[0].set_xlabel('Time (us)')
        ax2[0].set_ylabel('Probability')
        ax2[0].set_title('Qubit Probabilities')
        ax2[0].legend()

        ax2[1].plot(t_to_plot, F_vec, 'o-', color='tab:green', markersize=4, linewidth=1.2)
        ax2[1].set_xlabel('Time (us)')
        ax2[1].set_ylabel('Fidelity')
        ax2[1].set_title('Cavity state')
        fig2.tight_layout()

        # save the plots 
        gen_fit = GeneralFitting(data, threshold='dummy')
        gen_fit.save_plot(fig1, filename='photon_number_distribution')
        gen_fit.save_plot(fig2, filename='qubit_prob_and_fid')


    def save_data(self, data=None):

        if data is None:
            data = self.data_analyzed
        print(f'Saving {self.fname}')
        super().save_data(data=data)

            






    




                                            
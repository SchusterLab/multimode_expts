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


        self.parity_pulse_ = self.get_parity_str(1, 
                                                 return_pulse=True, 
                                                 second_phase=self.cfg.expt.phase_second_pulse,
                                                   fast=False)
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

    def display(self, data=None, mode_state_num=None, initial_state=None, rotate=None, state_label='', debug_components=False, **kwargs):
        """
        Display using WignerAnalysis reconstruction pipeline.

        Parameters:
        - mode_state_num: Hilbert space cutoff (default from cfg.expt.display_mode_state_num or 5)
        - initial_state: qutip.Qobj (ket) used for reconstruction; if None, defaults to |0> in the chosen dimension
        - rotate: whether to rotate the Wigner frame (default from cfg.expt.display_rotate or False)
        - state_label: optional label for the plotted state
        - debug_components: when True and pulse correction data are available, plot pe_plus/minus and parity_plus/minus vs |alpha|
        """
        if data is None:
            data = self.data

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', False))

        # Basic validation
        if 'parity' not in data or 'alpha' not in data:
            raise ValueError('Expected keys "parity" and "alpha" in data for Wigner display.')
        parity = data['parity']
        alphas = data['alpha']
        if len(parity) != len(alphas):
            raise ValueError(f'Length mismatch: parity ({len(parity)}) vs alpha ({len(alphas)}).')

        # Default initial state if not provided
        if initial_state is None:
            initial_state = qt.fock(mode_state_num, 0).unit()

        # Build analysis and plot
        wigner_analysis = WignerAnalysis(data=data, config=self.cfg, mode_state_num=mode_state_num, alphas=alphas)
        results = wigner_analysis.wigner_analysis_results(parity, initial_state=initial_state, rotate=rotate)
        fig = wigner_analysis.plot_wigner_reconstruction_results(results, initial_state=initial_state, state_label=state_label)

        # Optional debug components plot (requires pulse-correction products)
        if debug_components:
            has_pe = ('pe_plus' in data) and ('pe_minus' in data) and (data.get('pe_plus') is not None) and (data.get('pe_minus') is not None)
            has_par = ('parity_plus' in data) and ('parity_minus' in data) and (data.get('parity_plus') is not None) and (data.get('parity_minus') is not None)
            if has_pe and has_par:
                try:
                    pe_plus = np.asarray(data['pe_plus'])
                    pe_minus = np.asarray(data['pe_minus'])
                    parity_plus = np.asarray(data['parity_plus'])
                    parity_minus = np.asarray(data['parity_minus'])
                    parity = np.asarray(data['parity'])
                    alpha_abs = np.abs(alphas)

                    fig_dbg, ax = plt.subplots(1, 2, figsize=(12, 6))
                    pe = pe_plus + pe_minus
                    ax[0].plot(alpha_abs, pe, 'o', label='pe')
                    ax[0].plot(alpha_abs, pe_plus, 'o', label='pe_plus')
                    ax[0].plot(alpha_abs, pe_minus, 'o', label='pe_minus')

                    ax[1].plot(alpha_abs, 2/np.pi * parity, 'o', label='parity')
                    ax[1].plot(alpha_abs, 2/np.pi * parity_plus, 'o', label='parity_plus')
                    ax[1].plot(alpha_abs, 2/np.pi * parity_minus, 'o', label='parity_minus')

                    ax[0].set_xlabel('Alpha')
                    ax[0].set_ylabel('Probability')
                    ax[1].set_xlabel('Alpha')
                    ax[1].set_ylabel('Parity')
                    ax[0].legend()
                    ax[1].legend()
                    plt.tight_layout()
                except Exception as e:
                    print(f"Debug components plot skipped due to: {e}")
            else:
                print("Debug components requested, but pe+/pe- or parity+/parity- not found in data.")
        return fig

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


# ====================================================== #
#                Process Tomography (1 mode)             #
# ====================================================== #

class ProcessTomographyProgram(MMAveragerProgram):
    """
    Program sequence per shot:
    - Optional active reset
    - State preparation via custom_pulse (cfg.expt.state_prep) respecting cfg.expt.gate_based
    - Optional prepulse (cfg.expt.prepulse)
    - Wait for cfg.expt.wait_time [us]
    - Optional postpulse (cfg.expt.postpulse)
    - Displacement (amp_placeholder/phase_placeholder)
    - Parity pulse (phase set via cfg.expt.phase_second_pulse)
    - Measure
    """

    def __init__(self, soccfg, cfg, loaded_pulses=None):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = self.qubits[0]

        # Manipulation channel and displacement GAUSS
        self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_ch[0])
        self.displace_sigma = self.us2cycles(cfg.device.manipulate.displace_sigma[0], gen_ch=self.man_ch[0])
        self.add_gauss(ch=self.man_ch[0], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)

        # Readout setup
        self.set_pulse_registers(
            ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest],
            phase=self.deg2reg(cfg.device.readout.phase[qTest], gen_ch=self.man_ch[0]),
            gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest]
        )

        # Parity pulse (phase configured externally)
        self.parity_pulse_ = self.get_parity_str(
            1, return_pulse=True, second_phase=self.cfg.expt.phase_second_pulse, fast=False
        )

        self.sync_all(200)

        if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
            waveform_names = self.load_opt_ctrl_pulse(
                pulse_conf=cfg.expt.opt_pulse, IQ_table=cfg.expt.IQ_table,
            )
            self.waveforms_opt_ctrl = waveform_names

    def body(self):
        cfg = AttrDict(self.cfg)
        # reset phases
        self.reset_and_sync()

        # Optional active reset
        if 'active_reset' in cfg.expt and cfg.expt.active_reset:
            self.active_reset(
                man_reset=False, storage_reset=False, coupler_reset=False,
                pre_selection_reset=False, ef_reset=False
            )

        # State preparation
        if 'state_prep' in cfg.expt and cfg.expt.state_prep:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.state_prep)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='state_')
            else:
                self.custom_pulse(cfg, cfg.expt.state_prep, prefix='state_')

        # Prepulse
        if 'prepulse' in cfg.expt and cfg.expt.prepulse:
            # print("Applying prepulse")
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.prepulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='pre_')
            else:
                # print("Using custom pulse for prepulse")
                self.custom_pulse(cfg, cfg.expt.prepulse, prefix='pre_')

        # Wait
        wait_us = cfg.expt.get('wait_time', 0.0) or 0.0
        if wait_us > 0:
            self.sync_all(self.us2cycles(wait_us))

        # Postpulse
        if 'postpulse' in cfg.expt and cfg.expt.postpulse:
            # print("Applying postpulse")
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.postpulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='post_')
            else:
                # print("Using custom pulse for postpulse")
                self.custom_pulse(cfg, cfg.expt.postpulse, prefix='post_')

        # Displacement for Wigner tomography
        self.setup_and_pulse(
            ch=self.man_ch[0], style="arb", freq=self.f_cavity,
            phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch=self.man_ch[0]),
            gain=self.cfg.expt.amp_placeholder, waveform="displace"
        )
        self.sync_all()

        # Parity pulse and measure
        self.custom_pulse(self.cfg, self.parity_pulse_, prefix='ParityPulse')
        self.measure_wrapper()

    def collect_shots(self):
        # collect shots for 1 adc and I and Q channels
        cfg = self.cfg
        read_num = 1
        if 'active_reset' in cfg.expt and cfg.expt.active_reset:
            read_num += 1
        if 'post_select_pre_pulse' in cfg.expt and cfg.expt.post_select_pre_pulse:
            read_num += 1
        shots_i0 = self.di_buf[0].reshape((1, read_num*self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape((1, read_num*self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        return shots_i0, shots_q0


class ProcessTomographyExperiment(Experiment):
    """
    Process tomography over 4 cardinal states and a wait-time sweep, performing 1-mode Wigner tomography for each condition.

    expt keys:
    - qubits: [q]
    - cardinal_states: list of 4 pulses (list-of-lists) accepted by custom_pulse
    - gate_based: bool — applies to state_prep, prepulse, postpulse
    - prepulse: optional pulse (list-of-lists)
    - postpulse: optional pulse (list-of-lists)
    - wait_start, wait_step, wait_expts (us): defines wait_list
    - displacement_path: .npy path of complex alphas
    - displace_length: scaling ref, re-used from Wigner experiment
    - reps, rounds, active_reset, post_select_pre_pulse, opt_pulse, IQ_table: standard
    - pulse_correction: bool — if True, perform two acquisitions per alpha with phase_second_pulse=180 then 0 (non-inline)
    """

    def __init__(self, soccfg=None, path='', prefix='ProcessTomography', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def _build_wait_list(self):
        ex = self.cfg.expt
        if 'wait_list' in ex and ex.wait_list is not None:
            return np.array(ex.wait_list, dtype=float)
        start = float(ex.wait_start)
        step = float(ex.wait_step)
        expts = int(ex.wait_expts)
        return start + step * np.arange(expts, dtype=float)

    def acquire(self, progress=False, debug=False):
        # Expand singleton config entries
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        # Read control flags
        self.pulse_correction = bool(self.cfg.expt.get('pulse_correction', False))

        # Readouts per experiment lane count
        read_num = 1
        if self.cfg.expt.get('post_select_pre_pulse', False):
            read_num += 1
        if self.cfg.expt.get('active_reset', False):
            read_num += 1

        # Displacements
        alpha_list = np.load(self.cfg.expt["displacement_path"])  # complex ndarray
        man_mode_idx = 0  # using first manipulate channel index
        gain2alpha = self.cfg.device.manipulate.gain_to_alpha[man_mode_idx]
        displace_sigma = self.cfg.device.manipulate.displace_sigma[man_mode_idx]

        # Wait sweep and states
        wait_list = self._build_wait_list()
        states = self.cfg.expt.cardinal_states
        if not isinstance(states, (list, tuple)) or len(states) != 4:
            raise ValueError("expt.cardinal_states must be a list of 4 pulse payloads.")

        # Prepare data containers
        nS, nW, nA = 4, len(wait_list), len(alpha_list)
        pc_factor = 2 if self.pulse_correction else 1

        data = {
            "alpha": np.array(alpha_list),
            "wait_list": np.array(wait_list, dtype=float),
            "avgi": np.empty((nS, nW, nA*pc_factor), dtype=object),
            "avgq": np.empty((nS, nW, nA*pc_factor), dtype=object),
            "amps": np.empty((nS, nW, nA*pc_factor), dtype=float),
            "phases": np.empty((nS, nW, nA*pc_factor), dtype=float),
            "i0": np.empty((nS, nW, nA*pc_factor), dtype=object),
            "q0": np.empty((nS, nW, nA*pc_factor), dtype=object),
        }

        # Iterate: states -> waits -> alphas
        for si, state_prep in enumerate(states):
            for wi, wait_us in enumerate(wait_list):
                # reset phase for default parity (minus)
                self.cfg.expt.phase_second_pulse = 180
                self.cfg.expt.state_prep = state_prep
                self.cfg.expt.wait_time = float(wait_us)

                for ai, alpha in enumerate(tqdm(alpha_list, disable=not progress)):
                    # Map alpha to amp/phase placeholders
                    self.cfg.expt.phase_second_pulse = 180
                    scale = displace_sigma
                    _alpha = np.conj(alpha)
                    self.cfg.expt.amp_placeholder = int(np.abs(_alpha)/gain2alpha * scale / self.cfg.expt.displace_length)
                    self.cfg.expt.phase_placeholder = np.angle(_alpha)/np.pi*180 - 90

                    # First acquisition: parity minus
                    prog = ProcessTomographyProgram(soccfg=self.soccfg, cfg=self.cfg)
                    self.prog = prog
                    avgi, avgq = prog.acquire(
                        self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                        readouts_per_experiment=read_num,
                    )
                    avgi = avgi[0][0]
                    avgq = avgq[0][0]
                    amp = np.abs(alpha)
                    phase = np.angle(alpha)
                    # Index position for minus
                    idx = ai*pc_factor
                    data["avgi"][si, wi, idx] = avgi
                    data["avgq"][si, wi, idx] = avgq
                    data["amps"][si, wi, idx] = amp
                    data["phases"][si, wi, idx] = phase
                    i0, q0 = prog.collect_shots()
                    data["i0"][si, wi, idx] = i0
                    data["q0"][si, wi, idx] = q0

                    # Second acquisition: parity plus (if enabled)
                    if self.pulse_correction:
                        self.cfg.expt.phase_second_pulse = 0
                        prog = ProcessTomographyProgram(soccfg=self.soccfg, cfg=self.cfg)
                        avgi2, avgq2 = prog.acquire(
                            self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                            readouts_per_experiment=read_num,
                        )
                        avgi2 = avgi2[0][0]
                        avgq2 = avgq2[0][0]
                        idx2 = idx + 1
                        data["avgi"][si, wi, idx2] = avgi2
                        data["avgq"][si, wi, idx2] = avgq2
                        data["amps"][si, wi, idx2] = amp
                        data["phases"][si, wi, idx2] = phase
                        i02, q02 = prog.collect_shots()
                        data["i0"][si, wi, idx2] = i02
                        data["q0"][si, wi, idx2] = q02

        # Store meta
        self.data = data
        return data

    def analyze(self, data=None, mode_state_num=None, initial_state=None, rotate=None, **kwargs):
        if data is None:
            data = self.data

        # Read flags
        post_select = bool(self.cfg.expt.get('post_select_pre_pulse', False))
        active_reset = bool(self.cfg.expt.get('active_reset', False))
        pulse_correction = bool(self.cfg.expt.get('pulse_correction', False))

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', True))

        if initial_state is None:
            initial_state = qt.fock(mode_state_num, 0).unit()


        # Readout lanes
        read_num = 1
        if post_select:
            read_num += 1
        if active_reset:
            read_num += 1

        idx_start = read_num - 1
        idx_step = read_num
        idx_post_select = 0
        if active_reset:
            idx_post_select += 1

        nS, nW, nApc = data['avgi'].shape
        pc_factor = 2 if pulse_correction else 1

        # Containers for parity results
        results = {
            'parity': np.empty((nS, nW), dtype=object)
        }
        if pulse_correction:
            results.update({
                'pe_plus': np.empty((nS, nW), dtype=object),
                'pe_minus': np.empty((nS, nW), dtype=object),
                'parity_plus': np.empty((nS, nW), dtype=object),
                'parity_minus': np.empty((nS, nW), dtype=object),
                'rho': np.empty((nS, nW), dtype=object),
                'rho_rotated': np.empty((nS, nW), dtype=object),
                'theta_opt': np.empty((nS, nW), dtype=object),
            })

        # Ensure expts is set for downstream single-shot binning
        nA = len(data['alpha'])
        # We'll construct a local cfg clone with correct expts
        from copy import deepcopy

        for si in range(nS):
            for wi in range(nW):
                # Build small dicts like in Wigner analysis
                if pulse_correction:
                    cfg_local = AttrDict(deepcopy(self.cfg))
                    cfg_local.expt.expts = nA
                    data_minus = {
                        'i0': np.stack([data['i0'][si, wi, k][0] for k in range(0, nApc, 2)], axis=0)[:, idx_start::idx_step],
                        'q0': np.stack([data['q0'][si, wi, k][0] for k in range(0, nApc, 2)], axis=0)[:, idx_start::idx_step],
                    }
                    data_plus = {
                        'i0': np.stack([data['i0'][si, wi, k][0] for k in range(1, nApc, 2)], axis=0)[:, idx_start::idx_step],
                        'q0': np.stack([data['q0'][si, wi, k][0] for k in range(1, nApc, 2)], axis=0)[:, idx_start::idx_step],
                    }

                    if post_select:
                        I_eg = data_minus['i0'][:, 0, idx_post_select::idx_step]
                        Q_eg = data_minus['q0'][:, 0, idx_post_select::idx_step]
                        # Optionally store these if desired

                    wa_minus = WignerAnalysis(data=data_minus, config=cfg_local, mode_state_num=mode_state_num, alphas=data['alpha'])
                    wa_plus = WignerAnalysis(data=data_plus, config=cfg_local, mode_state_num=mode_state_num, alphas=data['alpha'])
                    pe_minus = wa_minus.bin_ss_data()
                    pe_plus = wa_plus.bin_ss_data()
                    parity_minus = (1 - pe_minus) - pe_minus
                    parity_plus = (1 - pe_plus) - pe_plus
                    parity = (parity_minus - parity_plus) / 2
                    data_slice = {'alpha': data['alpha'], 'parity': parity}

                    wigner_analysis = WignerAnalysis(data=data_slice, config=cfg_local,
                                         mode_state_num=mode_state_num, alphas=data['alpha'])

                    wigner_result = wigner_analysis.wigner_analysis_results(parity,
                                                                initial_state=initial_state,
                                                                rotate=rotate)


                    print('Storing results for state {}, wait {}'.format(si, wi))
                    print('rho', wigner_result['rho'])


                    results['pe_minus'][si, wi] = pe_minus
                    results['pe_plus'][si, wi] = pe_plus
                    results['parity_minus'][si, wi] = parity_minus
                    results['parity_plus'][si, wi] = parity_plus
                    results['parity'][si, wi] = parity
                    results['rho'][si, wi] = wigner_result['rho']
                    results['rho_rotated'][si, wi] = wigner_result['rho_rotated']
                    results['theta_opt'][si, wi] = wigner_result['theta_max']

                    print('rho stored:', results['rho'][si, wi])



                else:
                    # No correction: use only the last readout lane (stride select)
                    cfg_local = AttrDict(deepcopy(self.cfg))
                    cfg_local.expt.expts = nA
                    data_w = {
                        'i0': np.stack([data['i0'][si, wi, k][0] for k in range(0, nApc, 1)], axis=0)[:, idx_start::idx_step],
                        'q0': np.stack([data['q0'][si, wi, k][0] for k in range(0, nApc, 1)], axis=0)[:, idx_start::idx_step],
                    }
                    wa = WignerAnalysis(data=data_w, config=cfg_local, mode_state_num=mode_state_num, alphas=data['alpha'])
                    pe = wa.bin_ss_data()
                    parity = (1 - pe) - pe
                    results['parity'][si, wi] = parity

                    data_slice = {'alpha': data['alpha'], 'parity': parity}
                    wigner_analysis = WignerAnalysis(data=data_slice, config=cfg_local,
                                            mode_state_num=mode_state_num, alphas=data['alpha'])
                    wigner_result = wigner_analysis.wigner_analysis_results(parity,
                                                                initial_state=initial_state,
                                                                rotate=rotate)
                    results['rho'][si, wi] = wigner_result['rho']
                    results['rho_rotated'][si, wi] = wigner_result['rho_rotated']
                    results['theta_opt'][si, wi] = wigner_result['theta_max']







        # Merge into data
        for k, v in results.items():
            data[k] = v
        return data

    def display(self, data=None, state_idx=0, wait_idx=0, mode_state_num=None, initial_state=None, rotate=None, state_label='', debug_components=False, **kwargs):
        """
        Display a selected (state, wait) slice using WignerAnalysis reconstruction.

        Parameters:
        - state_idx: index in expt.cardinal_states (0..3)
        - wait_idx: index in wait_list
        - mode_state_num: Hilbert space cutoff (default cfg.expt.display_mode_state_num or 5)
        - initial_state: qutip.Qobj (ket); if None, defaults to |0>
        - rotate: default cfg.expt.display_rotate or False
        - state_label: annotation for plot
        """
        if data is None:
            data = self.data

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', False))

        alphas = data['alpha']
        parity = data['parity'][state_idx, wait_idx]
        if parity is None:
            print("No parity data available for the selected slice.")
            return
        if len(parity) != len(alphas):
            raise ValueError(f'Length mismatch: parity ({len(parity)}) vs alpha ({len(alphas)}).')

        if initial_state is None:
            initial_state = qt.fock(mode_state_num, 0).unit()

        # Minimal dict for analysis API compatibility if needed
        data_slice = {'alpha': alphas, 'parity': parity}
        wigner_analysis = WignerAnalysis(data=data_slice, config=self.cfg, mode_state_num=mode_state_num, alphas=alphas)
        results = wigner_analysis.wigner_analysis_results(parity, initial_state=initial_state, rotate=rotate)
        fig = wigner_analysis.plot_wigner_reconstruction_results(results, initial_state=initial_state, state_label=state_label)

        # Optional debug components: show pe_plus/minus and parity_plus/minus vs |alpha|
        if debug_components:
            has_pe = ('pe_plus' in data) and ('pe_minus' in data)
            has_par = ('parity_plus' in data) and ('parity_minus' in data)
            if has_pe and has_par:
                pe_plus = data['pe_plus'][state_idx, wait_idx] if data['pe_plus'][state_idx, wait_idx] is not None else None
                pe_minus = data['pe_minus'][state_idx, wait_idx] if data['pe_minus'][state_idx, wait_idx] is not None else None
                parity_plus = data['parity_plus'][state_idx, wait_idx] if data['parity_plus'][state_idx, wait_idx] is not None else None
                parity_minus = data['parity_minus'][state_idx, wait_idx] if data['parity_minus'][state_idx, wait_idx] is not None else None
                if pe_plus is not None and pe_minus is not None and parity_plus is not None and parity_minus is not None:
                    try:
                        alpha_abs = np.abs(alphas)
                        fig_dbg, ax = plt.subplots(1, 2, figsize=(12, 6))
                        pe = pe_plus + pe_minus
                        ax[0].plot(alpha_abs, pe, 'o', label='pe')
                        ax[0].plot(alpha_abs, pe_plus, 'o', label='pe_plus')
                        ax[0].plot(alpha_abs, pe_minus, 'o', label='pe_minus')

                        ax[1].plot(alpha_abs, 2/np.pi * np.asarray(parity), 'o', label='parity')
                        ax[1].plot(alpha_abs, 2/np.pi * np.asarray(parity_plus), 'o', label='parity_plus')
                        ax[1].plot(alpha_abs, 2/np.pi * np.asarray(parity_minus), 'o', label='parity_minus')

                        ax[0].set_xlabel('Alpha')
                        ax[0].set_ylabel('Probability')
                        ax[1].set_xlabel('Alpha')
                        ax[1].set_ylabel('Parity')
                        ax[0].legend()
                        ax[1].legend()
                        plt.tight_layout()
                    except Exception as e:
                        print(f"Debug components plot skipped due to: {e}")
                else:
                    print("Debug components requested, but components for selected slice are missing.")
            else:
                print("Debug components requested, but pe+/pe- or parity+/parity- not found in data.")
        return fig

    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data

        serial = {}

        # Metadata
        for key in ("alpha", "wait_list"):
            if key in data:
                serial[key] = np.asarray(data[key])

        # Numeric tensors
        for key in ("amps", "phases"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype != object:
                serial[key] = data[key]

        def stack_1d(obj_arr):
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = elem
                    break
            if sample is None:
                return np.zeros((*obj_arr.shape, 0), dtype=float)
            arr_sample = np.asarray(sample)
            # Handle scalar or 0-D array
            L = int(arr_sample.size) if arr_sample.ndim == 0 else int(arr_sample.shape[-1])
            out = np.zeros((*obj_arr.shape, L), dtype=float)
            for idx in np.ndindex(obj_arr.shape):
                if obj_arr[idx] is not None:
                    v = np.asarray(obj_arr[idx]).reshape(-1)
                    out[idx] = v[:L] if v.shape[0] >= L else np.pad(v, (0, L-v.shape[0]))
            return out

        def stack_2d(obj_arr):
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = elem
                    break
            if sample is None:
                return np.zeros((*obj_arr.shape, 0), dtype=float)
            arr_sample = np.asarray(sample)
            # Handle scalar or shape mismatch
            if arr_sample.ndim == 0:
                M = 1
            elif arr_sample.ndim == 1:
                M = arr_sample.shape[0]
            else:
                M = arr_sample.shape[-1]
            out = np.zeros((*obj_arr.shape, M), dtype=float)
            for idx in np.ndindex(obj_arr.shape):
                if obj_arr[idx] is not None:
                    a = np.asarray(obj_arr[idx])
                    v = a[0].reshape(-1) if a.ndim >= 2 else a.reshape(-1)
                    out[idx] = v[:M] if v.shape[0] >= M else np.pad(v, (0, M-v.shape[0]))
            return out
        
        def stack_3d(obj_arr):
            """Stack object array of 2D matrices to (obj_arr.shape, rows, cols) tensor"""
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = elem
                    break
            if sample is None:
                return np.zeros((*obj_arr.shape, 0, 0), dtype=complex)
            arr_sample = np.asarray(sample)
            # Determine matrix dimensions
            if arr_sample.ndim == 2:
                rows, cols = arr_sample.shape
            else:
                # Handle edge case: flatten and take sqrt to estimate square matrix
                size = arr_sample.size
                rows = cols = int(np.sqrt(size))
            
            out = np.zeros((*obj_arr.shape, rows, cols), dtype=complex)
            for idx in np.ndindex(obj_arr.shape):
                if obj_arr[idx] is not None:
                    mat = np.asarray(obj_arr[idx])
                    if mat.ndim == 2:
                        r, c = mat.shape
                        out[tuple(idx) + (slice(None, min(r, rows)), slice(None, min(c, cols)))] = mat[:rows, :cols]
                    else:
                        # Try to reshape if it's flat
                        mat_flat = mat.reshape(-1)
                        out[tuple(idx)] = mat_flat[:rows*cols].reshape(rows, cols)
            return out

        # Apply stacking
        if "avgi" in data and isinstance(data["avgi"], np.ndarray) and data["avgi"].dtype == object:
            serial["avgi"] = stack_1d(data["avgi"])
        if "avgq" in data and isinstance(data["avgq"], np.ndarray) and data["avgq"].dtype == object:
            serial["avgq"] = stack_1d(data["avgq"])
        if "i0" in data and isinstance(data["i0"], np.ndarray) and data["i0"].dtype == object:
            serial["i0"] = stack_2d(data["i0"])
        if "q0" in data and isinstance(data["q0"], np.ndarray) and data["q0"].dtype == object:
            serial["q0"] = stack_2d(data["q0"])

        # Parity arrays
        for key in ("parity", "pe_plus", "pe_minus", "parity_plus", "parity_minus", "theta_opt"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                serial[key] = stack_1d(data[key])
            elif key in data:
                serial[key] = np.asarray(data[key])

        # Density matrices (2D)
        for key in ("rho", "rho_rotated"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                serial[key] = stack_3d(data[key])
            elif key in data:
                serial[key] = np.asarray(data[key])


        super().save_data(data=serial)

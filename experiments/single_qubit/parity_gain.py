'''
Check version history for ECD running via this file 
Eesh 06/13/2025


'''


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
from fitting_folder.wigner import WignerAnalysis


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

        # declare register pages (gain)
        self.man_rp = [self.ch_page(self.man_ch[qTest])] # get register page for qubit_ch
        self.r_gain = self.sreg(self.man_ch[qTest], "gain") # get gain register for qubit_ch
        self.r_gain2 = 4 # dummy register for gain  (since multiple qubit pulses)
        self.safe_regwi(self.man_rp[qTest], self.r_gain2, self.cfg.expt.start) # set dummygain register to start value

        # cavity pulse param
        print(cfg.device.manipulate.f_ge[0])
        self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.manipulate - 1], gen_ch = self.man_ch[qTest])
        if cfg.expt.displace[0]:
            self.displace_sigma = self.us2cycles(cfg.expt.displace[1], gen_ch=self.man_ch[qTest])
            self.add_gauss(ch=self.man_ch[qTest],
                           name="displace",
                           sigma=self.displace_sigma,
                           length=self.displace_sigma*4)

        self.parity_pulse = self.get_parity_str(man_mode_no=1,
                                                return_pulse=True,
                                                second_phase=self.cfg.expt.phase_second_pulse,
                                                fast=False)
        print('parity pulse:', self.parity_pulse)

        self.sync_all(200)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest=0

        # add cavity reset
        self.reset_and_sync()
        # active reset 
        if self.cfg.expt.active_reset:
            self.active_reset(man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        # pre pulse
        if cfg.expt.prepulse:
            print(cfg.expt.pre_sweep_pulse)
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='Prepulse')

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
            self.set_pulse_registers(
                    ch=self.man_ch[qTest], 
                    style="const", 
                    freq=self.f_cavity, 
                    phase=self.deg2reg(0),
                    gain=self.cfg.expt.start, # placeholder
                    length=self.us2cycles(self.cfg.expt.const_pulse[1]),
                    gen_ch = self.man_ch[qTest])

        # Update gain and pulse  
        self.mathi(self.man_rp[qTest], self.r_gain, self.r_gain2, "+", 0) # update gain register
        self.pulse(ch = self.man_ch[qTest])
        self.sync_all() # align channels

        # Parity Measurement
        self.custom_pulse(cfg, self.parity_pulse, prefix='Parity')
        self.measure_wrapper()


    def update(self):
        qTest=0
        print('Updating gain register', self.cfg.expt.step)
        self.mathi(self.man_rp[qTest], self.r_gain2, self.r_gain2, '+', self.cfg.expt.step) # update gain register



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
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment( num_qubits_sample)  
        data = {}

        if self.cfg.expt.single_shot:
            from MM_dual_rail_base import MM_dual_rail_base
            mm_dr_base = MM_dual_rail_base(self.cfg)
            data = mm_dr_base.run_single_shot(self, data, True)
            print('Single shot data:', data)

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        if 'pulse_correction' in self.cfg.expt:
            print("Pulse correction is applied")
            self.pulse_correction = self.cfg.expt.pulse_correction
        else:
            self.pulse_correction = False

        self.cfg.expt.phase_second_pulse = 180 # reset the phase of the second pulse        
        prog = ParityGainProgram(soccfg=self.soccfg, cfg=self.cfg)

        x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress,
                                        #   debug=debug,
                                            readouts_per_experiment=read_num)        

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase

        data['xpts'] = x_pts
        data['avgi'] = avgi
        data['avgq'] = avgq
        data['amps'] = amps
        data['phases'] = phases
        data['idata'], data['qdata'] = prog.collect_shots()

        if self.pulse_correction:
            self.cfg.expt.phase_second_pulse = 0 # reset the phase of the second pulse
            prog = ParityGainProgram(soccfg=self.soccfg, cfg=self.cfg)
            x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress,
                                            # debug=debug,
                                            readouts_per_experiment=read_num)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq)
            phases = np.angle(avgi+1j*avgq)
            data['xpts'] = np.append(data['xpts'], x_pts)
            data['avgi'] = np.append(data['avgi'], avgi)
            data['avgq'] = np.append(data['avgq'], avgq)
            data['amps'] = np.append(data['amps'], amps)
            data['phases'] = np.append(data['phases'], phases)

            _idata, _qdata = prog.collect_shots()
            data['idata'] = np.append(data['idata'], _idata)
            data['qdata'] = np.append(data['qdata'], _qdata)

        self.data=data
        return data


    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        # data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        # data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        # data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)

        if 'pulse_correction' in self.cfg.expt and self.cfg.expt.pulse_correction:
            data_minus = {}
            data_plus = {}
            # split the idata, qdata into two parts
            data_minus["idata"] = data['idata'][: len(data['idata'])//2]
            data_minus["qdata"] = data['qdata'][: len(data['qdata'])//2]
            data_minus["xpts"] = data['xpts'][: len(data['xpts'])//2]
            data_plus["idata"] = data['idata'][len(data['idata'])//2:]
            data_plus["qdata"] = data['qdata'][len(data['qdata'])//2:]
            data_plus["xpts"] = data['xpts'][len(data['xpts'])//2:]


            wigner_analysis_minus = WignerAnalysis(data_minus, config=self.cfg)
            wigner_analysis_plus = WignerAnalysis(data_plus, config=self.cfg)

            pe_plus = wigner_analysis_plus.bin_ss_data()
            pe_minus = wigner_analysis_minus.bin_ss_data()
            parity_plus = (1 - pe_plus) - pe_plus
            parity_minus = (1 - pe_minus) - pe_minus
            parity = (parity_minus - parity_plus) / 2
            gain_to_alpha, result, ydata = wigner_analysis_minus.get_gain_to_alpha(parity, initial_guess=[0.001])
            data['gain_to_alpha'] = gain_to_alpha
            data['parity'] = parity

        else:
            wigner_analysis = WignerAnalysis(data, config=self.cfg)
            pe = wigner_analysis.bin_ss_data()
            parity = (1 - pe) - pe
            gain_to_alpha, result, ydata = wigner_analysis.get_gain_to_alpha(parity, initial_guess=[0.001])
            data['gain_to_alpha'] = gain_to_alpha
            data['parity'] = parity


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

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from copy import deepcopy # single shot dictionary cfg copy

import experiments.fitting as fitter
from MM_base import MMAveragerProgram
'''
Sweeps frequency of prepulse 
'''

class ParityFreqProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        # cfg = AttrDict(self.cfg)
        # print(self.cfg)
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = 0
        # cavity pulse param
        self.f_cavity = self.freq2reg(cfg.expt.cav_freq, gen_ch = self.man_ch[qTest ])
        if cfg.expt.displace[0]:
            self.displace_sigma = self.us2cycles(cfg.expt.displace[1], gen_ch=self.man_ch[qTest ])
            self.add_gauss(ch=self.man_ch[qTest ], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)
        self.parity_pulse_for_custom_pulse = self.get_parity_str(man_mode_no = 1, return_pulse = True, second_phase = 0 )

        self.sync_all(200)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = 0

        self.reset_and_sync()

        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        #----------------------------------------------------
        #  Now Parity Freq   
        #  Setup cavity pulse form
        if self.cfg.expt.displace[0]:
            self.set_pulse_registers(
                    ch=self.man_ch[qTest],
                    style="arb",
                    freq=self.f_cavity,
                    phase=self.deg2reg(0), 
                    gain=self.cfg.expt.displace[2], # placeholder
                    waveform="displace")

        if self.cfg.expt.const_pulse[0]:
            self.set_pulse_registers(ch=self.man_ch[qTest], 
                                 style="const", 
                                 freq=self.f_cavity, 
                                 phase=self.deg2reg(0),
                                gain=self.cfg.expt.start, # placeholder
                                length=self.us2cycles(self.cfg.expt.const_pulse[1]))
        # Update gain and pulse  
        #self.mathi(self.man_rp, self.r_gain, self.r_gain2, "+", 0) # update gain register
        self.pulse(ch = self.man_ch[qTest])
        self.sync_all() # align channels

        # Parity Measurement
        self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='Parity')
        self.measure_wrapper()


    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_length_adc
        shots_q0 = self.dq_buf[0] / self.readout_length_adc
        return shots_i0, shots_q0
        # return shots_i0[:5000], shots_q0[:5000]


class ParityFreqExperiment(Experiment):
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

        if not self.cfg.expt.single_shot:
            #t1 = ParityFreqProgram(soccfg=self.soccfg, cfg=self.cfg)

            x_pts = np.arange(self.cfg.expt.start,self.cfg.expt.stop, self.cfg.expt.step)

            data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

            for freq in tqdm(x_pts, disable=not progress):
                self.cfg.expt.cav_freq = float(freq)
                program = ParityFreqProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = program
                avgi, avgq = program.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, 
                                            #  debug=debug
                                             )
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                data["xpts"].append(freq)
                data["avgi"].append(avgi)
                data["avgq"].append(avgq)
                data["amps"].append(amp)
                data["phases"].append(phase)

        else:
            from experiments.single_qubit.single_shot import hist, HistogramProgram

            # ----------------- Single Shot Calibration -----------------
            data=dict()
            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps

            # Ground state shots
            # cfg.expt.reps = 10000
            sscfg.expt.qubit = 0
            sscfg.expt.rounds = 1
            sscfg.expt.pulse_e = False
            sscfg.expt.pulse_f = False
            # print(sscfg)

            data['Ig'] = []
            data['Qg'] = []
            data['Ie'] = []
            data['Qe'] = []
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            data['Ig'], data['Qg'] = histpro.collect_shots()

            # Excited state shots
            sscfg.expt.pulse_e = True 
            sscfg.expt.pulse_f = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            data['Ie'], data['Qe'] = histpro.collect_shots()
            # print(data)

            fids, thresholds, angle, confusion_matrix = hist(data=data, plot=False, verbose=False, span=self.cfg.expt.span)
            data['fids'] = fids
            data['angle'] = angle
            data['thresholds'] = thresholds
            data['confusion_matrix'] = confusion_matrix

            print(f'ge fidelity (%): {100*fids[0]}')
            print(f'rotation angle (deg): {angle}')
            print(f'threshold ge: {thresholds[0]}')
            print('Confusion matrix [Pgg, Pge, Peg, Pee]: ',confusion_matrix)

            # ------------------- Experiment -------------------

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
                x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress,
                                                #   debug=debug
                                                  )
                II, QQ = prog.collect_shots()
                # save data for each round
                data['I_data'].append(II)
                data['Q_data'].append(QQ)
                data['avgi'].append(avgi) # for debugging
                data['avgq'].append(avgq)
                data['xpts'] = x_pts # same for all rounds

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


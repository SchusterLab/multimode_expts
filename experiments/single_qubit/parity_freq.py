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
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
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

        # declare register pages (gain)
        self.man_rp = self.ch_page(self.man_ch) # get register page for qubit_ch
        self.r_freq = self.sreg(self.man_ch, "freq") # get gain register for qubit_ch
        self.r_freq2 = 4 # dummy register for gain  (since multiple qubit pulses)
        self.safe_regwi(self.man_rp, self.r_freq2, self.freq2reg(self.cfg.expt.start, gen_ch = self.man_ch)) # set dummygain register to start value

        # declare qubit dacs
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_ch
        if self.res_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == 'mux4':
            assert self.res_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # cavity pulse param
        self.f_cavity = self.freq2reg(cfg.expt.cav_freq, gen_ch = self.man_ch)
        if cfg.expt.displace[0]:
            self.displace_sigma = self.us2cycles(cfg.expt.displace[1], gen_ch=self.man_ch)
            self.add_gauss(ch=self.man_ch, name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)

        #f0g1 sideband
        self.f0g1 = self.freq2reg(cfg.device.QM.pulses.f0g1.freq[cfg.expt.f0g1_cavity-1], gen_ch=self.qubit_ch)
        self.f0g1_length = self.us2cycles(cfg.device.QM.pulses.f0g1.length[cfg.expt.f0g1_cavity-1], gen_ch=self.qubit_ch)
        self.pif0g1_gain = cfg.device.QM.pulses.f0g1.gain[cfg.expt.f0g1_cavity-1]
        
        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma, gen_ch=self.qubit_ch)
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain

        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
            self.add_gauss(ch=self.qubit_ch, name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)
            self.add_gauss(ch=self.qubit_ch, name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
            self.add_gauss(ch=self.f0g1_ch, name="f0g1",
                       sigma=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma), length=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma)*4)
            #self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        else:
            self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=cfg.expt.start, length=self.pi_sigma)


        # if self.res_ch_type == 'mux4':
        #     self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        

        self.sync_all(200)

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs 

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[qTest]s[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[qTest]s[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)


        #initialize the phase to be 0
        self.f_q = self.f_ge
        self.f_cav = self.f_cavity

        self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_ch)
        self.set_pulse_registers(ch=self.man_ch, freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_ch)
        self.set_pulse_registers(ch=self.storage_ch, freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.storage_ch)
        self.set_pulse_registers(ch=self.flux_low_ch, freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_low_ch)
        self.set_pulse_registers(ch=self.flux_high_ch, freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_high_ch)
        self.set_pulse_registers(ch=self.f0g1_ch, freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.f0g1_ch)

        self.sync_all(10)

    def body(self):
        cfg=AttrDict(self.cfg)
        # if cfg.expt.prepulse:
        #     for ii in range(len(cfg.expt.pre_sweep_pulse[0])):
        #         # translate ch id to ch
        #         if cfg.expt.pre_sweep_pulse[4][ii] == 1:
        #             self.tempch = self.flux_low_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 2:
        #             self.tempch = self.qubit_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 3:
        #             self.tempch = self.flux_high_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 4:
        #             self.tempch = self.storage_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 5:
        #             self.tempch = self.f0g1_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 6:
        #             self.tempch = self.man_ch
        #         # print(self.tempch)
        #         # determine the pulse shape
        #         if cfg.expt.pre_sweep_pulse[5][ii] == "gaussian":
        #             # print('gaussian')
        #             self.pisigma_resolved = self.us2cycles(
        #                 cfg.expt.pre_sweep_pulse[6][ii], gen_ch=self.tempch)
        #             self.add_gauss(ch=self.tempch, name="temp_gaussian",
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
        #             self.setup_and_pulse(ch=self.tempch, style="arb", 
        #                              freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch), 
        #                              phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
        #                              gain=cfg.expt.pre_sweep_pulse[1][ii], 
        #                              waveform="temp_gaussian")
        #         elif cfg.expt.pre_sweep_pulse[5][ii] == "flat_top":
        #             # print('flat_top')
        #             self.pisigma_resolved = self.us2cycles(
        #                 cfg.expt.pre_sweep_pulse[6][ii], gen_ch=self.tempch)
        #             self.add_gauss(ch=self.tempch, name="temp_gaussian",
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
        #             self.setup_and_pulse(ch=self.tempch, style="flat_top", 
        #                              freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch), 
        #                              phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
        #                              gain=cfg.expt.pre_sweep_pulse[1][ii], 
        #                              length=self.us2cycles(cfg.expt.pre_sweep_pulse[2][ii], 
        #                                                    gen_ch=self.tempch),
        #                             waveform="temp_gaussian")
        #         else:
        #             self.setup_and_pulse(ch=self.tempch, style="const", 
        #                              freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch), 
        #                              phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
        #                              gain=cfg.expt.pre_sweep_pulse[1][ii], 
        #                              length=self.us2cycles(cfg.expt.pre_sweep_pulse[2][ii], 
        #                                                    gen_ch=self.tempch))
        #         self.sync_all()

        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        if cfg.expt.f0g1_cavity > 0:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all() # align channels
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pief_gain, waveform="pief_qubit")
            self.sync_all() # align channels
            self.setup_and_pulse(
                    ch=self.f0g1_ch,
                    style="flat_top",
                    freq=self.f0g1,
                    length=self.f0g1_length,
                    phase=0,
                    gain=self.pif0g1_gain, 
                    waveform="f0g1")
            self.sync_all() # align channels
        if self.cfg.expt.prep_e:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all() # align channels

        #----------------------------------------------------
        #  Now Parity Freq   
        #  Setup cavity pulse form
        if self.cfg.expt.displace[0]:
            self.set_pulse_registers(
                    ch=self.man_ch,
                    style="arb",
                    freq=self.f_cavity,
                    phase=self.deg2reg(0), 
                    gain=self.cfg.expt.displace[2], # placeholder
                    waveform="displace")
            
        
        if self.cfg.expt.const_pulse[0]:
            self.set_pulse_registers(ch=self.man_ch, 
                                 style="const", 
                                 freq=self.f_cavity, 
                                 phase=self.deg2reg(0),
                                gain=self.cfg.expt.start, # placeholder
                                length=self.us2cycles(self.cfg.expt.const_pulse[1]))
        # Update gain and pulse  
        #self.mathi(self.man_rp, self.r_gain, self.r_gain2, "+", 0) # update gain register
        self.pulse(ch = self.man_ch)
        self.sync_all() # align channels

        # Parity Measurement
        self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=self.deg2reg(0), gain=cfg.device.qubit.pulses.hpi_ge.gain, waveform="hpi_qubit")
        self.sync_all() # align channels
        chi = 0.289 # MHz
        # self.sync_all(self.us2cycles(np.abs(1 / chi / 2))) # wait for pi/chi (noe chi in config is in MHz)
        # self.sync_all(self.us2cycles(self.cfg.device.manipulate.revival_time)) # align channels and wait 50ns
        self.sync_all(self.us2cycles(np.abs(self.cfg.device.manipulate.revival_time[self.cfg.expt.manipulate-1]))) # wait for parity revival time
        self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=self.deg2reg(180), gain=cfg.device.qubit.pulses.hpi_ge.gain, waveform="hpi_qubit")
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    # def update(self):
    #     self.mathi(self.man_rp, self.r_freq, self.r_freq, '+', self.cfg.expt.step) # update gain register

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
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})   
                                     
        if not self.cfg.expt.single_shot:
            #t1 = ParityFreqProgram(soccfg=self.soccfg, cfg=self.cfg)
            
            x_pts = np.arange(self.cfg.expt.start,self.cfg.expt.stop, self.cfg.expt.step)

            data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

            for freq in tqdm(x_pts, disable=not progress):
                self.cfg.expt.cav_freq = float(freq)
                program = ParityFreqProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = program
                avgi, avgq = program.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)        
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
                x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)
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
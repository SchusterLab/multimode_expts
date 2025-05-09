import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm
from MM_base import * 

def hist(data, plot=True, span=None, verbose=True):
    """
    span: histogram limit is the mean +/- span
    """
    Ig = data['Ig']
    Qg = data['Qg']
    Ie = data['Ie']
    Qe = data['Qe']
    plot_f = False 
    if 'If' in data.keys():
        plot_f = True
        If = data['If']
        Qf = data['Qf']

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f: xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print('Unrotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
        fig.tight_layout()
        
        axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.')
        axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.')
        
        if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.')
        axs[0,0].scatter(xg, yg, color='k', marker='o')
        axs[0,0].scatter(xe, ye, color='k', marker='o')
        if plot_f: axs[0,0].scatter(xf, yf, color='k', marker='o')

        axs[0,0].set_xlabel('I [ADC levels]')
        axs[0,0].set_ylabel('Q [ADC levels]')
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_title('Unrotated')
        axs[0,0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    if plot_f: theta = -np.arctan2((yf-yg),(xf-xg))

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print('Rotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [xg-span, xg+span]
    ylims = [yg-span, yg+span]

    if plot:
        axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.')
        axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.')
        if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.')
        axs[0,1].scatter(xg, yg, color='k', marker='o')
        axs[0,1].scatter(xe, ye, color='k', marker='o')    
        if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

        axs[0,1].set_xlabel('I [ADC levels]')
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Rotated')
        axs[0,1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        if plot_f:
            nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5)
        axs[1,0].set_ylabel('Counts')
        axs[1,0].set_xlabel('I [ADC levels]')       
        axs[1,0].legend(loc='upper right')

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])

    confusion_matrix = [np.cumsum(ng)[tind]/ng.sum(),
                        1-np.cumsum(ng)[tind]/ng.sum(),
                        np.cumsum(ne)[tind]/ne.sum(),
                        1-np.cumsum(ne)[tind]/ne.sum()]   # Pgg (prepare g measured g), Pge (prepare g measured e), Peg, Pee
    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
        
    if plot: 
        axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
        axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1,1].set_title('Cumulative Counts')
        axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
        axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
        axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1,1].legend()
        axs[1,1].set_xlabel('I [ADC levels]')
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)        
        plt.show()

    return fids, thresholds, theta*180/np.pi, confusion_matrix # fids: ge, gf, ef

# ====================================================== #

class HistogramProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        #qTest = 0
        self.qubits = self.cfg.expt.qubits
        #self.drive_freq = self.cfg.expt.freq

        qTest = self.qubits[0]
        
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        self.f0g1_chs = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_types = cfg.hw.soc.dacs.sideband.type

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
        
        # self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch[qTest])
        # self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch[qTest])
        # self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch[qTest])
        # self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch[qTest])
        # self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch[qTest])
        # self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
        # get register page for qubit_chs
        # print(self.qubit_chs)
        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]
        self.f_ge_reg_ss = [self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_ef_reg_ss = [self.freq2reg(
            cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])]

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(
            self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(
            self.cfg.device.readout.readout_length, self.adc_chs)]

        
        gen_chs = []

        # declare res dacs
        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = None
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest],
                         mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest],
                             freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])
        

        # declare qubit dacs
        
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(
                    ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # define pi_test_ramp as the pulse that we are calibrating with ramsey, update in outer loop over averager program
        # self.pi_test_ramp = self.us2cycles(
        #     cfg.device.qubit.ramp_sigma[qTest], gen_ch=self.qubit_chs[qTest])
        # self.f_pi_test_reg = self.freq2reg(self.drive_freq)  # freq we are trying to calibrate
        # self.gain_pi_test = self.cfg.expt.gain  # gain we are trying to play

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge_ss = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ge value
        self.pisigma_ef_ss = self.us2cycles(
            cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ef value
        self.f_ge_init_reg_ss = self.f_ge_reg_ss[qTest]
        self.f_ef_init_reg_ss = self.f_ef_reg_ss[qTest]
        self.gain_ge_init_ss = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_ef_init_ss = self.cfg.device.qubit.pulses.pi_ef.gain[qTest]

        # add qubit pulses to respective channels
        # self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test_ramp", sigma=self.pi_test_ramp,
        #                length=self.pi_test_ramp*2*cfg.device.qubit.ramp_sigma_num[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge_ss",
                       sigma=self.pisigma_ge_ss, length=self.pisigma_ge_ss*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef_ss",
                       sigma=self.pisigma_ef_ss, length=self.pisigma_ef_ss*4)
        # self.add_gauss(ch=self.f0g1_chs[qTest], name="pi_test",
        #                sigma=self.us2cycles(self.cfg.expt.ramp_sigma), length=self.us2cycles(self.cfg.expt.ramp_sigma)*4)

        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        
        #  ## ALL ACTIVE RESET REQUIREMENTS
        # # read val definition
        # self.r_read_q = 3   # ge active reset register
        # self.r_read_q_ef = 4   # ef active reset register
        # self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        # self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # # threshold definition
        # self.r_thresh_q = 5  # Define a location to store the threshold info

        # # # multiplication bc the readout is summed, so need common thing to compare to
        # self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))

        # # Define a location to store a counter for how frequently the condj is triggered
        # self.r_counter = 7
        # self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0
        # self.wait_all(self.us2cycles(0.2))
        # self.sync_all(200)
    
    def active_reset1(self, man_reset=False, storage_reset=False):

        # Reset ge level
        # ======================================================
        print("this codes active reset")
        qTest = 0
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.res_chs[qTest],
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                     t='auto', wait=True, syncdelay=self.us2cycles(2.0))#self.cfg["relax_delay"])#self.cfg["relax_delay"])  # self.us2cycles(1))
        
        self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced

        self.read(0, 0, "lower", self.r_read_q)  # read data from I buffer, QA, and store
        self.wait_all()  # to allow the read to be complete might be reduced
        self.sync_all()

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   "LABEL_1")  # compare the value recorded above to the value stored in threshold.

        #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
        self.setup_and_pulse(ch=self.qubit_chs[0], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge_ss")
        self.pulse(ch=self.qubit_chs[qTest])
        self.label("LABEL_1")  # location to be jumped to
        self.wait_all() 
        self.sync_all()
        # ======================================================

        # Reset ef level
        # ======================================================
        self.setup_and_pulse(ch=self.qubit_chs[0], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge_ss")
        self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05))
        self.measure(pulse_ch=self.res_chs[qTest],
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                     t='auto', wait=True, syncdelay=self.us2cycles(2.0))  # self.us2cycles(1))
        
        self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced
        
        self.read(0, 0, "lower", self.r_read_q_ef)  # read data from I buffer, QA, and store
        self.wait_all()  # to allow the read to be complete might be reduced
        self.sync_all()

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q_ef, "<", self.r_thresh_q,
                   "LABEL_2")  # compare the value recorded above to the value stored in threshold.

        #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
        self.setup_and_pulse(ch=self.qubit_chs[0], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge_ss")
        self.pulse(ch=self.qubit_chs[qTest])
        self.label("LABEL_2")  # location to be jumped to
        self.wait_all() 
        self.sync_all()

        # ======================================================

        # post selection

        # ======================================================
        self.measure(pulse_ch=self.res_chs[qTest],
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                     t='auto', wait=True, syncdelay=self.us2cycles(2.0))  # self.us2cycles(1))
        self.wait_all() 
        self.sync_all()

        # ======================================================


        self.sync_all(self.us2cycles(0.05))  # not sure if this is needed
        
    
    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = 0

        # initializations as necessary
        self.reset_and_sync()
        # self.sync_all()

        #do the active reset
        if cfg.expt.active_reset:
            self.active_reset(man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)


        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            if cfg.device.qubit.pulses.pi_ge.type[qTest] == 'gauss':
                self.setup_and_pulse(ch=self.qubit_chs[0], style="arb", freq=self.f_ge_init_reg_ss, phase=0, gain=self.gain_ge_init_ss, waveform="pi_qubit_ge_ss")
            else: # const pulse
                self.setup_and_pulse(ch=self.qubit_chs[0], style="const", freq=self.f_ge_init_reg_ss, phase=0, gain=self.gain_ge_init_ss, length=self.pisigma_ge_ss)
        self.sync_all()
        self.wait_all(self.us2cycles(0.01))

        if self.cfg.expt.pulse_f:
            if cfg.device.qubit.pulses.pi_ef.type[qTest] == 'gauss':
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ef_init_reg_ss, phase=0, gain=self.gain_ef_init_ss, waveform="pi_qubit_ef_ss")
            else: # const pulse
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=self.f_ef_init_reg_ss, phase=0, gain=self.gain_ef_init_ss, length=self.pisigma_ef_ss)
        self.sync_all()
        self.wait_all(self.us2cycles(0.01))


        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

    # def collect_shots(self):
    #     # collect shots for the relevant adc and I and Q channels
    #     cfg=AttrDict(self.cfg)
    #     # print(np.average(self.di_buf[0]))
    #     shots_i0 = self.di_buf[0] / self.readout_length_adc
    #     shots_q0 = self.dq_buf[0] / self.readout_length_adc
    #     return shots_i0, shots_q0
    #     # return shots_i0[:5000], shots_q0[:5000]


class HistogramExperiment(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubits[0]
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update(
                                    {key3: [value3]*num_qubits_sample})
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
        data = dict()

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        # Ground state shots
        cfg = self.cfg #AttrDict(deepcopy(self.cfg))
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False
        histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
        #i = histpro.acquire_decimated(self.im[self.cfg.aliases.soc])
        #print(i.to_list())
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, readouts_per_experiment=read_num)
        data['Ig'], data['Qg'] = histpro.collect_shots()

        # Excited state shots
        if 'check_e' not in self.cfg.expt:
            self.check_e = True
        else: self.check_e = self.cfg.expt.check_e
        if self.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, readouts_per_experiment=read_num)
            data['Ie'], data['Qe'] = histpro.collect_shots()
            # print(histpro)

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, readouts_per_experiment=read_num)
            data['If'], data['Qf'] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=True, **kwargs):
        if data is None:
            data=self.data
        
        fids, thresholds, angle, confusion_matrix = hist(data=data, plot=False, span=span, verbose=verbose)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        data['confusion_matrix'] = confusion_matrix
        
        return data

    def display(self, data=None, span=None, verbose=True, plot_e=True, plot_f=False, **kwargs):
        if data is None:
            data=self.data 
        
        fids, thresholds, angle, confusion_matrix = hist(data=data, plot=True, verbose=verbose, span=span)
            
        print(f'ge fidelity (%): {100*fids[0]}')
        if 'expt' not in self.cfg: 
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f'gf fidelity (%): {100*fids[1]}')
            print(f'ef fidelity (%): {100*fids[2]}')
        print(f'rotation angle (deg): {angle}')
        # print(f'set angle to (deg): {-angle}')
        print(f'threshold ge: {thresholds[0]}')
        if self.cfg.expt.check_f:
            print(f'threshold gf: {thresholds[1]}')
            print(f'threshold ef: {thresholds[2]}')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
# ------------------------ Old file --------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


# def hist(data, plot=True, span=None, verbose=True):
#     """
#     span: histogram limit is the mean +/- span
#     """
#     Ig = data['Ig']
#     Qg = data['Qg']
#     Ie = data['Ie']
#     Qe = data['Qe']
#     plot_f = False 
#     if 'If' in data.keys():
#         plot_f = True
#         If = data['If']
#         Qf = data['Qf']

#     numbins = 200

#     xg, yg = np.median(Ig), np.median(Qg)
#     xe, ye = np.median(Ie), np.median(Qe)
#     if plot_f: xf, yf = np.median(If), np.median(Qf)

#     if verbose:
#         print('Unrotated:')
#         print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
#         print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
#         if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

#     if plot:
#         fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
#         fig.tight_layout()
        
#         axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.')
#         axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.')
        
#         if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.')
#         axs[0,0].scatter(xg, yg, color='k', marker='o')
#         axs[0,0].scatter(xe, ye, color='k', marker='o')
#         if plot_f: axs[0,0].scatter(xf, yf, color='k', marker='o')

#         axs[0,0].set_xlabel('I [ADC levels]')
#         axs[0,0].set_ylabel('Q [ADC levels]')
#         axs[0,0].legend(loc='upper right')
#         axs[0,0].set_title('Unrotated')
#         axs[0,0].axis('equal')

#     """Compute the rotation angle"""
#     theta = -np.arctan2((ye-yg),(xe-xg))
#     if plot_f: theta = -np.arctan2((yf-yg),(xf-xg))

#     """Rotate the IQ data"""
#     Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
#     Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

#     Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
#     Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

#     if plot_f:
#         If_new = If*np.cos(theta) - Qf*np.sin(theta)
#         Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

#     """New means of each blob"""
#     xg, yg = np.median(Ig_new), np.median(Qg_new)
#     xe, ye = np.median(Ie_new), np.median(Qe_new)
#     if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
#     if verbose:
#         print('Rotated:')
#         print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
#         print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
#         if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


#     if span is None:
#         span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
#     xlims = [xg-span, xg+span]
#     ylims = [yg-span, yg+span]

#     if plot:
#         axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.')
#         axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.')
#         if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.')
#         axs[0,1].scatter(xg, yg, color='k', marker='o')
#         axs[0,1].scatter(xe, ye, color='k', marker='o')    
#         if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

#         axs[0,1].set_xlabel('I [ADC levels]')
#         axs[0,1].legend(loc='upper right')
#         axs[0,1].set_title('Rotated')
#         axs[0,1].axis('equal')

#         """X and Y ranges for histogram"""

#         ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
#         ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
#         if plot_f:
#             nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5)
#         axs[1,0].set_ylabel('Counts')
#         axs[1,0].set_xlabel('I [ADC levels]')       
#         axs[1,0].legend(loc='upper right')

#     else:        
#         ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
#         ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
#         if plot_f:
#             nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

#     """Compute the fidelity using overlap of the histograms"""
#     fids = []
#     thresholds = []
#     contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
#     tind=contrast.argmax()
#     thresholds.append(binsg[tind])
#     fids.append(contrast[tind])

#     confusion_matrix = [np.cumsum(ng)[tind]/ng.sum(),
#                         1-np.cumsum(ng)[tind]/ng.sum(),
#                         np.cumsum(ne)[tind]/ne.sum(),
#                         1-np.cumsum(ne)[tind]/ne.sum()]   # Pgg (prepare g measured g), Pge (prepare g measured e), Peg, Pee
#     if plot_f:
#         contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
#         tind=contrast.argmax()
#         thresholds.append(binsg[tind])
#         fids.append(contrast[tind])

#         contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
#         tind=contrast.argmax()
#         thresholds.append(binsg[tind])
#         fids.append(contrast[tind])
        
#     if plot: 
#         axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
#         axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
#         if plot_f:
#             axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
#             axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

#         axs[1,1].set_title('Cumulative Counts')
#         axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
#         axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
#         axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
#         if plot_f:
#             axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
#             axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
#             axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
#         axs[1,1].legend()
#         axs[1,1].set_xlabel('I [ADC levels]')
        
#         plt.subplots_adjust(hspace=0.25, wspace=0.15)        
#         plt.show()

#     return fids, thresholds, theta*180/np.pi, confusion_matrix # fids: ge, gf, ef
def hist(data, plot=True, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4.3):
    """
    span: histogram limit is the mean +/- span
    """
    # if active_reset:
    #     Ig = data['Ig'][readout_per_round-1::readout_per_round]
    #     Qg = data['Qg'][readout_per_round-1::readout_per_round]
    #     Ie = data['Ie'][readout_per_round-1::readout_per_round]
    #     Qe = data['Qe'][readout_per_round-1::readout_per_round]
    #     plot_f = False 
    #     if 'If' in data.keys():
    #         plot_f = True
    #         If = data['If'][readout_per_round-1::readout_per_round]
    #         Qf = data['Qf'][readout_per_round-1::readout_per_round]

    if active_reset:
        Ig, Qg = MM_base.filter_data_IQ(data['Ig'], data['Qg'], threshold, readout_per_experiment=readout_per_round)
        # Qg = filter_data(data['Qg'], threshold, readout_per_experiment=readout_per_round)
        Ie, Qe = MM_base.filter_data_IQ(data['Ie'], data['Qe'], threshold, readout_per_experiment=readout_per_round)
        # Qe = filter_data(data['Qe'], threshold, readout_per_experiment=readout_per_round)
        print(len(Ig))
        print(len(Ie))
        plot_f = False 
        if 'If' in data.keys():
            plot_f = True
            If, Qf = MM_base.filter_data_IQ(data['If'], data['Qf'], threshold, readout_per_experiment=readout_per_round)
            # Qf = filter_data(data['Qf'], threshold, readout_per_experiment=readout_per_round)
            print(len(If))
    else:

        Ig = data['Ig']
        Qg = data['Qg']
        Ie = data['Ie']
        Qe = data['Qe']
        plot_f = False 
        if 'If' in data.keys():
            plot_f = True
            If = data['If']
            Qf = data['Qf']

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f: xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print('Unrotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
        fig.tight_layout()
        
        axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.', s=1)
        axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.', s=1)
        
        if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.', s=1)
        axs[0,0].scatter(xg, yg, color='k', marker='o')
        axs[0,0].scatter(xe, ye, color='k', marker='o')
        if plot_f: axs[0,0].scatter(xf, yf, color='k', marker='o')

        axs[0,0].set_xlabel('I [ADC levels]')
        axs[0,0].set_ylabel('Q [ADC levels]')
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_title('Unrotated')
        axs[0,0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    if plot_f: theta = -np.arctan2((ye-yf),(xe-xf))

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print('Rotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [xg-span, xg+span]
    ylims = [yg-span, yg+span]

    if plot:
        axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.', s=1)
        axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.', s=1)
        if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.', s=1)
        axs[0,1].scatter(xg, yg, color='k', marker='o')
        axs[0,1].scatter(xe, ye, color='k', marker='o')    
        if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

        axs[0,1].set_xlabel('I [ADC levels]')
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Rotated')
        axs[0,1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5, density=True)
        ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5, density=True)
        if plot_f:
            nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5, density=True)
        axs[1,0].set_ylabel('Counts')
        axs[1,0].set_xlabel('I [ADC levels]')       
        axs[1,0].legend(loc='upper right')

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])

    confusion_matrix = [np.cumsum(ng)[tind]/ng.sum(),
                        1-np.cumsum(ng)[tind]/ng.sum(),
                        np.cumsum(ne)[tind]/ne.sum(),
                        1-np.cumsum(ne)[tind]/ne.sum()]   # Pgg (prepare g measured g), Pge (prepare g measured e), Peg, Pee
    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
        
    if plot: 
        axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
        axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1,1].set_title('Cumulative Counts')
        axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
        axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
        axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1,1].legend()
        axs[1,1].set_xlabel('I [ADC levels]')
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)        
        plt.show()

    return fids, thresholds, theta*180/np.pi, confusion_matrix # fids: ge, gf, ef

# ====================================================== #

class HistogramProgram_old(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        # self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        # self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        if self.cfg.expt.pulse_f: 
            self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare dacs
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

        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        if self.cfg.expt.pulse_f:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain
        
        # add qubit and readout pulses to respective channels
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f and cfg.device.qubit.pulses.pi_ge.type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        if self.cfg.expt.pulse_f and cfg.device.qubit.pulses.pi_ef.type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase= self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        ## ALL ACTIVE RESET REQUIREMENTS
        # read val definition
        self.r_read_q = 3   # ge active reset register
        self.r_read_q_ef = 4   # ef active reset register
        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # threshold definition
        self.r_thresh_q = 5  # Define a location to store the threshold info

        # # multiplication bc the readout is summed, so need common thing to compare to
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold * self.readout_length_adc))

        # Define a location to store a counter for how frequently the condj is triggered
        self.r_counter = 7
        self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0

        self.sync_all(200)
    
    def active_reset(self):

        # Reset ge level
        # ======================================================
        qTest = 0
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.res_ch, 
                    adcs=[self.adc_ch],
                    adc_trig_offset=cfg.device.readout.trig_offset,
                     t='auto', wait=True, syncdelay=self.us2cycles(2.5))#self.cfg["relax_delay"])  # self.us2cycles(1))
        
        self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced

        self.read(0, 0, "lower", self.r_read_q)  # read data from I buffer, QA, and store
        self.wait_all()  # to allow the read to be complete might be reduced
        self.sync_all()

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   "LABEL_1")  # compare the value recorded above to the value stored in threshold.

        #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
        self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ge, style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_gain, waveform='pi_qubit')
        self.pulse(ch=self.qubit_ch)
        self.label("LABEL_1")  # location to be jumped to
        self.wait_all() 
        self.sync_all()
        # ======================================================

        # Reset ef level
        # ======================================================
        self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ef, style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_ef_gain, waveform='pi_ef_qubit')
        self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05))
        self.measure(pulse_ch=self.res_ch, 
                    adcs=[self.adc_ch],
                    adc_trig_offset=cfg.device.readout.trig_offset,
                     t='auto', wait=True, syncdelay=self.us2cycles(2.5))  # self.us2cycles(1))
        
        self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced
        
        self.read(0, 0, "lower", self.r_read_q_ef)  # read data from I buffer, QA, and store
        self.wait_all()  # to allow the read to be complete might be reduced
        self.sync_all()

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q_ef, "<", self.r_thresh_q,
                   "LABEL_2")  # compare the value recorded above to the value stored in threshold.

        #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
        self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ge, style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_gain, waveform='pi_qubit')
        self.pulse(ch=self.qubit_ch)
        self.label("LABEL_2")  # location to be jumped to
        self.wait_all() 
        self.sync_all()

        # ======================================================

        # post selection

        # ======================================================
        self.measure(pulse_ch=self.res_ch, 
                    adcs=[self.adc_ch],
                    adc_trig_offset=cfg.device.readout.trig_offset,
                     t='auto', wait=True, syncdelay=self.us2cycles(2.5))  # self.us2cycles(1))
        self.wait_all() 
        self.sync_all()

        # ======================================================


        self.sync_all(self.us2cycles(0.05))  # not sure if this is needed
    
    def body(self):
        cfg=AttrDict(self.cfg)
        self.sync_all()

        if self.cfg.expt.active_reset:
            print('hello')
            self.active_reset() 

        # pre_pulse on manipulate:
        # if self.cfg.expt.pulse_manipulate:
        #     self.setup_and_pulse(ch=self.man_ch, style="const", 
        #                          freq=self.freq2reg(self.cfg.expt.cavity_freq, gen_ch=self.man_ch), phase=0, 
        #                          gain=self.cfg.expt.cavity_gain, 
        #                          length=self.us2cycles(self.cfg.expt.cavity_length, gen_ch=self.man_ch))
        # self.sync_all()


        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            if cfg.device.qubit.pulses.pi_ge.type == 'gauss':
                self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            else: # const pulse
                self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=self.pi_gain, length=self.pi_sigma)
        self.sync_all()

        if self.cfg.expt.pulse_f:
            if cfg.device.qubit.pulses.pi_ef.type == 'gauss':
                self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pi_ef_gain, waveform="pi_ef_qubit")
            else: # const pulse
                self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_ef, phase=0, gain=self.pi_ef_gain, length=self.pi_ef_sigma)
        self.sync_all()

        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay)
             )

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_length_adc
        shots_q0 = self.dq_buf[0] / self.readout_length_adc
        return shots_i0, shots_q0
        # return shots_i0[:5000], shots_q0[:5000]


class HistogramExperiment_old(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None):
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

        data=dict()

        # Ground state shots
        cfg = AttrDict(deepcopy(self.cfg))
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False
        histpro = HistogramProgram_old(soccfg=self.soccfg, cfg=cfg)
        #i = histpro.acquire_decimated(self.im[self.cfg.aliases.soc])
        #print(i.to_list())
        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, readouts_per_experiment=read_num)
        data['Ig'], data['Qg'] = histpro.collect_shots()

        # Excited state shots
        if 'check_e' not in self.cfg.expt:
            self.check_e = True
        else: self.check_e = self.cfg.expt.check_e
        if self.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = False
            histpro = HistogramProgram_old(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, readouts_per_experiment=read_num)
            data['Ie'], data['Qe'] = histpro.collect_shots()

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = True
            histpro = HistogramProgram_old(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, readouts_per_experiment=read_num)
            data['If'], data['Qf'] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=True, **kwargs):
        if data is None:
            data=self.data
        
        fids, thresholds, angle, confusion_matrix = hist(data=data, plot=False, span=span, verbose=verbose)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        data['confusion_matrix'] = confusion_matrix
        
        return data

    def display(self, data=None, span=None, verbose=True, plot_e=True, plot_f=False, **kwargs):
        if data is None:
            data=self.data 
        
        fids, thresholds, angle, confusion_matrix = hist(data=data, plot=True, verbose=verbose, span=span)
            
        print(f'ge fidelity (%): {100*fids[0]}')
        if 'expt' not in self.cfg: 
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f'gf fidelity (%): {100*fids[1]}')
            print(f'ef fidelity (%): {100*fids[2]}')
        print(f'rotation angle (deg): {angle}')
        # print(f'set angle to (deg): {-angle}')
        print(f'threshold ge: {thresholds[0]}')
        if self.cfg.expt.check_f:
            print(f'threshold gf: {thresholds[1]}')
            print(f'threshold ef: {thresholds[2]}')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

# ====================================================== #
# cfg = AttrDict(self.cfg)
        # self.cfg.update(cfg.expt)
        
        # self.adc_ch = cfg.hw.soc.adcs.readout.ch
        # self.res_ch = cfg.hw.soc.dacs.readout.ch
        # self.res_ch_type = cfg.hw.soc.dacs.readout.type
        # # self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        # # self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        # self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        # self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        
        # self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        # self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        # self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        # self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        # self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        # self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # # declare dacs
        # mask = None
        # mixer_freq = 0 # MHz
        # mux_freqs = None # MHz
        # mux_gains = None
        # ro_ch = self.adc_ch
        # if self.res_ch_type == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        # elif self.res_ch_type == 'mux4':
        #     assert self.res_ch == 6
        #     mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        #     mux_freqs = [0]*4
        #     mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
        #     mux_gains = [0]*4
        #     mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
        # self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # mixer_freq = 0
        # if self.qubit_ch_type == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        # self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # # declare adcs
        # self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        # self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain

        # self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        # self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain
        
        # # add qubit and readout pulses to respective channels
        # self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        # self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        # # add readout pulses to respective channels
        # self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        # ## ALL ACTIVE RESET REQUIREMENTS
        # # read val definition
        # self.r_read_q = 3   # ge active reset register
        # self.r_read_q_ef = 4   # ef active reset register
        # self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        # self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # # threshold definition
        # self.r_thresh_q = 5  # Define a location to store the threshold info

        # # # multiplication bc the readout is summed, so need common thing to compare to
        # self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold * self.readout_length_adc))

        # # Define a location to store a counter for how frequently the condj is triggered
        # self.r_counter = 7
        # self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0
        
        
        # self.sync_all(200)


    # def active_reset(self):

    #     # Reset ge level
    #     # ======================================================
    #     qTest = 0
    #     cfg=AttrDict(self.cfg)
    #     self.measure(pulse_ch=self.res_ch, 
    #                 adcs=[self.adc_ch],
    #                 adc_trig_offset=cfg.device.readout.trig_offset,
    #                  t='auto', wait=True, syncdelay=self.us2cycles(2.5))#self.cfg["relax_delay"])  # self.us2cycles(1))
        
    #     self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced

    #     self.read(0, 0, "lower", self.r_read_q)  # read data from I buffer, QA, and store
    #     self.wait_all()  # to allow the read to be complete might be reduced
    #     self.sync_all()

    #     # perform Qubit active reset comparison, jump if condition is true to the label1 location
    #     self.condj(0, self.r_read_q, "<", self.r_thresh_q,
    #                "LABEL_1")  # compare the value recorded above to the value stored in threshold.

    #     #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
    #     self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ge, style="arb",
    #                              phase=self.deg2reg(0),
    #                              gain=self.pi_gain, waveform='pi_qubit')
    #     self.pulse(ch=self.qubit_ch)
    #     self.label("LABEL_1")  # location to be jumped to
    #     self.wait_all() 
    #     self.sync_all()
    #     # ======================================================

    #     # Reset ef level
    #     # ======================================================
    #     self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ef, style="arb",
    #                              phase=self.deg2reg(0),
    #                              gain=self.pi_ef_gain, waveform='pi_ef_qubit')
    #     self.pulse(ch=self.qubit_ch)
    #     self.sync_all(self.us2cycles(0.05))
    #     self.measure(pulse_ch=self.res_ch, 
    #                 adcs=[self.adc_ch],
    #                 adc_trig_offset=cfg.device.readout.trig_offset,
    #                  t='auto', wait=True, syncdelay=self.us2cycles(2.5))  # self.us2cycles(1))
        
    #     self.wait_all(self.us2cycles(0.1))  # to allow the read to be complete might be reduced
        
    #     self.read(0, 0, "lower", self.r_read_q_ef)  # read data from I buffer, QA, and store
    #     self.wait_all()  # to allow the read to be complete might be reduced
    #     self.sync_all()

    #     # perform Qubit active reset comparison, jump if condition is true to the label1 location
    #     self.condj(0, self.r_read_q_ef, "<", self.r_thresh_q,
    #                "LABEL_2")  # compare the value recorded above to the value stored in threshold.

    #     #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
    #     self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ge, style="arb",
    #                              phase=self.deg2reg(0),
    #                              gain=self.pi_gain, waveform='pi_qubit')
    #     self.pulse(ch=self.qubit_ch)
    #     self.label("LABEL_2")  # location to be jumped to
    #     self.wait_all() 
    #     self.sync_all()

    #     # ======================================================

    #     # post selection

    #     # ======================================================
    #     self.measure(pulse_ch=self.res_ch, 
    #                 adcs=[self.adc_ch],
    #                 adc_trig_offset=cfg.device.readout.trig_offset,
    #                  t='auto', wait=True, syncdelay=self.us2cycles(2.5))  # self.us2cycles(1))
    #     self.wait_all() 
    #     self.sync_all()

    #     # ======================================================


    #     self.sync_all(self.us2cycles(0.05))  # not sure if this is needed

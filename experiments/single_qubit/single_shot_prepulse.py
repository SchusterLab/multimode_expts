import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm
from MM_base import *
from MM_rb_base import * 


 

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

    return fids, thresholds, theta*180/np.pi # fids: ge, gf, ef

# ====================================================== #

class HistogramPrepulseProgram(MMRBAveragerProgram):
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
        self.MM_base_initialize()
    
    # def reset_and_sync(self):
    #     # Phase reset all channels except readout DACs 

    #     # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
    #     # self.setup_and_pulse(ch=self.qubit_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[0]), phase=0, gain=5, length=10, phrst=1)
    #     # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
    #     # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
    #     # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
    #     # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
    #     # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)


    #     #initialize the phase to be 0
    #     self.set_pulse_registers(ch=self.qubit_chs, freq=self.f_ge,
    #                              phase=0, gain=0, length=10, style="const", phrst=1)
    #     self.pulse(ch=self.qubit_chs)
    #     self.set_pulse_registers(ch=self.man_chs, freq=self.f_ge,
    #                              phase=0, gain=0, length=10, style="const", phrst=1)
    #     self.pulse(ch=self.man_chs)
    #     # self.set_pulse_registers(ch=self.storage_ch, freq=self.f_ge,
    #     #                          phase=0, gain=0, length=10, style="const", phrst=1)
    #     # self.pulse(ch=self.storage_ch)
    #     self.set_pulse_registers(ch=self.flux_low_ch, freq=self.f_ge,
    #                              phase=0, gain=0, length=10, style="const", phrst=1)
    #     self.pulse(ch=self.flux_low_ch)
    #     self.set_pulse_registers(ch=self.flux_high_ch, freq=self.f_ge,
    #                              phase=0, gain=0, length=10, style="const", phrst=1)
    #     self.pulse(ch=self.flux_high_ch)
    #     self.set_pulse_registers(ch=self.f0g1_ch, freq=self.f_ge,
    #                              phase=0, gain=0, length=10, style="const", phrst=1)
    #     self.pulse(ch=self.f0g1_ch)

    #     self.sync_all(10)
    
    def body(self):
        qTest = 0
        cfg=AttrDict(self.cfg)


        # phase reset
        self.reset_and_sync()

        # Active Reset
        if cfg.expt.active_reset:
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        # Prepulse 
        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                print(creator.pulse)
                self.custom_pulse_with_preloaded_wfm(cfg, creator.pulse, prefix = 'pre_gate_based_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')
        
        # if cfg.expt.postpulse:
        #     if cfg.expt.gate_based: 
        #     # post pulse with phase correction 
        #         idling_time = cfg.expt.idling_time 
        #         f0g1_idling_freq, stor_idling_freq = cfg.expt.idling_freq

        #         # assuming the post pulse is M1_sx pi, f0g1 pi, ef pi
        #         creator = self.get_prepulse_creator(cfg.expt.post_sweep_pulse)
        #         post_pulse_str = creator.pulse
        #         post_pulse_str[0][-1] = stor_idling_freq * idling_time
        #         post_pulse_str[1][-1] = f0g1_idling_freq * (idling_time + 12*0.005*  

        #         self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')
 
        # if cfg.expt.gate_based:
        #     idling_times = []

        #     for idx, prepulse_str in enumerate(cfg.expt.pre_sweep_pulse):

        #         idling_times.append(prepulse_str[2])
            
        #     self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse_gate_based, prefix = 'pre_gate_')
        

        





        #  prepulse
        # self.sync_all()
        # if cfg.expt.prepulse:
        #     for ii in range(len(cfg.expt.pre_sweep_pulse[0])):
        #         # translate ch id to ch
        #         if cfg.expt.pre_sweep_pulse[4][ii] == 1:
        #             self.tempch = self.flux_low_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 2:
        #             self.tempch = self.qubit_chs
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 3:
        #             self.tempch = self.flux_high_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 6:
        #             self.tempch = self.storage_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 5:
        #             self.tempch = self.f0g1_ch
        #         elif cfg.expt.pre_sweep_pulse[4][ii] == 4:
        #             self.tempch = self.man_ch
        #         # print(self.tempch)
        #         # determine the pulse shape
        #         if cfg.expt.pre_sweep_pulse[5][ii] == "gaussian":
        #             # print('gaussian')
        #             self.pisigma_resolved = self.us2cycles(
        #                 cfg.expt.pre_sweep_pulse[6][ii], gen_ch=self.tempch)
        #             self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(ii),
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
        #             self.setup_and_pulse(ch=self.tempch, style="arb", 
        #                              freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch), 
        #                              phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
        #                              gain=cfg.expt.pre_sweep_pulse[1][ii], 
        #                              waveform="temp_gaussian"+str(ii))
        #         elif cfg.expt.pre_sweep_pulse[5][ii] == "flat_top":
        #             # print('flat_top')
        #             self.pisigma_resolved = self.us2cycles(
        #                 cfg.expt.pre_sweep_pulse[6][ii], gen_ch=self.tempch)
        #             self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(ii),
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
        #             self.setup_and_pulse(ch=self.tempch, style="flat_top", 
        #                              freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch), 
        #                              phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
        #                              gain=cfg.expt.pre_sweep_pulse[1][ii], 
        #                              length=self.us2cycles(cfg.expt.pre_sweep_pulse[2][ii], 
        #                                                    gen_ch=self.tempch),
        #                             waveform="temp_gaussian"+str(ii))
        #         else:
        #             self.setup_and_pulse(ch=self.tempch, style="const", 
        #                              freq=self.freq2reg(cfg.expt.pre_sweep_pulse[0][ii], gen_ch=self.tempch), 
        #                              phase=self.deg2reg(cfg.expt.pre_sweep_pulse[3][ii]), 
        #                              gain=cfg.expt.pre_sweep_pulse[1][ii], 
        #                              length=self.us2cycles(cfg.expt.pre_sweep_pulse[2][ii], 
        #                                                    gen_ch=self.tempch))
        #         self.sync_all()

        self.measure_wrapper()

    # def collect_shots(self):
    #     # collect shots for the relevant adc and I and Q channels
    #     cfg=AttrDict(self.cfg)
    #     # print(np.average(self.di_buf[0]))
    #     shots_i0 = self.di_buf[0] / self.readout_length_adc
    #     shots_q0 = self.dq_buf[0] / self.readout_length_adc
    #     return shots_i0, shots_q0
        # return shots_i0[:5000], shots_q0[:5000]


class HistogramPrepulseExperiment(Experiment):
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
        self.format_config_before_experiment( num_qubits_sample) 

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        # Ground state shots
        cfg = self.cfg #AttrDict((self.cfg))
        histpro = HistogramPrepulseProgram(soccfg=self.soccfg, cfg=cfg)
        # print(histpro)
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, 
                                     readouts_per_experiment=read_num)
        data = dict()
        data['I'], data['Q'] = histpro.collect_shots()

        self.data = data
        return data

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

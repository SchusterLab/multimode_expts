# Author: Ziqian 09/01/2024

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
import random

from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.single_qubit.single_shot import  HistogramProgram

import experiments.fitting as fitter
from MM_base import *

"""
Single Beam Splitter RB sequence generator
Gate set = {+-X/2, +-Y/2, X, Y}
"""
## generate sequences of random pulses
## 1:X,   2:Y, 3:X/2
## 4:Y/2, 5:-X/2, 6:-Y/2
## 0:I


## Calculate inverse rotation
matrix_ref = {}
# Z, X, Y, -Z, -X, -Y
matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
matrix_ref['1'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 0, 0]])
matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1]])
matrix_ref['3'] = np.matrix([[0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0]])
matrix_ref['4'] = np.matrix([[0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1]])
matrix_ref['5'] = np.matrix([[0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0, 0]])
matrix_ref['6'] = np.matrix([[0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1]])

def no2gate(no):
    g = 'I'
    if no==1:
        g = 'X'
    elif no==2:
        g = 'Y'
    elif no==3:
        g = 'X/2'
    elif no==4:
        g = 'Y/2'
    elif no==5:
        g = '-X/2'
    elif no==6:
        g = '-Y/2'  

    return g

def gate2no(g):
    no = 0
    if g=='X':
        no = 1
    elif g=='Y':
        no = 2
    elif g=='X/2':
        no = 3
    elif g=='Y/2':
        no = 4
    elif g=='-X/2':
        no = 5
    elif g=='-Y/2':
        no = 6

    return no

def generate_sequence(rb_depth, iRB_gate_no=-1, debug=False, matrix_ref=matrix_ref):
    gate_list = []
    for ii in range(rb_depth):
        gate_list.append(random.randint(1, 6))
        if iRB_gate_no > -1:   # performing iRB
            gate_list.append(iRB_gate_no)

    a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
    anow = a0
    for i in gate_list:
        anow = np.dot(matrix_ref[str(i)], anow)
    anow1 = np.matrix.tolist(anow.T)[0]
    max_index = anow1.index(max(anow1))
    # inverse of the rotation
    inverse_gate_symbol = ['-Y/2', 'X/2', 'X', 'Y/2', '-X/2']
    if max_index == 0:
        pass
    else:
        gate_list.append(gate2no(inverse_gate_symbol[max_index-1]))
    if debug:
        print(gate_list)
        print(max_index)
    return gate_list

class SingleBeamSplitterRBPostselectionrun(MMAveragerProgram):
    """
    RB program for single qubit gates
    """

    def __init__(self, soccfg, cfg):
        # gate_list should include the total gate!
        self.gate_list =  cfg.expt.running_list
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        super().__init__(soccfg, cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        # self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
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

        # get register page for qubit_chs
        # self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]
        # self.rf_rps = [self.ch_page(ch) for ch in self.rf_ch]

        self.f_ge_reg = [self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_ef_reg = [self.freq2reg(
            cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])]

        # self.f_ge_resolved_reg = [self.freq2reg(
        #     self.cfg.expt.qubit_resolved_pi[0], gen_ch=self.qubit_chs[qTest])]

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        # self.f_rf_reg = [self.freq2reg(self.cfg.expt.flux_drive[1], gen_ch=self.rf_ch[0])]

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

        # self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_chs)
        # self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_chs)

        # self.q_rps = self.ch_page(self.qubit_chs) # get register page for qubit_chs
        # self.f_ge_reg = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_chs)

        # defining BS gate frequency, gain, and channels, assume gaussian pulse shape for now
        ## bs_para = [[frequency], [gain], [length (us)], [sigma]]
        self.f_bs = cfg.expt.bs_para[0]
        self.gain_beamsplitter = cfg.expt.bs_para[1]
        self.length_beamsplitter = cfg.expt.bs_para[2]
        # self.phase_beamsplitter = cfg.expt.bs_para[3]
        self.ramp_beamsplitter = cfg.expt.bs_para[3]
        if self.f_bs < 1000:
            self.freq_beamsplitter = self.freq2reg(self.f_bs, gen_ch=self.flux_low_ch[0])
            self.pibs = self.us2cycles(self.ramp_beamsplitter, gen_ch=self.flux_low_ch[0])
            self.bs_ch = self.flux_low_ch
            self.add_gauss(ch=self.flux_low_ch[0], name="ramp_bs", sigma=self.pibs, length=self.pibs*6)
        else:
            self.freq_beamsplitter = self.freq2reg(self.f_bs, gen_ch=self.flux_high_ch[0])
            self.pibs = self.us2cycles(self.ramp_beamsplitter, gen_ch=self.flux_high_ch[0])
            self.bs_ch = self.flux_high_ch
            self.add_gauss(ch=self.flux_high_ch[0], name="ramp_bs", sigma=self.pibs, length=self.pibs*6)
        # print(f'BS channel: {self.bs_ch} MHz')
        # print(f'BS frequency: {self.f_bs} MHz')
        # print(f'BS frequency register: {self.freq_beamsplitter}')
        # print(f'BS gain: {self.gain_beamsplitter}')
        self.r_bs_phase = self.sreg(self.bs_ch[0], "phase") # register
        self.page_bs_phase = self.ch_page(self.bs_ch[0]) # page
        # print(f'BS page register: {self.page_bs_phase}')
        # print(f'Low BS page register: {self.ch_page(self.flux_low_ch[0])}')
        # print(f'High BS page register: {self.ch_page(self.flux_high_ch[0])}')
        # print(f'BS phase register: {self.r_phase}')
        self.safe_regwi(self.page_bs_phase, self.r_bs_phase, 0) 

        # self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_chs, ro_ch=self.adc_chs)
        # self.readout_lengths_dac = self.us2cycles(self.cfg.device.readout.readout_length, gen_ch=self.res_chs) 
        # self.readout_lengths_adc = 1+self.us2cycles(self.cfg.device.readout.readout_length, ro_ch=self.adc_chs) 

        # self.declare_readout(ch=self.adc_chs, length=self.readout_lengths_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_chs)
        # self.declare_gen(ch=self.qubit_chs, nqz=cfg.hw.soc.dacs.qubit.nyquist)
        # gen_chs.append(self.qubit_chs)


        # self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_chs)
        # self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        # self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma, gen_ch=self.qubit_chs)
        # self.hpi_gain = cfg.device.qubit.pulses.hpi_ge.gain
        


        # define all 2 different pulses
        # self.add_gauss(ch=self.qubit_chs, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        # self.add_gauss(ch=self.qubit_chs, name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)

        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])
        self.parity_pulse_for_custom_pulse = self.get_parity_str(man_mode_no = 1, return_pulse = True)

        self.wait_all(self.us2cycles(0.2))
        self.sync_all(self.us2cycles(0.2))

   
    def play_bs_gate(self, cfg, phase=0, times = 1, wait = False):
        if cfg.expt.setup:
            self.set_pulse_registers(ch=self.bs_ch[0], style="flat_top", 
                                     freq=self.freq_beamsplitter, 
                                     phase=self.deg2reg(phase), 
                                     gain=self.gain_beamsplitter, 
                                     length=self.us2cycles(self.length_beamsplitter, 
                                                           gen_ch=self.bs_ch[0]),
                                    waveform="ramp_bs")
        else: 
            self.safe_regwi(self.page_bs_phase, self.r_bs_phase, self.deg2reg(phase)) 
        
        for _ in range(times): 
            self.pulse(ch=self.bs_ch[0]) 
        if wait:
            self.sync_all(self.us2cycles(0.01))

        if cfg.expt.sync:
            self.sync_all()


    def body(self):
        cfg = AttrDict(self.cfg)

        self.vz = 0   # virtual Z phase in degree
        qTest = 0
        # phase reset
        self.reset_and_sync()

        # self.wait_all(self.us2cycles(0.2))
        self.sync_all(self.us2cycles(0.2))

        #do the active reset
        if cfg.expt.rb_active_reset:
            self.active_reset( man_reset= self.cfg.expt.rb_man_reset, storage_reset= self.cfg.expt.rb_storage_reset)

        # self.wait_all(self.us2cycles(0.2))
        # self.sync_all(self.us2cycles(0.2))

        # prepulse 
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre11')#, advance_qubit_phase=self.vz)
            # self.vz += self.cfg.expt.f0g1_offset 
        
        # prepare bs gate 
        self.set_pulse_registers(ch=self.bs_ch[0], style="flat_top", 
                                     freq=self.freq_beamsplitter, 
                                     phase=self.deg2reg(0), 
                                     gain=self.gain_beamsplitter, 
                                     length=self.us2cycles(self.length_beamsplitter, 
                                                           gen_ch=self.bs_ch[0]),
                                    waveform="ramp_bs")
        factor = self.cfg.expt.bs_repeat
        wait_bool = False
        # self.cfg.expt.running_list = [4,6]   #[3,5]
        for idx, ii in enumerate(self.cfg.expt.running_list):
            wait_bool = False
            if idx%self.cfg.expt.gates_per_wait == 0: # only wait after bs pulse every 10 gates
                wait_bool = True
        
            # add gate
            if ii == 0:
                pass
            if ii == 1:  #'X'
                #self.safe_regwi(self.bs_ch, self.r_phase, self.deg2reg(0)) 
                for _ in range(factor):
                    self.play_bs_gate(cfg, phase=0, times = 2, wait=wait_bool)

            if ii == 2:  #'Y'
                #self.safe_regwi(self.bs_ch, self.r_phase, self.deg2reg(90)) 
                for _ in range(factor):
                    self.play_bs_gate(cfg, phase=90, times = 2, wait=wait_bool)

            if ii == 3:  #'X/2'
                #self.safe_regwi(self.bs_ch, self.r_phase, self.deg2reg(0)) 
                for _ in range(factor):
                    self.play_bs_gate(cfg, phase=0, wait=wait_bool)

            if ii == 4:  #'Y/2'
                #self.safe_regwi(self.bs_ch, self.r_phase, self.deg2reg(90)) 
                for _ in range(factor):
                    self.play_bs_gate(cfg, phase=90, wait=wait_bool)

            if ii == 5:  #'-X/2'
                #self.safe_regwi(self.bs_ch, self.r_phase, self.deg2reg(180)) 
                for _ in range(factor):
                    self.play_bs_gate(cfg, phase=180, wait=wait_bool)
            if ii == 6:  #'-Y/2'
                #self.safe_regwi(self.bs_ch, self.r_phase, self.deg2reg(-90)) 
                for _ in range(factor):
                    self.play_bs_gate(cfg, phase=-90, wait=wait_bool)
                    

            ##postpulse
            #if idx == len(self.cfg.expt.running_list)-1:
        # self.wait_all(self.us2cycles(0.05))
        self.sync_all()
        # if cfg.expt.postpulse:
            # self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post22')#, advance_qubit_phase=self.vz)

                    # self.vz += self.cfg.expt.f0g1_offset 
                
        # align channels and wait 50ns and measure

        # align channels and measure
        
        # self.wait_all(self.us2cycles(0.05))
        self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas1')
        self.sync_all(self.us2cycles(0.05))

        # self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas11')
        # self.sync_all(self.us2cycles(0.05))
        # self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas21')
        # self.sync_all(self.us2cycles(0.05))
        # self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas13')
        # self.sync_all(self.us2cycles(0.05))
        # self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas14')
        # self.sync_all(self.us2cycles(0.05))
        

        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(self.cfg.expt.postselection_delay)
        )

        # self.wait_all(self.us2cycles(0.05))
        # self.sync_all()
        # parity meas to reset qubit 
        if self.cfg.expt.reset_qubit_after_parity: 
            self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_post_meas1')
        # Swap gate between two modes

        # self.custom_pulse(cfg, cfg.expt.post_selection_pulse, prefix='selection11')
        self.play_bs_gate(cfg, phase=0, times = 2, wait=True)
        self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas2')

        self.sync_all(self.us2cycles(0.05))
        # self.wait_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

    def collect_shots_rb(self, read_num):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg
        # print(self.di_buf[0])
        shots_i0 = self.di_buf[0].reshape((read_num, self.cfg["reps"]),order='F') / self.readout_lengths_adc
        # print(shots_i0)
        shots_q0 = self.dq_buf[0].reshape((read_num, self.cfg["reps"]),order='F') / self.readout_lengths_adc

        return shots_i0, shots_q0

# ===================================================================== #
# play the pulse
class SingleBeamSplitterRBPostSelection(Experiment):
    def __init__(self, soccfg=None, path='', prefix='SingleBeamSplitterRBPostSelection', config_file=None, progress=None):
            super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
    
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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        # ================= #
        # Get single shot calibration for all qubits
        # ================= #

        # g states for q0
        data=dict()
        # sscfg = AttrDict(deepcopy(self.cfg))
        sscfg = self.cfg
        sscfg.expt.reps = sscfg.expt.singleshot_reps
        # sscfg.expt.active_reset = 
        # print active reset inside sscfg 
        print('sscfg active reset ' + str(sscfg.expt.active_reset))
        # sscfg.expt.man_reset = kkk

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
        histpro_g = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
        avgi, avgq = histpro_g.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, 
                                       readouts_per_experiment=self.cfg.expt.readout_per_round)
        data['Ig'], data['Qg'] = histpro_g.collect_shots()

        # Excited state shots
        sscfg.expt.pulse_e = True 
        sscfg.expt.pulse_f = False
        histpro_e= HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
        avgi, avgq = histpro_e.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, 
                                       readouts_per_experiment=self.cfg.expt.readout_per_round)
        data['Ie'], data['Qe'] = histpro_e.collect_shots()
        # print(data)

        fids, thresholds, angle, confusion_matrix = histpro_e.hist(data=data, plot=False, verbose=False, span=self.cfg.expt.span, 
                                                         active_reset=self.cfg.expt.active_reset, threshold = self.cfg.expt.threshold,
                                                         readout_per_round=self.cfg.expt.readout_per_round)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        data['confusion_matrix'] = confusion_matrix


        print(f'ge fidelity (%): {100*fids[0]}')
        print(f'rotation angle (deg): {angle}')
        print(f'threshold ge: {thresholds[0]}')

        data['Idata'] = []
        data['Qdata'] = []

        #sequences = np.array([[0], [1]])#[1,1,1,1], [2,2,2,2],  [1,2,1,1], [1,2,2,2], [1,1,2,1]])
        #for var in sequences:
        self.cfg.expt.reps = self.cfg.expt.rb_reps
        # data['running_lists'] = []
        for var in tqdm(range(self.cfg.expt.variations)):   # repeat each depth by variations
            # generate random gate list
            self.cfg.expt.running_list =  generate_sequence(self.cfg.expt.rb_depth, iRB_gate_no=self.cfg.expt.IRB_gate_no)
            # data['running_lists'].append(self.cfg.expt.running_list)
            # print(f'Running list: {self.cfg.expt.running_list}')

        
            rb_shot = SingleBeamSplitterRBPostselectionrun(soccfg=self.soccfg, cfg=self.cfg)
            read_num =2 
            if self.cfg.expt.rb_active_reset: read_num = 5
            self.prog = rb_shot
            avgi, avgq = rb_shot.acquire(
                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug,
                        readouts_per_experiment=read_num) #,save_experiments=np.arange(0,5,1))
            II, QQ = rb_shot.collect_shots_rb(read_num)
            data['Idata'].append(II)
            data['Qdata'].append(QQ)
        #data['running_lists'] = running_lists   
        #print(self.prog)
            
        self.data = data

        return data
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
# ===================================================================== #

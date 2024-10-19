# Author: Ziqian 11/08/2023

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
import random

from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.single_qubit.single_shot_old import hist, HistogramProgram_oldold
from experiments.single_qubit.single_shot import  HistogramProgram

import experiments.fitting as fitter
from MM_base import MMAveragerProgram

"""
Single qubit RB sequence generator
Gate set = {I, +-X/2, +-Y/2, +-Z/2, X, Y, Z}
"""
## generate sequences of random pulses
## 1:Z,   2:X, 3:Y
## 4:Z/2, 5:X/2, 6:Y/2
## 7:-Z/2, 8:-X/2, 9:-Y/2
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
        gate_list.append(random.randint(1, 6))   # from 1 to 6
        if iRB_gate_no > -1:   # performing iRB
            gate_list.append(iRB_gate_no)

    a0 = np.matrix([[1], [0], [0], [0], [0], [0]]) # initial state
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

class SingleRBrun(MMAveragerProgram):
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
        self.MM_base_initialize()

    def play_ge_pulse(self, phase=0, times =1): 
        for _ in range(times): 
            # self.setup_and_pulse(ch=self.qubit_chs[0], style="flat_top", freq=self.f_hge_reg_defined,
            #                             phase=self.deg2reg(phase+self.vz), gain=self.cfg.expt.ge_pi2_pulse[1], 
            #                             length=self.f_hge_flat, waveform="ramp_up_hge")
            self.setup_and_pulse(ch=self.qubit_chs[0], style="arb", freq=self.f_ge_reg[0],
                                    phase=self.deg2reg(phase+self.vz), gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")

            #self.vz = 0
        self.sync_all()


    def body(self):
        cfg = AttrDict(self.cfg)

        # phase reset
        # self.reset_and_sync()
        #==================================================================== #
        self.vz = 0   # virtual Z phase in degree (ge phase)
        self.vz_ef = 0   # virtual Z phase in degree (ef phase)
        #==================================================================== #
        self.vz_f0g1 = 0   # virtual Z phase in degree (f0g1 phase)
        qTest = 0

        for idx, ii in enumerate(self.cfg.expt.running_list):
            ##prepulse before each gate

            if idx >0: 
                if cfg.expt.prepulse:
                    
                    creator = self.get_prepulse_creator(self.cfg.expt.pre_sweep_pulse)
                    # print(creator.pulse.tolistd())
                    self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb prepulse' + str(idx))
                    
                    # # self.vz_f0g1 += self.cfg.expt.f0g1_phase
                    # self.setup_and_pulse(ch=self.f0g1_ch, style="flat_top", 
                    #                 freq=self.f_f0g1_reg_defined, 
                    #                 phase=self.deg2reg(0), 
                    #                 gain=cfg.expt.f0g1_pi_pulse[1], 
                    #                 length=self.f_f0g1_flat,
                    #                 waveform="ramp_s")
                    
                    # self.sync_all()
                    # # self.vz_ef += self.cfg.expt.f0g1_ef_phase
                    # self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
                    #                 freq=self.f_ef_reg_defined, 
                    #                 phase=self.deg2reg(0), 
                    #                 gain=cfg.expt.ef_pi_pulse[1], 
                    #                 length=self.f_ef_flat,
                    #                 waveform="ramp_q") # ----------
                    self.vz += self.cfg.expt.f0g1_offset
                    # self.sync_all()
        
            # add gate
            if ii == 0:
                pass
            if ii == 1:  #'X'
                
                self.play_ge_pulse(phase=0, times=2)
                #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forpi
            if ii == 2:  #'Y'
                self.play_ge_pulse(phase=-90, times=2)
                
                #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forpi
            if ii == 3:  #'X/2'
                self.play_ge_pulse(phase=0, times=1)
                #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi
            if ii == 4:  #'Y/2'
                self.play_ge_pulse(phase=-90, times=1)
                #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi
            if ii == 5:  #'-X/2'
                self.play_ge_pulse(phase=-180, times=1)
                #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi
            if ii == 6:  #'-Y/2'
                self.play_ge_pulse(phase=90, times=1)
                #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi

            ##postpulse after each gate
            # if idx < len(self.cfg.expt.running_list)-1:
            #     if cfg.expt.postpulse:
            #         creator = self.get_prepulse_creator(self.cfg.expt.post_sweep_pulse)
            #         self.custom_pulse(self.cfg, creator.pulse.tolist(), prefix='Rb postpulse' + str(idx))

            if cfg.expt.postpulse:
                creator = self.get_prepulse_creator(self.cfg.expt.post_sweep_pulse)
                self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb postpulse' + str(idx))

                    # self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
                    #                 freq=self.f_ef_reg_defined, 
                    #                 phase=self.deg2reg(0), 
                    #                 gain=cfg.expt.ef_pi_pulse[1], 
                    #                 length=self.f_ef_flat,
                    #                 waveform="ramp_q") # ----------
                    # self.sync_all()
                    
                    # self.setup_and_pulse(ch=self.f0g1_ch, style="flat_top", 
                    #                 freq=self.f_f0g1_reg_defined, 
                    #                 phase=self.deg2reg(0), 
                    #                 gain=cfg.expt.f0g1_pi_pulse[1], 
                    #                 length=self.f_f0g1_flat,
                    #                 waveform="ramp_s")                        
                    # # self.vz += self.cfg.expt.f0g1_offset
                    # # self.vz = self.vz % 360
                    
                    # self.sync_all()
                
        # align channels and wait 50ns and measure
        # if cfg.expt.prepulse:
        #     self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse)#, advance_qubit_phase=self.vz)
        # print('measure')

        # parity measurement
        parity_str = [['qubit', 'ge', 'hpi', 0],
                        ['qubit', 'ge', 'parity_M1', 0 ],
                        ['qubit', 'ge', 'hpi', 180]]
        creator = self.get_prepulse_creator(parity_str)
        self.custom_pulse(self.cfg, creator.pulse.tolist(), prefix='Rb parity pulse')


        self.sync_all(self.us2cycles(0.05))
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
class SingleRB(Experiment):
    def __init__(self, soccfg=None, path='', prefix='SingleRB', config_file=None, progress=None):
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
        sscfg_readout_per_round = 1 #self.cfg.expt.readout_per_round
        if sscfg.expt.active_reset:
            sscfg_readout_per_round = 4
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
                                       readouts_per_experiment=sscfg_readout_per_round)
        data['Ig'], data['Qg'] = histpro_g.collect_shots()

        # Excited state shots
        sscfg.expt.pulse_e = True 
        sscfg.expt.pulse_f = False
        histpro_e= HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
        avgi, avgq = histpro_e.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, 
                                       readouts_per_experiment=sscfg_readout_per_round)
        data['Ie'], data['Qe'] = histpro_e.collect_shots()
        # print(data)

        fids, thresholds, angle, confusion_matrix = histpro_e.hist(data=data, plot=False, verbose=False, span=self.cfg.expt.span, 
                                                         active_reset=self.cfg.expt.active_reset, threshold = self.cfg.expt.threshold,
                                                         readout_per_round=sscfg_readout_per_round)
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

        
            rb_shot = SingleRBrun(soccfg=self.soccfg, cfg=self.cfg)
            read_num =1
            if self.cfg.expt.rb_active_reset: read_num = 4
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

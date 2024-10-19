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

import experiments.fitting as fitter

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

class SingleRBrun(AveragerProgram):
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
        qTest = 0

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

        self.man_chs = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_types = cfg.hw.soc.dacs.manipulate_in.type

        gen_chs = []

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_chs)
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_chs)

        self.q_rps = self.ch_page(self.qubit_chs) # get register page for qubit_chs
        self.f_ge_reg = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_chs)

        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_chs, ro_ch=self.adc_chs)
        self.readout_lengths_dac = self.us2cycles(self.cfg.device.readout.readout_length, gen_ch=self.res_chs) 
        self.readout_lengths_adc = 1+self.us2cycles(self.cfg.device.readout.readout_length, ro_ch=self.adc_chs) 

        self.declare_readout(ch=self.adc_chs, length=self.readout_lengths_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_chs)
        self.declare_gen(ch=self.qubit_chs, nqz=cfg.hw.soc.dacs.qubit.nyquist)
        gen_chs.append(self.qubit_chs)


        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_chs)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_chs)
        self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain
        self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma, gen_ch=self.qubit_chs)
        self.hpi_gain = cfg.device.qubit.pulses.hpi_ge.gain

        self.pi_sigma_ramp = self.us2cycles(cfg.expt.ge_pi_pulse[3], gen_ch=self.qubit_chs)
        self.hpi_sigma_ramp = self.us2cycles(cfg.expt.ge_pi2_pulse[3], gen_ch=self.qubit_chs)
        self.pief_sigma_ramp = self.us2cycles(cfg.expt.ef_pi_pulse[3], gen_ch=self.qubit_chs)


        # define all 2 different pulses
        self.add_gauss(ch=self.qubit_chs, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        self.add_gauss(ch=self.qubit_chs, name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
        self.add_gauss(ch=self.qubit_chs, name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)

        # define ramp pulses
        self.add_gauss(ch=self.qubit_chs, name="ramp_up_ge", sigma=self.pi_sigma_ramp, length=self.pi_sigma_ramp*4)
        self.add_gauss(ch=self.qubit_chs, name="ramp_up_ef", sigma=self.pief_sigma_ramp, length=self.pief_sigma_ramp*4)
        self.add_gauss(ch=self.qubit_chs, name="ramp_up_hge", sigma=self.hpi_sigma_ramp, length=self.hpi_sigma_ramp*4)

        self.defaultramp_q = self.us2cycles(0.005, gen_ch=self.qubit_chs)
        self.defaultramp_s = self.us2cycles(0.005, gen_ch=self.f0g1_ch)

        self.add_gauss(ch=self.qubit_chs, name="ramp_q", sigma=self.defaultramp_q, length=self.defaultramp_q*4)
        self.add_gauss(ch=self.f0g1_ch, name="ramp_s", sigma=self.defaultramp_s, length=self.defaultramp_s*6)
        # self.add_tanh(ch=self.f0g1_ch, name="ramp_s", sigma=self.defaultramp_s, length=self.defaultramp_s*6)
        # self.add_sin2(ch=self.f0g1_ch, name="ramp_s", length=self.defaultramp_s*4)

        self.f_ge_reg_defined = self.freq2reg(cfg.expt.ge_pi_pulse[0], gen_ch=self.qubit_chs)
        self.f_hge_reg_defined = self.freq2reg(cfg.expt.ge_pi2_pulse[0], gen_ch=self.qubit_chs)
        self.f_ge_flat = self.us2cycles(cfg.expt.ge_pi_pulse[2], gen_ch=self.qubit_chs)
        self.f_hge_flat = self.us2cycles(cfg.expt.ge_pi2_pulse[2], gen_ch=self.qubit_chs)
        self.f_ef_reg_defined = self.freq2reg(cfg.expt.ef_pi_pulse[0], gen_ch=self.qubit_chs)
        self.f_ef_flat = self.us2cycles(cfg.expt.ef_pi_pulse[2], gen_ch=self.qubit_chs)
        self.f_2ef_flat = self.us2cycles(cfg.expt.ef_2pi_pulse[2], gen_ch=self.qubit_chs)

        self.f_f0g1_reg_defined = self.freq2reg(cfg.expt.f0g1_pi_pulse[0], gen_ch=self.f0g1_ch)
        self.f_f0g1_flat = self.us2cycles(cfg.expt.f0g1_pi_pulse[2], gen_ch=self.f0g1_ch)


        self.set_pulse_registers(ch=self.res_chs, style="const", freq=self.f_res_reg, phase=self.deg2reg(
            cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_lengths_dac)

        self.sync_all(self.us2cycles(0.2))
        # print('starting rb ziqian'
        # )

    def custom_pulse(self, cfg, pulse_data, advance_qubit_phase = None, prefix='111'): 
        '''
        Executes prepulse or postpulse

        # [[frequency], [gain], [length (us)], [phases], [drive channel],
        #  [shape], [ramp sigma]],
        #  drive channel=1 (flux low), 
        # 2 (qubit),3 (flux high),4 (storage),5 (f0g1),6 (manipulate),
        '''
        if pulse_data is None:
            return None
        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        # for prepulse 
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        if advance_qubit_phase is not None:
            pulse_data[3] = [x + advance_qubit_phase for x in pulse_data[3]]

        for jj in range(len(pulse_data[0])):
                # translate ch id to ch
                if pulse_data[4][jj] == 1:
                    self.tempch = self.flux_low_ch
                elif pulse_data[4][jj] == 2:
                    self.tempch = self.qubit_ch
                elif pulse_data[4][jj] == 3:
                    self.tempch = self.flux_high_ch
                elif pulse_data[4][jj] == 6:
                    self.tempch = self.storage_ch
                elif pulse_data[4][jj] == 0:
                    self.tempch = self.f0g1_ch
                elif pulse_data[4][jj] == 4:
                    self.tempch = self.man_ch
                # print(self.tempch)
                # determine the pulse shape
                if pulse_data[5][jj] == "gaussian":
                    # print('gaussian')
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch)
                    self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch, style="arb", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     waveform="temp_gaussian"+str(jj)+prefix)
                elif pulse_data[5][jj] == "flat_top":
                    # print('flat_top')
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch)
                    self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.setup_and_pulse(ch=self.tempch, style="flat_top", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     length=self.us2cycles(pulse_data[2][jj], 
                                                           gen_ch=self.tempch),
                                    waveform="temp_gaussian"+str(jj)+prefix)
                else:
                    self.setup_and_pulse(ch=self.tempch, style="const", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                     phase=self.deg2reg(pulse_data[3][jj]), 
                                     gain=pulse_data[1][jj], 
                                     length=self.us2cycles(pulse_data[2][jj], 
                                                           gen_ch=self.tempch))
                self.sync_all()
    
    def reset_and_sync(self):
        # Phase reset all channels except readout DACs 

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)


        #initialize the phase to be 0
        self.set_pulse_registers(ch=self.qubit_chs, freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_chs)
        self.set_pulse_registers(ch=self.man_chs, freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_chs)
        # self.set_pulse_registers(ch=self.storage_ch, freq=self.f_ge,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.storage_ch)
        self.set_pulse_registers(ch=self.flux_low_ch, freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_low_ch)
        self.set_pulse_registers(ch=self.flux_high_ch, freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_high_ch)
        self.set_pulse_registers(ch=self.f0g1_ch, freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.f0g1_ch)

        self.sync_all(10)

    def play_ge_pulse(self, phase=0, times =1): 
        for _ in range(times): 
            self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", freq=self.f_hge_reg_defined,
                                        phase=self.deg2reg(phase+self.vz), gain=self.cfg.expt.ge_pi2_pulse[1], 
                                        length=self.f_hge_flat, waveform="ramp_up_hge")
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

        # self.cfg.expt.running_list = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
        # self.cfg.expt.running_list = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        # self.cfg.expt.running_list = [4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6,4,6]
        # self.cfg.expt.running_list = [3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3,5]
        # self.cfg.expt.running_list = [1,1,3,5]
        # print('------------------------------------')

        # initialize to |e> state
        # self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
        #                             phase=self.deg2reg(0+self.vz), gain=self.pi_gain, waveform="pi_qubit")
        # self.sync_all()
    
        if cfg.expt.qubit_gate_set:
            print(self.cfg.expt.running_list)


            # for idx, ii in enumerate(self.cfg.expt.running_list):
            #     ##prepulse before each gate
            #     print(idx, ii)

            #     if idx >0: 
            #         if cfg.expt.prepulse:
            #             # print('add prepulse')
            #             # self.vz_f0g1 += self.cfg.expt.f0g1_phase
            #             print('doing f0g1 pulse')
            #             # self.setup_and_pulse(ch=self.f0g1_ch, style="flat_top", 
            #             #              freq=self.f_f0g1_reg_defined, 
            #             #              phase=self.deg2reg(0+self.cfg.expt.f0g1_offset), 
            #             #              gain=cfg.expt.f0g1_pi_pulse[1], 
            #             #              length=self.f_f0g1_flat,
            #             #              waveform="ramp_s")
                        
            #             # self.sync_all()
            #             # self.vz_ef += self.cfg.expt.f0g1_ef_phase

            #             self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
            #                             freq=self.f_ef_reg_defined, 
            #                             phase=self.deg2reg(180+self.cfg.expt.f0g1_ef_phase), 
            #                             gain=cfg.expt.ef_pi_pulse[1], 
            #                             length=self.f_ef_flat,
            #                             waveform="ramp_q") # ----------
            #             #print('self.vz before the 2 swap:', self.vz)
            #             self.vz += self.cfg.expt.ef_offset_on_ge
            #             #print('self.vz after the 2 swap:', self.vz)
            #             self.sync_all()
            
            #     # add gate
            #     if ii == 0:
            #         pass
            #     if ii == 1:  #'X'
                    
            #         self.play_ge_pulse(phase=0, times=2)
            #         #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forpi
            #     if ii == 2:  #'Y'
            #         self.play_ge_pulse(phase=-90, times=2)
                    
            #         #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forpi
            #     if ii == 3:  #'X/2'
            #         self.play_ge_pulse(phase=0, times=1)
            #         #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi
            #     if ii == 4:  #'Y/2'
            #         self.play_ge_pulse(phase=-90, times=1)
            #         #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi
            #     if ii == 5:  #'-X/2'
            #         self.play_ge_pulse(phase=-180, times=1)
            #         #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi
            #     if ii == 6:  #'-Y/2'
            #         self.play_ge_pulse(phase=90, times=1)
            #         #self_temp_phase2ef = self.cfg.expt.f0g1_offset_forhpi

            #     ##postpulse after each gate
            #     if idx < len(self.cfg.expt.running_list)-1:
            #         if cfg.expt.postpulse:
            #             # print('add post pulse')
            #             self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
            #                             freq=self.f_ef_reg_defined, 
            #                             phase=self.deg2reg(0), 
            #                             gain=cfg.expt.ef_pi_pulse[1], 
            #                             length=self.f_ef_flat,
            #                             waveform="ramp_q") # ----------
                        
            #             # self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
            #             #                 freq=self.f_ef_reg_defined, 
            #             #                 phase=self.deg2reg(0), 
            #             #                 gain=cfg.expt.ef_2pi_pulse[1], 
            #             #                 length=self.f_2ef_flat,
            #             #                 waveform="ramp_q")
            #             self.sync_all()
            #             # self.setup_and_pulse(ch=self.f0g1_ch, style="flat_top", 
            #             #              freq=self.f_f0g1_reg_defined, 
            #             #              phase=self.deg2reg(0), 
            #             #              gain=cfg.expt.f0g1_pi_pulse[1], 
            #             #              length=self.f_f0g1_flat,
            #             #              waveform="ramp_s")                        
            #             # self.vz += self.cfg.expt.f0g1_offset
            #             # self.vz = self.vz % 360
                        
            #             self.sync_all()
        else:
            for idx, ii in enumerate(self.cfg.expt.running_list):
                ##prepulse before each gate

                if idx >0: 
                    if cfg.expt.prepulse:
                        # self.vz_f0g1 += self.cfg.expt.f0g1_phase
                        self.setup_and_pulse(ch=self.f0g1_ch, style="flat_top", 
                                     freq=self.f_f0g1_reg_defined, 
                                     phase=self.deg2reg(0), 
                                     gain=cfg.expt.f0g1_pi_pulse[1], 
                                     length=self.f_f0g1_flat,
                                     waveform="ramp_s")
                        
                        self.sync_all()
                        # self.vz_ef += self.cfg.expt.f0g1_ef_phase
                        self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
                                        freq=self.f_ef_reg_defined, 
                                        phase=self.deg2reg(0), 
                                        gain=cfg.expt.ef_pi_pulse[1], 
                                        length=self.f_ef_flat,
                                        waveform="ramp_q") # ----------
                        self.vz += self.cfg.expt.f0g1_ge_phase #self.cfg.expt.f0g1_offset
                        self.sync_all()
            
                # add gate
                if ii == 0:
                    pass
                if ii == 1:  #'X'
                    # self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                    #                 phase=self.deg2reg(0+self.vz), gain=self.pi_gain, waveform="pi_qubit")
                    # self.sync_all()
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(0+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(0+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.sync_all()
                if ii == 2:  #'Y'
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(-90+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(-90+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.sync_all()
                if ii == 3:  #'X/2'
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(0+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.sync_all()
                if ii == 4:  #'Y/2'
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(-90+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.sync_all()
                if ii == 5:  #'-X/2'
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(-180+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.sync_all()
                if ii == 6:  #'-Y/2'
                    self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
                                    phase=self.deg2reg(90+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
                    self.sync_all()

                ##postpulse after each gate
                if idx < len(self.cfg.expt.running_list)-1:
                    if cfg.expt.postpulse:
                        self.setup_and_pulse(ch=self.qubit_chs, style="flat_top", 
                                        freq=self.f_ef_reg_defined, 
                                        phase=self.deg2reg(0), 
                                        gain=cfg.expt.ef_pi_pulse[1], 
                                        length=self.f_ef_flat,
                                        waveform="ramp_q") # ----------
                        self.sync_all()
                        
                        self.setup_and_pulse(ch=self.f0g1_ch, style="flat_top", 
                                     freq=self.f_f0g1_reg_defined, 
                                     phase=self.deg2reg(0), 
                                     gain=cfg.expt.f0g1_pi_pulse[1], 
                                     length=self.f_f0g1_flat,
                                     waveform="ramp_s")                        
                        # self.vz += self.cfg.expt.f0g1_offset
                        # self.vz = self.vz % 360
                        
                        self.sync_all()
                
        # align channels and wait 50ns and measure
        # if cfg.expt.prepulse:
        #     self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse)#, advance_qubit_phase=self.vz)
        # print('measure')
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs,
            adcs=[self.adc_chs],
            adc_trig_offset=cfg.device.readout.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay)
        )

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc
        return shots_i0, shots_q0

# ===================================================================== #
# play the pulse
class SingleRB(Experiment):
    def __init__(self, soccfg=None, path='', prefix='SingleRB', config_file=None, progress=None):
            super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
    
    def acquire(self, progress=False, debug=False):
        qubits = self.cfg.expt.qubit

        # expand entries in config that are length 1 to fill all qubits
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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        # ================= #
        # Get single shot calibration for all qubits
        # ================= #

        # g states for q0
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
        histpro = HistogramProgram_oldold(soccfg=self.soccfg, cfg=sscfg)
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
        data['Ig'], data['Qg'] = histpro.collect_shots()

        # Excited state shots
        sscfg.expt.pulse_e = True 
        sscfg.expt.pulse_f = False
        histpro = HistogramProgram_oldold(soccfg=self.soccfg, cfg=sscfg)
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

        data['Idata'] = []
        data['Qdata'] = []
        for var in tqdm(range(self.cfg.expt.variations)):   # repeat each depth by variations
            # generate random gate list
            self.cfg.expt.running_list = generate_sequence(self.cfg.expt.rb_depth, iRB_gate_no=self.cfg.expt.IRB_gate_no)
            # print(self.cfg.expt.running_list)

        
            rb_shot = SingleRBrun(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = rb_shot
            avgi, avgq = rb_shot.acquire(
                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)
            II, QQ = rb_shot.collect_shots()
            data['Idata'].append(II)
            data['Qdata'].append(QQ)
            
        self.data = data

        return data
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
# ===================================================================== #

from qick import *
import numpy as np
from qick.helpers import gauss
import time
from slab import AttrDict
from dataset import * 
from dataset import storage_man_swap_dataset
import matplotlib.pyplot as plt
import random
from MM_base import * 



class MM_dual_rail_base(MM_base): 
    def __init__(self, cfg):
        ''' rb base is base class of f0g1 rb for storage modes '''
        super().__init__( cfg)
        # self.init_gate_length() # creates the dictionary of gate lengths
    
    
    def initialize_beam_splitter_pulse(self):
        ''' initializes the beam splitter pulse
         
        this is for characterizing a beam splitter pulse '''
        cfg = self.cfg
        qTest = 0 
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


    def prep_random_state_mode(self, state_num, mode_no): 
        '''
        preapre a cardinal state in a storage mode 
        formalism for state num 
        1: |0>
        2: |1>
        3: |+>
        4: |->
        5: |i>
        6: |-i>
        '''
        qubit_hpi_pulse_str = [['qubit', 'ge', 'hpi', 0 ]]
        qubit_ef_pulse_str = [['qubit', 'ef', 'pi', 0 ]]
        man_pulse_str = [['man', 'M1', 'pi', 0]]
        storage_pusle_str = [['storage', 'M1-S'+ str(mode_no), 'pi', 0]]

        if state_num == 5:
            qubit_hpi_pulse_str[0][3] = 90
        if state_num == 6:
            qubit_hpi_pulse_str[0][3] = -90
        
        pulse_str = []
        if state_num == 2: 
            pulse_str += qubit_hpi_pulse_str + qubit_hpi_pulse_str
        elif state_num == ( 3 or 4 or 5 or 6): 
            pulse_str += qubit_hpi_pulse_str 
        
        pulse_str += qubit_ef_pulse_str + man_pulse_str + storage_pusle_str

        return pulse_str
    
    def prepulse_str_for_random_ram_state(self, num_occupied_smodes, skip_modes): 
        '''
        prepare a random state in the storage modes

        num_occupied_smodes: number of occupied storage modes
        skip_modes: list of modes to skip [if have 7 modes, then 7- len(skip_modes) > num_occupied_smodes]
        '''
        # set up storage modes
        mode_list = []
        for i in range(1, 7+1): 
            if i in skip_modes: 
                continue
            mode_list.append(i)

        # set up states 
        state_list = [1+i for i in range(6)] # for 6 cardinal states
        
        prepulse_str = [] # gate based 
        for i in range(num_occupied_smodes): 
            state_num = random.choice(state_list)
            mode_num = random.choice(mode_list)
            print(f'Preparing state {state_num} in mode {mode_num}')
            mode_list.remove(mode_num) # remove the mode from the list
            prepulse_str += self.prep_random_state_mode(state_num, i+1)
        return prepulse_str

class MMDualRailAveragerProgram(AveragerProgram, MM_dual_rail_base):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)
    
        

    

    
        


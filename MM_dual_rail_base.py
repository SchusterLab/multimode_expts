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
from experiments.single_qubit.single_shot import  HistogramProgram
from copy import deepcopy



class MM_dual_rail_base(MM_base): 
    def __init__(self, cfg):
        ''' rb base is base class of f0g1 rb for storage modes '''
        super().__init__( cfg)
        # self.init_gate_length() # creates the dictionary of gate lengths
    
    def run_single_shot(self, self_expt, data,   progress=True, debug=False):
        '''
        self_expt: self method of expt class
        Runs single shot ; assumes follwing parameters in cfg.expt
        
        singleshot_reps: 20000
        singleshot_active_reset: True
        singleshot_man_reset: True
        singleshot_storage_reset: True

        Son't want to place this inside MMbase since then it would be circular import 

        '''
        # sscfg = AttrDict(deepcopy(self_expt.cfg))
        sscfg = deepcopy(self_expt.cfg)
        sscfg.expt.reps = sscfg.expt.singleshot_reps
        
        # sscfg.expt.active_reset = 
        # print active reset inside sscfg 
        print('sscfg active reset ' + str(sscfg.expt.singleshot_active_reset))
        sscfg.active_reset = sscfg.expt.singleshot_active_reset
        sscfg.man_reset = sscfg.expt.singleshot_man_reset
        sscfg.storage_reset = sscfg.expt.singleshot_storage_reset

        if sscfg.active_reset:
            readouts_per_experiment = 4
        else:
            readouts_per_experiment = 1

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
        histpro_g = HistogramProgram(soccfg=self_expt.soccfg, cfg=sscfg)
        avgi, avgq = histpro_g.acquire(self_expt.im[self_expt.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, 
                                       readouts_per_experiment=readouts_per_experiment)
        data['Ig'], data['Qg'] = histpro_g.collect_shots()

        # Excited state shots
        sscfg.expt.pulse_e = True 
        sscfg.expt.pulse_f = False
        histpro_e= HistogramProgram(soccfg=self_expt.soccfg, cfg=sscfg)
        avgi, avgq = histpro_e.acquire(self_expt.im[self_expt.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug, 
                                       readouts_per_experiment=readouts_per_experiment)
        data['Ie'], data['Qe'] = histpro_e.collect_shots()
        # print(data)

        fids, thresholds, angle, confusion_matrix = histpro_e.hist(data=data, plot=False, verbose=False, span=self_expt.cfg.expt.span, 
                                                         active_reset=self_expt.cfg.expt.active_reset, threshold = self_expt.cfg.device.readout.threshold[0],
                                                         readout_per_round=readouts_per_experiment)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        data['confusion_matrix'] = confusion_matrix
        return data
    
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

    def prep_man_photon(self, man_no, hpi = False): 
        ''' prepare a photon in the manipulate mode '''
        qubit_pi_pulse_str = [['qubit', 'ge', 'pi', 0 ]]
        if hpi: 
            qubit_pi_pulse_str[0][2] = 'hpi'
        qubit_ef_pulse_str = [['qubit', 'ef', 'pi', 0 ]]
        man_pulse_str = [['man', 'M' + str(man_no), 'pi', 0]]
        return qubit_pi_pulse_str + qubit_ef_pulse_str + man_pulse_str


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
        if state_num == 1:  # nothing to do for prepping 0 state
            return []
        qubit_hpi_pulse_str = [['qubit', 'ge', 'hpi', 0 ]]
        qubit_pi_pulse_str = [['qubit', 'ge', 'pi', 0 ]]
        qubit_ef_pulse_str = [['qubit', 'ef', 'pi', 0 ]]
        man_pulse_str = [['man', 'M1', 'pi', 0]]
        storage_pusle_str = [['storage', 'M1-S'+ str(mode_no), 'pi', 0]]

        if state_num == 4: 
            qubit_hpi_pulse_str[0][3] = 180
        if state_num == 5:
            qubit_hpi_pulse_str[0][3] = 90
        if state_num == 6:
            qubit_hpi_pulse_str[0][3] = -90
        
        pulse_str = []
        if state_num == 2: 
            pulse_str +=  qubit_pi_pulse_str #qubit_hpi_pulse_str + qubit_hpi_pulse_str
        elif state_num !=1:  # is 3,4,5,6
            pulse_str += qubit_hpi_pulse_str 
        
        pulse_str += qubit_ef_pulse_str + man_pulse_str + storage_pusle_str

        return pulse_str
    
    def prepulse_str_for_random_ram_state(self, num_occupied_smodes, skip_modes,
                                        target_spectator_mode = None, target_state = None): 
        '''
        prepare a random state in the storage modes

        num_occupied_smodes: number of occupied storage modes or total spectator storage modes with populations
        skip_modes: list of modes to skip [if have 7 modes, then 7- len(skip_modes) > num_occupied_smodes]
        
        target_spectator_mode: if not None, then the target spectator mode to prepare the state in (only compatible with num_occupied_smodes = 1)
        target_state: if not None, then the target state to prepare the state in (only compatible with num_occupied_smodes = 1)
        '''
        if target_spectator_mode is not None and num_occupied_smodes != 1: 
            raise ValueError('target_spectator_mode can only be used with num_occupied_smodes = 1')

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
            # print(f'Preparing state {state_num} in mode {mode_num}')
            mode_list.remove(mode_num) # remove the mode from the list

            if target_spectator_mode is not None: 
                mode_num = target_spectator_mode
            if target_state is not None:
                state_num = target_state
            prepulse_str += self.prep_random_state_mode(state_num, mode_num)
        return prepulse_str
    
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
            # print(f'Playing BS gate with phase {phase}')
            self.pulse(ch=self.bs_ch[0]) 
        if wait:
            self.sync_all(self.us2cycles(0.01))

        if cfg.expt.sync:
            self.sync_all()

    def get_total_time_from_running_list(self, running_list, bs_time):
        '''
        Calculate total time taken up by a RB sequence
        '''
        total_time = 0
        bs_gate_num = 0
        for ii in running_list:
            if ii == 0:
                total_time += 0
            elif ii == 1 or ii == 2:
                total_time += 2*bs_time
                bs_gate_num += 2
            else: 
                total_time += bs_time
                bs_gate_num += 1
        return total_time, bs_gate_num
    
    

    def no2gate(self, no):
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

    def gate2no(self, g):
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

    def generate_sequence(self, rb_depth, iRB_gate_no=-1, debug=False):
        
        # matrices 
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
            gate_list.append(self.gate2no(inverse_gate_symbol[max_index-1]))
        if debug:
            print(gate_list)
            print(max_index)
        return gate_list

class MMDualRailAveragerProgram(AveragerProgram, MM_dual_rail_base):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)

class MMDualRailRAveragerProgram(RAveragerProgram, MM_dual_rail_base):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)
    
        

    

    
        


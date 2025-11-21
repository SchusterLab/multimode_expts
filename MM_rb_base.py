from qick import *
import numpy as np
from qick.helpers import gauss
import time
from slab import AttrDict
from dataset import * 
from dataset import StorageManSwapDataset
import matplotlib.pyplot as plt
import random
from MM_base import * 



class MM_rb_base(MM_base): 
    def __init__(self, cfg):
        ''' rb base is base class of f0g1 rb for storage modes '''
        super().__init__( cfg)
        self.init_gate_length() # creates the dictionary of gate lengths
    
    def initialize_pulse_registers(self, storage_no = 1): 
        '''
        Initializes 
        -  f0g1 ch to be at M1 
        -  storage_ch to be at Si where i is the storage_no
        - if use_arb_waveform is true, preload arbitrary waveform for M1-Si and f0-g1, otherwise still use flat_top pulse
        '''
        qTest = 0 

        ## initialize qubit 
        pulse_str = [['qubit', 'ge', 'hpi', 0]]
        pulse = self.get_prepulse_creator(pulse_str).pulse.tolist() # [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]], drive channel=1 (flux low), 2 (qubit),3 (flux high),4 (storage),5 (f0g1),6 (manipulate),
        # print(pulse)
        self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb",
                                        freq=self.freq2reg(pulse[0][0], gen_ch=self.qubit_chs[qTest]),
                                        phase=self.deg2reg(0),
                                        gain=pulse[1][0],
                                        #length=self.us2cycles(pulse[2][0], gen_ch=self.qubit_chs[qTest]),
                                        waveform="pi_qubit_ge")
        self.r_qubit_phase = self.sreg(self.qubit_chs[qTest], "phase") # register # for phase update
        self.r_qubit_freq = self.sreg(self.qubit_chs[qTest], "freq") # register # for freq update
        self.r_qubit_gain = self.sreg(self.qubit_chs[qTest], "gain") # register # for gain update
        self.page_qubit = self.ch_page(self.qubit_chs[qTest])
        # self.f_ge_reg = self.freq2reg(self.cfg.device.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        # print('Register page for qubit phase:', self.r_qubit_phase)  
        # print('Register page for qubit freq:', self.r_qubit_freq)
        # print('Register page for qubit gain:', self.r_qubit_gain)




        ### initialize f0g1 to be at M1
        pulse_str = [['man', 'M1', 'pi', 0]]
        pulse = self.get_prepulse_creator(pulse_str).pulse.tolist() # [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]], drive channel=1 (flux low), 2 (qubit),3 (flux high),4 (storage),5 (f0g1),6 (manipulate),
        if self.cfg.expt.use_arb_waveform: 
            self.set_pulse_registers(ch=self.f0g1_ch[qTest], style="arb", 
                                     freq=self.freq2reg(pulse[0][0], gen_ch=self.f0g1_ch[qTest]), 
                                     phase=self.deg2reg(0), 
                                     gain=pulse[1][0], 
                                     #length=self.us2cycles(pulse[2][0], gen_ch=self.f0g1_ch[qTest]), 
                                    waveform="pi_f0g1_arb")
        else:
            self.set_pulse_registers(ch=self.f0g1_ch[qTest], style="flat_top", 
                                        freq=self.freq2reg(pulse[0][0], gen_ch=self.f0g1_ch[qTest]),
                                        phase=self.deg2reg(0), 
                                        gain=pulse[1][0], 
                                        length=self.us2cycles(pulse[2][0], gen_ch=self.f0g1_ch[qTest]), 
                                        waveform="pi_f0g1")
        self.r_f0g1_phase = self.sreg(self.f0g1_ch[qTest], "phase") # register # for phase update 
        self.page_f0g1_phase = self.ch_page(self.f0g1_ch[qTest]) # page

        ### initialize storage to be at Si
        pulse_str = [['storage', 'M1-S' + str(storage_no), 'pi', 0]]
        pulse = self.get_prepulse_creator(pulse_str).pulse.tolist() # [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]], drive channel=1 (flux low), 2 (qubit),3 (flux high),4 (storage),5 (f0g1),6 (manipulate),
        # print(pulse)
        if self.cfg.expt.use_arb_waveform: 
            if int(storage_no)<5:
                self.set_pulse_registers(ch=self.flux_low_ch[qTest], style="arb", 
                                        freq=self.freq2reg(pulse[0][0], gen_ch=self.flux_low_ch[qTest]), 
                                        phase=self.deg2reg(0), 
                                        gain=pulse[1][0], 
                                        #length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]), 
                                        waveform="pi_m1s" + str(storage_no) + "_arb")
            else:
                self.set_pulse_registers(ch=self.flux_high_ch[qTest], style="arb", 
                                        freq=self.freq2reg(pulse[0][0], gen_ch=self.flux_high_ch[qTest]), 
                                        phase=self.deg2reg(0), 
                                        gain=pulse[1][0], 
                                        #length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]), 
                                        waveform="pi_m1s" + str(storage_no) + "_arb")
        else:
            if int(storage_no)<5:
                self.set_pulse_registers(ch=self.flux_low_ch[qTest], style="flat_top",
                                                freq=self.freq2reg(pulse[0][0], gen_ch=self.flux_low_ch[qTest]),
                                                phase=self.deg2reg(0), 
                                                gain=pulse[1][0], 
                                                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]), 
                                                waveform="pi_m1si_low")
            else:
                self.set_pulse_registers(ch=self.flux_high_ch[qTest], style="flat_top",
                                                freq=self.freq2reg(pulse[0][0], gen_ch=self.flux_high_ch[qTest]),
                                                phase=self.deg2reg(0), 
                                                gain=pulse[1][0], 
                                                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_high_ch[qTest]), 
                                                waveform="pi_m1si_low")
        
        self.r_flux_low_phase = self.sreg(self.flux_low_ch[qTest], "phase") # register # for phase update 
        self.page_flux_low_phase = self.ch_page(self.flux_low_ch[qTest]) # page
        self.r_flux_high_phase = self.sreg(self.flux_high_ch[qTest], "phase") # register # for phase update
        self.page_flux_high_phase = self.ch_page(self.flux_high_ch[qTest]) # page

    def custom_pulse_with_preloaded_wfm(self, cfg, pulse_data, advance_qubit_phase = None, sync_zero_const = False, prefix='pre',
                                        same_storage = False, same_qubit_pulse = False, storage_no=1): 
        '''
        Executes prepulse or postpulse

        # [[frequency], [gain], [length (us)], [phases], [drive channel],
        #  [shape], [ramp sigma]],
        #  drive channel=1 (flux low), 
        # 2 (qubit),3 (flux high),4 (storage),0 (f0g1),6 (manipulate),

        same_storage: if True, then the storage mode is not changed, we can reuse already prgrammed pulse
        '''
        # print('------------------------------')
        # print(pulse_data)
        if pulse_data is None:
            return None
        
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
            elif pulse_data[4][jj] == 0:   # used to be 5
                self.tempch = self.f0g1_ch
            elif pulse_data[4][jj] == 4:
                self.tempch = self.man_ch
            # print(self.tempch)
            if type(self.tempch) == list:
                self.tempch = self.tempch[0]
            # determine the pulse shape

            waveform_name = None 

            if pulse_data[5][jj] == "gaussian" or pulse_data[5][jj] == "gauss" or pulse_data[5][jj] == "g": 
                # likely a qubit pulse on ge space with 35 ns sigma 
                waveform_name = "pi_qubit_ge"
                # self.sync_all(self.us2cycles(0.01))
                # if self.cfg.expt.preloaded_pulses and self.tempch == 2:
                #     self.safe_regwi(self.page_qubit_phase, self.r_qubit_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                #     self.pulse(ch=self.tempch) 
                # self.setup_and_pulse(ch=self.tempch, style="arb", 
                #                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                #                     phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                #                     gain=pulse_data[1][jj], 
                #                     waveform=waveform_name)
                if self.cfg.expt.preloaded_pulses and self.tempch == 2 and same_qubit_pulse: 
                    self.pulse(ch=self.tempch)
                #     # else:
                #         # print('reusing qubit')
                #         # print('Setting phase to ', pulse_data[3][jj])
                #         # print('Setting freq to ', self.f_ge_reg[0])
                #         # print('Setting gain to ', pulse_data[1][jj])

                #         # self.safe_regwi(self.page_qubit, self.r_qubit_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                #         # self.safe_regwi(self.page_qubit, self.r_qubit_freq, self.f_ge_reg[0])
                #         # self.safe_regwi(self.page_qubit, self.r_qubit_gain, pulse_data[1][jj])
                #         # # self.sync_all(self.us2cycles(0.02))
                #         # self.pulse(ch=self.tempch)
                else: 
                    self.setup_and_pulse(ch=self.tempch, style="arb", 
                                freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                gain=pulse_data[1][jj], 
                                waveform=waveform_name)
                
            elif pulse_data[5][jj] == "flat_top" or pulse_data[5][jj] == "f":
                if self.tempch == 0 : 
                    waveform_name = "pi_f0g1"
                elif self.tempch == 1:
                    waveform_name = "pi_m1si_low"
                elif self.tempch == 3:
                    waveform_name = "pi_m1si_high"
                # elif self.tempch == 2: 
                #     waveform_name = "pi_qubit_ef_ftop"

                # self.sync_all(self.us2cycles(0.01))
                if self.cfg.expt.preloaded_pulses and self.tempch == 0: # f0g1 resuse
                    self.safe_regwi(self.page_f0g1_phase, self.r_f0g1_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                    self.pulse(ch=self.tempch) 

                elif self.cfg.expt.preloaded_pulses and self.tempch == (1 or 3) and same_storage: # storage reuse
                    # print(self.tempch)
                    if self.tempch == 1: 
                        self.safe_regwi(self.page_flux_low_phase, self.r_flux_low_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                    else: 
                        self.safe_regwi(self.page_flux_high_phase, self.r_flux_high_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                    self.pulse(ch=self.tempch)
                
                # elif self.cfg.expt.preloaded_pulses and self.tempch == 2: # qubit reuse
                #     self.safe_regwi(self.page_qubit_phase, self.r_qubit_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                #     self.pulse(ch=self.tempch)
                else: 
                    # using arb waveform for flat top pulse
                    
                    if self.cfg.expt.use_arb_waveform:
                        print('printing arb waveform')
                        if self.tempch == 0:  # f0g1
                            self.setup_and_pulse(ch=self.tempch, style="arb", 
                                            freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                            phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                            gain=pulse_data[1][jj],
                                        waveform="pi_f0g1_arb")
                        else:  # M1-Si, need to specify storage number
                            self.setup_and_pulse(ch=self.tempch, style="arb", 
                                                freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                                phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                                gain=pulse_data[1][jj],
                                            waveform="pi_m1s" + str(storage_no) + "_arb")
                    else:                    
                        # using standard flat top pulse
                        # print('printing flat_top waveform')
                        self.setup_and_pulse(ch=self.tempch, style="flat_top", 
                                            freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                            phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                            gain=pulse_data[1][jj], 
                                            length=self.us2cycles(pulse_data[2][jj], 
                                                                gen_ch=self.tempch),
                                        waveform=waveform_name)
            else:
                if sync_zero_const and pulse_data[1][jj] ==0: 
                    self.sync_all(self.us2cycles(pulse_data[2][jj])) #, 
                                                        #gen_ch=self.tempch))
                else:
                    self.setup_and_pulse(ch=self.tempch, style="const", 
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                    gain=pulse_data[1][jj], 
                                    length=self.us2cycles(pulse_data[2][jj], 
                                                        gen_ch=self.tempch))
            # self.wait_all(self.us2cycles(0.01))
            self.sync_all(self.us2cycles(0.01))
            # print(waveform_name)
        
    
    def init_gate_length(self): 
        ''' Creates a dictionary of the form 
        gate_t_length = {
        'pi_ge_length': 60,
        'hpi_ge_length': 60,
        'pi_ef_length': 60,
        'f0g1_length': 270,
        'M1S1_length': 400,
        'M1S2_length': 400,
        'M1S3_length': 400,
        'M1S4_length': 400,
        'M1S5_length': 400,
        'M1S6_length': 400,
        'M1S7_length': 400,}

        Note gate time already includes the sync time  
        '''
        self.gate_t_length = {}
        self.gate_t_length['pi_ge_length'] = self.get_total_time([['qubit', 'ge', 'hpi', 0], ['qubit', 'ge', 'hpi', 0]], gate_based=True, cycles=True)
        self.gate_t_length['hpi_ge_length'] = self.get_total_time([['qubit', 'ge', 'hpi', 0]], gate_based=True, cycles=True)
        self.gate_t_length['pi_ef_length'] = self.get_total_time([['qubit', 'ef', 'pi', 0]], gate_based=True, cycles=True)
        self.gate_t_length['f0g1_length'] = self.get_total_time([['man', 'M1', 'pi', 0]], gate_based=True, cycles=True)
        for storage_no in range(1, 8):
            self.gate_t_length[f'M1S{storage_no}_length'] = self.get_total_time([['storage', 'M1-S' + str(storage_no), 'pi', 0]], gate_based=True, cycles=True)
        # print(self.gate_t_length)
        return None


      

    
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

    def generate_sequence(self, rb_depth, iRB_gate_no=-1, debug=False, matrix_ref=matrix_ref):
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
            gate_list.append(self.gate2no(inverse_gate_symbol[max_index-1]))
        if debug:
            print(gate_list)
            print(max_index)
        return gate_list

    def random_pick_from_lists(self, a):
        # Initialize index pointers for each sublist
        indices = [0] * len(a)
        # Total number of elements to pick
        total_elements = sum(len(sublist) for sublist in a)
        # Output list
        b = []
        # List to track which sublist each element was picked from
        origins = []

        # Continue until all elements are picked
        pick_no = 0
        while len(b) < total_elements:
            # Find all sublists that have elements left to pick
            available = [i for i in range(len(a)) if indices[i] < len(a[i])]
            # Randomly select one of the available sublists

            chosen_list = random.choice(available)
            # chosen_list = pick_no % len(a)
            # Pick the element from the chosen sublist and append to b
            b.append(a[chosen_list][indices[chosen_list]])
            # Record the origin of the picked element
            origins.append(chosen_list)
            # Update the index pointer for the chosen sublist
            indices[chosen_list] += 1
            pick_no += 1

        return b, origins
    def round_robin_pick(self, a):
        # Calculate the total number of elements
        total_elements = sum(len(lst) for lst in a)
        
        # Initialize indices for each list
        indices = [0] * len(a)
        
        # Output list
        b = []
        # List to track which sublist each element was picked from
        origins = []
        
        # Continue until all elements are picked
        pick_no = 0
        while len(b) < total_elements:
            # Find all sublists that have elements left to pick
            available = [i for i in range(len(a)) if indices[i] < len(a[i])]
            
            # Use round-robin approach to select the next list
            chosen_list = pick_no % len(a)
            
            # If the chosen list has elements left, pick the element
            if indices[chosen_list] < len(a[chosen_list]):
                # Pick the element from the chosen sublist and append to b
                b.append(a[chosen_list][indices[chosen_list]])
                # Record the origin of the picked element
                origins.append(chosen_list)
                # Update the index pointer for the chosen sublist
                indices[chosen_list] += 1
            
            pick_no += 1
        
        return b, origins

    def find_unique_elements_and_positions(self, lst):
        unique_elements = []
        first_positions = {}
        last_positions = {}

        # Iterate over the list to find the first and last occurrence of each element
        for idx, elem in enumerate(lst):
            # Update the last position for every occurrence
            last_positions[elem] = idx
            # If the element is encountered for the first time, record its first position
            if elem not in first_positions:
                unique_elements.append(elem)
                first_positions[elem] = idx

        # Create lists of the positions in the order of unique elements
        first_pos_list = [first_positions[elem] for elem in unique_elements]
        last_pos_list = [last_positions[elem] for elem in unique_elements]

        return unique_elements, first_pos_list, last_pos_list

    def gate2time(self, t0, gate_name, gate_t_length):

        # for each middle/final gate: M1-Si-->sync(10ns)-->f0g1-->sync(10ns)-->ef pi pulse-->sync(10ns)-->qubit rb gate-->sync(10ns)-->ef pi pulse-->sync(10ns)-->f0g1-->sync(10ns)-->M1-Si-->sync(10ns)
        # for each first gate: qubit rb gate-->sync(10ns)-->ef pi pulse-->sync(10ns)-->f0g1-->sync(10ns)-->M1-Si-->sync(10ns)
        # t0: 1*7 list keeps tracking the last completed gate on each storage mode

        # return 
        # tfinal: final time spot, it is a 1*7 list corresponding to previous last operation time (the end time) on Si

        sync_t = 0 #4   # 4 cycles of sync between pulses
        tfinal = []
        for i in t0:
            tfinal.append(i)

        if gate_name[1] == 'M' or gate_name[1] == 'L':

            sync_total = sync_t*7  # total time for sync
            f0g1_total = gate_t_length['f0g1_length']*2
            ef_total = gate_t_length['pi_ef_length']*2
            if int(gate_name[0]) in [1,2]:
                ge_total = gate_t_length['pi_ge_length']
            else:
                ge_total = gate_t_length['hpi_ge_length']

            m1si_name = 'M1S'+gate_name[-1]+'_length'
            M1Si_total = gate_t_length[m1si_name]*2

            tfinal[int(gate_name[2])-1] = sync_total+f0g1_total+ef_total+ge_total+M1Si_total + max(t0)
            gatelength = sync_total+f0g1_total+ef_total+ge_total+M1Si_total
        else:  # first pulse is different

            sync_total = sync_t*4  # total time for sync
            f0g1_total = gate_t_length['f0g1_length']*1
            ef_total = gate_t_length['pi_ef_length']*1
            if int(gate_name[0]) in [1,2]:
                ge_total = gate_t_length['pi_ge_length']
            else:
                ge_total = gate_t_length['hpi_ge_length']

            m1si_name = 'M1S'+gate_name[-1]+'_length'
            M1Si_total = gate_t_length[m1si_name]*1

            tfinal[int(gate_name[2])-1] = sync_total+f0g1_total+ef_total+ge_total+M1Si_total + max(t0)
            gatelength = sync_total+f0g1_total+ef_total+ge_total+M1Si_total

        return tfinal, gatelength

    def RAM_rb(self, storage_id, depth_list, cycles2us = 0.0023251488095238095):

        """
        Multimode RAM RB generator with VZ speicified
        Gate set = {+-X/2, +-Y/2, X, Y}
        storage_id: a list specifying the operation on storage i, eg [1,3,5] means operation on S1, S3,S5
        depth_list: a list specifying the individual rb depth on corresponding storage specified in storage_id list

        depth_list and storage_id should have the same length

        phase_overhead: a 7*7 matrix showing f0g1+[M1S1, ..., M1S7] pi swap's phase overhead to [S1, ..., S7] (time independent part). 
        phase_overhead[i][j] is M1-S(j+1) swap's+f0g1 phase overhead on M1-S(i+1) (only half of it, a V gate is 2*phase_overhead)

        phase_freq: a 1*7 list showing [M1S1, ..., M1S7]'s time-dependent phase accumulation rate during idle sessions.
        gate_t_length: a dictionary ,all in cycles
            'pi_ge_length': in cycles
            'hpi_ge_length': in cycles
            'pi_ef_length': in cycles
            'f0g1_length': in cycles
            'M1S1_length': in cycles
            'M1S2_length': in cycles
            'M1S3_length': in cycles
            'M1S4_length': in cycles
            'M1S5_length': in cycles
            'M1S6_length': in cycles
            'M1S7_length': in cycles

        Each storage operation has two parts:
        if it is not the initial gate, extract information, gates on qubit, then store information
        The initial gate only perform gate on qubit, then store information
        The last gate only extract information, gate on qubit and check |g> population

        gate_list: a list of strings, each string is gate_id+'F/L/M'+storage_id. 'F': first gate on the storage, 'L': last gate on the storage, 'M': any other gate between F and L
        vz_phase_list: virtual z phase (in degree)

        """
        phase_overhead = self.cfg.device.storage.idling_phase
        phase_freq = self.cfg.device.storage.idling_freq
        gate_t_length = self.gate_t_length

        # generate random gate_list for individual storage 
        individual_storage_gate = []
        for ii in range(len(depth_list)):
            individual_storage_gate.append(self.generate_sequence(depth_list[ii]))
            # invi
        stacked_gate, origins = self.round_robin_pick(individual_storage_gate)
        for ii in range(len(origins)):
            # convert origins to storage mode id
            origins[ii] = storage_id[origins[ii]]

        # check first or last element position
        unique_elements, first_pos_list, last_pos_list = self.find_unique_elements_and_positions(origins)


        # convert origins+stacked gate to gate_list form

        #cycles2us = self.cycles2us(1)   # coefficient
        # print('cycles2us ', cycles2us)

        gate_list = []
        vz_phase_list = []  # all in deg, length = gate_list
        vz_phase_current = [0]*7  # all in deg, position maps to different 7 storages
        t0_current = [0]*7  # initialize the time clock, each storage mode has its own clock
        for ii in range(len(stacked_gate)):
            gate_name = str(stacked_gate[ii])
            gate_symbol = 'M'
            vz = 0
            
            if ii in first_pos_list: 
                gate_symbol = 'F'
            if ii in last_pos_list: gate_symbol = 'L'

            gate_name = gate_name+gate_symbol+str(origins[ii])
            # calculate gate time (to be updated properly with experiment.cfg)
            t0_after, gate_length = self.gate2time(t0_current, gate_name, gate_t_length)

            gate_list.append(gate_name)
            

            

            # calculate vz_phase correction using t0_current and t0_after
            # operation is int(gate_name[-1])
            # overhead phase is overhead[0,1,2,3,4,5,6][int(gate_name[-1])-1]
            tophase = [0]*7
            if ii in first_pos_list:  # first gate 1 overhead
                # update 1* overhead
                # time independent phase 
                for i in range(7):
                    tophase[i] = phase_overhead[i][int(gate_name[-1])-1]   # in deg
                # to others that already applied, no need for self-correction, set self phase to 0
                tophase[int(gate_name[-1])-1] = 0
                vz_phase_current[int(gate_name[-1])-1] = 0
                # print(tophase)
            else:  # other case 2 overheads
                # time independent phase
                for i in range(7):
                    tophase[i] = phase_overhead[i][int(gate_name[-1])-1]*2   # in deg
                # time dependent phase
                tophase[int(gate_name[-1])-1] += phase_freq[int(gate_name[-1])-1]*(50*0+t0_after[int(gate_name[-1])-1]-t0_current[int(gate_name[-1])-1]-gate_length)*cycles2us/np.pi*180*2*np.pi   # in deg
                # print(t0_after[int(gate_name[-1])-1])
                # print(t0_current[int(gate_name[-1])-1])

            for i in range(7):
                vz_phase_current[i] += tophase[i]

            vz_phase_list.append(vz_phase_current[int(gate_name[-1])-1])

            # update the clock
            t0_current = t0_after
            # print(t0_current)

        vz_phase_list = np.array(vz_phase_list) % 360
        
        return gate_list, list(vz_phase_list), origins
        

class MMRBAveragerProgram(AveragerProgram, MM_rb_base):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)
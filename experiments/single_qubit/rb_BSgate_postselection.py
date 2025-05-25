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
from MM_dual_rail_base import *

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

class SingleBeamSplitterRBPostselectionrun(MMDualRailAveragerProgram):
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
        self.initialize_beam_splitter_pulse()

        # -------set up pulse parameters for measurement pulses -------

        self.parity_pulse_for_custom_pulse = self.get_parity_str(man_mode_no = 1, return_pulse = True, second_phase = 0 )
        
        self.f0g1_for_custom_pulse = self.get_prepulse_creator([['man', 'M1' , 'pi',0 ]]).pulse.tolist()
        self.ef_for_custom_pulse = self.get_prepulse_creator([['qubit', 'ef', 'pi', 0]]).pulse.tolist()
        self.ge_for_custom_pulse = self.get_prepulse_creator([['qubit', 'ge', 'pi', 0]]).pulse.tolist()

        # self.wait_all(self.us2cycles(0.2))
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
            # print(f'Playing BS gate with phase {phase}')
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
            self.active_reset( man_reset= self.cfg.expt.rb_man_reset, storage_reset= self.cfg.expt.rb_storage_reset, 
                              ef_reset = True, pre_selection_reset = True, prefix = 'base')

        # self.wait_all(self.us2cycles(0.2))
        # self.sync_all(self.us2cycles(0.2))

        # prepulse 
        if cfg.expt.prepulse:
            prepulse_for_custom_pulse = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse).pulse.tolist() # pre-sweep-pulse is not Gate based
            self.custom_pulse(cfg, prepulse_for_custom_pulse, prefix='pre10')#, advance_qubit_phase=self.vz)
            
        # prepare a photon in manipulate cavity 
        self.custom_pulse(cfg, self.ge_for_custom_pulse, prefix='pre11')#
        self.custom_pulse(cfg, self.ef_for_custom_pulse, prefix='pre12')#
        self.custom_pulse(cfg, self.f0g1_for_custom_pulse, prefix='pre13')#
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

        # store photon in storage 
        # self.play_bs_gate(cfg, phase=0, times = 2, wait=wait_bool)
        # self.cfg.expt.running_list = [4,6]   #[3,5]
        for idx, ii in enumerate(self.cfg.expt.running_list):
            wait_bool = False
            if idx%self.cfg.expt.gates_per_wait == 0: # only wait after bs pulse every 10 gates
                wait_bool = True

            print(ii)
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
                    

           
        self.sync_all()

        # ------------------Measurement------------------
        # Alice's meas
        self.measure_parity_or_f0g1(common_prefix = 'alice_meas', measure_wrapper_bool = False)
        self.sync_all(self.us2cycles(0.05))
        if cfg.expt.reset_qubit_via_active_reset_after_first_meas:
            self.active_reset(man_reset= False, storage_reset= False, ef_reset = False, pre_selection_reset = False, prefix = 'post_meas') #????
        else: 
            self.measure(
                pulse_ch=self.res_chs[qTest],
                adcs=[self.adc_chs[qTest]],
                adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                wait=True,
                syncdelay=self.us2cycles(self.cfg.expt.postselection_delay)
            )
        self.sync_all()


        # parity meas to reset qubit 
        if self.cfg.expt.reset_qubit_after_parity: 
            parity_str = self.parity_pulse_for_custom_pulse
            self.custom_pulse(cfg, parity_str, prefix='parity_post_meas1')

        # Post Selection measurement (Bob's meas)
        
        if not cfg.expt.parity_meas: 
            self.custom_pulse(cfg, self.ef_for_custom_pulse, prefix='ef_for_prepping_bob')
            self.custom_pulse(cfg, self.f0g1_for_custom_pulse, prefix='f0g1_for_prepping_bob')
        self.play_bs_gate(cfg, phase=0, times = 2, wait=True)
        self.measure_parity_or_f0g1(common_prefix = 'bob_meas')
    
    def measure_parity_or_f0g1(self, common_prefix = 'bob', measure_wrapper_bool = True):
        print(f'measure parity or f0g1 with prefix {common_prefix}')
        cfg = self.cfg
        if cfg.expt.parity_meas: 
            self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_' + common_prefix)
        else: 
            self.custom_pulse(cfg, self.f0g1_for_custom_pulse, prefix='f0g1'+ common_prefix)
            self.custom_pulse(cfg, self.ef_for_custom_pulse, prefix='ef'+ common_prefix)
        if measure_wrapper_bool:
            self.measure_wrapper()

    def prep_bob_meas(self, common_prefix = 'first'): 
        """
        Prepare the measurement for Bob mode
        """
        cfg = self.cfg
        if not cfg.expt.parity_meas: 
            self.custom_pulse(cfg, self.ef_for_custom_pulse, prefix='ef_'+ common_prefix)
            self.custom_pulse(cfg, self.f0g1_for_custom_pulse, prefix='f0g1_'+ common_prefix)
        self.play_bs_gate(cfg, phase=0, times = 2, wait=True)
        self.measure_wrapper()

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

        
        # ================= #
        # Get single shot calibration for all qubits
        # ================= #

        # g states for q0
        data=dict()
        data['Ig'] = []
        data['Qg'] = []
        data['Ie'] = []
        data['Qe'] = []
        data['fids'] = []
        data['angle'] = []
        data['thresholds'] = []
        data['confusion_matrix'] = []
        
        
        if self.cfg.expt.single_shot_bef_expt: 
            sscfg = self.cfg
            sscfg.expt.reps = sscfg.expt.singleshot_reps
            print('sscfg active reset ' + str(sscfg.expt.active_reset))
        

            # Ground state shots
            # cfg.expt.reps = 10000
            sscfg.expt.qubit = 0
            sscfg.expt.rounds = 1
            sscfg.expt.pulse_e = False
            sscfg.expt.pulse_f = False
            # print(sscfg)

            
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

        
        self.cfg.expt.reps = self.cfg.expt.rb_reps
        sequences = []
        for var in tqdm(range(self.cfg.expt.variations)):   # repeat each depth by variations
            print(f'Running variation {var+1} of {self.cfg.expt.variations}')
            self.cfg.expt.running_list =  generate_sequence(self.cfg.expt.rb_depth, iRB_gate_no=self.cfg.expt.IRB_gate_no)
            sequences.append(self.cfg.expt.running_list)
            
            #for ram prepulse 
            if self.cfg.expt.ram_prepulse_strs is None: 
                if self.cfg.expt.ram_prepulse[0]:
                    self.cfg.expt.prepulse = True
                    dummy = MM_dual_rail_base( cfg=self.cfg)
                    prepulse_strs = [dummy.prepulse_str_for_random_ram_state(num_occupied_smodes=self.cfg.expt.ram_prepulse[1],
                                                skip_modes=self.cfg.expt.ram_prepulse[2])
                                                for _ in range(self.cfg.expt.ram_prepulse[3])] 
                            #  for _ in range(self.cfg.expt.ram_prepulse[3])]
                else: 
                    self.cfg.expt.prepulse = False
                    prepulse_strs = [[None]]
                
            else: 
                prepulse_strs = self.cfg.expt.ram_prepulse_strs

            print(prepulse_strs)
            for prepulse_str in prepulse_strs:
            # print reps, rounds, etc
            #print(f'reps: {self.cfg.expt.reps}, rounds: {self.cfg.expt.rounds}, pulse_e: {self.cfg.expt.pulse_e}, pulse_f: {self.cfg.expt.pulse_f}')
                self.cfg.expt.pre_sweep_pulse = prepulse_str
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
        # seuqence = [[0,1], [1]]
        data['sequences'] = self.pad_sequences_to_homogeneous(sequences )  
        # data['expt_cfg'] = deepcopy(self.cfg.expt)
        self.data = data

        return data
    
    def pad_sequences_to_homogeneous(self, sequences):
        """
        Pads a list of 1D sequences (lists or arrays) so that all sequences have the same length,
        using np.nan for padding. Returns a list of lists.

        Example:
            Input: [[0,1], [1], [2]]
            Output: [
                [0, 1],
                [1, np.nan],
                [2, np.nan]
            ]
        """
        if not sequences:
            return []

        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            seq = list(seq)
            pad_width = max_len - len(seq)
            if pad_width > 0:
                seq = seq + [np.nan] * pad_width
            padded_sequences.append(seq)
        return padded_sequences
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
# ===================================================================== #

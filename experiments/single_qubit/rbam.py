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

import fitting.fitting as fitter
from MM_rb_base import * #MMRBAveragerProgram

"""
Single qubit RB sequence generator
Gate set = {I, +-X/2, +-Y/2, +-Z/2, X, Y, Z}
"""

class MultiRBAMrun(MMRBAveragerProgram):
    """
    RB program for single qubit gates
    """

    def __init__(self, soccfg, cfg):
        # gate_list should include the total gate!
        #self.gate_list =  cfg.expt.running_list
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        super().__init__(soccfg, cfg)

    def initialize(self):
        self.MM_base_initialize()

        if self.cfg.expt.preloaded_pulses: 
            self.initialize_pulse_registers()

    def play_ge_pulse(self, phase=0, times =1, idx = 0): 
        '''
        play gate based ge pulse 
        '''
        pulse_str = [['qubit', 'ge', 'hpi', phase + self.vz]]
        # print(times)
        creator = self.get_prepulse_creator(pulse_str)
        self.custom_pulse_with_preloaded_wfm(self.cfg,creator.pulse.tolist(), prefix='Rb ge pulse' +  str(idx)) # prefix to make it unique
        for jdx in range(times -1):
            self.custom_pulse_with_preloaded_wfm(self.cfg,creator.pulse.tolist(), prefix='Rb ge pulse' +  str(idx) + str(jdx), same_qubit_pulse= True) # prefix to make it unique
        #print('ge pulse played ' + str(creator.pulse.tolist()))
    
    
        
    
    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = 0 

        # phase reset
        # self.reset_and_sync()
        if self.cfg.expt.rb_active_reset: 
            self.active_reset(man_reset=self.cfg.expt.rb_man_reset, storage_reset=self.cfg.expt.rb_storage_reset)
        #==================================================================== #
        self.vz = 0   # virtual Z phase in degree (ge phase)
        # self.vz_ef = 0   # virtual Z phase in degree (ef phase)
        #==================================================================== #
        # self.vz_f0g1 = 0   # virtual Z phase in degree (f0g1 phase)
        # qTest = 0

        # cfg.expt.rb_gate_list = ['0F1', '0F2'] + list(np.concatenate([['0M1', '0M2'] for _ in range(19)]))
        #self.cfg.expt.phase_list = [0,0]
        
        # self.play_ge_pulse(phase=0, times=2, idx=0)
    
        prev_mode_no = 1
        for idx, gate_string in enumerate(self.cfg.expt.rb_gate_list):
            ##prepulse before each gate
            # print(gate_string)

            ii = int(gate_string[0])
            gate_loc = gate_string[1]
            mode_no = int(gate_string[2])


            # # load storage mode into ge subspace
            if gate_loc != 'F':
                pre_pulse = self.compound_storage_gate(input = False, storage_no=mode_no, man_no = 1)
                creator = self.get_prepulse_creator(pre_pulse)
                # print(creator.pulse.tolist())
                if self.cfg.expt.ref_rb: 
                    creator = self.get_prepulse_creator(pre_pulse[1:]) # skip first pulse in storage retrieval which is M1-Si

                if mode_no == prev_mode_no: # for storage optimization
                    self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb prepulse' + str(idx), 
                                                         same_storage=True, storage_no=mode_no)#, same_qubit_pulse=True)#, same_qubit_pulse= True) # ef is same
                else: self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb prepulse' + str(idx)
                                                           , storage_no=mode_no)#, same_qubit_pulse= True)
                self.vz = self.cfg.expt.phase_list[idx] #- 90
                # print('finished prepulse')
        
            # gate in ge subspace
            if ii == 0:
                pass
            if ii == 1:  #'X'               
                self.play_ge_pulse(phase=0, times=2, idx=idx)

            if ii == 2:  #'Y'
                self.play_ge_pulse(phase=-90, times=2, idx=idx)

            if ii == 3:  #'X/2'
                self.play_ge_pulse(phase=0, times=1, idx=idx)

            if ii == 4:  #'Y/2'
                self.play_ge_pulse(phase=-90, times=1, idx=idx)

            if ii == 5:  #'-X/2'
                self.play_ge_pulse(phase=-180, times=1, idx=idx)

            if ii == 6:  #'-Y/2'
                self.play_ge_pulse(phase=90, times=1, idx=idx)
            # print('finished qubit')

            # put ge state into storage mode

            post_pulse = self.compound_storage_gate(input = True, storage_no=mode_no, man_no = 1)
            creator = self.get_prepulse_creator(post_pulse)
            if self.cfg.expt.ref_rb: 
                    creator = self.get_prepulse_creator(post_pulse[:-1]) # skip last pulse in storage insertion which is M1-Si

            if gate_loc != 'F' and self.cfg.expt.preloaded_pulses: # for storage optimization
                self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb postpulse' + str(idx), 
                                                     same_storage=True, storage_no=mode_no, same_qubit_pulse= False)
            # elif prev_mode_no == mode_no and self.cfg.expt.preloaded_pulses: # for storage optimization
            #     self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb postpulse' + str(idx), same_storage=True)
            else: 
                self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb postpulse' + str(idx), storage_no=mode_no, same_qubit_pulse= False)
            # print('finished post pulse')
            
            prev_mode_no = mode_no
            # tunable idling time for test
            # self.sync_all(50)

                
        #==================================================================== #
        # picking a specific mode to measure ; note virtual phase accrued on mode to be measured is not important as we will not do any ge gates
        if self.cfg.expt.measure_mode_no == 0: # for reference measurement
            assert not cfg.expt.parity_meas, 'parity measurement for  reference measurement is not yet tested'
            measure_pulse = self.compound_storage_gate(input = False, storage_no=1, man_no = 1)[1:] # skip the M1-S1 pulse 
        else: 
            measure_pulse = self.compound_storage_gate(input = False, storage_no=self.cfg.expt.measure_mode_no, man_no = 1)#[:-1]
        if cfg.expt.parity_meas: 
            measure_pulse = measure_pulse[:-2]   # no need for last ef and f0g1 pulses, swap Si to M1

        # print('measure pulse ' + str(measure_pulse))
        creator = self.get_prepulse_creator(measure_pulse)   # other case:L Si-->M1, M1-->f, f-->e, and already at the ground state

        self.custom_pulse_with_preloaded_wfm(self.cfg, creator.pulse.tolist(), prefix='Rb measure pulse1')
        #==================================================================== #
        if cfg.expt.parity_meas: 
            parity_str = [['qubit', 'ge', 'hpi', 0],
                            ['qubit', 'ge', 'parity_M1', 0 ],
                          ['qubit', 'ge', 'hpi', 180]]
            creator = self.get_prepulse_creator(parity_str)
            self.custom_pulse(self.cfg, creator.pulse.tolist(), prefix='Rb parity pulse2')

        #================================Post Selection ==================================== #
        
           
        if self.cfg.expt.rb_post_select:
            sync_delay = self.cfg.expt.postselection_delay
        self.play_measure(sync_delay = sync_delay)

        if self.cfg.expt.rb_post_select: 
            self.play_ge_pulse(phase=0, times=2, idx=-1)
            self.play_measure()


    def play_measure(self, sync_delay = None):
        '''
        sync delay in us 
        plays measurement pulse and collects data
        '''
        qTest = 0
        if sync_delay is None: sync_delay = self.cfg.device.readout.relax_delay[0]
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=self.cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(sync_delay),
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
class MultiRBAM(Experiment):
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
                                                         active_reset=self.cfg.expt.active_reset, threshold = self.cfg.device.readout.threshold[q_ind],
                                                         readout_per_round=sscfg_readout_per_round)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        data['confusion_matrix'] = confusion_matrix


        print(f'ge fidelity (%): {100*fids[0]}')
        print(f'rotation angle (deg): {angle}')
        print(f'threshold ge: {thresholds[0]}')

        self.cfg.expt.measure_mode_list = self.cfg.expt.mode_list
        if self.cfg.expt.measure_all_modes: 
            self.cfg.expt.measure_mode_list = [i+1 for i in range(3)]
        if self.cfg.expt.ref_rb: 
            self.cfg.expt.measure_mode_list = [0] + self.cfg.expt.measure_mode_list # for measuring manipulate 1 mode

        data['Idata'] = [[] for _ in range(len(self.cfg.expt.measure_mode_list))]
        data['Qdata'] = [[] for _ in range(len(self.cfg.expt.measure_mode_list))]

        #sequences = np.array([[0], [1]])#[1,1,1,1], [2,2,2,2],  [1,2,1,1], [1,2,2,2], [1,1,2,1]])
        #for var in sequences:
        self.cfg.expt.reps = self.cfg.expt.rb_reps
        # data['running_lists'] = []
        read_num =1
        if self.cfg.expt.rb_active_reset: read_num = 4
        if self.cfg.expt.rb_post_select: read_num +=1
        
        for _ in tqdm(range(self.cfg.expt.variations)):   # repeat each depth by variations

            
            dummy = MM_rb_base( cfg=self.cfg)
            gate_list, vz_phase_list, origins = dummy.RAM_rb(self.cfg.expt.mode_list, self.cfg.expt.depth_list)


            self.cfg.expt.rb_gate_list = gate_list
            self.cfg.expt.phase_list = vz_phase_list
            # self.cfg.expt.origins = origins
            # print('gate list ' + str(gate_list))
            # print('phase list ' + str(vz_phase_list))

            


            for mode_idx, mode in enumerate(self.cfg.expt.measure_mode_list):
                self.cfg.expt.measure_mode_no = mode
                rb_shot = MultiRBAMrun(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rb_shot
                avgi, avgq = rb_shot.acquire(
                    self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug,
                            readouts_per_experiment=read_num) #,save_experiments=np.arange(0,5,1))
                II, QQ = rb_shot.collect_shots_rb(read_num)
                data['Idata'][mode_idx].append(II)
                data['Qdata'][mode_idx].append(QQ)
                # print(rb_shot)
        #data['running_lists'] = running_lists   
        #print(self.prog)
            
        self.data = data

        return data
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    
    def get_gate_list(self):
        dummy = MultiRBAMrun(soccfg=self.soccfg, cfg=self.cfg)
        print('finished running dummty ')
# ===================================================================== #

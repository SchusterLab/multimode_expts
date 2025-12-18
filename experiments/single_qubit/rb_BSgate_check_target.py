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

import fitting.fitting as fitter
from MM_dual_rail_base import *

"""
Single Beam Splitter RB sequence generator
Gate set = {+-X/2, +-Y/2, X, Y}
"""


class SingleBeamSplitterRB_check_target_prog(MMDualRailAveragerProgram):
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
        if cfg.expt.prep_man_photon: 
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
        
        if cfg.expt.postpulse:
            postpulse_for_custom_pulse = self.get_prepulse_creator(cfg.expt.post_sweep_pulse).pulse.tolist() # pre-sweep-pulse is not Gate based
            self.custom_pulse(cfg, postpulse_for_custom_pulse, prefix='post')#, advance_qubit_phase=self.vz)
                        

           
        # self.sync_all()
 
        # if cfg.expt.parity_meas: 
        #     self.custom_pulse(cfg, self.parity_pulse_for_custom_pulse, prefix='parity_meas1')
        # else: 
        #     self.custom_pulse(cfg, self.f0g1_for_custom_pulse, prefix='f0g1_meas1')
        #     self.custom_pulse(cfg, self.ef_for_custom_pulse, prefix='ef_meas1')

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
class SingleBeamSplitterRB_check_target(Experiment):
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
        data = dict()
        data['Ig'] = []
        data['Qg'] = []
        data['Ie'] = []
        data['Qe'] = []
        if self.cfg.expt.calibrate_single_shot:

            # g states for q0
            #data=dict()
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
        else: 
            fids = [0]
            thresholds = [0]
            angle = [0]
            confusion_matrix = [0]
            
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
        dummy = MM_dual_rail_base( cfg=self.cfg)
        # data['running_lists'] = []

        self.cfg.expt.rb_times = [] # for analysis
        self.cfg.expt.bs_gate_nums = [] # for analysis

        for var in tqdm(range(self.cfg.expt.variations)):   # repeat each depth by variations
            #rb sequence
            self.cfg.expt.running_list =  dummy.generate_sequence(self.cfg.expt.rb_depth, iRB_gate_no=self.cfg.expt.IRB_gate_no)
            # Need to calculate total time taken up by running list and consequently, the phase on the second pi/2 pulse
            
            rb_time, bs_gate_num = dummy.get_total_time_from_running_list( running_list=self.cfg.expt.running_list,
                                                                            bs_time=self.cfg.expt.bs_para[2])
            phase = self.cfg.expt.wait_freq*rb_time * 360 # convert to degrees
            self.cfg.expt.post_sweep_pulse[-1][-1] = phase
            # for analysis
            self.cfg.expt.rb_times.append(rb_time)
            self.cfg.expt.bs_gate_nums.append(bs_gate_num)
            
            # print(self.cfg.expt.rb_time)

            # #for ram prepulse 
            # if self.cfg.expt.ram_prepulse_strs is None: 
            #     if self.cfg.expt.ram_prepulse[0]:
            #         self.cfg.expt.prepulse = True
            #         #dummy = MM_dual_rail_base( cfg=self.cfg)
            #         prepulse_strs = [dummy.prepulse_str_for_random_ram_state(num_occupied_smodes=self.cfg.expt.ram_prepulse[1],
            #                                                                 skip_modes=self.cfg.expt.ram_prepulse[2])
            #                                                                 for _ in range(self.cfg.expt.ram_prepulse[3])] 
            #                         #  for _ in range(self.cfg.expt.ram_prepulse[3])]
            #     else: 
            #         self.cfg.expt.prepulse = False
            #         prepulse_strs = [[None]]
            # else: 
            #     prepulse_strs = self.cfg.expt.ram_prepulse_strs

            # for prepulse_str in prepulse_strs:
        
            # self.cfg.expt.pre_sweep_pulse = prepulse_str
            rb_shot = SingleBeamSplitterRB_check_target_prog(soccfg=self.soccfg, cfg=self.cfg)
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

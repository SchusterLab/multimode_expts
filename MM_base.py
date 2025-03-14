
from qick import *
import numpy as np
from qick.helpers import gauss
import time
from slab import AttrDict
from dataset import * 
from dataset import storage_man_swap_dataset
import matplotlib.pyplot as plt
import random

        
class MM_base(): 
    def __init__(self, cfg):
        '''
        Contains functions that are useful for both averager and raverager programs
        '''
        self.cfg=AttrDict(cfg)
        
    def initialize_idling_dataset(self): 
        '''
        Create a dictionary that will keep a record of idling times

        dict= {'key = transition' : value = []} 
        '''
    
    def get_prepulse_creator(self,  sweep_pulse = None):
        '''returns an instance  of  prepulse creator class '''
        #config_file = self.cfg
        creator = prepulse_creator2(self.cfg, self.cfg.device.storage.storage_man_file)

        if sweep_pulse is not None:
            for pulse_idx in range(len(sweep_pulse)):
                # for each pulse 
                #print(sweep_pulse)
                pulse_param = list(sweep_pulse[pulse_idx][1:])
                eval(f"creator.{sweep_pulse[pulse_idx][0]}({pulse_param})")

        return creator
    
    def compound_storage_gate(self, input = True, storage_no = 1, man_no = 1): 
        '''
        input: if True, then the storage gate is on, else output to storage mode

        input from ge state 

        returns gate based prepulse string 
        '''
        prepulse_str = [ ['qubit', 'ef', 'pi',0],
                    ['man', 'M1' , 'pi',0 ], 
                    ['storage', 'M' + str(man_no) + '-S' + str(storage_no), 'pi',0]]
        if not input: 
            prepulse_str = prepulse_str [::-1]
            for idx in range(len(prepulse_str)): 
                prepulse_str[idx][-1] = 180
        return prepulse_str 
    
    # for f0g1 randomized benchmarking 

    # def convert_rb_sequence_to_gates(self, rb_sequences): 
    #     '''
    #     rb_sequences: []
    #     '''

    
    def MM_base_initialize(self): 
        '''
        Shared Initialize method
        '''
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
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
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
        
        self.hpi_ge_gain = cfg.device.qubit.pulses.hpi_ge.gain[qTest]

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

        self.initialize_waveforms()

        # define ramp pulses
        # self.add_gauss(ch=self.qubit_chs[qTest], name="ramp_up_ge", sigma=self.pi_sigma_ramp, length=self.pi_sigma_ramp*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="ramp_up_ef", sigma=self.pief_sigma_ramp, length=self.pief_sigma_ramp*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="ramp_up_hge", sigma=self.hpi_sigma_ramp, length=self.hpi_sigma_ramp*4)


        # ---------- readout pulse parameters -----------
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # self.wait_all(self.us2cycles(0.2))
        self.sync_all(self.us2cycles(0.2))
        
    def get_total_time(self, test_pulse, gate_based = False, cycles = False, cycles2us = 0.0023251488095238095):
        '''
        Takes in pulse str of form 
        # [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]]s
        '''
        if gate_based: 
            test_pulse = self.get_prepulse_creator(test_pulse).pulse
            # print(test_pulse)
        t = 0 
        for i in range(len(test_pulse[0])):
            if test_pulse[5][i] == 'g' or test_pulse[5][i] == 'gauss' or test_pulse[5][i] == 'gaussian':
                t += test_pulse[-1][i] * 4
            elif test_pulse[5][i] == 'flat_top' or test_pulse[5][i] == 'f':
                t += test_pulse[-1][i] * 6 + test_pulse[2][i]
            t+= 0.01 # 10ns delay
        if cycles: 
            # QickConfig(im[yaml_cfg['aliases']['soc']].get_cfg())
            return int(round(t / cycles2us))
        return t 

    def initialize_waveforms(self): 
        '''
        Initialize waveforms for ge, ef_new, f0g1 and sidebands
        '''
        cfg = self.cfg 
        qTest = 0 

        # --------------------qubit pulse parameters 
        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.pief_sigma_g = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef_ftop.sigma[0], gen_ch=self.qubit_chs[qTest])

        # define all 2 different pulses
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pi_sigma, length=self.pi_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit_ge", sigma=self.hpi_sigma, length=self.hpi_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef", sigma=self.pief_sigma_g, length=self.pief_sigma_g*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef_ftop", sigma=self.pief_sigma, length=self.pief_sigma*6) # this is flat top 
        # self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)

        # frequencies
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_ch[qTest])
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_ch[qTest])


        # --------------------f0g1 pulse parameters 
        self.pi_f0g1_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_f0g1.sigma[0], gen_ch=self.f0g1_ch[qTest])
        self.add_gauss(ch=self.f0g1_ch[qTest], name="pi_f0g1", sigma=self.pi_f0g1_sigma, length=self.pi_f0g1_sigma*6)

        # -------------------- M1-Si sideband parameter 
        self.pi_m1_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_m1si.sigma[0], gen_ch=self.flux_low_ch[qTest])
        self.add_gauss(ch=self.flux_low_ch[qTest], name="pi_m1si_low", sigma=self.pi_m1_sigma, length=self.pi_m1_sigma*6)

        self.pi_m1_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_m1si.sigma[0], gen_ch=self.flux_high_ch[qTest])
        self.add_gauss(ch=self.flux_high_ch[qTest], name="pi_m1si_high", sigma=self.pi_m1_sigma, length=self.pi_m1_sigma*6)
        
        # # specify all f0g1 and M1-S1 sideband lengths for arb flat_top
        # pulse_str = [['man', 'M1', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.pi_f0g1_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_f0g1.sigma[0], gen_ch=self.f0g1_ch[qTest])
        # self.add_flat_top_gauss(ch=self.f0g1_ch[qTest], name="pi_f0g1_arb", sigma=self.pi_f0g1_sigma, 
        #                         length=self.us2cycles(pulse[2][0], gen_ch=self.f0g1_ch[qTest]))
        # print('f0g1 loaded')

        # self.pi_m1_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_m1si.sigma[0], gen_ch=self.flux_low_ch[qTest])
        # pulse_str = [['storage', 'M1-S1', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_low_ch[qTest], name="pi_m1s1_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]))
        # print('M1S1 loaded')
        # pulse_str = [['storage', 'M1-S2', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_low_ch[qTest], name="pi_m1s2_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]))
        # print('M1S2 loaded')
        # pulse_str = [['storage', 'M1-S3', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_low_ch[qTest], name="pi_m1s3_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]))
        # print('M1S3 loaded')
        # pulse_str = [['storage', 'M1-S4', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_low_ch[qTest], name="pi_m1s4_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_low_ch[qTest]))
        # print('M1S4 loaded')
        
        # self.pi_m1_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_m1si.sigma[0], gen_ch=self.flux_high_ch[qTest])
        # pulse_str = [['storage', 'M1-S5', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_high_ch[qTest], name="pi_m1s5_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_high_ch[qTest]))
        # print('M1S5 loaded')
        # pulse_str = [['storage', 'M1-S6', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_high_ch[qTest], name="pi_m1s6_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_high_ch[qTest]))
        # print('M1S6 loaded')
        # pulse_str = [['storage', 'M1-S7', 'pi', 0]]
        # pulse = self.get_prepulse_creator(pulse_str).pulse.tolist()
        # self.add_flat_top_gauss(ch=self.flux_high_ch[qTest], name="pi_m1s7_arb", sigma=self.pi_m1_sigma, 
        #                length=self.us2cycles(pulse[2][0], gen_ch=self.flux_high_ch[qTest]))
        # print('M1S7 loaded')
    

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs 

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[qTest]s[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[qTest]s[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)
        cfg = self.cfg
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

        # some dummy variables 
        qTest = 0
        self.f_q = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_cav = self.freq2reg(5000, gen_ch=self.man_ch[0])

        #initialize the phase to be 0
        self.set_pulse_registers(ch=self.qubit_ch[0], freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_ch[0])
        self.set_pulse_registers(ch=self.man_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_ch[0])
        # self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cav,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.storage_ch[0])
        self.set_pulse_registers(ch=self.flux_low_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_low_ch[0])
        self.set_pulse_registers(ch=self.flux_high_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_high_ch[0])
        self.set_pulse_registers(ch=self.f0g1_ch[0], freq=self.f_q,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.f0g1_ch[0])
        # self.wait_all(10)   
        self.sync_all(10)


    def custom_pulse(self, cfg, pulse_data, advance_qubit_phase = None, sync_zero_const = False, prefix='pre'): 
        '''
        Executes prepulse or postpulse

        # [[frequency], [gain], [length (us)], [phases], [drive channel],
        #  [shape], [ramp sigma]],
        #  drive channel=1 (flux low), 
        # 2 (qubit),3 (flux high),4 (storage),5 (f0g1),6 (manipulate),
        '''
        
        # print('------------------Beginning Custom Pulse----------------------------')
        # print(pulse_data)
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
            # print('here!')
            pulse_data[3] = [x + advance_qubit_phase for x in pulse_data[3]]

        # print(pulse_data)

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
                if pulse_data[5][jj] == "gaussian" or pulse_data[5][jj] == "gauss" or pulse_data[5][jj] == "g":
                    # print('gaussian') 
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch)
                    self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                       sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    # self.wait_all(self.us2cycles(0.01))
                    self.sync_all(self.us2cycles(0.01))
                    self.setup_and_pulse(ch=self.tempch, style="arb", 
                                     freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                     phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                     gain=pulse_data[1][jj], 
                                     waveform="temp_gaussian"+str(jj)+prefix)
                elif pulse_data[5][jj] == "flat_top" or pulse_data[5][jj] == "f":
                    # print('flat')
                    
                    self.pisigma_resolved = self.us2cycles(
                        pulse_data[6][jj], gen_ch=self.tempch)
                    if self.tempch==0 or self.tempch == 1 or self.tempch == 3: # f0r f0g1
                        self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                        sigma=self.pisigma_resolved, length=self.pisigma_resolved*6)
                    else:
                        self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                        sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                    self.sync_all(self.us2cycles(0.01))
                    self.setup_and_pulse(ch=self.tempch, style="flat_top", 
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch), 
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch), 
                                    gain=pulse_data[1][jj], 
                                    length=self.us2cycles(pulse_data[2][jj], 
                                                        gen_ch=self.tempch),
                                    waveform="temp_gaussian"+str(jj)+prefix)
                else:
                    # print('constant')
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
        # print('------------------End Custom Pulse----------------------------')

    def man_reset(self, man_idx, chi_dressed = True ): 
        '''
        Reset manipulate mode by swapping it to lossy mode 

        chi_dressed: if man freq shifted due to pop in qubit e, f states. 
        '''
        qTest = 0
        cfg=AttrDict(self.cfg)
        M_curr_lossy = cfg.device.active_reset.M_lossy[man_idx]
        chis = [0] # cfg.device.active_reset.chis
        N = 0
        if chi_dressed: 
            chis = cfg.device.active_reset.chis
            N = M_curr_lossy[4] 
        #print(M_curr_lossy)
        ### prepare waveform 
        sideband_sigma_high = self.sideband_sigma_high = self.us2cycles(
            cfg.device.active_reset.M1_S_sigma, gen_ch=self.flux_high_ch[qTest]) 
        self.add_gauss(ch=self.flux_high_ch[qTest], name="ramp_high",# + str(man_idx),
                       sigma=self.sideband_sigma_high, length=self.sideband_sigma_high*4)
        # self.wait_all(self.us2cycles(0.1))
        ### pulse 
        self.sync_all(self.us2cycles(0.1))

        for n in range(0, N + 1): 
            for chi in chis: 
                freq_chi_shifted = M_curr_lossy[0] - (n * chi) 
                self.set_pulse_registers(ch=self.flux_high_ch[qTest], 
                                        freq=self.freq2reg(freq_chi_shifted,gen_ch=self.flux_high_ch[qTest]), 
                                        style="flat_top",
                                        phase=self.deg2reg(0),
                                        length=self.us2cycles(M_curr_lossy[2]),
                                        gain=M_curr_lossy[1], waveform="ramp_high" )
                self.pulse(ch=self.flux_high_ch[qTest])
                # self.wait_all(self.us2cycles(0.025))
                self.sync_all(self.us2cycles(0.025))
        # self.wait_all(self.us2cycles(0.25))
        self.sync_all(self.us2cycles(M_curr_lossy[3]))
    

    def man_stor_swap(self, man_idx, stor_idx): 
        '''
        Perform Swap between manipulate mode and  storage mode 
        '''
        qTest = 0
        sweep_pulse = [['storage', 'M'+ str(man_idx) + '-' + 'S' + str(stor_idx), 'pi', 0], 
                       ]
        creator = self.get_prepulse_creator(sweep_pulse)
        # print(creator.pulse)
        # self.sync_all(self.us2cycles(0.2))
        self.custom_pulse(self.cfg, creator.pulse, prefix='Storage' + str(stor_idx) + 'dump')
        self.sync_all(self.us2cycles(0.2)) # without this sideband rabi of storage mode 7 has kinks
    
    def coup_stor_swap(self, man_idx): 
        '''
        Perform Swap between manipulate mode and  storage mode 
        '''
        qTest = 0
        sweep_pulse = [['storage', 'M'+ str(man_idx) + '-' + 'C', 'pi', 0], 
                       ]
        creator = self.get_prepulse_creator(sweep_pulse)
        # print(creator.pulse)
        # self.sync_all(self.us2cycles(0.2))
        self.custom_pulse(self.cfg, creator.pulse, prefix='Coupler')
        self.sync_all(self.us2cycles(0.2)) # without this sideband rabi of storage mode 7 has kinks

    def active_reset(self, man_reset = False, storage_reset = False, coupler_reset = False,
                      ef_reset = True, pre_selection_reset = True, prefix = 'base'):
        '''
        Performs active reset on g,e,f as well as man/storage modes 
        Includes post selection measurement
        '''
        cfg = self.cfg
        qTest = 0
        # print('I am here')

        # Prepare Active Reset 
        ## ALL ACTIVE RESET REQUIREMENTS
        # read val definition
        self.r_read_q = 9  # ge active reset register
        self.r_read_q_ef = 10   # ef active reset register
        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # threshold definition
        self.r_thresh_q = 11  # Define a location to store the threshold info

        # # multiplication bc the readout is summed, so need common thing to compare to
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))

        # Define a location to store a counter for how frequently the condj is triggered
        self.r_counter = 12
        self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0

        self.sync_all(self.us2cycles(0.2))
        # self.wait_all(self.us2cycles(0.2))

        ## Requirements for pi pulse 
        self.f_ge_init_reg = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.f_ef_init_reg = self.freq2reg(cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_ef_init = self.cfg.device.qubit.pulses.pi_ef.gain[qTest]
        self.qge_ramp = self.us2cycles(
            cfg.device.active_reset.qubit_ge[2], gen_ch=self.qubit_chs[qTest])  # default ramp value
        self.qef_ramp = self.us2cycles(
            cfg.device.active_reset.qubit_ef[2], gen_ch=self.qubit_chs[qTest])  # default ramp value
        self.pisigma_ge = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ge value
        self.pisigma_ef = self.us2cycles(
            cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ef value

        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge_active_reset",
                       sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef_active_reset",
                       sigma=self.pisigma_ef, length=self.pisigma_ef*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_ge_ramp",
                       sigma=self.qge_ramp, length=self.qge_ramp*6)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_ef_ramp",
                       sigma=self.qef_ramp, length=self.qef_ramp*6)
        
        self.sync_all(self.us2cycles(0.25))
        
        # First Reset Manipulate Modes 
        # =====================================
        if man_reset:
            self.man_reset(0)
            self.man_reset(1)

        # Reset ge level
        # ======================================================
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.res_chs[qTest],
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                     t='auto', wait=True, syncdelay=self.us2cycles(2.0))#self.cfg["relax_delay"])  # self.us2cycles(1))
        
        self.wait_all(self.us2cycles(0.2))  # to allow the read to be complete might be reduced

        self.read(0, 0, "lower", self.r_read_q)  # read data from I buffer, QA, and store
        # self.wait_all(self.us2cycles(0.05))  # to allow the read to be complete might be reduced
        self.sync_all(self.us2cycles(0.05)) # EG: this is not doing anything 

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   prefix + "LABEL_1")  # compare the value recorded above to the value stored in threshold.

        #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
        # self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ge_init_reg, style="arb",
        #                          phase=self.deg2reg(0),
        #                          gain=self.gain_ge_init, waveform='pi_qubit_ge_active_reset')
        self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ge_init_reg, style="flat_top",
                                 phase=self.deg2reg(0), length=self.us2cycles(cfg.device.active_reset.qubit_ge[1]),
                                 gain=cfg.device.active_reset.qubit_ge[0], waveform='pi_ge_ramp')
        self.pulse(ch=self.qubit_chs[qTest])
        self.label(prefix + "LABEL_1")  # location to be jumped to
        # self.wait_all(self.us2cycles(0.05)) 
        self.sync_all(self.us2cycles(0.25))
        # ======================================================

        # Reset ef level
        if ef_reset:    
            # ======================================================
            # self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ef_init_reg, style="arb",
            #                          phase=self.deg2reg(0),
            #                          gain=self.gain_ef_init, waveform='pi_qubit_ef_active_reset')
            self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ef_init_reg, style="flat_top",
                                    phase=self.deg2reg(0), length=self.us2cycles(cfg.device.active_reset.qubit_ef[1]),
                                    gain=cfg.device.active_reset.qubit_ef[0], waveform='pi_ef_ramp')
            self.pulse(ch=self.qubit_chs[qTest])
            # self.wait_all(self.us2cycles(0.05))
            self.sync_all(self.us2cycles(0.05))
            self.measure(pulse_ch=self.res_chs[qTest],
                        adcs=[self.adc_chs[qTest]],
                        adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                        t='auto', wait=True, syncdelay=self.us2cycles(2))  # self.us2cycles(1))
            
            self.wait_all(self.us2cycles(0.2))  # to allow the read to be complete might be reduced
            
            self.read(0, 0, "lower", self.r_read_q_ef)  # read data from I buffer, QA, and store
            # self.wait_all(self.us2cycles(0.05))  # to allow the read to be complete might be reduced
            self.sync_all(self.us2cycles(0.05))

            # perform Qubit active reset comparison, jump if condition is true to the label1 location
            self.condj(0, self.r_read_q_ef, "<", self.r_thresh_q,
                    prefix + "LABEL_2")  # compare the value recorded above to the value stored in threshold.

            #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
            # self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ge_init_reg, style="arb",
            #                          phase=self.deg2reg(0),
            #                          gain=self.gain_ge_init, waveform='pi_qubit_ge_active_reset')
            self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ge_init_reg, style="flat_top",
                                    phase=self.deg2reg(0), length=self.us2cycles(cfg.device.active_reset.qubit_ge[1]),
                                    gain=cfg.device.active_reset.qubit_ge[0], waveform='pi_ge_ramp')
            self.pulse(ch=self.qubit_chs[qTest])
            self.label(prefix + "LABEL_2")  # location to be jumped to
            # self.wait_all(self.us2cycles(0.05)) 
            self.sync_all(self.us2cycles(0.25))

        # ======================================================
        # Dump manipulate 1 and 2 to lossy mode
        # ======================================================
        # if man_reset: 
        #     self.man_reset(0)
        # # self.man_reset(1)

        # ======================================================
        # Dump storage population to manipulate, then to lossy mode
        # for ii in range(len(cfg.device.active_reset.M1_S_freq)):


        if storage_reset: 
            for ii in range(7):
            #      #7
            #ii = 0
                man_idx = 0 
                stor_idx = ii
                self.man_stor_swap(man_idx=man_idx+1, stor_idx=stor_idx+1) #self.man_stor_swap(1, ii+1)
                self.man_reset(0, chi_dressed = False)
                self.man_reset(1, chi_dressed = False)
        
        if coupler_reset:
            self.coup_stor_swap(man_idx=1) # M1
            self.man_reset(0, chi_dressed = False)
            self.man_reset(1, chi_dressed = False)


        
        # if man_reset:
        #     self.man_reset(0, chi_dressed = False)
        #     self.man_reset(1, chi_dressed = False)
        
        #self.man_reset(0, chi_dressed = False)
            
          

            #self.man_reset(0)
        # post selection

        # ======================================================
        if pre_selection_reset: 
            self.sync_all(self.us2cycles(self.cfg.device.active_reset.relax_delay[0]))

            self.measure(pulse_ch=self.res_chs[qTest],
                        adcs=[self.adc_chs[qTest]],
                        adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                        t='auto', wait=True, syncdelay=self.us2cycles(2.0))  # self.us2cycles(1))
            # self.wait_all() 
            # self.sync_all(self.us2cycles(self.cfg.device.active_reset.relax_delay[0]))
            self.sync_all(self.us2cycles(0.2))

    def get_parity_str(self, man_mode_no, return_pulse=False, second_phase = 0): 
        '''
        Create parity pulse 
        '''
        parity_str = [['qubit', 'ge', 'hpi', 0],
                    ['qubit', 'ge', 'parity_M' + str(man_mode_no), 0],
                    ['qubit', 'ge', 'hpi', second_phase]]
        if return_pulse:
            # mm_base = MM_rb_base(cfg = self.cfg)
            creator = self.get_prepulse_creator(parity_str)
            return creator.pulse.tolist()

        return parity_str

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        qTest = 0
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[qTest]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[qTest]
        return shots_i0, shots_q0
    
    # def post_select_histogram(self):

    # ----------------------------------------------------- #Single shot analysis code # ----------------------------------------------------- #    
    def filter_data_IQ(self, II, IQ, threshold, readout_per_experiment=2):
        # assume the last one is experiment data, the last but one is for post selection
        result_Ig = []
        result_Ie = []
        
        
        for k in range(len(II) // readout_per_experiment):
            index_4k_plus_2 = readout_per_experiment * k + readout_per_experiment-2
            index_4k_plus_3 = readout_per_experiment * k + readout_per_experiment-1
            
            # Ensure the indices are within the list bounds
            if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
                # Check if the value at 4k+2 exceeds the threshold
                if II[index_4k_plus_2] < threshold:
                    # Add the value at 4k+3 to the result list
                    result_Ig.append(II[index_4k_plus_3])
                    result_Ie.append(IQ[index_4k_plus_3])
        
        return np.array(result_Ig), np.array(result_Ie)

    def hist(self, data, plot=False, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4.3):
        """
        span: histogram limit is the mean +/- span
        """
        # if active_reset:
        #     Ig = data['Ig'][readout_per_round-1::readout_per_round]
        #     Qg = data['Qg'][readout_per_round-1::readout_per_round]
        #     Ie = data['Ie'][readout_per_round-1::readout_per_round]
        #     Qe = data['Qe'][readout_per_round-1::readout_per_round]
        #     plot_f = False 
        #     if 'If' in data.keys():
        #         plot_f = True
        #         If = data['If'][readout_per_round-1::readout_per_round]
        #         Qf = data['Qf'][readout_per_round-1::readout_per_round]

        if active_reset:
            Ig, Qg = self.filter_data_IQ(data['Ig'], data['Qg'], threshold, readout_per_experiment=readout_per_round)
            # Qg = filter_data(data['Qg'], threshold, readout_per_experiment=readout_per_round)
            Ie, Qe = self.filter_data_IQ(data['Ie'], data['Qe'], threshold, readout_per_experiment=readout_per_round)
            # Qe = filter_data(data['Qe'], threshold, readout_per_experiment=readout_per_round)
            print(len(Ig))
            print(len(Ie))
            plot_f = False 
            if 'If' in data.keys():
                plot_f = True
                If, Qf = filter_data_IQ(data['If'], data['Qf'], threshold, readout_per_experiment=readout_per_round)
                # Qf = filter_data(data['Qf'], threshold, readout_per_experiment=readout_per_round)
                print(len(If))
        else:

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
            
            axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.', s=1)
            axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.', s=1)
            
            if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.', s=1)
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
        if plot_f: theta = -np.arctan2((ye-yf),(xe-xf))

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
            axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.', s=1)
            axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.', s=1)
            if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.', s=1)
            axs[0,1].scatter(xg, yg, color='k', marker='o')
            axs[0,1].scatter(xe, ye, color='k', marker='o')    
            if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

            axs[0,1].set_xlabel('I [ADC levels]')
            axs[0,1].legend(loc='upper right')
            axs[0,1].set_title('Rotated')
            axs[0,1].axis('equal')

            """X and Y ranges for histogram"""

            ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5, density=True)
            ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5, density=True)
            if plot_f:
                nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5, density=True)
            axs[1,0].set_ylabel('Counts')
            axs[1,0].set_xlabel('I [ADC levels]')       
            axs[1,0].legend(loc='upper right')

        else:        
            ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
            ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
            if plot_f:
                nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

        """Compute the fidelity using overlap of the histograms"""
        fids = []
        thresholds = []
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        confusion_matrix = [np.cumsum(ng)[tind]/ng.sum(),
                            1-np.cumsum(ng)[tind]/ng.sum(),
                            np.cumsum(ne)[tind]/ne.sum(),
                            1-np.cumsum(ne)[tind]/ne.sum()]   # Pgg (prepare g measured g), Pge (prepare g measured e), Peg, Pee
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

        return fids, thresholds, theta*180/np.pi, confusion_matrix # fids: ge, gf, ef

    # g states for q0

    


    

# class MM_rb_base(MM_base): 
#     def __init__(self, cfg):
#         ''' rb base is base class of f0g1 rb for storage modes '''
#         super().__init__( cfg)
#         self.init_gate_length() # creates the dictionary of gate lengths
    
#     def init_gate_length(self): 
#         ''' Creates a dictionary of the form 
#         gate_t_length = {
#         'pi_ge_length': 60,
#         'hpi_ge_length': 60,
#         'pi_ef_length': 60,
#         'f0g1_length': 270,
#         'M1S1_length': 400,
#         'M1S2_length': 400,
#         'M1S3_length': 400,
#         'M1S4_length': 400,
#         'M1S5_length': 400,
#         'M1S6_length': 400,
#         'M1S7_length': 400,}

#         Note gate time already includes the sync time  
#         '''
#         self.gate_t_length = {}
#         self.gate_t_length['pi_ge_length'] = self.get_total_time([['qubit', 'ge', 'pi', 0]], gate_based=True, cycles=True)
#         self.gate_t_length['hpi_ge_length'] = self.get_total_time([['qubit', 'ge', 'hpi', 0]], gate_based=True, cycles=True)
#         self.gate_t_length['pi_ef_length'] = self.get_total_time([['qubit', 'ef', 'pi', 0]], gate_based=True, cycles=True)
#         self.gate_t_length['f0g1_length'] = self.get_total_time([['man', 'M1', 'pi', 0]], gate_based=True, cycles=True)
#         for storage_no in range(1, 8):
#             self.gate_t_length[f'M1S{storage_no}_length'] = self.get_total_time([['storage', 'M1-S' + str(storage_no), 'pi', 0]], gate_based=True, cycles=True)
#         # print(self.gate_t_length)
#         return None



      

    
#     """
#     Single qubit RB sequence generator
#     Gate set = {I, +-X/2, +-Y/2, +-Z/2, X, Y, Z}
#     """
#     ## generate sequences of random pulses
#     ## 1:Z,   2:X, 3:Y
#     ## 4:Z/2, 5:X/2, 6:Y/2
#     ## 7:-Z/2, 8:-X/2, 9:-Y/2
#     ## 0:I
#     ## Calculate inverse rotation
#     matrix_ref = {}
#     # Z, X, Y, -Z, -X, -Y
#     matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
#                                     [0, 1, 0, 0, 0, 0],
#                                     [0, 0, 1, 0, 0, 0],
#                                     [0, 0, 0, 1, 0, 0],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [0, 0, 0, 0, 0, 1]])
#     matrix_ref['1'] = np.matrix([[0, 0, 0, 1, 0, 0],
#                                     [0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 0, 1],
#                                     [1, 0, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [0, 0, 1, 0, 0, 0]])
#     matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [0, 0, 1, 0, 0, 0],
#                                     [1, 0, 0, 0, 0, 0],
#                                     [0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 0, 1]])
#     matrix_ref['3'] = np.matrix([[0, 0, 1, 0, 0, 0],
#                                     [0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 1, 0, 0],
#                                     [0, 0, 0, 0, 0, 1],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [1, 0, 0, 0, 0, 0]])
#     matrix_ref['4'] = np.matrix([[0, 0, 0, 0, 1, 0],
#                                     [1, 0, 0, 0, 0, 0],
#                                     [0, 0, 1, 0, 0, 0],
#                                     [0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 1, 0, 0],
#                                     [0, 0, 0, 0, 0, 1]])
#     matrix_ref['5'] = np.matrix([[0, 0, 0, 0, 0, 1],
#                                     [0, 1, 0, 0, 0, 0],
#                                     [1, 0, 0, 0, 0, 0],
#                                     [0, 0, 1, 0, 0, 0],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [0, 0, 0, 1, 0, 0]])
#     matrix_ref['6'] = np.matrix([[0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 1, 0, 0],
#                                     [0, 0, 1, 0, 0, 0],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [1, 0, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 0, 1]])

#     def no2gate(self, no):
#         g = 'I'
#         if no==1:
#             g = 'X'
#         elif no==2:
#             g = 'Y'
#         elif no==3:
#             g = 'X/2'
#         elif no==4:
#             g = 'Y/2'
#         elif no==5:
#             g = '-X/2'
#         elif no==6:
#             g = '-Y/2'  

#         return g

#     def gate2no(self, g):
#         no = 0
#         if g=='X':
#             no = 1
#         elif g=='Y':
#             no = 2
#         elif g=='X/2':
#             no = 3
#         elif g=='Y/2':
#             no = 4
#         elif g=='-X/2':
#             no = 5
#         elif g=='-Y/2':
#             no = 6

#         return no

#     def generate_sequence(self, rb_depth, iRB_gate_no=-1, debug=False, matrix_ref=matrix_ref):
#         gate_list = []
#         for ii in range(rb_depth):
#             gate_list.append(random.randint(1, 6))   # from 1 to 6
#             if iRB_gate_no > -1:   # performing iRB
#                 gate_list.append(iRB_gate_no)

#         a0 = np.matrix([[1], [0], [0], [0], [0], [0]]) # initial state
#         anow = a0
#         for i in gate_list:
#             anow = np.dot(matrix_ref[str(i)], anow)
#         anow1 = np.matrix.tolist(anow.T)[0]
#         max_index = anow1.index(max(anow1))
#         # inverse of the rotation
#         inverse_gate_symbol = ['-Y/2', 'X/2', 'X', 'Y/2', '-X/2']
#         if max_index == 0:
#             pass
#         else:
#             gate_list.append(self.gate2no(inverse_gate_symbol[max_index-1]))
#         if debug:
#             print(gate_list)
#             print(max_index)
#         return gate_list

#     def random_pick_from_lists(self, a):
#         # Initialize index pointers for each sublist
#         indices = [0] * len(a)
#         # Total number of elements to pick
#         total_elements = sum(len(sublist) for sublist in a)
#         # Output list
#         b = []
#         # List to track which sublist each element was picked from
#         origins = []

#         # Continue until all elements are picked
#         while len(b) < total_elements:
#             # Find all sublists that have elements left to pick
#             available = [i for i in range(len(a)) if indices[i] < len(a[i])]
#             # Randomly select one of the available sublists
#             chosen_list = random.choice(available)
#             # Pick the element from the chosen sublist and append to b
#             b.append(a[chosen_list][indices[chosen_list]])
#             # Record the origin of the picked element
#             origins.append(chosen_list)
#             # Update the index pointer for the chosen sublist
#             indices[chosen_list] += 1

#         return b, origins

#     def find_unique_elements_and_positions(self, lst):
#         unique_elements = []
#         first_positions = {}
#         last_positions = {}

#         # Iterate over the list to find the first and last occurrence of each element
#         for idx, elem in enumerate(lst):
#             # Update the last position for every occurrence
#             last_positions[elem] = idx
#             # If the element is encountered for the first time, record its first position
#             if elem not in first_positions:
#                 unique_elements.append(elem)
#                 first_positions[elem] = idx

#         # Create lists of the positions in the order of unique elements
#         first_pos_list = [first_positions[elem] for elem in unique_elements]
#         last_pos_list = [last_positions[elem] for elem in unique_elements]

#         return unique_elements, first_pos_list, last_pos_list

#     def gate2time(self, t0, gate_name, gate_t_length):

#         # for each middle/final gate: M1-Si-->sync(10ns)-->f0g1-->sync(10ns)-->ef pi pulse-->sync(10ns)-->qubit rb gate-->sync(10ns)-->ef pi pulse-->sync(10ns)-->f0g1-->sync(10ns)-->M1-Si-->sync(10ns)
#         # for each first gate: qubit rb gate-->sync(10ns)-->ef pi pulse-->sync(10ns)-->f0g1-->sync(10ns)-->M1-Si-->sync(10ns)
#         # t0: 1*7 list keeps tracking the last completed gate on each storage mode

#         # return 
#         # tfinal: final time spot, it is a 1*7 list corresponding to previous last operation time (the end time) on Si

#         sync_t = 0 #4   # 4 cycles of sync between pulses
#         tfinal = []
#         for i in t0:
#             tfinal.append(i)

#         if gate_name[1] == 'M' or gate_name[1] == 'L':

#             sync_total = sync_t*7  # total time for sync
#             f0g1_total = gate_t_length['f0g1_length']*2
#             ef_total = gate_t_length['pi_ef_length']*2
#             if int(gate_name[0]) in [1,2]:
#                 ge_total = gate_t_length['pi_ge_length']
#             else:
#                 ge_total = gate_t_length['hpi_ge_length']

#             m1si_name = 'M1S'+gate_name[-1]+'_length'
#             M1Si_total = gate_t_length[m1si_name]*2

#             tfinal[int(gate_name[2])-1] = sync_total+f0g1_total+ef_total+ge_total+M1Si_total + max(t0)
#             gatelength = sync_total+f0g1_total+ef_total+ge_total+M1Si_total
#         else:  # first pulse is different

#             sync_total = sync_t*4  # total time for sync
#             f0g1_total = gate_t_length['f0g1_length']*1
#             ef_total = gate_t_length['pi_ef_length']*1
#             if int(gate_name[0]) in [1,2]:
#                 ge_total = gate_t_length['pi_ge_length']
#             else:
#                 ge_total = gate_t_length['hpi_ge_length']

#             m1si_name = 'M1S'+gate_name[-1]+'_length'
#             M1Si_total = gate_t_length[m1si_name]*1

#             tfinal[int(gate_name[2])-1] = sync_total+f0g1_total+ef_total+ge_total+M1Si_total + max(t0)
#             gatelength = sync_total+f0g1_total+ef_total+ge_total+M1Si_total

#         return tfinal, gatelength

#     def RAM_rb(self, storage_id, depth_list, cycles2us = 0.0023251488095238095):

#         """
#         Multimode RAM RB generator with VZ speicified
#         Gate set = {+-X/2, +-Y/2, X, Y}
#         storage_id: a list specifying the operation on storage i, eg [1,3,5] means operation on S1, S3,S5
#         depth_list: a list specifying the individual rb depth on corresponding storage specified in storage_id list

#         depth_list and storage_id should have the same length

#         phase_overhead: a 7*7 matrix showing f0g1+[M1S1, ..., M1S7] pi swap's phase overhead to [S1, ..., S7] (time independent part). 
#         phase_overhead[i][j] is M1-S(j+1) swap's+f0g1 phase overhead on M1-S(i+1) (only half of it, a V gate is 2*phase_overhead)

#         phase_freq: a 1*7 list showing [M1S1, ..., M1S7]'s time-dependent phase accumulation rate during idle sessions.
#         gate_t_length: a dictionary ,all in cycles
#             'pi_ge_length': in cycles
#             'hpi_ge_length': in cycles
#             'pi_ef_length': in cycles
#             'f0g1_length': in cycles
#             'M1S1_length': in cycles
#             'M1S2_length': in cycles
#             'M1S3_length': in cycles
#             'M1S4_length': in cycles
#             'M1S5_length': in cycles
#             'M1S6_length': in cycles
#             'M1S7_length': in cycles

#         Each storage operation has two parts:
#         if it is not the initial gate, extract information, gates on qubit, then store information
#         The initial gate only perform gate on qubit, then store information
#         The last gate only extract information, gate on qubit and check |g> population

#         gate_list: a list of strings, each string is gate_id+'F/L/M'+storage_id. 'F': first gate on the storage, 'L': last gate on the storage, 'M': any other gate between F and L
#         vz_phase_list: virtual z phase (in degree)

#         """
#         phase_overhead = self.cfg.device.storage.idling_phase
#         phase_freq = self.cfg.device.storage.idling_freq
#         gate_t_length = self.gate_t_length

#         # generate random gate_list for individual storage 
#         individual_storage_gate = []
#         for ii in range(len(depth_list)):
#             individual_storage_gate.append(self.generate_sequence(depth_list[ii]))
#         stacked_gate, origins = self.random_pick_from_lists(individual_storage_gate)
#         for ii in range(len(origins)):
#             # convert origins to storage mode id
#             origins[ii] = storage_id[origins[ii]]

#         # check first or last element position
#         unique_elements, first_pos_list, last_pos_list = self.find_unique_elements_and_positions(origins)


#         # convert origins+stacked gate to gate_list form

#         #cycles2us = self.cycles2us(1)   # coefficient
#         # print('cycles2us ', cycles2us)

#         gate_list = []
#         vz_phase_list = []  # all in deg, length = gate_list
#         vz_phase_current = [0]*7  # all in deg, position maps to different 7 storages
#         t0_current = [0]*7  # initialize the time clock, each storage mode has its own clock
#         for ii in range(len(stacked_gate)):
#             gate_name = str(stacked_gate[ii])
#             gate_symbol = 'M'
#             vz = 0
            
#             if ii in first_pos_list: 
#                 gate_symbol = 'F'
#             if ii in last_pos_list: gate_symbol = 'L'

#             gate_name = gate_name+gate_symbol+str(origins[ii])
#             # calculate gate time (to be updated properly with experiment.cfg)
#             t0_after, gate_length = self.gate2time(t0_current, gate_name, gate_t_length)

#             gate_list.append(gate_name)
            

            

#             # calculate vz_phase correction using t0_current and t0_after
#             # operation is int(gate_name[-1])
#             # overhead phase is overhead[0,1,2,3,4,5,6][int(gate_name[-1])-1]
#             tophase = [0]*7
#             if ii in first_pos_list:  # first gate 1 overhead
#                 # update 1* overhead
#                 # time independent phase 
#                 for i in range(7):
#                     tophase[i] = phase_overhead[i][int(gate_name[-1])-1]   # in deg
#                 # to others that already applied, no need for self-correction, set self phase to 0
#                 tophase[int(gate_name[-1])-1] = 0
#                 vz_phase_current[int(gate_name[-1])-1] = 0
#                 # print(tophase)
#             else:  # other case 2 overheads
#                 # time independent phase
#                 for i in range(7):
#                     tophase[i] = phase_overhead[i][int(gate_name[-1])-1]*2   # in deg
#                 # time dependent phase
#                 tophase[int(gate_name[-1])-1] += phase_freq[int(gate_name[-1])-1]*(50*0+t0_after[int(gate_name[-1])-1]-t0_current[int(gate_name[-1])-1]-gate_length)*cycles2us/np.pi*180*2*np.pi   # in deg
#                 # print(t0_after[int(gate_name[-1])-1])
#                 # print(t0_current[int(gate_name[-1])-1])

#             for i in range(7):
#                 vz_phase_current[i] += tophase[i]

#             vz_phase_list.append(vz_phase_current[int(gate_name[-1])-1])

#             # update the clock
#             t0_current = t0_after
#             # print(t0_current)

#         vz_phase_list = np.array(vz_phase_list) % 360
        
#         return gate_list, list(vz_phase_list), origins
        
        
class MMAveragerProgram(AveragerProgram, MM_base):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)

# class MMRBAveragerProgram(AveragerProgram, MM_rb_base):
#     def __init__(self, soccfg, cfg):
#         super().__init__(soccfg, cfg)

    
        
    
class MMRAveragerProgram(RAveragerProgram, MM_base): 

    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)
        #self.mm_base = MM_base()

class prepulse_creator2: 
    def __init__(self, cfg, storage_man_file):
        '''
        Takes pulse param of form 
        [name of transition of cavity name like 'ge', 'ef' or 'M1', 'M1-S1', 
        name of pulse like pi, hpi, or parity_M1 or parity_M2,
        phase  (int form )]

        Creates pulses of the form 
        # [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]], drive channel=1 (flux low), 2 (qubit),3 (flux high),6 (storage),5 (f0g1),4 (manipulate),           
        '''
        # config 
        # with open(config_file, 'r') as cfg_file:
        #     yaml_cfg = yaml.safe_load(cfg_file)
        self.cfg = cfg#AttrDict(yaml_cfg)

        # man storage swap data 
        self.dataset = storage_man_swap_dataset(storage_man_file)
        
        # initialize pulse 
        self.pulse = np.array([[],[],[],[],[],[],[]], dtype = object)
    
    def flush(self):
        '''re initializes to empty array'''
        self.pulse = np.array([[],[],[],[],[],[],[]], dtype = object)
    
    def append(self, pulse):
        self.pulse = np.concatenate((self.pulse, pulse), axis=1)
        return None
        
    def qubit(self, pulse_param): #(self, transition_name, pulse_name, man_idx = 0):
        ''' pulse name comes from yaml file '''
        # print(pulse_param)
        transition_name, pulse_name, phase = pulse_param
        # frequency 
        if transition_name[:2] == 'ge': 
            freq = self.cfg.device.qubit.f_ge[0]
        else: 
            freq = self.cfg.device.qubit.f_ef[0]
        
        if pulse_name[:6] != 'parity':
            pulse_full_name = pulse_name + '_' + transition_name # like pi_ge or pi_ef or pi_ge_new or pi_ef_new

            # print(self.cfg.device.qubit.pulses[pulse_full_name])
            
            qubit_pulse = np.array([[freq], 
                    [self.cfg.device.qubit.pulses[pulse_full_name]['gain'][0]],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['length'][0]],
                    [phase],
                    [2],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['type'][0]],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['sigma'][0]]], dtype = object)
            

        else: # parity   string is 'parity_M1' or 'parity_M2'
            man_idx = int(pulse_name[-1:]) -1 # 1 for man1, 2 for man2
            qubit_pulse = np.array([[freq], 
                    [0],
                    [self.cfg.device.manipulate.revival_time[man_idx] ], # parity delay experiment doesn't involve 10 ns syncs 
                    [phase],
                    [2],
                    ['const'],
                    [0.0]], dtype = object)
        self.pulse = np.concatenate((self.pulse, qubit_pulse), axis=1)
        return None
    def man(self, pulse_param):
        '''name can be pi or hpi
        man_idx is not irrelvant
        '''
        cav_name, pulse_name, phase = pulse_param

        if pulse_name == 'pi': 
            length = self.dataset.get_pi(cav_name)
        else:
            length = self.dataset.get_h_pi(cav_name)
        
        f0g1  = np.array([[self.dataset.get_freq(cav_name)],
                [ self.dataset.get_gain(cav_name)],
                [length],
                [phase],
                [0], # f0g1 pulse 
                ['flat_top'],
                [0.005]], dtype = object)
        
        self.pulse = np.concatenate((self.pulse, f0g1), axis=1)
        return None
    def buffer(self, pulse_param): 
        '''here the last parameter is time '''
        buffer = np.array([[0],
                [0],
                [pulse_param[-1]],
                [0],
                [1],
                ['const'],
                [0.005]], dtype = object)
        self.pulse = np.concatenate((self.pulse, buffer), axis=1)
        return None

    def storage(self, pulse_param):
        '''
        plays sideband pulse on storage via coupler rf flux
        name can be pi or hpi'''
        stor_name, pulse_name, phase = pulse_param

        if pulse_name == 'pi': 
            # print(stor_name)
            length = self.dataset.get_pi(stor_name)
        else:
            length = self.dataset.get_h_pi(stor_name)
        freq = self.dataset.get_freq(stor_name)
        if freq<1000: 
            ch = 1
        else:
            ch = 3
        
        storage_pulse = np.array([[self.dataset.get_freq(stor_name)],
                [ self.dataset.get_gain(stor_name)],
                [length],
                [phase],
                [ch],
                ['flat_top'],
                [0.005]], dtype = object)
        
        self.pulse = np.concatenate((self.pulse, storage_pulse), axis=1)
        return None
    
    
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.fit_display_classes import (
    CavityRamseyGainSweepFitting,
    GeneralFitting,
    RamseyFitting,
)
from experiments.MM_base import *
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display import *


class CavityRamseyExcursionProgramMixin:
    def add_bipolar_gauss(self, 
                          ch, 
                          name, 
                          sigma, 
                          length, 
                          maxv = None, 
                          even_length = False):
        """
        +gauss then -gauss. Each gaussian has length = 4*sigma.
        sigma_cycles: in cycles (already converted)
        gap_cycles: optional zero gap between lobes (cycles)
        """
        gencfg = self.soccfg['gens'][ch]
        if maxv is None: 
            maxv = self.soccfg.get_maxv(ch)
        samps_per_clk = gencfg['samps_per_clk']

        if self.GAUSS_BUG:
            sigma /= np.sqrt(2.0)

        # convert to integer number of fabric clocks
        if self.USER_DURATIONS:
            if even_length:
                lenreg = 2*self.us2cycles(gen_ch=ch, us=length/2)
            else:
                lenreg = self.us2cycles(gen_ch=ch, us=length)
            sigreg = self.us2cycles(gen_ch=ch, us=sigma, as_float=True)
        else:
            lenreg = np.round(length)
            sigreg = sigma

        # convert to number of samples
        lenreg *= samps_per_clk
        sigreg *= samps_per_clk

        g = gauss(mu=lenreg/2-0.5, si=sigreg, length=lenreg, maxv=maxv)
        env = np.concatenate([g, -g])
        self.add_envelope(ch, name, idata=env)

    def fabric_to_timing(self, fabric_cycles):
        """
        Converts fabric clock cycles into timing clock cycles

        Args:
            fabric_cycles(int): the number of fabric clock cycles
        
        Returns:
            int: converted timing clock cycles

        """
        return int(np.round(fabric_cycles * 2 / 3))

    def timing_to_fabric(self, timing_cycles):
        """
        Converts timing dispatcher clock cycles into fabric clock cycles
        
        Args:
            timing_cycles(int): the number of tProc dispatcher timing clock cycles
        
        Returns:
            int: converted fabric clock cycles
        """
        return int(np.round(timing_cycles * 3 / 2))

    def align_fabric_to_timing(self, length):
        """
        Aligns the length of pulse in a unit of fabric cycle to the unit of tProc cycles
        when tProc = (2/3) * fabric 

        Args:
            length(int): length in a fabric clock cycle
        
        Returns:
            aligned_length(int): aligned length in a fabric clock cycle
            tProc_length(int): tProcessor clock cycle that matches the aligned length
        """
        aligned_length = int(3 * np.round(length / 3))
        tProc_length = int(2 * aligned_length / 3)
        return aligned_length, tProc_length


class CavityRamseyExcursionProgram(MMRAveragerProgram,
                                   CavityRamseyExcursionProgramMixin):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)
        if self.execute_gaussian == True:
            _start_us = self.cycles2us(self.excursion_length_timing)
            _step_us = self.cycles2us(self.excursion_length_timing)
        else:
            _start_us = self.cycles2us(self.us2cycles(self.time_step))
            _step_us = self.cycles2us(self.us2cycles(self.time_step))
            
        self.cfg.start = _start_us
        self.cfg.step = _step_us
        self.cfg.expt.start = _start_us
        self.cfg.expt.step = _step_us



    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = 0 # only one qubit for now
        
        # choose the channel on which ramsey will run 
        if cfg.expt.user_defined_pulse[5] == 1:
            self.cavity_ch = self.flux_low_ch
            self.cavity_ch_types = self.flux_low_ch_type
        elif cfg.expt.user_defined_pulse[5] == 2:
            self.cavity_ch= self.qubit_chs
            self.cavity_ch_types = self.qubit_ch_types
        elif cfg.expt.user_defined_pulse[5] == 3:
            self.cavity_ch = self.flux_high_ch
            self.cavity_ch_types = self.flux_high_ch_type
        elif cfg.expt.user_defined_pulse[5] == 6:
            self.cavity_ch = self.storage_ch
            self.cavity_ch_types = self.storage_ch_type
        elif cfg.expt.user_defined_pulse[5] == 0:
            self.cavity_ch = self.f0g1_ch
            self.cavity_ch_types = self.f0g1_ch_type
        elif cfg.expt.user_defined_pulse[5] == 4:
            self.cavity_ch = self.man_ch
            self.cavity_ch_types = self.man_ch_type

        self.q_rps = [self.ch_page(ch) for ch in self.cavity_ch] # get register page for f0g1 channel
        self.stor_rps = 0 # get register page for storage channel

        if self.cfg.expt.storage_ramsey[0]: 
            # decide which channel do we flux drive on 
            sweep_pulse = [
                ['storage', 
                 'M'+ str(self.cfg.expt.man_mode_no) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 
                 'pi',
                   0], 
            ]
            self.creator = self.get_prepulse_creator(sweep_pulse)
            freq = self.creator.pulse[0][0]
            self.flux_ch = self.flux_low_ch if freq < 1000 else self.flux_high_ch
            # get register page for that channel 
            self.flux_rps = [self.ch_page(self.flux_ch[qTest])]

        if self.cfg.expt.man_ramsey[0]: 
            print('using multiphoton conf for the f0-g1')
            sweep_pulse = [['multiphoton', 'f0-g1', 'pi', 0]]
            # sweep_pulse = [
            #     ['man', 'M'+ str(self.cfg.expt.man_ramsey[1]) , 'pi', 0], 
            # ]
            self.creator = self.get_prepulse_creator(sweep_pulse)

        if self.cfg.expt.coupler_ramsey: 
            # decide which channel do we flux drive on 
            pulse_str = self.cfg.expt.custom_coupler_pulse
            freq = pulse_str[0][0]
            self.flux_ch = self.flux_low_ch if freq < 1000 else self.flux_high_ch
            # get register page for that channel 
            self.flux_rps = [self.ch_page(self.flux_ch[qTest])]
        # if self.cfg.expt.custom_coupler_pulse[0]:
        #     self.ramse

        if self.cfg.expt.echoes[0]: 
            mm_base_dummy = MM_dual_rail_base(self.cfg, self.soccfg)
            if self.cfg.expt.storage_ramsey[0]:
                prep_stor = mm_base_dummy.prep_random_state_mode(3, self.cfg.expt.storage_ramsey[1])  # prepare the storage state + 
            elif self.cfg.expt.man_ramsey[0]:
                # prep_stor = mm_base_dummy.prep_man_photon(man_no=self.cfg.expt.man_ramsey[1], hpi = True)
                prep_stor = mm_base_dummy.prep_fock_state(man_no=self.cfg.expt.man_ramsey[1], 
                                                          photon_no_list=[0,1], broadband=True) # prepare the manipulate state +   

            get_stor = prep_stor[::-1] # get the storage state
            print('Echo pulse:', get_stor + prep_stor)
            self.echo_pulse_str = get_stor + prep_stor # echo pulse is the sum of the two pulse sequences
            self.echo_pulse = self.get_prepulse_creator(self.echo_pulse_str).pulse.tolist()
            # print(self.echo_pulse)

        # declare registers for phase incrementing
        self.r_wait = 3
        # self.r_wait_flux = 3
        self.r_phase2 = 4
        self.r_phase3 = 0
        self.r_phase4 = 6
        
        # if self.cavity_ch_types[qTest] == 'int4':
        #     self.r_phase = self.sreg(self.cavity_ch[qTest], "freq")
        #     self.r_phase3 = 5 # for storing the left shifted value
        # else:
        if (self.cfg.expt.storage_ramsey[0] and self.cfg.expt.storage_ramsey[2]) or self.cfg.expt.coupler_ramsey:
            self.phase_update_channel = self.flux_ch
            # self.q_rps = self.flux_rps
        elif self.cfg.expt.man_ramsey[0]:
            self.phase_update_channel = self.cavity_ch
        #QUESTION 1: Is not this REDUNDANCY???
        elif self.cfg.expt.user_defined_pulse[0] and self.cfg.expt.storage_ramsey[0]:
            # print('Running Kerr; will update phase ch')
            self.phase_update_channel = self.cavity_ch
        elif self.cfg.expt.user_defined_pulse[0] :
            # print('Running f0g1 ramsey')
            self.phase_update_channel = self.cavity_ch
        # print(f'phase update channel: {self.phase_update_channel}')
        self.phase_update_page = [self.ch_page(self.phase_update_channel[qTest])]
        self.r_phase = self.sreg(self.phase_update_channel[qTest], "phase")
        
        self.current_phase = 0   # in degree

        #for user defined
        if cfg.expt.user_defined_pulse[0]:
            # print('This is designed for displacing manipulate mode, not for swapping pi/2 into man')
            self.user_freq = self.freq2reg(cfg.expt.user_defined_pulse[1], gen_ch=self.cavity_ch[qTest])
            self.user_gain = cfg.expt.user_defined_pulse[2]
            self.user_sigma = self.us2cycles(cfg.expt.user_defined_pulse[3], gen_ch=self.cavity_ch[qTest])
            self.user_length  = self.us2cycles(cfg.expt.user_defined_pulse[4], gen_ch=self.cavity_ch[qTest])
            # print(f"if user length is 0, then it is a gaussian pulse with sigma {self.user_sigma} cycles")
            # print('user length:', self.user_length)
            self.add_gauss(ch=self.cavity_ch[qTest], 
                           name="user_test",
                           sigma=self.user_sigma, 
                           length=self.user_sigma*4)
            

        # For flux excursion
        self.excursion_page = [self.ch_page(self.flux_low_ch[qTest])]
        self.excursion_sigma = self.us2cycles(cfg.expt.excursion_sigma,
                                     gen_ch = self.flux_low_ch[qTest])
        self.excursion_gain = self.cfg.expt.excursion_gain
        # self.excursion_length, self.excursion_length_timing = self.align_fabric_to_timing(4 * self.excursion_sigma)
        # self.excursion_length_timing *= 2
        self.excursion_length_gen_ch = self.us2cycles(cfg.expt.excursion_sigma, 
                                                      gen_ch = self.flux_low_ch[qTest]) * 4
        
        self.excursion_length_timing = self.us2cycles(cfg.expt.excursion_sigma) * 8 #pulse length into tProc
        
        self.add_bipolar_gauss(ch = self.flux_low_ch[qTest],
                               name = 'flux_excursion',
                               sigma = self.excursion_sigma,
                               length = self.excursion_length_gen_ch)
                            #    length = self.excursion_length)
        self.safe_regwi(self.excursion_page[qTest], 
                        self.r_wait, 
                        self.excursion_length_timing)
        #For Sine waves
        self.execute_gaussian = self.cfg.expt.execute_gaussian
        if self.execute_gaussian != True:
            self.r_wait_sine = 4
            # self.r_wait_sine_gen = 5
            self.sine_freq = self.cfg.expt.sine_freq
            self.sine_freq_reg = self.freq2reg(self.sine_freq,
                                               gen_ch = self.flux_low_ch[qTest])
            self.number_per_cycle = self.cfg.expt.number_per_cycle
            self.time_step = 1/self.sine_freq * self.number_per_cycle # in us
            self.safe_regwi(self.excursion_page[qTest], 
                            self.r_wait_sine, 
                            self.us2cycles(self.time_step))
            # self.safe_regwi(self.excursion_page[qTest], 
            #                 self.r_wait_sine_gen, 
            #                 self.us2cycles(self.time_step,
            #                                gen_ch = self.flux_low_ch[qTest]))

        # for kerr engineering, drive a tone near the qubit
        if "qubit_drive_pulse" in cfg.expt and cfg.expt.qubit_drive_pulse[0]:
            print(self._gen_regmap)
            # print("register", self.sreg(self.qubit_chs[qTest], "len"))
            # self.qTest = self.qubits[0]
            # self.qubit_drive_freq = self.freq2reg(cfg.expt.qubit_drive_pulse[1], gen_ch=self.qubit_chs[self.qTest])
            # self.qubit_drive_gain = cfg.expt.qubit_drive_pulse[2]
            # self.qubit_drive_sigma = self.us2cycles(cfg.expt.qubit_drive_pulse[3], gen_ch=self.qubit_chs[self.qTest])
            # self.qubit_drive_length = self.us2cycles(cfg.expt.qubit_drive_pulse[4], gen_ch=self.qubit_chs[self.qTest])
            # # Flat top pulse
            # if self.qubit_drive_length == 0:
            #     self.add_gauss(ch=self.qubit_chs[self.qTest], name="test_qubit_drive",
            #                    sigma=self.qubit_drive_sigma, length=self.qubit_drive_sigma*4)

        # load the slow pulse waveform
        _sigma = cfg.device.qubit.pulses.slow_pi_ge.sigma[qTest]
        sigma_2_cycles = self.us2cycles(_sigma, gen_ch=self.qubit_chs[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="slow_pi_ge",
                       sigma=sigma_2_cycles, length=sigma_2_cycles*4)

        # initialize wait registers
        # self.safe_regwi(self.phase_update_page[qTest], 
        #                 self.r_wait, 
        #                 self.us2cycles(cfg.expt.start))
        #self.safe_regwi(self.flux_rps, self.r_wait_flux, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.phase_update_page[qTest], 
                        self.r_phase2, 
                        self.deg2reg(0)) 
        self.safe_regwi(self.phase_update_page[qTest], 
                        self.r_phase3, 
                        0) 
        self.safe_regwi(self.phase_update_page[qTest], 
                        self.r_phase4, 
                        0) 

        self.sync_all(200)
        self.parity_meas_pulse = self.get_parity_str(self.cfg.expt.man_mode_no, return_pulse=True, second_phase=180, fast = False)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0] 

        # reset and sync all channels
        self.reset_and_sync()

        # active reset 
        if self.cfg.expt.active_reset: 
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        # pre pulse
        if cfg.expt.prepulse:
            print('pre pulse')
            # print(cfg.expt.pre_sweep_pulse)
            if cfg.expt.gate_based: 
                print('gate based prepulse')
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else:
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')

        # play the prepulse for kerr experiment (displacement of manipulate)
        if self.cfg.user_defined_pulse[0]:
            if "prep_e_first" in self.cfg.expt.keys() and self.cfg.expt.prep_e_first:
                print('prep e first')
                _prepulse = [['qubit', 'ge', 'pi', 0]]
                creator = self.get_prepulse_creator(_prepulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre')

            if self.user_length == 0: # its a gaussian pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest],
                                     style="arb",
                                     freq=self.user_freq,
                                     phase=self.deg2reg(0, gen_ch=self.cavity_ch[qTest]), 
                                     gain=self.user_gain,
                                     waveform="user_test")
            else: # its a flat top pulse
                self.setup_and_pulse(ch=self.cavity_ch[qTest],
                                     style="flat_top",
                                     freq=self.user_freq,
                                     phase=0,
                                     gain=self.user_gain,
                                     length=self.user_length,
                                     waveform="user_test")
            self.sync_all(self.us2cycles(0.01))

        if cfg.expt.storage_ramsey[0]:
            # sweep_pulse = [['storage', 'M'+ str(self.cfg.expt.man_idx) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi'], ]
            # creator = self.get_prepulse_creator(sweep_pulse)
            self.custom_pulse(self.cfg, self.creator.pulse, prefix='Storage' + str(cfg.expt.storage_ramsey[1]))
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.coupler_ramsey:
            self.custom_pulse(cfg, cfg.expt.custom_coupler_pulse, prefix='CustomCoupler')
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.man_ramsey[0]:
            # man ramsey should be true if you are swapping in a 0+1 into manipulate instead of doing displacements; 
            # if displacements, then do user defined pulse
            self.custom_pulse(self.cfg, self.creator.pulse, prefix='Manipulate' + str(cfg.expt.man_ramsey[1]))
            self.sync_all(self.us2cycles(0.01))

        # wait advanced wait time
        if self.execute_gaussian == True:
            self.sync_all()
            self.setup_and_pulse(ch=self.flux_low_ch[qTest],
                                style="arb",
                                freq= 0,
                                phase=0,
                                gain=self.excursion_gain,
                                waveform="flux_excursion",
                                mode = "periodic") # periodic bipolar gaussian pulse
            # self.sync(self.phase_update_page[qTest], self.r_wait)
            self.sync(self.excursion_page[qTest], self.r_wait)
            self.reset_timestamps()
            self.setup_and_pulse(ch=self.flux_low_ch[qTest],
                                t =0,
                                style="const",
                                freq= 0,
                                phase=0,
                                gain=0,
                                length= 4,
                                #  waveform="flux_excursion",
                                mode = "oneshot")
        else:
            self.set_pulse_registers(ch=self.flux_low_ch[qTest],
                                     style="const",
                                     length = 400,
                                     freq= self.sine_freq_reg,
                                     phase=0,
                                     gain=self.excursion_gain,
                                     mode = "periodic")
            self.pulse(ch=self.flux_low_ch[qTest],
                       t = 0)
            self.sync(self.excursion_page[qTest], 
                      self.r_wait_sine)
            self.reset_timestamps()
            self.setup_and_pulse(ch=self.flux_low_ch[qTest],
                                 style = 'const',
                                 t = 0,
                                 phase = 0,
                                 length = 4,
                                 freq = self.sine_freq_reg,
                                 gain = 0)
            
            
            # _r_t = self.sreg(ch = self.flux_low_ch[qTest],
            #                 "t")
            # self.mathi(self.excursion_page,
            #            _r_t,
            #            self.r_wait_sine,
            #            "+",
            #            0)
            # self.pulse(ch = self.flux_low_ch[qTest],
            #            t = None)
            # self.mathi(self.excursion_page, 
            #            self.r_length_sine,
            #            self.r_wait_sine,
            #            '+',
            #            0)
            # self.sync_all()
            # self.pulse(ch = self.flux_low_ch[qTest],
            #            t = 0)
            self.sync_all()
        # self.sync_all()
        # commented out to ensure the displacement pulse right after the bipolar guassian pulse in the flux low line

        # echoes 
        if cfg.expt.echoes[0]:
            for i in range(cfg.expt.echoes[1]):
                if cfg.expt.storage_ramsey[0] or self.cfg.expt.man_ramsey[0] :
                    self.custom_pulse(cfg, self.echo_pulse, prefix='Echo')
                else:
                    # print('echoes not supported for coupler or user defined pulses')
                    self.sync_all()
                    self.sync(self.phase_update_page[qTest], self.r_wait)
                    self.sync_all()

        self.mathi(self.phase_update_page[qTest], self.r_phase, self.r_phase2, "+", 0)
        # self.sync_all(self.us2cycles(0.01))
        
        # commented out the original sync 
        # to ensure the displacement pulse right after the bipolar guassian pulse in the flux low line
        if cfg.expt.storage_ramsey[0] or self.cfg.expt.coupler_ramsey:
            self.pulse(ch=self.flux_ch[qTest])
            self.sync_all(self.us2cycles(0.01))
        elif self.cfg.expt.man_ramsey[0]:   
            self.pulse(ch=self.cavity_ch[qTest])
            self.sync_all(self.us2cycles(0.01))

        if self.cfg.user_defined_pulse[0]:
            self.pulse(ch=self.cavity_ch[qTest])
            self.sync_all(self.us2cycles(0.01))

        # postpulse 
        self.sync_all()
        if cfg.expt.postpulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.post_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'post_')
            else: 
                self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix = 'post_')

        if not self.cfg.user_defined_pulse[0]:
            # parity measurement
            if self.cfg.expt.parity_meas: 
                self.custom_pulse(self.cfg, self.parity_meas_pulse, prefix='ParityMeas')

        else: 
            _freq = cfg.device.qubit.f_ge[qTest]
            _phase = 0
            _gain = cfg.device.qubit.pulses.slow_pi_ge.gain[qTest]
            _sigma = cfg.device.qubit.pulses.slow_pi_ge.sigma[qTest]
            _length = cfg.device.qubit.pulses.slow_pi_ge.length[qTest]
            _style = cfg.device.qubit.pulses.slow_pi_ge.type[qTest]
            freq_2_reg = self.freq2reg(_freq, gen_ch=self.qubit_chs[qTest])
            _sigma_2_cycles = self.us2cycles(_sigma, gen_ch=self.qubit_chs[qTest])
            _length_2_cycles = self.us2cycles(_length, gen_ch=self.qubit_chs[qTest])
            phase_2_reg = self.deg2reg(_phase, gen_ch=self.qubit_chs[qTest])
            # print(f'_freq: {_freq}, _phase: {_phase}, _gain: {_gain}, _length: {_length}, _style: {_style}')

            self.setup_and_pulse(ch=self.qubit_chs[qTest],
                                 style=_style,
                                 freq=freq_2_reg, 
                                 phase=phase_2_reg,
                                 gain=_gain,
                                 length=_length_2_cycles,
                                 waveform="slow_pi_ge") # slow pi pulse for readout

        self.measure_wrapper()


    def update(self):
        '''
        Math i does not like values above 180 for the last argument 
        '''
        qTest = self.qubits[0]

        # update the phase of the LO for the second π/2 pulse
        phase_step_deg = 360 * self.cfg.expt.ramsey_freq *  self.cycles2us(self.excursion_length_timing)
        phase_step_deg = phase_step_deg % 360 # make sure it is between 0 and 360
        if phase_step_deg < 0: # given the wrapping statement above, this should never be true
            if phase_step_deg < -180:  # between -360 and -180
                phase_step_deg += 360
                logic = '+'
            else:                      # between -180 and 0
                phase_step_deg = abs(phase_step_deg)
                logic = '-'
        else:
            if phase_step_deg < 180: # between 0 and 180
                phase_step_deg = phase_step_deg 
                logic = '+'
            else:                     # between 180 and 360
                phase_step_deg = 360 - phase_step_deg
                logic = '-'
        # print(f'phase step deg: {phase_step_deg}')
        # print(f'phase step logic: {logic}')
        phase_step = self.deg2reg(phase_step_deg -85, 
                                  gen_ch=self.phase_update_channel[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        
        #self.safe_regwi(self.q_rps[qTest], self.r_phase3, phase_step) 
        # self.current_phase += 360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step
        # print(self.current_phase)
        # self.current_phase = self.current_phase % 360
        # if self.current_phase > 180: self.current_phase -= 360
        # if self.current_phase < -180: self.current_phase += 360
        if self.execute_gaussian == True:
            self.mathi(self.excursion_page[qTest], 
                    self.r_wait, 
                    self.r_wait, 
                    '+', 
                    self.excursion_length_timing) # update the time between two π/2 pulses
        else:
            self.mathi(self.excursion_page[qTest], 
                       self.r_wait_sine, 
                       self.r_wait_sine, 
                       '+', 
                       self.us2cycles(self.time_step))
            # self.mathi(self.excursion_page[qTest], 
            #            self.r_wait_sine_gen, 
            #            self.r_wait_sine_gen, 
            #            '+', 
            #            self.us2cycles(self.time_step,
            #                           gen_ch = self.flux_low_ch[qTest]))
        self.sync_all(self.us2cycles(0.01))
        # if self.cfg.expt.storage_ramsey[0]:
        #     self.mathi(self.flux_rps, self.r_wait_flux, self.r_wait_flux, '+', self.us2cycles(self.cfg.expt.step))
        #     self.sync_all(self.us2cycles(0.01))

        # Note that mathi only likes the last argument to be between 0 and 90!!!
        remaining_phase = phase_step_deg
        while remaining_phase != 0:
            if remaining_phase > 85: 
                phase_step = self.deg2reg(85, 
                                          gen_ch=self.phase_update_channel[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
                remaining_phase -= 85
            else:
                phase_step = self.deg2reg(remaining_phase, 
                                          gen_ch=self.phase_update_channel[qTest])
                remaining_phase = 0
            self.mathi(self.phase_update_page[qTest], 
                       self.r_phase2, 
                       self.r_phase2, 
                       logic, 
                       phase_step) # advance the phase of the LO for the second π/2 pulse

        # if phase_step_deg > 0:
        #     self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', phase_step)
        # else: 
        #     self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '-', phase_step) # advance the phase of the LO for the second π/2 pulse
        self.sync_all(self.us2cycles(0.01))
        # self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase4, '+', self.deg2reg(self.current_phase, gen_ch=self.cavity_ch[qTest])) # advance the phase of the LO for the second π/2 pulse
        # self.sync_all(self.us2cycles(0.01))


class CavityRamseyExcursionExperiment(Experiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Ramsey', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, 
                         path=path, 
                         prefix=prefix, 
                         config_file=config_file, 
                         progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        
        ramsey = CavityRamseyExcursionProgram(soccfg=self.soccfg, cfg=self.cfg)
        # print('inide t2 cavity acquire')

        print(self.cfg.expt.expts)

        x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc],
                                           threshold=None,
                                           load_pulses=True,
                                           progress=progress,
                                            # debug=debug,
                                            readouts_per_experiment=read_num)        
 
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases} 
        data['idata'], data['qdata'] = ramsey.collect_shots()  
        self.data = data    
        
        return data

    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        if data is None:
            data = self.data

        if fit:
            cavity_ramsey_analysis = RamseyFitting(
                data, config=self.cfg,
            )

            cavity_ramsey_analysis.analyze(fitparams=fitparams)

        return cavity_ramsey_analysis.data


    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        cavity_ramsey_analysis = RamseyFitting(
            data, config=self.cfg,
        )
        cavity_ramsey_analysis.display()


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


class CavityRamseyExcursionGainSweepExperiment(Experiment): #To Be Added
    def __init__(self, soccfg=None, path="", prefix="CavityRamseyGainSweep", config_file=None, progress=None):
        super().__init__(soccfg=soccfg,
                        path=path,
                        prefix=prefix,
                        config_file=config_file,
                        progress=progress)

    def acquire(self, progress=False, debug=False):

        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        gain_start = self.cfg.expt.gain_start
        gain_step = self.cfg.expt.gain_step
        gain_expts = self.cfg.expt.gain_expts
        gain_list = np.array([gain_start + i * gain_step for i in range(gain_expts)])
        self.cfg.expt.gain_list = gain_list

        do_g_and_e = self.cfg.expt.do_g_and_e

        data = {
            'gain_list': gain_list,
            'xpts': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'g_avgi': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'g_avgq': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'g_amps': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'g_phases': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'e_avgi': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'e_avgq': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'e_amps': np.zeros((len(gain_list), self.cfg.expt.expts)),
            'e_phases': np.zeros((len(gain_list), self.cfg.expt.expts))
        }

        self.cfg.expt.prep_e_first = False # if True prepare the qb in e before g

        for i_gain, gain in enumerate(tqdm(gain_list, disable = not progress)):
            self.cfg.expt.user_defined_pulse[2] = gain

            ramsey = CavityRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
            x_pts, avgi, avgq = ramsey.acquire(soc=self.im[self.cfg.aliases.soc],
                                               threshold=None,
                                               load_pulses=True,
                                               progress=False,
                                                # debug=debug,
                                                readouts_per_experiment=read_num)

            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi + 1j * avgq)
            phases = np.angle(avgi + 1j * avgq)

            data['xpts'][i_gain] = x_pts

            data['g_avgi'][i_gain] = avgi
            data['g_avgq'][i_gain] = avgq
            data['g_amps'][i_gain] = amps
            data['g_phases'][i_gain] = phases

            if do_g_and_e:
                self.cfg.expt.prep_e_first = True
                ramsey = CavityRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
                x_pts, avgi, avgq = ramsey.acquire(soc=self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                                    # debug=debug,
                                                    readouts_per_experiment=read_num)

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amps = np.abs(avgi + 1j * avgq)
                phases = np.angle(avgi + 1j * avgq)
                data['e_avgi'][i_gain] = avgi
                data['e_avgq'][i_gain] = avgq
                data['e_amps'][i_gain] = amps
                data['e_phases'][i_gain] = phases

                self.cfg.expt.prep_e_first = False # reset the flag for next gain

        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data
        return data


    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit: 
            cavity_ramsey_analysis = CavityRamseyGainSweepFitting(
                data, config=self.cfg, 
            )
            # forward any selection/debug kwargs to the fitter
            cavity_ramsey_analysis.analyze(fit=fit, **kwargs)

        return cavity_ramsey_analysis.data


    def display(self, data=None, **kwargs):

        if data is None:
            data=self.data

        cavity_ramsey_analysis = CavityRamseyGainSweepFitting(
            data, config=self.cfg,
        )

        if "save_fig" in kwargs and kwargs["save_fig"]:
            save_fit = True
        else:
            save_fit = False

        # forward any extra kwargs to display as well
        cavity_ramsey_analysis.display(
            save_fig=save_fit,
            **{k: v for k, v in kwargs.items() if k != 'save_fig'}
        )


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


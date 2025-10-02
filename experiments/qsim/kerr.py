import matplotlib.pyplot as plt
import numpy as np
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm.auto import tqdm

import experiments.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseProgram, QsimBaseExperiment
from fit_display_classes import (
    CavityRamseyGainSweepFitting,
    GeneralFitting,
    RamseyFitting,
)
from MM_dual_rail_base import MM_dual_rail_base

"""
In this file, each program looks at the effect of a particular combo of 
detuning, amplitude and duration of the Kerr engineering pulse on the qubit.
We need to know the following effects of such a tone:
- Qubit heating test: apply pump and check qubit state again to check heat up
- Qubit AC Stark shift: find the phase correction for the second qubit pi/2 pulse
  Q: do we need this for qsim if we never use non-pi gates on QB?
- Cavity Ramsey: extract the effective Kerr under this drive

A base class that extracts everything needed to do the Kerr pump pulse in init
For heating, this is just directly measuring the qubit again at the end
For Stark shift, this is a Ramsey on the qubit
For cavity Kerr, this is CavityRamseyProgram adapted with the pulse applied during the wait
"""

class KerrEngBaseProgram(QsimBaseProgram):
    def initialize(self):
        super().initialize()
        cfg = self.cfg

        # for kerr engineering, drive a tone near the qubit
        if "qubit_drive_pulse" in cfg.expt and cfg.expt.qubit_drive_pulse[0]:
            self.qTest = self.qubits[self.qTest]
            self.qubit_drive_freq = self.freq2reg(cfg.expt.qubit_drive_pulse[1], gen_ch=self.qubit_chs[self.qTest])
            self.qubit_drive_gain = cfg.expt.qubit_drive_pulse[2]
            self.qubit_drive_sigma = self.us2cycles(cfg.expt.qubit_drive_pulse[3], gen_ch=self.qubit_chs[self.qTest])
            self.qubit_drive_length = self.us2cycles(cfg.expt.qubit_drive_pulse[4], gen_ch=self.qubit_chs[self.qTest])
            # Flat top pulse
            if self.qubit_drive_length == 0:
                self.add_gauss(ch=self.qubit_chs[self.qTest],
                               name="test_qubit_drive",
                               sigma=self.qubit_drive_sigma,
                               length=self.qubit_drive_sigma*4)

    def core_pulses(self):
        qTest = 0
        ecfg = self.cfg.expt
        kerr_pulse = [
            [self.cfg.device.qubit.f_ge[qTest] + ecfg.kerr_detune],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.qubit_chs[qTest]],
            ['flat_top'],
            [self.cfg.device.qubit.ramp_sigma[qTest]],
        ]
        self.custom_pulse(self.cfg, kerr_pulse, prefix='kerr_')
        # [[frequency], [gain], [length (us)], [phases],
        # [drive channel], [shape], [ramp sigma]]


class KerrHeatingProgram(KerrEngBaseProgram):
    def body(self):
        pass


class KerrStarkProgram(KerrEngBaseProgram):
    def body(self):
        pass


class KerrCavityRamseyProgram(KerrEngBaseProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)


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
                ['storage', 'M'+ str(self.cfg.expt.man_mode_no) + '-' + 'S' + str(cfg.expt.storage_ramsey[1]), 'pi', 0], 
            ]
            self.creator = self.get_prepulse_creator(sweep_pulse)
            freq = self.creator.pulse[0][0]
            self.flux_ch = self.flux_low_ch if freq < 1000 else self.flux_high_ch
            # get register page for that channel 
            self.flux_rps = [self.ch_page(self.flux_ch[qTest])]

        if self.cfg.expt.man_ramsey[0]: 
            sweep_pulse = [
                ['man', 'M'+ str(self.cfg.expt.man_ramsey[1]) , 'pi', 0], 
            ]
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
                prep_stor = mm_base_dummy.prep_man_photon(man_no=self.cfg.expt.man_ramsey[1], hpi = True)
            get_stor = prep_stor[::-1] # get the storage state
            self.echo_pulse_str = get_stor + prep_stor # echo pulse is the sum of the two pulse sequences
            self.echo_pulse = self.get_prepulse_creator(self.echo_pulse_str).pulse.tolist()
            # print(self.echo_pulse)

        # declare registers for phase incrementing
        self.r_wait = 3
        self.r_wait_flux = 3
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
            self.add_gauss(ch=self.cavity_ch[qTest], name="user_test",
                       sigma=self.user_sigma, length=self.user_sigma*4)

        # load the slow pulse waveform
        _sigma = cfg.device.qubit.pulses.slow_pi_ge.sigma[qTest]
        sigma_2_cycles = self.us2cycles(_sigma, gen_ch=self.qubit_chs[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="slow_pi_ge",
                       sigma=sigma_2_cycles, length=sigma_2_cycles*4)

        # initialize wait registers
        self.safe_regwi(self.phase_update_page[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        #self.safe_regwi(self.flux_rps, self.r_wait_flux, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase2, self.deg2reg(0)) 
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase3, 0) 
        self.safe_regwi(self.phase_update_page[qTest], self.r_phase4 , 0) 

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
        self.sync_all()
        self.sync(self.phase_update_page[qTest], self.r_wait)
        self.sync_all()

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
        self.sync_all(self.us2cycles(0.01))

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



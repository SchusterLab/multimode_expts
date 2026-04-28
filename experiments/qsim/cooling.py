import json
import os

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.MM_base import MMAveragerProgram
from experiments.MM_dual_rail_base import MM_dual_rail_base

"""
cooling
"""

class CoolingSpectroscopyProgram(QsimBaseProgram):
    DRIVE_CHANNEL = 3 # bad hard coded :( change me to a cfg value

    def initialize(self):
        super().initialize()
        # self.declare_gen(ch=self.DRIVE_CHANNEL, nqz=2, mixer_freq=7000, mux_freqs=[0]) # bad hard coded :(
        self.declare_gen(ch=self.DRIVE_CHANNEL, nqz=2) # bad hard coded :(

    def core_pulses(self):
        qTest = 0
        ecfg = self.cfg.expt
        spec_pulse = [
            [ecfg.cooling_freq],
            [ecfg.cooling_gain],
            [ecfg.cooling_length],
            [0],
            [self.DRIVE_CHANNEL],
            ['flat_top'],
            [self.cfg.device.storage.ramp_sigma],
        ]
        self.custom_pulse(self.cfg, spec_pulse, prefix='cool_')
        # [[frequency], [gain], [length (us)], [phases],
        # [drive channel], [shape], [ramp sigma]]

        self.sync_all(self.us2cycles(1))


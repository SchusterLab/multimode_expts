import json
import os

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from scipy.ndimage import gaussian_filter1d
from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.MM_base import MMAveragerProgram
from experiments.MM_dual_rail_base import MM_dual_rail_base

"""
cooling
"""

class CoolingSpectroscopyProgram(QsimBaseProgram):
    def initialize(self):
        super().initialize()
        self.declare_gen(ch=4, nqz=2) # bad hard coded :(

    def core_pulses(self):
        qTest = 0
        ecfg = self.cfg.expt
        spec_pulse = [
            [ecfg.cooling_freq],
            [ecfg.cooling_gain],
            [ecfg.cooling_length],
            [0],
            [4], # bad hard coded :( change me to a cfg value
            ['flat_top'],
            [self.cfg.device.storage.ramp_sigma],
        ]
        self.custom_pulse(self.cfg, spec_pulse, prefix='cool_')
        # [[frequency], [gain], [length (us)], [phases],
        # [drive channel], [shape], [ramp sigma]]



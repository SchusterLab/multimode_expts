import os

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from dataset import storage_man_swap_dataset
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.utils import (
    ensure_list_in_cfg,
    guess_freq,
    post_select_raverager_data,
)

class SidebandStarkProgram(QsimBaseProgram):
    """
    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then do a rabi on the sideband
    """

    def core_pulses(self):
        m1s_kwarg = self.m1s_kwargs[self.cfg.expt.init_stor-1]
        m1s_kwarg['freq'] += self.freq2reg(self.cfg.expt.detune, gen_ch=m1s_kwarg['ch'])

        # first hpi
        self.setup_and_pulse(**m1s_kwarg)

        self.sync_all(self.us2cycles(self.cfg.expt.wait))

        # second hpi with updated phase
        m1s_kwarg.update({
            'phase': self.deg2reg(self.cfg.expt.advance_phase),
        })
        self.setup_and_pulse(**m1s_kwarg)

        self.sync_all(self.us2cycles(0.1))


import os

import matplotlib.pyplot as plt
from qick import QickConfig

import fitting.fitting as fitter
from experiments.dataset import StorageManSwapDataset
from experiments.qsim.qsim_base import QsimExperiment, QsimProgram
from experiments.qsim.utils import post_select_raverager_data
from fitting.fit_utils import guess_freq


class SidebandAmpRabiProgram(QsimProgram):
    """
    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then do a rabi on the sideband
    """

    def core_pulses(self):
        m1s_kwarg = self.m1s_kwargs[self.cfg.expt.init_stor-1]
        m1s_kwarg.update({
            'gain': self.cfg.expt.gain,
            'length': self.us2cycles(self.cfg.expt.length, gen_ch=m1s_kwarg['ch']),
        })
        m1s_kwarg['freq'] += self.freq2reg(self.cfg.expt.detune, gen_ch=m1s_kwarg['ch'])

        self.setup_and_pulse(**m1s_kwarg)
        self.sync_all(self.us2cycles(0.1))


class SidebandAmpRabiExperiment(QsimExperiment):
    """
    Sweep amplitude vs detuning
    Experimental Config:
    expt = dict(
        expts: number experiments should be 1 here as we do soft loops
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qubits: this is just 0 for the purpose of the currrent multimode sample
        init_stor: storage to initialize the photon into (1-7)
        ro_stor: storage to readout the photon from (1-7)
    )
    """
    default_program = SidebandAmpRabiProgram

    def sweep_axes(self):
        """Detune outer, gain inner, whatever ``swept_params`` says.

        Until step 10E this class had its own 2D loop; it counted the readouts
        itself and took the parity lane at 0, before the active-reset lanes.
        """
        self.cfg.expt.swept_params = ['detune', 'gain']
        return super().sweep_axes()

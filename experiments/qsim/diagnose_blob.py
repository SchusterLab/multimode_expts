from qick import QickConfig
from slab import AttrDict

from experiments.qsim.sideband_scramble import SidebandScrambleProgram

class ReadoutFreqSweepProgram(SidebandScrambleProgram):
    """
    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then (optionally) swap into init_stor
    Then do whatever in the core_pulses() that you override
    Finally swap ro_stor back into man and then man into qb and readout
    """
    def __init__(self, soccfg: QickConfig, cfg: AttrDict):
        cfg.device.readout.frequency = [cfg.expt.readout_freq] # override readout frequency
        super().__init__(soccfg, cfg)



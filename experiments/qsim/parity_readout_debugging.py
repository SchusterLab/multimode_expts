"""Test pulses for the configured Qsim parity-readout sequence.

The local program applies qubit ge and M1 pi pulses. Optional parity/Wigner
pulses and measurement are controlled by the inherited QsimBaseProgram sequence.
"""

from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseProgram


class ParityReadoutDebuggingProgram(QsimBaseProgram):
    """Apply qubit ge and M1 pi pulses for parity-readout debugging."""

    def core_pulses(self):
        pulse_cfg = [
            ['qubit', 'ge', 'pi', 0,],
            ['man', 'M1', 'pi', 0]
        ]
        pulse = self.get_prepulse_creator(pulse_cfg)
        self.sync_all()
        cfg = AttrDict(self.cfg)
        self.custom_pulse(cfg, pulse.pulse, prefix = 'test_')
        self.sync_all()
        # self.sync_all(self.us2cycles(20))
        

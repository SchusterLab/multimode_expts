from copy import deepcopy
from experiments.qsim.qsim_base import QsimBaseProgram


class SidebandScrambleProgram(QsimBaseProgram):
    """
    Scramble 1 photon via fractional beam splitters
    """
    def core_pulses(self):
        pulse_args = deepcopy(self.m1s_kwargs[self.cfg.expt.init_stor-1])
        pulse_args['gain'] //= self.cfg.expt.gain_div
        pulse_args['length'] //= self.cfg.expt.length_div

        for kk in range(self.cfg.expt.floquet_cycle):
            for jj in range(7):
                if jj+1==self.cfg.expt.init_stor:
                    pulse_args['phase'] = self.deg2reg(self.cfg.expt.advance_phase*kk)
                    self.setup_and_pulse(**pulse_args)
                    self.sync_all()
                elif jj+1==2:
                    pulse2 = deepcopy(self.m1s_kwargs[1])
                    pulse2['gain'] //= self.cfg.expt.gain_div
                    pulse2['length'] //= self.cfg.expt.length_div
                    pulse2['phase'] = self.deg2reg(-130*kk)
                    self.setup_and_pulse(**pulse2)
                else:
                    self.sync_all(self.us2cycles(0.3))


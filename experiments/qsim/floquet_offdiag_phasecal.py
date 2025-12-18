import fitting.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.utils import (
    fit_cos2d,
)

class FloquetPhaseCalProgram(QsimBaseProgram):
    """
    pi/2 from M1 to stor_row to make a dual rail + state,
    temporarily store the M1 half in stor_idle via pi,
    Then do a pulse of varying length on M1-S{stor_col}.
    Retrieve the half state from stor_idle via pi,
    Finally do pi/2 with varying phase to test for phase offset
    """
    def core_pulses(self):
        stor_row = self.cfg.expt.stor_row - 1
        stor_col = self.cfg.expt.stor_col - 1
        stor_idle = self.cfg.expt.stor_idle - 1
        row_kwarg = self.m1s_kwargs[stor_row]
        col_kwarg = self.m1s_kwargs[stor_col]
        idle_kwarg = self.m1s_kwargs[stor_idle]

        pi_fracs = self.m1s_pi_fracs[stor_row] 
        assert pi_fracs % 2 == 0, f'tryna do pi/2 pulse out of pi/{pi_fracs}, not a good idea'

        for _ in range(pi_fracs//2):
            self.setup_and_pulse(**row_kwarg)
        self.sync_all()

        for _ in range(self.m1s_pi_fracs[stor_idle]):
            self.setup_and_pulse(**idle_kwarg)
        self.sync_all()

        col_kwarg['length'] = self.us2cycles(self.cfg.expt.length, gen_ch=col_kwarg['ch'])
        self.setup_and_pulse(**col_kwarg)
        self.sync_all()

        for _ in range(self.m1s_pi_fracs[stor_idle]):
            self.setup_and_pulse(**idle_kwarg)
        self.sync_all()

        row_kwarg['phase'] = self.deg2reg(self.cfg.expt.advance_phase, gen_ch=row_kwarg['ch'])
        for _ in range(pi_fracs//2):
            self.setup_and_pulse(**row_kwarg)
        self.sync_all()


class FloquetPhaseCalExperiment(QsimBaseExperiment):

    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        if data is None:
            data = self.data

        #TODO: this should also handle 1D sweeps?
        if fit:
            self.fit_result = fit_cos2d(self.data['avgi'],
                                        self.data['xpts'],
                                        self.data['ypts'],
                                        plot=True)
            self.f_acstark = self.fit_result.best_values['f']
            self.data['best_fit'] = self.fit_result.best_fit.reshape(self.data['avgi'].shape)
            print(f'AC Stark freq: {self.f_acstark:.6f}MHz')


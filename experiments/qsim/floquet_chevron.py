import experiments.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram


class FloquetChevronProgram(QsimBaseProgram):
    """
    Do a chevron experiment with n floquet pulses
    """
    def core_pulses(self):
        i_stor = self.cfg.expt.init_stor - 1
        m1s_kwarg = self.m1s_kwargs[i_stor]
        ch = m1s_kwarg['ch']
        m1s_kwarg['freq'] += self.freq2reg(self.cfg.expt.detune, gen_ch=ch)
        m1s_kwarg['length'] += self.us2cycles(self.cfg.expt.length, gen_ch=ch)
        
        self.set_pulse_registers(**m1s_kwarg)
        for i in range(self.m1s_pi_fracs[i_stor]):
            self.pulse(ch)
        self.sync_all()
        self.sync_all()


class FloquetChevronExperiment(QsimBaseExperiment):
    def analyze(self, data=None, fit=True, fit_func="sin"):
        if data is None:
            data = self.data
        
        if len(data["avgi"].shape) > 1:
            print("Not implemented analysis for 2D chevron")
            return

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = [None, 1/max(data['xpts']), None, None]
            xdata = data["xpts"]
            fitparams = None
            if fit_func == "sin":
                fitparams = [None] * 4
            elif fit_func == "decaysin":
                fitparams = [None] * 5
            fitparams[1] = 2.0 / xdata[-1]
            if fit_func == "decaysin":
                fit_fitfunc = fitter.fitdecaysin
            elif fit_func == "sin":
                fit_fitfunc = fitter.fitsin
            p_avgi, pCov_avgi = fit_fitfunc(data["xpts"][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fit_fitfunc(data["xpts"][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fit_fitfunc(data["xpts"][:-1], data["amps"][:-1], fitparams=fitparams)
            data["fit_avgi"] = p_avgi
            data["fit_avgq"] = p_avgq
            data["fit_amps"] = p_amps
            data["fit_err_avgi"] = pCov_avgi
            data["fit_err_avgq"] = pCov_avgq
            data["fit_err_amps"] = pCov_amps
        return data

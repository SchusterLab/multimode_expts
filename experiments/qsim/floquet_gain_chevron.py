"""Floquet storage-swap gain chevrons: detuning x gain, with a chevron fit."""

import numpy as np

import fitting.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram


class FloquetGainChevronProgram(QsimBaseProgram):
    """
    Apply repeated Floquet swap pulses to expt.init_stor.

    Use expt.gain and expt.detune with the selected storage mode's configured
    pulse count and waveform.
    """

    def core_pulses(self):
        i_stor = self.cfg.expt.init_stor - 1
        m1s_kwarg = self.m1s_kwargs[i_stor]
        ch = m1s_kwarg['ch']
        m1s_kwarg['freq'] = self.freq2reg(self.m1s_freq_MHz[i_stor] + self.cfg.expt.detune, gen_ch=ch)
        m1s_kwarg['gain'] = self.cfg.expt.gain
        sync_cycles = int(self.cfg.expt.get("scramble_sync_cycles", 10))
        # print("freq", self.reg2freq(m1s_kwarg['freq'], gen_ch=ch), "length", self.cfg.expt.length)
        self.set_pulse_registers(**m1s_kwarg)
        for i in range(self.m1s_pi_fracs[i_stor]):
            self.pulse(ch)
            self.sync_all(sync_cycles)
        self.sync_all()


class FloquetGainChevronExperiment(QsimBaseExperiment):
    """Floquet swap gain chevron.

    A 2D sweep (``swept_params = ['detune', 'gain']``) is fitted with
    ``ChevronFitting``: ``chevron_analysis.results`` holds the best detuning
    (``best_frequency_contrast``) and the gain oscillation
    (``best_fit_params_contrast``). A 1D gain trace is fitted with a sine or
    a decaying sine.
    """

    def analyze(self, data=None, fit=True, fit_func="sin", station=None):
        if data is None:
            data = self.data

        if len(data["avgi"].shape) > 1:
            from fitting.fit_display_classes import ChevronFitting

            analysis = ChevronFitting(
                frequencies=np.array(data["ypts"]),
                time=np.array(data["xpts"]),
                response_matrix=data["avgi"],
                config=self.cfg,
                station=station,
            )
            analysis.analyze()
            self._chevron_analysis = analysis
            return data

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
            p_avgi, pCov_avgi = fit_fitfunc(
                data["xpts"][:-1], data["avgi"][:-1], fitparams=fitparams
            )
            p_avgq, pCov_avgq = fit_fitfunc(
                data["xpts"][:-1], data["avgq"][:-1], fitparams=fitparams
            )
            p_amps, pCov_amps = fit_fitfunc(
                data["xpts"][:-1], data["amps"][:-1], fitparams=fitparams
            )
            data["fit_avgi"] = p_avgi
            data["fit_avgq"] = p_avgq
            data["fit_amps"] = p_amps
            data["fit_err_avgi"] = pCov_avgi
            data["fit_err_avgq"] = pCov_avgq
            data["fit_err_amps"] = pCov_amps
        return data

    @property
    def chevron_analysis(self):
        """The ChevronFitting of the last 2D ``analyze``, or None."""
        return getattr(self, "_chevron_analysis", None)

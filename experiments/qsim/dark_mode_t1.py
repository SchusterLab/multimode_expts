# -*- coding: utf-8 -*-
"""Dark-mode T1: one photon into the dark mode, wait, read it back.

Split out of ``floquet_dark_mode_readout.py`` unchanged. ``DarkBaseProgram``
still lives there; it moves to the shared Floquet sequence layer later
(spec section 7.2).
"""
import matplotlib.pyplot as plt
import numpy as np

import fitting.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseExperiment
from experiments.qsim.floquet_dark_mode_readout import DarkBaseProgram

class DarkT1Program(DarkBaseProgram):

    def core_pulses(self):
        ecfg = self.cfg.expt

        wait_length = ecfg.get("wait_length", ecfg.get("wait", 0.0))

        if not ecfg.get("swap_man_dark", False):
            self.sync_all(self.us2cycles(wait_length))
            return

        swap_stors = list(ecfg.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)

        self.sync_all()

        # 1. M1 photon -> dark/normal mode
        self._prepare_dark_mode(phase_offsets)

        # 2. wait
        self.sync_all(self.us2cycles(wait_length))

        # 3. compensate dark-mode relative phase during wait
        self._apply_dark_wait_phase_tracking(phase_offsets, wait_length)

        # 4. dark/normal mode -> M1
        self._read_dark_mode(phase_offsets)

        self.sync_all()

class DarkT1Experiment(QsimBaseExperiment):
    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)

        T1 = data['fit_avgi'][3]  # decay rate
        T1_err = np.sqrt(data['fit_err_avgi'][3][3])
        kappa = 1/T1/2/ np.pi  # kappa = 1/T1/2/pi in unit of freq
        kappa_err = T1_err/T1**2 # kappa_err = T1_err/T1**2 * kappa

        data['T1'] = T1
        data['T1_err'] = T1_err
        data['kappa_in_freq'] = kappa
        data['kappa_err_in_freq'] = kappa_err


        return data
    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        T1 = data['T1']
        T1_err = data['T1_err']
        kappa = data['kappa_in_freq']
        kappa_err = data['kappa_err_in_freq']

        text = f"$T_1$ = {T1:.3f} $\pm$ {T1_err:.3f} us\n"
        text += f"$\kappa$ = {kappa*1e3:.3f} $\pm$ {kappa_err*1e3:.3f}KHz *2$\pi$\n"


        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'

            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')

        # add the text box with T1 and kappa values
        plt.gcf().text(0.15, 0.8, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

        plt.show()

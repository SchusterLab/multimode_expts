import os

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from tqdm import tqdm_notebook as tqdm

import lmfit
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


def cos2d(tau, phi, f, phi0, A, C):
    return A*np.cos(2*np.pi*f*tau + phi/180*np.pi + phi0/180*np.pi) + C


class Cos2dModel(lmfit.Model):
    """
    Incompatible with lmfit api but very short to call
    """
    def __init__(self, *args, **kwargs):
        super().__init__(cos2d, independent_vars=['tau', 'phi'], *args, **kwargs)

    def guess(self, data, tau, phi, **kwargs):
        verbose = kwargs.pop('verbose', None)
        phases = np.unwrap([guess_freq(tau, line)[1] for line in data]) / np.pi*180
        slope_sign = np.sign(np.corrcoef(phi, phases)[0, 1])
        # if we don't take care of the sign of the freq, it only finds the local minimum at f>0
        freq_guess = np.mean([guess_freq(tau, line)[0] for line in data]) * slope_sign
        offset_guess = np.mean(data)
        amp_guess = np.ptp(data) / 2
        if verbose:
            print(freq_guess, offset_guess, amp_guess)
            plt.plot(phi, phases)
        params = self.make_params(
            f=freq_guess,
            phi0=0,
            A=amp_guess,
            C=offset_guess
        )
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

    def fit(self, data, tau, phi, **kwargs):
        Tau, Phi = np.meshgrid(tau, phi)
        params = self.guess(data, tau, phi)
        return super().fit(data.ravel(), params, tau=Tau.ravel(), phi=Phi.ravel())


class SidebandStarkExperiment(QsimBaseExperiment):
    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        tau, phi = self.data['xpts'], self.data['ypts']
        z = self.data['avgi']
        model = Cos2dModel()
        result = model.fit(z, tau, phi)
        self.fit_result = result
        if result.rsquared<0.7:
            print('R rsquared small, fit likely failed')
        self.f_acstark = result.best_values['f']
        print(f'AC Stark freq: {self.f_acstark:.6f}MHz')

        fig, axs = plt.subplots(1,2,figsize=(12,5))
        mesh = axs[0].pcolormesh(tau, phi, z)
        fig.colorbar(mesh, ax=axs[0])
        mesh = axs[1].pcolormesh(tau, phi, result.best_fit.reshape(z.shape))
        fig.colorbar(mesh, ax=axs[1])



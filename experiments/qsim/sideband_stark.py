from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from dataset import storage_man_swap_dataset
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.utils import (
    fit_cos2d,
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


class SidebandStarkExperiment(QsimBaseExperiment):
    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        if data is None:
            data = self.data

        if fit:
            self.fit_result = fit_cos2d(self.data['avgi'],
                                        self.data['xpts'],
                                        self.data['ypts'],
                                        plot=True)
            self.f_acstark = self.fit_result.best_values['f']
            self.data['best_fit'] = self.fit_result.best_fit.reshape(self.data['avgi'].shape)
            print(f'AC Stark freq: {self.f_acstark:.6f}MHz')



class SidebandStarkAmplificationProgram(QsimBaseProgram):
    """
    1. Apply pi/2 swap pulse made of floquet pulses on stor_A
    2. Apply another floquet 2pi pulse on stor_B to calibrate the matrix element for. Do this xN times for error amplification
    3. Apply a -pi/2 swap pulse of floquet pulses on stor_A, with advanced phase
    
    Parameters in cfg.expt (sweepable):
    stor_A
    stor_B
    n_pulse: Nx pulses on stor B 
    advance_phase: phase of the last pulse on stor_A
    """

    def core_pulses(self):
        i_storA = self.cfg.expt.stor_A - 1
        i_storB = self.cfg.expt.stor_B - 1
        m1s_kwarg_A = self.m1s_kwargs[i_storA]
        m1s_kwarg_B = self.m1s_kwargs[i_storB]

        n_pulse_B = self.cfg.expt.n_pulse
        pi_frac_A = self.m1s_pi_fracs[i_storA]
        pi_frac_B = self.m1s_pi_fracs[i_storB]

        ch_A = m1s_kwarg_A['ch']
        ch_B = m1s_kwarg_B['ch']
        channel_page_B = self.ch_page(ch_B)
        r_phase_B= self.sreg(ch_B, "phase")

        # Apply pi/2 pulse on stor_A
        self.set_pulse_registers(**m1s_kwarg_A)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)
        self.sync_all()
        
        # # Apply a 2pi * n_pulse gate on stor_B
        # self.set_pulse_registers(**m1s_kwarg_B)
        # for i in range(n_pulse_B * 2 * pi_frac_B):
        #     self.pulse(ch_B)
        # advance_phase_A = self.deg2reg(n_pulse_B * pi_frac * self.cfg.expt.advance_phase)
        # self.sync_all()

        # Apply a (pi/12, -pi/12) * n_pulse gate on stor_B
        phase = 0
        self.set_pulse_registers(**m1s_kwarg_B)
        for i in range(n_pulse_B):
            for j in range(2):
                self.pulse(ch_B)
                # update the phase modulo 360
                phase += 180
                phase = phase % 360
                _phase_reg = self.deg2reg(phase, gen_ch=ch_B)
                self.safe_regwi(channel_page_B, r_phase_B, _phase_reg)
        advance_phase_A = self.deg2reg(2 * n_pulse_B * self.cfg.expt.advance_phase)
        self.sync_all()
        
        # Apply -pi/2 pulse on stor_A with advanced phase
        m1s_kwarg_A_advanced = deepcopy(m1s_kwarg_A)
        m1s_kwarg_A_advanced['phase'] = advance_phase_A
        self.set_pulse_registers(**m1s_kwarg_A_advanced)
        for i in range(pi_frac_A // 2):
            self.pulse(m1s_kwarg_A_advanced['ch'])
        self.sync_all()

class SidebandStarkAmplificationExperiment(QsimBaseExperiment):
    def analyze(self, data=None, fit=True, state_fin='g'):
        
        if data is None:
            data=self.data

        # use the fitting process implemented by MIT 
        # https://arxiv.org/pdf/2406.08295
        
        # for avgi, avgq, amp and phase take the product of the raws and

        # prod_avgi = np.abs(np.prod(data['avgi'], axis=0))
        # prod_avgq = np.abs(np.prod(data['avgq'], axis=0))
        # prod_amp = np.abs(np.prod(data['amp'], axis=0))
        # prod_phase = np.abs(np.prod(data['phase'], axis=0))


        Ie = self.cfg.device.readout.Ie[0]
        Ig = self.cfg.device.readout.Ig[0]

        # rescale avgi so that when equal to v_e it is 0 and when equal to v_g it is 1
        if state_fin == 'g':
            data_avgi_scaled = (data['avgi'] - Ie) / (Ig - Ie)
        elif state_fin == 'e':
            data_avgi_scaled = (data['avgi'] - Ig) / (Ie - Ig)
        else:
            raise ValueError("Invalid state_fin. Must be 'g' or 'e'.")

        prod_avgi = np.prod(data_avgi_scaled, axis=0)/ np.prod(data_avgi_scaled, axis=0).max()  # normalize the product
        data['prod_avgi'] = prod_avgi  # normalize the product

        if fit:
            p_avgi, pCov_avgi = fitter.fitgaussian(data['xpts'], data['prod_avgi'])
            data['prod_avgi_fit'] = fitter.gaussianfunc(data['xpts'], *p_avgi)
            # add the fit parameters to the data dictionary
            data['fit_avgi'] = p_avgi
            data['fit_prod_avgi_err'] = np.sqrt(np.diag(pCov_avgi))
    

    def display(self, data=None, fit=False):
        if data is None:
            data=self.data 
        
        fig, axs = super().display(data, fit=fit)

        x_sweep = data['xpts']
        xlabel = self.inner_param

        if fit: 
            if 'fit_avgi' in data:
                x_opt = data['fit_avgi'][2]
                axs[0].axvline(x_opt, color='black', linestyle='--')
                axs[1].axvline(x_opt, color='black', linestyle='--')

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(x_sweep, data['prod_avgi'], label='Avg I Product')
            # add the fit line if available
            if 'prod_avgi_fit' in data:
                ax2.plot(x_sweep, data['prod_avgi_fit'], label='Fit Avg I Product', color='black')
                # add a text annotation for the optimal point if available and put it in the upper left corner
                x_opt = data['fit_avgi'][2]
                text = f"Optimal Phase: {x_opt:.2f} deg"
                ax2.axvline(x_opt, color='black', linestyle='--')
                ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            ax2.set_xlabel(xlabel)
            ax2.set_ylabel('Avg I Product')
            ax2.legend(loc='lower left')
            ax2.grid()
        plt.show()

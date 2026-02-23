import os
from copy import deepcopy

import numpy as np
from lmfit.models import Model
from matplotlib import pyplot as plt

import fitting.fitting as fitter
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram


class StorageT1Program(QsimBaseProgram):
    """
    T1: just a wait

    expt params:
        init_stor
        ro_stor
        wait
    """
    def core_pulses(self):
        self.sync_all(self.us2cycles(self.cfg.expt.wait))


class FloquetCalibrationProgram(QsimBaseProgram):
    """
    Vary the phases to find out the optimal virtual Z correction because of AC Zeeman shift
    Will always do a series of M1-A and M1-B swaps as specified in the exp config
    The phase of each can be varied

    expt params:
        init_stor
        ro_stor
        storA_advance_phase
        storB_advance_phase
        floquet_cycle
        storA
        storB
    """
    def core_pulses(self):
        storA = self.cfg.expt.storA
        storB = self.cfg.expt.storB
        assert storA != storB, "storA and storB modes must be different for this calibration for now"
        assert storA>0 and storB>0, "storA and storB must be storage modes, not M1"
        storA_args = deepcopy(self.m1s_kwargs[storA-1])
        storB_args = deepcopy(self.m1s_kwargs[storB-1])

        for kk in range(self.cfg.expt.floquet_cycle):
            storA_args['phase'] = self.deg2reg(self.cfg.expt.storA_advance_phase*kk, storA_args['ch'])
            self.setup_and_pulse(**storA_args)
            self.sync_all()
            # pulse2['gain'] //= self.cfg.expt.gain_div
            # pulse2['length'] //= self.cfg.expt.length_div
            storB_args['phase'] = self.deg2reg(self.cfg.expt.storB_advance_phase*kk, storB_args['ch'])
            self.setup_and_pulse(**storB_args)
            self.sync_all()


class SidebandScrambleProgram(QsimBaseProgram):
    """
    Scramble photons via fractional beam splitters

    expt params:
        - swap_stors: list of storage modes to apply the floquet swaps to, will go in order of the list
        - detunings: list of detunings of drives in the same order as above
        - update_phases: boolean of whether to update each subsequent swap with the calibrated stark shift phase
        - floquet_cycle: number of floquet cycles to apply
    """
    def core_pulses(self):

        # process expt params
        swap_stors = self.cfg.expt.swap_stors

        if 'detunings' in self.cfg.expt and self.cfg.expt.detunings:
            detunings = self.cfg.expt.detunings
        else:
            detunings = [0.0] * len(swap_stors)
        assert len(detunings) == len(swap_stors), "length of detunings doesn't match that of swap_stors"

        update_phases = self.cfg.expt.update_phases
        swap_stor_phases = [0.0] * len(swap_stors)

        # deep copy floquet params and apply detunings if specified
        all_pulse_args = []
        for i_stor, stor in enumerate(swap_stors):
            pulse_args = deepcopy(self.m1s_kwargs[stor - 1])
            pulse_args['freq'] += self.freq2reg(detunings[i_stor], gen_ch=pulse_args['ch'])
            all_pulse_args.append(pulse_args)

        # start swapping
        self.sync_all()
        debug = self.cfg.expt.get('debug', False)
        if debug:
            print(all_pulse_args)
        for kk in range(self.cfg.expt.floquet_cycle):
            for i_stor, stor in enumerate(swap_stors):
                pulse_args = all_pulse_args[i_stor]
                pulse_args['phase'] = self.deg2reg(swap_stor_phases[i_stor], gen_ch=pulse_args['ch'])
                self.setup_and_pulse(**pulse_args)
                # this sync has to 10 at minimum if using setup_and_pulse to allow instructions to finish!
                self.sync_all(10)

                # Update the phases for all other swaps using the phases accumulated during this swap
                if update_phases:
                    for j_stor, stor_B in enumerate(swap_stors):
                        if stor_B != stor:
                            stor_B_name = f"M1-S{stor_B}"
                            stor_name = f"M1-S{stor}"
                            swap_stor_phases[j_stor] += self.swap_ds.get_phase_from(stor_B_name, stor_name)
                            swap_stor_phases[j_stor] = swap_stor_phases[j_stor] % 360
        self.sync_all()


class FloquetCalibrationAmplificationExperiment(QsimBaseExperiment):
    """
    expt params:
        init_stor
        ro_stor
        storA_advance_phase
        storB_advance_phase
        n_floquet_per_scramble # the number of floquet cycles (each cycle consists of the pi/pi_frac pulse for storA and storB) to implement one period in the random walk
        n_scramble_cycles # a list with the number of error amplification random walk periods to sweep over
    """
    def acquire(self, progress=False, debug=False):

        n_scramble_cycles = self.cfg.expt.n_scramble_cycles
        n_floquet_per_scramble = self.cfg.expt.n_floquet_per_scramble
        swept_params = ['storA_advance_phase', 'storB_advance_phase']
        self.cfg.expt.swept_params = swept_params

        all_data = dict()

        for n_scramble_cycle in n_scramble_cycles:
            floquet_cycle = (2*n_scramble_cycle+1) * n_floquet_per_scramble
            print("Starting experiment for n_scramble_cycle", n_scramble_cycle, "with total floquet cycles", floquet_cycle)
            self.cfg.expt.floquet_cycle = floquet_cycle
            super().acquire(progress=progress, debug=debug)
            for key in self.data:
                if key not in all_data.keys():
                    all_data[key] = [self.data[key]]
                else:
                    all_data[key].append(self.data[key])
            if debug:
                super().display()

        for key in all_data:
            all_data[key] = np.array(all_data[key])

        # data shape: (len(n_scramble_cycles), len(storA_advance_phases), len(storB_advance_phases))
        self.data = all_data


    def analyze(self, data=None, fit=True, state_fin='g', fit_model='sg'):
        """
        fit_model: 'sg' for a single Gaussian peak or 'dl' for double Lorentzian peaks
        """
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

        # data shape: ()
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
            if 'ypts' not in data.keys():
                p_avgi, pCov_avgi = fitter.fitgaussian(data['xpts'], data['prod_avgi'])
                data['prod_avgi_fit'] = fitter.gaussianfunc(data['xpts'], *p_avgi)
                # add the fit parameters to the data dictionary
                data['fit_avgi'] = p_avgi
                data['fit_prod_avgi_err'] = np.sqrt(np.diag(pCov_avgi))
            else: # fitting 3D sweeps
                xproj = np.mean(data['prod_avgi'], axis=0)
                yproj = np.mean(data['prod_avgi'], axis=1)
                fig, axs = plt.subplots(nrows=2, figsize=(8,8))
                m = axs[0].pcolormesh(data['xpts'][0], data['ypts'][0], data['prod_avgi'])
                fig.colorbar(m, ax=axs[0])
                axs[0].set_xlabel('storB advance phase (deg)')
                axs[0].set_ylabel('storA advance phase (deg)')
                axs[0].set_title(self.fname.split(os.path.sep)[-1], fontsize=12)
                axs[1].scatter(data['xpts'][0], xproj, label='storB')
                axs[1].scatter(data['ypts'][0], yproj, label='storA')
                axs[1].legend()
                axs[1].set_xlabel('advance phase (deg)')
                axs[1].set_ylabel('col/row mean')
                if fit_model == 'sg':
                    px, covx = fitter.fitgaussian(data['xpts'][0], xproj)
                    py, covy = fitter.fitgaussian(data['ypts'][0], yproj)
                    data['xproj_fit'] = fitter.gaussianfunc(data['xpts'][0], *px)
                    data['yproj_fit'] = fitter.gaussianfunc(data['ypts'][0], *py)
                    data['xproj_popts'] = px
                    data['xproj_err'] = np.sqrt(np.diag(covx))
                    data['yproj_popts'] = py
                    data['yproj_err'] = np.sqrt(np.diag(covy))

                    axs[0].scatter([data['xproj_popts'][2]], [data['yproj_popts'][2]], marker='x', color='r', s=100)
                    axs[1].plot(data['xpts'][0], data['xproj_fit'])
                    axs[1].plot(data['ypts'][0], data['yproj_fit'])
                    print(f'x center: {px[2]:.6f} (err: {np.sqrt(covx[2,2]):.6f})')
                    print(f'y center: {py[2]:.6f} (err: {np.sqrt(covy[2,2]):.6f})')

                elif fit_model == 'dl':
                    def dlorentz(x, amp1, amp2, cen, sep, wid):  # shared width g
                        c1 = cen - sep/2.0
                        c2 = cen + sep/2.0
                        return (amp1 / (1 + ((x-c1)/wid)**2) +
                                amp2 / (1 + ((x-c2)/wid)**2))

                    def fwhm(x, y):
                        halfmax = np.max(y) / 2.0
                        above = np.where(y >= halfmax)[0]
                        if len(above) < 2:
                            return 0  # nothing above halfmax
                        return x[above[-1]] - x[above[0]]

                    def guess_dlorentz(x, y):
                        return dict(
                            amp1 = dict(value=np.max(y), min=0),
                            amp2 = dict(value=np.max(y), min=0),
                            cen = dict(value=x[y.argmax()], min=x[0], max=x[-1]),
                            sep = dict(value=fwhm(x, y), min=0, max=x[-1]-x[0]),
                            wid = dict(value=fwhm(x, y)/2, min=0, max=x[-1]-x[0]),
                        )

                    mod = Model(dlorentz)
                    pars = mod.make_params(**guess_dlorentz(data['xpts'][0], xproj))
                    xresult = mod.fit(xproj, pars, x=data['xpts'][0])
                    xbestfit = xresult.best_fit
                    data['xresult'] = xresult

                    axs[1].plot(data['xpts'][0], xbestfit)
                    axs[1].axvline(xresult.best_values['cen'] - xresult.best_values['sep']/2, color='C0')
                    axs[1].axvline(xresult.best_values['cen'] + xresult.best_values['sep']/2, color='C0')

                    pars = mod.make_params(**guess_dlorentz(data['ypts'][0], yproj))
                    yresult = mod.fit(yproj, pars, x=data['ypts'][0])
                    ybestfit = yresult.best_fit
                    data['yresult'] = yresult
                    axs[1].plot(data['xpts'][0], ybestfit)
                    axs[1].axvline(yresult.best_values['cen'] - yresult.best_values['sep']/2, color='C1')
                    axs[1].axvline(yresult.best_values['cen'] + yresult.best_values['sep']/2, color='C1')

                    axs[0].scatter([xresult.best_values['cen']], [yresult.best_values['cen']], marker='x', color='r', s=100)
                    print(f'x center: {xresult.best_values["cen"]:.6f}')
                    print(f'y center: {yresult.best_values["cen"]:.6f}')
                else:
                    raise ValueError('fit_model must be sg or dl')


    def display(self, data=None, fit=False):
        if data is None:
            data=self.data 
        
        try:
            fig, axs = super().display(data, fit=fit)
            x_sweep = data['xpts']
            xlabel = self.inner_param
        except ValueError:
            plt.close()
            zlen, ylen, xlen = data['avgi'].shape
            fig, axs = plt.subplots(nrows=zlen, figsize=(6,3*zlen))
            for zind, ax in enumerate(axs):
                m = ax.pcolormesh(data['xpts'][0], data['ypts'][0], data['avgi'][zind])
                fig.colorbar(m, ax= ax, label='avgi')
            axs[0].set_title(self.fname.split(os.path.sep)[-1], fontsize=12)
            axs[-1].set_xlabel('storB advance phase (deg)')
            axs[-1].set_ylabel('storA advance phase (deg)')


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

        return fig, axs


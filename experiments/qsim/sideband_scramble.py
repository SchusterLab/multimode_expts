from copy import deepcopy

import numpy as np

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
    Scramble 1 photon via fractional beam splitters

    expt params:
    swap_stors: list of storage modes to apply the floquet swaps to, will go in order of the list
    update_phases: boolean of whether to update each subsequent swap with the calibrated stark shift phase
    """
    def core_pulses(self):
        pulse_args = deepcopy(self.m1s_kwargs[self.cfg.expt.init_stor-1])

        swap_stors = self.cfg.expt.swap_stors
        swap_stor_phases = np.zeros(len(swap_stors))
        update_phases = self.cfg.expt.update_phases
        for kk in range(self.cfg.expt.floquet_cycle):
            for i_stor, stor in enumerate(swap_stors):
                pulse_args = self.m1s_kwargs[stor - 1]
                pulse_args['phase'] = self.deg2reg(swap_stor_phases[i_stor], gen_ch=pulse_args['ch'])
                # print("phase on storage", stor, swap_stor_phases[i_stor])
                self.setup_and_pulse(**pulse_args)
                self.sync_all()

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

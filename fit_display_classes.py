
# Author : Eesh Gupta 
# Date : 2025-05-12
# Description : This file contains classes for analyzing and displaying data from qubit experiments.This is a simpler and cleaned up version of fit_display.py
# # %reload_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import os
import experiments.fitting as fitter
import datetime

class GeneralFitting:
    def __init__(self, data, readout_per_round=2, threshold=-4.0, config=None):
        self.cfg = config
        self.data = data
        self.readout_per_round = readout_per_round
        self.threshold = threshold

    def filter_data_IQ(self, II, IQ, threshold):
        result_Ig = []
        result_Ie = []

        for k in range(len(II) // self.readout_per_round):
            index_4k_plus_2 = self.readout_per_round * k + self.readout_per_round - 2
            index_4k_plus_3 = self.readout_per_round * k + self.readout_per_round - 1

            if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
                if II[index_4k_plus_2] < threshold:
                    result_Ig.append(II[index_4k_plus_3])
                    result_Ie.append(IQ[index_4k_plus_3])

        return np.array(result_Ig), np.array(result_Ie)

    def post_select_raverager_data(self, temp_data, attrs):
        read_num = self.readout_per_round
        rounds = attrs['config']['expt']['rounds']
        reps = attrs['config']['expt']['reps']
        expts = attrs['config']['expt']['expts']
        I_data = np.array(temp_data['idata'])
        Q_data = np.array(temp_data['qdata'])

        I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds * reps * read_num))
        Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds * reps * read_num))

        Ilist = []
        Qlist = []
        for ii in range(len(I_data) - 1):
            Ig, Qg = self.filter_data_IQ(I_data[ii], Q_data[ii], self.threshold)
            Ilist.append(np.mean(Ig))
            Qlist.append(np.mean(Qg))

        Ilist, Qlist
    
    def save_plot(self, fig, filename="plot.png"):
        
        """
        Save a matplotlib figure to the specified folder.
        Optionally append the image path to a markdown file for viewing.

        Parameters:
        - fig: matplotlib.figure.Figure object to save.
        - folder_path: Path to the folder where the plot will be saved.
        - filename: Name of the file (default: "plot.png").
        - markdown_path: Path to a markdown file to append the image (optional).
        """ 
        plots_folder_path = "plots"
        markdown_path = None

        # Extract markdown folder from config if available
        if self.cfg and hasattr(self.cfg, "data_management"):
            
            markdown_folder = getattr(self.cfg.data_management, "plot_and_logs_folder")
            plots_folder_path = markdown_folder + "/plots"
            if markdown_folder:
                os.makedirs(markdown_folder, exist_ok=True)
                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                markdown_path = os.path.join(markdown_folder, f"{today_str}.md")
                if not os.path.exists(markdown_path):
                    with open(markdown_path, "w") as f:
                        f.write(f"# Plots for {today_str}\n\n")

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        print("supertitle is ", fig._suptitle)
        if fig._suptitle is not None:
            fig._suptitle.set_text(fig._suptitle.get_text() + f" | {date_str} - {filename}")
        else:
            fig.suptitle(f"{date_str} - {filename}", fontsize=16)
        #get tight layout
        fig.tight_layout()
        filename = f"{date_str}_{filename}"
        os.makedirs(plots_folder_path, exist_ok=True)
        filepath = os.path.join(plots_folder_path, filename)
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

        if markdown_path is not None:
            # Use relative path if markdown file is in the same folder or subfolder
            rel_path = os.path.relpath(filepath, os.path.dirname(markdown_path))
            md_line = f"![Plot]({rel_path})\n"
            with open(markdown_path, "a") as md_file:
                md_file.write(md_line)
            print(f"Plot path appended to {markdown_path}")




           
class RamseyFitting(GeneralFitting):
    def __init__(self, data, readout_per_round=2, threshold=-4.0, config=None, fitparams=None):
        super().__init__(data, readout_per_round, threshold, config)
        self.fitparams = fitparams
        self.results = {}

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data

        if fit:
            if fitparams is None:
                fitparams = [200, 0.2, 0, 200, None, None]
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            if isinstance(p_avgi, (list, np.ndarray)):
                data['f_adjust_ramsey_avgi'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgi[1], self.cfg.expt.ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)):
                data['f_adjust_ramsey_avgq'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgq[1], self.cfg.expt.ramsey_freq + p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)):
                data['f_adjust_ramsey_amps'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_amps[1], self.cfg.expt.ramsey_freq + p_amps[1]), key=abs)

            self.results = {
                'fit_avgi': p_avgi,
                'fit_avgq': p_avgq,
                'fit_amps': p_amps,
                'fit_err_avgi': pCov_avgi,
                'fit_err_avgq': pCov_avgq,
                'fit_err_amps': pCov_amps
            }
        return data

    def display(self, data=None, fit=True, title_str = 'Ramsey', **kwargs):
        if data is None:
            data = self.data

        qubits = self.cfg.expt.qubits
        checkEF = self.cfg.expt.checkEF
        q = qubits[0]

        f_pi_test = self.cfg.device.qubit.f_ge[q]
        if checkEF:
            f_pi_test = self.cfg.device.qubit.f_ef[q]
        if getattr(self.cfg.expt, "f0g1_cavity", 0) > 0:
            ii = 0
            jj = 0
            if self.cfg.expt.f0g1_cavity == 1:
                ii = 1
                jj = 0
            if self.cfg.expt.f0g1_cavity == 2:
                ii = 0
                jj = 1
            f_pi_test = self.cfg.device.QM.chi_shift_matrix[0][self.cfg.expt.f0g1_cavity] + self.cfg.device.qubit.f_ge[0]

        title = ('EF' if checkEF else '') + 'Ramsey'

        fig = plt.figure(figsize=(10, 9))
        plt.subplot(211,
                    title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
                    ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1], 'o-')
        if fit:
            p = data['fit_avgi']
            if isinstance(p, (list, np.ndarray)):
                pCov = data['fit_err_avgi']
                try:
                    captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                except Exception:
                    print('Fit Failed ; aborting')
                    captionStr = None
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1], 'o-')
        if fit:
            p = data['fit_avgq']
            if isinstance(p, (list, np.ndarray)):
                pCov = data['fit_err_avgq']
                try:
                    captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                except Exception:
                    print('Fit Failed ; aborting')
                    captionStr = None
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')

        plt.tight_layout()
        plt.show()


        #make filename same as titlestr
        filename = title_str.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename)

class AmplitudeRabiFitting(GeneralFitting):
    def __init__(self, data, readout_per_round=2, threshold=-4.0, config=None, fitparams=None):
        super().__init__(data, readout_per_round, threshold, config)
        self.fitparams = fitparams
        self.results = {}

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        """
            Analyze the provided data by fitting decaying sine functions and extracting gain parameters.

            Parameters
            ----------
            data : dict, optional
                The data dictionary containing keys 'xpts', 'avgi', 'avgq', and 'amps'. If None, uses self.data.
            fit : bool, default True
                Whether to perform fitting on the data.
            fitparams : list or None, optional
                List of initial guess parameters for the fit. Should be of the form:
                [amplitude, frequency, phase, decay_time, offset, decay_offset]
                If None, uses default values.
            **kwargs
                Additional keyword arguments (currently unused).

            Returns
            -------
            data : dict
                The input data dictionary, updated with fit results and gain parameters.

            Notes
            -----
            - Fits decaying sine functions to 'avgi', 'avgq', and 'amps' using `fitter.fitdecaysin`.
            - Stores fit parameters and their covariance in the data dictionary.
            - Calculates and stores π and π/2 gain values for 'avgi' and 'avgq' using the fit results.
            - Updates `self.results` with all fit and gain results if fitting is performed.
            """

        if data is None:
            data = self.data

        def get_pi_hpi_gain_from_fit(p):
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if np.abs(p[2] - 90) > np.abs(p[2] + 90):
                pi_gain = (1 / 4 - p[2] / 360) / p[1]
                hpi_gain = (0 - p[2] / 360) / p[1]
            else:
                pi_gain = (3 / 4 - p[2] / 360) / p[1]
                hpi_gain = (1 / 2 - p[2] / 360) / p[1]
            return int(pi_gain), int(hpi_gain)

        if fit:
            xdata = data['xpts']
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            data['pi_gain_avgi'], data['hpi_gain_avgi'] = get_pi_hpi_gain_from_fit(p_avgi)
            data['pi_gain_avgq'], data['hpi_gain_avgq'] = get_pi_hpi_gain_from_fit(p_avgq)

            self.results = {
                'fit_avgi': p_avgi,
                'fit_avgq': p_avgq,
                'fit_amps': p_amps,
                'fit_err_avgi': pCov_avgi,
                'fit_err_avgq': pCov_avgq,
                'fit_err_amps': pCov_amps,
                'pi_gain_avgi': data['pi_gain_avgi'],
                'hpi_gain_avgi': data['hpi_gain_avgi'],
                'pi_gain_avgq': data['pi_gain_avgq'],
                'hpi_gain_avgq': data['hpi_gain_avgq'],
            }
        return data

    def display(self, data=None, fit=True, fitparams=None, vline=None, save_fig = False, title_str = 'AmpRabi', **kwargs):
        if data is None:
            data = self.data

        fig = plt.figure(figsize=(10, 10))
        plt.subplot(211, title=f"Amplitude Rabi (Pulse Length {self.cfg.expt.sigma_test})", ylabel="I [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1], 'o-')
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_gain = data['pi_gain_avgi']
            hpi_gain = data['hpi_gain_avgi']
            print(f'Pi gain from avgi data [dac units]: {pi_gain}')
            print(f'\tPi/2 gain from avgi data [dac units]: {hpi_gain}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(hpi_gain, color='0.2', linestyle='--')
            if vline is not None:
                plt.axvline(vline, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1], 'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_gain = data['pi_gain_avgq']
            hpi_gain = data['hpi_gain_avgq']
            print(f'Pi gain from avgq data [dac units]: {pi_gain}')
            print(f'\tPi/2 gain from avgq data [dac units]: {hpi_gain}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(hpi_gain, color='0.2', linestyle='--')

        plt.tight_layout()
        plt.show()

        if save_fig:
            filename = title_str.replace(' ', '_').replace(':', '') + '.png'
            self.save_plot(fig, filename=filename)

            
class Histogram(GeneralFitting):
    def __init__(self, data, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4.0, config=None):
        super().__init__(data, readout_per_round, threshold, config)
        print(self.data)
        self.span = span
        self.verbose = verbose
        self.active_reset = active_reset
        self.results = {}

    def analyze(self, plot=True):
        if self.active_reset:
            Ig, Qg = self.filter_data_IQ(self.data['Ig'], self.data['Qg'], self.thresholds)
            Ie, Qe = self.filter_data_IQ(self.data['Ie'], self.data['Qe'], self.thresholds)
            plot_f = 'If' in self.data.keys()
            if plot_f:
                If, Qf = self.filter_data_IQ(self.data['If'], self.data['Qf'], self.thresholds)
        else:
            Ig, Qg = self.data['Ig'], self.data['Qg']
            Ie, Qe = self.data['Ie'], self.data['Qe']
            plot_f = 'If' in self.data.keys()
            if plot_f:
                If, Qf = self.data['If'], self.data['Qf']

        numbins = 200
        xg, yg = np.median(Ig), np.median(Qg)
        xe, ye = np.median(Ie), np.median(Qe)
        if plot_f:
            xf, yf = np.median(If), np.median(Qf)

        if self.verbose:
            print('Unrotated:')
            print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
            print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
            if plot_f:
                print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

        theta = -np.arctan2((ye - yg), (xe - xg))
        if plot_f:
            theta = -np.arctan2((ye - yf), (xe - xf))

        Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
        Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)
        Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
        Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)
        print('updating temp data')
        self.data['Ig_rot'] = Ig_new
        self.data['Qg_rot'] = Qg_new
        self.data['Ie_rot'] = Ie_new
        self.data['Qe_rot'] = Qe_new


        if plot_f:
            If_new = If * np.cos(theta) - Qf * np.sin(theta)
            Qf_new = If * np.sin(theta) + Qf * np.cos(theta)

        xg, yg = np.median(Ig_new), np.median(Qg_new)
        xe, ye = np.median(Ie_new), np.median(Qe_new)
        if plot_f:
            xf, yf = np.median(If_new), np.median(Qf_new)

        if self.verbose:
            print('Rotated:')
            print(f'Ig {xg} +/- {np.std(Ig_new)} \t Qg {yg} +/- {np.std(Qg_new)} \t Amp g {np.abs(xg+1j*yg)}')
            print(f'Ie {xe} +/- {np.std(Ie_new)} \t Qe {ye} +/- {np.std(Qe_new)} \t Amp e {np.abs(xe+1j*ye)}')
            if plot_f:
                print(f'If {xf} +/- {np.std(If_new)} \t Qf {yf} +/- {np.std(Qf_new)} \t Amp f {np.abs(xf+1j*yf)}')

        if self.span is None:
            self.span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new)))) / 2
        xlims = [xg - self.span, xg + self.span]

        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        thresholds = [binsg[tind]]
        fids = [contrast[tind]]

        confusion_matrix = [np.cumsum(ng)[tind] / ng.sum(),
                            1 - np.cumsum(ng)[tind] / ng.sum(),
                            np.cumsum(ne)[tind] / ne.sum(),
                            1 - np.cumsum(ne)[tind] / ne.sum()]

        if plot_f:
            contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5 * ng.sum() + 0.5 * nf.sum())))
            tind = contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])

            contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum())))
            tind = contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])

        self.results = {
            'fids': fids,
            'thresholds': thresholds,
            'angle': theta * 180 / np.pi,
            'confusion_matrix': confusion_matrix
        }

        if plot:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
            fig.tight_layout()

            axs[0, 0].scatter(Ie, Qe, label='e', color='r', marker='.', s=1)
            axs[0, 0].scatter(Ig, Qg, label='g', color='b', marker='.', s=1)
            if plot_f:
                axs[0, 0].scatter(If, Qf, label='f', color='g', marker='.', s=1)
            axs[0, 0].scatter(xg, yg, color='k', marker='o')
            axs[0, 0].scatter(xe, ye, color='k', marker='o')
            if plot_f:
                axs[0, 0].scatter(xf, yf, color='k', marker='o')
            axs[0, 0].set_xlabel('I [ADC levels]')
            axs[0, 0].set_ylabel('Q [ADC levels]')
            axs[0, 0].legend(loc='upper right')
            axs[0, 0].set_title('Unrotated')
            axs[0, 0].axis('equal')

            axs[0, 1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.', s=1)
            axs[0, 1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.', s=1)
            if plot_f:
                axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.', s=1)
            axs[0, 1].scatter(xg, yg, color='k', marker='o')
            axs[0, 1].scatter(xe, ye, color='k', marker='o')
            if plot_f:
                axs[0, 1].scatter(xf, yf, color='k', marker='o')
            axs[0, 1].set_xlabel('I [ADC levels]')
            axs[0, 1].legend(loc='upper right')
            axs[0, 1].set_title('Rotated')
            axs[0, 1].axis('equal')

            axs[1, 0].hist(Ig_new, bins=numbins, range=xlims, alpha=0.5, label='g', color='blue', density=True)
            axs[1, 0].hist(Ie_new, bins=numbins, range=xlims, alpha=0.5, label='e', color='red', density=True)
            # self.data['Ig_rot'] = Ig_new
            # self.data['Qg_rot'] = Qg_new
            if plot_f:
                axs[1, 0].hist(If_new, bins=numbins, range=xlims, alpha=0.5, label='f', color='green', density=True)
            axs[1, 0].axvline(thresholds[0], color='black', linestyle='--', label='Threshold ge')
            if len(thresholds) > 1:
                axs[1, 0].axvline(thresholds[1], color='gray', linestyle='--', label='Threshold gf')
                axs[1, 0].axvline(thresholds[2], color='brown', linestyle='--', label='Threshold ef')
            axs[1, 0].set_title(f'Histogram (Fidelity g-e: {100 * fids[0]:.3}%)')
            axs[1, 0].set_xlabel('I [ADC levels]')
            axs[1, 0].set_ylabel('Density')
            axs[1, 0].legend()

            binsg = np.linspace(xlims[0], xlims[1], numbins + 1)
            ng, _ = np.histogram(Ig_new, bins=binsg, density=True)
            ne, _ = np.histogram(Ie_new, bins=binsg, density=True)
            cumsum_g = np.cumsum(ng) / np.sum(ng)
            cumsum_e = np.cumsum(ne) / np.sum(ne)
            axs[1, 1].plot(binsg[:-1], cumsum_g, label='g', color='blue')
            axs[1, 1].plot(binsg[:-1], cumsum_e, label='e', color='red')
            if plot_f:
                nf, _ = np.histogram(If_new, bins=binsg, density=True)
                cumsum_f = np.cumsum(nf) / np.sum(nf)
                axs[1, 1].plot(binsg[:-1], cumsum_f, label='f', color='green')
            axs[1, 1].axvline(thresholds[0], color='black', linestyle='--', label='Threshold ge')
            if len(thresholds) > 1:
                axs[1, 1].axvline(thresholds[1], color='gray', linestyle='--', label='Threshold gf')
                axs[1, 1].axvline(thresholds[2], color='brown', linestyle='--', label='Threshold ef')
            axs[1, 1].set_title('Cumulative Counts')
            axs[1, 1].set_xlabel('I [ADC levels]')
            axs[1, 1].set_ylabel('Cumulative Density')
            axs[1, 1].legend()

            plt.subplots_adjust(hspace=0.25, wspace=0.15)
            plt.show()
            # save into a file ;
            self.save_plot(fig, filename="histogram.png")
            

            

class Spectroscopy(GeneralFitting):
    def __init__(self, data, signs=[1, 1, 1], config=None):
        super().__init__(data, readout_per_round=2, threshold=-4.0, config=config)
        self.signs = signs


    def analyze(self, fit=True):
        xdata = self.data['xpts'][1:-1]
        if fit:
            self.data['fit_amps'], self.data['fit_err_amps'] = fitter.fitlor(xdata, self.signs[0] * self.data['amps'][1:-1])
            self.data['fit_avgi'], self.data['fit_err_avgi'] = fitter.fitlor(xdata, self.signs[1] * self.data['avgi'][1:-1])
            self.data['fit_avgq'], self.data['fit_err_avgq'] = fitter.fitlor(xdata, self.signs[2] * self.data['avgq'][1:-1])

    def display(self, title='Qubit Spectroscopy', vlines=None, fit=True):
        xpts = self.data['xpts'][1:-1]

        plt.figure(figsize=(9, 11))
        plt.subplot(311, title=title, ylabel="Amplitude [ADC units]")
        plt.plot(xpts, self.data["amps"][1:-1], 'o-')
        if fit and 'fit_amps' in self.data:
            plt.plot(xpts, self.signs[0] * fitter.lorfunc(self.data["xpts"][1:-1], *self.data["fit_amps"]))
            print(f'Found peak in amps at [MHz] {self.data["fit_amps"][2]}, HWHM {self.data["fit_amps"][3]}')
        if vlines:
            for vline in vlines:
                plt.axvline(vline, c='k', ls='--')

        plt.subplot(312, ylabel="I [ADC units]")
        plt.plot(xpts, self.data["avgi"][1:-1], 'o-')
        if fit and 'fit_avgi' in self.data:
            plt.plot(xpts, self.signs[1] * fitter.lorfunc(self.data["xpts"][1:-1], *self.data["fit_avgi"]))
            print(f'Found peak in I at [MHz] {self.data["fit_avgi"][2]}, HWHM {self.data["fit_avgi"][3]}')
        if vlines:
            for vline in vlines:
                plt.axvline(vline, c='k', ls='--')

        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, self.data["avgq"][1:-1], 'o-')
        if fit and 'fit_avgq' in self.data:
            plt.plot(xpts, self.signs[2] * fitter.lorfunc(self.data["xpts"][1:-1], *self.data["fit_avgq"]))
            print(f'Found peak in Q at [MHz] {self.data["fit_avgq"][2]}, HWHM {self.data["fit_avgq"][3]}')
        if vlines:
            for vline in vlines:
                plt.axvline(vline, c='k', ls='--')

        plt.tight_layout()
        plt.show()

class LengthRabiFitting(GeneralFitting):
    def __init__(self, data, fit=True, fitparams=None, normalize=[False, 'g_data', 'e_data'], vlines=None, title='length_rabi',
                    active_reset=False, readout_per_round=4, threshold=-4.0, fit_sin=False, config=None):
        super().__init__(data, readout_per_round, threshold, config)
        self.fit = fit
        self.fitparams = fitparams
        self.normalize = normalize
        self.vlines = vlines
        self.title = title
        self.active_reset = active_reset
        self.fit_sin = fit_sin
        self.results = {}

    def analyze(self):
        xlist = self.data['xpts'][0:-1]
        if self.active_reset:
            try:
                Ilist, Qlist = self.post_select_averager_data(self.data['Idata'][:-1], self.threshold, self.readout_per_round)
            except KeyError:
                Ilist, Qlist = self.post_select_averager_data(self.data['idata'][:-1], self.threshold, self.readout_per_round)
        else:
            Ilist = self.data["avgi"][0:-1]
            Qlist = self.data["avgq"][0:-1]

        fit_func = fitter.fitsin if self.fit_sin else fitter.fitdecaysin
        func = fitter.sinfunc if self.fit_sin else fitter.decaysin

        p_avgi, pCov_avgi = fit_func(xlist, Ilist, fitparams=self.fitparams)
        p_avgq, pCov_avgq = fit_func(xlist, Qlist, fitparams=self.fitparams)

        self.data['fit_avgi'] = p_avgi
        self.data['fit_avgq'] = p_avgq
        self.data['fit_err_avgi'] = pCov_avgi
        self.data['fit_err_avgq'] = pCov_avgq

        self.results = {
            'fit_avgi': p_avgi,
            'fit_avgq': p_avgq,
            'fit_err_avgi': pCov_avgi,
            'fit_err_avgq': pCov_avgq
        }

    def display(self, return_fit_params=False, title_str='Length Rabi', **kwargs):
        xlist = self.data['xpts'][0:-1]
        xpts_ns = self.data['xpts'] * 1e3
        Ilist = self.data["avgi"][0:-1]
        Qlist = self.data["avgq"][0:-1]

        func = fitter.sinfunc if self.fit_sin else fitter.decaysin

        fig = plt.figure(figsize=(10, 8))

        axi = plt.subplot(211, title=self.title, ylabel="I [adc level]")
        plt.plot(xpts_ns[1:-1], Ilist[1:], 'o-')
        if self.fit:
            p = self.data['fit_avgi']
            plt.plot(xpts_ns[0:-1], func(xlist, *p))
            pi_length, pi2_length = self._calculate_pi_lengths(p)
            self.results['pi_length'] = pi_length
            self.results['pi2_length'] = pi2_length
            plt.axvline(pi_length * 1e3, color='0.2', linestyle='--', label='pi')
            plt.axvline(pi2_length * 1e3, color='0.2', linestyle='--', label='pi/2')
            if self.vlines:
                for vline in self.vlines:
                    plt.axvline(vline, color='r', ls='--')

        axq = plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[1:-1], Qlist[1:], 'o-')
        if self.fit:
            p = self.data['fit_avgq']
            plt.plot(xpts_ns[0:-1], func(xlist, *p))
            pi_length, pi2_length = self._calculate_pi_lengths(p)
            plt.axvline(pi_length * 1e3, color='0.2', linestyle='--', label='pi')
            plt.axvline(pi2_length * 1e3, color='0.2', linestyle='--', label='pi/2')
            if self.vlines:
                for vline in self.vlines:
                    plt.axvline(vline, color='r', ls='--')

        # if self.normalize[0]:
        #     pass
        #     axi, axq = normalize_data(axi, axq, self.data, self.normalize)

        plt.tight_layout()
        plt.legend()
        plt.show()

        if return_fit_params:
            return self.results['fit_avgi'], self.results['fit_err_avgi'], xlist, Ilist
        # save figure 
        filename = title_str.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename)

    #@staticmethod
    def _calculate_pi_lengths(self, p):
        if p[2] > 180:
            p[2] -= 360
        elif p[2] < -180:
            p[2] += 360
        if p[2] < 0:
            pi_length = (1 / 2 - p[2] / 180) / 2 / p[1]
        else:
            pi_length = (3 / 2 - p[2] / 180) / 2 / p[1]
        T = 1/p[1]/2 # TIME PERIOD
        pi2_length = pi_length - T/2
        # pi2_length = pi_length - (1 / (2 * p[1]))
        print('Pi length:', pi_length)
        print('Pi/2 length:', pi2_length)
        return pi_length, pi2_length



class ChevronFitting(GeneralFitting):
    def __init__(self, frequencies, time, response_matrix, config=None):
        super().__init__(data=None, readout_per_round=2, threshold=-4.0, config=config)
        self.frequencies = frequencies
        self.time = time
        self.response_matrix = response_matrix
        self.results = {}

    @staticmethod
    def decaying_sine(t, A, omega, phi, tau, C):
        """
        A decaying sine function.
        t: time
        A: amplitude
        omega: angular frequency
        phi: phase
        tau: decay constant
        C: offset
        """
        return A * np.sin(omega * t + phi) * np.exp(-t / tau) + C

    @staticmethod
    def fit_slice(time, response):
        """
        Fit a decaying sine curve to a single frequency slice.
        time: array of time values
        response: array of response values
        Returns the fitted parameters and the contrast (max - min of the fitted curve).
        """
        initial_guess = [np.ptp(response) / 2, 2 * np.pi / (time[-1] - time[0]), 0, (time[-1] - time[0]) / 2, np.mean(response)]
        try:
            popt, _ = curve_fit(ChevronFitting.decaying_sine, time, response, p0=initial_guess)
            fitted_curve = ChevronFitting.decaying_sine(time, *popt)
            contrast = np.max(fitted_curve) - np.min(fitted_curve)
            return popt, contrast
        except RuntimeError:
            return None, 0

    def analyze(self):
        """
        Process the 2D data to find the frequency with the largest contrast,
        and then refine the search around it to find the frequency with the longest period.
        """
        max_contrast = 0
        best_frequency_contrast = None
        best_fit_params_contrast = None

        for i, freq in enumerate(self.frequencies):
            response = self.response_matrix[i, :]
            fit_params, contrast = self.fit_slice(self.time, response)
            if fit_params is not None and contrast > max_contrast:
                max_contrast = contrast
                best_frequency_contrast = freq
                best_fit_params_contrast = fit_params

        if best_frequency_contrast is not None:
            index = np.where(self.frequencies == best_frequency_contrast)[0][0]
            start = max(0, index - 2)
            end = min(len(self.frequencies), index + 3)

            longest_period = 0
            best_frequency_period = None
            best_fit_params_period = None

            for i in range(start, end):
                response = self.response_matrix[i, :]
                fit_params, _ = self.fit_slice(self.time, response)
                if fit_params is not None:
                    period = 2 * np.pi / fit_params[1]
                    if period > longest_period:
                        longest_period = period
                        best_frequency_period = self.frequencies[i]
                        best_fit_params_period = fit_params

            self.results = {
                'best_frequency_contrast': best_frequency_contrast,
                'best_frequency_period': best_frequency_period,
                'best_fit_params_contrast': best_fit_params_contrast,
                'best_fit_params_period': best_fit_params_period
            }
        else:
            self.results = {
                'best_frequency_contrast': None,
                'best_frequency_period': None,
                'best_fit_params_contrast': None,
                'best_fit_params_period': None
            }
    def display_results(self, save_fig=False, directory=None, title="chevron_plot.png", vlines=None, hlines=None):
        """
        Display the results of the analysis, including plots. Optionally save the figure.

        Parameters:
        - save_fig (bool): Whether to save the figure. Default is False.
        - directory (str): The directory where the figure will be saved (if save_fig is True).
        - title (str): The filename for the saved figure. Default is "chevron_plot.png".
        - vlines (list or None): List of time values to draw vertical lines on the 2D plot.
        - hlines (list or None): List of frequency values to draw horizontal lines on the 2D plot.
        """
        best_frequency_contrast = self.results.get('best_frequency_contrast')
        best_frequency_period = self.results.get('best_frequency_period')
        best_fit_params_contrast = self.results.get('best_fit_params_contrast')

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(self.time, self.frequencies, self.response_matrix, shading='auto', cmap='viridis')
        plt.colorbar(label='Response')
        if best_frequency_contrast is not None:
            plt.axhline(best_frequency_contrast, color='red', linestyle='--', label=f'Best Contrast: {best_frequency_contrast:.2f} Hz')
        if best_frequency_period is not None:
            plt.axhline(best_frequency_period, color='blue', linestyle='--', label=f'Longest Period: {best_frequency_period:.2f} Hz')
        if hlines is not None:
            for h in hlines:
                plt.axhline(h, color='orange', linestyle=':', label=f'hline: {h}')
        if vlines is not None:
            for v in vlines:
                plt.axvline(v, color='magenta', linestyle=':', label=f'vline: {v}')
        plt.title('2D Color Plot with Chosen Frequencies')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()

        if save_fig and directory:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, title)
            plt.savefig(filepath)
            print(f"Figure saved to {filepath}")

        plt.show()

        if best_fit_params_contrast is not None:
            best_index_contrast = np.argmin(np.abs(self.frequencies - best_frequency_contrast))
            best_response_contrast = self.response_matrix[best_index_contrast, :]
            fitted_curve_contrast = self.decaying_sine(self.time, *best_fit_params_contrast)

            plt.figure(figsize=(10, 6))
            plt.plot(self.time, best_response_contrast, 'b-', label="Original Data (Contrast)")
            plt.plot(self.time, fitted_curve_contrast, 'r--', label="Fitted Curve (Contrast)")
            plt.title(f"Best Fit for Frequency (Contrast) {best_frequency_contrast:.2f} Hz")
            plt.xlabel("Time (s)")
            plt.ylabel("Response")
            plt.legend()
            plt.show()
    

class MM_DualRail: 

    def __init__(self): 
        '''Analysis for dual rail experiments '''
    
    # ------------- Analyis for Storage state in presence of spectator BS ------------

    
    def filter_data_for_si_wrt_spec_BS(self, temp_data, attrs, threshold=None): 
        '''
        Filter data (based on active reset preselection) for single rail wrt to spectator BS
        '''
        if threshold == None:
            threshold = temp_data['thresholds']
        avg_idata = []
        for aa, var in enumerate(temp_data['Idata']):
            var_data, _ =  self.filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], None, threshold,post_selection = False)
            avg_data = np.mean(var_data, axis=0) # average wrt to shots
            # print(len(var_data))
            avg_idata.append(avg_data)
            # print(threshold)
        
        bs_gate_nums = attrs['config']['expt']['bs_gate_nums']
        rb_times = attrs['config']['expt']['rb_times']
        
        return avg_idata, bs_gate_nums, rb_times # the second average is over variations
    
    def extract_ramsey_seq_check_target(self, prev_data, expt_path, file_list, name, 
                                        wrong_way=False):
        '''
        Extract the data for the ramsey sequence as a function of gates 

        wrong_way: averaging over a single depth; ignoring that within a given depth, multiple times are covered
        '''

        var_datas = []
        # depth_list = []
        bs_gate_numss = []
        rb_timess = []

        wrong_fids = []
        wrong_bs_gate_nums = []
        wrong_times = []
        depth_list = []
        threshold = 0
        fnot_found_err_bool  = False

        for idx, file_no in enumerate(file_list): 
            try: 
                full_name = str(file_no).zfill(5)+name
                # print(full_name)
                temp_data, attrs = prev_data(expt_path, full_name)
                # analysis = MM_DualRail_Analysis()

                if attrs['config']['expt']['calibrate_single_shot']:
                    threshold = temp_data['thresholds']

                avg_idata, bs_gate_nums, rb_times = self.filter_data_for_si_wrt_spec_BS(temp_data, attrs, threshold)
                var_datas+= avg_idata
                bs_gate_numss += bs_gate_nums
                rb_timess += rb_times

                wrong_fids.append(np.average(avg_idata))
                wrong_bs_gate_nums.append(np.average(bs_gate_nums))
                wrong_times.append(np.average(rb_times))
                depth_list.append(attrs['config']['expt']['rb_depth'])

            except FileNotFoundError:
                print('FileNotFoundError')
                continue
        if wrong_way:
            return wrong_times, wrong_bs_gate_nums, wrong_fids, depth_list
        
        return self.reorganize_var_data_for_ramsey(var_datas, bs_gate_numss, rb_timess, attrs)
        
    def reorganize_var_data_for_ramsey(self, var_datas, bs_gate_numss, rb_timess, attrs, return_df=False, len_threshold = 0):
        # Re organize data so that we average over all the data points for a given BS gate number

        data = {'bs_gate_nums': bs_gate_numss, 'avg_idata': var_datas, 'rb_times': rb_timess}
        df = pd.DataFrame(data)
        if return_df: 
            return df
        bs_nums_range = np.arange(df['bs_gate_nums'].min(), df['bs_gate_nums'].max() + 1, 1)
        bs_nums_for_plot = [] # List to store the BS gate numbers that have a fidelity
        rb_times_for_plot = []
        fids_for_plot = []

        for idx, bs_gate_num in enumerate(bs_nums_range): 
            df_bs_num = df[df['bs_gate_nums'] == bs_gate_num]
            if len(df_bs_num) > len_threshold:
                # plt.plot(df_bs_num['rb_times'], df_bs_num['avg_idata'], '-o', label='BS gate ' + str(bs_gate_num))
                print('len of df_bs_num', len(df_bs_num))
                fids_for_plot.append(np.average(df_bs_num['avg_idata'].values))
                bs_nums_for_plot.append(bs_gate_num)
                rb_times_for_plot.append(np.average(df_bs_num['rb_times'].values))
            # print(f'Average idata for BS gate {bs_gate_num} is {avg_idata[idx]}')
        return bs_nums_for_plot, fids_for_plot, rb_times_for_plot, attrs['config']['expt']['wait_freq']
    def Ramsey_display(self, xdata, ydata, ramsey_freq=0.02, fit=True, fitparams = None, 
                       title='Ramsey'):

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = None
            # fitparams=[8, 0.5, 0, 20, None, None]
            p, pCov = fitter.fitdecaysin(xdata, ydata, fitparams=fitparams)
            
            if isinstance(p, (list, np.ndarray)): f_adjust_ramsey_avgi = sorted((ramsey_freq - p[1], ramsey_freq + p[1]), key=abs)

  

        title = title
        plt.figure(figsize=(10,5))
        axi = plt.subplot(111, 
            title=f"{title} (Ramsey Freq: {ramsey_freq} MHz)",
            ylabel="I [ADC level]")
        plt.plot(xdata, ydata,'o-')
        if fit:
            #p = data['fit_avgi']
            # print(p)
            # if isinstance(p, (list, np.ndarray)): 
                # pCov = data['fit_err_avgi']
            captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(xdata, fitter.decaysin(xdata, *p), label=captionStr)
            plt.plot(xdata, fitter.expfunc(xdata, p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.plot(xdata, fitter.expfunc(xdata, p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.legend()
            # print(f'Current pi pulse frequency: {f_pi_test}')
            print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
            if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
            # print('Suggested new pi pulse frequency from fit I [MHz]:\n',
            #         f'\t{f_pi_test + f_adjust_ramsey_avgi[0]}\n',
            #         f'\t{f_pi_test + f_adjust_ramsey_avgi[1]}')
            print(f'T2 Ramsey from fit I [us]: {p[3]}')
            return p[3], np.sqrt(pCov[3][3])
    
    # -------------------------------------------------------------------------

    def plot_rb(self, fids_list , fids_post_list , xlist, 
                gg_list , gg_list_err , ge_list, ge_list_err , 
                eg_list, eg_list_err , ee_list , ee_list_err , ebars_list, ebars_post_list,reset_qubit_after_parity = False, parity_meas = True, 
                title='M1-S4 RB Post selection'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        # Exponential fit subplot
        ax1.errorbar(xlist, fids_list, yerr=ebars_list, fmt='o', label='raw', capsize=5, color=colors[0])
        ax1.errorbar(xlist, fids_post_list, yerr=ebars_post_list, fmt='o', label='post selection', capsize=5, color=colors[1])

        # Fitting
        xpts = xlist
        ypts = fids_list
        fit, err = fitter.fitexp(xpts, ypts, fitparams=None)

        ypts = fids_post_list
        fit_post, err_post = fitter.fitexp(xpts, ypts, fitparams=[None, None, None, None])

        p = fit
        pCov = err
        rel_err = 1 / p[3] / p[3] * np.sqrt(pCov[3][3])
        abs_err = rel_err * np.exp(-1 / fit[3])
        fid = np.exp(-1 / fit[3])
        fid_err = abs_err
        captionStr = f'$t$ fit [gates]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}\nFidelity per gate: {np.exp(-1 / fit[3])*100:.6f} $\pm$ {abs_err*100:.6f} %'

        p_post = fit_post
        pCov_post = err_post
        rel_err_post = 1 / p_post[3] / p_post[3] * np.sqrt(pCov_post[3][3])
        abs_err_post = rel_err_post * np.exp(-1 / fit_post[3])
        fid_post = np.exp(-1 / fit_post[3])
        fid_err_post = abs_err_post
        captionStr_post = f'$t$ fit [gates]: {p_post[3]:.3} $\pm$ {np.sqrt(pCov_post[3][3]):.3}\nFidelity per gate: {np.exp(-1 / fit_post[3])*100:.6f} $\pm$ {abs_err_post*100:.6f}%'

        ax1.plot(xpts, fitter.expfunc(xpts, *fit), label=captionStr, color=colors[0])
        ax1.plot(xpts, [fitter.expfunc(x, *fit_post) for x in xpts], label=captionStr_post, color = colors[1])
        ax1.set_xlabel('Time [us]')
        ax1.set_ylabel('Man1 |1> population')
        #ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title('Exponential Fit')
        ax1.set_xlabel('RB depth')
        ax1.set_ylabel('Man1 |1> population')

        # Shots subplot
        gg_label = '|11>'
        ge_label = '|10>'
        eg_label = '|00>'
        ee_label = '|01>'

        if reset_qubit_after_parity:
            gg_label  = '|11>'
            ge_label = '|10>'
            eg_label = '|01>'
            ee_label = '|00>'
        elif not parity_meas: 
            gg_label  = '|00>'
            ge_label = '|01>'
            eg_label = '|10>'
            ee_label = '|11>'

        ax2.errorbar(xlist, gg_list, yerr=gg_list_err, fmt='-o', label=gg_label, capsize=5)
        ax2.errorbar(xlist, ge_list, yerr=ge_list_err, fmt='-o', label=ge_label, capsize=5)
        ax2.errorbar(xlist, eg_list, yerr=eg_list_err, fmt='-o', label=eg_label, capsize=5)
        ax2.errorbar(xlist, ee_list, yerr=ee_list_err, fmt='-o', label=ee_label, capsize=5)

        # print number of 10 and 01 counts
        # print(str(gg_label) + ' counts:', gg_list)
        # print(str(ge_label) + ' counts:', ge_list)
        # print(str(eg_label) + ' counts:', eg_list)
        # print(str(ee_label) + ' counts:', ee_list)

        ax2.set_yscale('log')
        ax2.legend()
        ax2.set_title('Shots')
        ax2.set_xlabel('RB depth')
        ax2.set_ylabel('Population Ratio')

        # Main title
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        return fid, fid_err, fid_post, fid_err_post
    
    def show_rb(self, prev_data, expt_path, file_list, name = '_SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep.h5', title = 'RB', 
                dual_rail_spec = False, skip_spec_state_idx = None):
        '''show the rb result for a list of files
        
        Args: dual_rail_spec: if True, then we use that rb data extract function 
        skip_spec_state_idx: if dual_rail_spec is True, then we skip the state index in the list
        '''
        from numpy.linalg import inv

        Pgg = 0.997573060976843
        Pge = 0.0024269390231570487
        Peg = 0.012984367839503025
        Pee = 0.9870156321604969

        print('Doing incorrect SIngle shot corretcion')


        P_matrix = np.matrix([[Pgg, Peg],[Pge, Pee]])
        conf_matrix = inv(P_matrix)

        tensor_product_matrix = np.kron(conf_matrix, conf_matrix)

        fids_list = []
        fids_post_list = []
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        gg_list_err = []
        ge_list_err = []
        eg_list_err = []
        ee_list_err = []
        xlist = []
        depth_list = []
        ebars_list = []
        ebars_post_list = []

        for file_no in file_list:
            # extract data from file 
            full_name = str(file_no).zfill(5)+name
            temp_data, attrs = prev_data(expt_path, full_name)  
            if not dual_rail_spec: # if not a photon in a spectator mode
                avg_readout, avg_readout_post, gg, ge, eg, ee = self.RB_extract_postselction_excited(temp_data,attrs, active_reset=True, conf_matrix=tensor_product_matrix)#, start_idx = start_idx)
            else: 
                avg_readout, avg_readout_post, gg, ge, eg, ee = self.RB_extract_postselction_excited_dual_rail_spec(temp_data,attrs, active_reset=True, conf_matrix=tensor_product_matrix, skip_spec_states_idx = skip_spec_state_idx)
            # print(gg[0]+ge[0]+eg[0]+ee[0])
            # print number of gg shots
            
            gg_list.append(np.average(gg))
            ge_list.append(np.average(ge))
            eg_list.append(np.average(eg))
            ee_list.append(np.average(ee))
            fids_list.append(np.average(avg_readout))
            ebars_list.append(np.std(avg_readout)/np.sqrt(len(avg_readout)))
            gg_list_err.append(np.std(ge)/np.sqrt(len(ge)))
            ge_list_err.append(np.std(ge)/np.sqrt(len(ge)))
            eg_list_err.append(np.std(eg)/np.sqrt(len(eg)))
            ee_list_err.append(np.std(ee)/np.sqrt(len(ee)))


            fids_post_list.append(np.average(avg_readout_post))
            ebars_post_list.append(np.std(avg_readout_post)/np.sqrt(len(avg_readout_post)))
            xlist.append(attrs['config']['expt']['rb_depth']*attrs['config']['expt']['bs_repeat'])
            depth_list.append(attrs['config']['expt']['rb_depth']*attrs['config']['expt']['bs_repeat'])

        try: 
            reset_bool = (attrs['config']['expt']['reset_qubit_after_parity'] or attrs['config']['expt']['reset_qubit_via_active_reset_after_first_meas'])
        except KeyError:
            reset_bool = attrs['config']['expt']['reset_qubit_after_parity']
        fid, fid_err, fid_post, fid_post_err = self.plot_rb(fids_list = fids_list, fids_post_list = fids_post_list, xlist=depth_list, 
                    gg_list = gg_list, gg_list_err = gg_list_err, ge_list = ge_list, ge_list_err = ge_list_err, 
                    eg_list = eg_list, eg_list_err = eg_list_err, ee_list = ee_list, ee_list_err = ee_list_err,
                    ebars_list=ebars_list, ebars_post_list=ebars_post_list, reset_qubit_after_parity = reset_bool,
                    parity_meas=attrs['config']['expt']['parity_meas'],
                    title=title)
        return fids_list, fids_post_list, gg_list, ge_list, eg_list, ee_list, gg_list_err, ge_list_err, eg_list_err, ee_list_err, xlist, depth_list, ebars_list, ebars_post_list, fid, fid_err, fid_post, fid_post_err
    def RB_extract_excited(self, temp_data):
        avg_readout = []
        for i in range(len(temp_data['Idata'])):
            counting = 0
            for j in temp_data['Idata'][i]:
                if j>temp_data['thresholds']:
                    counting += 1
            avg_readout.append(counting/len(temp_data['Idata'][i]))
        return avg_readout

    def RB_extract_ground(self, temp_data):
        avg_readout = []
        for i in range(len(temp_data['Idata'])):
            counting = 0
            for j in temp_data['Idata'][i]:
                if j<temp_data['thresholds']:
                    counting += 1
            avg_readout.append(counting/len(temp_data['Idata'][i]))
        return avg_readout


    
    def RB_extract_postselction_excited_old(self, temp_data):
        # remember the parity mapping rule:
        # 00 -> eg, 01 -> ee, 10 -> ge, 11 -> gg
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []
        for aa in range(len(temp_data['Idata'])):
            gg = 0
            ge = 0
            eg = 0
            ee = 0
            for j in range(len(temp_data['Idata'][aa][0])):
                #  check if the counts are the same as initial counts
                if temp_data['Idata'][aa][0][j]>temp_data['thresholds'][0]: # classified as e
                    if temp_data['Idata'][aa][1][j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if temp_data['Idata'][aa][1][j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))
            fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
            fid_post_list.append(ge/(ge+ee))
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list
    def filter_data_BS(self, a1, a2, a3, threshold, post_selection = False):
        # assume the last one  is experiment data, the last but one is for post selection
        '''
        This is for active reset post selection 

        the post selection parameter DOES not refer to active reset post selection
        a1: from active reset pre selection 
        a2: from actual experiment
        a3: from actual experiment post selection
        '''
        result_1 = []
        result_2 = []
        
        
        for k in range(len(a1)):

            if a1[k] < threshold:

                result_1.append(a2[k])
                if post_selection:
                    result_2.append(a3[k])
        
        return np.array(result_1), np.array(result_2)
    def RB_extract_postselction_excited(self, temp_data, attrs, active_reset = False, conf_matrix = None):
        # remember the parity mapping rule:
        # 00 -> eg, 01 -> ee, 10 -> ge, 11 -> gg # NOT active-reset-after-first-meas case (should have some thing to indicate this)
        # 00 -> gg, 01 -> ge, 10 -> eg, 11 -> ee # active-reset-after-first-meas case
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []

        threshold = 0 # for g, e assignment 
        if 'thresholds' in temp_data.keys():
            threshold = temp_data['thresholds'][0]
        else:
            threshold = attrs['config']['device']['readout']['threshold'][0]


        for aa in range(len(temp_data['Idata'])):
            gg = 0
            ge = 0
            eg = 0
            ee = 0

            #  post selection due to active reset
            if active_reset:
                data_init, data_post_select = self.filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], temp_data['Idata'][aa][4], temp_data['thresholds'],post_selection = True)
            else: 
                data_init = temp_data['Idata'][aa][0]
                data_post_select = temp_data['Idata'][aa][1]
            
            # print('len data_init', len(data_init))
            # print('len data_post_select', len(data_post_select))
            
            # beamsplitter post selection 
            for j in range(len(data_init)):
                #  check if the counts are the same as initial counts
                if data_init[j]>threshold: # classified as e
                    if data_post_select[j]>threshold:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if data_post_select[j]>threshold:  # second e
                        ge +=1
                    else:
                        gg += 1
            # print('gg', gg)
            # print('ge', ge)
            # print('eg', eg)
            # print('ee', ee)
            # print('total', eg + ge + gg + ee)
            if conf_matrix is not None: ## correct counts from histogram
                gg = gg * conf_matrix[0,0] + ge * conf_matrix[0,1] + eg * conf_matrix[0,2] + ee * conf_matrix[0,3]
                ge = gg * conf_matrix[1,0] + ge * conf_matrix[1,1] + eg * conf_matrix[1,2] + ee * conf_matrix[1,3]
                eg = gg * conf_matrix[2,0] + ge * conf_matrix[2,1] + eg * conf_matrix[2,2] + ee * conf_matrix[2,3]
                ee = gg * conf_matrix[3,0] + ge * conf_matrix[3,1] + eg * conf_matrix[3,2] + ee * conf_matrix[3,3]
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))

            print('gg_list', gg_list)
            print('ge_list', ge_list)
            print('eg_list', eg_list)
            print('ee_list', ee_list)

            try:
                if attrs['config']['expt']['reset_qubit_after_parity']:
                    # print('reset_qubit_after_parity')
                    # print('using new method to calculate post selection fidelity ')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                elif not attrs['config']['expt']['parity_meas']: 
                    # print('not parity_meas')
                    fid_raw_list.append((ee+eg)/(eg+ge+gg+ee))
                    print('ge', ge) 
                    print('eg', eg)
                    print('ee', ee)
                    print('gg', gg)
                    fid_post_list.append(eg/(ge+eg))
                elif attrs['config']['expt']['reset_qubit_via_active_reset_after_first_meas']:
                    # print('reset_qubit_via_active_reset_after_first_meas')
                    
                                            # gg_label = '|11>'
                                                # ge_label = '|10>'
                                                # eg_label = '|01>'
                                                # ee_label = '|00>'
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                else:
                    # print('using old method to calculate post selection fidelity ')
                    # print('old method')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+ee))
            except KeyError:
                print('using old method to calculate post selection fidelity ')
                fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                fid_post_list.append(ge/(ge+ee))
        print(eg + ge + gg + ee)
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list

    def RB_extract_postselction_excited_dual_rail_spec(self, temp_data, attrs, active_reset = False, conf_matrix = None,
                                        skip_spec_states_idx = None):
        '''
        This is specially for dual rail spectator analysis where we skip over 0 and 1 population of the spectator mode
        '''
        # remember the parity mapping rule:
        # 00 -> eg, 01 -> ee, 10 -> ge, 11 -> gg
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []

        
        for aa in range(len(temp_data['Idata'])):
            if aa%6 in skip_spec_states_idx:
                continue

            gg = 0
            ge = 0
            eg = 0
            ee = 0

            #  post selection due to active reset
            if active_reset:
                data_init, data_post_select = self.filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], temp_data['Idata'][aa][4], temp_data['thresholds'],post_selection = True)
            else: 
                data_init = temp_data['Idata'][aa][0]
                data_post_select = temp_data['Idata'][aa][1]
            
            # print('len data_init', len(data_init))
            # print('len data_post_select', len(data_post_select))
            
            # beamsplitter post selection 
            for j in range(len(data_init)):
                #  check if the counts are the same as initial counts
                if data_init[j]>temp_data['thresholds'][0]: # classified as e
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            # print('gg', gg)
            # print('ge', ge)
            # print('eg', eg)
            # print('ee', ee)
            # print('total', eg + ge + gg + ee)
            if conf_matrix is not None: ## correct counts from histogram
                gg = gg * conf_matrix[0,0] + ge * conf_matrix[0,1] + eg * conf_matrix[0,2] + ee * conf_matrix[0,3]
                ge = gg * conf_matrix[1,0] + ge * conf_matrix[1,1] + eg * conf_matrix[1,2] + ee * conf_matrix[1,3]
                eg = gg * conf_matrix[2,0] + ge * conf_matrix[2,1] + eg * conf_matrix[2,2] + ee * conf_matrix[2,3]
                ee = gg * conf_matrix[3,0] + ge * conf_matrix[3,1] + eg * conf_matrix[3,2] + ee * conf_matrix[3,3]
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))

            # print('gg_list', gg_list)
            # print('ge_list', ge_list)
            # print('eg_list', eg_list)
            # print('ee_list', ee_list)

            try:
                if attrs['config']['expt']['reset_qubit_after_parity']:
                    # print('reset_qubit_after_parity')
                    # print('using new method to calculate post selection fidelity ')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                elif not attrs['config']['expt']['parity_meas']: 
                    # print('not parity_meas')
                    fid_raw_list.append((ee+eg)/(eg+ge+gg+ee))
                    fid_post_list.append(eg/(ge+eg))
                elif attrs['config']['expt']['reset_qubit_via_active_reset_after_first_meas']:
                    # print('reset_qubit_via_active_reset_after_first_meas')
                    
                                            # gg_label = '|11>'
                                                # ge_label = '|10>'
                                                # eg_label = '|01>'
                                                # ee_label = '|00>'
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                else:
                    # print('using old method to calculate post selection fidelity ')
                    # print('old method')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+ee))
            except KeyError:
                print('using old method to calculate post selection fidelity ')
                fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                fid_post_list.append(ge/(ge+ee))
        print(eg + ge + gg + ee)
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list

    def RB_extract_postselction_parity_fixed_excited(self, temp_data, active_reset = False):
        # remember the parity mapping rule:
        # 00 -> ee, 01 -> eg, 10 -> ge, 11 -> gg # with active-reset-after-first-meas case
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []


        for aa in range(len(temp_data['Idata'])):
            gg = 0
            ge = 0
            eg = 0
            ee = 0

            #  post selection due to active reset
            if active_reset:
                data_init, data_post_select = filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], temp_data['Idata'][aa][4], temp_data['thresholds'],post_selection = True)
            else: 
                data_init = temp_data['Idata'][aa][0]
                data_post_select = temp_data['Idata'][aa][1]

            # print(data_init)
            # print(data_post_select)
            
            # beamsplitter post selection 
            for j in range(len(data_init)):
                #  check if the counts are the same as initial counts
                if data_init[j]>temp_data['thresholds'][0]: # classified as e
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))
            
            fid_raw_list.append((gg+ge)/(eg+ge+gg+ee))
            fid_post_list.append(ge/(ge+eg))
        print(eg + ge + gg + ee)
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list


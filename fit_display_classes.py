
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

class GeneralFitting:
    def __init__(self, data, readout_per_round=2, threshold=-4.0):
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

        return Ilist, Qlist




class Histogram(GeneralFitting):
    def __init__(self, data, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4.0):
        # Corrected to match the parent class's __init__ signature
        # pass
        super().__init__(data, readout_per_round, threshold)
        print(self.data)
        # self.data = data
        self.span = span
        self.verbose = verbose
        self.active_reset = active_reset
        self.results = {}
        # self.readout_per_round = readout_per_round
        # self.threshold = threshold

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

class Spectroscopy(GeneralFitting):
    def __init__(self, data, signs=[1, 1, 1]):
        super().__init__(data, readout_per_round=2, threshold=-4.0)
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
                 active_reset=False, readout_per_round=4, threshold=-4.0, fit_sin=False):
        super().__init__(data, readout_per_round, threshold)
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

    def display(self, return_fit_params=False):
        xlist = self.data['xpts'][0:-1]
        xpts_ns = self.data['xpts'] * 1e3
        Ilist = self.data["avgi"][0:-1]
        Qlist = self.data["avgq"][0:-1]

        func = fitter.sinfunc if self.fit_sin else fitter.decaysin

        plt.figure(figsize=(10, 8))

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
    def __init__(self, frequencies, time, response_matrix):
        super().__init__(data=None,     readout_per_round=2, threshold=-4.0)
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
    def display_results(self, save_fig=False, directory=None, title="chevron_plot.png"):
        """
        Display the results of the analysis, including plots. Optionally save the figure.

        Parameters:
        - save_fig (bool): Whether to save the figure. Default is False.
        - directory (str): The directory where the figure will be saved (if save_fig is True).
        - title (str): The filename for the saved figure. Default is "chevron_plot.png".
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
    

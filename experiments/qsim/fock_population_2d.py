"""Population versus time and detuning, with FFT and oscillation-frequency fits.

``detuningss`` is a list of per-storage detuning lists in MHz, for example
``[[-.02], [0.], [.02]]`` for M1 coupled to one storage.
Preparation, Floquet playback, readout and population analysis are inherited.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from experiments.qsim.fock_population import (
    FockPopulationExperiment,
    FockPopulationProgram,
    mode_name,
)
from fitting.fitting import sinfunc
from theory.qsim.central_boson import fixed_n_hamiltonian


class FockPopulation2DExperiment(FockPopulationExperiment):
    """One occupation and read mode; maps have shape (time, detuning).

    analyze() uses corrected populations when available; raw=True selects raw.
    cycle_us defaults to the compiled/saved timing. For M1 plus one storage,
    both frequency estimates fit coupling and an additive residual detuning.
    One photon uses sqrt((detuning + residual)**2 + 4*coupling**2).
    With multiple photons, Kerr defaults to the saved hardware config; the
    comparison then uses the strongest population line of the existing model.
    """
    default_program = FockPopulationProgram

    def sweep_axes(self):
        # The existing driver reads each sweep grid from key + "s".
        # Each detuningss entry becomes expt.detunings for the unchanged Program.
        self.cfg.expt.swept_params = ["floquet_cycle", "detunings"]
        return [(key, self.cfg.expt[key + "s"]) for key in self.cfg.expt.swept_params]

    def analyze(self, data=None, raw=False, fit=True, kerr_mhz=None, cycle_us=None, **kwargs):
        super().analyze(data=data, **kwargs)
        data, expt = self.data, self.cfg.expt
        recorded = self.recorded_derived_params()
        if cycle_us is None:
            cycle_us = recorded["floquet_cycle_us"]
        population_key = "nmod4_mean_corrected" if not raw and "nmod4_mean_corrected" in data else "nmod4_mean"
        time_us = np.asarray(data["ypts"]) * cycle_us
        population = np.asarray(data[population_key]) #shape (time, detuning)
        frequencies = np.fft.rfftfreq(len(time_us), time_us[1] - time_us[0])
        fft_amplitude = 2 * abs(np.fft.rfft(population - population.mean(axis=0), axis=0)) / len(time_us)
        fft_amplitude[0] = 0.
        fft_peaks = frequencies[np.argmax(fft_amplitude[1:], axis=0) + 1]
        sine_parameters = np.full((population.shape[1], 4), np.nan) #shape: (detuning, fit_params: amplitude, frequency_MHz, phase_deg, offset)
        sine_frequency_errors = np.full(population.shape[1], np.nan)
        for column, trace in enumerate(population.T):
            if np.ptp(trace) == 0:
                fft_peaks[column] = np.nan
            elif fit:
                phase = 90. if trace[0] >= np.mean(trace) else -90.
                initial = [np.ptp(trace) / 2, fft_peaks[column], phase, np.mean(trace)] # amplitude, frequency_MHz, phase_deg, offset
                try:
                    parameters, covariance = curve_fit(sinfunc, time_us, trace, p0=initial, x_scale="jac", maxfev=2000,
                                                       bounds=([0., 0., -np.inf, -np.inf], 
                                                               [np.inf, .5 / (time_us[1] - time_us[0]), np.inf, np.inf]))
                    sine_parameters[column] = parameters
                    sine_frequency_errors[column] = np.sqrt(covariance[1, 1])
                except RuntimeError:
                    pass  # Leave this trace's frequency NaN if the fit fails.
        data.update(time_us=time_us, 
                    detuning_MHz=np.asarray(data["xpts"]), 
                    population=population,
                    population_is_corrected=np.asarray(population_key == "nmod4_mean_corrected"),
                    fft_frequency_MHz=frequencies, 
                    fft_amplitude=fft_amplitude,
                    fft_peak_frequency_MHz=fft_peaks, 
                    sine_fit_parameters=sine_parameters,
                    sine_frequency_MHz=sine_parameters[:, 1], 
                    sine_frequency_error_MHz=sine_frequency_errors)
        
        for key in list(data): #this is to delete the previous one if analyze is rerun with fit = False
            if key.startswith(("sine_dispersion_", "fft_dispersion_")):
                del data[key]
                
        if fit and len(expt.swap_stors) == 1 and expt.ro_stor == 0:
            if kerr_mhz is None:
                kerr_mhz = 0. if sum(expt.occupations) <= 1 else float(np.asarray(self.cfg.device.manipulate.kerr).reshape(-1)[0])
            data["theory_kerr_MHz"] = np.asarray(kerr_mhz)
            coupling = recorded["couplings_MHz"][expt.swap_stors[0] - 1]
            data["theory_detuning_MHz"] = np.linspace(min(data["xpts"]), max(data["xpts"]), 300)
            
            def frequency_curve(detuning, coupling, residual):
                return self.two_mode_population_frequency(detuning, coupling, residual, kerr_mhz)
            
            for source in ["sine", "fft"]:
                measured = data["sine_frequency_MHz" if source == "sine" else "fft_peak_frequency_MHz"]
                measured_points = measured > 0  # NaN marks a constant trace or failed sine fit.
                parameters, covariance = curve_fit(frequency_curve, 
                                                   data["detuning_MHz"][measured_points], 
                                                   measured[measured_points],
                                                   p0=[coupling, 0.], 
                                                   bounds=([0., -np.inf], [np.inf, np.inf]), 
                                                   x_scale="jac")
                data[f"{source}_dispersion_parameters"] = parameters  # coupling, residual; MHz
                data[f"{source}_dispersion_covariance"] = covariance
                data[f"{source}_dispersion_curve_MHz"] = frequency_curve(data["theory_detuning_MHz"], *parameters)
        return data

    def two_mode_population_frequency(self, detunings, coupling, residual, kerr_mhz):
        """Analytic one-photon frequency; strongest population line with Kerr."""
        occupations = self.cfg.expt.occupations
        if sum(occupations) == 1 or kerr_mhz == 0:
            return np.sqrt((detunings + residual)**2 + 4 * coupling**2)
        frequencies = []
        for detuning in detunings:
            model = fixed_n_hamiltonian(sum(occupations), 2, [detuning + residual], [coupling], kerr_mhz)
            overlap = model.states[model.fock_index[tuple(occupations)]]
            numbers = np.asarray(model.fock_basis)[:, 0]
            number_matrix = model.states.T @ (numbers[:, None] * model.states)
            population_lines = [(abs(overlap[i] * overlap[j] * number_matrix[i, j]),
                                 model.energies_MHz[j] - model.energies_MHz[i])
                                for i in range(len(overlap)) for j in range(i + 1, len(overlap))]
            frequencies.append(max(population_lines)[1])
        return np.asarray(frequencies)

    def display(self, data=None, **kwargs):
        """Population, FFT, sine-frequency curve and FFT-peak curve, in a 2x2."""
        if data is None:
            data = self.data
        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        detuning_khz = 1e3 * data["detuning_MHz"]
        image = axes[0, 0].pcolormesh(detuning_khz, data["time_us"], data["population"], shading="auto")
        figure.colorbar(image, ax=axes[0, 0], label="Mean n mod 4")
        axes[0, 0].set(title="Population", ylabel="Time (us)")
        image = axes[0, 1].pcolormesh(detuning_khz, 1e3 * data["fft_frequency_MHz"], data["fft_amplitude"], shading="auto")
        figure.colorbar(image, ax=axes[0, 1], label="FFT amplitude")
        axes[0, 1].set(title="Population FFT", ylabel="Frequency (kHz)")
        for source, axis in [("sine", axes[1, 0]), ("fft", axes[1, 1])]:
            frequency_key = "sine_frequency_MHz" if source == "sine" else "fft_peak_frequency_MHz"
            axis.plot(detuning_khz, 1e3 * data[frequency_key], ".", label="Measured")
            if f"{source}_dispersion_curve_MHz" in data:
                coupling, residual = 1e3 * data[f"{source}_dispersion_parameters"]
                axis.plot(1e3 * data["theory_detuning_MHz"], 1e3 * data[f"{source}_dispersion_curve_MHz"],
                          label=f"g = {coupling:.2f}, residual = {residual:.2f} kHz")
            axis.axvline(0., color="gray", linestyle=":", linewidth=1)
            axis.set(title="Sine fitted frequency" if source == "sine" else "FFT peak frequency", ylabel="Frequency (kHz)")
            axis.legend(fontsize=8)
        for axis in axes.flat:
            axis.set_xlabel("Disorder amplitude (kHz)")
        expt = self.cfg.expt
        name = Path(getattr(self, "fname", None) or type(self).__name__).name
        kind = "corrected" if data["population_is_corrected"] else "raw"
        figure.suptitle(f"{list(expt.occupations)}, read {mode_name(expt.ro_stor)} ({kind})\n{name}", fontsize=10)
        figure.tight_layout()
        return figure, axes

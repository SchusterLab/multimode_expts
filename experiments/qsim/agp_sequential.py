"""Sequential AGP: one occupation union per W, with an explicit direction axis.

Population JOBs retain every readout mode. Each W and disorder scan saves all
requested directions together through the usual acquire/analyze/save methods.
"""

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.assembled_data import ASSEMBLED_DIR, experiment_root
from experiments.characterization_runner import json_plain
from experiments.qsim.agp import (
    CentralBosonAGPDisorderScan,
    CentralBosonAGPExperiment,
    calibration_shots,
    direction_sector,
    part_name,
    rms_khz,
    sector_dimension,
)
from experiments.qsim.fock_population import (
    FockPopulationSetExperiment,
    multiparity_moment_error,
)
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
)
from fitting.qsim.agp import (
    agp_kernel_weights,
    calibration_error_factors,
    spectral_function,
    spectrum_frequencies_khz,
    subtract_background,
)
from slab import AttrDict
from theory.qsim.agp import (
    probe_values,
    spectral_agp,
    spectral_weights,
    split_dc,
    two_time_correlation,
)
from theory.qsim.central_boson import fixed_n_hamiltonian


class CentralBosonAGPSequentialExperiment(CentralBosonAGPExperiment):
    """One W: unique initial-state JOBs and all requested AGP directions.

    Population arrays retain (state, cycle, readout step[, label]). Correlations
    have (direction, cycle), and AGP has (direction, cutoff), even for one
    direction. Kerr has no C2; its row is NaN when storage C2 rows are present.
    """

    child_class = FockPopulationSequentialReadoutExperiment

    def __init__(self, photon_sector_number, direction, swap_stors, cycles, detunings,
                 ro_stor=None, disorder_khz=None, calibration=None, notes="", ro_stors=None):
        swap_stors = list(swap_stors)
        self.photon_sector_number = int(photon_sector_number)
        self.directions = ([int(direction)] if isinstance(direction, (int, np.integer))
                           else [int(mode) for mode in direction])
        self.ro_stor = ro_stor
        if ro_stors is None:
            self.ro_stors = [0, *swap_stors]
        elif isinstance(ro_stors, (int, np.integer)):
            self.ro_stors = [int(ro_stors)]
        else:
            self.ro_stors = [int(mode) for mode in ro_stors]
        read_modes = self.directions if ro_stor is None else [ro_stor]
        if any(mode not in self.ro_stors for mode in read_modes):
            raise ValueError("AGP readout modes must be included in ro_stors")
        self.readout_calibration = str(calibration) if isinstance(calibration, Path) else json_plain(calibration)
        occupations = []
        for selected_direction in self.directions:
            for occupation in direction_sector(self.photon_sector_number, swap_stors, selected_direction):
                if tuple(occupation) not in occupations:
                    occupations.append(tuple(occupation))
        FockPopulationSetExperiment.__init__(
            self, [(occupation, self.ro_stors[0]) for occupation in sorted(occupations)],
            cycles, swap_stors, detunings, notes=notes)
        self.disorder_khz = rms_khz(self.detunings) if disorder_khz is None else float(disorder_khz)

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", direction=None, disorder_khz=None,
                      ro_stor=None):
        children = list(children)
        _, cycles, swap_stors, detunings, _ = cls.shared_settings(children)
        first = children[0].cfg.expt
        part = cls(sum(first.occupations), direction, swap_stors, cycles, detunings,
                   ro_stor=ro_stor, disorder_khz=disorder_khz, notes=notes,
                   ro_stors=children[0].data["ro_stors"],
                   calibration=first.get("multiparity_confusion_matrices"))
        part.children, part.job_ids = part._ordered(children, job_ids)
        part._check_children()
        return part

    def job_overrides(self):
        overrides = super().job_overrides()
        for override in overrides:
            override.update(ro_stors=self.ro_stors.copy(), multiparity_confusion_matrix=None,
                            multiparity_confusion_matrices=deepcopy(self.readout_calibration))
        return overrides

    def analyze(self, mu=None, t1_us=None, t1_err_us=None, kerr_mhz=None,
                background_start_us=None, window="hann", **kwargs):
        self._check_children()
        self.shared_settings(self.children)
        self.children, self.job_ids = self._ordered(self.children, self.job_ids)
        for job in self.children:
            if "p_mod" not in job.data:
                job.analyze()
            if list(job.data["ro_stors"]) != self.ro_stors:
                raise ValueError("Sequential AGP jobs must share the same ro_stors order")
        data = dict(occupations=np.asarray([occupation for occupation, _ in self.entries]),
                    directions=np.asarray(self.directions), ro_stors=np.asarray(self.ro_stors),
                    readout_steps=np.asarray([self.ro_stors.index(direction if self.ro_stor is None
                                                                else self.ro_stor)
                                              for direction in self.directions]),
                    cycles=np.asarray(self.cycles), swap_stors=np.asarray(self.swap_stors),
                    detunings=np.asarray(self.detunings), disorder_khz=np.asarray(self.disorder_khz),
                    photon_sector_number=np.asarray(self.photon_sector_number),
                    sector_dimension=np.asarray(sector_dimension(self.photon_sector_number,
                                                                  1 + len(self.swap_stors))))
        fields = ["p_mod", "n_mean", "n2_mean", "n_err", "n2_err", "shot_counts"]
        corrected = ["p_mod_corrected" in job.data for job in self.children]
        if any(corrected) and not all(corrected):
            raise ValueError("Some jobs are readout-corrected and some are not")
        self.calibration = None
        if all(corrected):
            fields += ["p_mod_corrected", "n_mean_corrected", "n2_mean_corrected"]
            matrices = [job.data["multiparity_confusion_matrices"] for job in self.children]
            if any(not np.allclose(matrix, matrices[0]) for matrix in matrices[1:]):
                raise ValueError("Sequential AGP jobs have different readout corrections")
            data["multiparity_confusion_matrices"] = np.asarray(matrices[0]).copy()
            self.calibration = self.children[0].cfg.expt.get("multiparity_confusion_matrices")
        for field in fields:
            data[field] = np.stack([job.data[field] for job in self.children])
        cycle_times = {(job.recorded_derived_params() or {}).get("floquet_cycle_us")
                       for job in self.children}
        if len(cycle_times) == 1 and None not in cycle_times:
            data["time_us"] = data["cycles"] * cycle_times.pop()
        self.data = AttrDict(data)
        self.analysis = dict(mu_khz=json_plain(mu), t1_us=t1_us, t1_err_us=t1_err_us,
                             kerr_mhz=kerr_mhz, background_start_us=background_start_us,
                             window=window)
        for index, direction in enumerate(self.directions):
            result = self._analyze_direction(direction, mu, t1_us, t1_err_us, kerr_mhz,
                                             background_start_us, window)
            for key, value in result.items():
                if key in ("agp_mu", "frequency_khz"):
                    self.data[key] = value
                else:
                    if key not in self.data:
                        self.data[key] = np.full((len(self.directions), *np.shape(value)), np.nan)
                    self.data[key][index] = value
        return self.data

    def _analyze_direction(self, direction, mu=None, t1_us=None, t1_err_us=None,
                           kerr_mhz=None, background_start_us=None, window="hann"):
        """Calculate one direction from the shared arrays, returning numeric results."""
        mode_index = [0, *self.swap_stors].index(direction)
        read_mode = direction if self.ro_stor is None else self.ro_stor
        readout_step = self.ro_stors.index(read_mode)
        first_probe = "kerr" if direction == 0 else "number"
        initial_probe = probe_values(self.data["occupations"], mode_index, first_probe)
        rows = [index for index, value in enumerate(initial_probe) if value > 0]
        occupations = self.data["occupations"][rows]
        raw = self.data["p_mod"][rows, :, readout_step, :]
        counts = self.data["shot_counts"][rows, :, readout_step]
        corrected = "p_mod_corrected" in self.data
        confusion = self.data["multiparity_confusion_matrices"][:, :, readout_step] if corrected else None
        corrected_probabilities = self.data["p_mod_corrected"][rows, :, readout_step, :] if corrected else None
        dimension = sector_dimension(self.photon_sector_number, 1 + len(self.swap_stors))
        numbers = np.arange(4, dtype=float)[:, None]
        shots = calibration_shots(self.calibration) if corrected else None
        if corrected and shots is None:
            warnings.warn("Calibration file not found: the AGP errors are shot-only.", stacklevel=2)

        result = {}
        probes = [(1, "kerr")] if direction == 0 else [(1, "number"), (2, "number_squared")]
        for moment, probe in probes:
            initial_weights = probe_values(occupations, mode_index, probe) / dimension
            measured_values = probe_values(numbers, 0, probe)
            result[f"C{moment}"] = initial_weights @ (raw @ measured_values)
            if corrected:
                result[f"C{moment}_corrected"] = initial_weights @ (corrected_probabilities @ measured_values)
            row_errors = multiparity_moment_error(
                raw.reshape(-1, 4), counts.reshape(-1), measured_values, confusion)
            result[f"C{moment}_err"] = np.sqrt(initial_weights**2 @ row_errors.reshape(raw.shape[:2])**2)
            if moment == 1 and shots is not None:
                row_factors = calibration_error_factors(
                    corrected_probabilities, confusion, shots, measured_values)
                factors = np.zeros((raw.shape[1], 4))
                for row, weight in enumerate(initial_weights):
                    factors += weight * row_factors[row]
                result["C1_calibration_factors"] = factors

        if read_mode != direction or "time_us" not in self.data:
            warnings.warn("AGP and spectrum skipped: need an autocorrelation and recorded timing.", stacklevel=2)
            return result
        time_us = self.data["time_us"]
        correlation = result["C1_corrected" if corrected else "C1"]
        if background_start_us is not None:
            connected, fit = subtract_background(time_us, correlation, result["C1_err"], background_start_us)
            result.update(background_amplitude=np.asarray(fit["amplitude"]),
                          background_rate_per_us=np.asarray(fit["rate_per_us"]),
                          background_start_us=np.asarray(float(background_start_us)))
        frequency_khz = spectrum_frequencies_khz(time_us)
        spectrum, spectrum_err = spectral_function(
            time_us, correlation if background_start_us is None else connected,
            result["C1_err"], frequency_khz, window)
        result.update(frequency_khz=frequency_khz, spectrum=spectrum, spectrum_err=spectrum_err)
        if background_start_us is not None:
            result["spectrum_with_background"] = spectral_function(
                time_us, correlation, result["C1_err"], frequency_khz, window)[0]

        couplings = self.recorded_couplings_mhz()
        if couplings is None:
            warnings.warn("AGP skipped: needs recorded couplings.", stacklevel=2)
            return result
        g_mhz = float(np.mean(np.abs(couplings)))
        cutoffs = self.DEFAULT_MU_KHZ if mu is None else mu
        if isinstance(cutoffs, np.ndarray):
            cutoffs = cutoffs.tolist()
        if isinstance(cutoffs, (int, float, np.integer, np.floating)):
            cutoffs = [cutoffs]
        mu_khz = np.asarray(cutoffs, dtype=float)
        factors = result.get("C1_calibration_factors")
        scale = (2 * np.pi * g_mhz)**2

        def estimate(lifetime_us):
            kernel = scale * np.array([agp_kernel_weights(time_us, cutoff / 1000, lifetime_us)
                                       for cutoff in mu_khz])
            variance = kernel**2 @ result["C1_err"]**2
            if factors is not None:
                variance = variance + ((kernel @ factors)**2).sum(axis=-1)
            return kernel, kernel @ correlation, variance

        kernel, agp, variance = estimate(None)
        result.update(agp_mu=mu_khz, agp=agp, agp_err=np.sqrt(variance),
                      coupling_mhz=np.asarray(g_mhz),
                      mu_gld_khz=np.asarray(1000 * g_mhz * (1 + len(self.swap_stors)) / dimension))
        if background_start_us is not None:
            result.update(agp_connected=kernel @ connected, agp_connected_err=np.sqrt(variance))
        if t1_us is not None:
            _, value, variance = estimate(t1_us)
            if t1_err_us:
                step = 1e-4 * t1_us
                slope = (estimate(t1_us + step)[1] - estimate(t1_us - step)[1]) / (2 * step)
                variance = variance + (slope * t1_err_us)**2
            result.update(agp_t1=value, agp_t1_err=np.sqrt(variance),
                          t1_us=np.asarray(float(t1_us)), t1_err_us=np.asarray(float(t1_err_us or 0)))

        kerr = self.recorded_kerr_mhz() if kerr_mhz is None else float(kerr_mhz)
        if kerr is None:
            warnings.warn("Theory skipped: no Kerr recorded; give kerr_mhz.", stacklevel=2)
            return result
        model = fixed_n_hamiltonian(self.photon_sector_number, 1 + len(self.swap_stors),
                                    self.detunings, couplings, kerr)
        theory_values = probe_values(model.fock_basis, mode_index, first_probe)
        differences, theory_weights = spectral_weights(model, theory_values)
        theory = two_time_correlation(differences, theory_weights, time_us)
        dc = split_dc(differences, theory_weights)[0] if background_start_us is not None else 0.
        result.update(C1_theory=theory, agp_theory_window=kernel @ theory,
                      agp_theory=spectral_agp(differences, theory_weights, mu_khz / 1000, g_mhz),
                      theory_kerr_mhz=np.asarray(kerr),
                      spectrum_theory=spectral_function(
                          time_us, theory - dc, np.zeros(len(theory)), frequency_khz, window)[0])
        return result

    def manifest_parameters(self):
        return dict(photon_sector_number=self.photon_sector_number, directions=self.directions,
                    swap_stors=self.swap_stors, cycles=self.cycles, detunings=self.detunings,
                    ro_stor=self.ro_stor, ro_stors=self.ro_stors, disorder_khz=self.disorder_khz,
                    readout_calibration=self.readout_calibration)

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        parameters = manifest["parameters"]
        return dict(direction=parameters.get("directions", parameters.get("direction")),
                    disorder_khz=parameters["disorder_khz"], ro_stor=parameters.get("ro_stor"))


class CentralBosonAGPSequentialDisorderScan(CentralBosonAGPDisorderScan):
    """One native sequential AGP Experiment per W; arrays have (W, direction, ...)."""

    child_class = CentralBosonAGPSequentialExperiment
    SHARED = ("photon_sector_number", "directions", "ro_stor", "swap_stors", "cycles", "ro_stors")

    @classmethod
    def from_pattern(cls, disorder_khz, pattern, photon_sector_number, direction, swap_stors,
                     cycles, ro_stor=None, calibration=None, notes="", ro_stors=None):
        pattern = np.asarray(pattern, dtype=float)
        rms = np.sqrt(np.mean(pattern**2)) if pattern.size else 0.
        if rms > 0:
            pattern = pattern / rms
        elif any(strength != 0 for strength in disorder_khz):
            raise ValueError("Nonzero disorder needs a nonzero pattern")
        parts = [cls.child_class(photon_sector_number, direction, swap_stors, cycles,
                                (1e-3 * strength * pattern).tolist(), ro_stor=ro_stor,
                                disorder_khz=strength, calibration=calibration,
                                notes=notes, ro_stors=ro_stors) for strength in disorder_khz]
        return cls(parts, notes=notes)

    def acquire(self, runner, name, analyze_kwargs=None, directory=None, batch_size=10,
                **execute_kwargs):
        """Acquire, analyze and save each W, then save the scan under its given name."""
        analyze_kwargs = dict(analyze_kwargs or {})
        if directory is None:
            experiment_path = getattr(getattr(runner, "station", None), "experiment_path", None)
            directory = None if experiment_path is None else Path(experiment_path) / ASSEMBLED_DIR
        if directory is not None:
            self._check_names(name, directory)
        for part in self.children:
            part.acquire(runner, batch_size=batch_size, **execute_kwargs)
            if directory is None:
                directory = experiment_root(part.children[0].fname) / ASSEMBLED_DIR
                self._check_names(name, directory)
            part.analyze(**analyze_kwargs)
            part.save(part_name(name, part.disorder_khz), directory=directory)
        self.analyze(**analyze_kwargs)
        return self.save(name, directory=directory)

    def analyze(self, **kwargs):
        super().analyze(**kwargs)
        self.data["directions"] = self.children[0].data["directions"].copy()
        self.data["ro_stors"] = self.children[0].data["ro_stors"].copy()
        return self.data

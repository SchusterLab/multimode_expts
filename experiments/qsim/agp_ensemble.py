"""AGP from one shot per saved disorder realization and initial Fock state.

Every child JOB uses the same ordered detuning table. Experimental estimates
use only the readout calibration and time kernel; theory is a separate overlay.
"""
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import ASSEMBLED_DIR, AssembledExperiment, experiment_root
from experiments.qsim.agp import (
    CentralBosonAGPDisorderScan, CentralBosonAGPExperiment, calibration_shots,
    direction_sector, part_name, rms_khz, sector_dimension,
)
from experiments.qsim.fock_population import calibration_record, recorded_cycle_us
from experiments.qsim.fock_population_ensemble import FockPopulationEnsembleExperiment
from fitting import plot_style
from fitting.qsim.agp import agp_kernel_weights
from fitting.qsim.agp_ensemble import ensemble_agp
from theory.qsim.agp import probe_values
from theory.qsim.agp_ensemble import ensemble_prediction


class CentralBosonAGPEnsembleExperiment(CentralBosonAGPExperiment):
    """One direction and W, with one population JOB per contributing Fock state."""

    child_class = FockPopulationEnsembleExperiment

    def __init__(self, photon_sector_number, direction, swap_stors, cycles,
                 disorder_detunings_mhz, calibration=None, disorder_khz=None,
                 disorder_seed=None, notes=""):
        AssembledExperiment.__init__(self, notes=notes)
        self.photon_sector_number = int(photon_sector_number)
        self.direction = self.ro_stor = int(direction)
        self.swap_stors = list(swap_stors)
        self.cycles = list(cycles)
        self.disorder_detunings_mhz = np.asarray(disorder_detunings_mhz, dtype=float).tolist()
        self.disorder_seed = disorder_seed
        self.disorder_khz = (rms_khz(disorder_detunings_mhz) if disorder_khz is None
                             else float(disorder_khz))
        self.calibration = calibration_record(calibration)
        self.entries = [(tuple(occupation), self.ro_stor) for occupation in
                        direction_sector(photon_sector_number, swap_stors, direction)]
        if not self.entries:
            raise ValueError("The photon sector has no states contributing to this direction")

    def job_overrides(self):
        """Use the same table, row order and number of shots in every Fock JOB."""
        return [dict(occupations=list(occupation), ro_stor=read_mode,
                     swap_stors=self.swap_stors.copy(), floquet_cycles=self.cycles.copy(),
                     disorder_detunings_mhz=deepcopy(self.disorder_detunings_mhz),
                     disorder_seed=self.disorder_seed, reps=len(self.disorder_detunings_mhz),
                     rounds=1, multiparity_confusion_matrix=deepcopy(self.calibration))
                for occupation, read_mode in self.entries]

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", direction=None, disorder_khz=None):
        children = list(children)
        first = children[0]
        settings = first.cfg.expt
        part = cls(sum(settings.occupations), direction, settings.swap_stors,
                   settings.floquet_cycles, first.data["disorder_detunings_mhz"],
                   calibration=settings.get("multiparity_confusion_matrix"),
                   disorder_khz=disorder_khz, disorder_seed=settings.get("disorder_seed"),
                   notes=notes)
        part.children, part.job_ids = part._ordered(children, job_ids)
        return part

    def _population_data(self):
        """Stack decoded results without losing the realization axis or rejected shots."""
        self._check_children()
        self.children, self.job_ids = self._ordered(self.children, self.job_ids)
        table_keys = ("disorder_detunings_mhz", "disorder_frequency_words", "realization_indices")
        for child in self.children:
            if "readout_labels" not in child.data:
                child.analyze()
            for key in table_keys:
                if not np.array_equal(child.data[key], self.children[0].data[key]):
                    raise ValueError(f"Fock JOBs must share the same {key}")
            if list(child.cfg.expt.floquet_cycles) != self.cycles:
                raise ValueError("Fock JOBs must share the same cycle grid")
            if not np.array_equal(child.data.get("confusion_matrix"),
                                  self.children[0].data.get("confusion_matrix")):
                raise ValueError("Fock JOBs must share the same readout calibration")
        if not np.array_equal(self.disorder_detunings_mhz,
                              self.children[0].data["disorder_detunings_mhz"]):
            raise ValueError("The saved disorder table differs from the requested table")
        cycle_times = {recorded_cycle_us(child) for child in self.children}
        if len(cycle_times) != 1 or None in cycle_times:
            raise ValueError("The Fock JOBs need a shared recorded Floquet cycle duration")
        data = dict(occupations=np.asarray([state for state, _ in self.entries]),
                    ro_stor=np.asarray([mode for _, mode in self.entries]),
                    swap_stors=np.asarray(self.swap_stors), cycles=np.asarray(self.cycles),
                    direction=np.asarray(self.direction), disorder_khz=np.asarray(self.disorder_khz),
                    photon_sector_number=np.asarray(self.photon_sector_number),
                    sector_dimension=np.asarray(sector_dimension(self.photon_sector_number,
                                                                 1 + len(self.swap_stors))))
        data["time_us"] = data["cycles"] * cycle_times.pop()
        for key in table_keys:
            data[key] = np.asarray(self.children[0].data[key]).copy()
        for key in ("readout_labels", "herald_keep", "n_err", "n2_err", "shot_counts"):
            data[key] = np.stack([child.data[key] for child in self.children])
        moment_keys = dict(n_mean="nmod4_mean", n2_mean="nmod4_second_moment",
                           n_mean_corrected="nmod4_mean_corrected",
                           n2_mean_corrected="nmod4_second_moment_corrected")
        for key, child_key in moment_keys.items():
            if all(child_key in child.data for child in self.children):
                data[key] = np.stack([child.data[child_key] for child in self.children])
        data["shot_realization_indices"] = self.children[0].data["shot_realization_indices"].copy()
        if self.disorder_seed is not None:
            data["disorder_seed"] = np.asarray(self.disorder_seed)
        return data

    def analyze(self, mu=None, bootstrap_samples=300, seed=0, theory=True, kerr_mhz=None):
        """Estimate g^2 ||A_mu||^2; mu is in kHz and defaults to g L/D."""
        data = self._population_data()
        couplings = self.recorded_couplings_mhz()
        if couplings is None:
            raise ValueError("AGP needs the recorded M1-storage couplings")
        coupling_mhz = float(np.mean(np.abs(couplings)))
        default_mu = 1000 * coupling_mhz * (1 + len(self.swap_stors)) / data["sector_dimension"]
        mu_khz = np.atleast_1d(np.asarray(default_mu if mu is None else mu, dtype=float))
        probe = self.probes[0][1]
        initial_weights = (probe_values(data["occupations"], self.mode_index, probe)
                           / data["sector_dimension"])
        measured_values = probe_values(np.arange(4)[:, None], 0, probe)
        confusion = None if self.calibration is None else np.asarray(self.calibration["matrix"])
        calibration_counts = calibration_shots(self.calibration)
        data.update(ensemble_agp(data["time_us"], data["readout_labels"], data["herald_keep"],
                                 initial_weights, mu_khz, coupling_mhz, measured_values,
                                 confusion_matrix=confusion,
                                 calibration_shots=calibration_counts,
                                 bootstrap_samples=bootstrap_samples, seed=seed))
        data.update(coupling_mhz=np.asarray(coupling_mhz), mu_gld_khz=np.asarray(default_mu))
        data["calibration_uncertainty_included"] = np.asarray(
            confusion is not None and calibration_counts is not None)
        if confusion is not None:
            data["confusion_matrix"] = confusion
        kerr = self.recorded_kerr_mhz() if kerr_mhz is None else kerr_mhz
        if theory and kerr is not None:
            data.update(ensemble_prediction(self.photon_sector_number,
                                           data["disorder_detunings_mhz"], couplings,
                                           kerr, self.mode_index, data["time_us"], mu_khz, probe))
            kernel = (2 * np.pi * coupling_mhz)**2 * np.array([
                agp_kernel_weights(data["time_us"], cutoff / 1000) for cutoff in mu_khz])
            data["agp_theory_sampled_window"] = kernel @ data["C1_theory"]
        self.analysis = dict(mu_khz=mu_khz.tolist(), bootstrap_samples=bootstrap_samples,
                             seed=seed, theory=theory, kerr_mhz=kerr,
                             uncertainty="Joint-realization sampling and shared readout calibration")
        self.data = AttrDict(data)
        return self.data

    def manifest_parameters(self):
        return dict(photon_sector_number=self.photon_sector_number, direction=self.direction,
                    swap_stors=self.swap_stors, cycles=self.cycles,
                    disorder_detunings_mhz=self.disorder_detunings_mhz,
                    disorder_khz=self.disorder_khz, disorder_seed=self.disorder_seed,
                    calibration=self.calibration)

    def display(self, **kwargs):
        with plot_style.context():
            return super().display(**kwargs)


class CentralBosonAGPEnsembleDisorderScan(CentralBosonAGPDisorderScan):
    """W sweep using one shared, explicitly supplied set of disorder patterns."""

    child_class = CentralBosonAGPEnsembleExperiment

    @classmethod
    def from_patterns(cls, disorder_khz, patterns, photon_sector_number, direction,
                      swap_stors, cycles, calibration=None, disorder_seed=None, notes=""):
        """Scale patterns by W in kHz; do not center or normalize individual rows."""
        patterns = np.asarray(patterns, dtype=float)
        # TODO: this implicit unit conversion (kHz to MHz) is really bad. Needs to be fixed.
        # The parent also uses the same implicit conversion as well
        parts = [cls.child_class(photon_sector_number, direction, swap_stors, cycles,
                                 0.001 * strength * patterns, calibration=calibration, #<---- 0.001 should be removed in the near future.
                                 disorder_khz=strength, disorder_seed=disorder_seed, notes=notes)
                 for strength in disorder_khz]
        return cls(parts, notes=notes)

    def acquire(self, runner, name, analyze_kwargs=None, directory=None, batch_size=1,
                **execute_kwargs):
        """Acquire one JOB per Fock state and save each completed W before proceeding."""
        analysis = dict(analyze_kwargs or {})
        if directory is None:
            experiment_path = getattr(getattr(runner, "station", None), "experiment_path", None)
            if experiment_path is not None:
                directory = Path(experiment_path) / ASSEMBLED_DIR
        if directory is not None:
            self._check_names(name, directory)
        for part in self.children:
            part.acquire(runner, batch_size=batch_size, **execute_kwargs)
            if directory is None:
                directory = experiment_root(part.children[0].fname) / ASSEMBLED_DIR
                self._check_names(name, directory)
            part.analyze(**analysis)
            part.save(part_name(name, part.disorder_khz), directory=directory)
        self._stack_parts()
        return self.save(name, directory=directory)

    def analyze(self, **kwargs):
        for part in self.children:
            part.analyze(**kwargs)
        return self._stack_parts()

    def _stack_parts(self):
        """Collect completed W results without rerunning the per-realization theory."""
        first = self.children[0].data
        data = dict(disorder_khz=np.asarray([part.disorder_khz for part in self.children]))
        for key in ("cycles", "time_us", "swap_stors", "direction", "photon_sector_number",
                    "sector_dimension", "realization_indices", "agp_mu"):
            data[key] = first[key].copy()
        for key in ("disorder_detunings_mhz", "disorder_frequency_words",
                    "C1", "C1_corrected", "C1_err",
                    "C1_covariance", "agp", "agp_err", "agp_sampling_err",
                    "agp_calibration_err", "mu_gld_khz", "C1_theory", "agp_theory",
                    "agp_theory_window", "agp_theory_sampled_window"):
            if all(key in part.data for part in self.children):
                data[key] = np.asarray([part.data[key] for part in self.children])
        self.data = AttrDict(data)
        self.analysis = self.children[0].analysis
        return self.data

    def display(self, mu=None, theory=True):
        """Use the original AGP scan appearance, with one linear panel per cutoff."""
        data = self.data
        columns = (list(range(len(data["agp_mu"]))) if mu is None else
                   [int(np.argmin(abs(data["agp_mu"] - mu)))])
        with plt.style.context(["default", {"font.size": 14}]):
            figure, axes = plt.subplots(1, len(columns), squeeze=False,
                                        figsize=(4 * len(columns), 3.6), constrained_layout=True)
            for axis, column in zip(axes[0], columns):
                axis.errorbar(data["disorder_khz"], data["agp"][:, column],
                              yerr=data["agp_err"][:, column], fmt="o-", ms=4, capsize=2,
                              label="measured")
                if theory and "agp_theory_window" in data:
                    axis.plot(data["disorder_khz"], data["agp_theory_window"][:, column],
                              "x:", color="gray", label="closed system, same window")
                    axis.plot(data["disorder_khz"], data["agp_theory"][:, column],
                              "k--", lw=1, label="closed system, infinite time")
                axis.set(xlabel="Disorder RMS (kHz)", ylabel=r"$g^2\|A_\mu\|^2$",
                         title=rf"$\mu = {data['agp_mu'][column]:g}$ kHz")
                axis.legend(fontsize=7)
        return figure, axes[0]

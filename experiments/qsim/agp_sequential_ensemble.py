"""Sequential disorder-ensemble AGP norms and measured metric tensors.

Each initial Fock state is prepared once per realization and evolution time.
The original disorder patterns survive at W = 0. Metric estimation uses only
the measured labels, matched readout calibration and the time-domain kernel;
closed-system predictions are separate comparison arrays.
"""
from copy import deepcopy
from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.characterization_runner import json_plain
from experiments.job_paths import _reroot, data_root
from experiments.qsim.agp import direction_sector, rms_khz
from experiments.qsim.agp_ensemble import (
    CentralBosonAGPEnsembleDisorderScan,
    CentralBosonAGPEnsembleExperiment,
)
from experiments.qsim.agp_sequential import (
    CentralBosonAGPSequentialDisorderScan,
    CentralBosonAGPSequentialExperiment,
)
from experiments.qsim.fock_population_sequential_ensemble import (
    FockPopulationSequentialEnsembleExperiment,
)
from experiments.qsim.fock_population import FockPopulationSetExperiment
from fitting import plot_style
from fitting.qsim.agp import agp_kernel_weights
from fitting.qsim.agp_metric import ensemble_agp_metric
from theory.qsim.agp_ensemble import ensemble_metric_prediction


def resolve_disorder_patterns(detunings_mhz, disorder_khz, patterns):
    """Preserve supplied patterns, including when every actual detuning is zero."""
    detunings = np.asarray(detunings_mhz, dtype=float)
    if patterns is None:
        if disorder_khz == 0:
            raise ValueError("W = 0 needs the original disorder_patterns to define the W direction")
        patterns = 1000 * detunings / disorder_khz
    patterns = np.asarray(patterns, dtype=float)
    if patterns.shape != detunings.shape or not np.allclose(
            disorder_khz * patterns / 1000, detunings, rtol=0, atol=1e-12):
        raise ValueError("Disorder patterns must match the ordered detuning table at the supplied W")
    return patterns.tolist()


def validate_sequential_ensemble_readouts(directions, readout_modes):
    """Require every selected AGP probe to have an acquired readout mode."""
    if any(direction not in readout_modes for direction in directions):
        raise ValueError("AGP readout modes must be included in ro_stors")


def validate_sequential_ensemble_children(children, read_modes, patterns):
    """Prevent averaging different mode orders, pattern identities or calibrations."""
    first_matrices = children[0].data.get("multiparity_confusion_matrices")
    for child in children:
        if list(child.data["ro_stors"]) != read_modes:
            raise ValueError("Sequential ensemble JOBs must share the same ro_stors order")
        if not np.array_equal(child.data["disorder_patterns"], patterns):
            raise ValueError("Sequential ensemble JOBs must share the original disorder_patterns")
        if not np.array_equal(child.data.get("multiparity_confusion_matrices"), first_matrices):
            raise ValueError("Sequential ensemble JOBs must share the per-step confusion matrices")


def sequential_calibration_counts(child):
    """Read recorded per-step calibration counts without changing any source file."""
    calibration = child.cfg.expt.get("multiparity_confusion_matrices")
    if isinstance(calibration, dict):
        if "calibration_shot_counts" in calibration:
            return np.asarray(calibration["calibration_shot_counts"])
        calibration = calibration.get("fname")
    if not calibration:
        return None
    path = Path(calibration)
    if not path.is_file():
        path = Path(_reroot(str(path), data_root()) or path)
    if path.is_file():
        with h5py.File(path, "r") as handle:
            return np.asarray(handle["calibration_shot_counts"][()])
    warnings.warn("Sequential calibration counts not found; calibration uncertainty is excluded.",
                  stacklevel=2)
    return None


class CentralBosonAGPSequentialEnsembleExperiment(CentralBosonAGPEnsembleExperiment):
    """One W and an occupation union, with a single ensemble parent.

    Occupation setup is copied and adapted from
    CentralBosonAGPSequentialExperiment.__init__ to avoid diamond inheritance.
    Its stateless plotting method is delegated
    explicitly rather than inheriting a second AGP Experiment hierarchy.
    """

    child_class = FockPopulationSequentialEnsembleExperiment

    def __init__(self, photon_sector_number, direction, swap_stors, cycles,
                 disorder_detunings_mhz, calibration=None, disorder_khz=None,
                 disorder_seed=None, notes="", ro_stors=None, disorder_patterns=None):
        """Copy/adapt the sequential constructor's union to avoid diamond inheritance."""
        swap_stors = list(swap_stors)
        self.photon_sector_number = int(photon_sector_number)
        self.directions = ([int(direction)] if isinstance(direction, (int, np.integer))
                           else [int(mode) for mode in direction])
        self.direction = self.directions
        self.ro_stor = None
        if ro_stors is None:
            self.ro_stors = [0, *swap_stors]
        elif isinstance(ro_stors, (int, np.integer)):
            self.ro_stors = [int(ro_stors)]
        else:
            self.ro_stors = [int(mode) for mode in ro_stors]
        validate_sequential_ensemble_readouts(self.directions, self.ro_stors)
        self.readout_calibration = str(calibration) if isinstance(calibration, Path) else json_plain(calibration)
        occupations = set()
        for selected_direction in self.directions:
            occupations.update(tuple(occupation) for occupation in direction_sector(
                self.photon_sector_number, swap_stors, selected_direction))
        # Actual detunings are supplied by the DMEM table, not this static field.
        FockPopulationSetExperiment.__init__(
            self, [(occupation, self.ro_stors[0]) for occupation in sorted(occupations)],
            cycles, swap_stors, [0.] * len(swap_stors), notes=notes)
        self.disorder_detunings_mhz = np.asarray(disorder_detunings_mhz, dtype=float).tolist()
        self.disorder_khz = (rms_khz(self.disorder_detunings_mhz) if disorder_khz is None
                             else float(disorder_khz))
        self.disorder_patterns = resolve_disorder_patterns(
            self.disorder_detunings_mhz, self.disorder_khz, disorder_patterns)
        self.disorder_seed = disorder_seed

    def job_overrides(self):
        """Reuse one DMEM table and pattern order across every prepared state."""
        overrides = super().job_overrides()
        for override in overrides:
            override.update(ro_stors=self.ro_stors.copy(), multiparity_confusion_matrix=None,
                            multiparity_confusion_matrices=deepcopy(self.readout_calibration),
                            disorder_patterns=deepcopy(self.disorder_patterns))
        return overrides

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", direction=None,
                      disorder_khz=None, ro_stors=None, disorder_patterns=None):
        children = list(children)
        first = children[0]
        expt = first.cfg.expt
        part = cls(sum(expt.occupations), direction, expt.swap_stors,
                   expt.floquet_cycles, first.data["disorder_detunings_mhz"],
                   calibration=expt.get("multiparity_confusion_matrices"),
                   disorder_khz=disorder_khz, disorder_seed=expt.get("disorder_seed"),
                   notes=notes, ro_stors=(first.data["ro_stors"].tolist()
                                         if ro_stors is None else ro_stors),
                   disorder_patterns=(first.data["disorder_patterns"]
                                      if disorder_patterns is None else disorder_patterns))
        part.children, part.job_ids = part._ordered(children, job_ids)
        return part

    def _population_data(self):
        for child in self.children:
            if "n_err" not in child.data:
                child.analyze()
        data = super()._population_data()
        validate_sequential_ensemble_children(self.children, self.ro_stors,
                                             self.disorder_patterns)
        data.update(directions=np.asarray(self.directions), ro_stors=np.asarray(self.ro_stors),
                    disorder_patterns=np.asarray(self.disorder_patterns))
        for key in ("n_mean", "n2_mean", "n_mean_corrected", "n2_mean_corrected"):
            if key in self.children[0].data:
                data[key] = np.stack([child.data[key] for child in self.children])
        return data

    def analyze(self, mu=None, bootstrap_samples=300, seed=0, theory=True, kerr_mhz=None):
        """Reconstruct metrics from shots; predictions never change measured arrays."""
        data = self._population_data()
        couplings_mhz = self.recorded_couplings_mhz()
        coupling_mhz = float(np.mean(np.abs(couplings_mhz)))
        default_mu = 1000 * coupling_mhz * (1 + len(self.swap_stors)) / data["sector_dimension"]
        cutoffs_khz = np.atleast_1d(np.asarray(default_mu if mu is None else mu, dtype=float))
        confusion_matrices = self.children[0].data.get("multiparity_confusion_matrices")
        counts = sequential_calibration_counts(self.children[0]) if confusion_matrices is not None else None
        data.update(ensemble_agp_metric(
            data["time_us"], data["readout_labels"], data["herald_keep"], data["occupations"],
            self.swap_stors, self.ro_stors, self.directions, data["disorder_patterns"],
            cutoffs_khz, coupling_mhz, confusion_matrices=confusion_matrices,
            calibration_shots=counts, bootstrap_samples=bootstrap_samples, seed=seed))
        data.update(coupling_mhz=np.asarray(coupling_mhz), mu_gld_khz=np.asarray(default_mu),
                    herald_acceptance=data["herald_keep"].mean(axis=-1))
        if confusion_matrices is not None:
            data["multiparity_confusion_matrices"] = np.asarray(confusion_matrices)
        if counts is not None:
            data["calibration_shot_counts"] = counts
        self.data = AttrDict(data)
        kerr = self.recorded_kerr_mhz() if kerr_mhz is None else kerr_mhz
        if theory and kerr is not None:
            self._add_metric_prediction(couplings_mhz, kerr, cutoffs_khz)
        self.analysis = dict(mu_khz=cutoffs_khz.tolist(), bootstrap_samples=bootstrap_samples,
                             seed=seed, theory=theory, kerr_mhz=kerr,
                             measured_window_us=float(data["time_us"][-1]),
                             estimator="Finite measured-window kernel integral; no tail completion",
                             metric_coordinates="K and W in cyclic kHz; metric units kHz^-2",
                             ensemble="Original patterns weighted before accepted-shot averaging",
                             uncertainty="Joint realization and shared calibration bootstrap")
        return self.data

    def _add_metric_prediction(self, couplings_mhz, kerr_mhz, cutoffs_khz):
        data = self.data
        data.update(ensemble_metric_prediction(
            self.photon_sector_number, data["disorder_detunings_mhz"], couplings_mhz,
            kerr_mhz, self.directions, self.swap_stors, data["disorder_patterns"],
            data["time_us"], cutoffs_khz))
        kernel = (2 * np.pi / 1000)**2 * np.asarray([
            agp_kernel_weights(data["time_us"], cutoff / 1000) for cutoff in cutoffs_khz])
        for suffix in ("", "_local"):
            data[f"agp_metric{suffix}_theory_sampled_window"] = np.einsum(
                "ut,tab->uab", kernel, data[f"C_metric{suffix}_theory"])
        data["agp_theory_sampled_window"] = (1000 * float(data["coupling_mhz"]))**2 * np.diagonal(
            data["agp_metric_local_theory_sampled_window"], axis1=-2, axis2=-1).T

    def manifest_parameters(self):
        parameters = super().manifest_parameters()
        parameters.update(ro_stors=self.ro_stors, disorder_patterns=self.disorder_patterns,
                          calibration=self.readout_calibration)
        return parameters

    def display(self, raw=False, **kwargs):
        """Delegate the stateless sequential plot to avoid diamond inheritance."""
        with plot_style.context():
            return CentralBosonAGPSequentialExperiment.display(self, raw=raw, **kwargs)

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        parameters = manifest["parameters"]
        return {key: parameters[key] for key in (
            "direction", "disorder_khz", "ro_stors", "disorder_patterns")}


class CentralBosonAGPSequentialEnsembleDisorderScan(CentralBosonAGPEnsembleDisorderScan):
    """One ensemble parent per scan, with explicit sequential plot delegation.

    The sequential plotting methods are reused without inheriting their Scan
    class to avoid diamond inheritance through CentralBosonAGPDisorderScan.
    """

    child_class = CentralBosonAGPSequentialEnsembleExperiment
    SHARED = ("photon_sector_number", "directions", "ro_stor", "swap_stors", "cycles",
              "ro_stors", "disorder_patterns")

    @classmethod
    def from_patterns(cls, disorder_khz, patterns, photon_sector_number, direction,
                      swap_stors, cycles, calibration=None, disorder_seed=None, notes="",
                      ro_stors=None):
        patterns = np.asarray(patterns, dtype=float)
        parts = [cls.child_class(
            photon_sector_number, direction, swap_stors, cycles,
            strength * patterns / 1000, calibration=calibration, disorder_khz=strength,
            disorder_seed=disorder_seed, notes=notes, ro_stors=ro_stors,
            disorder_patterns=patterns) for strength in disorder_khz]
        return cls(parts, notes=notes)

    def _stack_parts(self):
        data = super()._stack_parts()
        first = self.children[0].data
        for key in ("directions", "ro_stors", "disorder_patterns", "agp_metric_axes",
                    "agp_metric_local_axes", "agp_metric_units", "agp_metric_available"):
            data[key] = first[key].copy()
        for key in first:
            if key.startswith(("agp_metric", "C_metric")) and key not in data:
                data[key] = np.stack([part.data[key] for part in self.children])
        for key in ("coupling_mhz", "herald_acceptance"):
            data[key] = np.stack([part.data[key] for part in self.children])
        data["agp_list"] = data["agp"]
        data["agp_metric_list"] = data["agp_metric"]
        return data

    def display(self, mu=None, theory=True):
        """Delegate the stateless sequential plot to avoid diamond inheritance."""
        with plt.style.context(["default", {"font.size": 14}]):
            return CentralBosonAGPSequentialDisorderScan.display(self, mu=mu, theory=theory)

    display_correlation = CentralBosonAGPSequentialDisorderScan.display_traces

    def display_metric(self, mu=None, theory=True):
        """Show KK, KW and WW side by side in the original AGP scan style.

        Axes retain their (component, cutoff) indexing; every cutoff gets a row.
        """
        data = self.data
        cutoffs = data["agp_mu"]
        columns = range(len(cutoffs)) if mu is None else [int(np.argmin(abs(cutoffs - mu)))]
        entries = [(0, 0, "KK"), (0, 1, "KW"), (1, 1, "WW")]
        scale = (1000 * data["coupling_mhz"])**2
        with plt.style.context(["default", {"font.size": 14}]):
            figure, panels = plt.subplots(len(columns), 3, squeeze=False,
                                          figsize=(12, 3.6 * len(columns)), constrained_layout=True)
            axes = panels.T
            for row, (first, second, label) in enumerate(entries):
                for column, cutoff in enumerate(columns):
                    axis = axes[row, column]
                    axis.errorbar(data["disorder_khz"],
                                  data["agp_metric_scaled"][:, cutoff, first, second],
                                  data["agp_metric_scaled_err"][:, cutoff, first, second],
                                  fmt="o-", ms=4, capsize=2, label="measured")
                    if theory and "agp_metric_theory" in data:
                        axis.plot(data["disorder_khz"], scale * data[
                            "agp_metric_theory_window"][:, cutoff, first, second],
                            "x:", color="gray", label="closed system, same window")
                        axis.plot(data["disorder_khz"], scale * data[
                            "agp_metric_theory"][:, cutoff, first, second],
                            "k--", lw=1, label="closed system, infinite time")
                    axis.set(xlabel="Disorder RMS (kHz)", ylabel=rf"$g_{{ref}}^2 g_{{{label}}}$",
                             title=rf"$\mu = {cutoffs[cutoff]:g}$ kHz")
                    axis.legend(fontsize=7)
        return figure, axes

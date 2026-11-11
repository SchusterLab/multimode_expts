"""Sequential AGP: one occupation union per W, with an explicit direction axis.

Population JOBs retain every readout mode. Each W and disorder scan saves all
requested directions together through the usual acquire/analyze/save methods.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.assembled_data import ASSEMBLED_DIR, experiment_root
from experiments.characterization_runner import json_plain
from experiments.qsim.agp import (
    CentralBosonAGPDisorderScan,
    CentralBosonAGPExperiment,
    direction_sector,
    part_name,
    rms_khz,
)
from experiments.qsim.fock_population import (
    FockPopulationSetExperiment,
)
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
)
from slab import AttrDict


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

    # The analysis steps of the base class, on one readout step per direction.

    def _collect_populations(self, **kwargs):
        """Stack the jobs' arrays: (state, cycle, readout step[, label])."""
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
                    readout_steps=np.asarray([self.ro_stors.index(read_mode)
                                              for _, read_mode in self._read_directions()]),
                    cycles=np.asarray(self.cycles), swap_stors=np.asarray(self.swap_stors),
                    detunings=np.asarray(self.detunings))
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

    def _read_directions(self):
        return [(direction, direction if self.ro_stor is None else self.ro_stor)
                for direction in self.directions]

    def _correlation_inputs(self, read_mode):
        step = self.ro_stors.index(read_mode)
        corrected = "p_mod_corrected" in self.data
        return (self.data["occupations"], self.data["p_mod"][:, :, step, :],
                self.data["p_mod_corrected"][:, :, step, :] if corrected else None,
                self.data["shot_counts"][:, :, step],
                self.data["multiparity_confusion_matrices"][:, :, step] if corrected else None)

    def _store(self, index, result):
        """Rows (direction, ...); the shared axes once. A direction without a curve is NaN."""
        for key, value in result.items():
            if key in ("agp_mu", "frequency_khz"):
                self.data[key] = value
                continue
            if key not in self.data:
                self.data[key] = np.full((len(self.directions), *np.shape(value)), np.nan)
            self.data[key][index] = value

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

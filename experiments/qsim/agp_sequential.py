"""Sequential Fock readout with the existing AGP analysis and file structure.

Each child is an original sequential population JOB. Analysis selects one
readout mode in memory and reuses ``CentralBosonAGPExperiment``; no separate
per-mode files or raw-IQ copies are made. ``direction`` keeps its existing
single-mode meaning, while ``ro_stors`` specifies the physical readout order.

Pass the sequential calibration JOB filename as ``calibration``. Matrix
correction is performed by the child Experiment. AGP analysis keeps the existing
shot and calibration-error conventions; numeric tensors without a calibration
file give shot-only AGP errors. All results enter ``self.data`` before saving.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.assembled_data import ASSEMBLED_DIR, experiment_root
from experiments.characterization_runner import json_plain
from experiments.qsim.agp import (
    CentralBosonAGPDisorderScan,
    CentralBosonAGPExperiment,
    part_name,
)
from experiments.qsim.fock_population import FockPopulationSetExperiment
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
)
from slab import AttrDict


class _SequentialPopulationAnalysis(FockPopulationSetExperiment):
    """Collect one mode's arrays before the inherited AGP calculations run."""

    def analyze(self, **kwargs):
        self._check_children()
        self.shared_settings(self.children)
        self.children, self.job_ids = self._ordered(self.children, self.job_ids)
        readout_step = self.ro_stors.index(self.ro_stor)
        for source_job in self.children:
            if "p_mod" not in source_job.data:
                source_job.analyze()
            if list(source_job.data["ro_stors"]) != self.ro_stors:
                raise ValueError("Sequential AGP jobs must share the same ro_stors order")

        data = dict(occupations=np.asarray([occupation for occupation, _ in self.entries]),
                    ro_stor=np.full(len(self.children), self.ro_stor),
                    cycles=np.asarray(self.cycles), swap_stors=np.asarray(self.swap_stors),
                    detunings=np.asarray(self.detunings), ro_stors=np.asarray(self.ro_stors),
                    readout_step=np.asarray(readout_step))
        for field in ("p_mod", "n_mean", "n2_mean", "n_err", "n2_err", "shot_counts"):
            data[field] = np.stack([job.data[field][:, readout_step] for job in self.children])

        corrected = ["p_mod_corrected" in job.data for job in self.children]
        if any(corrected) and not all(corrected):
            raise ValueError("Some jobs are readout-corrected and some are not")
        self.calibration = None
        if all(corrected):
            for field in ("p_mod_corrected", "n_mean_corrected", "n2_mean_corrected"):
                data[field] = np.stack([job.data[field][:, readout_step] for job in self.children])
            matrices = [job.data["multiparity_confusion_matrices"][:, :, readout_step]
                        for job in self.children]
            if any(not np.allclose(matrix, matrices[0]) for matrix in matrices[1:]):
                raise ValueError("Sequential AGP jobs have different readout corrections")
            data["confusion_matrix"] = matrices[0].copy()
            self.calibration = dict(matrix=matrices[0].tolist())
            calibration = self.children[0].cfg.expt.get("multiparity_confusion_matrices")
            if isinstance(calibration, dict) and calibration.get("fname"):
                self.calibration["fname"] = calibration["fname"]

        cycle_times = {(job.recorded_derived_params() or {}).get("floquet_cycle_us")
                       for job in self.children}
        if len(cycle_times) == 1 and None not in cycle_times:
            data["time_us"] = data["cycles"] * cycle_times.pop()
        self.data = AttrDict(data)
        return self.data


# Existing AGP.analyze() calls super().analyze() to collect population data.
# Keep its calculations and replace that collection step with sequential arrays.
class CentralBosonAGPSequentialExperiment(CentralBosonAGPExperiment, _SequentialPopulationAnalysis):
    """One W and direction, assembled directly from sequential population JOBs."""

    child_class = FockPopulationSequentialReadoutExperiment

    def __init__(self, photon_sector_number, direction, swap_stors, cycles, detunings,
                 ro_stor=None, disorder_khz=None, calibration=None, notes="", ro_stors=None):
        swap_stors = list(swap_stors)
        if isinstance(ro_stors, np.ndarray):
            ro_stors = ro_stors.tolist()
        if ro_stors is None:
            self.ro_stors = [0, *swap_stors]
        elif isinstance(ro_stors, (int, np.integer)):
            self.ro_stors = [int(ro_stors)]
        else:
            self.ro_stors = list(ro_stors)
        self.readout_calibration = (str(calibration) if isinstance(calibration, Path)
                                    else json_plain(calibration))
        super().__init__(photon_sector_number, direction, swap_stors, cycles, detunings,
                         ro_stor=ro_stor, disorder_khz=disorder_khz, notes=notes)

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", direction=None, disorder_khz=None,
                      ro_stor=None):
        children = list(children)
        _, cycles, swap_stors, detunings, _ = cls.shared_settings(children)
        first_job_config = children[0].cfg.expt
        part = cls(sum(first_job_config.occupations), direction, swap_stors, cycles, detunings,
                   ro_stor=ro_stor, disorder_khz=disorder_khz, notes=notes,
                   ro_stors=children[0].data["ro_stors"],
                   calibration=first_job_config.get("multiparity_confusion_matrices"))
        part.children, part.job_ids = part._ordered(children, job_ids)
        part._check_children()
        return part

    def _ordered(self, children, job_ids):
        """Put original sequential JOBs in the direction sector's occupation order."""
        children, job_ids = list(children), list(job_ids)
        occupations = [tuple(child.cfg.expt.occupations) for child in children]
        expected = [occupation for occupation, _ in self.entries]
        if sorted(occupations) != sorted(expected):
            raise ValueError("The jobs must cover each direction-sector occupation exactly once")
        order = [occupations.index(occupation) for occupation in expected]
        return ([children[index] for index in order],
                [job_ids[index] for index in order] if job_ids else [])

    def job_overrides(self):
        overrides = super().job_overrides()
        for override in overrides:
            override.update(ro_stor=self.ro_stors[0], ro_stors=self.ro_stors.copy(),
                            multiparity_confusion_matrix=None,
                            multiparity_confusion_matrices=deepcopy(self.readout_calibration))
        return overrides

    def manifest_parameters(self):
        return dict(super().manifest_parameters(), ro_stors=self.ro_stors,
                    readout_calibration=self.readout_calibration)

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        return dict(super()._from_manifest_kwargs(manifest, path),
                    ro_stor=manifest["parameters"]["ro_stor"])


class CentralBosonAGPSequentialDisorderScan(CentralBosonAGPDisorderScan):
    """Use the existing disorder-scan analysis with sequential population JOBs."""

    child_class = CentralBosonAGPSequentialExperiment
    SHARED = (*CentralBosonAGPDisorderScan.SHARED, "ro_stors")

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

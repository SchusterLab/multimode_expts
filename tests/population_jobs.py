"""Synthetic FockPopulationExperiment jobs for the population-set and AGP tests.

A job is built from multiparity labels per (cycle, shot): the two final
readout lanes encode n mod 4 = b0 + 2 b1 against threshold 0, as the device
saves them. ``save_job`` writes the job-file layout a worker job has.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from slab import AttrDict

from experiments.qsim.fock_population import FockPopulationExperiment

CYCLE_US = 0.3
# C has 0.85 on its diagonal and 0.05 elsewhere: p_raw = 0.8 p_true + 0.05.
CONFUSION = .8 * np.eye(4) + .05


def shots_from_labels(labels):
    """-> idata rows (cycle, shots * 4 lanes) whose last two lanes give ``labels``."""
    labels = np.asarray(labels, dtype=int)
    lanes = np.full((*labels.shape, 4), 100.)
    lanes[..., -2] = 2 * (labels % 2) - 1
    lanes[..., -1] = 2 * (labels // 2) - 1
    return lanes.reshape(len(labels), -1)


def population_job(occupation, ro_stor, labels, cycles, swap_stors, detunings,
                   confusion=None):
    """-> an analyzed FockPopulationExperiment with shots, not yet saved."""
    job = FockPopulationExperiment.__new__(FockPopulationExperiment)
    calibration = None if confusion is None else dict(matrix=np.asarray(confusion).tolist(),
                                                      fname="calibration.h5")
    job.cfg = AttrDict(dict(
        read_num=4,
        device=dict(readout=dict(threshold=[0.])),
        expt=dict(qubits=[0], readout="multiparity", occupations=list(occupation),
                  ro_stor=ro_stor, swap_stors=list(swap_stors), detunings=list(detunings),
                  floquet_cycles=list(cycles), swept_params=["floquet_cycle"],
                  multiparity_confusion_matrix=calibration),
    ))
    idata = shots_from_labels(labels)
    job.data = dict(xpts=np.asarray(cycles), idata=idata, qdata=np.zeros_like(idata))
    job.fname = None
    job.analyze()
    return job


def save_job(job, path, cycle_us=CYCLE_US):
    """Write ``job`` in the job-file layout. -> the file loaded back without shots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["config"] = json.dumps(job.cfg)
        handle.attrs["derived_params"] = json.dumps(dict(floquet_cycle_us=cycle_us))
        for key, value in job.data.items():
            if key != "attrs":
                handle.create_dataset(key, data=np.asarray(value))
    job.fname = str(path)
    return FockPopulationExperiment.from_h5file(path, load_shots=False)


def random_labels(rng, cycles, shots=40):
    """-> labels (cycles, shots) drawn from a random distribution per cycle."""
    labels = []
    for _ in cycles:
        probabilities = rng.dirichlet(np.ones(4))
        labels.append(rng.choice(4, size=shots, p=probabilities))
    return np.asarray(labels)


class FakeRunner:
    """CharacterizationRunner stand-in: each override becomes one saved job file."""

    ExptClass = FockPopulationExperiment

    def __init__(self, data_dir, seed=0, labels=None):
        self.data_dir = Path(data_dir)
        self.rng = np.random.default_rng(seed)
        self.labels = labels  # optional callable (override) -> labels
        self.calls = []
        self.last_job_ids = []
        self.count = 0

    def execute(self, overrides, batch_size=10, **kwargs):
        self.calls.append(dict(count=len(overrides), batch_size=batch_size, **kwargs))
        jobs, self.last_job_ids = [], []
        for override in overrides:
            self.count += 1
            job_id = f"JOB-20990101-{self.count:05d}"
            cycles = override["floquet_cycles"]
            labels = (self.labels(override) if self.labels
                      else random_labels(self.rng, cycles))
            calibration = override.get("multiparity_confusion_matrix")
            job = population_job(override["occupations"], override["ro_stor"], labels, cycles,
                                 override["swap_stors"], override["detunings"],
                                 confusion=None if calibration is None else calibration["matrix"])
            save_job(job, self.data_dir / f"{job_id}_FockPopulationExperiment.h5")
            jobs.append(job)
            self.last_job_ids.append(job_id)
        return jobs

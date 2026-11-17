"""Saved disorder identity, kernel estimates and the named ensemble scan lifecycle."""
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from experiments.qsim.agp import direction_sector, sector_dimension
from experiments.qsim.agp_ensemble import (
    CentralBosonAGPEnsembleDisorderScan, CentralBosonAGPEnsembleExperiment,
)
from experiments.qsim.fock_population_ensemble import FockPopulationEnsembleExperiment
from fitting.qsim.agp import agp_kernel_weights
from tests.population_jobs import COUPLINGS_MHZ, population_job, save_job


PATTERNS = np.array([[.2, -1.], [.8, .1], [-.4, .7], [.1, -.3]])
CYCLES = [0, 3, 7]


class EnsembleRunner:
    """Write the decoded child HDF5 layout without contacting hardware."""

    ExptClass = FockPopulationEnsembleExperiment

    def __init__(self, directory):
        self.directory = Path(directory)
        self.last_job_ids = []
        self.overrides = []

    def execute(self, overrides, batch_size=1, **kwargs):
        self.overrides.extend(overrides)
        self.last_job_ids = []
        children = []
        for override in overrides:
            detunings = np.asarray(override["disorder_detunings_mhz"])
            cycles = override["floquet_cycles"]
            labels = np.array([[(time + realization) % 3 for realization in range(len(detunings))]
                               for time in range(len(cycles))])
            child = population_job(override["occupations"], override["ro_stor"], labels,
                                   cycles, override["swap_stors"], [0., 0.])
            child.cfg.expt.update(override)
            indices = np.arange(len(detunings))
            child.data.update(disorder_detunings_mhz=detunings,
                              disorder_frequency_words=np.arange(detunings.size, dtype=np.uint32)
                              .reshape(detunings.shape), realization_indices=indices,
                              shot_realization_indices=np.array([indices for _ in cycles]),
                              readout_labels=labels, herald_keep=np.ones(labels.shape, dtype=bool))
            job_id = f"JOB-20990101-{len(self.overrides) + len(children):05d}"
            path = self.directory / f"{job_id}_FockPopulationEnsembleExperiment.h5"
            save_job(child, path)
            children.append(FockPopulationEnsembleExperiment.from_h5file(path, load_shots=False))
            self.last_job_ids.append(job_id)
        return children


def planned_scan(strengths=(30, 0), direction=7):
    return CentralBosonAGPEnsembleDisorderScan.from_patterns(
        strengths, PATTERNS, photon_sector_number=2, direction=direction,
        swap_stors=[4, 7], cycles=CYCLES, disorder_seed=19)


@pytest.mark.parametrize("direction", [0, 7])
def test_same_realizations_in_all_fock_jobs_and_kernel_normalization(tmp_path, direction):
    part = planned_scan((30,), direction).children[0]
    runner = EnsembleRunner(tmp_path / "data")
    part.acquire(runner)
    assert len(runner.overrides) == len(direction_sector(2, [4, 7], direction))
    for override in runner.overrides:
        np.testing.assert_array_equal(override["disorder_detunings_mhz"], .03 * PATTERNS)
        assert override["reps"] == len(PATTERNS) and override["rounds"] == 1
    data = part.analyze(bootstrap_samples=30, theory=False)
    coupling = COUPLINGS_MHZ[0]
    expected_cutoff = coupling * 1000 * 3 / sector_dimension(2, 3)
    np.testing.assert_allclose(data["agp_mu"], [expected_cutoff])
    weights = (2 * np.pi * coupling)**2 * agp_kernel_weights(
        data["time_us"], expected_cutoff / 1000)
    np.testing.assert_allclose(data["agp"], [weights @ data["C1"]])
    assert data["readout_labels"].shape == (len(part.entries), len(CYCLES), len(PATTERNS))
    assert data["C1_covariance"].shape == (len(CYCLES), len(CYCLES))
    figure = part.display()
    matplotlib.pyplot.close(figure)


def test_save_reload_and_same_realization_theory_without_raw_iq(tmp_path):
    scan = planned_scan()
    runner = EnsembleRunner(tmp_path / "data")
    manifest = scan.acquire(runner, "ensemble_test", analyze_kwargs=dict(mu=[2., 5.915],
                                                                       bootstrap_samples=30))
    reloaded = CentralBosonAGPEnsembleDisorderScan.from_manifest(manifest)
    reloaded.analyze(mu=[2., 5.915], bootstrap_samples=30)
    assert len(reloaded.children) == 2
    np.testing.assert_array_equal(reloaded.data["disorder_khz"], [0, 30])
    for key in ("agp", "agp_err", "C1", "disorder_detunings_mhz", "agp_theory",
                "agp_theory_window", "agp_theory_sampled_window"):
        np.testing.assert_allclose(reloaded.data[key], scan.data[key])
    for part in reloaded.children:
        assert all("idata" not in child.data for child in part.children)
        with h5py.File(part.manifest_path.with_suffix(".h5"), "r") as handle:
            np.testing.assert_array_equal(handle["disorder_detunings_mhz"],
                                          .001 * part.disorder_khz * PATTERNS)
            assert handle["readout_labels"].shape[-1] == len(PATTERNS)
            assert json.loads(handle.attrs["analysis"])["bootstrap_samples"] == 30
    view = CentralBosonAGPEnsembleDisorderScan.from_h5file(manifest.with_suffix(".h5"))
    figure, axes = view.display()
    assert all(axis.get_xscale() == axis.get_yscale() == "linear" for axis in axes)
    matplotlib.pyplot.close(figure)


def test_mismatched_table_is_not_silently_averaged(tmp_path):
    part = planned_scan((30,)).children[0]
    part.acquire(EnsembleRunner(tmp_path / "data"))
    part.children[-1].data["disorder_detunings_mhz"][0, 0] += .01
    with pytest.raises(ValueError, match="same disorder_detunings_mhz"):
        part.analyze(theory=False)

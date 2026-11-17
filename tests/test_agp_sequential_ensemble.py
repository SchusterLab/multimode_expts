"""Joint ensemble acquisition planning, raw-file replay and metric persistence."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from experiments.qsim.agp_sequential_ensemble import (
    CentralBosonAGPSequentialEnsembleDisorderScan,
    CentralBosonAGPSequentialEnsembleExperiment,
)
from experiments.qsim.fock_population_sequential_ensemble import (
    FockPopulationSequentialEnsembleExperiment,
)
from tests.population_jobs import save_job
from tests.test_fock_population_sequential import MATRICES, MODES, synthetic_job


PATTERNS = np.array([[1., -1.], [-1., 1.], [.5, -.5], [-.5, .5], [1.2, -1.2], [-1.2, 1.2]])
CYCLES = [0, 3, 6]


@pytest.mark.parametrize("experiment_class", [CentralBosonAGPSequentialEnsembleExperiment,
                                             CentralBosonAGPSequentialEnsembleDisorderScan])
def test_assembly_hierarchies_have_no_repeated_nonobject_ancestor(experiment_class):
    ancestors = []

    def collect_parent_paths(child_class):
        for parent_class in child_class.__bases__:
            if parent_class is not object:
                ancestors.append(parent_class)
                collect_parent_paths(parent_class)

    collect_parent_paths(experiment_class)
    assert all(count == 1 for count in Counter(ancestors).values())


class SequentialEnsembleRunner:
    """Persist real decoded layouts without opening instruments or a job queue."""

    ExptClass = FockPopulationSequentialEnsembleExperiment

    def __init__(self, directory):
        self.directory = Path(directory)
        self.overrides = []
        self.last_job_ids = []

    def execute(self, overrides, batch_size=1, **kwargs):
        self.last_job_ids = []
        children = []
        for override in overrides:
            patterns = np.asarray(override["disorder_patterns"])
            labels = (np.arange(len(CYCLES) * len(patterns) * len(MODES)).reshape(
                len(CYCLES), len(patterns), len(MODES)) + len(self.overrides)) % 4
            original, *_ = synthetic_job(occupations=override["occupations"], labels=labels,
                                         calibration=override["multiparity_confusion_matrices"])
            child = self.ExptClass.__new__(self.ExptClass)
            child.cfg, child.data = deepcopy(original.cfg), deepcopy(original.data)
            child.cfg.expt.update(override)
            indices = np.arange(len(patterns))
            child.data.update(disorder_detunings_mhz=np.asarray(override["disorder_detunings_mhz"]),
                              disorder_frequency_words=np.arange(patterns.size, dtype=np.uint32)
                              .reshape(patterns.shape), disorder_patterns=patterns,
                              realization_indices=indices,
                              shot_realization_indices=np.broadcast_to(indices, labels.shape[:2]))
            child.analyze()
            job_id = f"JOB-20990101-{len(self.overrides):05d}"
            path = self.directory / f"{job_id}_FockPopulationSequentialEnsembleExperiment.h5"
            save_job(child, path)
            children.append(self.ExptClass.from_h5file(path, load_shots=False))
            self.last_job_ids.append(job_id)
            self.overrides.append(override)
        return children


def planned_scan(strengths=(30., 0.), directions=(0, 4, 7), calibrated=True):
    calibration = (dict(matrices=MATRICES.tolist(), ro_stors=MODES,
                        calibration_shot_counts=np.full((4, 3), 2000).tolist())
                   if calibrated else None)
    return CentralBosonAGPSequentialEnsembleDisorderScan.from_patterns(
        strengths, PATTERNS, photon_sector_number=2, direction=directions,
        swap_stors=[4, 7], cycles=CYCLES, calibration=calibration, disorder_seed=13)


def test_joint_plan_preserves_zero_strength_patterns_and_uses_state_union():
    scan = planned_scan()
    assert [part.disorder_khz for part in scan.children] == [0, 30]
    assert [len(part.entries) for part in scan.children] == [6, 6]
    for part in scan.children:
        overrides = part.job_overrides()
        for override in overrides:
            np.testing.assert_array_equal(override["disorder_patterns"], PATTERNS)
            np.testing.assert_array_equal(override["disorder_detunings_mhz"],
                                          part.disorder_khz * PATTERNS / 1000)
            assert override["ro_stors"] == MODES
            assert override["multiparity_confusion_matrix"] is None
    full = CentralBosonAGPSequentialEnsembleDisorderScan.from_patterns(
        [0], np.ones((2, 4)), 3, [0, 4, 5, 6, 7], [4, 5, 6, 7], CYCLES)
    assert len(full.children[0].entries) == 35


def test_saved_jobs_replay_metrics_and_bootstrap_without_iq(tmp_path):
    scan = planned_scan()
    runner = SequentialEnsembleRunner(tmp_path / "data")
    manifest = scan.acquire(runner, "joint_metric", analyze_kwargs=dict(
        mu=[3., 10.], bootstrap_samples=20, seed=8))
    assert len(runner.overrides) == 12
    assert scan.data["agp"].shape == (2, 3, 2)
    assert scan.data["agp_metric"].shape == (2, 2, 2, 2)
    assert scan.data["agp_metric_local"].shape == (2, 2, 3, 3)
    reloaded = CentralBosonAGPSequentialEnsembleDisorderScan.from_manifest(manifest)
    reloaded.analyze(mu=[3., 10.], bootstrap_samples=20, seed=8)
    for key in ("agp", "agp_err", "agp_metric", "agp_metric_err", "agp_metric_local",
                "agp_metric_theory", "agp_metric_theory_window",
                "agp_metric_theory_sampled_window", "agp_theory_sampled_window"):
        np.testing.assert_allclose(reloaded.data[key], scan.data[key], atol=1e-14)
    for part in reloaded.children:
        assert all("idata" not in child.data for child in part.children)
        np.testing.assert_array_equal(part.data["disorder_patterns"], PATTERNS)
        with h5py.File(part.manifest_path.with_suffix(".h5"), "r") as handle:
            assert handle["agp_metric_units"][()] == b"kHz^-2"
    view = CentralBosonAGPSequentialEnsembleDisorderScan.from_h5file(
        manifest.with_suffix(".h5"))
    for display in (view.display, view.display_metric, view.display_correlation):
        figure, _ = display()
        assert all(axis.get_xscale() == axis.get_yscale() == "linear" for axis in figure.axes)
        matplotlib.pyplot.close(figure)


def test_individual_point_display_and_partial_kerr_metric(tmp_path):
    part = planned_scan((0,), directions=(0,), calibrated=False).children[0]
    part.acquire(SequentialEnsembleRunner(tmp_path / "data"))
    part.analyze(mu=10, bootstrap_samples=10)
    assert part.data["agp_metric_local"].shape == (1, 1, 1)
    assert np.isfinite(part.data["agp_metric"][0, 0, 0])
    assert np.isnan(part.data["agp_metric"][0, 1, 1])
    figure = part.display()
    matplotlib.pyplot.close(figure)


@pytest.mark.parametrize("raw", [False, True])
def test_saved_scan_spectrum_resolves_each_direction_without_reanalysis(tmp_path, monkeypatch, raw):
    times = np.arange(201, dtype=float)
    peak_frequencies = np.array([[20., 40.], [60., 80.]])
    correlation = np.cos(2 * np.pi * peak_frequencies[..., None] * times / 1000)
    path = tmp_path / "saved_scan.h5"
    with h5py.File(path, "w") as handle:
        for key, value in dict(time_us=times, disorder_khz=[0., 10.], directions=[0, 4],
                               agp_mu=[10.], C1=2 * correlation, C1_corrected=correlation,
                               C1_err=np.zeros_like(correlation), C1_theory=.9 * correlation).items():
            handle.create_dataset(key, data=value)
    scan = CentralBosonAGPSequentialEnsembleDisorderScan.from_h5file(path)

    def forbid_reanalysis(*args, **kwargs):
        pytest.fail("Spectrum display must use saved correlators without reanalysis")

    monkeypatch.setattr(scan, "analyze", forbid_reanalysis)
    figure, axes = scan.display_spectrum(mu=[], window="raw", max_khz=100, raw=raw)
    assert axes.shape == (2, 2)
    for index, axis in np.ndenumerate(axes):
        measured, theory = axis.lines[:2]
        maximum = np.argmax(measured.get_ydata())
        assert measured.get_xdata()[maximum] == pytest.approx(peak_frequencies[index])
        assert measured.get_ydata()[maximum] == pytest.approx(.8 if raw else .4)
        assert theory.get_ydata()[maximum] == pytest.approx(.36)
        assert axis.get_xscale() == axis.get_yscale() == "linear"
    matplotlib.pyplot.close(figure)
    figure, axes = scan.display_spectrum(directions=[4], theory=False, mu=[], max_khz=100)
    assert axes.shape == (2, 1)
    assert all(len(axis.lines) == 1 for axis in axes.flat)
    matplotlib.pyplot.close(figure)


def test_mismatched_patterns_do_not_silently_change_disorder_direction(tmp_path):
    part = planned_scan((0,), calibrated=False).children[0]
    part.acquire(SequentialEnsembleRunner(tmp_path / "data"))
    part.children[-1].data["disorder_patterns"][0, 0] += .1
    with pytest.raises(ValueError, match="share the original disorder_patterns"):
        part.analyze(theory=False)


def test_zero_strength_without_original_patterns_is_explicit():
    with pytest.raises(ValueError, match="W = 0 needs"):
        CentralBosonAGPSequentialEnsembleExperiment(
            2, [0, 4, 7], [4, 7], CYCLES, np.zeros_like(PATTERNS))


def test_notebook_plans_one_joint_scan_with_original_patterns():
    source = Path("measurement_notebooks/jonginn/agp_sequential_ensemble_measurement.py").read_text(
        encoding="utf-8")
    planning_cell = source.split('# %% tags=["agp_plan"]', 1)[1].split("# %%", 1)[0]
    settings = dict(CentralBosonAGPSequentialEnsembleDisorderScan=
                    CentralBosonAGPSequentialEnsembleDisorderScan,
                    DISORDER_RMS_KHZ=[0, 30], disorder_patterns=PATTERNS,
                    TOTAL_PHOTONS=2, AGP_DIRECTIONS=[0, 4, 7], STORAGE_MODES=[4, 7],
                    AGP_CYCLES=CYCLES, CONFUSION_MATRIX_FNAME=None, DISORDER_SEED=13,
                    RO_STORS=MODES, NUMBER_OF_REALIZATIONS=len(PATTERNS))
    exec(planning_cell, settings)
    scan = settings["scan"]
    assert settings["jobs_per_strength"] == 6
    assert len(scan.children) == 2
    for override in scan.children[0].job_overrides():
        np.testing.assert_array_equal(override["disorder_patterns"], PATTERNS)

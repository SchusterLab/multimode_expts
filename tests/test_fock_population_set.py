"""FockPopulationExperiment moments and the FockPopulationSetExperiment layer."""
import h5py
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from experiments.qsim.fock_population import (  # noqa: E402
    FockPopulationExperiment,
    FockPopulationSetExperiment,
    calibration_record,
)
from tests.population_jobs import (  # noqa: E402
    CONFUSION,
    CYCLE_US,
    FakeRunner,
    population_job,
    random_labels,
    save_job,
)

CYCLES = [0, 2, 4]
STORS = [4, 7]
DETUNINGS = [.01, -.02]
WALK = [([1, 0, 0], mode) for mode in (0, 4, 7)]


@pytest.mark.parametrize("confusion", [None, CONFUSION])
def test_job_moments_and_errors_match_per_shot_sums(confusion):
    labels = random_labels(np.random.default_rng(1), CYCLES, shots=50)
    job = population_job([1, 0, 0], 0, labels, CYCLES, STORS, DETUNINGS, confusion=confusion)
    values = np.arange(4)
    for power, mean_key, error_key in [(1, "nmod4_mean", "n_err"),
                                       (2, "nmod4_second_moment", "n2_err")]:
        np.testing.assert_allclose(job.data[mean_key], (labels**power).mean(axis=1))
        weights = values**power if confusion is None else np.linalg.solve(confusion, values**power)
        # Each shot's weight; the error is the sample standard error of their mean.
        per_shot = weights[labels]
        np.testing.assert_allclose(job.data[error_key],
                                   per_shot.std(axis=1, ddof=1) / np.sqrt(labels.shape[1]))
        if confusion is not None:
            np.testing.assert_allclose(job.data[f"{mean_key}_corrected"], per_shot.mean(axis=1))
    if confusion is None:
        assert "nmod4_second_moment_corrected" not in job.data
    np.testing.assert_array_equal(job.data["shot_counts"], [50, 50, 50])


def test_sweep_axes_fix_the_cycle_sweep():
    job = population_job([1, 0, 0], 0, random_labels(np.random.default_rng(0), CYCLES),
                         CYCLES, STORS, DETUNINGS)
    job.cfg.expt.swept_params = ["something_else"]
    assert job.sweep_axes() == [("floquet_cycle", CYCLES)]
    assert job.cfg.expt.swept_params == ["floquet_cycle"]
    assert FockPopulationExperiment.default_program.__name__ == "FockPopulationProgram"


def test_job_file_loads_without_shots(tmp_path):
    job = population_job([1, 0, 0], 4, random_labels(np.random.default_rng(0), CYCLES),
                         CYCLES, STORS, DETUNINGS, confusion=CONFUSION)
    loaded = save_job(job, tmp_path / "data" / "JOB-1.h5")
    assert "idata" not in loaded.data and "qdata" not in loaded.data
    np.testing.assert_allclose(loaded.data["nmod4_mean_corrected"], job.data["nmod4_mean_corrected"])
    assert loaded.cfg.expt.ro_stor == 4
    full = FockPopulationExperiment.from_h5file(tmp_path / "data" / "JOB-1.h5")
    np.testing.assert_array_equal(full.data["idata"], job.data["idata"])
    fig, ax = full.display()
    assert "read S4" in ax.get_title()


def walk_jobs(tmp_path, confusion=CONFUSION, **changes):
    rng = np.random.default_rng(2)
    jobs = []
    for index, (occupation, mode) in enumerate(WALK):
        settings = dict(cycles=CYCLES, swap_stors=STORS, detunings=DETUNINGS)
        if index == 2:
            settings.update(changes)
        job = population_job(occupation, mode, random_labels(rng, settings["cycles"]),
                             settings["cycles"], settings["swap_stors"], settings["detunings"],
                             confusion=confusion)
        jobs.append(save_job(job, tmp_path / "data" / f"JOB-{index}.h5"))
    return jobs


def test_walk_set_stacks_saves_and_reloads_without_shots(tmp_path):
    jobs = walk_jobs(tmp_path)
    walk = FockPopulationSetExperiment.from_children(jobs, job_ids=["a", "b", "c"], notes="walk")
    data = walk.analyze()
    assert data["n_mean"].shape == (3, 3) and data["p_mod"].shape == (3, 3, 4)
    np.testing.assert_array_equal(data["ro_stor"], [0, 4, 7])
    for row, job in zip(data["n_mean_corrected"], jobs):
        np.testing.assert_allclose(row, job.data["nmod4_mean_corrected"])
    np.testing.assert_allclose(data["time_us"], np.asarray(CYCLES) * CYCLE_US)
    np.testing.assert_allclose(data["confusion_matrix"], CONFUSION)

    manifest_path = walk.save()
    assert manifest_path.parent == tmp_path / "assembled_data"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["class"] == "FockPopulationSetExperiment"
    assert manifest["raw_files"] == ["../data/JOB-0.h5", "../data/JOB-1.h5", "../data/JOB-2.h5"]
    assert manifest["parameters"]["calibration"]["fname"] == "calibration.h5"
    h5_path = manifest_path.with_suffix(".h5")
    with h5py.File(h5_path, "r") as handle:
        assert "idata" not in handle and "qdata" not in handle

    reloaded = FockPopulationSetExperiment.from_manifest(manifest_path)
    reloaded.analyze()
    for key, value in data.items():
        np.testing.assert_allclose(reloaded.data[key], value)
    view = FockPopulationSetExperiment.from_h5file(h5_path)
    assert view.manifest_path == manifest_path
    np.testing.assert_allclose(view.data["n_mean"], data["n_mean"])
    view.display()


@pytest.mark.parametrize("change, message", [
    (dict(cycles=[0, 2, 6]), "different Floquet cycles"),
    (dict(detunings=[.01, -.03]), "different detunings"),
])
def test_jobs_with_different_settings_are_not_combined(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        FockPopulationSetExperiment.from_children(walk_jobs(tmp_path, **change))


def test_jobs_with_different_corrections_are_not_combined(tmp_path):
    jobs = walk_jobs(tmp_path)
    other = population_job([1, 0, 0], 4, random_labels(np.random.default_rng(3), CYCLES),
                           CYCLES, STORS, DETUNINGS, confusion=None)
    with pytest.raises(ValueError, match="different readout correction"):
        FockPopulationSetExperiment.from_children([jobs[0], other])


def test_jobs_loaded_without_results_or_shots_are_refused(tmp_path):
    jobs = walk_jobs(tmp_path)
    del jobs[1].data["n2_err"]
    walk = FockPopulationSetExperiment.from_children(jobs)
    with pytest.raises(ValueError, match="no analyzed results"):
        walk.analyze()


def test_overrides_carry_the_correction_and_acquire_assembles(tmp_path):
    walk = FockPopulationSetExperiment(WALK, CYCLES, STORS, DETUNINGS, calibration=CONFUSION)
    overrides = walk.job_overrides()
    assert [(o["occupations"], o["ro_stor"]) for o in overrides] == [([1, 0, 0], m) for m in (0, 4, 7)]
    assert all(o["multiparity_confusion_matrix"]["matrix"] == CONFUSION.tolist() for o in overrides)
    assert all(o["floquet_cycles"] == CYCLES and o["detunings"] == DETUNINGS for o in overrides)

    runner = FakeRunner(tmp_path / "data")
    walk.acquire(runner, batch_size=3, use_queue=True)
    assert runner.calls == [dict(count=3, batch_size=3, use_queue=True)]
    assert walk.job_ids == runner.last_job_ids
    walk.analyze()
    assert "n_mean_corrected" in walk.data
    assert walk.save().exists()


def test_calibration_record_reads_a_calibration_file(tmp_path):
    path = tmp_path / "cal.h5"
    with h5py.File(path, "w") as handle:
        handle["multiparity_confusion_matrix"] = CONFUSION
    record = calibration_record(path)
    assert record == dict(fname=str(path), matrix=CONFUSION.tolist())
    assert calibration_record(None) is None
    assert calibration_record(CONFUSION)["matrix"] == CONFUSION.tolist()


def test_entries_are_checked():
    with pytest.raises(ValueError, match="appear once"):
        FockPopulationSetExperiment(WALK + WALK[:1], CYCLES, STORS, DETUNINGS)
    with pytest.raises(ValueError, match="read mode 5"):
        FockPopulationSetExperiment([([1, 0, 0], 5)], CYCLES, STORS, DETUNINGS)

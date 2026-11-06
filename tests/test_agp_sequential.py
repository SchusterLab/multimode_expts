"""Inherited AGP analysis over native sequential jobs, without exported files."""

import json
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import yaml

from slab.experiment import NpEncoder

matplotlib.use("Agg")

from experiments.qsim.agp import CentralBosonAGPExperiment, direction_sector
from experiments.qsim.agp_sequential import (
    CentralBosonAGPSequentialDisorderScan,
    CentralBosonAGPSequentialExperiment,
)
from experiments.qsim.fock_population import FockPopulationExperiment
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
)
from tests.test_fock_population_sequential import (
    CYCLES,
    MATRICES,
    MODES,
    save_through_go,
    synthetic_job,
)


def source_jobs(tmp_path, monkeypatch, direction, corrected):
    """Persist native sequential jobs and retain independent accepted-label inputs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jobs, labels_by_job, keep_by_job = [], [], []
    for index, occupation in enumerate(direction_sector(2, [4, 7], direction)):
        labels = (np.arange(len(CYCLES) * 6 * 3).reshape(len(CYCLES), 6, 3) + index) % 4
        calibration = (dict(matrices=MATRICES.tolist(), ro_stors=MODES.copy())
                       if corrected else None)
        job, labels, keep, *_ = synthetic_job(occupations=occupation, labels=labels,
                                              calibration=calibration)
        save_through_go(job, data_dir / f"JOB-20990101-{index:05d}_sequential.h5", monkeypatch)
        jobs.append(job)
        labels_by_job.append(labels)
        keep_by_job.append(keep)
    return jobs, labels_by_job, keep_by_job


def legacy_reference(job, labels, keep, mode, corrected):
    """Build scalar processed inputs independently from retained measured labels."""
    view = FockPopulationExperiment.__new__(FockPopulationExperiment)
    view.cfg = deepcopy(job.cfg)
    view.cfg.expt.ro_stor = mode
    view.cfg.expt.pop("ro_stors", None)
    view.cfg.expt.pop("multiparity_confusion_matrices", None)
    step = MODES.index(mode)
    matrix = MATRICES[:, :, step] if corrected else np.eye(4)
    view.cfg.expt.multiparity_confusion_matrix = (
        dict(matrix=matrix.tolist()) if corrected else None)
    raw = np.asarray([np.bincount(row[mask, step], minlength=4) / mask.sum()
                      for row, mask in zip(labels, keep)])
    view.data = dict(xpts=np.asarray(CYCLES), shot_counts=keep.sum(axis=1),
                     attrs=dict(derived_params=json.dumps(job.recorded_derived_params())))
    for number in range(4):
        view.data[f"p_mod{number}"] = raw[:, number]
    for power, mean_key, error_key in [(1, "nmod4_mean", "n_err"),
                                       (2, "nmod4_second_moment", "n2_err")]:
        values = np.arange(4.)**power
        weights = np.linalg.solve(matrix, values)
        measured = [weights[row[mask, step]] for row, mask in zip(labels, keep)]
        view.data[mean_key] = raw @ values
        view.data[error_key] = np.asarray([
            row.std(ddof=1) / np.sqrt(len(row)) for row in measured])
        if corrected:
            view.data[f"{mean_key}_corrected"] = np.asarray([row.mean() for row in measured])
    if corrected:
        probabilities = np.linalg.solve(matrix.T, raw.T).T
        for number in range(4):
            view.data[f"p_mod{number}_corrected"] = probabilities[:, number]
        view.data["confusion_matrix"] = matrix
    view.fname = job.fname
    return view


@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("direction, read_mode", [(7, 7), (0, 0), (7, 4)])
def test_native_children_match_legacy_analysis_and_reload_original_files(
        tmp_path, monkeypatch, corrected, direction, read_mode):
    jobs, labels, masks = source_jobs(tmp_path, monkeypatch, direction, corrected)
    configs_before = [json.dumps(job.cfg, cls=NpEncoder, sort_keys=True) for job in jobs]
    data_before = [deepcopy(job.data) for job in jobs]
    ids = [f"source-{index}" for index in range(len(jobs))]
    references = [legacy_reference(job, row, keep, read_mode, corrected)
                  for job, row, keep in zip(jobs, labels, masks)]
    reference = CentralBosonAGPExperiment.from_children(references, direction=direction)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Calibration file not found.*")
        warnings.filterwarnings("ignore", message="AGP and spectrum skipped.*")
        expected = reference.analyze(mu=[2., 10.], t1_us=600.)

    def unexpected_legacy_factory(*args, **kwargs):
        raise AssertionError("Native sequential analysis must not construct a legacy AGP object")

    with monkeypatch.context() as compatibility_guard:
        compatibility_guard.setattr(CentralBosonAGPExperiment, "from_children",
                                     classmethod(unexpected_legacy_factory))
        part = CentralBosonAGPSequentialExperiment.from_children(
            list(reversed(jobs)), job_ids=list(reversed(ids)), direction=direction,
            ro_stor=read_mode)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Calibration file not found.*")
            warnings.filterwarnings("ignore", message="AGP and spectrum skipped.*")
            actual = part.analyze(mu=[2., 10.], t1_us=600.)
    assert CentralBosonAGPSequentialExperiment.analyze is CentralBosonAGPExperiment.analyze
    for key, expected_value in expected.items():
        np.testing.assert_allclose(actual[key], expected_value)
    assert {id(job) for job in part.children} == {id(job) for job in jobs}
    assert all(isinstance(job, FockPopulationSequentialReadoutExperiment) for job in part.children)
    assert ("C2" in actual) == (direction != 0)
    assert ("agp" in actual) == (direction == read_mode)
    for job, cfg_before, previous_data in zip(jobs, configs_before, data_before):
        assert json.dumps(job.cfg, cls=NpEncoder, sort_keys=True) == cfg_before
        assert set(job.data) == set(previous_data)
        for key in previous_data:
            np.testing.assert_array_equal(job.data[key], previous_data[key])
    assert actual["p_mod"].shape == (len(jobs), len(CYCLES), 4)
    assert actual["n_mean"].shape == (len(jobs), len(CYCLES))
    np.testing.assert_array_equal(actual["ro_stor"], [read_mode] * len(jobs))
    np.testing.assert_array_equal(actual["ro_stors"], MODES)
    assert int(actual["readout_step"]) == MODES.index(read_mode)

    manifest_path = part.save("native_sequential", directory=tmp_path / "assembled_data")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    original_paths = {(manifest_path.parent / path).resolve() for path in manifest["raw_files"]}
    assert original_paths == {Path(job.fname).resolve() for job in jobs}
    assert manifest["parameters"]["ro_stors"] == MODES
    if corrected:
        np.testing.assert_allclose(manifest["parameters"]["readout_calibration"]["matrices"],
                                   MATRICES)
    assert not (tmp_path / "processed_data").exists()
    reloaded = CentralBosonAGPSequentialExperiment.from_manifest(manifest_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Calibration file not found.*")
        warnings.filterwarnings("ignore", message="AGP and spectrum skipped.*")
        restored = reloaded.analyze(mu=[2., 10.], t1_us=600.)
    assert reloaded.ro_stor == read_mode
    for key, expected_value in actual.items():
        np.testing.assert_allclose(restored[key], expected_value)
    arrays_only = CentralBosonAGPSequentialExperiment.from_h5file(
        manifest_path.with_suffix(".h5"))
    fig = arrays_only.display()
    matplotlib.pyplot.close(fig)


def test_job_overrides_keep_selected_probe_separate_from_physical_readout_order():
    part = CentralBosonAGPSequentialExperiment(
        2, 7, [4, 7], CYCLES, [.01, -.02], ro_stor=4,
        ro_stors=MODES, calibration=MATRICES)
    overrides = part.job_overrides()
    assert [override["occupations"] for override in overrides] == direction_sector(2, [4, 7], 7)
    for override in overrides:
        assert override["ro_stor"] == MODES[0]
        assert override["ro_stors"] == MODES
        assert override["multiparity_confusion_matrix"] is None
        np.testing.assert_allclose(override["multiparity_confusion_matrices"], MATRICES)
    assert part.direction == 7 and part.ro_stor == 4


class SequentialRunner:
    """Execute synthetic native jobs through their real analysis and save methods."""

    ExptClass = FockPopulationSequentialReadoutExperiment

    def __init__(self, directory, fail_on_call=None):
        self.station = SimpleNamespace(experiment_path=Path(directory))
        self.data_dir = Path(directory) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.last_job_ids = []
        self.fail_on_call = fail_on_call
        self.count = 0

    def execute(self, overrides, batch_size=10, **kwargs):
        self.calls.append(dict(overrides=deepcopy(overrides), batch_size=batch_size, **kwargs))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("stopped after the completed W")
        jobs, self.last_job_ids = [], []
        for override in overrides:
            self.count += 1
            job, *_ = synthetic_job(occupations=override["occupations"],
                                    cycles=override["floquet_cycles"],
                                    calibration=override["multiparity_confusion_matrices"])
            job.cfg.expt.update(deepcopy(override))
            job.analyze()
            job_id = f"JOB-20990101-{self.count:05d}"
            job.fname = str(self.data_dir / f"{job_id}_sequential.h5")
            job.save_data()
            jobs.append(job)
            self.last_job_ids.append(job_id)
        return jobs


def planned_scan(strengths=(30, 0)):
    return CentralBosonAGPSequentialDisorderScan.from_pattern(
        strengths, [1., -3.], photon_sector_number=2, direction=7,
        swap_stors=[4, 7], cycles=CYCLES, ro_stors=MODES)


def test_scan_acquires_and_reloads_native_layers_and_checks_names_before_jobs(tmp_path):
    scan = planned_scan()
    assert [part.disorder_khz for part in scan.children] == [0, 30]
    for part in scan.children:
        assert isinstance(part, CentralBosonAGPSequentialExperiment)
        expected = .001 * part.disorder_khz * np.array([1., -3.]) / np.sqrt(5.)
        np.testing.assert_allclose(part.detunings, expected)
    runner = SequentialRunner(tmp_path)
    manifest_path = scan.acquire(runner, "native_scan", batch_size=2,
                                 analyze_kwargs=dict(mu=[5., 10.]), use_queue=False)
    assert len(runner.calls) == 2
    assert all(len(call["overrides"]) == 3 for call in runner.calls)
    assert all(call["batch_size"] == 2 for call in runner.calls)
    assert scan.data["C1"].shape == (2, len(CYCLES))
    assert scan.data["agp"].shape == (2, 2)
    assert all(part.manifest_path.exists() for part in scan.children)
    for part in scan.children:
        for job in part.children:
            assert isinstance(job, FockPopulationSequentialReadoutExperiment)
            assert job.data["p_mod"].shape == (len(CYCLES), len(MODES), 4)
    restored = CentralBosonAGPSequentialDisorderScan.from_manifest(manifest_path)
    restored.analyze(mu=[5., 10.])
    for key, value in scan.data.items():
        np.testing.assert_allclose(restored.data[key], value)
    arrays_only = CentralBosonAGPSequentialDisorderScan.from_h5file(
        manifest_path.with_suffix(".h5"))
    fig, _ = arrays_only.display(mu=10.)
    matplotlib.pyplot.close(fig)
    before = len(runner.calls)
    with pytest.raises(FileExistsError):
        planned_scan().acquire(runner, "native_scan", analyze_kwargs=dict(mu=10.))
    assert len(runner.calls) == before


def test_interrupted_scan_leaves_completed_w_available_from_saved_parts(tmp_path):
    runner = SequentialRunner(tmp_path, fail_on_call=2)
    with pytest.raises(RuntimeError, match="stopped"):
        planned_scan().acquire(runner, "interrupted", analyze_kwargs=dict(mu=10.))
    completed = CentralBosonAGPSequentialDisorderScan.from_saved_parts(
        "interrupted", tmp_path / "assembled_data")
    assert [part.disorder_khz for part in completed.children] == [0]
    assert all(isinstance(job, FockPopulationSequentialReadoutExperiment)
               for job in completed.children[0].children)
    completed.analyze(mu=10.)
    assert completed.data["agp"].shape == (1, 1)
    assert len(runner.calls) == 2

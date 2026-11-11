# -*- coding: utf-8 -*-
"""Data roots and paths below them (experiments/data_paths.py).

Hermetic: every test builds its own data roots and provenance sidecar under
``tmp_path``; nothing touches the real shares or the job database.
"""
import json
import os

import pytest

from experiments import data_paths as dp


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    for var in (dp.DATA_ROOTS_ENV, dp.PROVENANCE_ENV, "MULTIMODE_DATA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    dp.clear_cache()
    yield
    dp.clear_cache()


def _tree(root, experiment, job_ids=(), suffix="_QsimBaseExperiment"):
    """Create ``root/experiment/data/{JOB}{suffix}.h5`` files; -> root."""
    data = root / experiment / "data"
    data.mkdir(parents=True, exist_ok=True)
    for job in job_ids:
        (data / f"{job}{suffix}.h5").write_bytes(b"")
    return root


def _sidecar(tmp_path, mapping):
    path = tmp_path / "prov.json"
    path.write_text(json.dumps({job: {"job_id": job, "data_file_path": recorded}
                                for job, recorded in mapping.items()}))
    return path


def _use(monkeypatch, *roots, sidecar=None):
    monkeypatch.setenv(dp.DATA_ROOTS_ENV, os.pathsep.join(str(root) for root in roots))
    monkeypatch.setenv(dp.PROVENANCE_ENV, str(sidecar or "absent.json"))
    dp.clear_cache()


def test_default_root_is_the_measurement_pc():
    assert dp.data_roots() == [dp.DEFAULT_DATA_ROOTS[0]]


def test_the_old_single_root_variable_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("MULTIMODE_DATA_ROOT", "/somewhere")
    with pytest.raises(dp.DataPathError, match=dp.DATA_ROOTS_ENV):
        dp.data_roots()


@pytest.mark.parametrize("recorded", [
    r"C:\experiments\x\data\f.h5",
    "C:/experiments/x/data/f.h5",
    "/Volumes/slab/Multimode/BF5_expts_autosync/x/data/f.h5",
    "x/data/f.h5",
])
def test_relative_cuts_any_known_root(recorded):
    assert str(dp.relative(recorded)) == "x/data/f.h5"


def test_relative_uses_the_first_root_folder(tmp_path):
    """The root folder is the first 'experiments', even if an experiment repeats the name."""
    assert str(dp.relative(r"C:\experiments\experiments\data\J.h5")) == "experiments/data/J.h5"
    assert dp.relative(r"D:\somewhere\else\J.h5") is None
    assert dp.relative("") is None


def test_relative_cuts_a_configured_root_with_any_name(tmp_path, monkeypatch):
    root = tmp_path / "my_data"
    _use(monkeypatch, root)
    assert str(dp.relative(root / "x" / "assembled_data" / "a.yaml")) == "x/assembled_data/a.yaml"


def test_locate_takes_the_first_root_that_has_the_file(tmp_path, monkeypatch):
    """Old data moved to the NAS, new data on pippin: each is found where it is."""
    nas, pippin = tmp_path / "nas", tmp_path / "pippin"
    (nas / "old" / "assembled_data").mkdir(parents=True)
    (nas / "old" / "assembled_data" / "scan.yaml").write_text("")
    (pippin / "new" / "data").mkdir(parents=True)
    (pippin / "new" / "data" / "job.h5").write_text("")
    (pippin / "old" / "assembled_data").mkdir(parents=True)  # a folder on both: the file decides
    _use(monkeypatch, pippin, nas)
    assert dp.locate("old/assembled_data/scan.yaml") == nas / "old" / "assembled_data" / "scan.yaml"
    assert dp.locate(r"C:\experiments\new\data\job.h5") == pippin / "new" / "data" / "job.h5"
    assert dp.locate("new") == pippin / "new"


def test_locate_names_the_roots_and_which_are_not_mounted(tmp_path, monkeypatch):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    _use(monkeypatch, mounted, tmp_path / "gone")
    with pytest.raises(dp.DataPathError) as error:
        dp.locate("x/data/f.h5")
    assert "gone (not mounted)" in str(error.value)
    assert isinstance(error.value, FileNotFoundError)


def test_locate_returns_an_existing_local_path_as_it_is(tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path / "root")
    path = tmp_path / "elsewhere.h5"
    path.write_text("")
    assert dp.locate(path) == path


def test_recorded_folder_resolves_without_walking(tmp_path, monkeypatch):
    """The sidecar's folder is used: a decoy with the same ID in another folder is not seen."""
    job = "JOB-20260815-00009"
    root = _tree(tmp_path / "experiments", "260814_qsim_encspec", [job])
    _tree(root, "260526_qsim_darkmode", [job])
    sidecar = _sidecar(tmp_path, {
        job: rf"C:\experiments\260814_qsim_encspec\data\{job}_QsimBaseExperiment.h5"})
    _use(monkeypatch, root, sidecar=sidecar)
    assert dp.resolve_job_path(job).parent.parent.name == "260814_qsim_encspec"


def test_jobs_resolve_across_roots_in_input_order(tmp_path, monkeypatch):
    old, new = "JOB-20260815-00009", "JOB-20261110-00001"
    nas = _tree(tmp_path / "nas", "260814_qsim_encspec", [old])
    pippin = _tree(tmp_path / "pippin", "261110_agp", [new])
    sidecar = _sidecar(tmp_path, {
        old: rf"C:\experiments\260814_qsim_encspec\data\{old}_QsimBaseExperiment.h5"})
    _use(monkeypatch, nas, pippin, sidecar=sidecar)
    paths = dp.resolve_job_paths([new, old])  # new is not in the sidecar: found by the walk
    assert list(paths) == [new, old]
    assert paths[new].is_relative_to(pippin) and paths[old].is_relative_to(nas)


def test_a_stale_record_falls_back_to_the_walk(tmp_path, monkeypatch):
    job = "JOB-20260815-00009"
    root = _tree(tmp_path / "experiments", "260814_qsim_encspec", [job])
    sidecar = _sidecar(tmp_path, {
        job: rf"C:\experiments\260526_qsim_darkmode\data\{job}_QsimBaseExperiment.h5"})
    _use(monkeypatch, root, sidecar=sidecar)
    assert dp.resolve_job_path(job).parent.parent.name == "260814_qsim_encspec"


def test_missing_jobs_are_reported_together(tmp_path, monkeypatch):
    root = _tree(tmp_path / "experiments", "260814_qsim_encspec")
    _use(monkeypatch, root, sidecar=_sidecar(tmp_path, {}))
    with pytest.raises(dp.DataPathError, match="4 of 4 jobs not found"):
        dp.resolve_job_paths([f"JOB-20260815-{n:05d}" for n in range(9, 13)])


def test_the_real_sidecar_covers_the_august_campaign():
    """Reads the shipped sidecar only, so this holds off-prod too."""
    table = dp._provenance_index(dp.DEFAULT_PROVENANCE)
    jobs = [f"JOB-20260815-{n:05d}" for n in range(9, 17)]
    assert all(job in table and dp.relative(table[job]) is not None for job in jobs)

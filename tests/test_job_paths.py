# -*- coding: utf-8 -*-
"""Job-ID to path resolution, per spec section 9.1.

Hermetic by construction: every test builds its own data tree and its own
provenance sidecar under ``tmp_path``. Nothing here touches the real data
share, the cloud-synced vault, or the job database -- which is the property
under test as much as it is a convenience, since the whole point of the
provenance backend is that offline analysis needs none of those three.
"""
import json

import pytest

from experiments import job_paths as jp


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    """Drop any resolver environment the developer's shell happens to export."""
    for var in (jp.DATA_ROOT_ENV, jp.BACKEND_ENV, jp.PROVENANCE_ENV,
                jp.VAULT_ROOT_ENV, jp.VAULT_USER_ENV):
        monkeypatch.delenv(var, raising=False)
    jp.clear_cache()
    yield
    jp.clear_cache()


def _tree(tmp_path, subdir, job_ids, suffix="_QsimBaseExperiment"):
    """Create ``{root}/{subdir}/data/{JOB}{suffix}.h5`` files, return the root."""
    root = tmp_path / "experiments"
    data = root / subdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    for job in job_ids:
        (data / f"{job}{suffix}.h5").write_bytes(b"")
    return root


def _sidecar(tmp_path, mapping, name="prov.json"):
    """Write a provenance sidecar recording Windows-style acquisition paths."""
    path = tmp_path / name
    path.write_text(json.dumps({
        job: {"job_id": job, "data_file_path": recorded}
        for job, recorded in mapping.items()
    }))
    return path


def _use(monkeypatch, root, sidecar=None, backend=None):
    monkeypatch.setenv(jp.DATA_ROOT_ENV, str(root))
    if sidecar is not None:
        monkeypatch.setenv(jp.PROVENANCE_ENV, str(sidecar))
    if backend is not None:
        monkeypatch.setenv(jp.BACKEND_ENV, backend)
    jp.clear_cache()


def test_provenance_is_the_default_backend():
    """The default must not be the glob: off-prod that walk is minutes long."""
    assert jp.backend() == "provenance"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv(jp.BACKEND_ENV, "database")
    with pytest.raises(jp.JobPathError, match="expected 'provenance'"):
        jp.backend()


def test_recorded_subdir_resolves_without_walking(tmp_path, monkeypatch):
    """The recorded subdirectory is enough; no directory walk is needed.

    Proven by giving the sidecar the right answer while leaving a decoy file
    under a *different* subdirectory. A glob would find the decoy too and have
    to disambiguate; reading the record cannot.
    """
    job = "JOB-20260815-00009"
    root = _tree(tmp_path, "260814_qsim_encspec", [job])
    _tree(tmp_path, "260526_qsim_darkmode", ["JOB-20260722-00683"])
    sidecar = _sidecar(tmp_path, {
        job: r"C:\experiments\260814_qsim_encspec\data\%s_QsimBaseExperiment.h5" % job,
    })
    _use(monkeypatch, root, sidecar)

    resolved = jp.resolve_job_paths([job])[job]
    assert resolved == root / "260814_qsim_encspec" / "data" / f"{job}_QsimBaseExperiment.h5"
    assert resolved.is_file()


def test_forward_slash_records_resolve_too(tmp_path, monkeypatch):
    """Records are Windows paths today, but the anchor split must not care."""
    job = "JOB-20260815-00010"
    root = _tree(tmp_path, "260814_qsim_encspec", [job])
    sidecar = _sidecar(tmp_path, {
        job: f"C:/experiments/260814_qsim_encspec/data/{job}_QsimBaseExperiment.h5",
    })
    _use(monkeypatch, root, sidecar)
    assert jp.resolve_job_paths([job])[job].is_file()


def test_input_order_is_preserved(tmp_path, monkeypatch):
    jobs = [f"JOB-20260815-{n:05d}" for n in (12, 9, 11, 10)]
    root = _tree(tmp_path, "260814_qsim_encspec", jobs)
    sidecar = _sidecar(tmp_path, {
        j: rf"C:\experiments\260814_qsim_encspec\data\{j}_QsimBaseExperiment.h5"
        for j in jobs
    })
    _use(monkeypatch, root, sidecar)
    assert list(jp.resolve_job_paths(jobs)) == jobs


def test_job_acquired_since_the_export_falls_back_to_the_glob(tmp_path, monkeypatch):
    """A sidecar miss must not be fatal, or fresh acquisition breaks on prod."""
    recorded, fresh = "JOB-20260815-00009", "JOB-20260826-00001"
    root = _tree(tmp_path, "260814_qsim_encspec", [recorded])
    _tree(tmp_path, "260826_job_worker", [fresh])
    sidecar = _sidecar(tmp_path, {
        recorded: rf"C:\experiments\260814_qsim_encspec\data\{recorded}_QsimBaseExperiment.h5",
    })
    _use(monkeypatch, root, sidecar)

    paths = jp.resolve_job_paths([recorded, fresh])
    assert paths[recorded].parent.parent.name == "260814_qsim_encspec"
    assert paths[fresh].parent.parent.name == "260826_job_worker"


def test_a_stale_record_falls_back_rather_than_lying(tmp_path, monkeypatch):
    """If the recorded path is absent on disk, fall back instead of returning it."""
    job = "JOB-20260815-00009"
    root = _tree(tmp_path, "260814_qsim_encspec", [job])
    sidecar = _sidecar(tmp_path, {
        job: rf"C:\experiments\260526_qsim_darkmode\data\{job}_QsimBaseExperiment.h5",
    })
    _use(monkeypatch, root, sidecar)
    assert jp.resolve_job_paths([job])[job].parent.parent.name == "260814_qsim_encspec"


def test_missing_jobs_are_reported_together(tmp_path, monkeypatch):
    """One diagnosis per bad ID list, not one per call."""
    root = _tree(tmp_path, "260814_qsim_encspec", [])
    sidecar = _sidecar(tmp_path, {})
    _use(monkeypatch, root, sidecar)
    absent = [f"JOB-20260815-{n:05d}" for n in range(9, 13)]
    with pytest.raises(jp.JobPathError) as excinfo:
        jp.resolve_job_paths(absent)
    message = str(excinfo.value)
    assert "4 of 4 jobs not found" in message
    # The remedy names the exporter, not the vault and not the database.
    assert "export_job_provenance.py" in message


def test_absent_sidecar_names_the_exporter(tmp_path, monkeypatch):
    root = _tree(tmp_path, "260814_qsim_encspec", [])
    _use(monkeypatch, root, tmp_path / "does_not_exist.json")
    with pytest.raises(jp.JobPathError, match="export_job_provenance.py"):
        jp.resolve_job_paths(["JOB-20260815-00009"])


def test_missing_data_root_names_the_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(jp.DATA_ROOT_ENV, str(tmp_path / "not_mounted"))
    with pytest.raises(jp.JobPathError, match=jp.DATA_ROOT_ENV):
        jp.data_root()


def test_reroot_declines_paths_outside_the_data_tree(tmp_path):
    """A record with no anchor component is unusable; guessing is worse."""
    assert jp._reroot(r"D:\somewhere\else\JOB-1.h5", tmp_path) is None
    assert jp._reroot("", tmp_path) is None


def test_reroot_anchors_on_the_root_component(tmp_path):
    """The first 'experiments' is the root, even if a subdir repeats the name."""
    out = jp._reroot(r"C:\experiments\experiments\data\JOB-1.h5", tmp_path)
    assert out == tmp_path / "experiments" / "data" / "JOB-1.h5"


def test_the_real_sidecar_covers_the_august_campaign():
    """The shipped export must cover the published sets, or analysis cannot run.

    Reads the sidecar only -- no data tree, so this holds off-prod too.
    """
    table = jp._provenance_index(jp.DEFAULT_PROVENANCE)
    quick_plot = [f"JOB-20260815-{n:05d}" for n in range(9, 17)]
    assert [j for j in quick_plot if j not in table] == []
    assert all(jp.DATA_ROOT_ANCHOR in table[j].replace("\\", "/").split("/")
               for j in quick_plot)

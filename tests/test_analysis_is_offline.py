"""The analysis path reads HDF5 and talks to nothing.

Four properties, each checked by exercising it rather than by reading source:

1. loading saved jobs opens no database connection;
2. loading saved jobs makes no HTTP request to the job server;
3. a saved job whose Floquet timing cannot be recovered raises, and the error
   says what to supply, rather than carrying a placeholder into the arithmetic;
4. the loaders accept no ``client``, and reject a pickle.

What is deliberately *not* tested: whether ``job_server`` gets imported.
Importing it is inert -- ``job_server/database.py`` only binds a path constant
and ``_db_instance = None`` at module level, and the engine plus
``create_tables()`` happen on the first ``get_database()`` call. So the import
costs nothing and proves nothing; what matters is whether anything *calls*
into it. Properties 1 and 2 check exactly that, which is why they assert on
``_db_instance`` and on a request guard rather than on ``sys.modules``.

The first two run in subprocesses. They have to: ``_db_instance`` is a
process-global, and other tests in the same pytest session legitimately open
the database, so its value in this process says nothing about what a load did.
"""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# One real load through the notebook helper, on the smallest pinned dataset.
# Eight files, so it costs a couple of seconds even over SMB.
A_REAL_LOAD = """
from experiments.qsim.legacy_mbr import MBRSpectrumExperiment
from experiments.saved_jobs import load_aggregate

JOBS = [f"JOB-20260815-{n:05d}" for n in range(9, 17)]
expt = load_aggregate(JOBS, owner=MBRSpectrumExperiment)
assert len(expt.batch_job_ids) == 8, expt.batch_job_ids
expt.analyze(cycle_branches={}, fft_window="raw", zero_padding=1,
             spectrum_method="fft")
"""

# Refuses every outgoing request, so any call to the job server raises rather
# than quietly succeeding on a machine that happens to have a server running.
NO_HTTP_GUARD = """
import requests

def _refuse(*args, **kwargs):
    raise AssertionError(f"analysis made an HTTP request: {args} {kwargs}")

requests.Session.request = _refuse
requests.Session.send = _refuse
requests.api.request = _refuse
"""


def _run(script):
    """Run `script` in a fresh interpreter at the repo root."""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    return result


def _run_ok(script):
    result = _run(script)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result


def test_loading_saved_jobs_opens_no_database():
    """A real load leaves the job database untouched.

    `get_database()` is what creates the engine and the jobs.db file, and it
    memoises into `_db_instance`. If that is still None after loading and
    analyzing real jobs, nothing in the path asked the database anything --
    which is the actual property wanted, and the one the pre-refactor loader
    broke by querying jobs.db for each job's subdirectory.
    """
    result = _run_ok(A_REAL_LOAD + """
import json, sys
opened = None
if "job_server.database" in sys.modules:
    opened = sys.modules["job_server.database"]._db_instance
print("RESULT", json.dumps(opened is None))
""")
    line = next(l for l in result.stdout.splitlines() if l.startswith("RESULT "))
    assert json.loads(line[len("RESULT "):]), "a database connection was opened"


def test_loading_saved_jobs_makes_no_http_request():
    """A real load performs no HTTP, so no job server need be running.

    This is the one that used to fail: `load_encoding_spectroscopy` called
    `client.get_status(job_id)` once per job -- 140 round trips for the N=3
    set -- purely to find each file and unpickle it.
    """
    _run_ok(NO_HTTP_GUARD + A_REAL_LOAD)


def test_the_http_guard_would_catch_a_job_server_call():
    """Mutation check for the test above: the guard must actually bite.

    Without this, a typo'd patch target would make the test above vacuous.
    """
    result = _run(NO_HTTP_GUARD + """
from job_server import JobClient
JobClient().get_status("JOB-20260815-00009")
""")
    assert result.returncode != 0
    assert "analysis made an HTTP request" in result.stdout + result.stderr


ANALYSIS_NOTEBOOKS = sorted(
    (REPO_ROOT / "analysis_notebooks" / "202609_qsim_migration").glob("*.py"))


def _import_block(path):
    """-> the notebook's top-level import statements, as runnable source."""
    lines = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    return "\n".join(
        "\n".join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))


def test_there_are_analysis_notebooks_to_check():
    """Guards the glob above: an empty parametrization would pass silently."""
    assert len(ANALYSIS_NOTEBOOKS) >= 4, [p.name for p in ANALYSIS_NOTEBOOKS]


@pytest.mark.parametrize("notebook", ANALYSIS_NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_imports_contact_nothing(notebook):
    """Each analysis notebook's imports run without contacting anything.

    This is what started the change: a pure analysis notebook constructed a
    `JobClient` at module scope, so opening it required a running server.
    Running the import block for real is the only check that catches it, since
    the failure is at import time and a notebook is not imported by anything.
    """
    block = _import_block(notebook)
    assert "experiments" in block, f"no repo imports found in {notebook.name}"
    _run_ok(NO_HTTP_GUARD + block + """
import sys
if "job_server.database" in sys.modules:
    assert sys.modules["job_server.database"]._db_instance is None, \
        "importing the notebook opened the job database"
""")


def _write_h5(path, cfg, avgi):
    with h5py.File(path, "w") as handle:
        handle.attrs["config"] = json.dumps(cfg)
        handle["avgi"] = np.asarray(avgi)
        handle["xpts"] = np.asarray([0.0, 180.0])
        handle["ypts"] = np.asarray([0])


MINIMAL_CFG = {
    "expt": {"swap_stors": [1, 2, 3, 4], "qubits": [0],
             "spectroscopy_occupations": [1, 0, 0, 0, 0]},
    "device": {"manipulate": {"kerr": [-0.0105]},
               "readout": {"Ig": [0.0], "Ie": [1.0]}},
}


def test_unrecoverable_timing_raises_and_says_what_to_supply(tmp_path):
    """A file with no provenance raises; it does not load with a placeholder.

    The cycle time divides into every energy in the result, so a NaN here
    would poison the whole spectrum quietly. The error names the two ways out.
    """
    from experiments.saved_jobs import SavedJobError, load_job

    path = tmp_path / "JOB-19990101-00001_EncodingHamiltonianSpectroscopyExperiment.h5"
    _write_h5(path, MINIMAL_CFG, [[0.1, 0.2]])

    with pytest.raises(SavedJobError) as error:
        load_job("JOB-19990101-00001", path=path, provenance={})

    message = str(error.value)
    assert "timing=" in message
    assert "export_job_provenance" in message


def test_supplied_timing_is_used_when_provenance_is_absent(tmp_path):
    """The escape hatch works: the same file loads once timing is given."""
    from experiments.saved_jobs import load_job

    path = tmp_path / "JOB-19990101-00002_EncodingHamiltonianSpectroscopyExperiment.h5"
    _write_h5(path, MINIMAL_CFG, [[0.1, 0.2]])

    job = load_job("JOB-19990101-00002", path=path, provenance={},
                   timing=dict(floquet_cycle_us=0.5, m1s_pi_fracs=[40] * 7))

    assert job.prog.calculate_floquet_cycle_us() == 0.5
    assert job.prog.source == "supplied by caller"


def test_derived_params_attribute_is_preferred_over_the_sidecar(tmp_path):
    """A file carrying its own `derived_params` needs no provenance sidecar.

    This is the format the acquisition side writes. Reading it here is what
    makes that change a pure addition: files that have the attribute stop
    needing the sidecar, and files that do not keep working as before.
    """
    from experiments.saved_jobs import load_job

    path = tmp_path / "JOB-19990101-00003_EncodingHamiltonianSpectroscopyExperiment.h5"
    _write_h5(path, MINIMAL_CFG, [[0.1, 0.2]])
    with h5py.File(path, "a") as handle:
        handle.attrs["derived_params"] = json.dumps(
            {"floquet_cycle_us": 0.75, "m1s_pi_fracs": [40] * 7})

    job = load_job("JOB-19990101-00003", path=path, provenance={})

    assert job.prog.calculate_floquet_cycle_us() == 0.75
    assert "derived_params" in job.prog.source


def test_derived_params_couplings_are_cross_checked(tmp_path):
    """A `derived_params` block that disagrees with itself raises."""
    from experiments.saved_jobs import SavedJobError, load_job

    path = tmp_path / "JOB-19990101-00004_EncodingHamiltonianSpectroscopyExperiment.h5"
    _write_h5(path, MINIMAL_CFG, [[0.1, 0.2]])
    with h5py.File(path, "a") as handle:
        handle.attrs["derived_params"] = json.dumps({
            "floquet_cycle_us": 0.75,
            "m1s_pi_fracs": [40] * 7,
            # 1/(4 * 40 * 0.75) = 0.008333..., so this is wrong on purpose.
            "couplings_MHz": [0.5] * 7,
        })

    with pytest.raises(SavedJobError, match="disagree"):
        load_job("JOB-19990101-00004", path=path, provenance={})


@pytest.mark.parametrize("loader", ["from_job_ids", "from_job_files"])
def test_loaders_take_no_client(loader):
    """The `client=` argument is gone, not merely unused."""
    import inspect

    from experiments.qsim.legacy_mbr import MBRSpectrumExperiment

    parameters = inspect.signature(getattr(MBRSpectrumExperiment, loader)).parameters
    assert "client" not in parameters, sorted(parameters)


def test_from_job_files_rejects_a_pickle(tmp_path):
    """Pointing the loader at a job pickle raises and says to use the .h5."""
    from experiments.qsim.legacy_mbr import MBRSpectrumExperiment

    pickle_path = tmp_path / "JOB-19990101-00005_expt.pkl"
    pickle_path.write_bytes(b"not really a pickle")

    with pytest.raises(ValueError, match="HDF5"):
        MBRSpectrumExperiment.from_job_files(pickle_path)


# --------------------------------------------------------------------------
# config_versions: written by the worker, read by the resolver
# --------------------------------------------------------------------------


def test_worker_records_config_versions_in_the_data_file(tmp_path):
    """The worker writes the config version IDs into the file it just saved.

    `_write_config_versions` needs nothing from worker state, so it is called
    on a bare instance rather than standing up a database and a station.
    """
    from job_server.worker import JobWorker

    path = tmp_path / "JOB-19990101-00010_MBRSpectrumExperiment.h5"
    _write_h5(path, MINIMAL_CFG, [[0.1, 0.2]])

    JobWorker._write_config_versions(
        JobWorker.__new__(JobWorker), path,
        {"hardware_config": "CFG-HW-20260814-00074",
         "floquet_storage_swap": "CFG-FL-20260814-00076",
         "man1_storage_swap": None})

    with h5py.File(path, "r") as handle:
        recorded = json.loads(handle.attrs["config_versions"])
    # None-valued entries are dropped rather than recorded as null: an absent
    # key means "not versioned", which is not the same as a null version.
    assert recorded == {"hardware_config": "CFG-HW-20260814-00074",
                        "floquet_storage_swap": "CFG-FL-20260814-00076"}


def test_a_failed_attribute_write_does_not_lose_the_job(tmp_path):
    """Provenance is best-effort: a job whose data is on disk stays completed.

    Failing the job because an attribute could not be added would discard a
    real measurement, and the exported sidecar still covers this case.
    """
    from job_server.worker import JobWorker

    missing = tmp_path / "does-not-exist.h5"
    JobWorker._write_config_versions(
        JobWorker.__new__(JobWorker), missing, {"hardware_config": "CFG-HW-1"})
    assert not missing.exists()


def test_config_versions_attribute_resolves_timing_without_the_sidecar(tmp_path):
    """A file recording its config versions needs no provenance sidecar.

    This is what the worker-side write buys: `provenance={}` here, so the only
    way the loader can find the Floquet version is the file's own attribute,
    and the timing then comes from the versioned archive.
    """
    from experiments.saved_jobs import load_job

    path = tmp_path / "JOB-20260815-00009_EncodingHamiltonianSpectroscopyExperiment.h5"
    # The real August config, so the resolver has a version it can actually
    # read and a value already pinned elsewhere in the suite.
    cfg, _ = load_h5_for_fixture()
    _write_h5(path, cfg, [[0.1, 0.2]])
    with h5py.File(path, "a") as handle:
        handle.attrs["config_versions"] = json.dumps(
            {"floquet_storage_swap": "CFG-FL-20260814-00076"})

    job = load_job("JOB-20260815-00009", path=path, provenance={})

    assert job.prog.calculate_floquet_cycle_us() == 0.7254464285714286
    assert job.prog.source.startswith("versioned config CFG-FL-")


def load_h5_for_fixture():
    """-> the real cfg of JOB-20260815-00009, for the resolver to read.

    The resolver needs genuine `expt`/`hw`/`device` sections -- swap_stors,
    the flux DAC channels, the manipulate ramp sigma -- so a hand-written
    minimal config will not do.
    """
    from experiments.job_paths import resolve_job_path
    from experiments.saved_jobs import load_h5

    return load_h5(resolve_job_path("JOB-20260815-00009"))

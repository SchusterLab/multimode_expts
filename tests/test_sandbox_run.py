# -*- coding: utf-8 -*-
"""A sandbox session never reaches the job queue.

A sandbox station runs code from a checkout that is not the main one: a
worktree, a refactor branch, the notebook test suite. The queue worker only
runs the main checkout, so a queued job would silently test the wrong code on
the real device. These tests hold the rule that makes a sandbox run safe:
every runner's `execute()` runs locally whatever `use_queue` says, and a
direct `run()` refuses.

They use a real-instrument station (`is_mock` False), because that is the case
the rule is for; mock stations already route locally on their own.

Also here: the `RunSettings` the notebooks read, and the suite driver's cell
filtering, which decides what a suite run executes.

Run:  pixi run pytest tests/test_sandbox_run.py -v
"""
from unittest.mock import MagicMock

import jupytext
import pytest
from slab import AttrDict

from experiments.batch_runner import BatchRunner
from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.notebook_helpers.run_mode import RunSettings, run_settings
from experiments.sweep_runner import SweepRunner
from tests.test_characterization_runner import MockExperiment
from tools.run_qsim_suite import load_notebook, skipped_tags


@pytest.fixture
def sandbox_station(station):
    station._is_mock = False
    station.sandbox = True
    return station


def _characterization(station, **kwargs):
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10, start=0)),
        job_client=MagicMock(),
        show=False,
        **kwargs,
    )
    runner.run_local = MagicMock(return_value="local")
    return runner


@pytest.mark.parametrize("runner_default, call_arg", [
    (True, None), (True, True), (False, True), (False, None),
])
def test_characterization_execute_runs_locally(sandbox_station, runner_default, call_arg):
    runner = _characterization(sandbox_station, use_queue=runner_default)
    assert runner.execute(use_queue=call_arg) == "local"
    runner.job_client.submit_job.assert_not_called()


def test_characterization_run_refuses(sandbox_station):
    runner = _characterization(sandbox_station)
    with pytest.raises(RuntimeError, match="sandbox"):
        runner.run()
    runner.job_client.submit_job.assert_not_called()


def test_without_sandbox_the_queue_is_still_used(station):
    # Guards against the rule leaking: a normal real-instrument station keeps
    # routing to the queue.
    station._is_mock = False
    station.sandbox = False
    runner = _characterization(station)
    runner.run = MagicMock(return_value="queued")
    assert runner.execute() == "queued"


def test_batch_execute_runs_locally_even_with_explicit_queue(sandbox_station):
    client = MagicMock()
    runner = BatchRunner(
        station=sandbox_station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10, start=0)),
        job_client=client,
        show=False,
    )
    runner.run_local = MagicMock(side_effect=lambda **kw: MagicMock())
    runner._aggregate = MagicMock(return_value="aggregate")
    assert runner.execute([dict(start=0), dict(start=1)], use_queue=True) == "aggregate"
    assert runner.run_local.call_count == 2
    client.submit_job.assert_not_called()


def test_sweep_execute_runs_locally_and_run_refuses(sandbox_station):
    runner = SweepRunner(
        station=sandbox_station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10, start=0)),
        sweep_param="freq",
        job_client=MagicMock(),
    )
    runner.run_local = MagicMock(return_value="local")
    assert runner.execute(0, 1, 2, use_queue=True) == "local"
    with pytest.raises(RuntimeError, match="sandbox"):
        runner.run(0, 1, 2)


def test_log_measurement_skips_in_sandbox():
    from experiments.station import MultimodeStation

    station = MultimodeStation.__new__(MultimodeStation)
    station._is_mock = False
    station.sandbox = True
    station.vault_root = "/nonexistent/vault"
    assert station.log_measurement(title="t", project="p") is None


# --------------------------------------------------------------------------
# RunSettings
# --------------------------------------------------------------------------

def test_default_settings_are_a_science_run(monkeypatch):
    for var in ("MULTIMODE_RUN_MODE", "MULTIMODE_RUN_PROFILE", "MULTIMODE_RUN_CONFIGS"):
        monkeypatch.delenv(var, raising=False)
    run = run_settings()
    assert (run.mode, run.profile, run.configs) == ("queue", "full", None)
    assert not run.sandbox
    assert run.pick(1000, smoke=10) == 1000
    assert run.experiment_name("260818_qsim_spectroscopy") == "260818_qsim_spectroscopy"
    notebook = {"hardware_config": "CFG-HW-1"}
    assert run.config_dict(notebook) == notebook


@pytest.mark.parametrize("mode", ["mock", "sandbox"])
def test_suite_modes_are_sandboxed(monkeypatch, mode):
    monkeypatch.setenv("MULTIMODE_RUN_MODE", mode)
    monkeypatch.setenv("MULTIMODE_RUN_PROFILE", "smoke")
    run = run_settings()
    assert run.sandbox
    assert run.mock == (mode == "mock")
    assert run.pick(1000, smoke=10) == 10
    name = run.experiment_name("260818_qsim_spectroscopy")
    assert name.endswith(f"_suite_{mode}") and "qsim_spectroscopy" not in name


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="MULTIMODE_RUN_MODE"):
        RunSettings(mode="hardware")


# --------------------------------------------------------------------------
# Suite driver cell filtering
# --------------------------------------------------------------------------

NOTEBOOK = '''# %%
a = 1

# %% tags=["suite-skip"]
b = 2

# %% tags=["mock-skip"]
c = 3
'''


@pytest.mark.parametrize("mode, kept", [
    ("mock", ["a = 1"]),
    ("sandbox", ["a = 1", "c = 3"]),
    ("analysis", ["a = 1", "c = 3"]),
])
def test_driver_drops_tagged_cells(tmp_path, mode, kept):
    path = tmp_path / "nb.py"
    path.write_text(NOTEBOOK)
    nb, dropped = load_notebook(path, skipped_tags(mode))
    assert [c.source for c in nb.cells] == kept
    assert dropped == 3 - len(kept)


def test_every_suite_notebook_parses_and_keeps_its_session_cell():
    # The tags are written by hand into jupytext headers; a malformed one
    # would silently turn a cell into markdown or drop it.
    from tools.run_qsim_suite import SUITES

    for path in SUITES["measurement"]:
        nb, _ = load_notebook(path, skipped_tags("mock"))
        sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "run=RUN" in sources, path.name
        raw = jupytext.read(path)
        for cell in raw.cells:
            for tag in cell.metadata.get("tags", []):
                assert tag in {"suite-skip", "mock-skip"}, (path.name, tag)

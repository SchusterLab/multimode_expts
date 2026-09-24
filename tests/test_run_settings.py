# -*- coding: utf-8 -*-
"""The `RunSettings` the qsim notebooks read, and the suite driver's cell filtering.

Run:  pixi run pytest tests/test_run_settings.py -v
"""
import jupytext
import pytest

from experiments.qsim.notebook_helpers import run_mode
from experiments.qsim.notebook_helpers.run_mode import (
    RunSettings,
    is_main_checkout,
    run_settings,
)
from tools.run_qsim_suite import SKIPPED_TAGS, load_notebook

RUN_VARS = ("MULTIMODE_RUN_USE_QUEUE", "MULTIMODE_RUN_PROFILE",
            "MULTIMODE_RUN_CONFIGS")

CONFIG_DICT = {
    "hardware_config": "CFG-HW-1",
    "multiphoton_config": "CFG-MP-1",
    "man1_storage_swap": "CFG-M1-1",
    "floquet_storage_swap": "CFG-FL-1",
}


@pytest.fixture
def clean_env(monkeypatch):
    for var in RUN_VARS:
        monkeypatch.delenv(var, raising=False)
    # run_settings() also reads the repo-root .env; keep it from setting these.
    monkeypatch.setattr("experiments.qsim.notebook_helpers.run_mode.load_env",
                        lambda: None)
    return monkeypatch


def test_default_settings_are_a_normal_run(clean_env):
    clean_env.setattr(run_mode, "is_main_checkout", lambda: True)
    run = run_settings()
    assert (run.use_queue, run.profile, run.configs) == (True, "full", None)
    assert run.pick(1000, smoke=10) == 1000
    # Through the queue, the station gets the version IDs unchanged.
    assert run.station_configs(CONFIG_DICT) == {
        "hardware_config": "CFG-HW-1",
        "multiphoton_config": "CFG-MP-1",
        "storage_man_file": "CFG-M1-1",
        "floquet_file": "CFG-FL-1",
    }


def test_a_worktree_runs_directly_by_default(clean_env):
    # The worker runs only the main checkout's code.
    clean_env.setattr(run_mode, "is_main_checkout", lambda: False)
    assert run_settings().use_queue is False


def test_use_queue_flag_overrides_the_checkout_default(clean_env):
    clean_env.setattr(run_mode, "is_main_checkout", lambda: False)
    clean_env.setenv("MULTIMODE_RUN_USE_QUEUE", "1")
    assert run_settings().use_queue is True
    clean_env.setattr(run_mode, "is_main_checkout", lambda: True)
    clean_env.setenv("MULTIMODE_RUN_USE_QUEUE", "0")
    clean_env.setenv("MULTIMODE_RUN_PROFILE", "smoke")
    run = run_settings()
    assert (run.use_queue, run.smoke) == (False, True)
    assert run.pick(1000, smoke=10) == 10


def test_is_main_checkout(tmp_path):
    # In a linked worktree, .git is a file that points to the main checkout.
    (tmp_path / "main" / ".git").mkdir(parents=True)
    (tmp_path / "worktree").mkdir()
    (tmp_path / "worktree" / ".git").write_text("gitdir: ../main/.git/worktrees/w")
    assert is_main_checkout(tmp_path / "main")
    assert not is_main_checkout(tmp_path / "worktree")


def test_bad_flag_is_refused(clean_env):
    clean_env.setenv("MULTIMODE_RUN_USE_QUEUE", "yes")
    with pytest.raises(ValueError, match="MULTIMODE_RUN_USE_QUEUE"):
        run_settings()


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="MULTIMODE_RUN_PROFILE"):
        RunSettings(profile="quick")


def test_missing_config_key_is_refused():
    with pytest.raises(KeyError, match="floquet_storage_swap"):
        RunSettings().station_configs({"hardware_config": "CFG-HW-1"})


# --------------------------------------------------------------------------
# Suite driver cell filtering
# --------------------------------------------------------------------------

NOTEBOOK = '''# %%
a = 1

# %% tags=["suite-skip"]
b = 2

# %% tags=["raises-exception"]
c = 3
'''


def test_driver_drops_suite_skip_cells(tmp_path):
    path = tmp_path / "nb.py"
    path.write_text(NOTEBOOK)
    nb, dropped = load_notebook(path, SKIPPED_TAGS)
    assert [c.source for c in nb.cells] == ["a = 1", "c = 3"]
    assert dropped == 1


def test_every_suite_notebook_parses_and_keeps_its_station_cell():
    # The tags are written by hand into jupytext headers; a malformed one
    # would silently turn a cell into markdown or drop it.
    from tools.run_qsim_suite import SUITES

    for path in SUITES["measurement"]:
        nb, _ = load_notebook(path, SKIPPED_TAGS)
        sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "MultimodeStation(" in sources, path.name
        raw = jupytext.read(path)
        for cell in raw.cells:
            for tag in cell.metadata.get("tags", []):
                assert tag in {"suite-skip", "raises-exception"}, (path.name, tag)


def test_expected_failures_split_raised_from_clean():
    import nbformat
    from tools.run_qsim_suite import expected_failures

    nb = nbformat.v4.new_notebook()
    raised = nbformat.v4.new_code_cell("1/0", metadata={"tags": ["raises-exception"]})
    raised.outputs = [nbformat.v4.new_output("error", ename="ZeroDivisionError",
                                             evalue="", traceback=[])]
    clean = nbformat.v4.new_code_cell("1", metadata={"tags": ["raises-exception"]})
    plain = nbformat.v4.new_code_cell("2")
    nb.cells = [plain, raised, clean]
    assert expected_failures(nb) == ([1], [2])

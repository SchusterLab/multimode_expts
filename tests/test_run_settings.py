# -*- coding: utf-8 -*-
"""The `RunSettings` the qsim notebooks read, and the suite driver's cell filtering.

Run:  pixi run pytest tests/test_run_settings.py -v
"""
import jupytext
import pytest

from experiments.qsim.notebook_helpers.run_mode import RunSettings, run_settings
from tools.run_qsim_suite import load_notebook, skipped_tags

RUN_VARS = ("MULTIMODE_RUN_MOCK", "MULTIMODE_RUN_USE_QUEUE",
            "MULTIMODE_RUN_PROFILE", "MULTIMODE_RUN_CONFIGS")

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
    run = run_settings()
    assert (run.mock, run.use_queue, run.profile, run.configs) == (False, True, "full", None)
    assert run.pick(1000, smoke=10) == 1000
    # Through the queue, the station gets the version IDs unchanged.
    assert run.station_configs(CONFIG_DICT) == {
        "hardware_config": "CFG-HW-1",
        "multiphoton_config": "CFG-MP-1",
        "storage_man_file": "CFG-M1-1",
        "floquet_file": "CFG-FL-1",
    }


def test_the_two_flags_are_independent(clean_env):
    clean_env.setenv("MULTIMODE_RUN_MOCK", "0")
    clean_env.setenv("MULTIMODE_RUN_USE_QUEUE", "0")
    clean_env.setenv("MULTIMODE_RUN_PROFILE", "smoke")
    run = run_settings()
    assert (run.mock, run.use_queue, run.smoke) == (False, False, True)
    assert run.pick(1000, smoke=10) == 10


def test_bad_flag_is_refused(clean_env):
    clean_env.setenv("MULTIMODE_RUN_MOCK", "yes")
    with pytest.raises(ValueError, match="MULTIMODE_RUN_MOCK"):
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

# %% tags=["mock-skip"]
c = 3
'''


@pytest.mark.parametrize("mode, kept", [
    ("mock", ["a = 1"]),
    ("hardware", ["a = 1", "c = 3"]),
    ("analysis", ["a = 1", "c = 3"]),
])
def test_driver_drops_tagged_cells(tmp_path, mode, kept):
    path = tmp_path / "nb.py"
    path.write_text(NOTEBOOK)
    nb, dropped = load_notebook(path, skipped_tags(mode))
    assert [c.source for c in nb.cells] == kept
    assert dropped == 3 - len(kept)


def test_every_suite_notebook_parses_and_keeps_its_station_cell():
    # The tags are written by hand into jupytext headers; a malformed one
    # would silently turn a cell into markdown or drop it.
    from tools.run_qsim_suite import SUITES

    for path in SUITES["measurement"]:
        nb, _ = load_notebook(path, skipped_tags("mock"))
        sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "mock=RUN.mock" in sources, path.name
        raw = jupytext.read(path)
        for cell in raw.cells:
            for tag in cell.metadata.get("tags", []):
                assert tag in {"suite-skip", "mock-skip"}, (path.name, tag)

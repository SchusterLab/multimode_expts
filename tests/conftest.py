"""Shared pytest fixtures for the test suite.

conftest.py is special: pytest auto-discovers it (no import anywhere) and makes
every @pytest.fixture defined here injectable BY NAME into any test under this
directory. A test that wants a fixture just names it as a parameter:

    def test_something(station):     # pytest builds `station` and passes it in
        ...

The prize here is `station`: a lightweight stand-in for the heavy, real
MultimodeStation (the global rig-state holder). The characterization- and
sweep-runner tests both need a station to drive their runners. They used to each
copy-paste their own ~80-line MockStation -- which is exactly how the
.soc -> .soccfg rename slipped past in two places at once. Now there is ONE
definition; an API change is a one-line fix here, not a suite-wide hunt.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from slab import AttrDict


class MockStation:
    """Minimal stand-in for MultimodeStation: temp dirs + a fake config, no real
    device.yaml and no Pyro proxy. Always reports is_mock=True, so the runner's
    render/log paths short-circuit (no display, no lab-notebook writes).

    The hardware_cfg here is a deliberate SUPERSET that satisfies both runner
    test files at once: characterization reads device.qubit.pulses, sweep reads
    device.manipulate / device.multiphoton, and both need device.storage to
    exist (run_local stashes dataset handles into device.storage._ds_*). Keeping
    it broad means neither test file needs its own station variant.
    """

    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()

        self.data_path = Path(self.temp_dir) / "data"
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.hardware_config_file = Path(self.temp_dir) / "device.yaml"
        self.hardware_config_file.touch()

        # Mock QickConfig (.soccfg, NOT the old .soc) and the instrument manager
        # the runner assigns onto each experiment.
        self.soccfg = MagicMock()
        self.im = {"soc": MagicMock()}

        self._is_mock = True
        self.user = "test_user"

        self.hardware_cfg = AttrDict({
            "device": {
                "readout": {"relax_delay": [1000]},
                "qubit": {
                    "f_ge": [5000],
                    "pulses": {
                        "pi_ge": {"gain": [4500], "sigma": [0.05]},
                        "hpi_ge": {"gain": [2250], "sigma": [0.05]},
                        "pi_ef": {"gain": [4000], "sigma": [0.05]},
                        "hpi_ef": {"gain": [2000], "sigma": [0.05]},
                    },
                },
                "manipulate": {"f0g1_freq": [2000]},
                "multiphoton": {"pi": {"fn-gn+1": {"frequency": [2000]}}},
                "storage": {"_ds_storage": None, "_ds_floquet": None},
            },
            "expt": {},
        })

        # Dataset handles the runner threads into cfg, plus the per-run dataset
        # the sweep postprocessor test pokes at.
        self.ds_storage = MagicMock()
        self.ds_floquet = None
        self.ds_thisrun = MagicMock()
        self.ds_thisrun.get_freq = MagicMock(return_value=5000)
        self.ds_thisrun.get_gain = MagicMock(return_value=8000)

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


@pytest.fixture
def station():
    """A fresh MockStation per test.

    Function scope (the default) means this runs once per test, so each test gets
    its own object -- mutations in one test can't leak into the next. The teardown
    after `yield` removes the temp directory, which Python's garbage collector
    would otherwise leak (it frees the object, not the folder on disk).
    """
    st = MockStation()
    yield st
    st.cleanup()

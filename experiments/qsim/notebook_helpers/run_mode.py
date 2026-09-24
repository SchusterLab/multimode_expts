"""How a qsim migration notebook is being run: for science, or as a test.

The notebooks under `measurement_notebooks/202609_qsim_migration/` are both
the working acquisition entry points and the refactor's test suite. One
notebook body serves both, and this module holds the only difference: a
`RunSettings` read from the environment, which `tools/run_qsim_suite.py` sets
and an interactive kernel normally leaves unset.

Modes (`MULTIMODE_RUN_MODE`):

- ``queue`` (default): the normal science run. Jobs go through the job
  queue, data goes to the notebook's own experiment folder, the notebook's
  config versions are used as written, and measurements are logged.
- ``mock``: mocked instruments. Programs are built and compiled for real,
  acquisition returns zeros. Needs no server and no hardware.
- ``sandbox``: real instruments, but the code under test is this checkout,
  not the main one the queue worker runs. Every runner executes locally, and
  nothing is logged to the vault. Stop the worker first: nothing else stops a
  worker job and a sandbox run from sharing the hardware.

`mock` and `sandbox` both write to a dated ``<yymmdd>_suite_<mode>`` folder,
not the notebook's, and load configs by absolute path from the version
archive, because a worktree's own config database is empty.

Profiles (`MULTIMODE_RUN_PROFILE`): ``full`` (default) or ``smoke``. A
notebook states both values where it sets a size, as
``RUN.pick(1000, smoke=100)``, so the smoke value sits beside the science
value it stands in for.

Configs (`MULTIMODE_RUN_CONFIGS`): unset uses the notebook's `config_dict`.
``main`` uses the current main versions read from the main checkout's job
database -- what the device is calibrated to today, which is usually what a
sandbox run on hardware wants. A path to a JSON file uses the four version
IDs in it.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments.local_env import load_env

MODE_VAR = "MULTIMODE_RUN_MODE"
PROFILE_VAR = "MULTIMODE_RUN_PROFILE"
CONFIGS_VAR = "MULTIMODE_RUN_CONFIGS"
MAIN_DB_VAR = "MULTIMODE_MAIN_JOBS_DB"

MODES = ("queue", "mock", "sandbox")
PROFILES = ("full", "smoke")

DEFAULT_MAIN_DB = "C:/python/multimode_expts/job_server/jobs.db"

# config_dict key -> MultimodeStation keyword and archive kind.
_STATION_KWARG = {
    "hardware_config": "hardware_config",
    "multiphoton_config": "multiphoton_config",
    "man1_storage_swap": "storage_man_file",
    "floquet_storage_swap": "floquet_file",
}


@dataclass(frozen=True)
class RunSettings:
    mode: str = "queue"
    profile: str = "full"
    configs: str | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"{MODE_VAR}={self.mode!r}; expected one of {MODES}")
        if self.profile not in PROFILES:
            raise ValueError(
                f"{PROFILE_VAR}={self.profile!r}; expected one of {PROFILES}"
            )

    @property
    def mock(self) -> bool:
        return self.mode == "mock"

    @property
    def sandbox(self) -> bool:
        return self.mode in ("mock", "sandbox")

    @property
    def smoke(self) -> bool:
        return self.profile == "smoke"

    def pick(self, full, smoke):
        """-> `smoke` under the smoke profile, else `full`."""
        return smoke if self.smoke else full

    def experiment_name(self, name: str) -> str:
        """-> the notebook's own folder for science, a dated suite folder otherwise."""
        if not self.sandbox:
            return name
        return f"{datetime.now():%y%m%d}_suite_{self.mode}"

    def config_dict(self, notebook_config_dict: dict) -> dict:
        """-> the four config version IDs this run uses."""
        if self.configs is None:
            return dict(notebook_config_dict)
        if self.configs == "main":
            return main_config_ids()
        return json.loads(Path(self.configs).read_text())

    def __str__(self):
        configs = self.configs or "notebook"
        return f"mode={self.mode} profile={self.profile} configs={configs}"


def run_settings() -> RunSettings:
    """-> the settings in the environment (or the repo-root .env)."""
    load_env()
    return RunSettings(
        mode=os.environ.get(MODE_VAR, "queue"),
        profile=os.environ.get(PROFILE_VAR, "full"),
        configs=os.environ.get(CONFIGS_VAR) or None,
    )


def station_config_paths(config_dict: dict) -> dict:
    """-> MultimodeStation keywords, each an absolute path to an archived version.

    Absolute paths bypass the station's database lookup, which is what makes a
    worktree work: its own jobs.db is empty. The archive is the pinned copies
    under tests/data/config_set first, then $MULTIMODE_CONFIG_ARCHIVE.
    """
    from experiments.qsim.mbr_campaign import archived

    missing = set(_STATION_KWARG) - set(config_dict)
    if missing:
        raise KeyError(f"config_dict is missing keys: {sorted(missing)}")
    return {
        _STATION_KWARG[key]: str(archived(_STATION_KWARG[key], config_dict[key]))
        for key in _STATION_KWARG
    }


def main_config_ids(db_path: str | Path | None = None) -> dict:
    """-> the current main config version IDs, read from the main job database.

    Read-only: the database belongs to the main checkout's server and worker.
    $MULTIMODE_MAIN_JOBS_DB overrides the measurement PC's default location.
    """
    load_env()
    path = Path(db_path or os.environ.get(MAIN_DB_VAR, DEFAULT_MAIN_DB))
    if not path.is_file():
        raise FileNotFoundError(
            f"main job database not found at {path}; set {MAIN_DB_VAR}"
        )
    query = "SELECT config_type, version_id FROM main_configs"
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
            rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError:
        # SQLite locking does not work over an SMB mount. immutable=1 skips
        # it, at the cost of not seeing writes still in the -wal file.
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(query).fetchall()
    # SQLAlchemy's Enum column stores the member name (HARDWARE_CONFIG).
    by_type = {str(kind).lower(): version for kind, version in rows}
    missing = set(_STATION_KWARG) - set(by_type)
    if missing:
        raise KeyError(f"no main version set for {sorted(missing)} in {path}")
    return {key: by_type[key] for key in _STATION_KWARG}

"""Run settings a qsim migration notebook reads from the environment.

`tools/run_qsim_suite.py` sets these to drive the notebooks from the command
line. An interactive kernel normally leaves them unset, which gives a normal
science run on real hardware. (The notebooks do not run in mock mode; pytest
checks what mock mode can. See docs/qsim/mock_suite_plan.md.)

- `MULTIMODE_RUN_USE_QUEUE`: pass to each runner's `use_queue`. Submit to the
  job server, or run directly on this kernel's station. Unset: 1 in the main
  checkout, 0 in a linked git worktree. The worker runs only the main
  checkout's code, so a queued job from a worktree would not run the
  worktree's code.
- `MULTIMODE_RUN_PROFILE`: ``full`` (default) or ``smoke``. A notebook writes
  both sizes as ``RUN.pick(1000, smoke=100)``.
- `MULTIMODE_RUN_CONFIGS`: unset uses the notebook's `config_dict`. ``main``
  uses the current main versions in the main checkout's job database. A path
  to a JSON file uses the four version IDs in it.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from experiments.local_env import load_env
from experiments.qsim.mbr_campaign import archived

USE_QUEUE_VAR = "MULTIMODE_RUN_USE_QUEUE"
PROFILE_VAR = "MULTIMODE_RUN_PROFILE"
CONFIGS_VAR = "MULTIMODE_RUN_CONFIGS"
MAIN_DB_VAR = "MULTIMODE_MAIN_JOBS_DB"

PROFILES = ("full", "smoke")

DEFAULT_MAIN_DB = "C:/python/multimode_expts/job_server/jobs.db"

# config_dict key -> MultimodeStation keyword.
_STATION_KWARG = {
    "hardware_config": "hardware_config",
    "multiphoton_config": "multiphoton_config",
    "man1_storage_swap": "storage_man_file",
    "floquet_storage_swap": "floquet_file",
}


REPO_ROOT = Path(__file__).resolve().parents[3]


def is_main_checkout(repo_root: Path = REPO_ROOT) -> bool:
    """Is this the main checkout, not a linked worktree?

    In a linked worktree `.git` is a file that points to the main one.
    """
    return (repo_root / ".git").is_dir()


@dataclass(frozen=True)
class RunSettings:
    use_queue: bool = True
    profile: str = "full"
    configs: str | None = None

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(
                f"{PROFILE_VAR}={self.profile!r}; expected one of {PROFILES}"
            )

    @property
    def smoke(self) -> bool:
        return self.profile == "smoke"

    def pick(self, full, smoke):
        """-> `smoke` under the smoke profile, else `full`."""
        return smoke if self.smoke else full

    def station_configs(self, config_dict: dict) -> dict:
        """-> the config keywords for `MultimodeStation`.

        Through the queue: the version IDs, as before. Direct: absolute paths
        from the version archive, because the local job database may not know
        the versions (a worktree, or a machine other than the measurement PC).
        """
        if self.configs == "main":
            config_dict = main_config_ids()
        elif self.configs is not None:
            config_dict = json.loads(Path(self.configs).read_text())
        missing = set(_STATION_KWARG) - set(config_dict)
        if missing:
            raise KeyError(f"config_dict is missing keys: {sorted(missing)}")
        if self.use_queue:
            return {_STATION_KWARG[key]: config_dict[key] for key in _STATION_KWARG}
        return station_config_paths(config_dict)

    def __str__(self):
        return (f"use_queue={self.use_queue} "
                f"profile={self.profile} configs={self.configs or 'notebook'}")


def _flag(var: str, default: bool) -> bool:
    raw = os.environ.get(var)
    if raw is None or raw == "":
        return default
    if raw in ("0", "1"):
        return raw == "1"
    raise ValueError(f"{var}={raw!r}; expected 0 or 1")


def run_settings() -> RunSettings:
    """-> the settings in the environment (or the repo-root .env)."""
    load_env()
    return RunSettings(
        use_queue=_flag(USE_QUEUE_VAR, is_main_checkout()),
        profile=os.environ.get(PROFILE_VAR, "full"),
        configs=os.environ.get(CONFIGS_VAR) or None,
    )


def station_config_paths(config_dict: dict) -> dict:
    """-> MultimodeStation keywords, each an absolute path to an archived version.

    The archive is the pinned copies under tests/data/config_set first, then
    $MULTIMODE_CONFIG_ARCHIVE.
    """
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

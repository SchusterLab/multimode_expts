"""Session setup shared by the qsim migration measurement notebooks.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells 2-6
by the stage-2 notebook split. Every themed measurement notebook needs the same
database handle, job client and station, and the same three blocks of default
settings; before the split they all lived in one notebook and so shared one
copy. Keeping them in one module preserves that -- each notebook does not get
its own drifting copy.

What stays in the notebook is the scientific choice: which config versions to
pin, the experiment name and project, and every per-theme override. What moved
here is the boilerplate around them. Cell 2 built `db`, `config_manager` and
`client` twice in a row, which is also why this is worth having in one place.

Callers: the notebooks under `measurement_notebooks/202609_qsim_migration/`.

This is a temporary home, per the stage-2 instructions -- reconciling it with
`MultimodeStation` and the runners is a later task.
"""

import os
from dataclasses import dataclass
from typing import Any

from experiments import MultimodeStation
from experiments.local_env import load_env
from job_server import JobClient
from job_server.config_versioning import ConfigVersionManager
from job_server.database import get_database

# Cell 2 hard-coded the main checkout. Worktrees share the same config store,
# so this stays pointed at the main path rather than the importing tree -- but
# where that checkout lives is per-machine, so it comes from
# MULTIMODE_CONFIG_DIR in the repo-root .env (see experiments/local_env.py),
# defaulting to the acquisition workstation's path.
load_env()
CONFIG_DIR = os.environ.get("MULTIMODE_CONFIG_DIR", "C:/python/multimode_expts/configs")

# Cell 6, verbatim. Shared by every theme, so a change here is meant to reach
# all of them, exactly as editing cell 6 used to.
MEASUREMENT_CONFIG_DEFAULTS = {
    "avoid_yoko": False,
    "use_multiphoton_swap": False,
}

ACTIVE_RESET_DEFAULTS = {
    "reset_dump_mode": 2,
    "dump_reset_iter_num": 1,
}

FLOQUET_DEFAULTS = {
    # For phase accumulation:
    "include_10cycles_buffer": True,
    "include_10cycles_buffer_in_pi_half": True,
    # flat_top: legacy 3-segment pulse; preload_flattop: one preloaded arb envelope
    "floquet_waveform": "preload_flattop",
    "floquet_hardware_loop": True,
    "scramble_sync_cycles": 1,
    "palindrome_scramble": False,
}


@dataclass
class QsimSession:
    """The handles cell 2 left in the notebook namespace."""

    user: str
    station: Any
    client: JobClient
    db: Any
    config_manager: ConfigVersionManager
    config_dict: dict


def open_session(
    user,
    experiment_name,
    project,
    config_dict,
    log_measurements=True,
    config_dir=CONFIG_DIR,
    verbose=True,
    mock=False,
):
    """Open the database, job client and station for a measurement notebook.

    `config_dict` is required rather than defaulted: which config versions a
    campaign ran against is a scientific choice, so it stays written down in
    the calling notebook. It needs the keys `hardware_config`,
    `man1_storage_swap` and `floquet_storage_swap`; `multiphoton_config` is
    carried through for the record even though the station does not take it.

    `mock=True` builds the station with mocked instruments, per
    `docs/reference/mock_mode_architecture.md`: the qick program build and ASM
    compile run for real, so parameter-validation errors still fire, but
    nothing reaches the FPGA. The runners' `execute()` notices
    `station.is_mock` and dispatches locally instead of through the job queue,
    so a mock session needs no server. The job-server health check is skipped
    for the same reason -- there may not be one running.
    """
    missing = {"hardware_config", "man1_storage_swap", "floquet_storage_swap"} - set(
        config_dict
    )
    if missing:
        raise KeyError(f"config_dict is missing required keys: {sorted(missing)}")

    db = get_database()
    config_manager = ConfigVersionManager(config_dir)
    client = JobClient()

    if verbose and not mock:
        health = client.health_check()
        print(f"Server status: {health['status']}")
        print(f"Pending jobs: {health['pending_jobs']}")
        client.print_queue()
        print(f"Welcome {user}!")
    elif verbose:
        print(f"Welcome {user}! (mock instruments; job queue not used)")

    station = MultimodeStation(
        user=user,
        experiment_name=experiment_name,
        project=project,
        log_measurements=log_measurements,
        mock=mock,
        storage_man_file=config_dict["man1_storage_swap"],
        hardware_config=config_dict["hardware_config"],
        floquet_file=config_dict["floquet_storage_swap"],
    )

    return QsimSession(
        user=user,
        station=station,
        client=client,
        db=db,
        config_manager=config_manager,
        config_dict=config_dict,
    )

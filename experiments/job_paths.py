"""Job ID -> data file resolution, per section 9.1 of docs/qsim_mbr_refactor.md.

Why this exists as its own module
---------------------------------
Finding the HDF5 file for ``JOB-20260815-00009`` is not one problem with one
answer; it is two problems selected by where you are sitting:

* **On the acquisition workstation** the data tree is local, so globbing
  ``{root}/*/data/{JOB-ID}_*.h5`` is instant and resolves uniquely. The
  directory a job landed in is the station's ``experiment_name`` at acquisition
  time, which tracks neither the job date nor the project, so the glob has to
  span projects rather than name one.

* **Anywhere else** the same tree arrives over SMB, where walking it is far too
  slow to do per analysis. There the vault (Obsidian lab notebook, cloud
  synced and therefore fast) is the practical index: ``station.log_measurement``
  writes a full ``data_path`` per run, which is the only place the data
  subdirectory is recorded outside the tree itself.

Both are legitimate and neither is universal, so the choice is deployment
configuration rather than a naming problem to fix upstream.

This module is deliberately a top-level file under ``experiments/``. The
package's flattened exporter skips top-level modules (see the "Skipped
top-level files" line it prints on import), so nothing here can collide in the
``experiments`` namespace the way section 6 warns about.

Failure policy
--------------
Unresolvable roots and missing jobs **raise**, and the error names the
environment variable to set and what it currently points at. This project has
no CI, so every run of the suite is a human on a machine that should have the
data; a silent skip there is not caution, it is a test that never runs.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

# Where the acquisition workstation keeps its data. Used as the default so the
# common case needs no environment at all.
PRODUCTION_DATA_ROOT = Path("C:/experiments")

DATA_ROOT_ENV = "MULTIMODE_DATA_ROOT"
VAULT_ROOT_ENV = "MULTIMODE_VAULT_ROOT"
BACKEND_ENV = "MULTIMODE_PATH_BACKEND"
VAULT_USER_ENV = "MULTIMODE_VAULT_USER"

JOB_ID_RE = re.compile(r"JOB-\d{8}-\d{5}")


class JobPathError(RuntimeError):
    """Raised when a job's data file cannot be located.

    A distinct type so callers (and tests) can tell "your environment is not
    set up" apart from "this analysis is wrong".
    """


def data_root() -> Path:
    """Root of the experiment data tree, or raise telling the caller what to set."""
    raw = os.environ.get(DATA_ROOT_ENV)
    root = Path(raw) if raw else PRODUCTION_DATA_ROOT
    if not root.is_dir():
        source = f"${DATA_ROOT_ENV}={raw!r}" if raw else f"default {PRODUCTION_DATA_ROOT}"
        raise JobPathError(
            f"Experiment data root does not exist: {root} (from {source}).\n"
            f"Set {DATA_ROOT_ENV} to the mounted data tree, e.g.\n"
            f"  {DATA_ROOT_ENV}=/Volumes/experiments"
        )
    return root


def vault_root() -> Path:
    """Root of the Obsidian lab-notebook vault, or raise."""
    raw = os.environ.get(VAULT_ROOT_ENV)
    if not raw:
        raise JobPathError(
            f"The 'vault' backend needs {VAULT_ROOT_ENV} set to the synced vault "
            f"root (the directory containing 'Lab/')."
        )
    root = Path(raw)
    if not root.is_dir():
        raise JobPathError(f"Vault root does not exist: {root} (from ${VAULT_ROOT_ENV}).")
    return root


def backend() -> str:
    """Which resolution strategy to use: 'index' (default) or 'vault'."""
    name = os.environ.get(BACKEND_ENV, "index").strip().lower()
    if name not in ("index", "vault"):
        raise JobPathError(
            f"Unknown {BACKEND_ENV}={name!r}; expected 'index' or 'vault'."
        )
    return name


@lru_cache(maxsize=4)
def _index(root: Path) -> dict:
    """job_id -> path, by one walk of ``{root}/*/data``.

    Cached per root because the walk is the expensive part: doing it once and
    reusing it is what makes a multi-job analysis cheap on the workstation.
    lru_cache means a long-lived kernel will not notice files written after the
    first call; call ``clear_cache()`` if that matters.
    """
    found = {}
    for path in root.glob("*/data/JOB-*.h5"):
        match = JOB_ID_RE.match(path.name)
        if match:
            # Collisions would mean the same job wrote into two projects, which
            # should not happen; keep the first and let validation elsewhere
            # catch it rather than silently preferring a later glob order.
            found.setdefault(match.group(0), path)
    return found


@lru_cache(maxsize=4)
def _vault_index(vault: Path, user: str) -> dict:
    """job_id -> recorded data_path, scraped from the vault's daily notes.

    Reads the ``data_path:`` lines that ``station.log_measurement`` writes. The
    recorded path is a Windows path from the acquisition machine, so only its
    tail below ``experiments/`` is portable; it is re-rooted onto the local
    data root by the caller.
    """
    found = {}
    for note in vault.glob(f"Lab/{user}/*/*/*/*.md"):
        for match in re.finditer(r"data_path:\s*(.+\.h5)", note.read_text(errors="replace")):
            recorded = match.group(1).strip()
            job = JOB_ID_RE.search(recorded)
            if job:
                found.setdefault(job.group(0), recorded)
    return found


def clear_cache() -> None:
    """Drop the cached indexes, so a later call sees newly written files."""
    _index.cache_clear()
    _vault_index.cache_clear()


def resolve_job_paths(job_ids) -> dict:
    """Map job IDs to local HDF5 paths.

    Args:
        job_ids: iterable of ``JOB-YYYYMMDD-NNNNN`` strings.

    Returns:
        dict preserving the input order, job_id -> existing ``Path``.

    Raises:
        JobPathError: if the environment is not resolvable, or if any requested
            job is absent. Absent jobs are reported together rather than one
            per call, so a bad ID list is diagnosed in one go.
    """
    ids = list(job_ids)
    root = data_root()
    which = backend()

    if which == "index":
        table = _index(root)
        resolved = {j: table.get(j) for j in ids}
    else:
        user = os.environ.get(VAULT_USER_ENV)
        if not user:
            raise JobPathError(
                f"The 'vault' backend needs {VAULT_USER_ENV} set to the vault's "
                f"lab-notebook user directory name."
            )
        table = _vault_index(vault_root(), user)
        resolved = {}
        for j in ids:
            recorded = table.get(j)
            if recorded is None:
                resolved[j] = None
                continue
            # Re-root the acquisition machine's absolute path onto this machine.
            tail = recorded.replace("\\", "/").split("experiments/", 1)
            resolved[j] = root / tail[1] if len(tail) == 2 else None

    missing = [j for j, p in resolved.items() if p is None or not Path(p).is_file()]
    if missing:
        raise JobPathError(
            f"{len(missing)} of {len(ids)} jobs not found under {root} "
            f"via the {which!r} backend: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
            + f"\nIf this tree is remote and the walk is slow, set "
              f"{BACKEND_ENV}=vault (also needs {VAULT_ROOT_ENV} and {VAULT_USER_ENV})."
        )
    return {j: Path(resolved[j]) for j in ids}


def resolve_job_path(job_id: str) -> Path:
    """Single-job convenience wrapper around :func:`resolve_job_paths`."""
    return resolve_job_paths([job_id])[job_id]

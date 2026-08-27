"""Job ID -> data file resolution, per section 9.1 of docs/qsim_mbr_refactor.md.

Why this exists as its own module
---------------------------------
Finding the HDF5 file for ``JOB-20260815-00009`` needs one fact that the ID
does not carry: the subdirectory it landed in, which is the station's
``experiment_name`` at acquisition time and so tracks neither the job date nor
the project. Two ways to obtain that were tried first, each tied to where you
are sitting:

* **On the acquisition workstation** the tree is local, so globbing
  ``{root}/*/data/{JOB-ID}_*.h5`` is instant and resolves uniquely. It has to
  span projects rather than name one, precisely because the ID does not say
  which project.

* **Anywhere else** the tree arrives over SMB, where that walk is far too slow
  per analysis: it measured about three minutes for 148 project directories.
  The vault (Obsidian lab notebook, cloud synced and therefore fast) looked
  like the index, since ``station.log_measurement`` writes a full ``data_path``
  per run.

Neither is a good default. The vault only knows runs whose acquirer opted into
``station.log_measurements`` (default False), so whole campaigns are absent
from it -- the 2026-08 MBR campaign among them, in the workstation's own vault
copy as much as any synced one. Do not rely on it for data you did not
personally log.

So the default is the third option: read the subdirectory that was already
recorded. It is an immutable fact about the job, and
``tools/export_job_provenance.py`` writes it to a checked-in sidecar, so
resolving is a JSON read plus one ``is_file`` -- 0.18 s for eight jobs over
SMB against the three minutes above. No walk, no vault, no database, and no
environment beyond the data root. The glob stays as the fallback for jobs
acquired since the last export.

Querying the job database for the subdirectory -- which the pre-refactor loader
did -- is ruled out on three counts, not just speed: it dominated load time
(about half of 40 s for eight files), it contends with the job server for a
resource that belongs to submission and execution, and it cannot be published,
since the database will not ship with the paper.

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

import json
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
PROVENANCE_ENV = "MULTIMODE_PROVENANCE"

# The provenance sidecar exported by tools/export_job_provenance.py. It lives
# under tests/data/ because the golden harness was its first consumer; now that
# the runtime resolver reads it too, it wants a home outside tests/ (follow-up,
# not moved here so the golden fixtures keep their paths).
DEFAULT_PROVENANCE = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "job_provenance.json"
)

# Recorded paths are absolute Windows paths from the acquisition workstation.
# Only the tail below this component is portable across machines.
DATA_ROOT_ANCHOR = "experiments"

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
    """Which resolution strategy to use.

    'provenance' (default) reads the recorded subdirectory and falls back to a
    glob for jobs acquired since the last export. 'index' forces the glob;
    'vault' forces the lab-notebook scrape. The default is correct on the
    workstation too -- a recorded subdirectory does not go stale, because a
    job's data file never moves.
    """
    name = os.environ.get(BACKEND_ENV, "provenance").strip().lower()
    if name not in ("provenance", "index", "vault"):
        raise JobPathError(
            f"Unknown {BACKEND_ENV}={name!r}; "
            f"expected 'provenance', 'index' or 'vault'."
        )
    return name


def provenance_path() -> Path:
    """Location of the provenance sidecar, or raise if it is absent."""
    raw = os.environ.get(PROVENANCE_ENV)
    path = Path(raw) if raw else DEFAULT_PROVENANCE
    if not path.is_file():
        source = f"${PROVENANCE_ENV}={raw!r}" if raw else f"default {path}"
        raise JobPathError(
            f"Provenance sidecar not found: {path} (from {source}).\n"
            f"Export it on the acquisition workstation with\n"
            f"  pixi run python tools/export_job_provenance.py --range ... -o {path.name}"
        )
    return path


def _reroot(recorded: str, root: Path):
    """Re-root an acquisition-machine absolute path onto the local data root.

    Returns None if the path does not sit under a ``{DATA_ROOT_ANCHOR}``
    component, which would mean the recorded value is not a data-tree path at
    all and guessing is worse than reporting it missing.
    """
    parts = recorded.replace("\\", "/").split("/")
    try:
        cut = parts.index(DATA_ROOT_ANCHOR)
    except ValueError:
        return None
    tail = parts[cut + 1:]
    return root.joinpath(*tail) if tail else None


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
def _provenance_index(sidecar: Path) -> dict:
    """job_id -> recorded ``data_file_path``, from the exported sidecar.

    Cheap: one JSON read of a file already in the repo. No directory walk, no
    cloud-synced vault, and above all no live database -- section 3.2 of the
    spec is explicit that offline analysis must not depend on it.
    """
    with sidecar.open() as handle:
        records = json.load(handle)
    found = {}
    for job, record in records.items():
        recorded = record.get("data_file_path") if isinstance(record, dict) else None
        if recorded:
            found[job] = recorded
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
    _provenance_index.cache_clear()


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

    if which == "provenance":
        table = _provenance_index(provenance_path())
        resolved = {}
        for j in ids:
            recorded = table.get(j)
            resolved[j] = _reroot(recorded, root) if recorded else None
        # Jobs acquired since the last export are simply absent from the
        # sidecar. Falling back to the glob for exactly those keeps fresh
        # acquisition working on the workstation, and costs nothing when the
        # sidecar already covers everything asked for.
        stragglers = [j for j, path in resolved.items()
                      if path is None or not Path(path).is_file()]
        if stragglers:
            table = _index(root)
            for j in stragglers:
                resolved[j] = table.get(j)
    elif which == "index":
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
            resolved[j] = _reroot(recorded, root)

    missing = [j for j, p in resolved.items() if p is None or not Path(p).is_file()]
    if missing:
        raise JobPathError(
            f"{len(missing)} of {len(ids)} jobs not found under {root} "
            f"via the {which!r} backend: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
            + f"\nIf these are recent acquisitions, re-export the provenance "
              f"sidecar on the workstation:\n"
              f"  pixi run python tools/export_job_provenance.py --range ... "
              f"-o tests/data/job_provenance.json"
        )
    return {j: Path(resolved[j]) for j in ids}


def resolve_job_path(job_id: str) -> Path:
    """Single-job convenience wrapper around :func:`resolve_job_paths`."""
    return resolve_job_paths([job_id])[job_id]

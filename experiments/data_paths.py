"""Where the data files are: the data roots of this machine, and paths below them.

Every data file has a path below a data root that is the same on every machine:

    <experiment>/data/<JOB-ID>_<Class>.h5        one job (the worker writes it)
    <experiment>/assembled_data/<Class>_<name>.*  an assembled data set (manifest + HDF5)
    <experiment>/derived_data/...                 files made by an analysis

``<experiment>`` is the station's ``experiment_name``. Only the roots differ per
machine. Set them in the repo-root ``.env`` (``experiments/local_env.py``):

    MULTIMODE_DATA_ROOTS=/Volumes/slab/Multimode/BF5_expts_autosync:/Volumes/pippin/experiments

Separate roots with ``os.pathsep`` (``;`` on Windows, ``:`` elsewhere). Unset, the
root is ``C:/experiments`` (the measurement PC). A path is looked up in each root in
turn and the first hit is used, so the order matters only for speed: put the root
with most of the data first. Roots that are not mounted are skipped; the error
for a file that is in no root says which roots are not mounted.

Why more than one root: old experiment folders move from pippin's ``C:\\experiments``
to the slabnas copy (``BF5_expts_autosync``), while new data lands on pippin.

The functions:

- ``locate(path)``: -> the file or folder on this machine. ``path`` is relative to a
  data root (``"261027AGPlowKerr/assembled_data/x.yaml"``), or an absolute path
  recorded on another machine (``C:\\experiments\\...``), or a local absolute path.
- ``relative(path)``: -> the part of a path below its data root.
- ``resolve_job_paths(job_ids)``: job IDs -> files. The experiment folder of a job
  is read from the provenance sidecar (``tools/export_job_provenance.py``); jobs
  not in it are found by one walk of ``*/data`` per root (slow over SMB).

Missing files raise ``DataPathError`` (a ``FileNotFoundError``) with the roots that
were tried. Do not write other loaders that know about mounts or machines: give
them a path below the data root and call ``locate``.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

from experiments.local_env import load_env

load_env()

DATA_ROOTS_ENV = "MULTIMODE_DATA_ROOTS"
PROVENANCE_ENV = "MULTIMODE_PROVENANCE"
DEFAULT_DATA_ROOTS = (Path("C:/experiments"),)

# The provenance sidecar exported by tools/export_job_provenance.py.
DEFAULT_PROVENANCE = Path(__file__).resolve().parent.parent / "tests" / "data" / "job_provenance.json"

# Names of data-root folders on the machines we know. A recorded absolute path is
# cut after the first of these (or after the name of a configured root).
ROOT_NAMES = ("experiments", "BF5_expts_autosync")

JOB_ID_RE = re.compile(r"JOB-\d{8}-\d{5}")


class DataPathError(FileNotFoundError):
    """A data file is not on this machine, or the data roots are not set up."""


def data_roots():
    """-> the configured data roots, in lookup order (mounted or not)."""
    raw = os.environ.get(DATA_ROOTS_ENV)
    if raw is None and os.environ.get("MULTIMODE_DATA_ROOT"):
        raise DataPathError("MULTIMODE_DATA_ROOT is now MULTIMODE_DATA_ROOTS (one or more roots, "
                            f"separated by {os.pathsep!r}); rename it in .env")
    if not raw:
        return list(DEFAULT_DATA_ROOTS)
    return [Path(os.path.expanduser(part.strip())) for part in raw.split(os.pathsep) if part.strip()]


def mounted_roots():
    """-> the data roots that exist on this machine now."""
    return [root for root in data_roots() if root.is_dir()]


def _is_absolute(text):
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def relative(path):
    """-> the part of ``path`` below its data root (a PurePosixPath), or None.

    A relative path is already below the root. An absolute one is cut after a
    configured root, else after the first folder named like a data root
    (``ROOT_NAMES``): ``C:\\experiments\\x\\data\\f.h5`` -> ``x/data/f.h5``.
    """
    text = str(path).replace("\\", "/")
    if not text:
        return None
    if not _is_absolute(text):
        return PurePosixPath(text)
    for root in data_roots():
        prefix = str(root).replace("\\", "/").rstrip("/") + "/"
        if text.startswith(prefix):
            return PurePosixPath(text[len(prefix):])
    names = set(ROOT_NAMES) | {root.name for root in data_roots()}
    parts = text.split("/")
    for index, part in enumerate(parts):
        if part in names and index + 1 < len(parts):
            return PurePosixPath(*parts[index + 1:])
    return None


@lru_cache(maxsize=None)
def _has_folder(root, experiment):
    # One check per (root, experiment folder), so a root without that folder costs
    # one round trip per kernel, not one per file. clear_cache() forgets.
    return (root / experiment).is_dir()


def _candidates(tail):
    for root in mounted_roots():
        if _has_folder(root, tail.parts[0]):
            yield root / tail


def locate(path):
    """-> ``path`` on this machine (see the module docstring). Raises DataPathError."""
    if _is_absolute(str(path)) and Path(path).exists():
        return Path(path)
    tail = relative(path)
    if tail is None:
        raise DataPathError(f"{path} is not below a data root ({', '.join(ROOT_NAMES)})")
    for candidate in _candidates(tail):
        if candidate.exists():
            return candidate
    raise DataPathError(f"{tail} is in none of the data roots: {_describe_roots()}")


def _describe_roots():
    roots = data_roots()
    text = ", ".join(f"{root}{'' if root.is_dir() else ' (not mounted)'}" for root in roots)
    return f"{text}. Set {DATA_ROOTS_ENV} in .env (experiments/data_paths.py)."


def provenance_path():
    """-> the provenance sidecar, or raise if it is absent."""
    raw = os.environ.get(PROVENANCE_ENV)
    path = Path(raw) if raw else DEFAULT_PROVENANCE
    if not path.is_file():
        raise DataPathError(
            f"Provenance sidecar not found: {path}. Export it on the measurement PC with\n"
            f"  pixi run python tools/export_job_provenance.py --range ... -o {path.name}")
    return path


@lru_cache(maxsize=4)
def _provenance_index(sidecar):
    """job_id -> recorded ``data_file_path``, from the exported sidecar."""
    with sidecar.open() as handle:
        records = json.load(handle)
    return {job: record["data_file_path"] for job, record in records.items()
            if isinstance(record, dict) and record.get("data_file_path")}


@lru_cache(maxsize=None)
def _job_index(root):
    """job_id -> path, by one walk of ``root/*/data``. Slow over SMB; cached per root."""
    found = {}
    for path in root.glob("*/data/JOB-*.h5"):
        match = JOB_ID_RE.match(path.name)
        if match:
            found.setdefault(match.group(0), path)
    return found


def clear_cache():
    """Forget the cached folder checks and job indexes (to see files written since)."""
    _has_folder.cache_clear()
    _provenance_index.cache_clear()
    _job_index.cache_clear()


def resolve_job_paths(job_ids):
    """-> {job_id: Path} in input order. Raises DataPathError naming all missing jobs."""
    ids = list(job_ids)
    try:
        recorded = _provenance_index(provenance_path())
    except DataPathError:
        recorded = {}
    resolved = {}
    for job in ids:
        try:
            resolved[job] = locate(recorded[job]) if job in recorded else None
        except DataPathError:
            resolved[job] = None
    # Jobs acquired since the last sidecar export: walk the roots, in order.
    for root in mounted_roots():
        stragglers = [job for job in ids if resolved[job] is None]
        if not stragglers:
            break
        index = _job_index(root)
        for job in stragglers:
            resolved[job] = index.get(job)
    missing = [job for job in ids if resolved[job] is None]
    if missing:
        raise DataPathError(
            f"{len(missing)} of {len(ids)} jobs not found: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "") + f"\nRoots: {_describe_roots()}")
    return resolved


def resolve_job_path(job_id):
    """-> the file of one job (``resolve_job_paths``)."""
    return resolve_job_paths([job_id])[job_id]

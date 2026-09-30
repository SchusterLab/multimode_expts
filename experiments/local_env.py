"""Per-machine settings from a gitignored ``.env`` at the repo root.

Why a file and not just the shell
---------------------------------
The path-resolving modules here (``job_paths``, ``floquet_timing``) each read a
``MULTIMODE_*`` variable to find data that lives somewhere different on every
machine: ``C:/experiments`` on the acquisition workstation, an SMB mount on a
laptop. Leaving that to the shell means every Jupyter kernel, pytest run and
editor-launched process has to inherit an export that only one interactive
shell had -- which they routinely do not, and the symptom is a JobPathError
pointing at a default nobody set.

One file at the repo root fixes that for every entry point at once, because it
is found relative to this source tree rather than to whoever started the
process. It is gitignored, so each checkout keeps its own.

Precedence and format
---------------------
The real environment always wins: a variable already set in ``os.environ`` is
left alone, so ``MULTIMODE_BACKEND=index pixi run pytest`` and pytest's
``monkeypatch.setenv`` still override the file. The file only fills gaps.

Format is the common subset of the many ``.env`` dialects -- ``KEY=value``, one
per line, ``#`` comments, blank lines, an optional ``export`` prefix, and
optional surrounding quotes. No interpolation, no multi-line values: this
parser is deliberately small so the repo needs no dependency for it, and
anything fancier belongs in the shell.

``.env.example`` is checked in and lists the variables with the values that are
right for the acquisition workstation; copy it to ``.env`` and edit.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Escape hatch for a file elsewhere (a shared one on prod, say). Read from the
# real environment only -- a .env cannot redirect which .env is read.
ENV_FILE_VAR = "MULTIMODE_ENV_FILE"

DEFAULT_ENV_FILE = REPO_ROOT / ".env"

_loaded = False


def env_file() -> Path:
    """Which file ``load_env`` reads. May not exist; absence is not an error."""
    raw = os.environ.get(ENV_FILE_VAR)
    return Path(raw) if raw else DEFAULT_ENV_FILE


def parse_env_file(path: Path) -> dict:
    """Parse ``path`` into a dict. Malformed lines are skipped, not raised on."""
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(force: bool = False) -> dict:
    """Fill unset variables from the ``.env`` file. Idempotent.

    Args:
        force: re-read the file even if it was already loaded this process,
            for an interactive kernel that just edited it. Still does not
            overwrite variables that are set.

    Returns:
        The variables this call actually set (empty if the file is absent or
        everything in it was already set).
    """
    global _loaded
    if _loaded and not force:
        return {}
    _loaded = True

    path = env_file()
    if not path.is_file():
        return {}

    applied = {}
    for key, value in parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied

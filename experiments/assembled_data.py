# -*- coding: utf-8 -*-
"""Save and read what an assembled Experiment writes: a manifest and an HDF5.

An assembled Experiment holds many job Experiments and never goes to the
worker (docs/qsim/mbr_redesign.md, section 5). Its ``save()`` writes two
files to ``<experiment root>/assembled_data/``:

- a **manifest** YAML: the class, the job IDs, the raw job HDF5 paths, the
  calibration manifest it used, and notes. This is the source of truth.
  ``from_manifest`` re-assembles from the raw job files it lists.
- an **assembled HDF5**: the assembled arrays, with the manifest path and the
  code version as attributes. It is a copy that can be rebuilt from the
  manifest; never edit it.

Raw file paths are stored relative to the manifest's directory when they are
under the same experiment root, so a manifest still resolves when the data is
mounted somewhere else.

Infrastructure, not physics: nothing here knows what is being measured.
"""
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from experiments.characterization_runner import json_plain

REPO_ROOT = Path(__file__).resolve().parents[1]

ASSEMBLED_DIR = "assembled_data"
CONVERTED_DIR = "converted_data"


def experiment_root(job_file):
    """-> ``C:\\experiments\\<experiment_name>`` for a job file in its data dir."""
    return Path(job_file).resolve().parent.parent


def code_version():
    """-> the git commit of this checkout, with ``+dirty`` if it has changes."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


def new_stem(class_name):
    """-> a file stem that sorts by time, e.g. ``260924_153012_MBRCalibrationSetExperiment``."""
    return f"{datetime.now():%y%m%d_%H%M%S}_{class_name}"


def _relative(path, base):
    """-> ``path`` relative to ``base`` if they share a drive, else absolute."""
    path = Path(path).resolve()
    try:
        return Path(os.path.relpath(path, Path(base).resolve())).as_posix()
    except ValueError:  # different drives on Windows
        return str(path)


def write_manifest(path, manifest, raw_files):
    """Write the manifest YAML. ``raw_files`` are stored relative to its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dict(manifest)
    content["raw_files"] = [_relative(f, path.parent) for f in raw_files]
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(json_plain(content), handle, sort_keys=False,
                       default_flow_style=None, width=100)
    return path


def read_manifest(path):
    """-> the manifest as a dict, with ``raw_files`` resolved to absolute paths."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    manifest["raw_files"] = [
        f if Path(f).is_absolute() else (path.parent / f).resolve()
        for f in manifest.get("raw_files", [])
    ]
    return manifest


def write_assembled_h5(path, arrays, attrs):
    """Write the assembled arrays and their provenance attributes.

    Attribute values that are not plain strings or numbers are stored as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in arrays.items():
            handle.create_dataset(key, data=np.asarray(value))
        for key, value in attrs.items():
            if not isinstance(value, (str, int, float)):
                value = json.dumps(json_plain(value))
            handle.attrs[key] = value
    return path


def read_assembled_h5(path):
    """-> (arrays, attrs) from an assembled HDF5."""
    with h5py.File(path, "r") as handle:
        arrays = {key: handle[key][()] for key in handle}
        attrs = dict(handle.attrs)
    return arrays, attrs

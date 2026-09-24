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


def new_stem(class_name, directory):
    """-> a file stem that sorts by time, e.g. ``260924_153012_MBRCalibrationSetExperiment``.

    A second save in the same second in ``directory`` gets ``_2``, ``_3``, ...
    so it never overwrites the first.
    """
    stem = f"{datetime.now():%y%m%d_%H%M%S}_{class_name}"
    candidate, count = stem, 1
    while any((Path(directory) / f"{candidate}{suffix}").exists() for suffix in (".yaml", ".h5")):
        count += 1
        candidate = f"{stem}_{count}"
    return candidate


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
    calibration = manifest.get("calibration_manifest")
    if calibration and not Path(calibration).is_absolute():
        manifest["calibration_manifest"] = (path.parent / calibration).resolve()
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


class AssembledExperiment:
    """Base of the assembled Experiments: holds job Experiments of one class.

    Not a slab ``Experiment``: it never goes to the worker and needs no
    soccfg, config file or instruments. Same method names as one, though:
    ``acquire``, ``analyze``, ``display``, ``save``, ``from_manifest``
    (docs/qsim/mbr_redesign.md, section 5).

    A subclass sets ``child_class`` and implements

    - ``job_overrides()``: one runner override dict per job;
    - ``from_children(children, job_ids, notes, **kwargs)``: assemble jobs;
    - ``analyze(...)`` and ``display(...)``;
    - ``manifest_parameters()``, ``assembled_arrays()``, ``assembled_attrs()``:
      what ``save()`` writes.
    """

    child_class = None

    def __init__(self, notes=""):
        self.notes = notes
        self.children = []
        self.job_ids = []
        self.data = {}
        self.manifest_path = None

    def job_overrides(self):
        raise NotImplementedError

    def acquire(self, runner, batch_size=10, **execute_kwargs):
        """Run one job per ``job_overrides()`` entry through ``runner``.

        ``execute_kwargs`` go to ``runner.execute`` (``use_queue``, ``log``,
        ``show``, ...). Returns the job Experiments.
        """
        if runner.ExptClass is not self.child_class:
            raise TypeError(
                f"runner.ExptClass is {runner.ExptClass.__name__}; "
                f"{type(self).__name__} needs {self.child_class.__name__}")
        children = runner.execute(overrides=self.job_overrides(),
                                  batch_size=batch_size, **execute_kwargs)
        self.children = list(children)
        self.job_ids = list(runner.last_job_ids)
        self.data = {}
        return self.children

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", **kwargs):
        raise NotImplementedError

    def _check_children(self):
        if not self.children:
            raise ValueError(f"{type(self).__name__} has no jobs; acquire or load them first")
        for child in self.children:
            if not isinstance(child, self.child_class):
                raise TypeError(f"{child!r} is not a {self.child_class.__name__}")

    # -- persistence ------------------------------------------------------

    def calibration_manifest(self):
        """-> the path of the calibration manifest this set used, or None."""
        return None

    def manifest_parameters(self):
        return {}

    def assembled_arrays(self):
        raise NotImplementedError

    def assembled_attrs(self):
        return {}

    def _child_files(self):
        """-> the files ``save()`` lists as ``raw_files``: the job HDF5s."""
        return [Path(child.fname) for child in self.children]

    @classmethod
    def _load_children(cls, manifest, timing=None):
        from experiments.saved_jobs import load_experiment

        return [load_experiment(cls.child_class, raw, timing=timing)
                for raw in manifest["raw_files"]]

    @classmethod
    def from_manifest(cls, path, timing=None):
        """Re-assemble from the raw job files a saved manifest lists.

        ``timing`` goes to :func:`experiments.saved_jobs.load_experiment` for
        job files that carry no Floquet timing of their own.
        """
        manifest = read_manifest(path)
        if manifest["class"] != cls.__name__:
            raise ValueError(f"{path} is a {manifest['class']} manifest")
        assembled = cls.from_children(
            cls._load_children(manifest, timing=timing),
            job_ids=manifest["job_ids"], notes=manifest.get("notes", ""),
            **cls._from_manifest_kwargs(manifest, Path(path)))
        assembled.manifest_path = Path(path)
        return assembled

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        """-> extra ``from_children`` keyword arguments read from a manifest."""
        return {}

    def save(self, directory=None, notes=None):
        """Write the manifest YAML and the assembled HDF5. -> the manifest path.

        ``directory`` defaults to ``assembled_data/`` beside the first job's
        data directory. Needs ``analyze()`` to have run.
        """
        if not self.data:
            raise ValueError("run analyze() before save()")
        if notes is not None:
            self.notes = notes
        raw_files = self._child_files()
        if directory is None:
            directory = experiment_root(raw_files[0]) / ASSEMBLED_DIR
        stem = new_stem(type(self).__name__, directory)
        manifest_path = Path(directory) / f"{stem}.yaml"
        h5_path = Path(directory) / f"{stem}.h5"
        version = code_version()
        calibration = self.calibration_manifest()
        if calibration is not None:
            calibration = _relative(calibration, manifest_path.parent)

        write_manifest(manifest_path, {
            "class": type(self).__name__,
            "module": type(self).__module__,
            "child_class": self.child_class.__name__,
            "created": datetime.now().isoformat(timespec="seconds"),
            "code_version": version,
            "job_ids": list(self.job_ids),
            "calibration_manifest": calibration,
            "assembled_h5": h5_path.name,
            "parameters": self.manifest_parameters(),
            "notes": self.notes,
        }, raw_files)
        write_assembled_h5(h5_path, self.assembled_arrays(), {
            "class": type(self).__name__,
            "manifest": manifest_path.name,
            "code_version": version,
            **self.assembled_attrs(),
        })
        self.manifest_path = manifest_path
        return manifest_path

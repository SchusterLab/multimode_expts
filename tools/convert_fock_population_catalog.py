# -*- coding: utf-8 -*-
"""Convert the data sets of ``configs/datasets/fock_population_datasets.yaml`` to the new layout.

A one-off driver (docs/qsim/mbr_redesign.md, section 6) for the old population
jobs (``QsimExperiment`` + ``FockPopulationProgram``) of the quantum walks and
AGP scans of 2026-10-17..19. Raw files are only read; their SHA-256 is checked
before and after.

- Each calibration (four jobs, Fock 0..3 in M1, read M1) becomes one
  ``MultiParityConfusionMatrixExperiment`` file.
- Each population job becomes one ``FockPopulationExperiment`` file, analyzed
  with its data set's calibration: shots plus raw and corrected moments. Its
  raw probabilities are checked against the independent classifier of
  ``fitting/qsim/agp.py`` (``load_population_job``).
- Both go to ``<experiment root>/converted_data/`` with ``converted_from``
  (old job IDs, files, this tool, the code version) and the old jobs'
  ``derived_params`` and ``config_versions``.
- A walk becomes one ``FockPopulationSetExperiment``. An AGP scan becomes one
  ``CentralBosonAGPExperiment`` per W and one ``CentralBosonAGPDisorderScan``.
  Their manifests and small HDF5s go to ``<experiment root>/assembled_data/``,
  written by the classes' own ``save()``.

An existing converted file stops the run, unless ``--reuse`` loads it as it is.
At the end the tool prints each data set's top manifest, for the catalog's
``manifest`` field (the catalog is hand-commented YAML; the tool does not rewrite it).

    pixi run python tools/convert_fock_population_catalog.py --check     # resolve the files only
    pixi run python tools/convert_fock_population_catalog.py             # every data set
    pixi run python tools/convert_fock_population_catalog.py walk_n3_20261017 --reuse
"""
import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import yaml
from slab import AttrDict
from slab.experiment import NpEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import assembled_data  # noqa: E402
from experiments.job_paths import resolve_job_paths  # noqa: E402
from experiments.qsim.agp import (  # noqa: E402
    CentralBosonAGPDisorderScan,
    CentralBosonAGPExperiment,
    rms_khz,
)
from experiments.qsim.fock_population import (  # noqa: E402
    FockPopulationExperiment,
    FockPopulationSetExperiment,
)
from experiments.qsim.multiparity_confusion_matrix import (  # noqa: E402
    MultiParityConfusionMatrixExperiment,
)
from fitting.qsim.agp import load_population_job  # noqa: E402

CATALOG = REPO_ROOT / "configs" / "datasets" / "fock_population_datasets.yaml"
TOOL = "tools/convert_fock_population_catalog.py"
POINT_KEYS = ("xpts", "avgi", "avgq", "amps", "phases", "idata", "qdata")


def read_old_job(path):
    """-> (cfg, data, attrs) of an old job file; data has one shot row per cycle."""
    with h5py.File(path, "r") as handle:
        cfg = json.loads(handle.attrs["config"])
        data = {key: np.asarray(handle[key][()]) for key in POINT_KEYS if key in handle}
        attrs = {key: handle.attrs[key] for key in ("derived_params", "config_versions")
                 if key in handle.attrs}
    if "expt" not in cfg and isinstance(cfg.get("config"), dict):
        cfg = cfg["config"]
    cfg = AttrDict(cfg)
    if int(cfg.get("read_num", 0)) < 2:
        raise ValueError(f"{path}: the config must record read_num")
    data["xpts"] = data["xpts"].reshape(-1)
    for key in ("idata", "qdata"):
        if data[key].ndim == 1:
            data[key] = data[key][None, :]
        if data[key].shape[0] != len(data["xpts"]):
            raise ValueError(f"{path}: {key} must have one shot row per cycle")
    for key in ("avgi", "avgq", "amps", "phases"):
        if key in data:
            data[key] = data[key].reshape(-1)
    return cfg, data, attrs


def readout_settings(cfg):
    """-> what must agree between a calibration and the jobs it corrects."""
    expt = cfg.expt
    qubit = int(expt.get("qubits", [0])[0])
    threshold = float(np.asarray(cfg.device.readout.threshold).reshape(-1)[qubit])
    return dict(read_num=int(cfg.read_num), threshold=threshold,
                readout=expt.get("readout"), cond_sec_phase=expt.get("cond_sec_phase"),
                phase_second_pulse=expt.get("phase_second_pulse"),
                active_reset=expt.get("active_reset"),
                pre_selection_reset=bool(expt.get("pre_selection_reset")))


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_job(path, job, attrs):
    """Write one converted job in the layout ``Experiment.save_data`` produces; never overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as handle:
        handle.attrs["config"] = json.dumps(job.cfg, cls=NpEncoder)
        for key, value in attrs.items():
            handle.attrs[key] = value if isinstance(value, str) else json.dumps(value, cls=NpEncoder)
        for key, value in job.data.items():
            handle.create_dataset(key, data=np.asarray(value))
    job.fname = str(path)


def converted_from(job_ids, paths):
    return dict(job_ids=list(job_ids), files=[str(path) for path in paths], tool=TOOL,
                code_version=assembled_data.code_version())


def old_attrs(attrs, job_class, extra):
    """-> the converted file's attributes: class, provenance and the old timing and versions."""
    result = dict(job_class=job_class.__name__, job_module=job_class.__module__, **extra)
    result.update({key: value for key, value in attrs.items()})
    return result


class Converter:
    """Converts one data set at a time; remembers the calibrations it converted."""

    def __init__(self, catalog, reuse=False, check=False):
        self.catalog = catalog
        self.reuse = reuse
        self.check = check
        self.calibrations = {}
        self.hashes = {}

    def paths(self, job_ids):
        paths = resolve_job_paths(job_ids)
        missing = [job_id for job_id in job_ids if job_id not in paths]
        if missing:
            raise FileNotFoundError(f"no file for {missing}")
        for path in paths.values():
            self.hashes.setdefault(str(path), sha256(path))
        return [Path(paths[job_id]) for job_id in job_ids]

    def converted_dir(self, path):
        return assembled_data.experiment_root(path) / assembled_data.CONVERTED_DIR

    def existing(self, path, loader):
        if not path.exists():
            return None
        if not self.reuse:
            raise FileExistsError(f"{path} exists; pass --reuse to load it as it is")
        return loader(path)

    # -- calibration -------------------------------------------------------

    def calibration(self, name):
        """-> (calibration file path, readout settings) of a catalog calibration."""
        if name in self.calibrations:
            return self.calibrations[name]
        job_ids = self.catalog["calibrations"][name]["job_ids"]
        paths = self.paths(job_ids)
        out = self.converted_dir(paths[0]) / (
            f"converted_{'_'.join(job_ids)}_{MultiParityConfusionMatrixExperiment.__name__}.h5")
        jobs = [read_old_job(path) for path in paths]
        settings = readout_settings(jobs[0][0])
        if self.check:
            self.calibrations[name] = (out, settings)
            return self.calibrations[name]
        if self.existing(out, MultiParityConfusionMatrixExperiment.from_h5file) is None:
            prepared = [int(cfg.expt.occupations[0]) for cfg, _, _ in jobs]
            if sorted(prepared) != [0, 1, 2, 3]:
                raise ValueError(f"{name}: needs one job each with 0, 1, 2, 3 photons in M1")
            for (cfg, data, _), path in zip(jobs, paths):
                if (not np.array_equal(data["xpts"], [0]) or any(cfg.expt.occupations[1:])
                        or cfg.expt.ro_stor != 0):
                    raise ValueError(f"{path}: a calibration job prepares M1 only, at zero cycles, read M1")
                if readout_settings(cfg) != settings:
                    raise ValueError(f"{path}: different readout settings in one calibration")
            order = np.argsort(prepared)
            calibration = MultiParityConfusionMatrixExperiment.__new__(
                MultiParityConfusionMatrixExperiment)
            cfg = deepcopy(jobs[0][0])
            cfg.expt.pop("occupations", None)
            cfg.expt.swept_params = ["init_man_fock_state"]
            cfg.expt.init_man_fock_states = ["0", "1", "2", "3"]
            cfg.expt.multiparity_confusion_matrix = None
            calibration.cfg = cfg
            calibration.data = {key: np.concatenate([np.atleast_1d(jobs[i][1][key]) for i in order])
                                for key in POINT_KEYS if key != "xpts"}
            calibration.data["xpts"] = np.arange(4)
            calibration.data["prepared_occupations"] = np.asarray(
                [jobs[i][0].expt.occupations for i in order])
            calibration.analyze()
            write_job(out, calibration, old_attrs(
                jobs[0][2], MultiParityConfusionMatrixExperiment,
                dict(converted_from=converted_from([job_ids[i] for i in order],
                                                   [paths[i] for i in order]))))
            print(f"  calibration {name}: C diagonal "
                  f"{np.round(np.diag(calibration.data['multiparity_confusion_matrix']), 3)}")
        self.calibrations[name] = (out, settings)
        return self.calibrations[name]

    # -- population jobs ---------------------------------------------------

    def population_jobs(self, job_ids, calibration_name):
        """-> the converted FockPopulationExperiments of ``job_ids``, loaded without shots."""
        calibration_path, settings = self.calibration(calibration_name)
        paths = self.paths(job_ids)
        if self.check:
            for path in paths:
                cfg, _, _ = read_old_job(path)
                if readout_settings(cfg) != settings:
                    raise ValueError(f"{path}: readout settings differ from {calibration_name}")
            return paths
        with h5py.File(calibration_path, "r") as handle:
            matrix = np.asarray(handle["multiparity_confusion_matrix"][()])
        record = dict(matrix=matrix.tolist(), fname=str(calibration_path),
                      job_id=",".join(self.catalog["calibrations"][calibration_name]["job_ids"]))
        jobs = []
        for job_id, path in zip(job_ids, paths):
            out = self.converted_dir(path) / f"converted_{job_id}_{FockPopulationExperiment.__name__}.h5"
            loaded = self.existing(out, lambda p: FockPopulationExperiment.from_h5file(p, load_shots=False))
            if loaded is None:
                cfg, data, attrs = read_old_job(path)
                if readout_settings(cfg) != settings:
                    raise ValueError(f"{path}: readout settings differ from {calibration_name}")
                job = FockPopulationExperiment.__new__(FockPopulationExperiment)
                job.cfg = cfg
                job.cfg.expt.swept_params = ["floquet_cycle"]
                job.cfg.expt.floquet_cycles = data["xpts"].astype(int).tolist()
                job.cfg.expt.multiparity_confusion_matrix = deepcopy(record)
                job.data = dict(data)
                job.analyze()
                self.check_against_fitting(path, job, matrix)
                write_job(out, job, old_attrs(attrs, FockPopulationExperiment,
                                              dict(converted_from=converted_from([job_id], [path]))))
                loaded = FockPopulationExperiment.from_h5file(out, load_shots=False)
            jobs.append(loaded)
        return jobs

    @staticmethod
    def check_against_fitting(path, job, matrix):
        """Raw probabilities must match fitting/qsim/agp.py's classifier; corrected ones C^-T p."""
        reference = load_population_job(path)
        raw = np.stack([job.data[f"p_mod{n}"] for n in range(4)], axis=-1)
        np.testing.assert_allclose(raw, reference["probabilities"], rtol=0, atol=1e-12,
                                   err_msg=f"{path}: raw probabilities")
        corrected = np.stack([job.data[f"p_mod{n}_corrected"] for n in range(4)], axis=-1)
        np.testing.assert_allclose(corrected, np.linalg.solve(matrix.T, raw.T).T, rtol=0,
                                   atol=1e-12, err_msg=f"{path}: corrected probabilities")

    # -- data sets ---------------------------------------------------------

    def walk(self, name, entry):
        jobs = self.population_jobs(entry["job_ids"], entry["calibration"])
        if self.check:
            return None
        walk = FockPopulationSetExperiment.from_children(jobs, job_ids=entry["job_ids"],
                                                         notes=f"{name}: {entry['label']}")
        walk.analyze()
        return walk.save(directory=self.assembled_dir(jobs[0]))

    def agp_scan(self, name, entry):
        parts = []
        for strength, part in sorted(entry["parts"].items()):
            jobs = self.population_jobs(part["job_ids"], part["calibration"])
            if self.check:
                continue
            agp = CentralBosonAGPExperiment.from_children(
                jobs, job_ids=part["job_ids"], direction=entry["direction"],
                disorder_khz=strength, notes=f"{name}, W = {strength} kHz")
            if not np.isclose(rms_khz(agp.detunings), strength, rtol=1e-3, atol=1e-6):
                raise ValueError(f"{name}: W = {strength} kHz, but the detuning RMS is "
                                 f"{rms_khz(agp.detunings):.3f} kHz")
            agp.analyze()
            agp.save(directory=self.assembled_dir(jobs[0]))
            parts.append(agp)
        if self.check:
            return None
        scan = CentralBosonAGPDisorderScan.from_children(parts, notes=f"{name}: {entry['label']}")
        scan.analyze()
        return scan.save(directory=self.assembled_dir(parts[0].children[0]))

    @staticmethod
    def assembled_dir(job):
        return assembled_data.experiment_root(job.fname) / assembled_data.ASSEMBLED_DIR

    def run(self, names):
        manifests = {}
        for name in names:
            entry = self.catalog["datasets"][name]
            print(f"{name} ({entry['kind']})")
            manifests[name] = getattr(self, entry["kind"])(name, entry)
        changed = [path for path, digest in self.hashes.items() if sha256(path) != digest]
        if changed:
            raise RuntimeError(f"raw files changed: {changed}")
        print(f"{len(self.hashes)} raw files read; none changed")
        return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", help="catalog names (default: all)")
    parser.add_argument("--check", action="store_true", help="resolve the files and settings only")
    parser.add_argument("--reuse", action="store_true", help="load existing converted files as they are")
    args = parser.parse_args(argv)
    with open(CATALOG, encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    names = args.datasets or list(catalog["datasets"])
    manifests = Converter(catalog, reuse=args.reuse, check=args.check).run(names)
    if not args.check:
        root = assembled_data.experiment_root
        for name, path in manifests.items():
            print(f"{name}:\n  manifest: {Path(root(path).name) / path.parent.name / path.name}")


if __name__ == "__main__":
    main()

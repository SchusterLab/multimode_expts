# -*- coding: utf-8 -*-
"""Convert old MBR job HDF5s to the new job layout, one dataset at a time.

docs/qsim/mbr_redesign.md, section 6. The new job classes read only the new
layout; old data reaches them only through this script. Raw files are never
changed: converted job files go to ``<experiment root>/converted_data/``, and
the assembled output -- written by the new assembled class's own ``save()``
-- to ``<experiment root>/assembled_data/``.

The input is a hand-written list of the old job IDs of **one dataset**, as a
YAML (or JSON) file::

    dataset: september_N3          # a label, for the manifest notes
    kind: stark_cal                # which conversion
    notes: N=3 sector, preload_flattop swaps, 65 cycle pairs
    job_ids: [JOB-20260905-00075, JOB-20260905-00076, ...]

Do not guess the grouping of jobs into datasets. If no list exists for a
dataset, build it from the lab's OneNote logs.

Kinds
-----
``stark_cal``
    Old ``EntireFloquetCyclePhaseCalibrationProgram`` jobs, two per
    occupation (analyzer phase 0 and 90) -> one ``MBRStarkCalExperiment``
    file per occupation, then one ``MBRCalibrationSetExperiment`` manifest and
    assembled HDF5. Example: 70 files (35 occupations x 2) -> 35 + 1 + 1.

Each converted file records the old job IDs and file paths it came from
(``converted_from``) and its new job class (``job_class``), and carries the
Floquet timing as a ``derived_params`` attribute, so it loads without the
provenance sidecar.

Usage::

    pixi run python tools/migrate_mbr_jobs.py JOBS.yaml [--out-root DIR] [--no-shots]
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slab import AttrDict  # noqa: E402
from slab.experiment import NpEncoder  # noqa: E402

from experiments import assembled_data  # noqa: E402
from experiments.job_paths import job_records, resolve_job_paths  # noqa: E402
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment  # noqa: E402
from experiments.qsim.mbr_stark_cal import (  # noqa: E402
    RAMSEY_PHASES,
    MBRStarkCalExperiment,
)
from experiments.saved_jobs import load_experiment, load_job  # noqa: E402

TOOL = "tools/migrate_mbr_jobs.py"

# Point-by-point arrays of the old 2D sweep, shape (n_cycle_pairs, 2 prep phases).
POINT_KEYS = ("avgi", "avgq", "amps", "phases")
# Per-shot arrays, one row per point in sweep order (cycle-major, prep inner).
SHOT_KEYS = ("idata", "qdata")
# Config keys that differ between the two old jobs of one occupation.
ANALYZER_KEY = "spectroscopy_analyzer_phase"


def read_job_list(path):
    """-> the job-list file as a dict (YAML or JSON)."""
    with open(path, encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not content.get("job_ids"):
        raise ValueError(f"{path} lists no job_ids")
    return content


def write_job_file(path, cfg, data, attrs):
    """Write one job file in the layout ``Experiment.save_data`` produces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["config"] = json.dumps(cfg, cls=NpEncoder)
        for key, value in attrs.items():
            handle.attrs[key] = value if isinstance(value, str) else json.dumps(value, cls=NpEncoder)
        for key, value in data.items():
            handle.create_dataset(key, data=np.asarray(value))
    return path


def _config_versions(job):
    """-> the config version IDs of an old job: its own attribute, else the sidecar."""
    raw = job.data.get("attrs", {}).get("config_versions")
    if raw is not None:
        return json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    record = job_records(required=False).get(job.job_id, {})
    names = {"hardware_config_version_id": "hardware_config",
             "multiphoton_config_version_id": "multiphoton_config",
             "man1_storage_version_id": "man1_storage_swap",
             "floquet_storage_version_id": "floquet_storage_swap"}
    return {new: record[old] for old, new in names.items() if record.get(old)}


def _derived_params(job, converted_job_class):
    """-> the ``derived_params`` attribute for a converted file."""
    prog = job.prog
    cycle_us = float(prog.calculate_floquet_cycle_us())
    pi_fracs = [int(frac) for frac in prog.m1s_pi_fracs]
    return dict(floquet_cycle_us=cycle_us,
                m1s_pi_fracs=pi_fracs,
                couplings_MHz=[1. / (4. * frac * cycle_us) for frac in pi_fracs],
                source=f"converted to {converted_job_class} from {prog.source}")


# --------------------------------------------------------------------------
# stark_cal
# --------------------------------------------------------------------------


def convert_stark_cal_pair(phi0, phi90):
    """-> (cfg, data, attrs) of one MBRStarkCalExperiment file.

    ``phi0`` and ``phi90`` are the two old jobs of one occupation (loaded
    with :func:`experiments.saved_jobs.load_job`). The new inner sweep is
    :data:`RAMSEY_PHASES`: [prep, analyzer] = [0,0], [180,0], [0,90], [180,90],
    i.e. phi0's two columns, then phi90's.
    """
    for job, phi in ((phi0, 0.), (phi90, 90.)):
        ecfg = job.cfg.expt
        if float(ecfg[ANALYZER_KEY]) != phi:
            raise ValueError(f"{job.job_id}: analyzer phase {ecfg[ANALYZER_KEY]}, expected {phi}")
        if ecfg.get("phase_unwrap_mode", "pair") != "pair":
            raise ValueError(f"{job.job_id}: phase_unwrap_mode "
                             f"{ecfg.phase_unwrap_mode!r} has no new-layout equivalent")
        if not np.allclose(job.data["xpts"], [0., 180.]):
            raise ValueError(f"{job.job_id}: preparation phases {job.data['xpts']}")
    cfg0, cfg90 = deepcopy(phi0.cfg), deepcopy(phi90.cfg)
    # Old configs also carry a stray top-level copy of the swept keys.
    for cfg in (cfg0, cfg90):
        cfg.expt.pop(ANALYZER_KEY)
        cfg.pop(ANALYZER_KEY, None)
    if json.dumps(cfg0, cls=NpEncoder, sort_keys=True) != json.dumps(cfg90, cls=NpEncoder, sort_keys=True):
        raise ValueError(f"{phi0.job_id} and {phi90.job_id} differ in more than the analyzer phase")
    if not np.array_equal(phi0.data["ypts"], phi90.data["ypts"]):
        raise ValueError(f"{phi0.job_id} and {phi90.job_id} sweep different cycle pairs")

    cfg = AttrDict(cfg0)
    ecfg = cfg.expt
    for key in ("spectroscopy_prep_phases", "spectroscopy_prep_phase", "phase_unwrap_mode"):
        ecfg.pop(key, None)
    ecfg.ramsey_phases = deepcopy(RAMSEY_PHASES)
    ecfg.swept_params = ["n_cycle_pair", "ramsey_phase"]
    ecfg.QickProgramName = "MBRStarkCalProgram"

    n_pairs = len(phi0.data["ypts"])
    data = dict(xpts=np.asarray(RAMSEY_PHASES), ypts=np.asarray(phi0.data["ypts"]))
    for key in POINT_KEYS:
        data[key] = np.concatenate([np.asarray(phi0.data[key]).reshape(n_pairs, 2),
                                    np.asarray(phi90.data[key]).reshape(n_pairs, 2)], axis=1)
    if all(key in phi0.data and key in phi90.data for key in SHOT_KEYS):
        for key in SHOT_KEYS:
            old0 = np.asarray(phi0.data[key]).reshape(n_pairs, 2, -1)
            old90 = np.asarray(phi90.data[key]).reshape(n_pairs, 2, -1)
            data[key] = np.concatenate([old0, old90], axis=1).reshape(4 * n_pairs, -1)

    attrs = dict(
        job_class=MBRStarkCalExperiment.__name__,
        converted_from=dict(job_ids=[phi0.job_id, phi90.job_id],
                            files=[str(phi0.fname), str(phi90.fname)],
                            tool=TOOL, code_version=assembled_data.code_version()),
        derived_params=_derived_params(phi0, MBRStarkCalExperiment.__name__),
    )
    versions = _config_versions(phi0)
    if versions:
        attrs["config_versions"] = versions
    return cfg, data, attrs


def pair_stark_cal_jobs(jobs):
    """-> [(phi0, phi90)] in the order occupations first appear in ``jobs``."""
    grouped = {}
    for job in jobs:
        occupation = tuple(int(n) for n in job.cfg.expt.spectroscopy_occupations)
        phi = float(job.cfg.expt[ANALYZER_KEY])
        if phi not in (0., 90.):
            raise ValueError(f"{job.job_id}: analyzer phase {phi}; expected 0 or 90")
        slot = grouped.setdefault(occupation, {0.: [], 90.: []})
        slot[phi].append(job)
    pairs = []
    for occupation, slot in grouped.items():
        if len(slot[0.]) != 1 or len(slot[90.]) != 1:
            raise ValueError(
                f"{occupation}: {len(slot[0.])} phi=0 and {len(slot[90.])} phi=90 jobs; "
                f"expected one of each (repeated jobs are not converted)")
        pairs.append((slot[0.][0], slot[90.][0]))
    return pairs


def migrate_stark_cal(job_ids, out_root=None, load_shots=True, notes="", timing=None):
    """Convert one calibration dataset. -> the saved MBRCalibrationSetExperiment.

    ``out_root`` defaults to the experiment root of the first old job file.
    """
    paths = resolve_job_paths(list(job_ids))
    jobs = [load_job(job_id, path=paths[job_id], timing=timing, load_shots=load_shots,
                     provenance=job_records(required=False))
            for job_id in job_ids]
    out_root = Path(out_root) if out_root else assembled_data.experiment_root(paths[job_ids[0]])
    converted_dir = out_root / assembled_data.CONVERTED_DIR

    children, pair_ids = [], []
    for phi0, phi90 in pair_stark_cal_jobs(jobs):
        cfg, data, attrs = convert_stark_cal_pair(phi0, phi90)
        path = converted_dir / (f"converted_{phi0.job_id}_{phi90.job_id}_"
                                f"{MBRStarkCalExperiment.__name__}.h5")
        # Run the job's own analysis once, so the file holds what a new job
        # would: the raw sweep plus complex_return and its fit.
        child = MBRStarkCalExperiment.__new__(MBRStarkCalExperiment)
        child.cfg, child.data, child.fname = cfg, AttrDict(data), str(path)
        child.analyze()
        write_job_file(path, cfg, dict(child.data), attrs)
        children.append(load_experiment(MBRStarkCalExperiment, path))
        pair_ids.append(f"{phi0.job_id}+{phi90.job_id}")

    # A converted job has no queue ID of its own; name the old pair it came from.
    calibration = MBRCalibrationSetExperiment.from_children(
        children, job_ids=pair_ids, notes=notes)
    calibration.analyze()
    calibration.save(directory=out_root / assembled_data.ASSEMBLED_DIR)
    return calibration


MIGRATIONS = {"stark_cal": migrate_stark_cal}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("job_list", help="YAML/JSON file with kind, job_ids, notes")
    parser.add_argument("--out-root", help="experiment root to write under "
                        "(default: the old files' experiment root)")
    parser.add_argument("--no-shots", action="store_true",
                        help="do not copy the per-shot idata/qdata arrays")
    args = parser.parse_args(argv)

    job_list = read_job_list(args.job_list)
    kind = job_list.get("kind")
    if kind not in MIGRATIONS:
        parser.error(f"kind must be one of {sorted(MIGRATIONS)}, got {kind!r}")
    notes = " ".join(str(part) for part in (job_list.get("dataset"), job_list.get("notes")) if part)
    result = MIGRATIONS[kind](job_list["job_ids"], out_root=args.out_root,
                              load_shots=not args.no_shots, notes=notes)
    print(f"{len(result.children)} converted job files; manifest {result.manifest_path}")


if __name__ == "__main__":
    main()

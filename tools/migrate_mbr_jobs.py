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
    calibration_job_ids: [...]     # kind spectrum only, optional

Do not guess the grouping of jobs into datasets. If no list exists for a
dataset, build it from the lab's OneNote logs.

Kinds
-----
``stark_cal``
    Old ``EntireFloquetCyclePhaseCalibrationProgram`` jobs, two per
    occupation (analyzer phase 0 and 90) -> one ``MBRStarkCalExperiment``
    file per occupation, then one ``MBRCalibrationSetExperiment`` manifest and
    assembled HDF5. Example: 70 files (35 occupations x 2) -> 35 + 1 + 1.
``spectrum``
    Old diagonal ``NPhotonHamiltonianSpectroscopyProgram`` jobs: per
    occupation, analyzer phase 0 and 90, each possibly in time chunks -> one
    ``MBRTimeTraceExperiment`` file per occupation, then one
    ``MBRSpectrumExperiment`` manifest and assembled HDF5. The optional
    ``calibration_job_ids`` are converted first, as ``stark_cal``, and become
    the spectrum's calibration set.

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
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment  # noqa: E402
from experiments.qsim.mbr_stark_cal import (  # noqa: E402
    RAMSEY_PHASES,
    MBRStarkCalExperiment,
)
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment  # noqa: E402
from experiments.saved_jobs import load_experiment, load_job  # noqa: E402

TOOL = "tools/migrate_mbr_jobs.py"

# Point-by-point arrays of the old 2D sweep, shape (n_cycles, 2 prep phases).
POINT_KEYS = ("avgi", "avgq", "amps", "phases")
# Per-shot arrays, one row per point in sweep order (cycle-major, prep inner).
SHOT_KEYS = ("idata", "qdata")
# The config key that differs between the two analyzer-phase jobs of a trace.
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
# Shared: merge the analyzer-phase jobs (and time chunks) of one trace
# --------------------------------------------------------------------------


def _strip(cfg, keys):
    """-> a copy of ``cfg`` without ``keys`` in cfg.expt or at the top level.

    Old configs also carry a stray top-level copy of the swept keys.
    """
    cfg = deepcopy(cfg)
    for key in keys:
        cfg.expt.pop(key, None)
        cfg.pop(key, None)
    return cfg


def group_phase_jobs(jobs, state_of):
    """-> {state: {0.: [jobs], 90.: [jobs]}}, states in first-seen order."""
    grouped = {}
    for job in jobs:
        phi = float(job.cfg.expt[ANALYZER_KEY])
        if phi not in (0., 90.):
            raise ValueError(f"{job.job_id}: analyzer phase {phi}; expected 0 or 90")
        grouped.setdefault(state_of(job), {0.: [], 90.: []})[phi].append(job)
    return grouped


def _joined(phase_jobs, key, shots):
    """-> (sorted cycles, rows of ``key``) from the time chunks of one phase."""
    cycles = np.concatenate([np.asarray(job.data["ypts"]) for job in phase_jobs])
    order = np.argsort(cycles, kind="stable")
    shape = (2, -1) if shots else (2,)
    rows = np.concatenate([np.asarray(job.data[key]).reshape(len(job.data["ypts"]), *shape)
                           for job in phase_jobs])
    return cycles[order], rows[order]


def merge_phase_jobs(phi0_jobs, phi90_jobs, cycle_key, swept_cycle):
    """-> (cfg, data) of one new-layout trace from its old jobs.

    ``phi0_jobs``/``phi90_jobs`` are the old jobs at analyzer phase 0 and 90;
    several per phase are time chunks of one trace. Each job's ``ypts`` are
    its cycles and ``cfg.expt[cycle_key]`` their config list. The chunks are
    joined and sorted by cycle; both phases must cover the same cycles. The
    new inner sweep is :data:`RAMSEY_PHASES`: [prep, analyzer] = [0,0],
    [180,0], [0,90], [180,90], i.e. the phi=0 columns, then phi=90's.
    """
    jobs = list(phi0_jobs) + list(phi90_jobs)
    if not phi0_jobs or not phi90_jobs:
        raise ValueError(f"{[job.job_id for job in jobs]}: need both analyzer phases")
    for job in jobs:
        ecfg = job.cfg.expt
        if ecfg.get("phase_unwrap_mode", "pair") != "pair":
            raise ValueError(f"{job.job_id}: phase_unwrap_mode "
                             f"{ecfg.phase_unwrap_mode!r} has no new-layout equivalent")
        if not np.allclose(job.data["xpts"], [0., 180.]):
            raise ValueError(f"{job.job_id}: preparation phases {job.data['xpts']}")
        if not np.array_equal(job.data["ypts"], ecfg[cycle_key]):
            raise ValueError(f"{job.job_id}: saved cycles do not match its config")
    reference = json.dumps(_strip(jobs[0].cfg, (ANALYZER_KEY, cycle_key)),
                           cls=NpEncoder, sort_keys=True)
    for job in jobs[1:]:
        if json.dumps(_strip(job.cfg, (ANALYZER_KEY, cycle_key)),
                      cls=NpEncoder, sort_keys=True) != reference:
            raise ValueError(f"{jobs[0].job_id} and {job.job_id} differ in more than "
                             f"the analyzer phase and the cycles")

    cycles, _ = _joined(phi0_jobs, "avgi", False)
    if len(np.unique(cycles)) != len(cycles):
        raise ValueError(f"{[job.job_id for job in phi0_jobs]}: cycles overlap")
    if not np.array_equal(cycles, _joined(phi90_jobs, "avgi", False)[0]):
        raise ValueError(f"{[job.job_id for job in jobs]}: the two analyzer phases "
                         f"cover different cycles")

    data = dict(xpts=np.asarray(RAMSEY_PHASES), ypts=cycles)
    for key in POINT_KEYS:
        data[key] = np.concatenate([_joined(phi0_jobs, key, False)[1],
                                    _joined(phi90_jobs, key, False)[1]], axis=1)
    if all(key in job.data for job in jobs for key in SHOT_KEYS):
        for key in SHOT_KEYS:
            both = np.concatenate([_joined(phi0_jobs, key, True)[1],
                                   _joined(phi90_jobs, key, True)[1]], axis=1)
            data[key] = both.reshape(4 * len(cycles), -1)

    cfg = AttrDict(_strip(phi0_jobs[0].cfg, (ANALYZER_KEY,)))
    ecfg = cfg.expt
    for key in ("spectroscopy_prep_phases", "spectroscopy_prep_phase", "phase_unwrap_mode"):
        ecfg.pop(key, None)
    ecfg[cycle_key] = [int(n) for n in cycles]
    ecfg.ramsey_phases = deepcopy(RAMSEY_PHASES)
    ecfg.swept_params = [swept_cycle, "ramsey_phase"]
    return cfg, data


def write_converted(job_class, program_name, jobs, cfg, data, path):
    """Analyze, write and reload one converted job file. -> the loaded job.

    The job's own ``analyze`` runs once, so the file holds what a new job
    would: the raw sweep plus ``complex_return``.
    """
    cfg.expt.QickProgramName = program_name
    attrs = dict(
        job_class=job_class.__name__,
        converted_from=dict(job_ids=[job.job_id for job in jobs],
                            files=[str(job.fname) for job in jobs],
                            tool=TOOL, code_version=assembled_data.code_version()),
        derived_params=_derived_params(jobs[0], job_class.__name__),
    )
    versions = _config_versions(jobs[0])
    if versions:
        attrs["config_versions"] = versions
    child = job_class.__new__(job_class)
    child.cfg, child.data, child.fname = cfg, AttrDict(data), str(path)
    child.analyze()
    write_job_file(path, cfg, dict(child.data), attrs)
    return load_experiment(job_class, path)


def _load_old_jobs(job_ids, timing, load_shots):
    paths = resolve_job_paths(list(job_ids))
    records = job_records(required=False)
    return paths, [load_job(job_id, path=paths[job_id], timing=timing,
                            load_shots=load_shots, provenance=records)
                   for job_id in job_ids]


# --------------------------------------------------------------------------
# stark_cal
# --------------------------------------------------------------------------


def migrate_stark_cal(job_ids, out_root=None, load_shots=True, notes="", timing=None):
    """Convert one calibration dataset. -> the saved MBRCalibrationSetExperiment.

    ``out_root`` defaults to the experiment root of the first old job file.
    """
    paths, jobs = _load_old_jobs(job_ids, timing, load_shots)
    out_root = Path(out_root) if out_root else assembled_data.experiment_root(paths[job_ids[0]])
    converted_dir = out_root / assembled_data.CONVERTED_DIR

    children, sources = [], []
    grouped = group_phase_jobs(
        jobs, lambda job: tuple(int(n) for n in job.cfg.expt.spectroscopy_occupations))
    for occupation, slot in grouped.items():
        if len(slot[0.]) != 1 or len(slot[90.]) != 1:
            raise ValueError(
                f"{occupation}: {len(slot[0.])} phi=0 and {len(slot[90.])} phi=90 jobs; "
                f"expected one of each (repeated jobs are not converted)")
        phi0, phi90 = slot[0.][0], slot[90.][0]
        cfg, data = merge_phase_jobs([phi0], [phi90], "n_cycle_pairs", "n_cycle_pair")
        path = converted_dir / (f"converted_{phi0.job_id}_{phi90.job_id}_"
                                f"{MBRStarkCalExperiment.__name__}.h5")
        children.append(write_converted(MBRStarkCalExperiment, "MBRStarkCalProgram",
                                        [phi0, phi90], cfg, data, path))
        sources.append(f"{phi0.job_id}+{phi90.job_id}")

    # A converted job has no queue ID of its own; name the old jobs it came from.
    calibration = MBRCalibrationSetExperiment.from_children(
        children, job_ids=sources, notes=notes)
    calibration.analyze()
    calibration.save(directory=out_root / assembled_data.ASSEMBLED_DIR)
    return calibration


# --------------------------------------------------------------------------
# spectrum
# --------------------------------------------------------------------------


def migrate_spectrum(job_ids, out_root=None, load_shots=True, notes="", timing=None,
                     calibration_job_ids=None):
    """Convert one diagonal spectroscopy dataset. -> the saved MBRSpectrumExperiment.

    Old off-diagonal pair jobs (``offdiag_cycles``) are not converted yet;
    they belong to the disorder datasets (later phase).
    """
    paths, jobs = _load_old_jobs(job_ids, timing, load_shots)
    out_root = Path(out_root) if out_root else assembled_data.experiment_root(paths[job_ids[0]])
    converted_dir = out_root / assembled_data.CONVERTED_DIR
    for job in jobs:
        if "offdiag_cycles" in job.cfg.expt:
            raise NotImplementedError(
                f"{job.job_id} is an old off-diagonal pair job; their conversion "
                f"comes with the disorder datasets (later phase)")
        if "floquet_cycles" not in job.cfg.expt:
            raise ValueError(f"{job.job_id} is not a spectroscopy job")
    calibration = None
    if calibration_job_ids:
        calibration = migrate_stark_cal(calibration_job_ids, out_root=out_root,
                                        load_shots=load_shots, notes=notes, timing=timing)

    def state_of(job):
        ecfg = job.cfg.expt
        initial = tuple(int(n) for n in ecfg.spectroscopy_occupations)
        final = tuple(int(n) for n in ecfg.get("spectroscopy_final_occupations", initial))
        return initial, final

    children, sources = [], []
    for (initial, final), slot in group_phase_jobs(jobs, state_of).items():
        if initial != final:
            raise ValueError(f"{initial} -> {final}: a spectrum takes diagonal traces")
        cfg, data = merge_phase_jobs(slot[0.], slot[90.], "floquet_cycles", "floquet_cycle")
        trace_jobs = slot[0.] + slot[90.]
        path = converted_dir / (f"converted_{trace_jobs[0].job_id}_x{len(trace_jobs)}_"
                                f"{MBRTimeTraceExperiment.__name__}.h5")
        children.append(write_converted(MBRTimeTraceExperiment, "MBRTimeTraceProgram",
                                        trace_jobs, cfg, data, path))
        sources.append("+".join(job.job_id for job in trace_jobs))

    spectrum = MBRSpectrumExperiment.from_children(
        children, job_ids=sources, notes=notes, calibration=calibration)
    spectrum.analyze()
    spectrum.save(directory=out_root / assembled_data.ASSEMBLED_DIR)
    return spectrum


MIGRATIONS = {"stark_cal": migrate_stark_cal, "spectrum": migrate_spectrum}


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
    extra = {}
    if kind == "spectrum" and job_list.get("calibration_job_ids"):
        extra["calibration_job_ids"] = job_list["calibration_job_ids"]
    result = MIGRATIONS[kind](job_list["job_ids"], out_root=args.out_root,
                              load_shots=not args.no_shots, notes=notes, **extra)
    print(f"{len(result.children)} converted job files; manifest {result.manifest_path}")


if __name__ == "__main__":
    main()

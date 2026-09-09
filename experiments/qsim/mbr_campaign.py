# -*- coding: utf-8 -*-
"""Acquire a many-body-Ramsey campaign: one config, four stages, either target.

This is the acquisition half of the post-refactor caller. It exists so the
exemplar scripts stay thin, and so the same code can run in two places:

* on the measurement PC, submitting through the job queue, and
* off-prod in mock mode, where the qick library still builds every program and
  compiles the ASM but no bytes reach an FPGA.

The second mode is what makes the MBR programs testable at all. The whole
suite can be green while a Floquet swap plays the wrong envelope, because
nothing in it builds a program from a real dataset row -- that is exactly how
the preload_flattop envelope bug survived. A mock acquisition catches it.

Three things about acquisition that are easy to get wrong
--------------------------------------------------------

**Drive stages through the Experiment, never the Program.** A stage config
carries a plural key (``cycle_decoder_analyzers``) and the program body reads
the singular one (``cycle_decoder_analyzer``). The expansion in between lives
in ``DarkBaseExperiment.acquire``, so instantiating a Program directly fails
with a bare ``AttributeError``.

**Pass a matched config set.** The four configs are versioned independently
and the working files in ``configs/`` drift from what any campaign actually
used. A mismatch usually surfaces as a pointed ``KeyError`` naming the missing
pulse, but it can also be quietly wrong -- the swap dataset decides the Floquet
envelope per mode, so the wrong CSV changes the physics rather than raising.
Never assemble a set by taking the newest of each.

Two sets are committed under ``tests/data/config_set/`` (48 KB), so this module
runs on a fresh checkout with no mount and no environment variables:
:func:`pinned_config_set` resolves them by repo-relative path. To reproduce a
recorded campaign instead, :func:`config_set_for_job` reads the four version
IDs out of job provenance; anything not pinned is then fetched from the archive
on the measurement PC via ``$MULTIMODE_CONFIG_ARCHIVE``.

**The queue is optional.** ``BatchRunner.execute`` requires a ``job_client``,
so off-prod :func:`run_stage` falls back to acquiring in-process. A local job
server plus ``worker --mock`` is the way to exercise the queue itself; that is
a separate concern from whether the programs are right.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from slab import AttrDict

from experiments.floquet_timing import config_archive
from experiments.qsim import floquet_dark_mode_readout as fdmr
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_propagator import MBRPropagatorExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = REPO_ROOT / "tests" / "data" / "job_provenance.json"

# Config versions committed to the repo, so acquisition can be exercised on a
# fresh checkout with no mount and no environment. 48 KB for two whole sets.
# The archive on the measurement PC stays the source of truth; this is a
# pinned copy of the few versions the tests and exemplars name.
PINNED_DIR = REPO_ROOT / "tests" / "data" / "config_set"
PINNED_SETS_FILE = PINNED_DIR / "SETS.json"

# Archive layout, and the station keyword each config feeds.
_ARCHIVE = {
    "hardware_config": ("hardware_config", "yml"),
    "multiphoton_config": ("multiphoton_config", "yml"),
    "storage_man_file": ("man1_storage_swap", "csv"),
    "floquet_file": ("floquet_storage_swap", "csv"),
}
# Provenance field naming each version, keyed the same way.
_PROVENANCE_FIELD = {
    "hardware_config": "hardware_config_version_id",
    "multiphoton_config": "multiphoton_config_version_id",
    "storage_man_file": "man1_storage_version_id",
    "floquet_file": "floquet_storage_version_id",
}


class ConfigSetError(RuntimeError):
    """Raised when a requested config version is not in the archive."""


def archived(kind: str, version_id: str) -> Path:
    """-> absolute path to one archived config version.

    Absolute on purpose. ``MultimodeStation`` joins a config name onto its own
    ``configs/`` directory, and joining an absolute path discards that prefix,
    so this reaches the archive without copying anything into the repo.
    """
    if kind not in _ARCHIVE:
        raise KeyError(f"unknown config kind {kind!r}; expected {sorted(_ARCHIVE)}")
    subdir, suffix = _ARCHIVE[kind]
    name = f"{version_id}.{suffix}"

    # Pinned copy first, so a fresh checkout needs neither the mount nor
    # $MULTIMODE_CONFIG_ARCHIVE. Same file either way -- these are immutable
    # version snapshots, so there is no stale-copy risk to weigh.
    pinned = PINNED_DIR / name
    if pinned.is_file():
        return pinned

    try:
        path = config_archive() / subdir / name
    except Exception as error:  # archive unreachable, and not pinned
        raise ConfigSetError(
            f"{kind} {version_id} is not pinned under {PINNED_DIR.name}/ and "
            f"the version archive is unreachable: {error}"
        ) from error
    if not path.is_file():
        raise ConfigSetError(
            f"no {kind} {version_id}: not pinned under {PINNED_DIR.name}/, and "
            f"absent from the archive at {path}. To pin it, copy the file from "
            f"the measurement PC into {PINNED_DIR}."
        )
    return path


def config_set(**version_ids) -> dict:
    """-> station kwargs for an explicit set of version IDs.

    >>> config_set(hardware_config="CFG-HW-20260907-00001", ...)   # doctest: +SKIP
    """
    unknown = set(version_ids) - set(_ARCHIVE)
    if unknown:
        raise KeyError(f"unknown config kinds {sorted(unknown)}")
    return {kind: str(archived(kind, vid)) for kind, vid in version_ids.items()}


def job_provenance(path: Path | str = PROVENANCE) -> dict:
    """-> the recorded provenance sidecar, job_id -> record."""
    return json.loads(Path(path).read_text())


def config_set_for_job(job_id: str, provenance: dict | None = None) -> dict:
    """-> station kwargs for the exact configs one recorded job ran with.

    The reliable way to reproduce a campaign: the four version IDs are written
    down per job, so this cannot drift the way "newest of each" does.
    """
    provenance = provenance if provenance is not None else job_provenance()
    if job_id not in provenance:
        raise KeyError(f"{job_id} is not in the provenance sidecar")
    record = provenance[job_id]
    resolved = {}
    for kind, field in _PROVENANCE_FIELD.items():
        version_id = record.get(field)
        if version_id:
            resolved[kind] = str(archived(kind, version_id))
    return resolved


def pinned_sets() -> dict:
    """-> {set_name: {kind: version_id}} for the config sets kept in the repo."""
    return json.loads(PINNED_SETS_FILE.read_text())


def pinned_config_set(name: str = "preload_current") -> dict:
    """-> station kwargs for one committed config set, by name.

    ``august_n3`` is the set ``JOB-20260815-00009`` ran with, taken from job
    provenance, so it reproduces the golden-baseline campaign. Its seven modes
    are all ``gauss``.

    ``preload_current`` is a snapshot of the live calibration taken on
    2026-09-08, where every mode is ``preload_flattop``. It is *not* a
    reproduced campaign -- no recorded job in the provenance sidecar used it,
    because preload_flattop postdates the export -- so treat it as "does
    today's calibration still build", not as ground truth for saved data. It
    is the default here precisely because it is the set that exercises the
    preloaded register-bank path.
    """
    sets = pinned_sets()
    if name not in sets:
        raise KeyError(f"unknown pinned set {name!r}; have {sorted(sets)}")
    return {kind: str(archived(kind, vid)) for kind, vid in sets[name].items()}


def latest_config_set() -> dict:
    """-> station kwargs for the newest version of each config in the archive.

    Convenience for "does today's calibration still build a program". Do not
    use it to reproduce saved data: the four are versioned independently, so
    the newest of each is not necessarily a set that ever ran together.
    """
    resolved = {}
    for kind, (subdir, suffix) in _ARCHIVE.items():
        candidates = sorted((config_archive() / subdir).glob(f"*.{suffix}"))
        if not candidates:
            raise ConfigSetError(f"archive has no {kind} under {subdir}/")
        resolved[kind] = str(candidates[-1])
    return resolved


def mock_station(user: str = "guan", **station_kwargs):
    """-> a MultimodeStation wired to mocks, with a matched config set.

    Defaults to :func:`pinned_config_set`, which needs nothing outside the
    repo. Pass explicit config kwargs -- from :func:`config_set_for_job` or
    :func:`pinned_config_set` -- to reproduce a particular campaign.
    """
    from experiments.station import MultimodeStation

    resolved = {k: v for k, v in station_kwargs.items() if k in _ARCHIVE}
    if not resolved:
        resolved = pinned_config_set()
    rest = {k: v for k, v in station_kwargs.items() if k not in _ARCHIVE}
    return MultimodeStation(mock=True, user=user, **resolved, **rest)


# --------------------------------------------------------------------------
# The one default config. Stage builders override from here; nothing else
# should define these keys, so a campaign has a single place to read.
# --------------------------------------------------------------------------

def mbr_defaults(swap_stors, **overrides) -> AttrDict:
    """-> the shared expt config for every MBR stage.

    Deliberately does not set ``floquet_waveform``: the envelope is a property
    of the swap calibration, read per mode from the dataset row. Setting it
    here would have hidden the preload_flattop bug rather than exposing it.
    """
    swap_stors = [int(stor) for stor in swap_stors]
    defaults = AttrDict(dict(
        expts=1, reps=1000, rounds=1, qubits=[0],
        normalize=False,
        active_reset=True, man_reset=True, storage_reset=swap_stors,
        pre_relax_delay=100, relax_delay=200,
        reset_dump_mode=2, dump_reset_iter_num=1, use_qubit_man_reset=False,
        prepulse=False, postpulse=False, init_fock=False,
        perform_wigner=False, parity_readout=False, multiparity_readout=False,
        load_man_dark=False, swap_man_dark=False, swap_man_large_dark=False,
        update_phases=True,
        floquet_cycle=0,
        palindrome_scramble=False,
        scramble_sync_cycles=1,
        floquet_hardware_loop=True,
        swap_stors=swap_stors,
        detunings=[0.] * len(swap_stors),
        spectroscopy_prep_phases=[0., 180.],
        include_10cycles_buffer=True,
        include_10cycles_buffer_in_pi_half=True,
        avoid_yoko=False,
        use_multiphoton_swap=False,
    ))
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------
# Stages. Each entry says which Experiment owns the analysis, which Program
# acquires, and how to turn campaign parameters into a batch.
# --------------------------------------------------------------------------

def _calibration_batch(defaults, swap_stors, occupations, cycle_pairs=(0, 1, 2),
                       sync_cycles=1, reps=1500, **kwargs):
    batch = MBRPhaseCorrectionExperiment.calibration_batch(
        defaults, swap_stors, occupations, np.asarray(cycle_pairs),
        sync_cycles=sync_cycles, reps=reps, **kwargs)
    return batch


def _spectrum_batch(defaults, swap_stors, occupations, cycles=(0, 4),
                    phase_by_occupation=None, sync_cycles=1, reps=1000,
                    **kwargs):
    phase_by_occupation = phase_by_occupation or {
        tuple(o): 0.0 for o in occupations}
    return MBRSpectrumExperiment.spectroscopy_batch(
        defaults, swap_stors, occupations, [np.asarray(cycles)],
        phase_by_occupation=phase_by_occupation,
        sync_cycles=sync_cycles, reps=reps, **kwargs)


def _propagator_batch(defaults, swap_stors, occupations, cycles=(0, 4),
                      phase_by_occupation=None, sync_cycles=1, reps=1000,
                      **kwargs):
    phase_by_occupation = phase_by_occupation or {
        tuple(o): 0.0 for o in occupations}
    return MBRPropagatorExperiment.propagator_batch(
        defaults, swap_stors, occupations, list(cycles),
        phase_by_occupation=phase_by_occupation,
        sync_cycles=sync_cycles, reps=reps, **kwargs)


def _orthogonality_batch(defaults, swap_stors, occupations, sync_cycles=1,
                         reps=1000, **kwargs):
    return MBROrthogonalityExperiment.orthogonality_batch(
        defaults, swap_stors, occupations,
        sync_cycles=sync_cycles, reps=reps, **kwargs)


STAGES = {
    "calibration": (MBRPhaseCorrectionExperiment,
                    "EntireFloquetCyclePhaseCalibrationProgram",
                    _calibration_batch),
    "spectrum": (MBRSpectrumExperiment,
                 "NPhotonHamiltonianSpectroscopyProgram",
                 _spectrum_batch),
    "propagator": (MBRPropagatorExperiment,
                   "EncodingPropagatorProgram",
                   _propagator_batch),
    "orthogonality": (MBROrthogonalityExperiment,
                      "EncodingOrthogonalityProgram",
                      _orthogonality_batch),
}


def build_stage(stage, defaults, swap_stors, occupations, **kwargs):
    """-> (OwnerExperiment, ProgramClass, batch) for one stage."""
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; expected {sorted(STAGES)}")
    owner, program_name, builder = STAGES[stage]
    program = getattr(fdmr, program_name)
    batch = builder(defaults, swap_stors, occupations, **kwargs)
    return owner, program, batch


def run_stage(station, stage, defaults, swap_stors, occupations,
              job_client=None, batch_size=1, show=False, **kwargs):
    """Acquire one stage, through the queue if a client is given.

    With a ``job_client`` this is the production path and goes through
    ``BatchRunner``, so provenance and HDF5 output happen as usual. Without
    one it acquires in-process, which is the only option off-prod; the return
    value is then a list of acquired Experiments rather than an aggregate.
    """
    owner, program, batch = build_stage(
        stage, defaults, swap_stors, occupations, **kwargs)

    if job_client is not None:
        runner = fdmr.BatchRunner(
            station=station, ExptClass=owner, ExptProgram=program,
            default_expt_cfg=batch.default_expt_cfg,
            job_client=job_client, show=show)
        return runner.execute(batch.configs, batch_size=batch_size, show=show)

    acquired = []
    for override in batch.configs:
        expt = owner(soccfg=station.soccfg, path=station.data_path,
                     prefix=f"{owner.__name__}_{stage}",
                     config_file=station.hardware_config_file,
                     program=program)
        expt.cfg = AttrDict(deepcopy(station.hardware_cfg))
        expt.cfg.expt = AttrDict(deepcopy(batch.default_expt_cfg))
        expt.cfg.expt.update(override)
        expt.im = station.im
        expt.acquire(progress=False)
        acquired.append(expt)
    return acquired


def smoke(station=None, swap_stors=(1, 2, 3, 4), occupations=None,
          stages=tuple(STAGES), reps=10, **kwargs):
    """Build, compile and acquire every stage at negligible depth.

    The cheapest end-to-end check that the acquisition path is intact. Returns
    ``{stage: n_jobs_acquired}``.
    """
    station = station if station is not None else mock_station()
    occupations = occupations or [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]
    defaults = mbr_defaults(swap_stors, reps=reps)
    out = {}
    for stage in stages:
        acquired = run_stage(station, stage, defaults, swap_stors, occupations,
                             reps=reps, **kwargs)
        out[stage] = len(acquired)
    return out

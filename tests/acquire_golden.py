# -*- coding: utf-8 -*-
"""Acquisition golden: what each qsim sweep driver asks for and how it slices.

The net under step 10E of ``docs/qsim/program_tree_plan.md``: four copies of
the sweep loop (``QsimBaseExperiment``, ``DarkBaseExperiment``,
``MBRJobExperiment``, ``QsimWignerBaseExperiment``), a fifth in
``SidebandAmpRabiExperiment``, and two loop wrappers become one driver. The
program golden cannot see this: the driver decides how many readouts it asks
each Program for, in which order it visits the sweep points, and which lane of
the shots goes into ``avgi``/``avgq`` and the parity lanes.

So the real Programs are built (the driver builds them), but their
``acquire`` and ``collect_shots`` are replaced by lane-tagged fake shots:
in point ``p`` (the p-th Program the driver builds), lane ``l`` of shot ``s``
reads ``I = 100 p + l + (s % 2) / 2`` and ``Q = -I``. A value in ``avgi``
therefore names the point and the lane it came from. The shots vary between
even and odd shots, so a pre-selection keeps some shots and not others.

Per case, the pinned record holds: the readout count the driver passed to
each Program (``readouts_per_experiment``), the swept values each Program saw,
and the keys, shapes and values of the data.

This is a characterization pin: it records what the drivers do now, including
the lane errors listed in the plan (7.3). Regenerate with
``pixi run python -m tests.acquire_golden`` and read the diff before committing
it -- that diff is the review.
"""
import contextlib
import importlib
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import numpy as np
from slab import AttrDict

from experiments.qsim.mbr_campaign import mbr_defaults, mock_station, pinned_config_set
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS,
    FLOQUET_DEFAULTS,
)

GOLDEN = Path(__file__).parent / "data" / "acquire_golden.json"
CONFIG_SET = "preload_current"
REPS = 4
QSIM = "experiments.qsim"
SWAP_STORS = [1, 2, 3, 4]

BASE = dict(
    expts=1, reps=REPS, rounds=1, qubits=[0], normalize=False, relax_delay=200,
    active_reset=False, man_reset=False, storage_reset=False,
    prepulse=True, postpulse=True, init_fock=True, init_stor=1, ro_stor=1,
    parity_fast=False, phase_second_pulse=180, man_mode_no=1,
    **ACTIVE_RESET_DEFAULTS,
    **FLOQUET_DEFAULTS,
)
ACTIVE_RESET = dict(active_reset=True, man_reset=True, storage_reset=[1, 2],
                    pre_relax_delay=100)


class Case(NamedTuple):
    key: str
    experiment: str             # "module:Class"
    program: str                # "module:Class", or "" for the Experiment's default_program
    expt: dict                  # the full cfg.expt
    watch: tuple                # cfg.expt keys to record per Program build
    stub_core: bool = False     # replace core_pulses by a wait (for leaves that do not compile)


def _t1(**overrides):
    cfg = dict(BASE, wait=0.0, waits=[1.0, 2.0, 3.0], swept_params=["wait"])
    cfg.update(overrides)
    return cfg


def _dark_t1(**overrides):
    cfg = dict(BASE, swap_stors=SWAP_STORS, wait_length=0.0,
               wait_lengths=[1.0, 2.0, 3.0], swept_params=["wait_length"])
    cfg.update(overrides)
    return cfg


def _wigner(**overrides):
    cfg = dict(BASE, wait_us_time=0.0, wait_us_times=[1.0, 2.0],
               swept_params=["wait_us_time"],
               alpha_list=[[0.5, 0.0], [0.0, 0.5], [0.25, 0.25]])
    cfg.update(overrides)
    return cfg


def _mbr(**overrides):
    from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment

    cfg = dict(mbr_defaults(SWAP_STORS, reps=REPS))
    cfg.update(MBRTimeTraceExperiment.job_config(
        [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 2], SWAP_STORS, reps=REPS))
    cfg.update(overrides)
    return cfg


QSIM_BASE = f"{QSIM}.qsim_base:QsimBaseExperiment"
DARK_BASE = f"{QSIM}.dark_base:DarkBaseExperiment"
WIGNER = f"{QSIM}.qsim_base_wigner:QsimWignerBaseExperiment"
T1 = f"{QSIM}.sideband_scramble:StorageT1Program"
DARK_T1 = f"{QSIM}.dark_mode_t1:DarkT1Program"
KERR_WAIT = f"{QSIM}.kerr:KerrWaitProgram"

CASES = [
    # QsimBaseExperiment: 1D and 2D, each herald option.
    Case("qsim_base__1d", QSIM_BASE, T1, _t1(), ("wait",)),
    Case("qsim_base__2d", QSIM_BASE, T1,
         _t1(init_stors=[1, 2], swept_params=["init_stor", "wait"]), ("init_stor", "wait")),
    Case("qsim_base__parity_check", QSIM_BASE, T1, _t1(parity_check=True), ("wait",)),
    Case("qsim_base__active_reset", QSIM_BASE, T1, _t1(**ACTIVE_RESET), ("wait",)),
    Case("qsim_base__pre_selection", QSIM_BASE, T1,
         _t1(**ACTIVE_RESET, pre_selection_reset=True), ("wait",)),
    Case("qsim_base__active_reset_parity_check", QSIM_BASE, T1,
         _t1(**ACTIVE_RESET, parity_check=True), ("wait",)),
    # QsimBaseExperiment does not count this readout (plan 7.3).
    Case("qsim_base__multiparity", QSIM_BASE, DARK_T1, _dark_t1(readout="multiparity"),
         ("wait_length",)),
    # DarkBaseExperiment.
    Case("dark_base__1d", DARK_BASE, DARK_T1, _dark_t1(), ("wait_length",)),
    Case("dark_base__2d", DARK_BASE, DARK_T1,
         _dark_t1(init_stors=[1, 2], swept_params=["init_stor", "wait_length"]),
         ("init_stor", "wait_length")),
    Case("dark_base__parity_check", DARK_BASE, DARK_T1, _dark_t1(parity_check=True),
         ("wait_length",)),
    Case("dark_base__pre_selection", DARK_BASE, DARK_T1,
         _dark_t1(**ACTIVE_RESET, pre_selection_reset=True), ("wait_length",)),
    Case("dark_base__multiparity", DARK_BASE, DARK_T1, _dark_t1(readout="multiparity"),
         ("wait_length",)),
    Case("dark_base__active_reset_multiparity", DARK_BASE, DARK_T1,
         _dark_t1(**ACTIVE_RESET, readout="multiparity"), ("wait_length",)),
    # MBRJobExperiment (through a job class).
    Case("mbr_job__time_trace", f"{QSIM}.mbr_time_trace:MBRTimeTraceExperiment", "", _mbr(),
         ("floquet_cycle", "ramsey_phase")),
    Case("mbr_job__pre_selection", f"{QSIM}.mbr_time_trace:MBRTimeTraceExperiment", "",
         _mbr(pre_selection_reset=True), ("floquet_cycle", "ramsey_phase")),
    # QsimWignerBaseExperiment: an alpha axis inside the sweep.
    Case("wigner__default", WIGNER, KERR_WAIT, _wigner(), ("wait_us_time", "wigner_alpha")),
    Case("wigner__pulse_correction", WIGNER, KERR_WAIT, _wigner(pulse_correction=True),
         ("wait_us_time", "wigner_alpha", "phase_second_pulse")),
    Case("wigner__parity_check", WIGNER, KERR_WAIT, _wigner(parity_check=True),
         ("wait_us_time", "wigner_alpha")),
    # wigner__post_select_pre_pulse was here until step 10D: the driver
    # counted a readout that no qsim Program plays; the Program refuses it now.
    Case("wigner__pre_selection", WIGNER, KERR_WAIT,
         _wigner(**ACTIVE_RESET, pre_selection_reset=True), ("wait_us_time", "wigner_alpha")),
    # Hand-written 2D loop. Its Program sets its own Floquet length, which the
    # pinned set's envelopes reject (see program_asm_golden), so core_pulses is
    # stubbed: only the driver is under test here. Until step 10D it raised
    # unless the config set perform_wigner (the template read it).
    Case("sideband_amp_rabi__2d", f"{QSIM}.sideband_amp_rabi:SidebandAmpRabiExperiment",
         f"{QSIM}.sideband_amp_rabi:SidebandAmpRabiProgram",
         dict(BASE, init_stor=2, length=0.5,
              detunes=[-0.1, 0.1], gains=[1000, 2000, 3000]),
         ("detune", "gain"), stub_core=True),
    Case("sideband_amp_rabi__active_reset_parity_check",
         f"{QSIM}.sideband_amp_rabi:SidebandAmpRabiExperiment",
         f"{QSIM}.sideband_amp_rabi:SidebandAmpRabiProgram",
         dict(BASE, **ACTIVE_RESET, parity_check=True,
              init_stor=2, length=0.5,
              detunes=[-0.1, 0.1], gains=[1000, 2000]),
         ("detune", "gain"), stub_core=True),
    # Loop wrappers.
    Case("floquet_calibration_amplification",
         f"{QSIM}.sideband_scramble:FloquetCalibrationAmplificationExperiment",
         f"{QSIM}.sideband_scramble:FloquetCalibrationProgram",
         dict(BASE, storA=1, storB=2, n_scramble_cycles=[0, 1], n_floquet_per_scramble=1,
              storA_advance_phases=[0.0, 10.0], storB_advance_phases=[0.0, 20.0, 40.0]),
         ("floquet_cycle", "storA_advance_phase", "storB_advance_phase")),
    Case("floquet_displacement_kerr",
         f"{QSIM}.floquet_displacement_kerr:FloquetDisplacementKerrExperiment",
         f"{QSIM}.floquet_displacement_kerr:FloquetDisplacementKerrProgram",
         dict(BASE, **ACTIVE_RESET, swap_stors=SWAP_STORS, zero_floquet_gain=False,
              ramsey_freq=0.2, displace_gains=[2000, 4000],
              n_cycle_pairs=[0, 3], swept_params=["displace_gain", "n_cycle_pair"]),
         ("displace_gain", "n_cycle_pair")),
]


def _load(target):
    module, name = target.split(":")
    return getattr(importlib.import_module(module), name)


def _plain(value):
    """-> a JSON value; complex numbers as [re, im], floats rounded."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (complex, np.complexfloating)):
        return [round(float(value.real), 6), round(float(value.imag), 6)]
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _summary(value):
    """-> shape and values of one data entry."""
    try:
        arr = np.asarray(value)
    except ValueError:              # ragged
        return {"ragged": [_summary(v) for v in value]}
    if arr.dtype == object:
        return {"object": [_summary(v) for v in value]}
    out = {"shape": list(arr.shape)}
    if arr.size <= 64:
        out["values"] = _plain(arr.tolist())
    else:
        out["first"] = _plain(arr.reshape(-1)[:8].tolist())
        out["last"] = _plain(arr.reshape(-1)[-8:].tolist())
    return out


class _LaneTagger:
    """Replaces ``acquire``/``collect_shots`` of one Program class."""

    def __init__(self, program_class, watch, stub_core):
        self.cls = program_class
        self.watch = watch
        self.stub_core = stub_core
        self.builds = []

    def __enter__(self):
        tagger = self
        self.saved = {name: self.cls.__dict__.get(name)
                      for name in ("acquire", "collect_shots", "core_pulses")}

        def acquire(prog, soc, threshold=None, load_pulses=True, progress=False,
                    debug=False, readouts_per_experiment=1, **kwargs):
            point = len(tagger.builds)
            n = int(readouts_per_experiment)
            reps = int(prog.cfg.expt.reps)
            lanes = np.arange(n, dtype=float)
            shots = 100.0 * point + lanes[None, :] + (np.arange(reps)[:, None] % 2) / 2
            prog._fake_i = shots.reshape(-1)
            tagger.builds.append(dict(
                readouts_per_experiment=n,
                **{key: _plain(prog.cfg.expt.get(key)) for key in tagger.watch}))
            avg = shots.mean(axis=0)
            return np.array([avg]), np.array([-avg])

        def collect_shots(prog):
            return prog._fake_i.copy(), -prog._fake_i.copy()

        self.cls.acquire = acquire
        self.cls.collect_shots = collect_shots
        if self.stub_core:
            self.cls.core_pulses = lambda prog: prog.sync_all(prog.us2cycles(0.1))
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                if name in self.cls.__dict__:
                    delattr(self.cls, name)
            else:
                setattr(self.cls, name, value)


def record(station, case, tmp_path):
    """Acquire one case with lane-tagged shots; -> its pinned record."""
    Expt = _load(case.experiment)
    program = _load(case.program) if case.program else Expt.default_program
    expt = Expt(soccfg=station.soccfg, path=str(tmp_path), prefix=Expt.__name__,
                config_file=station.hardware_config_file, program=program)
    expt.cfg = AttrDict(deepcopy(station.hardware_cfg))
    expt.cfg.device.storage._ds_storage = station.ds_storage
    expt.cfg.device.storage._ds_floquet = station.ds_floquet
    expt.cfg.expt = AttrDict(deepcopy(case.expt))
    expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]
    with _LaneTagger(program, case.watch, case.stub_core) as tagger:
        with contextlib.redirect_stdout(io.StringIO()):
            expt.acquire()
    data = expt.data
    return {
        "builds": tagger.builds,
        "read_num": _plain(expt.cfg.get("read_num")),
        "data": {key: _summary(data[key]) for key in sorted(data)},
    }


def records(tmp_path):
    station = mock_station(**pinned_config_set(CONFIG_SET))
    return {case.key: record(station, case, tmp_path) for case in CASES}


def read():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def dumps(recs):
    return json.dumps(recs, indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        text = dumps(records(Path(tmp)))
    GOLDEN.write_text(text, encoding="utf-8")
    print(f"wrote {GOLDEN.name} ({len(CASES)} cases, {len(text) / 1024:.1f} KiB)")

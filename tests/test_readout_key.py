# -*- coding: utf-8 -*-
"""The ``readout`` key: one input for the final readout (step 10D).

Rules from ``docs/qsim/program_tree_plan.md``, section 4. The pulses each mode
plays are pinned by ``tests/program_asm_golden.py``; this module pins the
rules around them: what a Program refuses, how many readouts a shot has, and
how saved configs from before the key are read.

Run:  pixi run python -m pytest tests/test_readout_key.py -v
"""
import contextlib
import io

import pytest
from slab import AttrDict

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.qsim_base import (
    READOUT_MODES,
    RETIRED_READOUT_FLAGS,
    QsimBaseProgram,
    readout_lane_count,
    readout_mode,
    saved_readout_mode,
)

pytest.importorskip("qick")

ACTIVE_RESET = dict(active_reset=True, man_reset=True, storage_reset=[1, 2])


@pytest.fixture(scope="module")
def station():
    from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
    return mock_station(**pinned_config_set("preload_current"))


def _build(station, target, **overrides):
    from tests import program_asm_golden as golden
    with contextlib.redirect_stdout(io.StringIO()):
        return golden.build(station, target, overrides)


TEMPLATE = "experiments.qsim.qsim_base:QsimBaseProgram"


@pytest.mark.parametrize("flag", RETIRED_READOUT_FLAGS)
@pytest.mark.parametrize("value", [True, False])
def test_a_retired_flag_is_refused_even_when_false(station, flag, value):
    with pytest.raises(ValueError, match="replaced by cfg.expt.readout"):
        _build(station, TEMPLATE, **{flag: value})


def test_post_select_pre_pulse_is_refused_only_when_set(station):
    _build(station, TEMPLATE, post_select_pre_pulse=False)
    with pytest.raises(ValueError, match="post_select_pre_pulse"):
        _build(station, TEMPLATE, post_select_pre_pulse=True)


def test_an_unknown_mode_is_refused(station):
    with pytest.raises(ValueError, match="expected one of"):
        _build(station, TEMPLATE, readout="wigner_tomography")


@pytest.mark.parametrize("mode", sorted(READOUT_MODES))
@pytest.mark.parametrize("postpulse", [True, False])
def test_every_mode_plays_with_or_without_postpulse(station, mode, postpulse):
    _build(station, TEMPLATE, readout=mode, postpulse=postpulse, wigner_alpha=0.5)


def test_the_r_program_refuses_the_same():
    from experiments.qsim.dark_base import DarkBaseRProgram
    cfg = AttrDict({"expt": {"perform_wigner": False}})
    with pytest.raises(ValueError, match="replaced by cfg.expt.readout"):
        DarkBaseRProgram(soccfg=None, cfg=cfg)


@pytest.mark.parametrize("mode", ["qubit", "parity"])
def test_mbr_refuses_any_readout_but_the_default(station, mode):
    from copy import deepcopy

    from experiments.qsim.mbr_campaign import mbr_defaults
    from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment, MBRTimeTraceProgram
    from experiments.qsim.utils import ensure_list_in_cfg

    expt = dict(mbr_defaults([1, 2, 3, 4], reps=4))
    expt.update(MBRTimeTraceExperiment.job_config(
        [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 2], [1, 2, 3, 4], reps=4))
    # One point of the job's sweep, as the driver sets it.
    expt.update(floquet_cycle=0, ramsey_phase=expt["ramsey_phases"][0],
                postpulse=True, readout=mode)
    cfg = AttrDict(deepcopy(station.hardware_cfg))
    cfg.device.storage._ds_storage = station.ds_storage
    cfg.device.storage._ds_floquet = station.ds_floquet
    cfg.expt = AttrDict(expt)
    ensure_list_in_cfg(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        if mode == "qubit":
            MBRTimeTraceProgram(soccfg=station.soccfg, cfg=cfg)
        else:
            with pytest.raises(ValueError, match="the MBR jobs read the qubit Ramsey"):
                MBRTimeTraceProgram(soccfg=station.soccfg, cfg=cfg)


# ------------------------------------------------------------------ counting

def _cfg(**expt):
    return AttrDict({"expt": dict(dict(postpulse=True), **expt)})


def _active_reset_lanes():
    params = MMAveragerProgram.get_active_reset_params(_cfg(**ACTIVE_RESET))
    return MMAveragerProgram.active_reset_read_num(**params)


@pytest.mark.parametrize("mode", sorted(READOUT_MODES))
@pytest.mark.parametrize("parity_check", [False, True])
@pytest.mark.parametrize("active_reset", [False, True])
def test_readouts_per_shot(mode, parity_check, active_reset):
    expt = dict(readout=mode, parity_check=parity_check)
    if active_reset:
        expt.update(ACTIVE_RESET)
    heralds = int(parity_check) + (_active_reset_lanes() if active_reset else 0)
    expected = heralds + (2 if mode == "multiparity" else 1)
    cfg = _cfg(**expt)
    assert QsimBaseProgram.readouts_per_shot(cfg) == expected
    # For a new config, the saved-config reader counts the same.
    assert readout_lane_count(cfg) == expected


# The retired booleans, as jobs saved before step 10D record them, and the
# mode the template played for each (perform_wigner won over the parity
# flags, multiparity_readout over parity_readout).
SAVED = [
    (dict(), "qubit"),
    (dict(perform_wigner=False, parity_readout=False, multiparity_readout=False), "qubit"),
    (dict(parity_readout=True), "parity"),
    (dict(multiparity_readout=True), "multiparity"),
    (dict(parity_readout=True, multiparity_readout=True), "multiparity"),
    (dict(perform_wigner=True), "wigner"),
    (dict(perform_wigner=True, parity_readout=True, multiparity_readout=True), "wigner"),
    (dict(slow_pi_ge_readout=True), "slow_pi_ge"),
    (dict(readout="parity", perform_wigner=True), "parity"),   # the key wins
]


@pytest.mark.parametrize("expt, mode", SAVED)
def test_saved_configs_from_before_the_key(expt, mode):
    assert saved_readout_mode(AttrDict(dict(expt))) == mode


@pytest.mark.parametrize("expt, mode", SAVED)
def test_saved_lane_count_is_the_old_count(expt, mode):
    """The pre-10D ``readout_lane_count``: 1 + parity check + active reset + multiparity."""
    for extra in (dict(), dict(parity_check=True), dict(ACTIVE_RESET)):
        cfg = AttrDict({"expt": dict(expt, **extra)})
        old = 1
        old += int(cfg.expt.get("parity_check", False))
        if cfg.expt.get("active_reset", False):
            old += _active_reset_lanes()
        old += int(bool(cfg.expt.get("multiparity_readout", False))) if "readout" not in expt \
            else int(expt["readout"] == "multiparity")
        assert readout_lane_count(cfg) == old


def test_a_new_config_is_read_strictly():
    assert readout_mode(AttrDict({"postpulse": True})) == "qubit"
    with pytest.raises(ValueError):
        readout_mode(AttrDict({"postpulse": True, "parity_readout": False}))

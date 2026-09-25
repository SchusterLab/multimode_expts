# -*- coding: utf-8 -*-
"""Tests of MBR code that moved to `experiments/qsim/deprecated/` (step 7a).

See `docs/qsim/mbr_step7_plan.md`. This code is not maintained. When a test here
breaks, mark it `pytest.mark.skip` with the reason below. Do not fix the code.

Moved from `tests/test_mbr_acquire_mock.py` on 2026-09-24: the disorder SFF
tests (`DisorderSFFExperiment` is to be deleted, plan decision 3).
"""
import pytest

from experiments.qsim.mbr_campaign import mbr_defaults, mock_station, pinned_config_set

DEPRECATED_SKIP = "moved to deprecated/, see docs/qsim/mbr_step7_plan.md"


# --------------------------------------------------------------------------
# Disorder SFF. Ported from main in its own module; see experiments/qsim/deprecated/mbr_sff.py.
# Jonginn marked its mixins "NOT PERUSED", and these tests treat it that way:
# they pin what it does today rather than asserting it is right.
# --------------------------------------------------------------------------

SFF_SWAP_STORS = [1, 2, 3]
# The complete N=1 basis over M1 plus three storage modes, so Tr(U)/D is a sum
# over a real basis rather than an arbitrary subset.
SFF_OCCUPATIONS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _sff_plan(detunings):
    from experiments.qsim.deprecated.mbr_sff import DisorderSFFExperiment

    defaults = mbr_defaults(SFF_SWAP_STORS, reps=10)
    return DisorderSFFExperiment.batch(
        defaults, SFF_SWAP_STORS, SFF_OCCUPATIONS, [0, 1, 2],
        phase_by_occupation={tuple(o): 0.0 for o in SFF_OCCUPATIONS},
        realization_detunings_MHz=detunings,
        sync_cycles=10,          # the module floors this at 10 for register setup
        shots_per_replica=1, visibility_reps=10, realizations_per_job=1)


def _acquire_sff(station, plan):
    from copy import deepcopy

    from slab import AttrDict

    from experiments.qsim.deprecated.mbr_sff import DisorderSFFExperiment

    out = []
    for override in plan.configs:
        expt = DisorderSFFExperiment(
            soccfg=station.soccfg, path=station.data_path, prefix="mock_sff",
            config_file=station.hardware_config_file, program=plan.program)
        expt.cfg = AttrDict(deepcopy(station.hardware_cfg))
        expt.cfg.expt = AttrDict(deepcopy(plan.default_expt_cfg))
        expt.cfg.expt.update(override)
        expt.im = station.im
        expt.acquire(progress=False)
        out.append(expt)
    return out


def test_sff_acquires_with_positive_detunings():
    """Both SFF job kinds build, compile and acquire.

    The RAverager depth-sweep path is separate machinery from the fixed-depth
    spectroscopy programs above -- its own register allocation and counted
    Floquet loop -- so it needs its own coverage.
    """
    st = mock_station(**pinned_config_set("preload_current"))
    plan = _sff_plan([[0.05, 0.03, 0.04], [0.02, 0.06, 0.01]])
    acquired = _acquire_sff(st, plan)

    assert len(acquired) == len(plan.configs) == 3
    kinds = [e.cfg.expt.sff_job_kind for e in acquired]
    assert kinds.count("visibility") == 1, "expected one visibility job"
    assert kinds.count("disorder") == 2, "expected two disorder jobs"


def test_sff_rejects_negative_detunings():
    """Documents a real limitation, deliberately not fixed here.

    ``_setup_phase_updated_by_depth`` guards its ``mathi`` immediate with
    ``0 <= deg2reg(phase_change) < 2**31``. ``deg2reg`` wraps a negative angle
    to near ``2**32`` -- ``deg2reg(-0.5)`` is 4289002064 -- so *every* negative
    phase change per depth trips the guard. The comment directly above it says
    the opposite: that negative phase changes are represented by their wrapped
    positive value and repeated ``mathi`` additions still evolve correctly.

    This matters because a disorder ensemble is naturally zero-mean, so about
    half of every realization's detunings are negative and raise. Whether the
    bound should be ``2**32`` depends on the tProc ``mathi`` immediate width,
    which is jonginn's call -- hence a test that pins the behaviour and
    explains it, rather than a guess at a fix.
    """
    st = mock_station(**pinned_config_set("preload_current"))
    plan = _sff_plan([[-0.05, 0.03, 0.04], [0.02, 0.06, 0.01]])

    with pytest.raises(RuntimeError, match="does not fit mathi"):
        _acquire_sff(st, plan)

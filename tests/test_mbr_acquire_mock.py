# -*- coding: utf-8 -*-
"""Build, compile and acquire every MBR stage in mock mode.

Why this exists
---------------
The refactor split the MBR acquisition code four ways, and the rest of the
suite checks only what happens *after* acquisition: it loads saved HDF5 and
compares numbers. Nothing in it constructs a qick program from a real swap
dataset row. So the suite was fully green while every Floquet swap played a
0.08 us gaussian in place of the calibrated 0.037 us flat top -- main had moved
the envelope choice into the dataset and added a third waveform mode, and a
duplicated registration loop kept calling ``add_gauss`` regardless. It built,
it compiled, it "acquired", and it was wrong.

This test closes that gap. It runs the real ``initialize()`` and ``body()`` and
the real ASM compile, with ``MockQickSoc`` standing in only at the FPGA
boundary, so the qick library's own parameter validators fire exactly as they
do on the measurement PC.

It replaces ``test_qsim_measurement_split.py``, which compared moved code
against a pinned commit by AST. That checked only that code had not changed,
which stops being useful the moment the modules evolve on purpose -- and it
would not have caught the envelope bug either, since both copies were faithful.

Both pinned config sets run, because they exercise different branches:
``august_n3`` is all ``gauss`` and ``preload_current`` is all
``preload_flattop``, which is the only one that reaches the preloaded
register-bank playback path.

No mount, no environment
------------------------
Everything needed is committed: the config sets live in
``tests/data/config_set/`` and the firmware shape in
``configs/soccfg_snapshot.json``. Unlike the golden-baseline tests, this one
needs no measurement data, so it runs anywhere.
"""
import numpy as np
import pytest

from experiments.qsim.mbr_campaign import (
    STAGES,
    mbr_defaults,
    mock_station,
    pinned_config_set,
    pinned_sets,
    run_stage,
)

CONFIG_SETS = sorted(pinned_sets())

# Two occupations of the same total photon number, so the encoder and decoder
# paths differ and the phase bookkeeping actually has something to carry.
OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]
SWAP_STORS = [1, 2, 3, 4]

# Jobs each stage builds from OCCUPATIONS. Pinned so a stage silently
# collapsing to zero jobs fails instead of passing vacuously.
EXPECTED_JOBS = {
    "calibration": 4,
    "spectrum": 4,
    "propagator": 2,
    "orthogonality": 2,
}


@pytest.fixture(scope="module", params=CONFIG_SETS)
def station(request):
    """A mock station per pinned config set, built once."""
    return request.param, mock_station(**pinned_config_set(request.param))


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_stage_acquires(station, stage):
    """Every stage builds, compiles and acquires at negligible depth."""
    set_name, st = station
    assert st.is_mock, "refusing to acquire against real instruments"

    defaults = mbr_defaults(SWAP_STORS, reps=10)
    acquired = run_stage(st, stage, defaults, SWAP_STORS, OCCUPATIONS, reps=10)

    assert len(acquired) == EXPECTED_JOBS[stage], (
        f"{stage} on {set_name} built {len(acquired)} jobs, "
        f"expected {EXPECTED_JOBS[stage]}"
    )
    for expt in acquired:
        for field in ("avgi", "avgq", "amps", "phases"):
            assert field in expt.data, f"{stage}: {field} missing from acquired data"


def test_waveform_mode_follows_the_dataset(station):
    """The envelope comes from the swap dataset, not from a config default.

    The regression this pins: ``m1s_wf_name`` naming a preload_flattop mode
    while the registered envelope is a gaussian. Checking the mode string is
    not enough, so this also asserts the registered envelope is a real flat
    top -- most of its samples sit at the plateau, which is false for a
    gaussian.
    """
    set_name, st = station
    defaults = mbr_defaults(SWAP_STORS, reps=10)
    acquired = run_stage(st, "propagator", defaults, SWAP_STORS, OCCUPATIONS,
                         reps=10)
    prog = acquired[0].prog

    expected = {"august_n3": "gauss", "preload_current": "preload_flattop"}[set_name]
    modes = [prog.m1s_waveform_mode[stor - 1] for stor in SWAP_STORS]
    assert set(modes) == {expected}, f"{set_name}: got modes {modes}"

    for stor in SWAP_STORS:
        index = stor - 1
        name = prog.m1s_wf_name[index]
        assert expected in name, f"{name} does not name a {expected} envelope"

        envelope = prog.envelopes[prog.m1s_ch[index]]["envs"].get(name)
        assert envelope is not None, f"{name} was never registered"

        data = np.asarray(envelope["data"] if isinstance(envelope, dict)
                          and "data" in envelope else envelope)
        profile = data[:, 0] if data.ndim == 2 else data
        at_plateau = (profile > 0.95 * profile.max()).mean()
        if expected == "preload_flattop":
            assert at_plateau > 0.3, (
                f"{name} is registered but only {at_plateau:.0%} of samples are "
                f"at the plateau -- that is a gaussian, not a flat top")
        else:
            assert at_plateau < 0.3, (
                f"{name} should be a gaussian but {at_plateau:.0%} of samples "
                f"sit at the plateau")


def test_program_is_not_driven_directly():
    """Instantiating a Program instead of an Experiment fails, as documented.

    Pins the reason ``run_stage`` goes through ``Experiment.acquire``: the
    plural-to-singular sweep expansion lives there, so a Program built from a
    stage config alone is missing the key its body reads. Worth a test because
    the failure is an opaque AttributeError that has cost time more than once.
    """
    from copy import deepcopy

    from slab import AttrDict

    from experiments.qsim import floquet_dark_mode_readout as fdmr
    from experiments.qsim.mbr_campaign import build_stage

    st = mock_station(**pinned_config_set("preload_current"))
    defaults = mbr_defaults(SWAP_STORS, reps=10)
    _, program, batch = build_stage(
        "propagator", defaults, SWAP_STORS, OCCUPATIONS, reps=10)

    cfg = AttrDict(deepcopy(st.hardware_cfg))
    cfg.expt = AttrDict(deepcopy(batch.default_expt_cfg))
    cfg.expt.update(batch.configs[0])
    assert "cycle_decoder_analyzers" in cfg.expt
    assert "cycle_decoder_analyzer" not in cfg.expt

    with pytest.raises(AttributeError, match="cycle_decoder_analyzer"):
        program(soccfg=st.soccfg, cfg=cfg)

    assert program is fdmr.EncodingPropagatorProgram

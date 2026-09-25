# -*- coding: utf-8 -*-
"""Build, compile and acquire every MBR job class in mock mode.

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
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.floquet_timing import resolve_floquet_timing
from experiments.saved_jobs import load_job
from experiments.qsim.mbr_campaign import (
    mbr_defaults,
    mock_station,
    pinned_config_set,
    pinned_sets,
    smoke,
)

CONFIG_SETS = sorted(pinned_sets())

# Two occupations of the same total photon number, so the encoder and decoder
# paths differ and the phase bookkeeping actually has something to carry.
OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]
SWAP_STORS = [1, 2, 3, 4]

# The products smoke() acquires, one job per occupation each. Pinned so a
# product silently collapsing to zero jobs fails instead of passing vacuously.
PRODUCTS = ["ortho_column_q0", "ortho_column_q4", "stark_cal", "time_trace"]


@pytest.fixture(scope="module", params=CONFIG_SETS)
def station(request):
    """A mock station per pinned config set, built once."""
    return request.param, mock_station(**pinned_config_set(request.param))


@pytest.fixture(scope="module")
def products(station):
    """Every MBR product, acquired once per config set."""
    _, st = station
    assert st.is_mock, "refusing to acquire against real instruments"
    return smoke(st, SWAP_STORS, OCCUPATIONS, reps=10)


@pytest.mark.parametrize("name", PRODUCTS)
def test_product_acquires(station, products, name):
    """Every job class builds, compiles and acquires at negligible depth."""
    set_name, _ = station
    assert sorted(products) == PRODUCTS
    acquired = products[name].children
    assert len(acquired) == len(OCCUPATIONS), (
        f"{name} on {set_name} built {len(acquired)} jobs, expected {len(OCCUPATIONS)}")
    for expt in acquired:
        for field in ("avgi", "avgq", "amps", "phases"):
            assert field in expt.data, f"{name}: {field} missing from acquired data"


def test_waveform_mode_follows_the_dataset(station, products):
    """The envelope comes from the swap dataset, not from a config default.

    The regression this pins: ``m1s_wf_name`` naming a preload_flattop mode
    while the registered envelope is a gaussian. Checking the mode string is
    not enough, so this also asserts the registered envelope is a real flat
    top -- most of its samples sit at the plateau, which is false for a
    gaussian.
    """
    set_name, _ = station
    prog = products["ortho_column_q4"].children[0].prog

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

    Pins the reason jobs go through ``Experiment.acquire``: the
    plural-to-singular sweep expansion lives there, so a Program built from a
    job config alone is missing the keys its body reads. Worth a test because
    the failure is an opaque AttributeError that has cost time more than once.
    """
    from copy import deepcopy

    from slab import AttrDict

    from experiments.qsim.mbr_ortho_column import (
        MBROrthoColumnExperiment,
        MBROrthoColumnProgram,
    )

    st = mock_station(**pinned_config_set("preload_current"))
    cfg = AttrDict(deepcopy(st.hardware_cfg))
    cfg.expt = AttrDict(mbr_defaults(SWAP_STORS, reps=10))
    cfg.expt.update(MBROrthoColumnExperiment.job_config(
        OCCUPATIONS[0], OCCUPATIONS, SWAP_STORS, cycle=4))
    assert "ramsey_phases" in cfg.expt and "decoder_occupations" in cfg.expt
    assert "ramsey_phase" not in cfg.expt and "decoder_occupation" not in cfg.expt

    with pytest.raises(AttributeError, match="ramsey_phase"):
        MBROrthoColumnProgram(soccfg=st.soccfg, cfg=cfg)


# --------------------------------------------------------------------------
# The derived-parameter provenance attribute
# --------------------------------------------------------------------------
#
# The Floquet cycle time and couplings are the one thing saved data cannot
# otherwise carry: they existed only on the compiled program, which lives in
# the job pickle, and pickles are ephemeral. Acquisition now records them in
# the HDF5 `derived_params` attribute -- beside `config`, not inside it,
# because `cfg.expt` is the input a notebook overrides by hand and this is
# generated output.
#
# These two tests are the pair that matters: the value is written where the
# reader looks, and it agrees with the independent way of recovering it.


def _acquire_one(station, tmp_path):
    """-> one acquired time-trace job, saved under `tmp_path`.

    The mock station's own output root is a prod path
    (`C:/experiments/mock_data`), so point the file somewhere the test can
    read and save it again.
    """
    products = smoke(station, SWAP_STORS, OCCUPATIONS[:1], reps=10)
    expt = products["time_trace"].children[0]
    expt.fname = str(tmp_path / "JOB-19990101-00001_MBRTimeTraceExperiment.h5")
    expt.save_data(expt.data)
    return expt


def test_saved_h5_carries_the_derived_timing(station, tmp_path):
    """Acquisition writes the timing, and analysis reads it back with no sidecar.

    The round trip is the point. `provenance={}` below means the loader has no
    sidecar entry to fall back on, so the only way it can answer is the
    attribute the save path just wrote.
    """
    _, st = station
    assert st.is_mock, "refusing to acquire against real instruments"

    expt = _acquire_one(st, tmp_path)
    expected = expt.derived_params()
    assert expected["floquet_cycle_us"] > 0.
    assert len(expected["m1s_pi_fracs"]) == 7
    assert len(expected["couplings_MHz"]) == 7

    with h5py.File(expt.fname, "r") as handle:
        assert "derived_params" in handle.attrs, sorted(handle.attrs)
        # Still beside `config`, never inside it: a derived value in cfg.expt
        # would be indistinguishable from a hand-set input.
        assert "config" in handle.attrs
        recorded = json.loads(handle.attrs["derived_params"])
        cfg = json.loads(handle.attrs["config"])
    assert recorded == expected
    assert "floquet_cycle_us" not in cfg["expt"]

    job = load_job("JOB-19990101-00001", path=expt.fname, provenance={})
    assert job.prog.calculate_floquet_cycle_us() == expected["floquet_cycle_us"]
    assert job.prog.m1s_pi_fracs == expected["m1s_pi_fracs"]
    assert "derived_params" in job.prog.source


def test_recorded_timing_agrees_with_the_archive_resolver(station, tmp_path):
    """The two ways of recovering the timing give the same number.

    One reads what acquisition recorded; the other recomputes it from the
    versioned swap CSV (`resolve_floquet_timing`, the path used for files
    written before the attribute existed). If these ever disagree, then files
    with and without the attribute would analyze differently, and the
    attribute would have made old and new data incomparable.
    """
    set_name, st = station
    expt = _acquire_one(st, tmp_path)
    recorded = expt.derived_params()

    # The resolver wants an archive laid out as {root}/floquet_storage_swap/,
    # while the pinned sets are a flat directory. Build the shape it expects,
    # still entirely from committed files -- no mount, per this module's note.
    csv = Path(pinned_config_set(set_name)["floquet_file"])
    archive = tmp_path / "archive"
    (archive / "floquet_storage_swap").mkdir(parents=True)
    shutil.copy2(csv, archive / "floquet_storage_swap" / csv.name)

    resolved = resolve_floquet_timing(expt.cfg, csv.stem, archive=archive)

    assert resolved["floquet_cycle_us"] == pytest.approx(
        recorded["floquet_cycle_us"], rel=1e-12), set_name
    assert list(resolved["m1s_pi_fracs"]) == list(recorded["m1s_pi_fracs"])

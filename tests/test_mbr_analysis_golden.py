"""Characterization test for the MBR spectrum analysis.

What this is for
----------------
The refactor's first real work is moving numerical code out of the 8k-line
``floquet_dark_mode_readout.py`` without changing what it computes. A pure code
move has exactly one failure mode -- the numbers changed -- so one end-to-end
comparison against a pinned baseline is a better net than a pile of per-function
unit tests: it covers every extraction at once and costs one file.

It pins *current* behaviour, bugs included. That is the point of a
characterization test, not an endorsement. Sections 2.1 to 2.5 of
``docs/qsim_mbr_refactor.md`` describe defects this baseline currently locks in.

Re-blessing
-----------
When behaviour changes **on purpose**, regenerate the baseline in the same
commit as the change and say why in the commit message::

    MBR_GOLDEN_BLESS=1 pixi run python -m pytest tests/test_mbr_analysis_golden.py

Consequence for ordering: do every pure move while this test is green, because
extraction is free only until the first re-blessing. See spec section 14.

Running it
----------
Needs the raw HDF5 for ``JOB-20260815-00009..16``. On the acquisition
workstation that is automatic. Elsewhere point it at your mount::

    MULTIMODE_DATA_ROOT=/Volumes/experiments pixi run python -m pytest tests/

It fails rather than skips when the data is unreachable -- there is no CI here,
so every run is a human on a machine that should have the data, and a skip
would just hide the fact that the suite never ran.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests.mbr_reference import (
    CHARACTERIZATION_ANALYSIS,
    CHARACTERIZATION_JOB_IDS,
    CHARACTERIZATION_TIMING,
    flatten_result,
    run_reference_analysis,
)

# One baseline per spectrum method. Both are needed: the FFT path and the
# Matrix-Pencil path share the reconstruction but almost nothing after it, so a
# baseline for one does not protect moves in the other. The first extraction
# proved that the hard way -- a missing import in the moved Matrix-Pencil code
# was invisible to the FFT baseline.
BASELINES = {
    "fft": Path(__file__).parent / "data" / "mbr_spectrum_20260815.npz",
    "matrix_pencil": Path(__file__).parent / "data" / "mbr_matrix_pencil_20260815.npz",
}

# Tolerances. The analysis is deterministic -- same inputs, same floating-point
# operations -- so a pure move should reproduce bit-for-bit. rtol is left just
# above zero so a genuine change is loud while a harmless last-bit difference
# from, say, a numpy version bump does not block the refactor.
RTOL = 1e-12
ATOL = 1e-12

# Fields whose meaning is structural rather than numerical. Compared exactly.
EXACT_PATHS = (
    "phase_frame",
    "spectrum_method",
    "hardware.source",
    "legacy_analyzer_migration",
    "photon_number",
)


def _blessing() -> bool:
    return os.environ.get("MBR_GOLDEN_BLESS", "").strip() not in ("", "0", "false")


def _save_baseline(flat, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_pickle for the object-dtype placeholders flatten_result emits for
    # None and for non-numerical members.
    np.savez_compressed(path, **flat)
    return path


@pytest.fixture(scope="module")
def analysis():
    """The default (FFT) reference analysis, run once per module."""
    expt, result = run_reference_analysis()
    return expt, result, flatten_result(result)


@pytest.mark.parametrize("method", sorted(BASELINES))
def test_baseline_matches(method):
    """Every pinned field of the analysis result is unchanged, per method."""
    _, result = run_reference_analysis(spectrum_method=method)
    flat = flatten_result(result)
    baseline = BASELINES[method]

    if _blessing() or not baseline.exists():
        if not baseline.exists() and not _blessing():
            pytest.fail(
                f"No baseline at {baseline}.\n"
                f"Create it once from known-good code:\n"
                f"  MBR_GOLDEN_BLESS=1 pixi run python -m pytest {__file__}"
            )
        _save_baseline(flat, baseline)
        pytest.skip(f"baseline written to {baseline} ({len(flat)} fields); re-run to compare")

    with np.load(baseline, allow_pickle=True) as handle:
        expected = {k: handle[k] for k in handle.files}

    new = set(flat) - set(expected)
    gone = set(expected) - set(flat)
    assert not gone, f"fields disappeared from the analysis result: {sorted(gone)}"
    assert not new, f"fields appeared without re-blessing: {sorted(new)}"

    mismatched = []
    for path, want in sorted(expected.items()):
        got = flat[path]
        want_arr, got_arr = np.asarray(want), np.asarray(got)

        if want_arr.shape != got_arr.shape:
            mismatched.append(f"{path}: shape {want_arr.shape} -> {got_arr.shape}")
            continue

        if path in EXACT_PATHS or want_arr.dtype.kind in "OUSb":
            if not np.array_equal(want_arr, got_arr):
                mismatched.append(f"{path}: {want_arr!r} -> {got_arr!r}")
            continue

        if not np.allclose(want_arr, got_arr, rtol=RTOL, atol=ATOL, equal_nan=True):
            worst = np.nanmax(np.abs(np.asarray(got_arr, float) - np.asarray(want_arr, float)))
            mismatched.append(f"{path}: max abs deviation {worst:.3e}")

    assert not mismatched, "analysis output changed:\n  " + "\n  ".join(mismatched)


def test_analysis_needs_no_station_or_database(analysis):
    """The reference path resolves its hardware from the saved sources only.

    Guards spec section 2.2: if this ever reports the current station, the
    analysis has silently substituted today's timing for August's.
    """
    expt, _, _ = analysis
    assert expt.data.hardware.source == "saved program", expt.data.hardware.source
    assert not hasattr(expt, "_analysis_station") or expt._analysis_station is None


def test_floquet_timing_is_the_historical_value(analysis):
    """The cycle time used is August's, not today's.

    Today's configuration gives roughly half this value (spec section 2.2), so
    this single number distinguishes correct historical analysis from the
    station-fallback bug even if the baseline were regenerated carelessly.
    """
    _, result, _ = analysis
    assert result["hardware"]["floquet_cycle_us"] == pytest.approx(
        CHARACTERIZATION_TIMING["floquet_cycle_us"], rel=1e-15
    )


def test_source_mutation_is_pinned():
    """Records exactly which source fields aggregate analysis writes back.

    Spec section 2.5 wants aggregate reconstruction to treat loaded sources as
    immutable; today quadrature extraction attaches fields to them as a side
    effect, so reusing a leaf in two analyses depends on call order. Like the
    baseline above, this pins current behaviour rather than the target, so the
    section 2.5 fix has a precise thing to flip: when it lands, this expectation
    becomes ``set()`` and the assertion below is inverted.
    """
    from tests.mbr_reference import load_aggregate

    # Known side-effect fields, from the current quadrature-extraction path.
    EXPECTED_INJECTED = {"Pe", "return_quadrature"}

    expt = load_aggregate(CHARACTERIZATION_JOB_IDS)
    before = [set(src.data.keys()) for src in expt.batch_expts]
    expt.analyze(**CHARACTERIZATION_ANALYSIS)
    after = [set(src.data.keys()) for src in expt.batch_expts]

    injected = set().union(*(a - b for b, a in zip(before, after)))
    assert injected <= EXPECTED_INJECTED, (
        f"aggregate analysis writes back more than the known fields: "
        f"{sorted(injected - EXPECTED_INJECTED)}"
    )

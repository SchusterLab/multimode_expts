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


# --------------------------------------------------------------------------
# Complete-basis coverage: level statistics and the SFF
# --------------------------------------------------------------------------
#
# The August characterization set covers four occupations, so
# analyze_level_statistics, analyze_sff and merge_spectra all refuse to run on
# it and were extracted with no execution coverage at all. The N=3 July sector
# completes the 35-state fixed-N basis and does exercise them.
#
# Slower than the rest of this module (140 source files plus a Matrix-Pencil
# fit), hence its own marker: `-m "not slow"` skips it during quick iteration.

COMPLETE_BASIS_BASELINE = Path(__file__).parent / "data" / "mbr_complete_basis_20260723.npz"


@pytest.mark.slow
def test_complete_basis_baseline_matches():
    """Level statistics and SFF outputs are unchanged on the N=3 sector."""
    from tests.mbr_reference import run_complete_basis_analysis

    expt, data = run_complete_basis_analysis()
    assert data.spectrum.complete_basis, "N=3 sector no longer completes the basis"
    assert int(data.photon_number) == 3

    flat = {}
    flat.update(flatten_result(data, "spectrum_run"))
    flat.update(flatten_result(expt.analyze_level_statistics(data=data), "levels"))
    flat.update(flatten_result(expt.analyze_sff(data=data), "sff"))

    if _blessing() or not COMPLETE_BASIS_BASELINE.exists():
        if not COMPLETE_BASIS_BASELINE.exists() and not _blessing():
            pytest.fail(f"No baseline at {COMPLETE_BASIS_BASELINE}; bless it once.")
        _save_baseline(flat, COMPLETE_BASIS_BASELINE)
        pytest.skip(f"baseline written ({len(flat)} fields); re-run to compare")

    with np.load(COMPLETE_BASIS_BASELINE, allow_pickle=True) as handle:
        expected = {k: handle[k] for k in handle.files}

    assert not set(expected) - set(flat), "fields disappeared from the analysis result"
    mismatched = []
    for path, want in sorted(expected.items()):
        want_arr, got_arr = np.asarray(want), np.asarray(flat[path])
        if want_arr.shape != got_arr.shape:
            mismatched.append(f"{path}: shape {want_arr.shape} -> {got_arr.shape}")
        elif want_arr.dtype.kind in "OUSb":
            if not np.array_equal(want_arr, got_arr):
                mismatched.append(f"{path}: {want_arr!r} -> {got_arr!r}")
        elif not np.allclose(want_arr, got_arr, rtol=RTOL, atol=ATOL, equal_nan=True):
            mismatched.append(f"{path}: values differ")
    assert not mismatched, "analysis output changed:\n  " + "\n  ".join(mismatched)


@pytest.mark.slow
def test_timing_resolves_from_versioned_configs_not_a_constant():
    """Every source in the N=3 sector resolves its own historical timing.

    The July sector was taken under a different configuration than August
    (0.4135 us vs 0.7340 us per cycle), so loading it at all proves the section
    2.2 resolver works rather than a constant happening to fit.
    """
    from experiments.floquet_timing import resolve_floquet_timing
    from tests.mbr_reference import (COMPLETE_BASIS_SPECTROSCOPY_IDS, job_provenance,
                                     load_h5)
    from experiments.job_paths import resolve_job_paths

    provenance = job_provenance()
    ids = COMPLETE_BASIS_SPECTROSCOPY_IDS[:3]
    paths = resolve_job_paths(ids)
    for job_id in ids:
        cfg, _ = load_h5(paths[job_id])
        timing = resolve_floquet_timing(
            cfg, provenance[job_id]["floquet_storage_version_id"])
        assert timing["floquet_cycle_us"] == pytest.approx(0.41351877289377287, rel=1e-15)
        assert timing["source"].startswith("versioned config CFG-FL-")


def test_august_timing_reproduces_the_pickled_value():
    """The resolver reproduces the value the historical pickle held, exactly.

    Fast, and the single sharpest check that section 2.2 is correct: today's
    station gives roughly half this (gauss_sigma moved 0.04 -> 0.02 us).
    """
    from experiments.floquet_timing import resolve_floquet_timing
    from experiments.job_paths import resolve_job_path
    from tests.mbr_reference import load_h5

    cfg, _ = load_h5(resolve_job_path("JOB-20260815-00009"))
    timing = resolve_floquet_timing(cfg, "CFG-FL-20260814-00076")
    assert timing["floquet_cycle_us"] == 0.7340315934065934
    assert timing["m1s_pi_fracs"] == [40] * 7

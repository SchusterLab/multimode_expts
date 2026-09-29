"""The gap score and the fixed synthetic set (plan docs/qsim/pole_finding_explore.md, T0)."""
import numpy as np
import pytest

from fitting.qsim.poles.fixed_set import Condition, conditions, load_fits, load_set, make_case, save_fit, save_set
from fitting.qsim.poles.gap_score import gauge_shift, ordered_match, pooled_small_gap_fraction, score_gaps
from fitting.qsim.poles.pole_fit import PoleFit

LEVELS = np.array([0., 1., 1.2, 3., 4.5, 5., 7.])       # small gaps: 1-1.2 and 4.5-5 (mean 7/6)
SAMPLING = 100.


def test_exact_poles_find_every_gap():
    score = score_gaps(LEVELS[::-1], LEVELS, SAMPLING, {"real": np.full(6, 0.01)})
    assert score.gap_found.all() and score.false_poles == 0
    np.testing.assert_allclose(score.gap_errors_MHz, 0, atol=1e-12)
    summary = score.summary()
    assert summary["small_gaps"] == summary["small_found"] == 2
    assert summary["resolvable_real"] == 6 and summary["P_true"] == summary["P_found"]


def test_common_shift_is_removed():
    score = score_gaps(LEVELS + 0.07, LEVELS, SAMPLING, max_shift_MHz=0.2)
    assert score.shift_MHz == pytest.approx(-0.07, abs=0.03) and score.gap_found.all()


def test_gauge_shift_puts_found_levels_on_the_true_gauge():
    """Rows see E + delta_b; a fit with offsets delta - s has levels E + s."""
    true_offsets = np.array([0.01, -0.03, 0.05])
    shift = gauge_shift(true_offsets - 0.02, true_offsets)
    score = score_gaps(LEVELS + 0.02, LEVELS, SAMPLING, shift_MHz=shift)
    assert shift == pytest.approx(-0.02) and score.gap_found.all()
    np.testing.assert_allclose(score.found_MHz, LEVELS, atol=1e-12)
    assert gauge_shift(None, true_offsets) == pytest.approx(-0.01)


def test_merged_pair_loses_its_gap_and_one_neighbour_gap():
    """The merged pole matches one of the pair; the gap on the other side is lost."""
    found = np.r_[LEVELS[:1], 1.1, LEVELS[3:]]
    score = score_gaps(found, LEVELS, SAMPLING, max_shift_MHz=0.)
    assert list(score.gap_found[:3]) in ([True, False, False], [False, False, True])
    assert all(score.gap_found[3:]) and score.summary()["small_found"] == 1


def test_false_pole_between_two_levels_breaks_their_gap():
    score = score_gaps(np.r_[LEVELS, 2.0], LEVELS, SAMPLING, max_shift_MHz=0.)
    assert score.false_poles == 1
    assert list(score.gap_found) == [True, True, False, True, True, True]


def test_ordered_match_is_monotone_and_within_tolerance():
    level_pole = ordered_match(np.array([0.05, 0.9, 1.25, 9.]), LEVELS, np.full(7, 0.1))
    assert list(level_pole) == [0, 1, 2, -1, -1, -1, -1]


def test_found_gaps_are_the_found_poles_differences():
    found = LEVELS + np.array([0, 0.01, -0.01, 0, 0, 0.02, 0])
    score = score_gaps(found, LEVELS, SAMPLING, {"real": np.full(6, 0.01)}, max_shift_MHz=0.)
    np.testing.assert_allclose(score.gap_errors_MHz, np.diff(found - LEVELS), atol=1e-12)
    np.testing.assert_allclose(score.gap_z("real"), np.diff(found - LEVELS) / 0.01, atol=1e-9)


def test_pooled_fraction_pools_the_level_sets():
    scores = [score_gaps(LEVELS, LEVELS, SAMPLING), score_gaps(LEVELS[:-1], LEVELS, SAMPLING)]
    true, found = pooled_small_gap_fraction(scores)
    assert true == pytest.approx(pooled_small_gap_fraction(scores[:1])[0])
    assert 0 <= found <= 1


def test_set_has_80_distinct_cases():
    keys = [condition.key for condition in conditions()]
    assert len(keys) == len(set(keys)) == 80


def test_set_and_fits_round_trip(tmp_path):
    condition = Condition(point="august", decay_per_us=0.005, offset_sigma_MHz=0.5e-3, partial_rows=10, draw=0)
    case = make_case(condition)
    assert case.A.shape == (10, 300) and len(case.levels_MHz) == 35
    assert set(case.gap_bounds_MHz) == {"complex", "real"}
    assert np.all(case.gap_bounds_MHz["real"] <= case.gap_bounds_MHz["complex"] * (1 + 1e-6))
    loaded = load_set(save_set(tmp_path / "set.h5", [case]))[0]
    assert loaded.condition == condition
    np.testing.assert_array_equal(loaded.A, case.A)
    np.testing.assert_array_equal(loaded.gap_bounds_MHz["real"], case.gap_bounds_MHz["real"])

    fit = PoleFit(case.levels_MHz, np.full(35, 0.005), np.ones((10, 35)), 35, row_offsets_MHz=case.offsets_MHz)
    save_fit(tmp_path / "fits.h5", condition.key, fit, 1.5, condition)
    stored, seconds = load_fits(tmp_path / "fits.h5")[condition.key]
    assert seconds == 1.5 and stored.frequency_errors_MHz is None
    np.testing.assert_array_equal(stored.row_offsets_MHz, case.offsets_MHz)
    assert score_gaps(stored.frequencies_MHz, loaded.levels_MHz, loaded.sampling_MHz).gap_found.all()
    assert load_fits(tmp_path / "missing.h5") == {}

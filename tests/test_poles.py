"""The pole-finding building blocks (docs/qsim/pole_finding.md): generator, fitters, matching,
statistic. Benchmark 1 is ``tests/test_pole_bench_ideal.py``.

The hand-made returns are the setup of ``tests/test_matrix_pencil_synthetic.py``: 64 points
at 1 us, frequencies in FFT bins, rows normalized to A(0) = 1.
"""
from math import comb

import numpy as np
import pytest

from experiments.job_paths import JobPathError, data_root
from fitting.qsim.mbr_disorder import disorder_direction
from fitting.qsim.poles import fft_peaks, joint_pencil, per_row_reconciled
from fitting.qsim.poles.bench_io import load_scores, save_results
from fitting.qsim.poles.benchmarks import FITTERS, run_ideal_bench, run_nonideal_bench
from fitting.qsim.poles.matching import level_tolerances, match_poles, resolved_levels
from fitting.qsim.poles.offsets import calibration_sigma, model_offset_sigma
from fitting.qsim.poles.rank import RankRule
from fitting.qsim.poles.registry import load_registry, manifest_path
from fitting.qsim.poles.statistics import small_gap_ratio_fraction
from fitting.qsim.poles.synthetic import (Hardware, Nonideal, distinct_levels, sample_phase_diagram,
                                          synthetic_returns)

SAMPLES = 64
TIME_US = np.arange(SAMPLES) * 1.0
BIN_MHz = 1. / SAMPLES
FREQUENCIES_BINS = np.array([-10., -3., 4., 12.])
DECAYS_PER_US = np.array([0.01, 0.02, 0.005, 0.015])
WEIGHTS = np.array([[.4, .3, .2, .1], [.1, .2, .3, .4], [.25, .25, .25, .25]])


def returns(frequencies_bins, decays_per_us, weights, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    exponent = -np.asarray(decays_per_us)[:, None] - 2j * np.pi * BIN_MHz * np.asarray(frequencies_bins)[:, None]
    A = np.asarray(weights) @ np.exp(exponent * TIME_US[None, :])
    return A + noise * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape))


# --- Fitter B -------------------------------------------------------------

def test_joint_pencil_fits_noise_free_returns_exactly():
    fit = joint_pencil.fit(returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS, noise=0.), TIME_US)
    assert fit.rank == 4
    np.testing.assert_allclose(fit.frequencies_MHz / BIN_MHz, FREQUENCIES_BINS, atol=1e-9)
    np.testing.assert_allclose(fit.decays_per_us, DECAYS_PER_US, atol=1e-9)
    np.testing.assert_allclose(fit.amplitudes, WEIGHTS, atol=1e-9)


@pytest.mark.parametrize("rule", ["mdl", "threshold"])
def test_both_rank_rules_find_four_poles_at_one_percent_noise(rule):
    settings = joint_pencil.JointPencilSettings(rank_rule=RankRule(rule=rule))
    fit = joint_pencil.fit(returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS), TIME_US, settings)
    assert fit.rank == 4
    np.testing.assert_allclose(fit.frequencies_MHz / BIN_MHz, FREQUENCIES_BINS, atol=0.01)


def test_a_pole_past_nyquist_comes_back_as_its_alias():
    fit = joint_pencil.fit(returns([-10., -3., 4., 40.], DECAYS_PER_US, WEIGHTS), TIME_US)
    np.testing.assert_allclose(fit.frequencies_MHz / BIN_MHz, [-24., -10., -3., 4.], atol=0.01)


def test_joint_pencil_resolves_a_pair_half_a_bin_apart():
    """Both poles in every seed; at most one extra (noise) pole over all 20 seeds.
    The current code resolves this in 0 of 20 (docs/log/2026-09-28_pole-finding-design.md)."""
    extra_poles = 0
    for seed in range(20):
        fit = joint_pencil.fit(returns([0., 0.5], [.01, .01], [[.5, .5], [.6, .4]], seed=seed), TIME_US)
        match = match_poles(fit.frequencies_MHz, [0., 0.5 * BIN_MHz], 0.05 * BIN_MHz, 1.)
        assert len(match.missed_levels) == 0
        extra_poles += len(match.false_poles)
    assert extra_poles <= 1


# --- Fitters A and E ------------------------------------------------------

def test_fitter_a_is_the_current_code_on_well_separated_poles():
    fit = per_row_reconciled.fit(returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS), TIME_US)
    np.testing.assert_allclose(fit.frequencies_MHz / BIN_MHz, FREQUENCIES_BINS, atol=0.02)
    np.testing.assert_allclose(fit.weights, WEIGHTS.sum(axis=0), atol=0.03)
    assert per_row_reconciled.CAMPAIGN_SETTINGS.requested_max_modes == comb(7, 3)


def test_fft_peaks_find_well_separated_levels_to_a_tenth_of_a_bin():
    fit = fft_peaks.fit(returns(FREQUENCIES_BINS, [0.] * 4, WEIGHTS, noise=0.), TIME_US)
    np.testing.assert_allclose(fit.frequencies_MHz / BIN_MHz, FREQUENCIES_BINS, atol=0.1)


# --- Generator ------------------------------------------------------------

def test_a_disorder_draw_is_a_zero_mean_unit_vector():
    point = sample_phase_diagram([-3.], [3.], draws=2, seed=5)[1]
    np.testing.assert_allclose(point.direction, disorder_direction(6, 4))
    assert abs(point.direction.sum()) < 1e-12 and np.linalg.norm(point.direction) == pytest.approx(1.)


def test_complete_basis_weights_are_the_multiplicities():
    for point in sample_phase_diagram([0., -3.], [0., 3.], draws=1):
        case = synthetic_returns(point)
        truth = case.truth
        np.testing.assert_allclose(case.A[:, 0], 1.)
        np.testing.assert_allclose(truth.row_weights.sum(axis=0), truth.multiplicities, atol=1e-12)
        assert truth.multiplicities.sum() == comb(7, 3) == len(case.A)


def test_no_disorder_and_no_kerr_gives_the_degenerate_ladder():
    """At K = delta = 0 the model is linear: 7 distinct levels for N = 3 on a 4-leaf star."""
    truth = synthetic_returns(sample_phase_diagram([0.], [0.], 1)[0]).truth
    assert len(truth.levels_MHz) == 7


def test_partial_rows_are_chosen_as_the_campaign_chose_them():
    case = synthetic_returns(sample_phase_diagram([-3.], [3.], 1)[0], Hardware(partial_rows=10))
    assert case.A.shape == (10, 100)
    assert np.all(case.truth.row_weights.sum(axis=0) <= case.truth.multiplicities + 1e-12)


def test_nonidealities_are_reproducible_by_seed():
    point = sample_phase_diagram([-3.], [3.], 1)[0]
    nonideal = Nonideal(snr=30, decay_per_us=0.01, offset_sigma_MHz=1e-3, seed=4)
    np.testing.assert_array_equal(synthetic_returns(point, nonideal=nonideal).A,
                                  synthetic_returns(point, nonideal=nonideal).A)


def test_distinct_levels_sum_the_weights_of_a_multiplet():
    levels, multiplicities, weights = distinct_levels(np.array([1., 0., 1. + 1e-9]),
                                                      np.array([[.2, .5, .3]]), 1e-6)
    np.testing.assert_allclose(levels, [0., 1. + 5e-10])
    np.testing.assert_array_equal(multiplicities, [1, 2])
    np.testing.assert_allclose(weights, [[.5, .5]])


# --- Matching and statistic -----------------------------------------------

def test_matching_is_one_to_one_on_the_alias_circle():
    match = match_poles(found_MHz=[-0.49, 0.1, 0.12, 0.3], levels_MHz=[0.49, 0.11, -0.2],
                        tolerance_MHz=0.03, sampling_MHz=1.)
    assert sorted(zip(match.pole_index, match.level_index)) == [(0, 0), (1, 1)]
    np.testing.assert_allclose(match.errors_MHz[match.pole_index == 0], [0.02])
    np.testing.assert_array_equal(match.missed_levels, [2])
    np.testing.assert_array_equal(match.false_poles, [2, 3])


def test_a_level_tolerance_is_capped_by_a_quarter_of_its_separation():
    np.testing.assert_allclose(level_tolerances(np.array([0., 0.4, 3.]), bin_MHz=1.), [0.1, 0.1, 0.25])


def test_a_merged_pair_counts_as_unresolved():
    levels = np.array([0., 0.1, 1.])
    match = match_poles([0.05, 1.], levels, tolerance_MHz=0.06, sampling_MHz=10.)
    np.testing.assert_array_equal(resolved_levels(match, levels), [False, False, False])
    match = match_poles([0., 0.1, 1.], levels, tolerance_MHz=0.06, sampling_MHz=10.)
    np.testing.assert_array_equal(resolved_levels(match, levels), [True, True, True])


def test_small_gap_ratio_fraction_pools_level_sets():
    # gaps 1, 0.1, 1 -> r = 0.1, 0.1; gaps 1, 1 -> r = 1.
    assert small_gap_ratio_fraction([[0., 1., 1.1, 2.1], [0., 1., 2.]], r0=0.25, edge_fraction=0.) == pytest.approx(2 / 3)


# --- Registry -------------------------------------------------------------

def test_the_registry_loads_and_its_manifests_exist():
    data_sets = load_registry()
    assert len({data_set.label for data_set in data_sets}) == len(data_sets)
    try:
        root = data_root()
    except JobPathError:
        pytest.skip("no data tree on this machine")
    assert all(manifest_path(data_set, root).is_file() for data_set in data_sets)


# --- Saved results --------------------------------------------------------

def test_bench_results_save_and_load(tmp_path):
    points = sample_phase_diagram([-3.], [3.], 1)
    fitters = {"B": FITTERS["B"]}
    results = [run_ideal_bench(points, fitters=fitters, workers=1),
               run_nonideal_bench(points, [Nonideal(snr=100)], 1, fitters=fitters, workers=1),
               run_nonideal_bench(points, [Nonideal(snr=100)], 1, fitters=fitters, name="window_100", workers=1)]
    path = save_results(tmp_path / "results.h5", results, note="test")
    loaded = load_scores(path, "window_100")
    assert list(loaded.fitter) == ["B"] and loaded.levels[0] == 35
    with pytest.raises(ValueError, match="distinct names"):
        save_results(tmp_path / "again.h5", results[1:2] * 2)


def test_parallel_fits_agree_with_serial_fits():
    """Workers use one BLAS thread, so results agree only up to rounding."""
    points = sample_phase_diagram([-3.], [1., 3.3], 1)
    args = (points, [Nonideal(snr=100)], 1)
    serial = run_nonideal_bench(*args, workers=1)
    parallel = run_nonideal_bench(*args, workers=2)  # B, E and A on 2 cases: seconds, low load
    for a, b in zip(serial.scores, parallel.scores):
        assert a.fitter == b.fitter
        np.testing.assert_allclose(a.found_MHz, b.found_MHz, atol=1e-9)
        np.testing.assert_array_equal(a.level_resolved, b.level_resolved)


# --- Row offsets (spec 6) -------------------------------------------------

def test_calibration_sigma_is_the_phase_slope_error_as_a_frequency():
    # 0.36 deg per 1 us cycle is 1e-3 of a turn per us: 1 kHz.
    np.testing.assert_allclose(calibration_sigma([0.36, -0.72], cycle_us=1.), [1e-3, 2e-3])


def test_model_offsets_recover_injected_row_offsets():
    point = sample_phase_diagram([-1.22], [5.8], 1)[0]
    hardware = Hardware(coupling_MHz=8.615e-3, dt_us=1.4509, samples=300, partial_rows=10)
    case = synthetic_returns(point, hardware, Nonideal(snr=100, decay_per_us=0.01, offset_sigma_MHz=1e-3, seed=3))
    injected = np.random.default_rng(3).normal(0, 1e-3, size=10)  # add_nonidealities' first draw
    clean = synthetic_returns(point, hardware)
    model = clean.A  # the model's returns on the same rows, no decay, no offsets
    sigma, offsets = model_offset_sigma(case.A, model, case.time_us)
    np.testing.assert_allclose(offsets, injected, atol=5e-5)
    assert sigma == pytest.approx(np.std(injected), rel=0.05)

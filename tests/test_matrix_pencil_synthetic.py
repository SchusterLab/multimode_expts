"""Matrix Pencil on synthetic returns with known poles and weights.

Each row is ``A_i(t) = sum_m W_im exp((-g_m - 2 pi i f_m) t)`` plus complex
Gaussian noise. Rows of ``W`` sum to 1, so ``A_i(0) = 1`` and the weights
Matrix Pencil reports are ``W`` itself. The time grid is 64 points at 1 us
(FFT bin 1/64 MHz, Nyquist 32 bins), in frequency units of FFT bins.

The ``xfail(strict=True)`` tests are known defects of the current algorithm,
found by these tests (docs/log/2026-09-27_matrix-pencil.md). They pass once
the defect is fixed; then remove the mark.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from slab import AttrDict

from fitting.qsim.matrix_pencil import MatrixPencilSettings, analyze_matrix_pencil, refit_row

SAMPLES = 64
TIME_US = np.arange(SAMPLES) * 1.0
BIN_MHz = 1. / SAMPLES

# Four poles, distinct decays, three rows of different weights.
FREQUENCIES_BINS = np.array([-10., -3., 4., 12.])
DECAYS_PER_US = np.array([0.01, 0.02, 0.005, 0.015])
WEIGHTS = np.array([[.4, .3, .2, .1],
                    [.1, .2, .3, .4],
                    [.25, .25, .25, .25]])


def returns(frequencies_bins, decays_per_us, weights, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    exponent = -np.asarray(decays_per_us)[:, None] - 2j * np.pi * BIN_MHz * np.asarray(frequencies_bins)[:, None]
    A = np.asarray(weights) @ np.exp(exponent * TIME_US[None, :])
    return A + noise * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape))


def fit(A, final_occupations=None, **settings):
    occupations = [(row,) for row in range(len(A))]
    reconstruction = AttrDict(dict(A=A, occupations=occupations,
                                   final_occupations=final_occupations or occupations))
    settings.setdefault("requested_max_modes", 30)
    return analyze_matrix_pencil(reconstruction, TIME_US, MatrixPencilSettings(**settings))


def found_bins(result):
    return result.modes.frequencies_MHz / BIN_MHz


@pytest.mark.parametrize("noise, frequency_bins, decay_per_us, weight", [
    (1e-4, 0.005, 1e-4, 0.002),
    (0.01, 0.02, 7e-4, 0.01),
    (0.05, 0.04, 2e-3, 0.03),
])
def test_recovers_the_poles_and_weights(noise, frequency_bins, decay_per_us, weight):
    result = fit(returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS, noise))
    np.testing.assert_allclose(found_bins(result), FREQUENCIES_BINS, atol=frequency_bins)
    np.testing.assert_allclose(result.modes.decay_per_us, DECAYS_PER_US, atol=decay_per_us)
    np.testing.assert_allclose(result.modes.local_weights, WEIGHTS, atol=weight)
    np.testing.assert_allclose(result.modes.DOS_weights, WEIGHTS.sum(axis=0), atol=2 * weight)
    np.testing.assert_array_equal(result.modes.supporting_row_counts, 3)


def test_a_pole_past_nyquist_comes_back_as_its_alias():
    frequencies_bins = np.array([-10., -3., 4., 40.])
    result = fit(returns(frequencies_bins, DECAYS_PER_US, WEIGHTS))
    np.testing.assert_allclose(found_bins(result), [-24., -10., -3., 4.], atol=0.03)


def test_off_diagonal_rows_are_weighted_as_measured():
    """Diagonal rows are divided by A(0); an off-diagonal row is not."""
    A = 0.3 * returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS)
    result = fit(A, final_occupations=[(9,), (1,), (2,)])
    expected = np.vstack([0.3 * WEIGHTS[0], WEIGHTS[1:]])
    np.testing.assert_allclose(result.modes.local_weights, expected, atol=0.01)
    np.testing.assert_allclose(result.row_normalization, [1., *A[1:, 0]])


# A third pole, weight 0.1, only in the last row.
WEAK_FREQUENCIES_BINS = np.array([-5., 5., 15.])
WEAK_WEIGHTS = np.array([[.5, .5, 0.], [.5, .5, 0.], [.45, .45, .1]])


def test_a_pole_in_one_row_is_kept_with_one_supporting_row():
    result = fit(returns(WEAK_FREQUENCIES_BINS, [.01] * 3, WEAK_WEIGHTS))
    np.testing.assert_allclose(found_bins(result), WEAK_FREQUENCIES_BINS, atol=0.02)
    np.testing.assert_array_equal(result.modes.supporting_row_counts, [3, 3, 1])
    assert result.modes.local_weights[2, 2] == pytest.approx(0.1, abs=0.01)


def test_a_pole_in_one_row_is_rejected_with_two_supporting_rows():
    result = fit(returns(WEAK_FREQUENCIES_BINS, [.01] * 3, WEAK_WEIGHTS), minimum_supporting_rows=2)
    np.testing.assert_allclose(found_bins(result), WEAK_FREQUENCIES_BINS[:2], atol=0.02)
    assert len(result.candidates.rejected_clusters) == 1


def test_a_row_refit_uses_only_the_poles_that_row_found():
    A = returns(WEAK_FREQUENCIES_BINS, [.01] * 3, WEAK_WEIGHTS)
    result = fit(A)
    np.testing.assert_allclose(refit_row(result, A[0], 0).frequencies_MHz / BIN_MHz,
                               WEAK_FREQUENCIES_BINS[:2], atol=0.02)
    last = refit_row(result, A[2], 2)
    np.testing.assert_allclose(last.frequencies_MHz / BIN_MHz, WEAK_FREQUENCIES_BINS, atol=0.02)
    np.testing.assert_allclose(last.normalized_amplitudes.real, WEAK_WEIGHTS[2], atol=0.01)


PAIR_WEIGHTS = np.array([[.5, .5], [.6, .4]])


def test_a_pair_two_bins_apart_is_resolved():
    result = fit(returns([0., 2.], [.01, .01], PAIR_WEIGHTS))
    np.testing.assert_allclose(found_bins(result), [0., 2.], atol=0.02)


def test_a_growing_pole_is_found_when_growth_is_not_clipped():
    decays_per_us = np.array([0.01, -0.01, 0.005, 0.015])
    result = fit(returns(FREQUENCIES_BINS, decays_per_us, WEIGHTS), clip_growth=False)
    np.testing.assert_allclose(result.modes.decay_per_us, decays_per_us, atol=1e-3)
    np.testing.assert_allclose(result.modes.local_weights, WEIGHTS, atol=0.02)


# --- Known defects --------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="rank sweep stops at the numerical rank, which on noise-free "
                   "data is the true pole count, so no pole persists minimum_consecutive_ranks ranks")
def test_noise_free_returns_are_fit_exactly():
    result = fit(returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS, noise=0.))
    np.testing.assert_allclose(found_bins(result), FREQUENCIES_BINS, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="requested_max_modes caps the rank sweep as well as the "
                   "number of poles kept; at the true pole count, late poles cannot persist")
def test_asking_for_exactly_the_true_pole_count_finds_them_all():
    result = fit(returns(FREQUENCIES_BINS, DECAYS_PER_US, WEIGHTS), requested_max_modes=4)
    np.testing.assert_allclose(found_bins(result), FREQUENCIES_BINS, atol=0.02)


@pytest.mark.xfail(strict=True, reason="the default 1.5-bin tracking and dedup tolerances merge "
                   "poles closer than that, although the pencil resolves them")
def test_a_pair_one_bin_apart_is_resolved_with_the_defaults():
    result = fit(returns([0., 1.], [.01, .01], PAIR_WEIGHTS))
    np.testing.assert_allclose(found_bins(result), [0., 1.], atol=0.05)


# --- Settings -------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(track_frequency_tolerance_bin=1.0),   # misspelled
    dict(track_frequency_tolerance_bins=0.),
    dict(track_frequency_tolerance_bins=np.inf),
    dict(minimum_consecutive_ranks=2.5),
    dict(minimum_pole_radius=1.1),              # above the maximum
])
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValidationError):
        MatrixPencilSettings(**bad)

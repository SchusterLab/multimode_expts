"""The per-level diagnosis (fitting.qsim.poles.diagnosis, spec 5.5) on small synthetic cases."""
import numpy as np

from fitting.qsim.poles import joint_pencil
from fitting.qsim.poles.diagnosis import (DiagnosisSettings, cause_counts, diagnose_levels, oracle_starts,
                                          residual_excess, row_noise, seen_frequencies)
from fitting.qsim.poles.resolution import clusters, cramer_rao_bounds
from fitting.qsim.poles.pole_fit import PoleFit
from fitting.qsim.poles.real_benchmarks import RealSpectrum

SAMPLES = 64
TIME_US = np.arange(SAMPLES) * 1.0
BIN_MHz = 1. / SAMPLES
LEVELS_MHz = BIN_MHz * np.array([-10., -3., 4., 12.])
WEIGHTS = np.array([[.4, .3, .2, .1], [.1, .2, .3, .4], [.25, .25, .25, .25]])


def rows(weights=WEIGHTS, levels_MHz=LEVELS_MHz, noise=0.01, decay_per_us=0.01, seed=0):
    rng = np.random.default_rng(seed)
    A = weights @ np.exp(np.outer(-decay_per_us - 2j * np.pi * levels_MHz, TIME_US))
    return A + noise / np.sqrt(2) * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape))


def spectrum(A, weights=WEIGHTS, levels_MHz=LEVELS_MHz):
    return RealSpectrum("toy", "toy", False, TIME_US, A, levels_MHz, np.ones(len(levels_MHz)), weights)


def full_fisher(levels_MHz, weights, decay, noise, prior_MHz, step=1e-7):
    """Finite-difference Fisher matrix over every parameter, amplitudes included."""
    M, R = len(levels_MHz), len(weights)

    def model(p):
        E, gamma, delta = p[:M], p[M:2 * M], p[2 * M:2 * M + R]
        c = (p[2 * M + R:2 * M + R + M * R] + 1j * p[2 * M + R + M * R:]).reshape(R, M)
        a = c @ np.exp(np.outer(-gamma - 2j * np.pi * E, TIME_US))
        return (a * np.exp(-2j * np.pi * np.outer(delta, TIME_US)) / noise[:, None]).ravel()

    p0 = np.concatenate([levels_MHz, np.full(M, decay), np.zeros(R), weights.ravel(), np.zeros(M * R)])
    J = np.stack([(model(p0 + step * e) - model(p0 - step * e)) / (2 * step) for e in np.eye(len(p0))], axis=1)
    F = 2 * np.real(J.conj().T @ J)
    F[2 * M:2 * M + R, 2 * M:2 * M + R] += np.eye(R) / prior_MHz ** 2
    return F


def test_cramer_rao_bound_matches_the_full_fisher_matrix():
    noise = np.array([0.01, 0.02, 0.015])
    crb, _ = cramer_rao_bounds(TIME_US, LEVELS_MHz, WEIGHTS, 0.01, noise, 0.1 * BIN_MHz, np.inf)
    covariance = np.linalg.inv(full_fisher(LEVELS_MHz, WEIGHTS, 0.01, noise, 0.1 * BIN_MHz))
    np.testing.assert_allclose(crb, np.sqrt(np.diag(covariance)[:4]), rtol=1e-4)


def test_row_noise_is_the_injected_noise():
    A = np.vstack([rows(noise=0.01, seed=s) for s in range(20)])
    noise = row_noise(A / A[:, :1], 1.0, LEVELS_MHz, 4.)
    assert abs(np.mean(noise) / 0.01 - 1) < 0.1


def test_a_good_fit_leaves_only_noise_in_band():
    A = rows()
    s = spectrum(A)
    assert np.all(residual_excess(joint_pencil.fit(A, TIME_US), s) < 2)
    missing_one = joint_pencil.fit(A, TIME_US)
    keep = np.arange(4) != 2
    missing_one = PoleFit(missing_one.frequencies_MHz[keep], missing_one.decays_per_us[keep],
                          missing_one.amplitudes[:, keep], 3)
    assert np.all(residual_excess(missing_one, s) > 10)


def test_each_missed_level_gets_its_cause():
    """Level 0 not in the rows, level 1 found, level 2 missed by the fitter, 3 and 4 too close."""
    levels = BIN_MHz * np.array([-10., -3., 4., 12., 12.02])
    weights = np.array([[0., .3, .3, .2, .2], [0., .2, .3, .3, .2], [0., .25, .25, .25, .25]])
    A = rows(weights, levels, noise=0.01)
    fit = PoleFit(levels[[1, 3]], np.zeros(2), np.ones((3, 2)), 2)
    table, _ = diagnose_levels(spectrum(A, weights, levels), {"X": fit}, DiagnosisSettings())
    assert list(table.cause_X) == ["not in rows", "found", "search", "found", "unresolvable"]
    assert cause_counts(table, ["X"]).loc["found", "X"] == 2


def test_clusters_join_levels_across_small_gaps():
    assert [list(g) for g in clusters([10., 1., 2., 10.], 4.)] == [[0], [1, 2, 3], [4]]


def test_oracle_merges_start_poles_only_past_the_condition_limit():
    groups, centers = oracle_starts(LEVELS_MHz, np.ones(4), TIME_US, 0.01)
    assert groups == [[0], [1], [2], [3]]
    levels = BIN_MHz * np.array([-10., 4., 4. + 1e-9, 12.])
    groups, centers = oracle_starts(levels, np.array([1., 1., 3., 1.]), TIME_US, 0.01)
    assert groups == [[0], [1, 2], [3]]
    np.testing.assert_allclose(centers[1], BIN_MHz * (4. + 0.75e-9))


def test_seen_frequencies_move_with_the_rows_that_hold_the_pole():
    fit = PoleFit(np.array([0.1, 0.2]), np.zeros(2), np.array([[1., 0.], [0., 1.]]), 2,
                  row_offsets_MHz=np.array([0.001, -0.002]))
    np.testing.assert_allclose(seen_frequencies(fit), [0.101, 0.198])

"""Fitter T3 (``fitting.qsim.poles.sparse_fit``): the convex sparse fit on a fine grid."""
import numpy as np
import pytest

from fitting.qsim.poles import sparse_fit
from fitting.qsim.poles.joint_refined import remove_offsets
from fitting.qsim.poles.pole_fit import PoleFit

DT_US, SAMPLES, DECAY = 1.0, 200, 0.01
LEVELS = np.array([-0.060, -0.030, -0.0285, 0.010, 0.045])   # MHz; bin 5 kHz, one pair 0.3 bin apart


def rows(offsets, noise=0.01, seed=0):
    """-> (A, time_us, weights): 4 diagonal rows, c >= 0 summing to 1, white noise."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(LEVELS)), size=4)
    time_us = DT_US * np.arange(SAMPLES)
    V = np.exp(np.outer(-DECAY - 2j * np.pi * LEVELS, time_us))
    A = (weights @ V) * np.exp(-2j * np.pi * np.outer(offsets, time_us))
    A += noise * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape)) / np.sqrt(2)
    return A, time_us, weights


def start_fit(offsets):
    """-> a start PoleFit as C would give: the pair merged, the offsets as given."""
    E = np.array([-0.060, -0.029, 0.010, 0.045])
    return PoleFit(E, np.full(4, DECAY), np.zeros((4, 4)), 4, None, np.asarray(offsets, dtype=float))


def test_recovers_levels_with_true_offsets():
    offsets = np.array([0.3e-3, -0.2e-3, 0.1e-3, 0.])
    A, time_us, _ = rows(offsets)
    fit = sparse_fit.fit(A, time_us, sparse_fit.SparseFitSettings(), start=start_fit(offsets))
    assert len(fit.frequencies_MHz) == len(LEVELS)                       # the merged pair is split
    assert np.max(np.abs(np.sort(fit.frequencies_MHz) - LEVELS)) < 0.2e-3
    assert np.all(fit.amplitudes >= 0)
    np.testing.assert_allclose(fit.row_offsets_MHz, offsets)


def test_working_set_meets_the_optimality_test():
    A, time_us, _ = rows(np.zeros(4))
    a = A / A[:, :1]
    settings = sparse_fit.SparseFitSettings()
    grid = sparse_fit.make_grid(start_fit(np.zeros(4)), SAMPLES, DT_US, settings)
    solution = sparse_fit.sparse_amplitudes(a, np.full(4, 0.01), time_us, grid, settings, LEVELS[:2])
    assert solution.worst_kkt <= 1 + settings.kkt_tolerance
    assert np.all(solution.amplitudes >= 0)
    assert np.all(solution.amplitudes[:, :, ~grid.band] == 0)


@pytest.mark.parametrize("penalty", ["group", "l1"])
def test_large_lambda_gives_no_grid_points(penalty):
    A, time_us, _ = rows(np.zeros(4))
    settings = sparse_fit.SparseFitSettings(lambda_chi2=1e12, penalty=penalty)
    grid = sparse_fit.make_grid(start_fit(np.zeros(4)), SAMPLES, DT_US, settings)
    solution = sparse_fit.sparse_amplitudes(A / A[:, :1], np.full(4, 0.01), time_us, grid, settings, LEVELS)
    assert solution.working_set == 0 and np.all(solution.amplitudes == 0)


def test_offset_scan_finds_a_row_offset():
    offsets = np.array([0., 0.8e-3, 0., 0.])
    A, time_us, _ = rows(offsets)
    data = (A / A[:, :1], time_us, np.arange(4), np.full(4, 0.01))
    found = sparse_fit.fit_offsets(data, LEVELS, np.full(len(LEVELS), DECAY), np.zeros(4),
                                   sparse_fit.SparseFitSettings())
    np.testing.assert_allclose(found, offsets, atol=0.05e-3)


def test_cluster_and_split():
    assert [len(part) for part in sparse_fit.split_run(np.array([1., 2., 0.1, 2., 1.]), 0.5)] == [2, 3]
    assert len(sparse_fit.split_run(np.array([1., 2., 1.5, 2., 1.]), 0.5)) == 1


def test_merge_undoes_a_split_level():
    A, time_us, _ = rows(np.zeros(4), seed=1)
    data = (A / A[:, :1], time_us, np.arange(4), np.full(4, 0.01))
    split = np.r_[LEVELS[:3], 0.0095, 0.0105, LEVELS[4:]]            # 0.010 as two spikes
    E, _ = sparse_fit.merge_and_drop(data, split, np.full(len(split), DECAY), np.zeros(4),
                                     sparse_fit.SparseFitSettings())
    assert len(E) == len(LEVELS)
    assert np.max(np.abs(np.sort(E) - LEVELS)) < 0.2e-3


def test_remove_offsets_convention():
    """The rows carry exp(-2 pi i delta t); the fit removes it with remove_offsets."""
    A, time_us, _ = rows(np.array([1e-3, 0., 0., 0.]), noise=0.)
    clean, _, _ = rows(np.zeros(4), noise=0.)
    np.testing.assert_allclose(remove_offsets(A, time_us, np.array([1e-3, 0., 0., 0.])), clean, atol=1e-12)

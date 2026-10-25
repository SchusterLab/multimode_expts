"""Shared helpers for the level-statistics lab book (2026-10-25 night, Claude).

Model: the central boson in one N sector (``theory.qsim.central_boson.fixed_n_hamiltonian``),
M1 + 4 storage modes, equal couplings g, disorder W = RMS of the storage onsite energies.
All energies in units of g unless a name says MHz/kHz.
"""
from pathlib import Path

import numpy as np

from theory.qsim.central_boson import fixed_n_hamiltonian

VAULT = Path("G:/Shared drives/SLab/Multimode/Lab/guan/LevelStatistics")
FIGURES = VAULT / "figures"
CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

MODES = 5
LEAVES = MODES - 1
POISSON_MEAN_R = 2 * np.log(2) - 1
GOE_MEAN_R = 0.5307
EDGE_FRACTION = 0.10


def disorder_shape(rng):
    """-> a zero-mean storage onsite pattern with RMS 1 (as ``disorder_direction(..., 'rms')``)."""
    u = rng.normal(size=LEAVES)
    u -= u.mean()
    return u / np.sqrt(np.mean(u ** 2))


def spectrum(n, kerr, disorder, shape, g=1.0):
    """-> (levels, weights) of the N = n sector; kerr, disorder in units of g. weights[b, k] = |<b|k>|^2."""
    m = fixed_n_hamiltonian(n, MODES, -disorder * g * shape, [g] * LEAVES, kerr * g)
    return m.energies_MHz, m.basis_eigenstate_weights


def gap_ratios(levels, edge_fraction=EDGE_FRACTION, tol=1e-9):
    """-> bulk r_n = min/max of adjacent gaps; exact degeneracies (gap < tol) give r = 0."""
    levels = np.sort(levels)
    cut = int(np.ceil(edge_fraction * len(levels)))
    if cut:
        levels = levels[cut:-cut]
    s = np.diff(levels)
    s = np.where(s < tol, 0.0, s)
    lo, hi = np.minimum(s[:-1], s[1:]), np.maximum(s[:-1], s[1:])
    return np.where(hi > 0, lo / np.where(hi > 0, hi, 1.0), 0.0)


def bulk_spacings(levels, edge_fraction=EDGE_FRACTION):
    """-> the bulk adjacent gaps divided by their own mean (a crude per-realization unfolding)."""
    levels = np.sort(levels)
    cut = int(np.ceil(edge_fraction * len(levels)))
    s = np.diff(levels[cut:-cut] if cut else levels)
    return s / s.mean()


def goe_r_density(r):
    """Atas et al. surmise for GOE, P(r) on r in [0, 1] (folded)."""
    z = 27 / 8
    return 2 * z * (r + r ** 2) / (1 + r + r ** 2) ** 2.5


def poisson_r_density(r):
    return 2 / (1 + r) ** 2

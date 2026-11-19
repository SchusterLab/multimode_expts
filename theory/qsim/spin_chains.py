"""Reference spin models of Kim and Polkovnikov (arXiv:2308.09745), to validate the AGP tools.

c-XXZ (their Eq. 7): an open XXZ chain of sites 2..L coupled to one boundary spin (site 1),

    H = (1/2) sum_{i=2}^{L-1} [X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}]
        + (g/2) [X_1 X_2 + Y_1 Y_2 + Delta Z_1 Z_2] + h Z_1,

with Pauli matrices X, Y, Z. Integrable at g = 1 (XXZ with a boundary field) and g = 0 for all h.
dH/dh = Z_1 (the integrability-preserving direction at g = 1), dH/dg = the boundary bond
(breaking). H conserves m = sum_i Z_i; a sector is given by the number of up spins.

Spin i is bit i - 1 of the basis integer; bit 1 is up (Z = +1). Dimensionless energies, as in the
paper. Spectral sums follow the paper: only the central ``fraction`` of the eigenstates of a sector
are used as the state |n>, all eigenstates as |m>, and pairs m = n are left out.
"""
from itertools import combinations
from math import comb

import numpy as np

from slab import AttrDict


def sector_basis(length, ups):
    """-> sorted basis integers of ``length`` spins with ``ups`` spins up."""
    return np.array(sorted(sum(1 << i for i in c) for c in combinations(range(length), ups)), dtype=np.int64)


def _bond(basis, index, i, j, delta):
    """-> the matrix of (1/2)[X_i X_j + Y_i Y_j + delta Z_i Z_j] (spins counted from 0) in ``basis``."""
    out = np.zeros((len(basis), len(basis)))
    for k, state in enumerate(basis):
        zi = 1 if state >> i & 1 else -1
        zj = 1 if state >> j & 1 else -1
        out[k, k] += 0.5 * delta * zi * zj
        if zi != zj:   # (1/2)(XX + YY) = S+S- + S-S+ flips an anti-aligned pair with amplitude 1
            out[index[state ^ (1 << i | 1 << j)], k] += 1.0
    return out


def cxxz_operators(length, ups, delta=1.2):
    """-> AttrDict(bulk, boundary, field, basis) of the c-XXZ model in one magnetization sector.

    H = bulk + g * boundary + h * field; boundary = dH/dg and field = dH/dh = Z_1.
    """
    basis = sector_basis(length, ups)
    index = {int(state): k for k, state in enumerate(basis)}
    bulk = sum(_bond(basis, index, i, i + 1, delta) for i in range(1, length - 1))
    boundary = _bond(basis, index, 0, 1, delta)
    field = np.diag(np.where(basis & 1, 1.0, -1.0))
    return AttrDict(dict(bulk=bulk, boundary=boundary, field=field, basis=basis))


def cxxz_sectors(length):
    """-> the up-spin counts of the paper's sectors: m = 0, +-2 for even L, m = +-1 for odd L."""
    half = length // 2
    return [half - 1, half, half + 1] if length % 2 == 0 else [half, half + 1]


def largest_sector_dimension(length):
    """-> D_s = C(L, floor(L / 2)), the paper's cutoff unit: mu = alpha L / D_s."""
    return comb(length, length // 2)


def sector_spectrum(hamiltonian, operators, fraction=0.5):
    """-> AttrDict for one sector: ``omega`` (D, n) = E_m - E_n for the central states n,
    ``cross`` (k, k, D, n) = Re[<m|O_a|n><n|O_b|m>] with the pairs m = n set to 0,
    ``spacings`` of the central states, ``count`` n.
    """
    energies, states = np.linalg.eigh(hamiltonian)
    dim = len(energies)
    keep = int(round(fraction * dim))
    start = (dim - keep) // 2
    window = np.arange(start, start + keep)
    matrices = [states.T @ op @ states[:, window] for op in operators]
    cross = np.empty((len(operators), len(operators), dim, keep))
    for a, A in enumerate(matrices):
        for b, B in enumerate(matrices):
            cross[a, b] = np.real(A * B.conj())
    cross[:, :, window, np.arange(keep)] = 0.0   # m = n
    return AttrDict(dict(omega=energies[:, None] - energies[window][None, :], cross=cross,
                         spacings=np.diff(energies[window]), count=keep))


def cxxz_spectrum(length, h, g, delta=1.2, fraction=0.5, operators=None):
    """-> list of :func:`sector_spectrum` results over the paper's sectors, for the operators
    (field, boundary) = (dH/dh, dH/dg). ``operators`` caches :func:`cxxz_operators` per sector.
    """
    out = []
    for ups in cxxz_sectors(length):
        ops = operators[ups] if operators is not None else cxxz_operators(length, ups, delta)
        H = ops.bulk + g * ops.boundary + h * ops.field
        out.append(sector_spectrum(H, [ops.field, ops.boundary], fraction))
    return out


def pooled_sticks(sectors, direction):
    """-> (omega, weights): all sticks of all sectors for the direction n = (n_h, n_g), weights / (central states).

    Phi_n(omega) is these sticks broadened; sum over both signs of omega.
    """
    n = np.asarray(direction, dtype=float)
    total = sum(s.count for s in sectors)
    omega = np.concatenate([s.omega.ravel() for s in sectors])
    weights = np.concatenate([np.einsum("a,b,abij->ij", n, n, s.cross).ravel() for s in sectors]) / total
    return omega, weights


def qgt(sectors, cutoff):
    """-> (2, 2) regularized QGT g_ab = (1/N) sum_n sum_m omega^2 / (omega^2 + mu^2)^2 Re[...] (paper Eq. 4)."""
    total = sum(s.count for s in sectors)
    out = np.zeros((2, 2))
    for s in sectors:
        kernel = s.omega**2 / (s.omega**2 + cutoff**2)**2
        out += np.einsum("abij,ij->ab", s.cross, kernel)
    return out / total


def typical_spacing(sectors):
    """-> (typical, minimum) level spacing of the central states: the median and the smallest spacing."""
    spacings = np.concatenate([s.spacings for s in sectors])
    return float(np.median(spacings)), float(spacings.min())

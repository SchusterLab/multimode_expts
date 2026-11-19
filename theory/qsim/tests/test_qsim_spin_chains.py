"""Checks of the c-XXZ reference model against a full-space Pauli build."""
from functools import reduce

import numpy as np
import pytest

from theory.qsim.spin_chains import cxxz_operators, cxxz_sectors, cxxz_spectrum, pooled_sticks, qgt

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
UP_IS_FIRST = True   # in kron order, basis vector 0 of a site is up (Z = +1)


def site(op, i, length):
    return reduce(np.kron, [op if j == i else np.eye(2) for j in range(length)])


def full_cxxz(length, h, g, delta):
    bond = lambda i: 0.5 * (site(X, i, length) @ site(X, i + 1, length) + site(Y, i, length) @ site(Y, i + 1, length)
                            + delta * site(Z, i, length) @ site(Z, i + 1, length))
    return sum(bond(i) for i in range(1, length - 1)) + g * bond(0) + h * site(Z, 0, length), bond(0), site(Z, 0, length)


def full_index(state, length):
    # Spin i (from 0) is bit i of ``state``, 1 = up. In kron order spin 0 is the most significant digit
    # and digit 0 is up.
    return sum((0 if state >> i & 1 else 1) << (length - 1 - i) for i in range(length))


@pytest.mark.parametrize("length", [5, 6])
def test_sector_matrices_match_the_full_pauli_build(length):
    h, g, delta = 1.6, 1.3, 1.2
    H_full, boundary_full, field_full = full_cxxz(length, h, g, delta)
    for ups in range(length + 1):
        ops = cxxz_operators(length, ups, delta)
        idx = [full_index(int(s), length) for s in ops.basis]
        block = np.ix_(idx, idx)
        np.testing.assert_allclose(ops.bulk + g * ops.boundary + h * ops.field, H_full[block].real, atol=1e-14)
        np.testing.assert_allclose(ops.boundary, boundary_full[block].real, atol=1e-14)
        np.testing.assert_allclose(ops.field, field_full[block].real, atol=1e-14)
        assert np.abs(H_full[block].imag).max() < 1e-14
    # H conserves the magnetization: no matrix element between sectors.
    m = np.array([sum(1 if (k >> (length - 1 - i)) & 1 == 0 else -1 for i in range(length)) for k in range(2**length)])
    assert np.abs(H_full[m[:, None] != m[None, :]]).max() < 1e-14


def test_sectors_of_the_paper():
    assert cxxz_sectors(14) == [6, 7, 8]
    assert cxxz_sectors(15) == [7, 8]


def test_cross_weights_sum_to_the_connected_variance_of_each_central_state():
    # sum_{m != n} |<m|O|n>|^2 = <n|O^2|n> - <n|O|n>^2.
    length, ups, h, g = 8, 4, 1.6, 1.2
    ops = cxxz_operators(length, ups)
    H = ops.bulk + g * ops.boundary + h * ops.field
    sector = cxxz_spectrum(length, h, g, operators={u: cxxz_operators(length, u) for u in cxxz_sectors(length)})[1]
    energies, states = np.linalg.eigh(H)
    window = np.arange((len(energies) - sector.count) // 2, (len(energies) - sector.count) // 2 + sector.count)
    for a, O in enumerate((ops.field, ops.boundary)):
        M = states.T @ O @ states
        expected = (M @ M).diagonal()[window] - M.diagonal()[window]**2
        np.testing.assert_allclose(sector.cross[a, a].sum(axis=0), expected, atol=1e-12)


def test_qgt_diagonal_is_the_kernel_sum_of_the_pooled_sticks():
    sectors = cxxz_spectrum(8, 1.6, 1.2)
    mu = 0.05
    g = qgt(sectors, mu)
    for a, n in enumerate(([1, 0], [0, 1])):
        omega, w = pooled_sticks(sectors, n)
        assert g[a, a] == pytest.approx(np.sum(w * omega**2 / (omega**2 + mu**2)**2), rel=1e-12)
    np.testing.assert_allclose(g, g.T, atol=1e-14)

"""Known-truth tests of the general-operator AGP tools in theory.qsim.agp."""
import numpy as np
import pytest

from theory.qsim.agp import (broadened_spectral_function, cross_weights, energy_differences,
                             operator_weights, positive_sticks, qgt_directions, regularized_qgt,
                             spectral_agp, spectral_weights)
from slab import AttrDict

SX = np.array([[0, 1], [1, 0]], dtype=float)
SZ = np.diag([1.0, -1.0])


def two_level(omega0):
    energies = np.array([0.0, omega0])
    return energies, np.eye(2)


def test_two_level_weights_and_agp_norm():
    # H = diag(0, w0), O = sigma_x: w_01 = w_10 = 1/2, chi = w0^2 / (w0^2 + mu^2)^2.
    omega0, mu = 0.3, 0.05
    energies, states = two_level(omega0)
    weights = operator_weights(states, SX)
    np.testing.assert_allclose(weights, [[0, 0.5], [0.5, 0]])
    differences = energy_differences(energies)
    chi = spectral_agp(differences, weights, mu, 1.0)
    assert chi == pytest.approx(omega0**2 / (omega0**2 + mu**2)**2)
    f, w, dc = positive_sticks(differences, weights)
    np.testing.assert_allclose([f[0], w[0], dc], [omega0, 0.5, 0.0])


def test_qgt_of_a_conserved_and_a_breaking_direction():
    # sigma_z commutes with H: no off-diagonal weight, so the minimal direction is (0, 1).
    omega0, mu = 0.3, 0.05
    energies, states = two_level(omega0)
    cross = cross_weights(states, [SX, SZ])
    g = regularized_qgt(energy_differences(energies), cross, mu)
    np.testing.assert_allclose(g, [[omega0**2 / (omega0**2 + mu**2)**2, 0], [0, 0]], atol=1e-15)
    directions = qgt_directions(g)
    assert directions["min_value"] == pytest.approx(0, abs=1e-15)
    assert directions["min_angle"] == pytest.approx(np.pi / 2)
    assert directions["max_angle"] == pytest.approx(0, abs=1e-12)


def test_qgt_of_a_rotated_pair_has_rotated_directions():
    # A = sigma_x, B = sigma_x + sigma_z: the conserved combination is B - A, angle 3 pi / 4.
    energies, states = two_level(0.3)
    g = regularized_qgt(energy_differences(energies), cross_weights(states, [SX, SX + SZ]), 0.05)
    assert qgt_directions(g)["min_angle"] == pytest.approx(3 * np.pi / 4)


def test_operator_weights_match_spectral_weights_for_a_diagonal_probe():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(6, 6))
    energies, states = np.linalg.eigh(H + H.T)
    values = rng.normal(size=6)
    model = AttrDict(dict(states=states, energies_MHz=energies))
    _, expected = spectral_weights(model, values)
    np.testing.assert_allclose(operator_weights(states, np.diag(values)), expected, atol=1e-14)


def test_broadened_function_integrates_to_the_weight_and_peaks_at_the_line():
    omega = np.linspace(-1, 2, 30001)
    phi = broadened_spectral_function([0.4, 1.1], [0.25, 0.5], omega, 0.02)
    assert np.trapz(phi, omega) == pytest.approx(0.75, rel=1e-6)
    assert phi[np.argmin(np.abs(omega - 0.4))] == pytest.approx(0.25 / (0.02 * np.sqrt(2 * np.pi)), rel=1e-6)


def test_qgt_quadratic_form_is_the_agp_norm_of_the_combined_operator():
    # n^T g n = chi of n_a A + n_b B, for complex Hermitian operators and a random H.
    rng = np.random.default_rng(1)

    def hermitian(d):
        m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        return m + m.conj().T

    energies, states = np.linalg.eigh(hermitian(8))
    A, B = hermitian(8), hermitian(8)
    differences = energy_differences(energies)
    g = regularized_qgt(differences, cross_weights(states, [A, B]), 0.3)
    for phi in (0.0, 0.7, 2.0):
        n = np.array([np.cos(phi), np.sin(phi)])
        chi = spectral_agp(differences, operator_weights(states, n[0] * A + n[1] * B), 0.3, 1.0)
        assert n @ g @ n == pytest.approx(chi, rel=1e-12)

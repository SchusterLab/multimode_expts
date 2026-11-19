"""Checks of the central-boson spectral functions against independent calculations.

The checks C1 to C7 of analysis_notebooks/guan/agp/understanding/spectral_function.py.
"""
import numpy as np
import pytest
from scipy import stats
from scipy.linalg import expm

from theory.qsim.agp import probe_values, spectral_weights, two_time_correlation
from theory.qsim.central_boson import disorder_pattern, disorder_values, fixed_n_hamiltonian, operator_form_hamiltonian
from theory.qsim.spectral_ensemble import (DEFAULT_PROBES, StarSetup, ensemble_spectra, mode_spectrum,
                                           stick_spectrum, zero_disorder_lines)

SETUP = StarSetup(photon_number=3, storage_modes=4, coupling_mhz=0.02)
KERR, RMS = -0.012, 0.035


@pytest.fixture(scope="module")
def point():
    unit = disorder_pattern(np.random.default_rng(11), SETUP.storage_modes, 1.0)
    detunings = RMS * unit
    independent = operator_form_hamiltonian(SETUP.photon_number, SETUP.storage_modes + 1, detunings,
                                            [SETUP.coupling_mhz] * SETUP.storage_modes, KERR)
    return unit, SETUP.model(detunings, KERR), independent


def test_c1_hamiltonian_matches_the_operator_form_build(point):
    _, model, independent = point
    np.testing.assert_allclose(model.energies_MHz, np.linalg.eigvalsh(independent.hamiltonian_MHz), atol=1e-12)
    np.testing.assert_allclose(model.hamiltonian_MHz, independent.hamiltonian_MHz, atol=1e-15)


def test_c2_correlation_from_the_matrix_exponential_matches_the_sticks(point):
    unit, model, independent = point
    n0 = independent.numbers[0]
    probes = {"hub Kerr": (n0 * (n0 - 1) / 2, probe_values(model.fock_basis, 0, "kerr")),
              "hub number": (n0, probe_values(model.fock_basis, 0, "number")),
              "storage 2": (independent.numbers[2], probe_values(model.fock_basis, 2, "number")),
              "disorder V": (sum(u * n for u, n in zip(unit, independent.numbers[1:])),
                             disorder_values(model.fock_basis, unit))}
    times = np.linspace(0, 300, 7)
    H = independent.hamiltonian_MHz
    for diagonal, values in probes.values():
        O = np.diag(diagonal)
        direct = []
        for t in times:
            U = expm(-2j * np.pi * H * t)
            direct.append(np.trace(U.conj().T @ O @ U @ O).real / len(O))
        differences, weights = spectral_weights(model, values)
        np.testing.assert_allclose(two_time_correlation(differences, weights, times), direct, atol=1e-10)


def test_c3_sum_rule(point):
    unit, model, _ = point
    for values in (probe_values(model.fock_basis, 0, "kerr"), probe_values(model.fock_basis, 0, "number"),
                   probe_values(model.fock_basis, 2, "number"), disorder_values(model.fock_basis, unit)):
        _, w, dc = stick_spectrum(model, values)
        assert dc + 2 * w.sum() == pytest.approx(np.mean(values**2), abs=1e-12)


def test_c4_free_limit_sticks_lie_on_single_particle_differences():
    rng = np.random.default_rng(3)
    for _ in range(30):
        detunings = disorder_pattern(rng, SETUP.storage_modes, RMS)
        single = fixed_n_hamiltonian(1, SETUP.storage_modes + 1, detunings,
                                     [SETUP.coupling_mhz] * SETUP.storage_modes, 0.0).energies_MHz
        lines = np.abs(single[:, None] - single[None, :]).ravel()
        lines = lines[lines > 1e-9]
        model = SETUP.model(detunings, 0.0)
        for mode in (0, 1):
            f, w, _ = mode_spectrum(model, mode, "number")
            f = f[w > 1e-10]
            assert np.abs(f[:, None] - lines[None, :]).min(axis=1).max() < 1e-9


def test_c5_pattern_has_zero_mean_and_the_set_rms():
    patterns = np.array([disorder_pattern(np.random.default_rng(k), 4, RMS) for k in range(200)])
    np.testing.assert_allclose(patterns.mean(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(np.sqrt((patterns**2).mean(axis=1)), RMS, rtol=1e-12)


def test_c6_binned_density_integrates_to_the_pooled_weight_in_range():
    edges = np.geomspace(1e-5, 0.5, 60)
    spectra = ensemble_spectra(SETUP, KERR, RMS, 30, edges, seed=3)
    for s in spectra.values():
        inside = (s["f"] >= edges[0]) & (s["f"] < edges[-1])
        assert (s["density"] * np.diff(edges)).sum() == pytest.approx(s["w"][inside].sum() / 30, abs=1e-12)


def test_c7_gaussian_pattern_is_uniform_on_the_sphere():
    n = SETUP.storage_modes
    rng = np.random.default_rng(1)
    p = np.array([disorder_pattern(rng, n, 1.0) for _ in range(40000)])
    basis = np.linalg.qr(np.eye(n) - 1 / n)[0][:, :n - 1].T
    unit = p @ basis.T / np.sqrt(n)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, rtol=1e-12)
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        # Archimedes: the component along any axis of a uniform unit vector in 3D is uniform on [-1, 1].
        assert stats.kstest(unit @ axis, stats.uniform(-1, 2).cdf).pvalue > 1e-3


def test_zero_disorder_covariance_formula_is_exact_over_an_orthonormal_set_of_patterns():
    # Patterns sqrt(n) q_k, q_k an orthonormal basis of the zero-mean plane, have second moment
    # exactly c (I - J / n); at W = 0 H does not depend on them, so their mean V spectrum is the formula.
    n = SETUP.storage_modes
    lines = zero_disorder_lines(SETUP, KERR)["disorder V"]
    model = SETUP.model(np.zeros(n), KERR)
    basis = np.linalg.qr(np.eye(n) - 1 / n)[0][:, :n - 1].T
    f = []
    w = []
    for q in basis:
        fk, wk, _ = stick_spectrum(model, disorder_values(model.fock_basis, np.sqrt(n) * q))
        f.append(fk)
        w.append(wk / (n - 1))
    f, w = np.concatenate(f), np.concatenate(w)
    index = np.searchsorted(lines["f"], np.round(f, 9))
    assert np.all(lines["f"][index] == np.round(f, 9))
    np.testing.assert_allclose(np.bincount(index, weights=w, minlength=len(lines["f"])), lines["w"], atol=1e-12)


def test_zero_disorder_lines_keep_the_sum_rule():
    lines = zero_disorder_lines(SETUP, KERR)
    model = SETUP.model(np.zeros(SETUP.storage_modes), KERR)
    for name, (mode, kind) in DEFAULT_PROBES.items():
        if kind == "disorder":
            continue
        values = probe_values(model.fock_basis, 0 if mode is None else mode, kind) if mode is not None else \
            probe_values(model.fock_basis, 1, kind)
        assert lines[name]["dc"] + 2 * lines[name]["w"].sum() == pytest.approx(np.mean(values**2), abs=1e-12)

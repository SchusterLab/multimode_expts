"""The Hamiltonian fit (plan docs/qsim/pole_finding_explore.md, T1)."""
import numpy as np
import pytest

from fitting.qsim.mbr_disorder import disorder_direction
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles import hamiltonian_fit as hf
from fitting.qsim.poles.synthetic import Hardware, ModelPoint, Nonideal, synthetic_returns

POINT = (8.615e-3, -1.22, 5.80)                     # the August point: g, K / g, delta / g
HARDWARE = Hardware(coupling_MHz=POINT[0], dt_us=1.4509, samples=200, partial_rows=10)


def august_case(nonideal):
    direction = disorder_direction(100, 4)
    case = synthetic_returns(ModelPoint(POINT[1], POINT[2], direction, 0), HARDWARE, nonideal)
    return case, hf.true_parameters(POINT, direction)


def test_generators_rebuild_the_model():
    truth = hf.true_parameters(POINT, disorder_direction(100, 4))
    occupations = fixed_n_hamiltonian(3, 5, np.zeros(4), np.zeros(4), 0.).fock_basis
    model = hf.model_of(occupations)
    reference = fixed_n_hamiltonian(3, 5, truth.detunings_MHz, truth.couplings_MHz, truth.kerr_MHz)
    np.testing.assert_allclose(np.tensordot(truth.vector, model.generators, axes=1), reference.hamiltonian_MHz, atol=1e-15)
    np.testing.assert_array_equal(model.rows, np.arange(35))


@pytest.mark.parametrize("amplitudes", ["model", "free"])
def test_jacobian_matches_finite_differences(amplitudes):
    """On noiseless rows at the truth (residual 0) the projected Jacobian is exact."""
    case, truth = august_case(Nonideal(decay_per_us=0.01, offset_sigma_MHz=0.5e-3))
    offsets = np.random.default_rng(0).normal(0, 0.5e-3, size=len(case.A))
    model = hf.model_of(case.truth.occupations)
    data = (0.7 * case.A, case.time_us, np.arange(len(case.A)), np.full(len(case.A), 0.1), model)
    x = np.r_[truth.vector, 0.01, offsets]
    settings = hf.HamiltonianFitSettings(amplitudes=amplitudes)
    function = hf.residual_and_jacobian_model if amplitudes == "model" else hf.residual_and_jacobian_free
    r, J = function(x, data, settings)
    assert np.sum(r[:-len(offsets)] ** 2) < 1e-12
    h = 1e-8
    numeric = np.stack([(function(x + h * e, data, settings)[0] - function(x - h * e, data, settings)[0]) / (2 * h)
                        for e in np.eye(len(x))], axis=1)
    np.testing.assert_allclose(J, numeric, atol=1e-3 * np.abs(numeric).max())


@pytest.mark.parametrize("amplitudes", ["model", "free"])
def test_recovers_perturbed_parameters(amplitudes):
    """From a start off by the real model error (1 kHz, 5 %), the fit returns the truth within
    4 of its errors, and its levels within a small fraction of a bin."""
    nonideal = Nonideal(snr=1 / 0.075, decay_per_us=0.005, offset_sigma_MHz=0.5e-3, seed=0)
    case, truth = august_case(nonideal)
    start = hf.perturbed(truth, 1e-3, 0.05, 1e-3, seed=1)
    settings = hf.HamiltonianFitSettings(amplitudes=amplitudes, starts=3)
    result = hf.fit_hamiltonian(case.A, case.time_us, case.truth.occupations, start, settings)
    z = (result.parameters.vector - truth.vector) / result.errors.vector
    assert np.all(np.abs(z) < 4)
    assert result.decay_per_us == pytest.approx(0.005, rel=0.05)
    assert result.reduced_chi2 == pytest.approx(1, abs=0.1)
    fit = result.pole_fit
    assert len(fit.frequencies_MHz) == len(case.truth.levels_MHz)
    assert np.max(np.abs(np.sort(fit.frequencies_MHz) - case.truth.levels_MHz)) < 0.05 * HARDWARE.bin_MHz


def test_fit_is_the_common_interface():
    case, truth = august_case(Nonideal(decay_per_us=0.005))
    fit = hf.fit(case.A, case.time_us, hf.HamiltonianFitSettings(starts=1), occupations=case.truth.occupations,
                 start=truth)
    np.testing.assert_allclose(fit.returns(case.time_us), case.A / case.A[:, :1], atol=1e-8)
    assert fit.row_offsets_MHz.shape == (len(case.A),) and fit.frequency_errors_MHz.shape == fit.frequencies_MHz.shape

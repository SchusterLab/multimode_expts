"""Ensemble averaging, common-realization errors, and independent predictions."""
import numpy as np

from fitting.qsim.agp import agp_kernel_weights
from fitting.qsim.agp_ensemble import ensemble_agp
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights, two_time_correlation
from theory.qsim.agp_ensemble import ensemble_prediction
from theory.qsim.central_boson import fixed_n_hamiltonian


def test_masked_state_means_and_confusion_correction_use_the_existing_convention():
    times = np.array([0., 10., 30.])
    labels = np.array([[[0, 1, 2, 3], [0, 0, 2, 2], [3, 2, 1, 0]],
                       [[1, 1, 1, 1], [1, 2, 2, 3], [0, 0, 0, 0]]])
    accepted = np.ones(labels.shape, dtype=bool)
    accepted[0, 1, 2:] = False
    confusion = .8 * np.eye(4) + .05
    result = ensemble_agp(times, labels, accepted, [.5, .25], [5.915, 10.], .02,
                          np.arange(4), confusion, bootstrap_samples=0)
    expected_raw = np.array([1., .5, .75])
    expected_corrected = (expected_raw - .3 * .75) / .8
    np.testing.assert_allclose(result["C1"], expected_raw)
    np.testing.assert_allclose(result["C1_corrected"], expected_corrected)
    np.testing.assert_array_equal(result["shot_counts"], [[4, 2, 4], [4, 4, 4]])
    expected = [(2 * np.pi * .02)**2 * agp_kernel_weights(times, cutoff / 1000)
                @ expected_corrected for cutoff in [5.915, 10.]]
    np.testing.assert_allclose(result["agp"], expected)


def test_constant_trace_has_the_exact_finite_window_dc_contribution_and_units():
    times = np.array([0., 8., 25., 40.])
    cutoff_khz, coupling_mhz = 10., .02
    labels = np.ones((1, len(times), 4), dtype=int)
    result = ensemble_agp(times, labels, np.ones(labels.shape, dtype=bool), [1.],
                          cutoff_khz, coupling_mhz, np.arange(4), bootstrap_samples=0)
    angular_cutoff = 2 * np.pi * cutoff_khz / 1000
    expected = (2 * np.pi * coupling_mhz)**2 * times[-1] * np.exp(
        -angular_cutoff * times[-1]) / (2 * angular_cutoff)
    np.testing.assert_allclose(result["agp"], [expected], rtol=1e-12)


def test_shared_realization_bootstrap_preserves_time_covariance():
    times = np.array([0., 8., 20.])
    labels = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]])
    result = ensemble_agp(times, labels, np.ones(labels.shape, dtype=bool), [1.],
                          5.915, .02, np.arange(4), bootstrap_samples=300, seed=14)
    covariance = result["C1_covariance"]
    np.testing.assert_allclose(covariance, np.full((3, 3), covariance[0, 0]), atol=1e-15)
    kernel = (2 * np.pi * .02)**2 * agp_kernel_weights(times, .005915)
    expected_error = abs(kernel.sum()) * result["C1_sampling_err"][0]
    np.testing.assert_allclose(result["agp_sampling_err"], [expected_error], rtol=1e-12)
    assert result["bootstrap_valid_samples"] == 300


def test_shared_realization_bootstrap_preserves_cross_state_cancellation():
    times = np.array([0., 10.])
    first_state = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    labels = np.stack([first_state, 3 - first_state])
    result = ensemble_agp(times, labels, np.ones(labels.shape, dtype=bool), [.5, .5],
                          10., .02, np.arange(4), bootstrap_samples=100, seed=4)
    np.testing.assert_allclose(result["C1"], [1.5, 1.5])
    np.testing.assert_allclose(result["C1_sampling_err"], 0, atol=1e-14)
    np.testing.assert_allclose(result["agp_sampling_err"], 0, atol=1e-14)


def test_calibration_error_does_not_shrink_when_the_same_outcomes_are_repeated():
    times = np.array([0., 10., 30.])
    labels = np.array([[[0, 1, 2, 3], [1, 1, 2, 2], [2, 2, 3, 3]]])
    confusion = .8 * np.eye(4) + .05
    errors = []
    for repeats in [1, 4]:
        repeated = np.tile(labels, (1, 1, repeats))
        result = ensemble_agp(times, repeated, np.ones(repeated.shape, dtype=bool),
                              [1.], [5.915, 10.], .02, np.arange(4), confusion,
                              calibration_shots=500, bootstrap_samples=0)
        errors.append(result["agp_calibration_err"])
    assert np.all(errors[0] > 0)
    np.testing.assert_allclose(errors[0], errors[1], rtol=1e-14)


def test_signed_finite_window_result_is_retained():
    times = np.array([0., 100.])
    labels = np.array([[[0, 0, 0, 0], [3, 3, 3, 3]]])
    result = ensemble_agp(times, labels, np.ones(labels.shape, dtype=bool), [1.],
                          10., .02, np.arange(4), bootstrap_samples=0)
    assert result["agp"][0] < 0


def test_one_realization_does_not_claim_a_sampling_error():
    labels = np.ones((1, 2, 1), dtype=int)
    result = ensemble_agp([0., 10.], labels, np.ones(labels.shape, dtype=bool),
                          [1.], 10., .02, np.arange(4), bootstrap_samples=20)
    assert np.isnan(result["agp_sampling_err"]).all()
    assert result["bootstrap_valid_samples"] == 0


def test_theory_averages_the_saved_hamiltonians_not_the_mean_vector():
    times = np.linspace(0., 90., 51)
    detunings = np.array([[0.], [.08]])
    result = ensemble_prediction(1, detunings, [.02], -.027, 1, times, [5.915, 10.])
    traces, norms = [], []
    for detuning in [0., .08]:
        model = fixed_n_hamiltonian(1, 2, [detuning], [.02], -.027)
        differences, weights = spectral_weights(model, probe_values(model.fock_basis, 1))
        traces.append(two_time_correlation(differences, weights, times))
        norms.append(spectral_agp(differences, weights, np.array([.005915, .01]), .02))
    np.testing.assert_allclose(result["C1_theory"], np.mean(traces, axis=0))
    np.testing.assert_allclose(result["agp_theory"], np.mean(norms, axis=0))
    np.testing.assert_allclose(result["C1_theory"][0], .5)
    mean_vector = ensemble_prediction(1, [[.04]], [.02], -.027, 1, times, [5.915, 10.])
    assert np.max(abs(result["C1_theory"] - mean_vector["C1_theory"])) > .02


def test_kerr_probe_uses_pair_occupations():
    times = [0., 5.]
    result = ensemble_prediction(2, [[0.], [.04]], [.02], -.027, 0, times, 10.,
                                  probe="kerr")
    # The N=2 two-mode sector has three states; only |2,0> has Kerr probe 1.
    np.testing.assert_allclose(result["C1_theory"][0], 1 / 3)

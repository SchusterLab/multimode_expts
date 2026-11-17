"""Independent checks of measured metrics, disorder weighting, and units."""
import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from fitting.qsim.agp import agp_kernel_weights
from fitting.qsim.agp_metric import ensemble_agp_metric
from theory.qsim.agp_ensemble import ensemble_metric_prediction, ensemble_prediction
from theory.qsim.central_boson import fixed_n_hamiltonian


def two_mode_measurement():
    """Exact four-shot Born distributions for a two-photon beam splitter."""
    occupations = np.array([[0, 2], [1, 1], [2, 0]])
    times = np.array([0., 6.25, 12.5])
    m1 = np.empty((3, 3, 4), dtype=int)
    for state, occupation in enumerate(occupations):
        m1[state, 0] = occupation[0]
        m1[state, 1] = [0, 0, 2, 2] if occupation[0] == 1 else [0, 1, 1, 2]
        m1[state, 2] = occupation[1]
    labels = np.stack([m1, 2 - m1], axis=-1)
    accepted = np.ones(labels.shape[:-1], dtype=bool)
    return times, occupations, labels, accepted


def test_measured_metric_matches_independent_schrodinger_correlator_and_units():
    times, occupations, labels, accepted = two_mode_measurement()
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], np.ones((4, 1)), [5., 10.], .02,
                                 bootstrap_samples=0)
    model = fixed_n_hamiltonian(2, 2, [0.], [.02], 0.)
    numbers = np.asarray(model.fock_basis)
    probes = np.column_stack([numbers[:, 0] * (numbers[:, 0] - 1) / 2, -numbers[:, 1]])
    reference = []
    for time in times:
        evolution = expm(-2j * np.pi * model.hamiltonian_MHz * time)
        output = (np.abs(evolution)**2).T @ probes
        reference.append(probes.T @ output / len(numbers))
    np.testing.assert_allclose(result["C_metric_local"], reference, atol=1e-14)
    expected = np.array([(2 * np.pi / 1000)**2 * np.einsum(
        "t,tab->ab", agp_kernel_weights(times, cutoff / 1000), reference)
        for cutoff in [5., 10.]])
    np.testing.assert_allclose(result["agp_metric"], expected, atol=1e-17)
    np.testing.assert_allclose(result["agp"], np.diagonal(expected, axis1=1, axis2=2).T * 20**2)
    np.testing.assert_allclose(result["agp_metric_scaled"], expected * 20**2)


def test_zero_disorder_patterns_are_weighted_before_averaging():
    times, occupations, labels, accepted = two_mode_measurement()
    labels = labels[:, :, [0, 0]]
    accepted = accepted[:, :, :2]
    patterns = np.array([[1.], [-1.]])
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], patterns, 10., .02, bootstrap_samples=0)
    np.testing.assert_allclose(result["C_metric"][:, 0, 1], 0, atol=1e-15)
    np.testing.assert_allclose(result["C_metric"][:, 1, 1], result["C_metric_local"][:, 1, 1])
    assert np.max(np.abs(result["C_metric"][:, 1, 1])) > 0


def test_common_herald_mask_preserves_realization_pairing_and_state_weights():
    times, occupations, labels, accepted = two_mode_measurement()
    patterns = np.array([[1.], [2.], [-1.], [.5]])
    accepted[0, 1, 1] = False
    accepted[1, 2, 3] = False
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], patterns, 10., .02, bootstrap_samples=0)
    reference = np.zeros((len(times), 2, 2))
    for state, occupation in enumerate(occupations):
        initial_kerr = occupation[0] * (occupation[0] - 1) / 2
        for time in range(len(times)):
            retained = np.flatnonzero(accepted[state, time])
            for realization in retained:
                initial = np.array([initial_kerr, -patterns[realization, 0] * occupation[1]])
                photon = labels[state, time, realization, 0]
                output = np.array([photon * (photon - 1) / 2,
                                   -patterns[realization, 0] * labels[state, time, realization, 1]])
                reference[time] += (np.outer(initial, output) + np.outer(output, initial)) / (
                    2 * len(occupations) * len(retained))
    np.testing.assert_allclose(result["C_metric"], reference)
    np.testing.assert_array_equal(result["shot_counts"], accepted.sum(axis=-1))


def test_confusion_correction_uses_kerr_label_scores_not_mean_photon_number():
    times, occupations, labels, accepted = two_mode_measurement()
    matrix = .8 * np.eye(4) + .05
    calibration = np.stack([matrix, matrix], axis=-1)
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], np.ones((4, 1)), 10., .02,
                                 confusion_matrices=calibration, bootstrap_samples=0)
    q_values = np.array([0., 0., 1., 3.])
    corrected_scores = np.linalg.solve(matrix, q_values)[labels[..., 0]]
    initial_q = occupations[:, 0] * (occupations[:, 0] - 1) / 2
    expected = initial_q @ corrected_scores.mean(axis=-1) / len(occupations)
    np.testing.assert_allclose(result["C1_corrected"][0], expected)
    assert not np.allclose(result["C1"][0], expected)


def test_partial_metric_is_explicitly_unavailable_and_signed_values_are_retained():
    times, occupations, labels, accepted = two_mode_measurement()
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0], np.ones((4, 1)), 10., .02, bootstrap_samples=0)
    np.testing.assert_array_equal(result["agp_metric_available"], [[True, False], [False, False]])
    assert np.isnan(result["agp_metric"][:, 1]).all()
    assert np.isnan(result["agp_metric_eigenvectors"]).all()
    labels[:, 0, :, 0] = 0
    labels[:, -1, :, 0] = 2
    negative = ensemble_agp_metric([0., 100.], labels[:, [0, -1]], accepted[:, [0, -1]],
                                   occupations, [4], [0, 4], [0], np.ones((4, 1)),
                                   10., .02, bootstrap_samples=0)
    assert negative["agp_metric_local"][0, 0, 0] < 0


def test_missing_initial_states_are_not_reported_as_a_complete_metric():
    times, occupations, labels, accepted = two_mode_measurement()
    with pytest.raises(ValueError, match="complete initial-state sector"):
        ensemble_agp_metric(times, labels[:-1], accepted[:-1], occupations[:-1],
                             [4], [0, 4], [0, 4], np.ones((4, 1)), 10., .02,
                             bootstrap_samples=0)


def test_joint_bootstrap_includes_shared_calibration_counts():
    times, occupations, labels, accepted = two_mode_measurement()
    matrix = .8 * np.eye(4) + .05
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], np.ones((4, 1)), 10., .02,
                                 confusion_matrices=np.stack([matrix, matrix], axis=-1),
                                 calibration_shots=np.full((4, 2), 500),
                                 bootstrap_samples=20, seed=2)
    assert result["bootstrap_valid_samples"] == 20
    assert result["calibration_uncertainty_included"]
    assert np.isfinite(result["agp_metric_err"]).all()
    assert np.max(result["agp_metric_err"]) > 0


def test_zero_calibration_counts_are_rejected_before_sampling():
    times, occupations, labels, accepted = two_mode_measurement()
    with pytest.raises(ValueError, match="Calibration row counts must be positive"):
        ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                             [0, 4], np.ones((4, 1)), 10., .02,
                             calibration_shots=np.zeros((4, 2)), bootstrap_samples=10)


def test_singular_calibration_draws_expose_the_valid_bootstrap_fraction():
    times, occupations, labels, accepted = two_mode_measurement()
    matrix = .8 * np.eye(4) + .05
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], np.ones((4, 1)), 10., .02,
                                 confusion_matrices=np.stack([matrix, matrix], axis=-1),
                                 calibration_shots=np.ones((4, 2)),
                                 bootstrap_samples=50, seed=4)
    assert 0 < result["bootstrap_valid_samples"] < result["bootstrap_requested_samples"]
    assert result["bootstrap_requested_samples"] == 50


def test_histogram_bootstrap_matches_direct_weighted_single_shots():
    times, occupations, labels, accepted = two_mode_measurement()
    patterns = np.array([[1.], [2.], [-1.], [.5]])
    accepted[0, 1, 1] = False
    matrix = .8 * np.eye(4) + .05
    calibration = np.stack([matrix, matrix], axis=-1)
    result = ensemble_agp_metric(times, labels, accepted, occupations, [4], [0, 4],
                                 [0, 4], patterns, 10., .02,
                                 confusion_matrices=calibration,
                                 bootstrap_samples=30, seed=15)
    lookup = np.stack([np.linalg.solve(matrix, [0., 0., 1., 3.]),
                       np.linalg.solve(matrix, -np.arange(4.))])
    random_generator = np.random.default_rng(15)
    target_samples, local_samples = [], []
    kernel = (2 * np.pi / 1000)**2 * agp_kernel_weights(times, .01)
    for _ in range(30):
        indices = random_generator.integers(4, size=4)
        target = np.zeros((len(times), 2, 2))
        local = np.zeros_like(target)
        valid = True
        for state, occupation in enumerate(occupations):
            initial = np.array([occupation[0] * (occupation[0] - 1) / 2, -occupation[1]])
            for time in range(len(times)):
                retained = indices[accepted[state, time, indices]]
                if not len(retained):
                    valid = False
                    continue
                for realization in retained:
                    output = lookup[np.arange(2), labels[state, time, realization]]
                    jacobian = np.diag([1., patterns[realization, 0]])
                    local[time] += np.outer(initial, output) / (3 * len(retained))
                    target[time] += np.outer(initial @ jacobian, output @ jacobian) / (3 * len(retained))
        if valid:
            local_samples.append(np.einsum("t,tab->ab", kernel, (local + local.swapaxes(1, 2)) / 2))
            target_samples.append(np.einsum("t,tab->ab", kernel, (target + target.swapaxes(1, 2)) / 2))
    np.testing.assert_allclose(result["agp_metric_err"][0], np.std(target_samples, axis=0, ddof=1))
    np.testing.assert_allclose(result["agp_metric_local_err"][0], np.std(local_samples, axis=0, ddof=1))
    assert result["bootstrap_valid_samples"] == len(target_samples)


def test_theory_signed_cross_metric_matches_independent_time_integration():
    times = np.array([0., 90.])
    cutoff_khz = 10.
    prediction = ensemble_metric_prediction(2, [[.03]], [.02], -.027,
                                            [0, 4], [4], [[1.]], times, cutoff_khz)
    model = fixed_n_hamiltonian(2, 2, [.03], [.02], -.027)
    numbers = np.asarray(model.fock_basis)
    probes = np.column_stack([numbers[:, 0] * (numbers[:, 0] - 1) / 2, -numbers[:, 1]])
    rate = 2 * np.pi * cutoff_khz / 1000

    def integrand(time):
        evolution = expm(-2j * np.pi * model.hamiltonian_MHz * time)
        output = np.abs(evolution.T)**2 @ probes
        correlation = probes.T @ output / len(numbers)
        kernel = np.exp(-rate * time) * (1 - rate * time) / (2 * rate)
        return (2 * np.pi / 1000)**2 * kernel * correlation

    finite, _ = quad_vec(integrand, 0., times[-1], epsabs=1e-12)
    infinite, _ = quad_vec(integrand, 0., 800., epsabs=1e-12)
    np.testing.assert_allclose(prediction["agp_metric_theory_window"][0], finite, atol=1e-11)
    np.testing.assert_allclose(prediction["agp_metric_theory"][0], infinite, atol=1e-11)
    assert abs(finite[0, 1]) > 1e-8


def test_theory_diagonals_reuse_the_existing_scaled_norm_convention():
    arguments = dict(photon_number=2, detunings_mhz=[[0.], [.04]], couplings_mhz=[.02],
                     kerr_mhz=-.027, time_us=[0., 10., 90.], mu_khz=[5.915, 10.])
    metric = ensemble_metric_prediction(**arguments, directions=[0, 4], swap_stors=[4],
                                         disorder_patterns=[[1.], [-1.]])
    for direction, mode, probe in [(0, 0, "kerr"), (1, 1, "number")]:
        scalar = ensemble_prediction(**arguments, mode_index=mode, probe=probe)
        for key in ["C1_theory", "agp_theory", "agp_theory_window"]:
            np.testing.assert_allclose(metric[key][direction], scalar[key], atol=1e-13)

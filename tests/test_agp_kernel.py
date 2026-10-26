"""Numerical checks of the AGP kernel and the calibration errors against independent calculations."""

import numpy as np

from scipy.integrate import trapezoid

from fitting.qsim.agp import agp_kernel_weights, calibration_error_factors, finite_time_agp


def test_finite_window_constant_and_cosine():
    times = np.linspace(0, 145, 50001)
    cutoff_mhz, frequency_mhz = 0.005915, 0.0173
    mu, omega = 2 * np.pi * np.array([cutoff_mhz, frequency_mhz])
    offsets = np.array([0.8, 0.4])
    amplitudes = np.array([0.35, -0.3])
    signal = offsets[:, None] + amplitudes[:, None] * np.cos(omega * times)

    # Exact finite-window integrals of exp(-z*t) and t*exp(-z*t).
    duration = times[-1]
    z = mu - 1j * omega
    integral = (1 - np.exp(-z * duration)) / z
    first_moment = (1 - (1 + z * duration) * np.exp(-z * duration)) / z**2
    cosine = np.real(integral - mu * first_moment) / (2 * mu)
    constant = duration * np.exp(-mu * duration) / (2 * mu)
    expected = offsets * constant + amplitudes * cosine

    values = [finite_time_agp(times, row, cutoff_mhz)["values"] for row in signal]
    np.testing.assert_allclose(values, expected, rtol=1e-6)
    assert values[1] < 0  # Do not clip signed finite-window estimates.


def test_long_window_matches_angular_spectral_filter():
    cutoff_mhz, frequency_mhz = 0.01, 0.027
    mu, omega = 2 * np.pi * np.array([cutoff_mhz, frequency_mhz])
    times = np.linspace(0, 30 / mu, 100001)
    result = finite_time_agp(times, np.cos(omega * times), cutoff_mhz)
    expected = omega**2 / (omega**2 + mu**2)**2
    np.testing.assert_allclose(result["values"], expected, rtol=1e-6)


def test_weights_integrate_the_linear_interpolation_exactly_with_t1():
    # A coarse, uneven grid: the weights integrate K(t) e^{t/T1} C(t) for C linear between samples.
    times = np.array([0., .7, 2., 2.5, 6., 11.])
    correlation = np.array([1., .3, -.4, .2, .5, .1])
    cutoff_mhz, lifetime_us = .02, 40.
    mu = 2 * np.pi * cutoff_mhz
    dense = np.linspace(0, times[-1], 400001)
    kernel = np.exp(-mu * dense) * (1 - mu * dense) / (2 * mu) * np.exp(dense / lifetime_us)
    expected = trapezoid(kernel * np.interp(dense, times, correlation), x=dense)
    weights = agp_kernel_weights(times, cutoff_mhz, lifetime_us)
    np.testing.assert_allclose(weights @ correlation, expected, rtol=1e-9)
    result = finite_time_agp(times, correlation, cutoff_mhz, covariance=np.eye(len(times)),
                             lifetime_us=lifetime_us)
    np.testing.assert_allclose(result["standard_error"], np.linalg.norm(weights))
    with np.testing.assert_raises(ValueError):
        agp_kernel_weights(times + 1, cutoff_mhz)  # must start at 0
    with np.testing.assert_raises(ValueError):
        agp_kernel_weights(times, cutoff_mhz, lifetime_us=1.)  # 1/T1 above mu


def test_calibration_factors_match_the_delta_method_by_finite_differences():
    rng = np.random.default_rng(3)
    matrix = .8 * np.eye(4) + .05 + rng.uniform(0, .02, (4, 4))
    matrix /= matrix.sum(axis=1, keepdims=True)
    raw = rng.dirichlet(np.ones(4), size=3)
    values, shots = np.arange(4.)**2, np.array([500, 600, 700, 800])
    corrected = np.linalg.solve(matrix.T, raw.T).T
    factors = calibration_error_factors(corrected, matrix, shots, values)

    def moments(confusion):
        return np.linalg.solve(confusion.T, raw.T).T @ values

    # Covariance by the delta method, from a numerical Jacobian over each prepared row.
    expected = np.zeros((3, 3))
    for row, count in enumerate(shots):
        jacobian = np.zeros((3, 4))
        for column in range(4):
            step = np.zeros((4, 4))
            step[row, column] = 1e-7
            jacobian[:, column] = (moments(matrix + step) - moments(matrix - step)) / 2e-7
        covariance = (np.diag(matrix[row]) - np.outer(matrix[row], matrix[row])) / (count - 1)
        expected += jacobian @ covariance @ jacobian.T
    np.testing.assert_allclose(factors @ factors.T, expected, rtol=1e-6)


def test_exponential_background_fit_recovers_amplitude_and_rate():
    from fitting.qsim.agp import fit_exponential_background

    rng = np.random.default_rng(2)
    times = np.linspace(0, 245, 800)
    error = np.full(len(times), .003)
    trace = .7 * np.exp(-times / 300) + .3 * np.exp(-times / 4) * np.cos(times) + rng.normal(0, error)
    fit = fit_exponential_background(times, trace, error, start_us=30.)
    np.testing.assert_allclose([fit["amplitude"], fit["lifetime_us"]], [.7, 300], rtol=.02)
    assert fit["covariance"].shape == (2, 2) and fit["start_us"] == 30.


def test_cosine_transform_peak_area_and_sum_rule():
    from fitting.qsim.agp import cosine_transform_weights

    times = np.linspace(0, 400, 2001)
    weights, frequencies_mhz = np.array([.5, .3, .2]), np.array([0., .05, .12])
    trace = weights @ np.cos(2 * np.pi * frequencies_mhz[:, None] * times)
    grid = np.linspace(0, 1, 20001)
    for window in ("hann", "raw"):
        spectrum = cosine_transform_weights(times, grid, window) @ trace
        np.testing.assert_allclose(trapezoid(spectrum, grid), trace[0], rtol=2e-3)
        for weight, frequency in zip(weights[1:], frequencies_mhz[1:]):
            near = abs(grid - frequency) < .02
            np.testing.assert_allclose(trapezoid(spectrum[near], grid[near]), weight, rtol=.03)
            assert abs(grid[near][np.argmax(spectrum[near])] - frequency) < 1e-3


def test_cosine_transform_filtered_matches_the_spectral_agp():
    from fitting.qsim.agp import cosine_transform_weights

    cutoff_mhz, frequency_mhz = .01, .027
    times = np.linspace(0, 3000, 30001)
    grid = np.linspace(0, .2, 40001)
    spectrum = cosine_transform_weights(times, grid) @ np.cos(2 * np.pi * frequency_mhz * times)
    value = trapezoid(spectrum * grid**2 / (grid**2 + cutoff_mhz**2)**2, grid)
    expected = frequency_mhz**2 / (frequency_mhz**2 + cutoff_mhz**2)**2
    np.testing.assert_allclose(value, expected, rtol=1e-2)

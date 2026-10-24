"""Numerical checks of the direct AGP kernel against independent integrals."""

import numpy as np

from fitting.qsim.agp import finite_time_agp


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

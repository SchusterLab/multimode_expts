"""The noise per sample of each row, measured out of band (spec 5.5).

White noise of variance s^2 per sample has E|X|^2 = s^2 in every bin of the normalized
Hann-windowed spectrum; far from every level (more than a few bins) X is noise only. Shared by
the diagnosis and by fitters that weigh rows by their noise. Pure numerics.
"""
import numpy as np

from fitting.qsim.poles.matching import alias_distance


def hann_spectrum(a, dt_us):
    """-> (E, X_b(E)) with X = sum_n w_n a_n e^{+2 pi i E t_n} / sqrt(sum w^2), w Hann:
    white noise of variance s^2 per sample has E|X|^2 = s^2 in every bin."""
    window = np.hanning(a.shape[1])
    X = np.fft.ifft(a * window, axis=1) * a.shape[1] / np.sqrt(np.sum(window ** 2))
    return np.fft.fftfreq(a.shape[1], d=dt_us), X


def in_band(frequencies_MHz, levels_MHz, bin_MHz, margin_bins, dt_us):
    """-> True where a frequency is within ``margin_bins`` of a level (alias-wrapped)."""
    distance = np.abs(alias_distance(frequencies_MHz, levels_MHz, 1 / dt_us))
    return distance.min(axis=1) <= margin_bins * bin_MHz


def row_noise(a, dt_us, levels_MHz, margin_bins):
    """-> s_b, each row's noise per sample: the rms of X_b(E) out of band."""
    frequencies, X = hann_spectrum(a, dt_us)
    band = in_band(frequencies, levels_MHz, 1 / (a.shape[1] * dt_us), margin_bins, dt_us)
    return np.sqrt(np.mean(np.abs(X[:, ~band]) ** 2, axis=1))

"""The result every fitter returns, and the numerics the fitters share.

Spec: ``docs/qsim/pole_finding.md``, sections 1 and 4. A pole is
``z = exp((-gamma - 2 pi i E) dt)``; a row is ``a_b[n] = sum_lambda c_{b,lambda} z_lambda^n``.
Frequencies are principal aliases, in ``[-1/(2 dt), 1/(2 dt))``.

Pure numerics.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoleFit:
    """Poles shared by all rows, and each row's amplitudes on them.

    ``amplitudes[b, lambda]`` is normalized to ``A_b(0)``: for a diagonal row it
    is ``<b|P_lambda|b>``. ``rank`` is the number of poles the rank rule chose.
    ``row_offsets_MHz[b]`` is the frequency offset a fitter found for row b (fitter C;
    None for the others): row b sees each pole at ``E_lambda + delta_b``.
    """
    frequencies_MHz: np.ndarray
    decays_per_us: np.ndarray
    amplitudes: np.ndarray
    rank: int
    frequency_errors_MHz: np.ndarray | None = None
    row_offsets_MHz: np.ndarray | None = None

    @property
    def weights(self):
        """-> w_lambda = Re sum_b c_{b,lambda}, the pole weights (spec 3)."""
        return np.real(np.sum(self.amplitudes, axis=0))

    def returns(self, time_us):
        """-> the fitted rows a_b(t) = exp(-2 pi i delta_b t) sum_lambda c_{b,lambda}
        exp((-gamma_lambda - 2 pi i E_lambda) t), normalized as the fit was (a_b = A_b / A_b(0))."""
        time_us = np.asarray(time_us, dtype=float)
        V = np.exp(np.outer(-self.decays_per_us - 2j * np.pi * self.frequencies_MHz, time_us))
        a = self.amplitudes @ V
        if self.row_offsets_MHz is None:
            return a
        return a * np.exp(-2j * np.pi * np.outer(self.row_offsets_MHz, time_us))


def sample_time(time_us):
    """-> dt of a uniform grid that starts at 0 (the one shape check of a fit)."""
    time_us = np.asarray(time_us, dtype=float)
    dt_us = time_us[1] - time_us[0]
    if time_us[0] != 0 or not np.allclose(np.diff(time_us), dt_us):
        raise ValueError("time_us must be uniform and start at 0")
    return float(dt_us)


def check_shape(A, time_us):
    """-> A as complex (row x time); refuses a shape that does not match time_us."""
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[1] != len(time_us):
        raise ValueError("A must have shape (row, len(time_us))")
    return A


def normalize_to_initial_return(A):
    """-> a_b[n] = A_b[n] / A_b[0]: every row as a diagonal row, weight 1 at t = 0."""
    return A / A[:, :1]


def wrap_frequency(frequencies_MHz, dt_us):
    """-> E wrapped to the principal alias, [-1/(2 dt), 1/(2 dt))."""
    sampling_MHz = 1 / dt_us
    return (np.asarray(frequencies_MHz) + sampling_MHz / 2) % sampling_MHz - sampling_MHz / 2


def poles_from(frequencies_MHz, decays_per_us, dt_us):
    """-> z = exp((-gamma - 2 pi i E) dt)."""
    return np.exp((-np.asarray(decays_per_us) - 2j * np.pi * np.asarray(frequencies_MHz)) * dt_us)


def frequencies_and_decays(poles, dt_us):
    """-> (E, gamma) of poles z: E = -arg(z) / (2 pi dt), gamma = -ln|z| / dt, E wrapped."""
    frequencies_MHz = wrap_frequency(-np.angle(poles) / (2 * np.pi * dt_us), dt_us)
    return frequencies_MHz, -np.log(np.abs(poles)) / dt_us


def least_squares_amplitudes(a, poles):
    """-> c (row x pole) minimizing sum_n |a_b[n] - sum_lambda c_{b,lambda} z_lambda^n|^2."""
    vandermonde = poles[None, :] ** np.arange(a.shape[1])[:, None]
    amplitudes, *_ = np.linalg.lstsq(vandermonde, a.T, rcond=None)
    return amplitudes.T


def pole_fit(poles, a, dt_us, rank):
    """-> the PoleFit of fixed poles: amplitudes by least squares, sorted by frequency."""
    frequencies_MHz, decays_per_us = frequencies_and_decays(poles, dt_us)
    order = np.argsort(frequencies_MHz)
    amplitudes = least_squares_amplitudes(a, poles[order])
    return PoleFit(frequencies_MHz[order], decays_per_us[order], amplitudes, int(rank))

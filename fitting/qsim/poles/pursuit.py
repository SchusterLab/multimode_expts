"""Fitter F: pole pursuit with real, non-negative amplitudes (spec 4.8).

What the diagnosis and the design calculator taught (spec 5.5, 5.6), in one model:

    a_b(t) = exp(-2 pi i delta_b t) sum_lambda c_{b,lambda} exp((-gamma_lambda - 2 pi i E_lambda) t),

c_{b,lambda} = <b|P_lambda|b> real and >= 0 for a diagonal row, delta_b a row offset with the
calibration's prior, and white noise s_b per sample measured out of band, so that chi^2 is in
absolute units. The pencil only starts it (it merges close pairs, and splits smeared levels at
high SNR); the pole count is then chosen by chi^2:

1. Start from C (complex): its poles and row offsets (``joint_refined.fit``).
2. Refine E, gamma and the offsets, the amplitudes by non-negative least squares per row
   (``refine``).
3. Drop every pole whose removal costs less than ``drop_chi2`` (``prune``).
4. The candidate: the frequency where one more pole, real and >= 0 in each row, lowers chi^2
   most, on a fine grid, from the rows' residuals (``best_candidate``). Keep it if the refined
   chi^2 drops by at least ``add_chi2``; then 3 again. Stop when no candidate pays.

Pure numerics.
"""
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares, nnls

from fitting.qsim.poles import joint_refined
from fitting.qsim.poles.joint_refined import remove_offsets
from fitting.qsim.poles.noise import row_noise
from fitting.qsim.poles.pole_fit import (PoleFit, check_shape, normalize_to_initial_return, sample_time,
                                         wrap_frequency)


class PursuitSettings(BaseModel):
    """The free parameters of fitter F."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The fit that gives the starting poles and offsets (C, complex).
    start: joint_refined.JointRefinedSettings = joint_refined.JointRefinedSettings()
    #: Prior width of each group's offset (MHz), as C.
    offset_prior_MHz: Annotated[float, Field(gt=0.)] = 0.5e-3
    #: c >= 0 (true for diagonal rows); False: real only. >= 0 keeps false poles out, as they
    #: often need negative amplitudes.
    nonnegative: bool = True
    #: A new pole stays if chi^2 drops by at least this. Lower: more levels and more false poles.
    add_chi2: Annotated[float, Field(gt=0.)] = 25.
    #: A pole goes if removing it (the rest fixed) costs less than this.
    drop_chi2: Annotated[float, Field(gt=0.)] = 25.
    #: Candidate grid: this many points per FFT bin.
    grid_per_bin: Annotated[int, Field(ge=1)] = 16
    max_poles: Annotated[int, Field(ge=1)] = 80
    #: Out of band (for the noise): farther than this many bins from every starting pole.
    band_margin_bins: Annotated[float, Field(gt=0.)] = 4.


@dataclass(frozen=True)
class Poles:
    """A state of the pursuit: poles, group offsets, and chi^2 (with the offsets' prior)."""
    frequencies_MHz: np.ndarray
    decays_per_us: np.ndarray
    offsets_MHz: np.ndarray
    chi2: float


def fit(A, time_us, settings=PursuitSettings(), row_groups=None):
    """-> the PoleFit of rows ``A``; ``row_groups`` labels rows sharing one offset (default: each row)."""
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(check_shape(A, time_us))
    groups = np.unique(np.arange(len(a)) if row_groups is None else np.asarray(row_groups), return_inverse=True)[1]
    start = joint_refined.fit(A, time_us, settings.start, row_groups)
    noise = row_noise(a, dt_us, start.frequencies_MHz, settings.band_margin_bins)
    group_offsets = np.bincount(groups, weights=start.row_offsets_MHz) / np.bincount(groups)
    data = (a, np.asarray(time_us, dtype=float), groups, noise)
    poles = prune(refine(data, start.frequencies_MHz, start.decays_per_us, group_offsets, settings), data, settings)
    while len(poles.frequencies_MHz) < settings.max_poles:
        candidate = best_candidate(data, poles, settings)
        trial = refine(data, np.r_[poles.frequencies_MHz, candidate], np.r_[poles.decays_per_us, np.median(poles.decays_per_us)],
                       poles.offsets_MHz, settings)
        if poles.chi2 - trial.chi2 < settings.add_chi2:
            break
        poles = prune(trial, data, settings)
    return pole_fit(data, poles, settings)


def amplitudes(a, time_us, frequencies_MHz, decays_per_us, nonnegative):
    """-> (residual a - V c, c) per row, c real (>= 0 if ``nonnegative``): least squares on
    [Re V; Im V] c = [Re a_b; Im a_b], V_{n,lambda} = exp((-gamma - 2 pi i E) t_n)."""
    V = np.exp(np.outer(time_us, -np.asarray(decays_per_us) - 2j * np.pi * np.asarray(frequencies_MHz)))
    stacked = np.vstack([V.real, V.imag])
    rhs = np.vstack([a.T.real, a.T.imag])
    if nonnegative:
        c = np.stack([nnls(stacked, column, maxiter=50 * stacked.shape[1])[0] for column in rhs.T], axis=1)
    else:
        c = np.linalg.lstsq(stacked, rhs, rcond=None)[0]
    return a - (V @ c).T, c.T


def residuals(p, data, M, settings):
    """-> the whitened residual (real, imaginary) of every row, and delta_g / sigma_cal."""
    a, time_us, groups, noise = data
    offsets = p[2 * M:]
    r = amplitudes(remove_offsets(a, time_us, offsets[groups]), time_us, p[:M], p[M:2 * M], settings.nonnegative)[0]
    r = r / noise[:, None]
    return np.concatenate([r.real.ravel(), r.imag.ravel(), offsets / settings.offset_prior_MHz])


def refine(data, frequencies_MHz, decays_per_us, offsets_MHz, settings):
    """-> the Poles minimizing chi^2 = sum_b |a_b e^{2 pi i delta t} - V c_b|^2 / s_b^2 + sum (delta / sigma)^2."""
    M = len(frequencies_MHz)
    x0 = np.concatenate([frequencies_MHz, np.maximum(decays_per_us, 0.), offsets_MHz])
    scale = np.concatenate([np.full(M, 1e-4), np.full(M, 1e-3), np.full(len(offsets_MHz), 1e-4)])
    solution = least_squares(residuals, x0, x_scale=scale, args=(data, M, settings))
    p = solution.x
    return Poles(p[:M], p[M:2 * M], p[2 * M:], float(2 * solution.cost))


def chi2_without(data, poles, k, settings):
    """-> chi^2 with pole k removed, the rest and the offsets fixed, the amplitudes refitted."""
    keep = np.arange(len(poles.frequencies_MHz)) != k
    p = np.concatenate([poles.frequencies_MHz[keep], poles.decays_per_us[keep], poles.offsets_MHz])
    return float(np.sum(residuals(p, data, int(keep.sum()), settings) ** 2))


def prune(poles, data, settings):
    """-> the Poles after dropping, one at a time and refining after each, the pole whose
    removal costs least, while that cost is under ``drop_chi2``."""
    while len(poles.frequencies_MHz) > 1:
        costs = [chi2_without(data, poles, k, settings) - poles.chi2 for k in range(len(poles.frequencies_MHz))]
        k = int(np.argmin(costs))
        if costs[k] >= settings.drop_chi2:
            break
        keep = np.arange(len(costs)) != k
        poles = refine(data, poles.frequencies_MHz[keep], poles.decays_per_us[keep], poles.offsets_MHz, settings)
    return poles


def best_candidate(data, poles, settings):
    """-> the E maximizing sum_b g_b(E) / s_b^2, the chi^2 gain of one more real pole in row b:

        g_b(E) = max(Re <v_E, r_b>, 0)^2 / <v_E, v_E>,   v_E(t) = exp((-gamma_0 - 2 pi i E) t),

    r_b the offset-corrected residual, gamma_0 the median decay; the inner products for all E at
    once by a zero-padded FFT (without the max for real amplitudes)."""
    a, time_us, groups, noise = data
    corrected = remove_offsets(a, time_us, poles.offsets_MHz[groups])
    r = amplitudes(corrected, time_us, poles.frequencies_MHz, poles.decays_per_us, settings.nonnegative)[0]
    gamma = max(float(np.median(poles.decays_per_us)), 0.)
    envelope = np.exp(-gamma * time_us)
    n_fft = settings.grid_per_bin * len(time_us)
    projections = np.fft.ifft(r * envelope, n=n_fft, axis=1).real * n_fft        # Re <v_E, r_b>
    if settings.nonnegative:
        projections = np.maximum(projections, 0.)
    gain = np.sum(projections ** 2 / noise[:, None] ** 2, axis=0) / np.sum(envelope ** 2)
    return float(np.fft.fftfreq(n_fft, d=time_us[1] - time_us[0])[np.argmax(gain)])


def pole_fit(data, poles, settings):
    """-> the PoleFit of the final Poles, sorted by frequency, amplitudes real."""
    a, time_us, groups, _ = data
    corrected = remove_offsets(a, time_us, poles.offsets_MHz[groups])
    c = amplitudes(corrected, time_us, poles.frequencies_MHz, poles.decays_per_us, settings.nonnegative)[1]
    E = wrap_frequency(poles.frequencies_MHz, time_us[1] - time_us[0])
    order = np.argsort(E)
    return PoleFit(E[order], poles.decays_per_us[order], c[:, order], len(E), None, poles.offsets_MHz[groups])

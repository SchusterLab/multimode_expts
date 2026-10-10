"""What a measurement design could resolve, before building a fitter or measuring (spec 5.6).

The Cramer-Rao bound of ``fitting.qsim.poles.resolution``, with the amplitudes as explicit
parameters, so that what we know about them can enter:

- ``complex``: free complex amplitudes, as every fitter assumes (equal to ``resolution``);
- ``real``: c_{b,lambda} = <b|P_lambda|b> is real for a diagonal row in the right phase frame;
- ``real_sums``: real, and sum_b c_{b,lambda} = m_lambda (the multiplicity) for a complete basis.

And what we assume of the row offsets:

- ``free``: one per row, Gaussian prior ``offset_prior_MHz``;
- ``per_photon``: delta_b = sum_i n_{b,i} eps_i + u_b, eps free (a weak prior fixes only the
  frame), u_b with the prior: a systematic shift per photon in each mode plus a random part.

Each gap uses the levels within ``window_bins`` of its ends, as ``resolution``. Pure numerics.
"""
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.linalg import null_space

from fitting.qsim.poles.resolution import MAX_CONDITION, clusters, scaled_condition

#: The prior of the per-photon shifts: weak, it only fixes the frame (sum_i n_i = N for every row).
PATTERN_PRIOR_MHz = 10e-3


class Design(BaseModel):
    """What the bound assumes of the amplitudes, offsets and decay."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    amplitudes: Literal["complex", "real", "real_sums"] = "complex"
    offsets: Literal["free", "per_photon"] = "free"
    #: Prior width of each row's (random) offset (MHz): the calibration's error.
    offset_prior_MHz: Annotated[float, Field(gt=0.)] = 0.5e-3
    #: The decay of every level (per us).
    decay_per_us: Annotated[float, Field(ge=0.)] = 0.005
    #: Levels within this many bins of a gap's ends enter its bound.
    window_bins: Annotated[float, Field(gt=0.)] = 3.
    #: A level is resolvable if both its gaps are at least this many of their errors.
    min_gap_snr: Annotated[float, Field(gt=0.)] = 4.


def parameter_count(rows, levels, modes, design):
    """-> (offset parameters, amplitude parameters) after the 2 x levels frequencies and decays."""
    offsets = rows + (modes if design.offsets == "per_photon" else 0)
    return offsets, rows * levels * (2 if design.amplitudes == "complex" else 1)


def fisher_matrix(time_us, levels_MHz, weights, occupations, noise, design):
    """-> F = sum_b 2 Re(J_b^h J_b) / s_b^2 + priors, over (E, gamma, offsets, amplitudes), at
    delta = 0 and c = the model's weights; J_b the derivatives of

        a_b(t) = exp(-2 pi i delta_b t) sum_lambda c_{b,lambda} exp((-gamma_lambda - 2 pi i E_lambda) t).
    """
    t = np.asarray(time_us)[:, None]
    R, M = weights.shape
    modes = occupations.shape[1]
    n_offsets, n_amplitudes = parameter_count(R, M, modes, design)
    P, first_amplitude = 2 * M + n_offsets + n_amplitudes, 2 * M + n_offsets
    Z = np.exp(t * (-design.decay_per_us - 2j * np.pi * np.asarray(levels_MHz)))
    F = np.zeros((P, P))
    for b, c in enumerate(weights):
        J = np.zeros((len(t), P), dtype=complex)
        J[:, :M], J[:, M:2 * M] = -2j * np.pi * t * Z * c, -t * Z * c
        roll = -2j * np.pi * t[:, 0] * (Z @ c)
        if design.offsets == "per_photon":
            J[:, 2 * M:2 * M + modes] = roll[:, None] * occupations[b]
        J[:, first_amplitude - R + b] = roll
        J[:, first_amplitude + b * M:first_amplitude + (b + 1) * M] = Z
        if design.amplitudes == "complex":
            J[:, first_amplitude + (R + b) * M:first_amplitude + (R + b + 1) * M] = 1j * Z
        F += 2 * np.real(J.conj().T @ J) / noise[b] ** 2
    prior = np.zeros(P)
    prior[first_amplitude - R:first_amplitude] = design.offset_prior_MHz ** -2
    if design.offsets == "per_photon":
        prior[2 * M:2 * M + modes] = PATTERN_PRIOR_MHz ** -2
    return F + np.diag(prior)


def sum_constraints(rows, levels, modes, design):
    """-> C with C p = 0 on a change p of the parameters for sum_b c_{b,lambda} fixed (``real_sums``)."""
    n_offsets, n_amplitudes = parameter_count(rows, levels, modes, design)
    first_amplitude = 2 * levels + n_offsets
    C = np.zeros((levels, first_amplitude + n_amplitudes))
    for k in range(levels):
        C[k, first_amplitude + k + levels * np.arange(rows)] = 1.
    return C


def covariance(time_us, levels_MHz, weights, occupations, noise, design):
    """-> the covariance bound of the parameters (on the constraint's null space for
    ``real_sums``), or None when F is beyond double precision (``scaled_condition``)."""
    F = fisher_matrix(time_us, levels_MHz, weights, occupations, noise, design)
    U = np.eye(len(F))
    if design.amplitudes == "real_sums":
        U = null_space(sum_constraints(*weights.shape, occupations.shape[1], design))
    reduced = U.T @ F @ U
    if scaled_condition(reduced) > MAX_CONDITION:
        return None
    return U @ np.linalg.inv(reduced) @ U.T


def gap_errors(time_us, levels_MHz, weights, occupations, noise, bin_MHz, design):
    """-> the bound on each adjacent gap E_{k+1} - E_k (MHz; inf beyond double precision), levels sorted."""
    levels_MHz = np.asarray(levels_MHz)
    near = np.abs(np.subtract.outer(levels_MHz, levels_MHz)) <= design.window_bins * bin_MHz
    errors = np.full(len(levels_MHz) - 1, np.inf)
    for k in range(len(levels_MHz) - 1):
        subset = np.flatnonzero(near[k] | near[k + 1])
        cov = covariance(time_us, levels_MHz[subset], weights[:, subset], occupations, noise, design)
        if cov is not None:
            i, j = np.searchsorted(subset, [k, k + 1])
            errors[k] = np.sqrt(abs(cov[i, i] + cov[j, j] - 2 * cov[i, j]))
    return errors


def design_summary(time_us, levels_MHz, weights, occupations, noise, bin_MHz, design):
    """-> dict: resolvable levels (both gaps at least ``min_gap_snr`` errors), the median gap
    error (kHz), and the small gaps (under half the mean gap, the ones P(r < r0) counts)
    resolved of all small gaps."""
    gaps = np.diff(levels_MHz)
    errors = gap_errors(time_us, levels_MHz, weights, np.asarray(occupations, dtype=float), noise, bin_MHz, design)
    small = gaps < 0.5 * np.mean(gaps)
    return dict(resolvable=sum(len(g) == 1 for g in clusters(gaps / errors, design.min_gap_snr)),
                median_gap_error_kHz=1e3 * float(np.median(errors)),
                small_gaps=int(small.sum()), small_gaps_resolved=int(np.sum(small & (gaps >= design.min_gap_snr * errors))))

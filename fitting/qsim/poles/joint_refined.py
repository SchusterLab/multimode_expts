"""Fitter C: the joint pencil, then a joint refinement with one offset per row group (spec 4.2).

B assumes all rows share the same poles exactly. With a frequency offset delta_g per row
group (the Stark calibration's error) a level sits at slightly different frequencies in
different rows, and B splits it. C fits the offsets: nonlinear least squares of

    a_b(t) = exp(-2 pi i delta_g(b) t) sum_lambda c_{b,lambda} exp((-gamma_lambda - 2 pi i E_lambda) t)

over E, gamma and delta (the amplitudes c by linear least squares inside), each delta with a
Gaussian prior of mean 0 and width sigma_cal. Then the rows are corrected by their offsets and
B's pencil runs again (so that a split level can merge), and the refinement repeats.

Pure numerics.
"""
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares

from fitting.qsim.poles import joint_pencil
from fitting.qsim.poles.pole_fit import PoleFit, check_shape, normalize_to_initial_return, sample_time


class JointRefinedSettings(BaseModel):
    """The free parameters of fitter C."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The pencil that gives the starting poles (B).
    pencil: joint_pencil.JointPencilSettings = joint_pencil.JointPencilSettings()
    #: Prior width of each group's offset (MHz): the Stark calibration's error. Larger lets
    #: the offsets absorb more (and trade against the poles); smaller trusts the calibration.
    offset_prior_MHz: Annotated[float, Field(gt=0.)] = 0.5e-3
    #: Pencil-then-refine rounds; the second pencil runs on offset-corrected rows.
    rounds: Annotated[int, Field(ge=1)] = 2
    #: ``real``: c_{b,lambda} = <b|P_lambda|b> of a diagonal row is real in the right phase frame
    #: (after the offsets); half the amplitude freedom, so close levels split better (spec 5.6).
    #: Wrong if the rows carry a phase drift that is not linear in t. A last refinement, from
    #: the complex rounds' solution.
    amplitudes: Literal["complex", "real"] = "complex"


def fit(A, time_us, settings=JointRefinedSettings(), row_groups=None):
    """-> the PoleFit of rows ``A``; ``row_groups`` labels rows sharing one offset (default: each row)."""
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(check_shape(A, time_us))
    groups = np.unique(np.arange(len(a)) if row_groups is None else np.asarray(row_groups), return_inverse=True)[1]
    offsets = np.zeros(groups.max() + 1)
    for _ in range(settings.rounds):
        start = joint_pencil.fit(remove_offsets(a, time_us, offsets[groups]), time_us, settings.pencil)
        frequencies, decays, offsets, errors = refine_with_group_offsets(
            a, time_us, groups, start.frequencies_MHz, start.decays_per_us, offsets, settings.offset_prior_MHz)
    real = settings.amplitudes == "real"
    if real:   # a polish from the complex solution: from B's split start it finds wrong offsets
        frequencies, decays, offsets, errors = refine_with_group_offsets(
            a, time_us, groups, frequencies, decays, offsets, settings.offset_prior_MHz, real)
    corrected = remove_offsets(a, time_us, offsets[groups])
    amplitudes = variable_projection(corrected, time_us, frequencies, decays, real)[1]
    order = np.argsort(frequencies)
    return PoleFit(frequencies[order], decays[order], amplitudes[:, order], len(frequencies), errors[order],
                   offsets[groups])


def remove_offsets(a, time_us, offsets_MHz):
    """-> a_b(t) exp(+2 pi i delta_b t)."""
    return a * np.exp(2j * np.pi * np.outer(offsets_MHz, time_us))


def variable_projection(a, time_us, frequencies_MHz, decays_per_us, real=False):
    """-> (residual a - V c, c) with V_{n,lambda} = exp((-gamma - 2 pi i E) t_n), c by least squares;
    ``real``: c real, from [Re V; Im V] c = [Re a; Im a]."""
    V = np.exp(np.outer(time_us, -np.asarray(decays_per_us) - 2j * np.pi * np.asarray(frequencies_MHz)))
    if real:
        c = np.linalg.lstsq(np.vstack([V.real, V.imag]), np.vstack([a.T.real, a.T.imag]), rcond=None)[0]
    else:
        c = np.linalg.lstsq(V, a.T, rcond=None)[0]
    return a - (V @ c).T, c.T


def refine_with_group_offsets(a, time_us, groups, frequencies_MHz, decays_per_us, offsets_MHz, prior_MHz, real=False):
    """-> (E, gamma, delta per group, frequency errors) minimizing

        sum_b |a_b e^{2 pi i delta_g(b) t} - V c_b|^2 / s^2 + sum_g (delta_g / sigma_cal)^2,

    s the noise per sample, taken from the starting residual. Frequency errors from the
    Jacobian at the solution (``frequency_errors``)."""
    M = len(frequencies_MHz)
    s = np.std(variable_projection(remove_offsets(a, time_us, offsets_MHz[groups]), time_us,
                                   frequencies_MHz, decays_per_us, real)[0]) + 1e-12

    def residuals(p):
        r = variable_projection(remove_offsets(a, time_us, p[2 * M:][groups]), time_us, p[:M], p[M:2 * M], real)[0] / s
        return np.concatenate([r.real.ravel(), r.imag.ravel(), p[2 * M:] / prior_MHz])

    solution = least_squares(residuals, np.concatenate([frequencies_MHz, decays_per_us, offsets_MHz]),
                             x_scale=np.concatenate([np.full(M, 1e-4), np.full(M, 1e-3), np.full(len(offsets_MHz), 1e-4)]))
    p = solution.x
    return p[:M], p[M:2 * M], p[2 * M:], frequency_errors(solution.jac, M)


def frequency_errors(jacobian, pole_count):
    """-> the standard errors of the frequencies: sqrt(diag((J^T J)^-1)) at the solution."""
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    return np.sqrt(np.abs(np.diag(covariance)[:pole_count]))

"""Fitter T1: a fit of the Hamiltonian to all rows at once (plan ``docs/qsim/pole_finding_explore.md``, T1).

The model is :func:`fitting.qsim.mbr_hamiltonian.fixed_n_hamiltonian`, linear in its parameters
p = (detunings per storage mode, couplings per link, Kerr on M1): H(p) = sum_k p_k D_k. With
H = V diag(E) V^T a diagonal row b is

    a_b(t) = s_b exp(-(gamma + 2 pi i delta_g(b)) t) sum_lambda V_{b,lambda}^2 exp(-2 pi i E_lambda t),

one decay gamma for all levels, an offset delta_g per row group (Gaussian prior, as C), and a
complex scale s_b per row (projected out; it takes the noise of A_b(0) and the readout
contrast). Variant ``amplitudes="free"``: the model's levels E_lambda(p) only, the amplitudes
c_{b,lambda} >= 0 free per row (non-negative least squares, as F). chi^2 is in absolute units
(noise per row out of band, ``row_noise``). Many starts: the given parameters and random ones
around them; the lowest chi^2 wins.

The Jacobian is analytic. For the model amplitudes, by the Daleckii-Krein formula,
d <b|exp(-2 pi i H t)|b> / dp_k = sum_{lambda,mu} V_{b,lambda} V_{b,mu} F_{lambda,mu}(t) (V^T D_k V)_{lambda,mu},
F_{lambda,mu}(t) = -2 pi i t exp(-i pi (E_lambda + E_mu) t) sinc(t (E_lambda - E_mu)), smooth
through degeneracies; for free amplitudes, dE_lambda / dp_k = (V^T D_k V)_{lambda,lambda}. The
projected (s_b, c_b) are held fixed in the Jacobian (Kaufman's approximation).

The logic caveat (guan): the levels of a model fit carry the model's statistics. For
P(r < 0.25) they are a start (for F or T3) or a check of the model, never the answer.

Inputs beyond the common interface (keyword arguments of ``fit``): ``occupations``, the Fock
occupation of each row (row x mode, M1 first; photon number and mode count follow from it), and
``start``, the recorded ``ModelParameters``. Pure numerics.
"""
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares

from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles.noise import row_noise
from fitting.qsim.poles.pole_fit import (PoleFit, check_shape, normalize_to_initial_return, sample_time,
                                         wrap_frequency)
from fitting.qsim.poles.pursuit import amplitudes as nonnegative_amplitudes


class ModelParameters(BaseModel):
    """The arguments of ``fixed_n_hamiltonian`` that the fit varies (MHz; per storage mode)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    detunings_MHz: tuple[float, ...]
    couplings_MHz: tuple[float, ...]
    kerr_MHz: float

    @property
    def vector(self):
        return np.array([*self.detunings_MHz, *self.couplings_MHz, self.kerr_MHz])

    @classmethod
    def from_vector(cls, p):
        storage = (len(p) - 1) // 2
        return cls(detunings_MHz=tuple(map(float, p[:storage])), couplings_MHz=tuple(map(float, p[storage:2 * storage])),
                   kerr_MHz=float(p[-1]))


class HamiltonianFitSettings(BaseModel):
    """The free parameters of the Hamiltonian fit."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``model``: c_{b,lambda} = s_b |<b|lambda>|^2; ``free``: the model's levels, c >= 0 free per row.
    amplitudes: Literal["model", "free"] = "model"
    #: Starts: the given parameters, then ``starts - 1`` random ones around them.
    starts: Annotated[int, Field(ge=1)] = 12
    #: Spread of the random starts: detunings and Kerr (MHz) additive, couplings relative.
    detuning_spread_MHz: Annotated[float, Field(ge=0.)] = 1e-3
    coupling_spread: Annotated[float, Field(ge=0.)] = 0.05
    kerr_spread_MHz: Annotated[float, Field(ge=0.)] = 1e-3
    seed: int = 0
    decay_start_per_us: Annotated[float, Field(ge=0.)] = 0.005
    #: Prior width of each group's offset (MHz), as C and F.
    offset_prior_MHz: Annotated[float, Field(gt=0.)] = 0.5e-3
    #: Out of band (for the noise): farther than this many bins from every starting level.
    band_margin_bins: Annotated[float, Field(gt=0.)] = 4.
    #: Levels closer than this many bins are one pole of the PoleFit (as the truth of ``synthetic``).
    merge_bins: Annotated[float, Field(gt=0.)] = 1e-3
    max_nfev: Annotated[int, Field(ge=1)] = 200


@dataclass(frozen=True)
class HamiltonianFit:
    """The best start's solution. chi^2 = sum |r|^2 / s^2 + sum (delta / sigma)^2 (as F: about one
    per complex sample if the residual is noise; ``dof`` counts complex samples less half the real
    parameters). ``errors`` are 1-sigma from the Fisher matrix, not scaled by the reduced chi^2;
    ``start_chi2`` is each start's final chi^2 (how often the best is reached)."""
    parameters: ModelParameters
    errors: ModelParameters
    decay_per_us: float
    row_offsets_MHz: np.ndarray
    chi2: float
    dof: int
    start_chi2: np.ndarray
    pole_fit: PoleFit

    @property
    def reduced_chi2(self):
        return self.chi2 / self.dof


@dataclass(frozen=True)
class Model:
    """H(p) = sum_k p_k D_k on the Fock basis, and the measured rows' indices in it."""
    generators: np.ndarray
    rows: np.ndarray

    def eigen(self, p):
        return np.linalg.eigh(np.tensordot(p, self.generators, axes=1))


def model_of(occupations):
    """-> the Model of rows with these occupations (row x mode, M1 first): D_k = H at the unit vector e_k."""
    occupations = [tuple(int(round(n)) for n in row) for row in occupations]
    photons, modes = sum(occupations[0]), len(occupations[0])
    S = modes - 1
    generators = [fixed_n_hamiltonian(photons, modes, e[:S], e[S:2 * S], e[-1]).hamiltonian_MHz for e in np.eye(2 * S + 1)]
    basis = fixed_n_hamiltonian(photons, modes, np.zeros(S), np.zeros(S), 0.)
    return Model(np.stack(generators), np.array([basis.fock_index[o] for o in occupations]))


def fit(A, time_us, settings=HamiltonianFitSettings(), row_groups=None, *, occupations, start):
    """-> the PoleFit of rows ``A`` (``fit_hamiltonian``'s ``pole_fit``); ``occupations`` (row x mode)
    and ``start`` (ModelParameters, e.g. the recorded values) are required keywords."""
    return fit_hamiltonian(A, time_us, occupations, start, settings, row_groups).pole_fit


def fit_hamiltonian(A, time_us, occupations, start, settings=HamiltonianFitSettings(), row_groups=None):
    """-> the HamiltonianFit of rows ``A``: min over p, gamma, delta of
    chi^2 = sum_b |a_b - model_b|^2 / s_b^2 + sum_g (delta_g / sigma)^2, from ``settings.starts`` starts."""
    dt_us = sample_time(time_us)
    time_us = np.asarray(time_us, dtype=float)
    a = normalize_to_initial_return(check_shape(A, time_us))
    groups = np.unique(np.arange(len(a)) if row_groups is None else np.asarray(row_groups), return_inverse=True)[1]
    model = model_of(occupations)
    noise = row_noise(a, dt_us, model.eigen(start.vector)[0], settings.band_margin_bins)
    data = (a, time_us, groups, noise, model)
    solutions = [solve(data, p, settings) for p in starting_points(start, settings)]
    chi2 = np.array([2 * s.cost for s in solutions])
    best = solutions[int(np.argmin(chi2))]
    K = len(start.vector)
    errors = parameter_errors(best.jac, groups.max() + 1)
    return HamiltonianFit(ModelParameters.from_vector(best.x[:K]), ModelParameters.from_vector(errors[:K]),
                          float(best.x[K]), best.x[K + 1:][groups], float(chi2.min()),
                          int(a.size - (len(best.x) + 2 * len(a)) / 2), chi2, pole_fit(data, best.x, errors, settings))


def starting_points(start, settings):
    """-> the given p, then p + spread x N(0, 1) (couplings: p (1 + spread N(0, 1)))."""
    rng = np.random.default_rng(settings.seed)
    p0, S = start.vector, len(start.detunings_MHz)
    spread = np.r_[np.full(S, settings.detuning_spread_MHz), settings.coupling_spread * np.abs(p0[S:2 * S]),
                   settings.kerr_spread_MHz]
    return [p0] + [p0 + spread * rng.normal(size=len(p0)) for _ in range(settings.starts - 1)]


def solve(data, p0, settings):
    """-> scipy's least_squares result over x = (p, gamma, delta_g) from p0."""
    groups, K = data[2], len(p0)
    x0 = np.r_[p0, settings.decay_start_per_us, np.zeros(groups.max() + 1)]
    lower = np.r_[np.full(K, -np.inf), 0., np.full(groups.max() + 1, -np.inf)]
    scale = np.r_[np.full(K, 1e-3), 1e-3, np.full(groups.max() + 1, 1e-4)]
    function = residual_and_jacobian_model if settings.amplitudes == "model" else residual_and_jacobian_free
    cache = {}

    def evaluate(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = function(x, data, settings)
        return cache[key]
    return least_squares(lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], bounds=(lower, np.inf),
                         x_scale=scale, max_nfev=settings.max_nfev)


def _split(x, data):
    K = data[4].generators.shape[0]
    return x[:K], x[K], x[K + 1:]


def _stack(r, J_rows, offsets, settings):
    """-> the real residual [Re r; Im r; delta / sigma] and its Jacobian from complex parts."""
    prior = offsets / settings.offset_prior_MHz
    J_prior = np.zeros((len(offsets), J_rows.shape[-1]))
    J_prior[:, -len(offsets):] = np.eye(len(offsets)) / settings.offset_prior_MHz
    J = J_rows.reshape(-1, J_rows.shape[-1])
    return np.r_[r.real.ravel(), r.imag.ravel(), prior], np.vstack([J.real, J.imag, J_prior])


def residual_and_jacobian_model(x, data, settings):
    """-> (residual, Jacobian) of the model amplitudes, s_b projected out."""
    a, time_us, groups, noise, model = data
    p, gamma, offsets = _split(x, data)
    E, V = model.eigen(p)
    Vb = V[model.rows]                                                   # (row, level)
    phase = np.exp(np.outer(-2j * np.pi * E, time_us))                   # (level, t)
    envelope = np.exp(-gamma * time_us[None, :] - 2j * np.pi * np.outer(offsets[groups], time_us))
    m = envelope * ((Vb ** 2) @ phase)                                   # (row, t)
    s = np.sum(np.conj(m) * a, axis=1) / np.sum(np.abs(m) ** 2, axis=1)
    fitted = s[:, None] * m
    r = (a - fitted) / noise[:, None]
    # Daleckii-Krein: dm_b/dp_k = envelope sum_{l,u} Vb_l Vb_u F_lu(t) G_k,lu
    G = np.einsum("al,kab,bu->klu", V, model.generators, V)
    mean, diff = 0.5 * (E[:, None] + E[None, :]), E[:, None] - E[None, :]
    F = (-2j * np.pi * time_us[:, None, None] * np.exp(-2j * np.pi * mean[None] * time_us[:, None, None])
         * np.sinc(diff[None] * time_us[:, None, None]))                # (t, l, u)
    L = len(E)
    X = (Vb[:, None, :, None] * Vb[:, None, None, :] * G[None]).reshape(len(Vb) * len(p), L * L)
    dm_dp = (X @ F.reshape(len(time_us), L * L).T).reshape(len(Vb), len(p), len(time_us))
    dm_dp = dm_dp * envelope[:, None, :]
    J = np.zeros((len(a), len(time_us), len(x)), dtype=complex)
    J[:, :, :len(p)] = np.moveaxis(dm_dp, 1, 2)
    J[:, :, len(p)] = -time_us[None, :] * m
    rows = np.arange(len(a))
    J[rows, :, len(p) + 1 + groups] = -2j * np.pi * time_us[None, :] * m
    J = -J * (s / noise)[:, None, None]
    J -= m[:, :, None] * (np.einsum("bt,btk->bk", np.conj(m), J) / np.sum(np.abs(m) ** 2, axis=1)[:, None])[:, None, :]
    return _stack(r, J, offsets, settings)


def residual_and_jacobian_free(x, data, settings):
    """-> (residual, Jacobian) of the model's levels with c_{b,lambda} >= 0 free (NNLS inside)."""
    a, time_us, groups, noise, model = data
    p, gamma, offsets = _split(x, data)
    E, V = model.eigen(p)
    turn = np.exp(2j * np.pi * np.outer(offsets[groups], time_us))
    corrected = a * turn
    r, c = nonnegative_amplitudes(corrected, time_us, E, np.full(len(E), gamma), True)
    basis = np.exp(np.outer(-gamma - 2j * np.pi * E, time_us))          # (level, t)
    dE_dp = np.einsum("al,kab,bl->kl", V, model.generators, V)           # Hellmann-Feynman
    J = np.zeros((len(a), len(time_us), len(x)), dtype=complex)
    fitted_t = c @ basis                                                 # (row, t)
    J[:, :, :len(p)] = np.einsum("bl,lt,kl->btk", c, -2j * np.pi * time_us[None, :] * basis, dE_dp)
    J[:, :, len(p)] = -time_us[None, :] * fitted_t
    rows = np.arange(len(a))
    J[rows, :, len(p) + 1 + groups] = -2j * np.pi * time_us[None, :] * corrected
    # r = corrected - fitted; dr = d corrected - d fitted
    J = -J / noise[:, None, None]
    for b in range(len(a)):          # Kaufman: project out the span of row b's active levels
        active = basis[c[b] > 0].T
        if active.size:
            M = np.vstack([active.real, active.imag])
            Jb = np.vstack([J[b].real, J[b].imag])
            Jb -= M @ np.linalg.lstsq(M, Jb, rcond=None)[0]
            J[b] = Jb[:len(time_us)] + 1j * Jb[len(time_us):]
    return _stack(r / noise[:, None], J, offsets, settings)


def parameter_errors(J, priors):
    """-> sqrt diag of the inverse Fisher matrix 2 J_data^T J_data + J_prior^T J_prior (the last
    ``priors`` rows): chi^2 counts |r|^2 / s^2 per complex sample, as F, i.e. half the Gaussian
    log-likelihood's weight on the data (pseudo-inverse where a direction is not fixed)."""
    data, prior = J[:-priors], J[-priors:]
    return np.sqrt(np.abs(np.diag(np.linalg.pinv(2 * data.T @ data + prior.T @ prior))))


def pole_fit(data, x, errors, settings):
    """-> the PoleFit at x: the model's distinct levels (closer than ``merge_bins``: one pole),
    one decay, the row offsets, the amplitudes (s_b W_{b,lambda}, or c) summed over merged levels,
    and each level's error from the parameters' (uncorrelated; a guide only)."""
    a, time_us, groups, noise, model = data
    p, gamma, offsets = _split(x, data)
    E, V = model.eigen(p)
    if settings.amplitudes == "model":
        m = np.exp(-gamma * time_us[None, :] - 2j * np.pi * np.outer(offsets[groups], time_us)) * (
            (V[model.rows] ** 2) @ np.exp(np.outer(-2j * np.pi * E, time_us)))
        s = np.sum(np.conj(m) * a, axis=1) / np.sum(np.abs(m) ** 2, axis=1)
        c = s[:, None] * V[model.rows] ** 2
    else:
        corrected = a * np.exp(2j * np.pi * np.outer(offsets[groups], time_us))
        c = nonnegative_amplitudes(corrected, time_us, E, np.full(len(E), gamma), True)[1]
    bin_MHz = 1 / (len(time_us) * (time_us[1] - time_us[0]))
    order = np.argsort(E)
    E, c, V = E[order], c[:, order], V[:, order]
    group = np.concatenate(([0], np.cumsum(np.diff(E) > settings.merge_bins * bin_MHz)))
    levels = np.bincount(group, weights=E) / np.bincount(group)
    amplitudes = np.stack([np.bincount(group, weights=row.real) + 1j * np.bincount(group, weights=row.imag)
                           for row in c])
    dE_dp = np.einsum("al,kab,bl->kl", V, model.generators, V)
    level_errors = np.sqrt((dE_dp ** 2).T @ errors[:len(p)] ** 2)
    level_errors = np.bincount(group, weights=level_errors) / np.bincount(group)
    if settings.amplitudes == "free":
        amplitudes = amplitudes.real
    return PoleFit(wrap_frequency(levels, time_us[1] - time_us[0]), np.full(len(levels), gamma), amplitudes,
                   len(levels), level_errors, offsets[groups])


def true_parameters(point, direction, storage_modes=4):
    """-> the ModelParameters of a synthetic point (g, K / g, delta / g) and disorder direction,
    as ``synthetic.synthetic_returns`` builds them."""
    g, kerr_over_g, disorder_over_g = point
    return ModelParameters(detunings_MHz=tuple(-g * disorder_over_g * np.asarray(direction)),
                           couplings_MHz=(g,) * storage_modes, kerr_MHz=g * kerr_over_g)


def perturbed(parameters, detuning_MHz, coupling, kerr_MHz, seed):
    """-> parameters + a random error of these sizes (couplings relative): a stand-in for the
    recorded values on synthetic data."""
    rng = np.random.default_rng(seed)
    p, S = parameters.vector, len(parameters.detunings_MHz)
    spread = np.r_[np.full(S, detuning_MHz), coupling * np.abs(p[S:2 * S]), kerr_MHz]
    return ModelParameters.from_vector(p + spread * rng.normal(size=len(p)))

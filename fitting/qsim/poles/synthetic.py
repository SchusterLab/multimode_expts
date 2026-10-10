"""Synthetic MBR returns from our own model, with exact answers (spec 5).

A case is a point of the phase diagram (Kerr ``K / g``, disorder ``delta / g``, one
disorder draw), the hardware grid, and the non-idealities. The model is
:func:`fitting.qsim.mbr_hamiltonian.fixed_n_hamiltonian`: a star of storage modes
around M1, coupling ``g``, Kerr on M1, onsite energies ``delta u_i``.

Pure numerics.
"""
from dataclasses import dataclass
from itertools import product
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fitting.qsim.mbr_disorder import disorder_direction, select_diagonal_rows
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian

_Positive = Annotated[float, Field(gt=0., allow_inf_nan=False)]
_NonNegative = Annotated[float, Field(ge=0., allow_inf_nan=False)]


class Hardware(BaseModel):
    """The model size and time grid. Defaults: the 7-1 disorder campaign (2026-08-29)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    photon_number: int = 3
    mode_count: int = 5
    coupling_MHz: _Positive = 0.0152727
    dt_us: _Positive = 0.818452
    samples: Annotated[int, Field(ge=4)] = 100
    #: Rows measured: None for the complete basis, else this many, chosen as the
    #: campaign chose them (``select_diagonal_rows``, max-min eigenstate support).
    partial_rows: Annotated[int, Field(ge=1)] | None = None

    @property
    def bin_MHz(self):
        """The FFT bin, 1 / (N dt)."""
        return 1 / (self.samples * self.dt_us)


class Nonideal(BaseModel):
    """What separates the data from the model. The defaults are ideal."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: |A_b(0)| over the complex noise standard deviation per sample.
    snr: Annotated[float, Field(gt=0.)] = np.inf
    #: One decay rate for every level (T2 of order 100 us: 0.01 per us).
    decay_per_us: _NonNegative = 0.
    #: Width of the frequency offset of each row group (here: each row).
    offset_sigma_MHz: _NonNegative = 0.
    seed: int = 0


@dataclass(frozen=True)
class ModelPoint:
    """One disorder draw at one point of the phase diagram, in units of g."""
    kerr_over_g: float
    disorder_over_g: float
    direction: np.ndarray
    draw: int


@dataclass(frozen=True)
class Truth:
    """The distinct levels, their multiplicities, and each measured row's weight on them.

    ``conditioning`` is s_min / s_max of the Vandermonde matrix exp(-2 pi i E_lambda t_n)
    of the distinct levels on the time grid: how well the grid can tell them apart at all,
    for any fitter. Near 1: separated by several bins; below about 1e-2, errors on
    clean data grow above 1e-6 bin in double precision.
    """
    levels_MHz: np.ndarray
    multiplicities: np.ndarray
    row_weights: np.ndarray
    occupations: list
    conditioning: float


@dataclass(frozen=True)
class SyntheticCase:
    point: ModelPoint
    hardware: Hardware
    nonideal: Nonideal
    time_us: np.ndarray
    A: np.ndarray
    truth: Truth


def sample_phase_diagram(kerr_over_g, disorder_over_g, draws, hardware=Hardware(), seed=0):
    """-> ModelPoints on the grid kerr_over_g x disorder_over_g, ``draws`` each.

    Draw ``d`` has the direction ``disorder_direction(seed + d)``, the same at
    every point, so the grid varies only K and delta.
    """
    directions = [disorder_direction(seed + draw, hardware.mode_count - 1) for draw in range(draws)]
    return [ModelPoint(float(kerr), float(disorder), directions[draw], draw)
            for kerr, disorder in product(kerr_over_g, disorder_over_g) for draw in range(draws)]


def synthetic_returns(point, hardware=Hardware(), nonideal=Nonideal()):
    """-> the SyntheticCase: A_b(t) = sum_lambda <b|P_lambda|b> exp((-gamma - 2 pi i E_lambda) t),

    then a frequency offset per row, and complex white noise.
    """
    g = hardware.coupling_MHz
    model = fixed_n_hamiltonian(hardware.photon_number, hardware.mode_count,
                                -g * point.disorder_over_g * point.direction,
                                [g] * (hardware.mode_count - 1), g * point.kerr_over_g)
    rows = measured_rows(model.basis_eigenstate_weights, hardware)
    time_us = hardware.dt_us * np.arange(hardware.samples)
    weights = model.basis_eigenstate_weights[rows]
    A = weights @ np.exp(np.outer(-nonideal.decay_per_us - 2j * np.pi * model.energies_MHz, time_us))
    A = add_nonidealities(A, time_us, nonideal)
    truth = distinct_levels(model.energies_MHz, weights, 1e-3 * hardware.bin_MHz)
    occupations = [tuple(model.fock_basis[row]) for row in rows]
    conditioning = vandermonde_conditioning(truth[0], time_us)
    return SyntheticCase(point, hardware, nonideal, time_us, A, Truth(*truth, occupations, conditioning))


def measured_rows(eigenstate_weights, hardware):
    """-> the measured basis rows: all, or ``partial_rows`` by max-min support."""
    if hardware.partial_rows is None:
        return np.arange(len(eigenstate_weights))
    return select_diagonal_rows(eigenstate_weights, hardware.partial_rows)[0]


def add_nonidealities(A, time_us, nonideal):
    """-> A_b(t) exp(-2 pi i delta_b t) + noise, delta_b ~ N(0, sigma), noise ~ CN(0, 1/snr^2)."""
    rng = np.random.default_rng(nonideal.seed)
    offsets_MHz = rng.normal(0, nonideal.offset_sigma_MHz, size=len(A))
    A = A * np.exp(-2j * np.pi * np.outer(offsets_MHz, time_us))
    noise = (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape)) / np.sqrt(2)
    return A + noise / nonideal.snr


def vandermonde_conditioning(levels_MHz, time_us):
    """-> s_min / s_max of V_{n,lambda} = exp(-2 pi i E_lambda t_n)."""
    s = np.linalg.svd(np.exp(-2j * np.pi * np.outer(time_us, levels_MHz)), compute_uv=False)
    return float(s[-1] / s[0])


def distinct_levels(energies_MHz, row_weights, tolerance_MHz):
    """-> (levels, multiplicities, row weights) with energies closer than the tolerance
    as one level: its mean energy, its count, and the summed weights."""
    order = np.argsort(energies_MHz)
    energies_MHz, row_weights = energies_MHz[order], row_weights[:, order]
    group = np.concatenate(([0], np.cumsum(np.diff(energies_MHz) > tolerance_MHz)))
    multiplicities = np.bincount(group)
    levels_MHz = np.bincount(group, weights=energies_MHz) / multiplicities
    summed = np.stack([np.bincount(group, weights=row) for row in row_weights])
    return levels_MHz, multiplicities, summed

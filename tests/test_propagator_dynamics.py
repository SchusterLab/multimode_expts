# -*- coding: utf-8 -*-
"""Hamiltonian tomography from propagator matrices.

Why this file exists
--------------------
`analyze_propagator_dynamics` was the one method of 114 that the god-file
split lost outright: the `stage='propagator'` dispatch's `calibration=` branch
called it, the branch was not carried into `MBRPropagatorExperiment.analyze`,
and no definition survived anywhere in `experiments/` or `fitting/`.
`qsim_experiments.ipynb` cells 242 and 263 are its only callers, and both ask
for it by passing `calibration=`.

It is restored verbatim, so these tests are characterization rather than
specification: they pin what the method does, not that it is right. It never
had a test, which is part of why its loss was invisible.

The fixture is a *synthetic* propagator built from a known Hamiltonian, so
the two estimators have a right answer to be checked against:

    M_q = E_endpoint . exp(-2.pi.i.H.q.T) . E_endpoint

with `E_endpoint` diagonal, which is the structure the method assumes
(`<i|U(q)|j> = D_i U_q E_j`). That makes the eigenphase route exact and the
finite-difference route accurate to its own truncation error. No hardware, no
saved data.

Run:  pixi run python -m pytest tests/test_propagator_dynamics.py -v
"""
from itertools import product

import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.mbr_propagator import MBRPropagatorExperiment

FLOQUET_CYCLE_US = 0.5
STEP = 2                        # even, so [0, 2, 4] is a valid FD triple
CYCLES = [0, STEP, 2 * STEP]


def _fixed_n_basis(photon_number, mode_count):
    """Every occupation of `photon_number` photons over `mode_count` modes."""
    states = [state for state in
              product(range(photon_number + 1), repeat=mode_count)
              if sum(state) == photon_number]
    states.sort(reverse=True)
    return states


@pytest.fixture
def tomography():
    """A synthetic propagator with a Hamiltonian we know, plus its calibration.

    Two modes and two photons give a 3-dimensional fixed-N sector, which is
    the complete basis the method requires.
    """
    occupations = _fixed_n_basis(photon_number=2, mode_count=2)
    dimension = len(occupations)
    assert dimension == 3

    rng = np.random.default_rng(20260914)
    # A real symmetric H, so its eigenfrequencies are real and sortable.
    hamiltonian_MHz = rng.normal(scale=0.02, size=(dimension, dimension))
    hamiltonian_MHz = (hamiltonian_MHz + hamiltonian_MHz.T) / 2

    # Diagonal endpoint factors: the fixed decoder/encoder inefficiency the
    # method divides out. Complex, so the phase has to be handled too.
    endpoint = np.array([0.81 + 0.05j, 0.74 - 0.03j, 0.88 + 0.01j])

    matrices = []
    for cycle in CYCLES:
        time_us = cycle * FLOQUET_CYCLE_US
        evolution = _expm(-2j * np.pi * hamiltonian_MHz * time_us)
        matrices.append(endpoint[:, None] * evolution * endpoint[None, :])
    matrices = np.asarray(matrices)

    reconstruction = AttrDict(dict(
        occupations=[list(state) for state in occupations],
        mode_labels=["M1", "S1"],
        cycles=np.asarray(CYCLES, dtype=int),
        matrices=matrices,
    ))
    # The calibration supplies the q=0 self-return per occupation, which is
    # endpoint^2 for this construction.
    calibration = AttrDict(dict(
        mode_labels=["M1", "S1"],
        results=[AttrDict(dict(occupation=list(state),
                               physical_cycles=[0],
                               complex_return=[endpoint[i] ** 2]))
                 for i, state in enumerate(occupations)],
    ))
    return reconstruction, calibration, hamiltonian_MHz, endpoint


def _expm(matrix):
    """Matrix exponential by eigendecomposition; avoids a scipy import here."""
    values, vectors = np.linalg.eig(matrix)
    return vectors @ np.diag(np.exp(values)) @ np.linalg.inv(vectors)


def _run(reconstruction, calibration, **kwargs):
    return MBRPropagatorExperiment.analyze_propagator_dynamics(
        reconstruction, calibration, FLOQUET_CYCLE_US, **kwargs)


def test_eigenphase_recovers_the_known_hamiltonian(tomography):
    """The generalized eigenvalues against M_0 give H's eigenfrequencies.

    This is the estimator's whole claim: taking eigenvalues relative to the
    same batch's q=0 matrix cancels the fixed D and E factors, so the answer
    does not depend on the endpoint inefficiency at all.
    """
    reconstruction, calibration, hamiltonian_MHz, _ = tomography

    result = _run(reconstruction, calibration)

    expected = np.sort(np.linalg.eigvalsh(hamiltonian_MHz))
    got = np.sort(result.eigenphase.eigenfrequencies_MHz)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_eigenphase_poles_sit_on_the_unit_circle(tomography):
    """Unitary evolution, so no decay: |lambda| = 1.

    A radius far from 1 is how this estimator reports that the data are not
    describable by a single unitary generator.
    """
    reconstruction, calibration, _, _ = tomography

    result = _run(reconstruction, calibration)

    np.testing.assert_allclose(result.eigenphase.pole_radii, 1.0, atol=1e-9)
    assert result.eigenphase.cycle == STEP
    assert result.eigenphase.single_cycle_branch_ambiguous is True
    np.testing.assert_allclose(
        result.eigenphase.alias_period_MHz,
        1.0 / (STEP * FLOQUET_CYCLE_US))


def test_finite_difference_recovers_the_same_hamiltonian(tomography):
    """The [0, s, 2s] derivative route agrees, to its truncation error.

    It is the independent estimator: a three-point derivative rather than a
    generalized eigenproblem. Agreement is the cross-check the two exist for.

    The tolerance is not arbitrary -- it is this estimator's truncation error
    at the fixture's step, which `test_finite_difference_is_second_order`
    below measures rather than assumes. Unlike the eigenphase route, which is
    exact here, the derivative is only as good as `2.pi.H.s.T << 1`, and at
    `s.T = 1 us` with frequencies of ~0.05 MHz that product is ~0.3.
    """
    reconstruction, calibration, hamiltonian_MHz, _ = tomography

    result = _run(reconstruction, calibration)

    assert result.finite_difference is not None
    assert list(result.finite_difference.cycles) == CYCLES
    assert result.finite_difference.step_cycles == STEP
    np.testing.assert_allclose(
        result.finite_difference.step_time_us, STEP * FLOQUET_CYCLE_US)

    expected = np.sort(np.linalg.eigvalsh(hamiltonian_MHz))
    got = np.sort(result.finite_difference.eigenfrequencies_MHz)
    np.testing.assert_allclose(got, expected, atol=2e-3)


def test_finite_difference_is_second_order(tomography):
    """Halving the step quarters the error, over a 16x range.

    This is what makes the tolerance above a measurement rather than a
    fudge factor: the gap between the two estimators is truncation error and
    nothing else. A first-order gap, or a floor that stops improving, would
    mean the derivative is assembled wrong -- and that is exactly the kind of
    defect the missing test let through, since the eigenphase route would
    keep looking perfect either way.
    """
    _, calibration, hamiltonian_MHz, endpoint = tomography
    occupations = _fixed_n_basis(2, 2)
    exact = np.sort(np.linalg.eigvalsh(hamiltonian_MHz))

    errors = {}
    for cycle_us in (0.5, 0.25, 0.125, 0.0625):
        matrices = np.asarray([
            endpoint[:, None]
            * _expm(-2j * np.pi * hamiltonian_MHz * (cycle * cycle_us))
            * endpoint[None, :]
            for cycle in CYCLES])
        reconstruction = AttrDict(dict(
            occupations=[list(state) for state in occupations],
            mode_labels=["M1", "S1"],
            cycles=np.asarray(CYCLES, dtype=int),
            matrices=matrices,
        ))
        result = MBRPropagatorExperiment.analyze_propagator_dynamics(
            reconstruction, calibration, cycle_us)
        got = np.sort(result.finite_difference.eigenfrequencies_MHz)
        errors[STEP * cycle_us] = np.max(np.abs(got - exact))

    # error / step_time^2 is the same constant at every step.
    constants = [error / step_time ** 2 for step_time, error in errors.items()]
    assert max(constants) / min(constants) < 1.05, errors
    # And it really is shrinking, not just proportional to something flat.
    assert max(errors.values()) / min(errors.values()) > 50, errors


def test_semigroup_residual_is_small_for_a_real_semigroup(tomography):
    """M_2s should equal M_s M_0^-1 M_s when one generator explains the data.

    This is the diagnostic that says whether the finite-difference number
    means anything, so it must actually be near zero on clean input.
    """
    reconstruction, calibration, _, _ = tomography

    result = _run(reconstruction, calibration)

    assert result.finite_difference.semigroup_relative_residual < 1e-9


def test_endpoint_normalization_makes_the_zero_cycle_the_identity(tomography):
    """Dividing by sqrt(self-return) outer product should undo D and E at q=0."""
    reconstruction, calibration, _, endpoint = tomography

    result = _run(reconstruction, calibration)

    np.testing.assert_allclose(result.calibration_self_returns, endpoint ** 2)
    assert result.endpoint_normalized_zero_cycle_identity_residual < 1e-9
    assert result.calibration_diagonal_relative_mismatch < 1e-9
    assert result.zero_cycle_condition_number < 1e3


def test_an_explicit_eigenphase_cycle_is_honoured(tomography):
    """And it changes the alias period, which is the reason to choose it."""
    reconstruction, calibration, hamiltonian_MHz, _ = tomography

    result = _run(reconstruction, calibration, eigenphase_cycle=2 * STEP)

    assert result.eigenphase.cycle == 2 * STEP
    np.testing.assert_allclose(
        result.eigenphase.alias_period_MHz,
        1.0 / (2 * STEP * FLOQUET_CYCLE_US))
    # Still the right answer, just aliased twice as tightly.
    np.testing.assert_allclose(
        np.sort(result.eigenphase.eigenfrequencies_MHz),
        np.sort(np.linalg.eigvalsh(hamiltonian_MHz)), atol=1e-9)


@pytest.mark.parametrize("bad_cycle, why", [
    (3, "odd"),
    (0, "not positive"),
    (STEP + 1, "not acquired"),
])
def test_bad_eigenphase_cycles_are_rejected(tomography, bad_cycle, why):
    reconstruction, calibration, _, _ = tomography
    with pytest.raises(ValueError, match="eigenphase_cycle"):
        _run(reconstruction, calibration, eigenphase_cycle=bad_cycle)


def test_finite_difference_cycles_must_be_an_even_triple(tomography):
    reconstruction, calibration, _, _ = tomography
    with pytest.raises(ValueError, match="finite_difference_cycles"):
        _run(reconstruction, calibration, finite_difference_cycles=[0, 1, 2])


def test_an_incomplete_basis_is_rejected(tomography):
    """Generalized eigenanalysis needs the whole fixed-N sector.

    Dropping a row leaves a matrix that is still square-looking per cycle but
    no longer represents the sector, and the answer would be silently wrong.
    """
    reconstruction, calibration, _, _ = tomography
    reconstruction.occupations = reconstruction.occupations[:2]
    reconstruction.matrices = reconstruction.matrices[:, :2, :2]
    with pytest.raises(ValueError, match="complete fixed-N basis"):
        _run(reconstruction, calibration)


def test_a_missing_zero_cycle_is_rejected(tomography):
    """Everything here is relative to M_0, so q=0 is not optional."""
    reconstruction, calibration, _, _ = tomography
    reconstruction.cycles = np.asarray([STEP, 2 * STEP], dtype=int)
    reconstruction.matrices = reconstruction.matrices[1:]
    with pytest.raises(ValueError, match="needs q=0"):
        _run(reconstruction, calibration)


def test_a_calibration_missing_an_occupation_is_rejected(tomography):
    reconstruction, calibration, _, _ = tomography
    calibration.results = calibration.results[:-1]
    with pytest.raises(ValueError, match="calibration is missing"):
        _run(reconstruction, calibration)


def test_mismatched_cycle_times_are_rejected(tomography):
    """The calibration and the propagator must describe the same Floquet cycle."""
    reconstruction, calibration, _, _ = tomography
    calibration.hardware = AttrDict(dict(
        floquet_cycle_us=FLOQUET_CYCLE_US * 1.1))
    with pytest.raises(ValueError, match="cycle times differ"):
        _run(reconstruction, calibration)


def test_analyze_runs_tomography_only_when_given_a_calibration(tomography):
    """`analyze(calibration=...)` is how cells 242 and 263 reach all this.

    Without the argument `analyze` is reconstruction only, which is what the
    class did for every caller between the split and the restoration.
    """
    reconstruction, calibration, _, _ = tomography

    expt = MBRPropagatorExperiment.__new__(MBRPropagatorExperiment)
    expt.batch_expts = []
    expt._analysis_station = None
    expt.data = AttrDict()

    # Stand in for the reconstruction step, which needs saved jobs.
    MBRPropagatorExperiment.reconstruct_propagator = staticmethod(
        lambda expts, occupations=None: reconstruction)
    try:
        plain = expt.analyze()
        assert "eigenphase" not in plain

        with_tomography = expt.analyze(calibration=calibration,
                                       floquet_cycle_us=FLOQUET_CYCLE_US)
        assert "eigenphase" in with_tomography
        assert with_tomography.floquet_cycle_us == FLOQUET_CYCLE_US
        assert with_tomography.calibration is not None
    finally:
        del MBRPropagatorExperiment.reconstruct_propagator

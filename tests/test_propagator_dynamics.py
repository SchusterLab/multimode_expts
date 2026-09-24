# -*- coding: utf-8 -*-
"""Hamiltonian tomography from propagator matrices, through MBRHamTomoExperiment.

Why this file exists
--------------------
`analyze_propagator_dynamics` was the one method of 114 that the god-file
split lost outright, and it never had a test, which is part of why its loss
was invisible. It was restored verbatim; the redesign
(docs/qsim/mbr_redesign.md, step 5) moved it to
`fitting/qsim/mbr_propagator.py`, and `MBRHamTomoExperiment.analyze` calls it
when it has a calibration set. These tests are characterization rather than
specification: they pin what the method does, not that it is right.

The fixture is a *synthetic* propagator built from a known Hamiltonian, so
the two estimators have a right answer to be checked against:

    M_q = E_endpoint . exp(-2.pi.i.H.q.T) . E_endpoint

with `E_endpoint` diagonal, which is the structure the method assumes
(`<i|U(q)|j> = D_i U_q E_j`). That makes the eigenphase route exact and the
finite-difference route accurate to its own truncation error. No hardware, no
saved data: the matrices are written into synthetic `MBROrthoColumnExperiment`
jobs as excited-state probabilities, and the q = 0 self-returns into
synthetic `MBRStarkCalExperiment` jobs, so the whole path from job data to
tomography runs.

Run:  pixi run python -m pytest tests/test_propagator_dynamics.py -v
"""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_ham_tomo import MBRHamTomoExperiment
from experiments.qsim.mbr_ortho_column import MBROrthoColumnExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_stark_cal import RAMSEY_PHASES, MBRStarkCalExperiment

FLOQUET_CYCLE_US = 0.5
STEP = 2                        # even, so [0, 2, 4] is a valid FD triple
CYCLES = [0, STEP, 2 * STEP]
SWAP_STORS = [1]                # modes M1, S1


def _fixed_n_basis(photon_number, mode_count):
    """Every occupation of `photon_number` photons over `mode_count` modes."""
    states = [state for state in
              product(range(photon_number + 1), repeat=mode_count)
              if sum(state) == photon_number]
    states.sort(reverse=True)
    return states


def _synthetic_job(job_class, expt, sweep, returns, cycle_us, name):
    """A job whose data give ``complex_return == returns`` in its ``analyze``.

    With Ig = 0 and Ie = 1 the signal is Pe, and Pe at [prep, analyzer] =
    [0, 0], [180, 0], [0, 90], [180, 90] is chosen so that Q_0 = Re A and
    Q_90 = -Im A. The stand-in ``prog`` carries the Floquet timing, like the
    one ``experiments.saved_jobs`` attaches to saved data.
    """
    returns = np.asarray(returns, dtype=complex)
    job = job_class.__new__(job_class)
    job.cfg = AttrDict(dict(
        expt=dict(expt, qubits=[0]),
        device=dict(readout=dict(Ig=[0.], Ie=[1.]), manipulate=dict(kerr=[-0.001]))))
    job.data = dict(xpts=np.asarray(RAMSEY_PHASES), ypts=np.asarray(sweep),
                    avgi=np.stack([(1 + returns.real) / 2, (1 - returns.real) / 2,
                                   (1 - returns.imag) / 2, (1 + returns.imag) / 2], axis=1))
    job.prog = SimpleNamespace(calculate_floquet_cycle_us=lambda: cycle_us,
                               m1s_pi_fracs=[40] * 7, source="synthetic")
    job.fname = f"{name}.h5"
    return job


def _tomography(occupations, matrices, cycles, self_returns, cycle_us=FLOQUET_CYCLE_US,
                calibration_cycle_us=None):
    """-> MBRHamTomoExperiment over synthetic jobs, with its calibration set.

    ``self_returns`` maps occupation -> q = 0 self-return; None gives no
    calibration set.
    """
    parts = []
    for cycle, matrix in zip(cycles, matrices):
        columns = [_synthetic_job(
            MBROrthoColumnExperiment,
            MBROrthoColumnExperiment.job_config(initial, occupations, SWAP_STORS, cycle=cycle),
            occupations, matrix[:, i], cycle_us, f"column_{i}_q{cycle}")
            for i, initial in enumerate(occupations)]
        parts.append(MBROrthogonalityExperiment.from_children(columns))
    calibration = None
    if self_returns is not None:
        calibration = MBRCalibrationSetExperiment.from_children([_synthetic_job(
            MBRStarkCalExperiment,
            MBRStarkCalExperiment.job_config(occupation, [0, 1, 2], SWAP_STORS),
            [0, 1, 2], [value] * 3, calibration_cycle_us or cycle_us, f"stark_{i}")
            for i, (occupation, value) in enumerate(self_returns.items())])
    return MBRHamTomoExperiment.from_parts(parts, calibration=calibration)


def _matrices(hamiltonian_MHz, endpoint, cycle_us):
    return np.asarray([
        endpoint[:, None]
        * _expm(-2j * np.pi * hamiltonian_MHz * (cycle * cycle_us))
        * endpoint[None, :]
        for cycle in CYCLES])


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

    matrices = _matrices(hamiltonian_MHz, endpoint, FLOQUET_CYCLE_US)
    # The calibration supplies the q=0 self-return per occupation, which is
    # endpoint^2 for this construction.
    self_returns = {state: endpoint[i] ** 2 for i, state in enumerate(occupations)}
    return occupations, matrices, self_returns, hamiltonian_MHz, endpoint


def _expm(matrix):
    """Matrix exponential by eigendecomposition; avoids a scipy import here."""
    values, vectors = np.linalg.eig(matrix)
    return vectors @ np.diag(np.exp(values)) @ np.linalg.inv(vectors)


def _run(tomography, **kwargs):
    occupations, matrices, self_returns, _, _ = tomography
    return _tomography(occupations, matrices, CYCLES, self_returns).analyze(**kwargs)


def test_eigenphase_recovers_the_known_hamiltonian(tomography):
    """The generalized eigenvalues against M_0 give H's eigenfrequencies.

    This is the estimator's whole claim: taking eigenvalues relative to the
    same batch's q=0 matrix cancels the fixed D and E factors, so the answer
    does not depend on the endpoint inefficiency at all.
    """
    hamiltonian_MHz = tomography[3]

    result = _run(tomography)

    expected = np.sort(np.linalg.eigvalsh(hamiltonian_MHz))
    got = np.sort(result.eigenphase.eigenfrequencies_MHz)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_eigenphase_poles_sit_on_the_unit_circle(tomography):
    """Unitary evolution, so no decay: |lambda| = 1.

    A radius far from 1 is how this estimator reports that the data are not
    describable by a single unitary generator.
    """
    result = _run(tomography)

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
    hamiltonian_MHz = tomography[3]

    result = _run(tomography)

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
    occupations, _, self_returns, hamiltonian_MHz, endpoint = tomography
    exact = np.sort(np.linalg.eigvalsh(hamiltonian_MHz))

    errors = {}
    for cycle_us in (0.5, 0.25, 0.125, 0.0625):
        result = _tomography(occupations, _matrices(hamiltonian_MHz, endpoint, cycle_us),
                             CYCLES, self_returns, cycle_us=cycle_us).analyze()
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
    result = _run(tomography)

    assert result.finite_difference.semigroup_relative_residual < 1e-9


def test_endpoint_normalization_makes_the_zero_cycle_the_identity(tomography):
    """Dividing by sqrt(self-return) outer product should undo D and E at q=0."""
    endpoint = tomography[4]

    result = _run(tomography)

    np.testing.assert_allclose(result.calibration_self_returns, endpoint ** 2)
    assert result.endpoint_normalized_zero_cycle_identity_residual < 1e-9
    assert result.calibration_diagonal_relative_mismatch < 1e-9
    assert result.zero_cycle_condition_number < 1e3


def test_an_explicit_eigenphase_cycle_is_honoured(tomography):
    """And it changes the alias period, which is the reason to choose it."""
    hamiltonian_MHz = tomography[3]

    result = _run(tomography, eigenphase_cycle=2 * STEP)

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
    with pytest.raises(ValueError, match="eigenphase_cycle"):
        _run(tomography, eigenphase_cycle=bad_cycle)


def test_finite_difference_cycles_must_be_an_even_triple(tomography):
    with pytest.raises(ValueError, match="finite_difference_cycles"):
        _run(tomography, finite_difference_cycles=[0, 1, 2])


def test_an_incomplete_basis_is_rejected(tomography):
    """Generalized eigenanalysis needs the whole fixed-N sector.

    Dropping a state leaves a matrix that is still square per cycle but no
    longer represents the sector, and the answer would be silently wrong.
    """
    occupations, matrices, self_returns, _, _ = tomography
    tomo = _tomography(occupations[:2], matrices[:, :2, :2], CYCLES, self_returns)
    with pytest.raises(ValueError, match="complete fixed-N basis"):
        tomo.analyze()


def test_a_missing_zero_cycle_is_rejected(tomography):
    """Everything here is relative to M_0, so q=0 is not optional."""
    occupations, matrices, self_returns, _, _ = tomography
    tomo = _tomography(occupations, matrices[1:], CYCLES[1:], self_returns)
    with pytest.raises(ValueError, match="needs q=0"):
        tomo.analyze()


def test_a_calibration_missing_an_occupation_is_rejected(tomography):
    occupations, matrices, self_returns, _, _ = tomography
    del self_returns[occupations[-1]]
    tomo = _tomography(occupations, matrices, CYCLES, self_returns)
    with pytest.raises(ValueError, match="calibration is missing"):
        tomo.analyze()


def test_mismatched_cycle_times_are_rejected(tomography):
    """The calibration and the propagator must describe the same Floquet cycle."""
    occupations, matrices, self_returns, _, _ = tomography
    tomo = _tomography(occupations, matrices, CYCLES, self_returns,
                       calibration_cycle_us=FLOQUET_CYCLE_US * 1.1)
    with pytest.raises(ValueError, match="cycle times differ"):
        tomo.analyze()


def test_analyze_runs_tomography_only_when_given_a_calibration(tomography):
    """Without a calibration set, ``analyze`` only stacks the matrices."""
    occupations, matrices, self_returns, _, _ = tomography

    plain = _tomography(occupations, matrices, CYCLES, None).analyze()
    assert "eigenphase" not in plain
    np.testing.assert_allclose(plain.matrices, matrices, atol=1e-12)
    assert plain.floquet_cycle_us == FLOQUET_CYCLE_US

    with_tomography = _tomography(occupations, matrices, CYCLES, self_returns).analyze()
    assert "eigenphase" in with_tomography
    assert with_tomography.calibration is not None

'''
Offline validation of the single-qubit state-tomography reconstruction core
(``fitting/state_tomography.py``).

These tests use synthetic counts generated from known density matrices -- no
hardware required. They check that:
  * cardinal states reconstruct with fidelity > 0.99,
  * the reconstruction is always physical (PSD, unit trace),
  * readout confusion-matrix correction recovers the true state,
  * shot noise degrades gracefully,
  * the Cholesky MLE agrees with the fast projection on clean data.
'''

import numpy as np
import pytest

from fitting.state_tomography import (
    I2, PX, PY, PZ,
    rho_from_expectations,
    project_to_physical,
    reconstruct_single_qubit,
    correct_readout_probs,
    as_confusion_matrix,
    state_fidelity,
)

# --- target states ----------------------------------------------------------
G = np.array([[1], [0]], dtype=complex)
E = np.array([[0], [1]], dtype=complex)
PLUS = (G + E) / np.sqrt(2)
MINUS = (G - E) / np.sqrt(2)
PLUS_I = (G + 1j * E) / np.sqrt(2)
MINUS_I = (G - 1j * E) / np.sqrt(2)

CARDINALS = {
    '0': G, '1': E, '+': PLUS, '-': MINUS, '+i': PLUS_I, '-i': MINUS_I,
}


def _rho(psi):
    return psi @ psi.conj().T


def _synth_counts(rho, n_shots=100000, rng=None, confusion=None):
    '''Generate ideal (or noisy) X/Y/Z counts for a density matrix.

    Probability of |g> in each basis is the +axis projector expectation.
    If ``confusion`` is given, apply it to the true probabilities before
    sampling (simulating readout error).
    '''
    axes = {'X': PX, 'Y': PY, 'Z': PZ}
    counts = {}
    for basis, op in axes.items():
        ev = np.real(np.trace(rho @ op))      # <A> in [-1, 1]
        p_g = (1 + ev) / 2
        p = np.array([p_g, 1 - p_g])
        if confusion is not None:
            p = confusion @ p                  # measured = M @ true
        if rng is not None:
            n_g = rng.binomial(n_shots, p[0])
        else:
            n_g = int(round(n_shots * p[0]))
        counts[basis] = (n_g, n_shots - n_g)
    return counts


def _is_physical(rho, tol=1e-6):
    herm = np.allclose(rho, rho.conj().T, atol=tol)
    trace_ok = abs(np.trace(rho) - 1) < 1e-6
    psd = np.min(np.linalg.eigvalsh(rho)) > -tol
    return herm and trace_ok and psd


@pytest.mark.parametrize('label', list(CARDINALS))
def test_cardinal_reconstruction_noiseless(label):
    rho_true = _rho(CARDINALS[label])
    counts = _synth_counts(rho_true)
    rho_est = reconstruct_single_qubit(counts, method='fast')
    assert _is_physical(rho_est)
    assert state_fidelity(rho_est, CARDINALS[label]) > 0.999


@pytest.mark.parametrize('label', list(CARDINALS))
def test_cardinal_reconstruction_shot_noise(label):
    rng = np.random.default_rng(0)
    rho_true = _rho(CARDINALS[label])
    counts = _synth_counts(rho_true, n_shots=50000, rng=rng)
    rho_est = reconstruct_single_qubit(counts, method='fast')
    assert _is_physical(rho_est)
    assert state_fidelity(rho_est, CARDINALS[label]) > 0.99


def test_maximally_mixed_state():
    rho_true = I2 / 2
    counts = _synth_counts(rho_true)
    rho_est = reconstruct_single_qubit(counts, method='fast')
    assert _is_physical(rho_est)
    assert state_fidelity(rho_est, rho_true) > 0.999


def test_projection_fixes_unphysical_linear_inversion():
    # an over-rotated estimate with |Bloch vector| > 1 must be pulled to PSD
    rho_bad = rho_from_expectations(0.9, 0.9, 0.9)  # |r| ~ 1.56
    assert np.min(np.linalg.eigvalsh(rho_bad)) < 0  # confirm it's unphysical
    rho_fixed = project_to_physical(rho_bad)
    assert _is_physical(rho_fixed)


def test_readout_correction_recovers_state():
    # 5% / 8% asymmetric assignment error
    confusion = np.array([[0.95, 0.08], [0.05, 0.92]])
    rho_true = _rho(PLUS_I)
    counts = _synth_counts(rho_true, confusion=confusion)

    rho_uncorr = reconstruct_single_qubit(counts, method='fast')
    rho_corr = reconstruct_single_qubit(counts, confusion=confusion,
                                        method='fast')
    f_uncorr = state_fidelity(rho_uncorr, PLUS_I)
    f_corr = state_fidelity(rho_corr, PLUS_I)
    assert f_corr > f_uncorr
    assert f_corr > 0.999


def test_confusion_inversion_is_a_distribution():
    confusion = np.array([[0.9, 0.1], [0.1, 0.9]])
    measured = np.array([0.7, 0.3])
    corrected = correct_readout_probs(measured, confusion)
    assert abs(corrected.sum() - 1) < 1e-6
    assert np.all(corrected >= -1e-9)


def test_flat4_confusion_matches_2x2():
    # lab native flat format [Pgg, Pge, Peg, Pee] must equal the 2x2 form
    Pgg, Pge, Peg, Pee = 0.95, 0.05, 0.08, 0.92
    flat = [Pgg, Pge, Peg, Pee]
    M = as_confusion_matrix(flat)
    assert np.allclose(M, [[Pgg, Peg], [Pge, Pee]])
    assert np.allclose(M.sum(axis=0), [1, 1])  # columns sum to 1

    rho_true = _rho(PLUS_I)
    counts = _synth_counts(rho_true, confusion=M)
    rho_flat = reconstruct_single_qubit(counts, confusion=flat, method='fast')
    rho_2x2 = reconstruct_single_qubit(counts, confusion=M.tolist(),
                                       method='fast')
    assert np.allclose(rho_flat, rho_2x2)
    assert state_fidelity(rho_flat, PLUS_I) > 0.999


@pytest.mark.parametrize('label', ['0', '+', '+i'])
def test_cholesky_matches_fast(label):
    rho_true = _rho(CARDINALS[label])
    counts = _synth_counts(rho_true)
    rho_fast = reconstruct_single_qubit(counts, method='fast')
    rho_chol = reconstruct_single_qubit(counts, method='cholesky')
    assert _is_physical(rho_chol)
    # both should be close to the true state and to each other
    assert state_fidelity(rho_chol, CARDINALS[label]) > 0.99
    assert np.max(np.abs(rho_fast - rho_chol)) < 0.05

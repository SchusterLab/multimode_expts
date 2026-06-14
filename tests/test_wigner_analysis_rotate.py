"""
Offline coverage for the worker-facing `WignerAnalysis.wigner_analysis_results`
after generalizing it to (a) accept a mixed density-matrix target (not just a
pure ket) and (b) take rotate in {False/None, True/'optimal', <fixed angle>}.

This is the function the Wigner experiment's display() (run by the job worker)
calls, so a parse/logic error here would fail every Wigner job. We synthesize
the parity grid from a known state via the same forward model the
reconstruction inverts (WignerAnalysis.extracted_W_single_analytic).
"""

import numpy as np
import pytest
import qutip
from scipy.linalg import expm

from fitting.wigner import WignerAnalysis

FOCK_DIM = 5


def _grid_alphas(extent=2.2, n=11):
    axis = np.linspace(-extent, extent, n)
    re, im = np.meshgrid(axis, axis)
    return (re + 1j * im).ravel()


def _wa(alphas_c):
    return WignerAnalysis(data={"alpha": alphas_c}, threshold=0.0, config=None,
                          mode_state_num=FOCK_DIM, alphas=alphas_c)


def _parity_from_rho(rho, alphas_c):
    """Forward model: allocated_readout = 2/pi*parity = W(alpha); so parity = pi/2 * W.
    extracted_W_single_analytic expects a numpy density matrix (not a Qobj)."""
    rho_np = rho.full() if hasattr(rho, 'full') else np.asarray(rho)
    wa = _wa(alphas_c)
    W = wa.extracted_W_single_analytic(rho_np, alphas_c, FOCK_DIM)
    return (np.pi / 2) * np.asarray(W, dtype=float)


def test_worker_default_path_ket_no_rotation():
    # display()'s default call: pure-ket target, rotate=None/False
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.ket2dm(psi), a)
    res = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=False)
    assert res['theta_max'] == 0.0
    assert res['fidelity'] > 0.99
    # rotate=None behaves like False
    res_none = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=None)
    assert res_none['theta_max'] == 0.0


def test_mixed_density_matrix_target_accepted():
    # the generalization: an operator (mixed) target must not crash and returns
    # a valid fidelity (the channel model's ideal output is mixed).
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.ket2dm(psi), a)
    mixed = 0.5 * qutip.ket2dm(psi) + 0.5 * qutip.ket2dm(qutip.basis(FOCK_DIM, 1))
    res = _wa(a).wigner_analysis_results(parity, initial_state=mixed, rotate=False)
    assert 0.0 <= res['fidelity'] <= 1.0
    # pure ket still works identically
    res_ket = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=False)
    assert res_ket['fidelity'] > 0.99


def test_rotate_optimal_realigns_phase():
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    N = np.diag(np.arange(FOCK_DIM))
    t = 0.7
    R = expm(1j * t * N)                       # rotate the true state off-phase
    rho_rot = R @ qutip.ket2dm(psi).full() @ R.conj().T
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.Qobj(rho_rot), a)

    f_none = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=False)['fidelity']
    res_opt = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate='optimal')
    assert res_opt['fidelity'] >= f_none - 1e-6
    assert res_opt['fidelity'] > 0.99         # optimal rotation recovers the phase


def test_rotate_fixed_angle_echoes_and_applies():
    # passing the optimal angle back as a FIXED angle must reproduce the optimal
    # result (and echo the angle) -- this is how the F_e ledger applies a known phi_ch.
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    N = np.diag(np.arange(FOCK_DIM))
    R = expm(1j * 0.7 * N)
    rho_rot = R @ qutip.ket2dm(psi).full() @ R.conj().T
    a = _grid_alphas()
    parity = _parity_from_rho(rho_rot, a)

    opt = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate='optimal')
    fix = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=opt['theta_max'])
    assert fix['theta_max'] == pytest.approx(opt['theta_max'])
    assert fix['fidelity'] == pytest.approx(opt['fidelity'], abs=1e-3)
    assert fix['fidelity'] > 0.99

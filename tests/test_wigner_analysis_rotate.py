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


# --- linear (unprojected, unbiased) fidelity ---------------------------------

def _counts_for_parity(parity, n_total=4000):
    """Identity-confusion, no-pulse-correction counts whose deterministic parity
    equals `parity` (pe = (1 - parity)/2), so the bootstrap point matches."""
    pe = np.clip((1.0 - np.asarray(parity)) / 2.0, 0.0, 1.0)
    return {
        "pulse_correction": False, "alpha_scale": 1.0, "threshold": 0.0,
        "confusion_matrix": [1.0, 0.0, 0.0, 1.0],
        "n_total": [int(n_total)] * len(pe),
        "n_excited": [int(round(p * n_total)) for p in pe],
    }


def test_linear_fidelity_present_and_matches_when_noiseless():
    # On a clean (noiseless) reconstruction the unprojected state is already ~physical,
    # so linear_fidelity ~ projected fidelity ~ 1, and rho_linear has the right shape.
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.ket2dm(psi), a)
    res = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=False)
    assert 'linear_fidelity' in res and 'rho_linear' in res
    assert np.asarray(res['rho_linear']).shape == (FOCK_DIM, FOCK_DIM)
    assert res['linear_fidelity'] > 0.99
    assert res['linear_fidelity'] == pytest.approx(res['fidelity'], abs=1e-2)


def test_linear_fidelity_uses_fixed_rotation_consistently():
    # linear_fidelity is gauge-consistent: fitting theta ('optimal') and passing the
    # same theta back as a fixed angle give the same linear fidelity.
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    N = np.diag(np.arange(FOCK_DIM))
    rho_rot = expm(1j * 0.7 * N) @ qutip.ket2dm(psi).full() @ expm(-1j * 0.7 * N)
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.Qobj(rho_rot), a)

    opt = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate='optimal')
    fix = _wa(a).wigner_analysis_results(parity, initial_state=psi, rotate=opt['theta_max'])
    assert fix['linear_fidelity'] == pytest.approx(opt['linear_fidelity'], abs=1e-3)
    assert opt['linear_fidelity'] > 0.99


def test_bootstrap_freezes_theta_across_draws():
    # The bug we fixed: theta must be fit ONCE and frozen, not re-maximized per draw.
    # Proof: bootstrapping with rotate='optimal' must be identical (same seed) to
    # bootstrapping with the frozen angle passed as a fixed number -- which is only
    # true if 'optimal' does NOT re-fit theta inside each draw.
    psi = (qutip.basis(FOCK_DIM, 0) + qutip.basis(FOCK_DIM, 2)).unit()
    N = np.diag(np.arange(FOCK_DIM))
    rho_rot = expm(1j * 0.7 * N) @ qutip.ket2dm(psi).full() @ expm(-1j * 0.7 * N)
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.Qobj(rho_rot), a)
    pc = _counts_for_parity(parity)

    b_opt = _wa(a).bootstrap_reconstruction(pc, initial_state=psi, rotate='optimal',
                                            n_boot=30, seed=1)
    tf = b_opt['theta_frozen']
    b_fix = _wa(a).bootstrap_reconstruction(pc, initial_state=psi, rotate=tf,
                                            n_boot=30, seed=1)
    # theta_frozen equals the point estimate's theta_max, and 'optimal' actually fit
    # the ~0.7 gauge (not a degenerate no-op).
    point_theta = _wa(a).wigner_analysis_results(
        parity, initial_state=psi, rotate='optimal')['theta_max']
    assert tf == pytest.approx(point_theta)
    assert abs(tf) > 0.1
    # Same frozen angle + same seed => same per-draw reconstruction, up to BLAS
    # last-bit noise (~1e-8). A per-draw re-maximization would instead shift theta
    # each draw and move the fidelities by O(1e-3+), so 1e-6 cleanly discriminates
    # "frozen" from "re-fit".
    np.testing.assert_allclose(b_opt['linear_fidelity_samples'],
                               b_fix['linear_fidelity_samples'], rtol=0, atol=1e-6)
    np.testing.assert_allclose(b_opt['fidelity_samples'],
                               b_fix['fidelity_samples'], rtol=0, atol=1e-6)


def test_bootstrap_linear_fidelity_is_less_biased_than_projected():
    # Bias direction (Schwemmer): for a near-pure target the projected fidelity is
    # biased DOWN, the linear one is not. Over binomial-noise draws the linear mean
    # should sit at/above the projected mean.
    psi = qutip.basis(FOCK_DIM, 1)                 # near-pure Fock-1 target
    a = _grid_alphas()
    parity = _parity_from_rho(qutip.ket2dm(psi), a)
    pc = _counts_for_parity(parity, n_total=1500)  # modest shots -> visible noise
    boot = _wa(a).bootstrap_reconstruction(pc, initial_state=psi, rotate=False,
                                           n_boot=200, seed=0)
    assert boot['linear_fidelity_mean'] >= boot['fidelity_mean'] - 1e-9
    assert 'linear_fidelity_ci' in boot and len(boot['linear_fidelity_ci']) == 2

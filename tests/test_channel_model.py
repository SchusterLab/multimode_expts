"""
Offline tests for the analytical lossy-beamsplitter channel model
(experiments/transduction/channel_model.py). Pure qutip -- no hardware.
"""

import numpy as np
import pytest
import qutip as qt

from experiments.transduction.channel_model import (
    eta_to_theta, logical_ket, beamsplitter, parity_projector,
    ideal_channel_output, even_survival, state_fidelity_qobj,
    logical_z, apply_logical_z,
    channel_output, offdiag_from_states, decoder_kraus, apply_decoder,
    choi_matrix, entanglement_fidelity, coherent_information,
    process_operators,
)

DIM = 6


def test_eta_to_theta_endpoints():
    assert eta_to_theta(1.0) == pytest.approx(0.0)
    assert eta_to_theta(0.0) == pytest.approx(np.pi)
    assert eta_to_theta(0.5) == pytest.approx(np.pi / 2)


def test_beamsplitter_unitary():
    U = beamsplitter(0.7, DIM)
    assert (U.dag() * U - qt.qeye([DIM, DIM])).norm() < 1e-9


def test_eta1_is_identity_on_system():
    # no swap: system unchanged, environment irrelevant
    for label in ('0', '1', '+', '-', '+i', '-i'):
        psi = logical_ket(label, DIM)
        rho = ideal_channel_output(psi, eta=1.0, dim=DIM, env_fock=1)
        assert state_fidelity_qobj(rho, psi) > 0.999


def test_eta0_full_swap_gives_env_photon():
    # full swap: the environment photon ends up in the system
    psi = logical_ket('+', DIM)
    rho = ideal_channel_output(psi, eta=0.0, dim=DIM, env_fock=1)
    assert state_fidelity_qobj(rho, qt.basis(DIM, 1)) > 0.999


def test_output_is_physical_density_matrix():
    psi = logical_ket('+i', DIM)
    for eta in (0.9, 0.75, 0.5):
        for ps in (None, 'even', 'odd'):
            rho = ideal_channel_output(psi, eta, DIM, postselect=ps)
            assert abs(rho.tr() - 1) < 1e-9
            assert np.min(np.linalg.eigvalsh(rho.full())) > -1e-9
            assert (rho - rho.dag()).norm() < 1e-9


def test_even_postselect_at_eta1_returns_input():
    # input is even; post-selecting even at eta=1 must return it unchanged
    psi = logical_ket('-', DIM)
    rho = ideal_channel_output(psi, eta=1.0, dim=DIM, postselect='even')
    assert state_fidelity_qobj(rho, psi) > 0.999


def test_high_eta_even_postselect_near_input():
    # the protocol's premise: at high eta, even-post-selected output ~ input
    psi = logical_ket('+', DIM)
    rho = ideal_channel_output(psi, eta=0.9, dim=DIM, postselect='even')
    assert state_fidelity_qobj(rho, psi) > 0.95


def test_even_survival_monotonic_and_bounded():
    psi = logical_ket('+', DIM)
    s1 = even_survival(psi, 0.98, DIM)
    s2 = even_survival(psi, 0.70, DIM)
    assert 0.0 <= s2 <= s1 <= 1.0          # more swap -> more odd -> lower survival
    assert s1 > 0.9                        # almost all even at eta->1


def test_parity_projector():
    P = parity_projector(4, 'even')
    assert np.allclose(np.diag(P.full()), [1, 0, 1, 0])
    P2 = parity_projector(4, 'odd')
    assert np.allclose(np.diag(P2.full()), [0, 1, 0, 1])


def test_apply_logical_z_removes_known_phase():
    # |+_L> rotated by +50 deg, then de-rotated by -50 deg, returns to |+_L>
    plus = logical_ket('+', DIM)
    rotated = apply_logical_z(plus * plus.dag(), np.deg2rad(50))
    assert state_fidelity_qobj(rotated, plus) < 0.99           # rotation moved it
    derotated = apply_logical_z(rotated, -np.deg2rad(50))
    assert state_fidelity_qobj(derotated, plus) > 0.999        # de-rotation restores
    # +50 deg maps |+_L> toward |+i_L>-ish; exact: rotated == |0>+e^{i50}|2>
    expect = (qt.basis(DIM, 0) + np.exp(1j * np.deg2rad(50)) * qt.basis(DIM, 2)).unit()
    assert state_fidelity_qobj(rotated, expect) > 0.999


def test_state_fidelity_qobj_accepts_numpy_and_mixed():
    psi = logical_ket('+', DIM)
    rho = (psi * psi.dag()).full()                 # numpy
    assert state_fidelity_qobj(rho, psi) > 0.999    # pure ket target
    mixed = 0.5 * psi * psi.dag() + 0.5 * (qt.basis(DIM, 1) * qt.basis(DIM, 1).dag())
    f = state_fidelity_qobj(mixed, mixed)           # mixed-mixed Uhlmann
    assert f == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Process-tomography metrics (F_e via Choi, coherent information)
# --------------------------------------------------------------------------

def test_channel_output_matches_ptrace_model():
    # the analytic process operators must equal the BS+ptrace model
    for eta in (1.0, 0.9, 0.75, 0.5):
        N00, N02, N20, N22, Npp, Nipip = channel_output(eta, DIM)
        m00 = ideal_channel_output(qt.basis(DIM, 0), eta, DIM, env_fock=1)
        m22 = ideal_channel_output(qt.basis(DIM, 2), eta, DIM, env_fock=1)
        assert (N00 - m00).norm() < 1e-9
        assert (N22 - m22).norm() < 1e-9
        # N_pp is the channel applied to |+><+|
        mpp = ideal_channel_output((qt.basis(DIM, 0) + qt.basis(DIM, 2)).unit(),
                                   eta, DIM)
        assert (Npp - mpp).norm() < 1e-9


def test_offdiag_from_states_inverts_channel_output():
    # recovering N_02/N_20 from the four states must reproduce the analytic ones
    for eta in (0.95, 0.7, 0.5):
        N00, N02, N20, N22, Npp, Nipip = channel_output(eta, DIM)
        rec02, rec20 = offdiag_from_states(N00, N22, Npp, Nipip)
        assert (rec02 - N02).norm() < 1e-9
        assert (rec20 - N20).norm() < 1e-9


def test_decoder_kraus_completeness_and_sign_flip():
    for eta in (0.9, 0.5):
        K0, K1 = decoder_kraus(eta, DIM)
        comp = K0.dag() * K0 + K1.dag() * K1
        # identity on {0,1,2,3}, zero elsewhere
        diag = np.real(np.diag(comp.full()))
        assert np.allclose(diag[:4], 1.0)
        assert np.allclose(diag[4:], 0.0)
    # sign flips across eta = 2/3
    K0_hi, _ = decoder_kraus(0.9, DIM)
    K0_lo, _ = decoder_kraus(0.5, DIM)
    assert K0_hi.full()[1, 2] == pytest.approx(+1.0)
    assert K0_lo.full()[1, 2] == pytest.approx(-1.0)


def test_entanglement_fidelity_unity_at_eta1():
    N00, N02, N20, N22, _, _ = channel_output(1.0, DIM)
    Fe = entanglement_fidelity(N00, N20, N02, N22, 1.0, DIM, decode=True)
    assert Fe == pytest.approx(1.0, abs=1e-9)


def test_process_Fe_matches_octahedron_average():
    # RIGOR CROSS-CHECK: on the deterministic (no-postselect) decoded channel the
    # Choi-based F_e must equal the octahedron 2-design average (3*Fbar-1)/2.
    for eta in (0.95, 0.8, 0.7, 0.55):
        N00, N02, N20, N22, _, _ = channel_output(eta, DIM)
        Fe_choi = entanglement_fidelity(N00, N20, N02, N22, eta, DIM,
                                        decode=True, physical=True)
        # octahedron: decode the full channel output for each cardinal, fidelity
        # to the decoded input |0>+c|1>
        fids = []
        for L in ('0', '1', '+', '-', '+i', '-i'):
            rho_out = ideal_channel_output(logical_ket(L, DIM), eta, DIM)  # no PS
            dec = apply_decoder(rho_out, eta, DIM)
            # decoded ideal input: |0_L>->|0>, |2>->|1>
            coeffs = {'0': (1, 0), '1': (0, 1), '+': (1, 1), '-': (1, -1),
                      '+i': (1, 1j), '-i': (1, -1j)}[L]
            tgt = (coeffs[0] * qt.basis(DIM, 0) + coeffs[1] * qt.basis(DIM, 1)).unit()
            fids.append(state_fidelity_qobj(dec, tgt))
        Fe_octa = (3 * float(np.mean(fids)) - 1) / 2
        assert Fe_choi == pytest.approx(Fe_octa, abs=1e-6)


def test_entanglement_fidelity_physical_projection_bounds_noisy_choi():
    # add noise to the process operators; physical=True must keep F_e <= 1
    rng_offsets = [0.05, -0.03, 0.04, -0.02]   # deterministic, no RNG
    N00, N02, N20, N22, _, _ = channel_output(0.98, DIM)
    noisy = []
    for op, off in zip((N00, N02, N20, N22), rng_offsets):
        M = op.full().copy()
        M[0, 1] += off
        M[1, 0] += np.conj(off)
        noisy.append(qt.Qobj(M))
    nn00, nn02, nn20, nn22 = noisy
    Fe_phys = entanglement_fidelity(nn00, nn20, nn02, nn22, 0.98, DIM,
                                    decode=True, physical=True)
    assert Fe_phys <= 1.0 + 1e-9


def test_coherent_information_identity_channel():
    # perfect channel at eta=1: I_c of maximally mixed input = ln 2 (nats)
    N00, N02, N20, N22, _, _ = channel_output(1.0, DIM)
    Ic = coherent_information(N00, N20, N02, N22, 1.0, DIM, decode=False)
    assert Ic == pytest.approx(np.log(2), abs=1e-6)


def test_coherent_information_guard_handles_nonpsd():
    # operators built from independent noisy reconstructions can be non-PSD;
    # the projection must keep entropy finite (no NaN/complex)
    N00, N02, N20, N22, _, _ = channel_output(0.8, DIM)
    M = N00.full().copy()
    M[2, 2] -= 0.1            # force a negative eigenvalue
    Ic = coherent_information(qt.Qobj(M), N20, N02, N22, 0.8, DIM, decode=False)
    assert np.isfinite(Ic)


# --------------------------------------------------------------------------
# Rung 2: even-branch decoder + post-selected process operators
# --------------------------------------------------------------------------

def test_process_operators_order_and_inversion():
    # process_operators must return (N_00, N_20, N_02, N_22) and round-trip the
    # off-diagonals against the analytic channel_output
    for eta in (0.95, 0.7):
        N00, N02, N20, N22, Npp, Nipip = channel_output(eta, DIM)
        p00, p20, p02, p22 = process_operators(N00, N22, Npp, Nipip)
        assert (p00 - N00).norm() < 1e-9
        assert (p22 - N22).norm() < 1e-9
        assert (p02 - N02).norm() < 1e-9
        assert (p20 - N20).norm() < 1e-9


def test_even_branch_equals_full_on_even_states():
    # on a genuinely even post-selected state, K0-only ('even') equals K0+K1
    # ('full') because K1 annihilates the even subspace
    for eta in (0.95, 0.8, 0.7):
        rho = {L: ideal_channel_output(logical_ket(L, DIM), eta, DIM,
                                       postselect='even')
               for L in ('0', '1', '+', '+i')}
        N = process_operators(rho['0'], rho['1'], rho['+'], rho['+i'])
        Fe_even = entanglement_fidelity(*N, eta, DIM, decode=True, branch='even')
        Fe_full = entanglement_fidelity(*N, eta, DIM, decode=True, branch='full')
        assert Fe_even == pytest.approx(Fe_full, abs=1e-9)


def _Fe_rung1(eta):
    N = channel_output(eta, DIM)
    return entanglement_fidelity(N[0], N[2], N[1], N[3], eta, DIM,
                                 decode=True, branch='full')


def _Fe_rung2(eta):
    rho = {L: ideal_channel_output(logical_ket(L, DIM), eta, DIM,
                                   postselect='even')
           for L in ('0', '1', '+', '+i')}
    N = process_operators(rho['0'], rho['1'], rho['+'], rho['+i'])
    return entanglement_fidelity(*N, eta, DIM, decode=True, branch='even')


def test_rung2_postselect_helps_at_high_eta():
    # at high transmissivity, post-selecting even discards the few odd events and
    # the surviving even state is clean -> even-branch F_e beats the no-PS rung 1
    for eta in (0.95, 0.90):
        assert _Fe_rung2(eta) > _Fe_rung1(eta)


def test_rung2_postselect_loses_to_full_decoder_at_low_eta():
    # below the crossover (~0.8) the full decoder's odd-branch recovery (K1) wins:
    # post-selection throws away recoverable odd shots AND the even state is
    # corrupted. This crossover is the motivation for the rung-4/5 feed-forward.
    assert _Fe_rung2(0.70) < _Fe_rung1(0.70)

"""
Analytical model of the lossy-beamsplitter channel for the Fock decoder.

The channel: a logical system state (in the manipulate mode, encoded in the
even-Fock manifold {|0>, |2>}) is mixed with a single-photon environment
(|1> in a storage mode) by a two-mode beamsplitter of transmissivity eta.

Conventions (matching the experiment):
  * eta = single-photon transmissivity; theta = 2*arccos(sqrt(eta)) so
    eta = cos^2(theta/2). eta -> 1 = no swap (identity); eta -> 0 = full swap.
  * Environment is exactly Fock |1>.
  * Ideal, lossless beamsplitter (no prep leakage / decoherence). The absolute
    channel phase (iSWAP phase) is NOT modeled -- it is extracted empirically
    (phi_ch) from the |+_L> reconstruction; populations and parity are
    phase-independent, so this model is exact for those.

These are used to (a) overlay the ideal post-channel cavity state on measured
Wigner reconstructions, and (b) provide the theory ceiling for the decoder
entanglement fidelity F_e(eta).
"""

import numpy as np
import qutip as qt

from fitting.state_tomography import project_to_physical


# logical label -> list of (fock_n, coeff) in the cavity
LOGICAL_FOCK = {
    '0':  [(0, 1)],
    '1':  [(2, 1)],
    '+':  [(0, 1), (2, 1)],
    '-':  [(0, 1), (2, -1)],
    '+i': [(0, 1), (2, 1j)],
    '-i': [(0, 1), (2, -1j)],
}


def eta_to_theta(eta):
    """Beamsplitter angle theta = 2*arccos(sqrt(eta)) (eta = cos^2(theta/2))."""
    return 2.0 * np.arccos(np.sqrt(np.clip(float(eta), 0.0, 1.0)))


def logical_ket(label, dim):
    """Cavity Fock ket for a logical cardinal state (|0>,|2> manifold)."""
    psi = 0 * qt.basis(dim, 0)
    for n, c in LOGICAL_FOCK[label]:
        psi += c * qt.basis(dim, n)
    return psi.unit()


def beamsplitter(theta, dim):
    """Two-mode beamsplitter U = exp[theta/2 (a^dag b - a b^dag)] on dim x dim.

    Single-mode transmissivity is cos^2(theta/2): a -> a cos(theta/2) +/- b sin(...).
    """
    a = qt.tensor(qt.destroy(dim), qt.qeye(dim))
    b = qt.tensor(qt.qeye(dim), qt.destroy(dim))
    return (0.5 * theta * (a.dag() * b - a * b.dag())).expm()


def parity_projector(dim, parity='even'):
    """Diagonal projector onto even or odd Fock states (single mode)."""
    want_even = (parity == 'even')
    diag = [1.0 if ((n % 2 == 0) == want_even) else 0.0 for n in range(dim)]
    return qt.Qobj(np.diag(diag))


def ideal_channel_output(psi_sys, eta, dim, env_fock=1, postselect=None):
    """Ideal reduced system state after the beamsplitter channel.

    Parameters
    ----------
    psi_sys : qt.Qobj ket (dim) -- input system (manipulate) state.
    eta : transmissivity.
    dim : Fock cutoff per mode (>= 4 recommended; the swap can populate |3>).
    env_fock : environment Fock number (default 1).
    postselect : None | 'even' | 'odd' -- project the system onto a parity
        subspace (a syndrome post-selection) and renormalize.

    Returns
    -------
    qt.Qobj density matrix (dim x dim) of the system mode.
    """
    psi0 = qt.tensor(psi_sys.unit(), qt.basis(dim, env_fock))
    out = beamsplitter(eta_to_theta(eta), dim) * psi0
    rho_sys = (out * out.dag()).ptrace(0)
    if postselect in ('even', 'odd'):
        P = parity_projector(dim, postselect)
        rho_sys = P * rho_sys * P
        tr = rho_sys.tr()
        if abs(tr) > 0:
            rho_sys = rho_sys / tr
    return rho_sys


def even_survival(psi_sys, eta, dim, env_fock=1):
    """Probability the system is found even (the post-selection survival)."""
    psi0 = qt.tensor(psi_sys.unit(), qt.basis(dim, env_fock))
    out = beamsplitter(eta_to_theta(eta), dim) * psi0
    rho_sys = (out * out.dag()).ptrace(0)
    return float(np.real((parity_projector(dim, 'even') * rho_sys).tr()))


def logical_z(phi, dim, n_excited=2):
    """Logical-Z phase gate: multiplies the |n_excited> Fock component by e^{i phi}
    (the |2> of the {|0>,|2>} code), identity elsewhere."""
    diag = [np.exp(1j * phi) if n == n_excited else 1.0 for n in range(dim)]
    return qt.Qobj(np.diag(diag))


def apply_logical_z(rho, phi, n_excited=2):
    """Return Z(phi) rho Z(phi)^dag -- de-rotate the logical equatorial phase
    (e.g. to remove the channel phase phi_ch from a reconstructed state)."""
    rho = rho if isinstance(rho, qt.Qobj) else qt.Qobj(np.asarray(rho))
    Z = logical_z(phi, rho.shape[0], n_excited)
    return Z * rho * Z.dag()


def state_fidelity_qobj(rho, target):
    """State fidelity F = <psi|rho|psi> (pure target) or Uhlmann F (mixed),
    accepting qt.Qobj or numpy arrays. Returns the (non-sqrt) fidelity in [0,1].
    """
    rho = rho if isinstance(rho, qt.Qobj) else qt.Qobj(np.asarray(rho))
    target = target if isinstance(target, qt.Qobj) else qt.Qobj(np.asarray(target))
    if target.isket:
        return float(np.real(qt.expect(rho, target)))  # <psi|rho|psi>
    return float(qt.fidelity(rho, target) ** 2)  # qutip fidelity is the sqrt form


# ---------------------------------------------------------------------------
# Process-tomography metrics for the {|0>,|2>} -> {|0>,|1>} decoded channel.
#
# The channel is characterized by the four "process operators"
#   N_jk = (channel)(|j_L><k_L|),   j,k in {0,1},  |0_L>=Fock|0>, |1_L>=Fock|2>,
# which are obtained either analytically (`channel_output`) or by reconstructing
# the post-channel cavity state for four input states {|0>,|2>,|+>,|+i>} and
# combining them (`offdiag_from_states`). The optimal high-eta decoder
# (`decoder_kraus`) downshifts both parity branches into the qubit {|0>,|1>}.
#
# Two figures of merit:
#   * entanglement_fidelity -- F_e of the *decoded* channel vs identity, computed
#     from the (PSD-projected) Choi matrix. F_e = <Phi+|(C/tr C)|Phi+>.
#   * coherent_information -- I_c = S(rho_1) - S(rho_2) of the channel; a genuine
#     channel quantity only when the map is trace-preserving (the deterministic
#     rungs). rho_1/rho_2 are projected to the nearest physical state before the
#     von Neumann entropy (they are built from independent noisy reconstructions
#     and can go slightly non-PSD, which would make entropy_vn NaN/complex).
# ---------------------------------------------------------------------------


def _to_qobj(x):
    return x if isinstance(x, qt.Qobj) else qt.Qobj(np.asarray(x))


def _entropy_physical(rho, base=np.e):
    """Von Neumann entropy after PSD-projecting (Smolin water-fill) the operator,
    so independent-reconstruction noise can't produce NaN/complex entropy."""
    arr = rho.full() if isinstance(rho, qt.Qobj) else np.asarray(rho)
    return float(qt.entropy_vn(qt.Qobj(project_to_physical(arr)), base=base))


def channel_output(eta, dim):
    """Analytic process operators for the single-photon-environment beamsplitter
    channel on the {|0>,|2>} code (matches the BS+ptrace model to machine eps).

    Returns the tuple ``(N_00, N_02, N_20, N_22, N_pp, N_ipip)`` where
    ``N_jk = channel(|j_L><k_L|)`` and ``N_pp``/``N_ipip`` are the channel applied
    to ``|+><+|`` / ``|+i><+i|`` (provided for the data-reconstruction inversion).
    """
    f0, f1, f2, f3 = (qt.basis(dim, k) for k in (0, 1, 2, 3))
    N_00 = eta * qt.ket2dm(f0) + (1 - eta) * qt.ket2dm(f1)
    N_02 = (eta * (3 * eta - 2) * f0 * f2.dag()
            + np.sqrt(3) * eta * (1 - eta) * f1 * f3.dag())
    N_20 = N_02.dag()
    N_22 = (eta * (3 * eta - 2) ** 2 * qt.ket2dm(f2)
            + 3 * eta * (1 - eta) ** 2 * qt.ket2dm(f0)
            + 3 * eta ** 2 * (1 - eta) * qt.ket2dm(f3)
            + (1 - eta) * (1 - 3 * eta) ** 2 * qt.ket2dm(f1))
    N_pp = 0.5 * (N_00 + N_22 + N_02 + N_20)
    N_ipip = 0.5 * (N_00 + N_22 - 1j * N_02 + 1j * N_20)
    return N_00, N_02, N_20, N_22, N_pp, N_ipip


def offdiag_from_states(rho_00, rho_22, rho_pp, rho_ipip):
    """Recover the process coherences (N_02, N_20) from the four measured/ideal
    output states for inputs {|0>, |2>, |+>, |+i>}.

    Inverts  N_pp   = 1/2 (N_00 + N_22 + N_02 + N_20)
             N_ipip = 1/2 (N_00 + N_22 - i N_02 + i N_20).
    """
    rho_00, rho_22 = _to_qobj(rho_00), _to_qobj(rho_22)
    rho_pp, rho_ipip = _to_qobj(rho_pp), _to_qobj(rho_ipip)
    A = rho_pp - 0.5 * (rho_00 + rho_22)
    B = rho_ipip - 0.5 * (rho_00 + rho_22)
    N_20 = A - 1j * B
    N_02 = A + 1j * B
    return N_02, N_20


def process_operators(rho_00, rho_22, rho_pp, rho_ipip):
    """Assemble the four process operators in the order the metric functions
    expect, ``(N_00, N_20, N_02, N_22)``, from the four input-state outputs
    {|0>, |2>, |+>, |+i>}. Used identically for measured and ideal states."""
    N_02, N_20 = offdiag_from_states(rho_00, rho_22, rho_pp, rho_ipip)
    return _to_qobj(rho_00), N_20, N_02, _to_qobj(rho_22)


def decoder_kraus(eta, dim):
    """Optimal recovery Kraus operators K0 (even branch), K1 (odd branch).

    Maps the post-channel Fock {0,1,2,3} back to the logical qubit {|0>,|1>}:
      even:  |0>->|0>, |2>->(+/-)|1>   (sign flips at eta = 2/3)
      odd:   |1>->|0>, |3>->|1>
    K0^dag K0 + K1^dag K1 = I on the {0,1,2,3} subspace (trace-preserving there).
    """
    f0, f1, f2, f3 = (qt.basis(dim, k) for k in (0, 1, 2, 3))
    sign = 1.0 if eta >= 2.0 / 3.0 else -1.0
    K0 = f0 * f0.dag() + sign * f1 * f2.dag()
    K1 = f0 * f1.dag() + f1 * f3.dag()
    return K0, K1


def apply_decoder(rho, eta, dim, branch='full'):
    """Return the recovery applied to rho (linear; works on any operator,
    including the non-positive process coherences N_02/N_20).

    branch='full' -> K0 rho K0^dag + K1 rho K1^dag (rung 1, both syndromes).
    branch='even' -> K0 rho K0^dag only (rung 2: even-post-selected decoder; on a
                     genuinely even state this equals 'full' since K1 kills it).
    branch='odd'  -> K1 rho K1^dag only.
    """
    K0, K1 = decoder_kraus(eta, dim)
    rho = _to_qobj(rho)
    if branch == 'even':
        return K0 * rho * K0.dag()
    if branch == 'odd':
        return K1 * rho * K1.dag()
    return K0 * rho * K0.dag() + K1 * rho * K1.dag()


def choi_matrix(N_00, N_20, N_02, N_22, eta, dim, decode=True, branch='full'):
    """4x4 Choi matrix C = sum_jk |j><k| (x) N_jk of the (optionally decoded)
    qubit channel, restricted to the 2-dim output block.

    decode=True  -> apply the recovery (branch selects K0/K1); output Fock {0,1}.
    decode=False -> raw channel; output block is the code {0,2} (diagnostic only;
                    the raw channel leaks into the odd subspace so tr C < 2).
    """
    ops = {(0, 0): N_00, (0, 1): N_02, (1, 0): N_20, (1, 1): N_22}
    if decode:
        ops = {jk: apply_decoder(op, eta, dim, branch=branch) for jk, op in ops.items()}
        out = (0, 1)
    else:
        ops = {jk: _to_qobj(op) for jk, op in ops.items()}
        out = (0, 2)
    C = np.zeros((4, 4), dtype=complex)
    for j in (0, 1):
        for k in (0, 1):
            M = ops[(j, k)].full()
            for a in (0, 1):
                for b in (0, 1):
                    C[2 * j + a, 2 * k + b] = M[out[a], out[b]]
    return C


def entanglement_fidelity(N_00, N_20, N_02, N_22, eta, dim, decode=True,
                          branch='full', physical=True):
    """Entanglement fidelity F_e of the decoded channel vs the identity, via the
    Choi matrix: F_e = <Phi+|(C / tr C)|Phi+>, |Phi+> = (|00>+|11>)/sqrt(2).

    branch selects the recovery (rung 1: 'full' = K0+K1; rung 2: 'even' = K0).
    physical=True PSD-projects the normalized Choi (Smolin water-fill) before
    evaluating, enforcing complete positivity (the rigorous estimate; required
    because raw linear inversion of noisy data can give a non-PSD Choi and
    F_e > 1). physical=False returns the raw linear-inversion value.
    """
    C = choi_matrix(N_00, N_20, N_02, N_22, eta, dim, decode=decode, branch=branch)
    tr = np.real(np.trace(C))
    rho_C = C / tr if abs(tr) > 0 else C
    if physical:
        rho_C = project_to_physical(rho_C)
    # <Phi+|rho_C|Phi+> with |00> at index 0, |11> at index 3
    F = 0.5 * (rho_C[0, 0] + rho_C[0, 3] + rho_C[3, 0] + rho_C[3, 3])
    return float(np.real(F))


def coherent_information(N_00, N_20, N_02, N_22, eta, dim, decode=False,
                         branch='full', base=np.e):
    """Coherent information I_c = S(rho_1) - S(rho_2) for the maximally mixed
    input, with both operators projected to the nearest physical state first.

    decode=True applies the recovery (branch selects K0/K1; output basis Fock
    {0,1}); decode=False uses the raw channel (output basis the code {0,2}). I_c
    is a genuine channel capacity quantity only for trace-preserving maps -- on
    post-selected rungs it is a conditional figure of merit; report it with the
    survival fraction.
    """
    if decode:
        r00 = apply_decoder(N_00, eta, dim, branch=branch)
        r22 = apply_decoder(N_22, eta, dim, branch=branch)
        r02 = apply_decoder(N_02, eta, dim, branch=branch)
        r20 = apply_decoder(N_20, eta, dim, branch=branch)
        b0, b1 = qt.basis(dim, 0), qt.basis(dim, 1)
    else:
        r00, r22 = _to_qobj(N_00), _to_qobj(N_22)
        r02, r20 = _to_qobj(N_02), _to_qobj(N_20)
        b0, b1 = qt.basis(dim, 0), qt.basis(dim, 2)
    rho_1 = 0.5 * (r00 + r22)
    rho_2 = 0.5 * (qt.tensor(b0 * b0.dag(), r00) + qt.tensor(b1 * b1.dag(), r22)
                   + qt.tensor(b0 * b1.dag(), r02) + qt.tensor(b1 * b0.dag(), r20))
    S1 = _entropy_physical(rho_1, base=base)
    S2 = _entropy_physical(rho_2, base=base)
    return S1 - S2

'''
Single-qubit quantum state tomography reconstruction.

Pure-python, hardware-independent reconstruction utilities used by the
single-qubit state-tomography experiment (and, downstream, by the Fock-state
decoder readout). The numerics mirror the SchusterLab QRAM repo's
``TomoAnalysis.py``:

  * measure in three bases (Z, X, Y) via pre-rotations,
  * turn thresholded g/e counts into Pauli expectation values,
  * optional readout confusion-matrix correction,
  * build the (possibly unphysical) linear-inversion density matrix,
  * physicalize it with the Smolin/Gambetta/Knill fast maximum-likelihood
    projection (Phys. Rev. Lett. 108, 070502) -- the default -- or a full
    Cholesky T-matrix + L-BFGS-B chi^2 MLE.

Convention
----------
Readout maps |g> -> outcome 0 and |e> -> outcome 1. For each measurement
basis the qubit expectation along that axis is

    <A> = P(g) - P(e) = (n_g - n_e) / (n_g + n_e),

provided the basis pre-rotation maps the +1 eigenstate of A onto |g>. The
experiment that feeds this module is responsible for using pre-rotations
consistent with that convention (see ``state_tomography_1q.py``).
'''

import numpy as np
import scipy as sp
import scipy.optimize

# Pauli matrices
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI = {'I': I2, 'X': PX, 'Y': PY, 'Z': PZ}


def expectation_from_counts(n_g, n_e):
    '''Pauli expectation value from thresholded counts.

    <A> = (n_g - n_e) / (n_g + n_e), where the basis pre-rotation has mapped
    the +1 eigenstate of A onto |g> (outcome 0).
    '''
    total = n_g + n_e
    if total == 0:
        return 0.0
    return (n_g - n_e) / total


def as_confusion_matrix(confusion):
    '''Normalize a confusion matrix to 2x2 ``M[i, j] = P(measure i | prepared j)``.

    Accepts either:
      * the lab's native flat 4-list ``[Pgg, Pge, Peg, Pee]`` as saved in
        ``device.readout.confusion_matrix_with_active_reset`` /
        ``confusion_matrix_without_reset`` (g = outcome 0 = below threshold), or
      * an already-shaped ``(2, 2)`` matrix.

    Returns a ``(2, 2)`` numpy array with columns summing to 1.
    '''
    M = np.asarray(confusion, dtype=float)
    if M.shape == (4,):
        Pgg, Pge, Peg, Pee = M
        M = np.array([[Pgg, Peg], [Pge, Pee]])   # [measure, prepare]
    elif M.shape != (2, 2):
        raise ValueError(
            f"confusion must be flat [Pgg, Pge, Peg, Pee] or a 2x2 matrix; "
            f"got shape {M.shape}")
    return M


def correct_readout_probs(probs, confusion):
    '''Correct measured probabilities for readout (assignment) error.

    Parameters
    ----------
    probs : array_like, shape (2,)
        Measured ``[P(0), P(1)]``.
    confusion : array_like, shape (2, 2)
        Assignment matrix ``M[i, j] = P(measure i | prepared j)`` with
        columns summing to 1.

    Returns
    -------
    np.ndarray, shape (2,)
        Corrected ``[P(prep 0), P(prep 1)]``, constrained to be a valid
        probability distribution (non-negative, sums to 1).
    '''
    probs = np.asarray(probs, dtype=float)
    M = np.asarray(confusion, dtype=float)

    def cost(x):
        return np.sum((M @ x - probs) ** 2)

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    x0 = np.clip(np.linalg.solve(M, probs)
                 if np.linalg.det(M) != 0 else probs, 0, 1)
    x0 = x0 / x0.sum() if x0.sum() > 0 else np.array([0.5, 0.5])
    res = sp.optimize.minimize(cost, x0, method='SLSQP',
                               bounds=bounds, constraints=cons)
    return res.x


def rho_from_expectations(ex, ey, ez):
    '''Linear-inversion density matrix: rho = (I + ex X + ey Y + ez Z) / 2.'''
    return 0.5 * (I2 + ex * PX + ey * PY + ez * PZ)


def project_to_physical(rho):
    '''Closest physical density matrix (PSD, unit trace) in Frobenius norm.

    Smolin, Gambetta & Knill, Phys. Rev. Lett. 108, 070502 (2012):
    eigendecompose the Hermitian (trace-1) input, then water-fill the
    eigenvalues so the smallest negatives are zeroed and their weight is
    redistributed uniformly over the surviving eigenvalues.
    '''
    rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity
    vals, vecs = np.linalg.eigh(rho)         # ascending
    vals = vals[::-1]
    vecs = vecs[:, ::-1]                     # descending
    d = len(vals)

    lam = vals.copy().astype(float)
    acc = 0.0
    i = d - 1
    while i >= 0:
        if lam[i] + acc / (i + 1) < 0:
            acc += lam[i]
            lam[i] = 0.0
            i -= 1
        else:
            break
    if i >= 0:
        lam[:i + 1] += acc / (i + 1)

    rho_phys = vecs @ np.diag(lam) @ vecs.conj().T
    tr = np.real(np.trace(rho_phys))
    if tr > 0:
        rho_phys = rho_phys / tr
    return rho_phys


def _t_from_rho(rho):
    '''Cholesky factor (lower triangular) of a (regularized) density matrix,
    returned as a flat real parameter vector.'''
    d = rho.shape[0]
    # regularize to guarantee positive-definiteness for the factorization
    rho_reg = rho + 1e-9 * np.eye(d)
    L = np.linalg.cholesky(rho_reg)
    t = []
    for col in range(d):
        t.append(np.real(L[col, col]))
    for row in range(d):
        for col in range(row):
            t.append(np.real(L[row, col]))
            t.append(np.imag(L[row, col]))
    return np.array(t)


def _rho_from_t(t, d):
    '''Inverse of :func:`_t_from_rho`: build rho = T T^dagger / Tr(T T^dagger).'''
    L = np.zeros((d, d), dtype=complex)
    idx = 0
    for col in range(d):
        L[col, col] = t[idx]
        idx += 1
    for row in range(d):
        for col in range(row):
            L[row, col] = t[idx] + 1j * t[idx + 1]
            idx += 2
    rho = L @ L.conj().T
    tr = np.real(np.trace(rho))
    return rho / tr if tr > 0 else rho


def rho_mle_cholesky(basis_states, probs, n_total=1.0):
    '''Full numerical MLE via Cholesky T-matrix + L-BFGS-B chi^2.

    Parameters
    ----------
    basis_states : list of np.ndarray
        Measurement projector kets ``|psi>`` (column vectors), one per
        outcome whose probability is recorded in ``probs``.
    probs : array_like
        Measured probabilities aligned with ``basis_states``.
    n_total : float
        Total shots per setting (sets the chi^2 weighting). Defaults to 1.

    Returns
    -------
    np.ndarray
        Physical density matrix maximizing the (Gaussian) likelihood.
    '''
    probs = np.asarray(probs, dtype=float)
    d = basis_states[0].shape[0]

    # initial guess: physicalized linear inversion of the projectors
    rho0 = project_to_physical(
        sum(p * (psi @ psi.conj().T)
            for p, psi in zip(probs, basis_states)) * (d / max(len(probs), 1)))

    def likelihood(t):
        rho = _rho_from_t(t, d)
        val = 0.0
        for psi, p in zip(basis_states, probs):
            proj = np.real((psi.conj().T @ rho @ psi)[0, 0])
            expected = n_total * proj
            measured = n_total * p
            if expected > 1e-12:
                val += (expected - measured) ** 2 / expected
        return val

    res = sp.optimize.minimize(
        likelihood, _t_from_rho(rho0), method='L-BFGS-B',
        options={'maxiter': 100000, 'maxfun': 100000})
    return _rho_from_t(res.x, d)


def state_fidelity(rho, target):
    '''State fidelity F(rho, target).

    If ``target`` is a ket (1-D or column vector) the pure-state shortcut
    F = <psi|rho|psi> is used; otherwise the mixed-state Uhlmann fidelity
    F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.
    '''
    target = np.asarray(target, dtype=complex)
    if target.ndim == 1 or 1 in target.shape:
        psi = target.reshape(-1, 1)
        psi = psi / np.linalg.norm(psi)
        return float(np.real((psi.conj().T @ rho @ psi)[0, 0]))

    sigma = target
    # sqrt(rho)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0, None)
    sqrt_rho = vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T
    inner = sqrt_rho @ sigma @ sqrt_rho
    ivals = np.linalg.eigvalsh(inner)
    ivals = np.clip(ivals, 0, None)
    return float(np.real(np.sum(np.sqrt(ivals))) ** 2)


def reconstruct_single_qubit(counts, confusion=None, method='fast'):
    '''Reconstruct a single-qubit density matrix from X/Y/Z basis counts.

    Parameters
    ----------
    counts : dict
        ``{'X': (n_g, n_e), 'Y': (n_g, n_e), 'Z': (n_g, n_e)}``.
    confusion : array_like, shape (2, 2), optional
        Readout assignment matrix ``M[i, j] = P(measure i | prepared j)``.
        If given, per-basis probabilities are corrected before forming
        expectation values.
    method : {'fast', 'cholesky', 'linear'}
        ``'fast'`` (default): linear inversion then PRL-108-070502 projection.
        ``'cholesky'``: full numerical MLE. ``'linear'``: no physicalization.

    Returns
    -------
    np.ndarray, shape (2, 2)
        The reconstructed density matrix.
    '''
    if confusion is not None:
        confusion = as_confusion_matrix(confusion)

    exp = {}
    probs_by_basis = {}
    for basis in ('X', 'Y', 'Z'):
        n_g, n_e = counts[basis]
        total = n_g + n_e
        p = np.array([n_g, n_e], dtype=float) / total if total else np.array([0.5, 0.5])
        if confusion is not None:
            p = correct_readout_probs(p, confusion)
        probs_by_basis[basis] = p
        exp[basis] = p[0] - p[1]

    if method == 'cholesky':
        # projector kets: +axis -> |g> outcome, -axis -> |e> outcome
        g = np.array([[1], [0]], dtype=complex)
        e = np.array([[0], [1]], dtype=complex)
        plus = (g + e) / np.sqrt(2)
        minus = (g - e) / np.sqrt(2)
        plus_i = (g + 1j * e) / np.sqrt(2)
        minus_i = (g - 1j * e) / np.sqrt(2)
        basis_states = [plus, minus, plus_i, minus_i, g, e]
        probs = [probs_by_basis['X'][0], probs_by_basis['X'][1],
                 probs_by_basis['Y'][0], probs_by_basis['Y'][1],
                 probs_by_basis['Z'][0], probs_by_basis['Z'][1]]
        return rho_mle_cholesky(basis_states, probs)

    rho = rho_from_expectations(exp['X'], exp['Y'], exp['Z'])
    if method == 'linear':
        return rho
    return project_to_physical(rho)

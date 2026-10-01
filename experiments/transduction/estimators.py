"""
Maximum-likelihood estimate of the transduction channel (for Ic).

Why: linear inversion of each output state alone, then projection, reads Ic too
low (shot noise adds false eigenvalues to the joint state rho_RB). The study
(``C:\\experiments\\260601_Transduction_sandbox\\ic_debug\\estimator\\REPORT.md``,
2026-09-30) found, on the real grid and counts at eta 0.35:

    estimator                                  Ic bias, ideal ch. / channel like ours
    linear inversion + projection (old)        -0.12 / -0.03..-0.06
    joint ML, CPTP (``fit_channel``)           -0.10 / -0.00..-0.04
    joint ML, CPTP + photon-number covariant   -0.05 / -0.00..-0.01   <- default for Ic

Fe: keep the linear estimate (unbiased); the ML fits read Fe 0.01-0.03 low.

Measurement model (one Wigner job): for each displacement alpha_i the parity
operator E_i = [D(-a) P D(-a)^dag] on the DIM block, W_i = tr(rho E_i). The fit
uses the difference parity y = (pm - pp)/2/alpha_scale (confusion corrected),
exactly the lab parity, with a Gaussian likelihood (sigma from the counts). The
separate plus/minus counts have a common offset (pp + pm)/2 of about -0.15 that
cancels in y; a binomial fit of the separate counts would need it.

Channel: Choi J (2 DIM x 2 DIM) = sum_jk |j><k| (x) N_jk, input index major,
|0_L> = Fock 0, |1_L> = Fock 2. TP <=> tr_out J = I_2. rho_RB = J/2,
rho_B = tr_R J / 2, Ic = S(rho_B) - S(rho_RB).
"""

import numpy as np
import qutip as qt
from scipy.optimize import minimize
from scipy.stats import chi2 as _chi2

from experiments.transduction.sequences import PROC_INPUTS

_NT = 30                          # Fock cutoff for the displaced parity operators
RHO_IN = {'0': np.array([[1, 0], [0, 0]], complex),
          '1': np.array([[0, 0], [0, 1]], complex),
          '+': 0.5 * np.array([[1, 1], [1, 1]], complex),
          '+i': 0.5 * np.array([[1, -1j], [1j, 1]], complex)}
COV_P_MIN = 0.02                  # photon-number check: p-value below this -> use the non-covariant fit


# ---------------------------------------------------------------------------
# Choi helpers
# ---------------------------------------------------------------------------

def outputs_to_choi(rhos):
    """Output states {label: array} of the four inputs -> Choi matrix (Hermitian part)."""
    r = {L: np.asarray(v.full() if hasattr(v, 'full') else v) for L, v in rhos.items()}
    d = r['0'].shape[0]
    A = r['+'] - 0.5 * (r['0'] + r['1'])
    B = r['+i'] - 0.5 * (r['0'] + r['1'])
    blocks = {(0, 0): r['0'], (0, 1): A + 1j * B, (1, 0): A - 1j * B, (1, 1): r['1']}
    J = np.zeros((2 * d, 2 * d), complex)
    for (j, k), M in blocks.items():
        J[j * d:(j + 1) * d, k * d:(k + 1) * d] = M
    return 0.5 * (J + J.conj().T)


def choi_to_outputs(J):
    d = J.shape[0] // 2
    J4 = J.reshape(2, d, 2, d)
    return {L: np.einsum('jk,jakb->ab', RHO_IN[L], J4) for L in PROC_INPUTS}


def _entropy(rho):
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log(w)))


def ic_from_choi(J):
    """Ic = S(rho_B) - S(rho_RB) of a PSD, TP Choi matrix (nats)."""
    d = J.shape[0] // 2
    rho_B = np.einsum('jajb->ab', J.reshape(2, d, 2, d)) / 2
    return _entropy(rho_B) - _entropy(J / 2)


def charge_mask(dim):
    """Choi elements allowed by photon-number covariance: (n_out - n_in) equal on both sides."""
    n_in = (0, 2)
    c = np.array([n - n_in[j] for j in range(2) for n in range(dim)])
    return c[:, None] == c[None, :]


# ---------------------------------------------------------------------------
# Measurement model of one Wigner job
# ---------------------------------------------------------------------------

def _parity_ops(alpha, dim):
    P = np.diag((-1.0) ** np.arange(_NT)).astype(complex)
    out = []
    for a in alpha:
        D = qt.displace(_NT, -a).full()
        out.append((D @ P @ D.conj().T)[:dim, :dim])
    return np.array(out)


class WignerData:
    """Counts of one Wigner job and its parity model. ``data`` = expt.data (needs
    'alpha' and 'parity_counts')."""

    def __init__(self, data, dim):
        pc = data['parity_counts']
        self.dim = dim
        self.alpha = np.asarray(data['alpha'])
        self.cm = np.asarray(pc['confusion_matrix'], float)       # [Pgg, Pge, Peg, Pee]
        self.ascale = float(pc.get('alpha_scale', 1.0))
        self.pulse_correction = bool(pc.get('pulse_correction'))
        if self.pulse_correction:
            self.nt = np.stack([np.asarray(pc[s]['n_total'], float) for s in ('plus', 'minus')])
            self.ne = np.stack([np.asarray(pc[s]['n_excited'], float) for s in ('plus', 'minus')])
        else:
            self.nt = np.asarray(pc['n_total'], float)[None]
            self.ne = np.asarray(pc['n_excited'], float)[None]
        E = _parity_ops(self.alpha, dim)
        self.Ef = E.reshape(len(self.alpha), -1)               # W = real(ET @ rho.ravel())
        self.ET = E.transpose(0, 2, 1).reshape(len(self.alpha), -1)
        self.offset = self._offset(self.ne)

    # parity and its sigma, as fitting.wigner.parity_from_counts
    def _pe(self, ne):
        c = self.cm
        Pinv = np.linalg.inv(np.array([[c[0], c[2]], [c[1], c[3]]]))
        f = ne / np.where(self.nt > 0, self.nt, 1)
        return Pinv[1, 0] * (1 - f) + Pinv[1, 1] * f

    def parity(self, ne=None):
        p = 1 - 2 * self._pe(self.ne if ne is None else ne)
        if self.pulse_correction:
            return (p[1] - p[0]) / 2 / self.ascale
        return p[0] / self.ascale

    def sigma(self, ne=None):
        ne = self.ne if ne is None else ne
        f = np.clip(ne / self.nt, 1.0 / self.nt, 1 - 1.0 / self.nt)
        v = f * (1 - f) / self.nt / (self.cm[3] - self.cm[1]) ** 2
        if self.pulse_correction:
            return np.sqrt(v[0] + v[1]) / self.ascale
        return 2 * np.sqrt(v[0]) / self.ascale

    def _offset(self, ne):
        if not self.pulse_correction:
            return np.zeros(len(self.alpha))
        p = 1 - 2 * self._pe(ne)
        return (p[0] + p[1]) / 2

    def W(self, rho):
        return np.real(self.ET @ rho.ravel())

    def nll_grad(self, rho, y, s):
        r = (y - self.W(rho)) / s
        return 0.5 * np.sum(r ** 2), (-(r / s) @ self.Ef).reshape(self.dim, self.dim)

    def chi2(self, rho, ne=None):
        return float(np.sum(((self.parity(ne) - self.W(rho)) / self.sigma(ne)) ** 2))

    def simulate(self, rho, rng):
        """Excited counts for state ``rho`` with this job's shots, confusion, alpha_scale and offset."""
        W = self.W(rho)
        # plus: parity -a W + o, minus: +a W + o  ->  (pm - pp)/2/a = W;  single: +a W
        sgn = np.array([1.0, -1.0]) if self.pulse_correction else np.array([-1.0])
        pe = (1 + sgn[:, None] * self.ascale * W[None] - self.offset[None]) / 2
        p = np.clip(self.cm[1] + (self.cm[3] - self.cm[1]) * pe, 0, 1)
        return rng.binomial(self.nt.astype(int), p).astype(float)

    def linear(self, ne=None):
        """Trace-1 least-squares state (the lab linear inversion) from the counts."""
        if not hasattr(self, '_lin'):
            d = self.dim
            B = []
            for i in range(d):
                M = np.zeros((d, d), complex); M[i, i] = 1; B.append(M)
            for i in range(d):
                for j in range(i + 1, d):
                    M = np.zeros((d, d), complex); M[i, j] = M[j, i] = 1; B.append(M)
                    M = np.zeros((d, d), complex); M[i, j] = -1j; M[j, i] = 1j; B.append(M)
            B = np.array(B)
            A = np.real(self.ET @ B.reshape(len(B), -1).T)          # A[k, b] = tr(E_k B_b)
            t = np.real(np.trace(B, axis1=1, axis2=2))
            K = np.block([[A.T @ A, t[:, None]], [t[None, :], np.zeros((1, 1))]])
            self._lin = (np.linalg.inv(K), A, B)
        Kinv, A, B = self._lin
        x = (Kinv @ np.concatenate([A.T @ self.parity(ne), [1.0]]))[:-1]
        return np.tensordot(x, B, axes=1)


# ---------------------------------------------------------------------------
# Joint ML fit of the channel
# ---------------------------------------------------------------------------

def _sqrtm_psd(M):
    w, v = np.linalg.eigh(0.5 * (M + M.conj().T))
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def fit_channel(wd, J0, covariant=True, ne=None):
    """Joint ML Choi matrix over CPTP maps (optionally photon-number covariant).

    ``wd``: {label: WignerData}; ``J0``: start point (e.g. the linear Choi);
    ``ne``: {label: excited counts} to fit instead of the measured ones (bootstrap).
    Parametrization J = (S^-1/2 (x) I) A A^dag (S^-1/2 (x) I), S = tr_out(A A^dag),
    which is CPTP for any A."""
    dim = J0.shape[0] // 2
    D = 2 * dim
    mask = charge_mask(dim) if covariant else np.ones((D, D), bool)
    ne = ne or {L: None for L in PROC_INPUTS}
    ys = {L: (wd[L].parity(ne[L]), wd[L].sigma(ne[L])) for L in PROC_INPUTS}
    norm = sum(wd[L].nt.sum() for L in PROC_INPUTS)
    w, v = np.linalg.eigh(0.5 * (J0 + J0.conj().T))
    J0 = (v * np.clip(w, 0, None)) @ v.conj().T
    J0 = 0.9 * J0 / (np.real(np.trace(J0)) / 2) + 0.1 * np.eye(D) / dim
    if covariant:
        J0 = np.where(mask, J0, 0)
    idx = np.where(mask.ravel())[0]
    a0 = (_sqrtm_psd(J0) * mask).ravel()[idx]

    def build(a):
        A = np.zeros(D * D, complex)
        A[idx] = a
        A = A.reshape(D, D)
        M = A @ A.conj().T
        S = np.einsum('jaka->jk', M.reshape(2, dim, 2, dim))
        s, U = np.linalg.eigh(0.5 * (S + S.conj().T))
        T = np.kron((U * s ** -0.5) @ U.conj().T, np.eye(dim))
        return A, M, s, U, T, T @ M @ T

    def fg(x):
        A, M, s, U, T, J = build(x[:len(idx)] + 1j * x[len(idx):])
        J4 = J.reshape(2, dim, 2, dim)
        f, Gam = 0.0, np.zeros((D, D), complex)
        for L in PROC_INPUTS:
            fL, G = wd[L].nll_grad(np.einsum('jk,jakb->ab', RHO_IN[L], J4), *ys[L])
            f += fL
            Gam += np.kron(RHO_IN[L].T, G)
        # gradient through the TP normalization T = S^-1/2 (x) I
        Y = np.einsum('jaka->jk', (M @ T @ Gam + Gam @ T @ M).reshape(2, dim, 2, dim))
        fs = s ** -0.5
        Lm = np.array([[(-0.5 * s[i] ** -1.5) if abs(s[i] - s[j]) < 1e-12 * max(s) else (fs[i] - fs[j]) / (s[i] - s[j])
                        for j in range(2)] for i in range(2)])
        YS = U @ ((U.conj().T @ Y @ U) * Lm) @ U.conj().T
        gA = (2 * (T @ Gam @ T + np.kron(YS, np.eye(dim))) @ A).ravel()[idx]
        return f / norm, np.concatenate([gA.real, gA.imag]) / norm

    r = minimize(fg, np.concatenate([a0.real, a0.imag]), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
    J = build(r.x[:len(idx)] + 1j * r.x[len(idx):])[-1]
    return 0.5 * (J + J.conj().T)


def chi2_of(wd, J, ne=None):
    out = choi_to_outputs(J)
    ne = ne or {L: None for L in PROC_INPUTS}
    return sum(wd[L].chi2(out[L], ne[L]) for L in PROC_INPUTS)


def estimate_ic(expts, rhos_linear, dim, n_boot=100, n_thru=0, eta=None, seed=0):
    """Ic from the joint ML fit of one set.

    Fits the covariant and the non-covariant CPTP channel. The photon-number check
    compares their chi2 (the covariant fit has 50 fewer parameters for DIM 4); if it
    fails (p < COV_P_MIN) the non-covariant fit is used. ``n_boot`` parametric
    bootstrap refits (data simulated from the chosen fit) give the spread and a bias
    estimate. ``n_thru`` > 0 also runs the ideal channel at ``eta`` through the same
    estimator on this set's grid and shots ("theory through the estimator")."""
    wd = {L: WignerData(expts[L].data, dim) for L in PROC_INPUTS}
    J_lin = outputs_to_choi(rhos_linear)
    J_cov = fit_channel(wd, J_lin, covariant=True)
    J_full = fit_channel(wd, J_lin, covariant=False)
    c_cov, c_full = chi2_of(wd, J_cov), chi2_of(wd, J_full)
    # parameters removed by the covariance: masked Choi elements minus the 2 TP
    # constraints (off-diagonal of tr_out J) that the covariant map meets anyway; 50 for DIM 4
    dof = int(charge_mask(dim).size - charge_mask(dim).sum()) - 2
    p_cov = float(_chi2.sf(max(c_cov - c_full, 0.0), dof))
    covariant = p_cov >= COV_P_MIN
    J = J_cov if covariant else J_full
    out = dict(Ic=ic_from_choi(J), estimator='ml_covariant' if covariant else 'ml_cptp',
               Ic_ml_covariant=ic_from_choi(J_cov), Ic_ml_cptp=ic_from_choi(J_full),
               chi2=c_cov if covariant else c_full, n_points=int(sum(len(wd[L].alpha) for L in PROC_INPUTS)),
               cov_check_p=p_cov, choi=J)
    rng = np.random.default_rng(seed)

    def resample(rhos):
        return {L: wd[L].simulate(rhos[L], rng) for L in PROC_INPUTS}

    def refit(ne):
        start = outputs_to_choi({L: wd[L].linear(ne[L]) for L in PROC_INPUTS})
        return ic_from_choi(fit_channel(wd, start, covariant=covariant, ne=ne))

    if n_boot:
        fit_out = choi_to_outputs(J)
        xs = np.array([refit(resample(fit_out)) for _ in range(int(n_boot))])
        out['Ic_std'] = float(np.std(xs, ddof=1))
        out['Ic_ci'] = [float(np.percentile(out['Ic'] - (xs - out['Ic']), 16)),
                        float(np.percentile(out['Ic'] - (xs - out['Ic']), 84))]
        out['Ic_bias_est'] = float(xs.mean() - out['Ic'])
        out['Ic_bias_corrected'] = float(2 * out['Ic'] - xs.mean())
    if n_thru:
        from experiments.transduction.channel_model import ideal_channel_output, logical_ket
        ideal = {L: ideal_channel_output(logical_ket(L, dim), eta, dim).full() for L in PROC_INPUTS}
        xs = np.array([refit(resample(ideal)) for _ in range(int(n_thru))])
        out['theory_thru'] = dict(mean=float(xs.mean()), lo=float(np.percentile(xs, 16)),
                                  hi=float(np.percentile(xs, 84)), n=int(n_thru))
    return out

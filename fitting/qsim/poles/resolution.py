"""What the data could resolve, for any method: Cramer-Rao bounds of the model's levels (spec 5.5).

The Fisher matrix of all frequencies, decays and row offsets, the amplitudes eliminated by
variable projection, at the model's levels and amplitudes and each row's own noise. Levels whose
gap is smaller than a few of its Cramer-Rao errors form a cluster: no unbiased method tells them
apart. Pure numerics.
"""
import numpy as np

#: A Fisher matrix above this condition number is beyond double precision.
MAX_CONDITION = 1e12


def fisher_matrix(time_us, levels_MHz, amplitudes, decay_per_us, noise, offset_prior_MHz):
    """-> F over (E_1..M, gamma_1..M, delta_1..R), amplitudes eliminated (variable projection):

        F = sum_b 2 Re(G_b^h G_b) / s_b^2,   G_b = P_perp [c_b (-2 pi i t) z, c_b (-t) z, -2 pi i t m_b],

    P_perp the projector orthogonal to the pole columns z_lambda(t), m_b = sum c_b z; plus
    1 / sigma_cal^2 on each offset."""
    M, R = len(levels_MHz), len(amplitudes)
    V = np.exp(np.outer(time_us, -decay_per_us - 2j * np.pi * np.asarray(levels_MHz)))
    Q = np.linalg.qr(V)[0]
    t = np.asarray(time_us)[:, None]
    F = np.zeros((2 * M + R, 2 * M + R))
    for b, c in enumerate(amplitudes):
        columns = np.hstack([-2j * np.pi * t * V * c, -t * V * c, -2j * np.pi * t * (V @ c)[:, None]])
        G = columns - Q @ (Q.conj().T @ columns)
        index = np.r_[np.arange(2 * M), 2 * M + b]
        F[np.ix_(index, index)] += 2 * np.real(G.conj().T @ G) / noise[b] ** 2
    F[2 * M:, 2 * M:] += np.eye(R) / offset_prior_MHz ** 2
    return F


def local_covariance(time_us, levels_MHz, amplitudes, decay_per_us, noise, offset_prior_MHz, subset):
    """-> F^-1 of the levels in ``subset`` alone (all rows and offsets), or None when F is
    beyond double precision (condition above ``MAX_CONDITION``): the levels cannot be told apart."""
    F = fisher_matrix(time_us, levels_MHz[subset], amplitudes[:, subset], decay_per_us, noise, offset_prior_MHz)
    return None if np.linalg.cond(F) > MAX_CONDITION else np.linalg.inv(F)


def cramer_rao_bounds(time_us, levels_MHz, amplitudes, decay_per_us, noise, offset_prior_MHz, window_MHz):
    """-> (sigma_E per level, sigma of each adjacent gap E_{k+1} - E_k), in MHz, levels sorted.

    Each from the Fisher matrix of the levels within ``window_MHz`` of the level (or of either
    end of the gap) only: far levels are nearly orthogonal on the grid, and the full matrix of
    a dense spectrum is beyond double precision. inf where the local matrix is too."""
    levels_MHz, crb, gap_crb = np.asarray(levels_MHz), np.full(len(levels_MHz), np.inf), np.full(len(levels_MHz) - 1, np.inf)
    near = np.abs(np.subtract.outer(levels_MHz, levels_MHz)) <= window_MHz
    for k in range(len(levels_MHz)):
        subset = np.flatnonzero(near[k])
        covariance = local_covariance(time_us, levels_MHz, amplitudes, decay_per_us, noise, offset_prior_MHz, subset)
        if covariance is not None:
            i = np.searchsorted(subset, k)
            crb[k] = np.sqrt(covariance[i, i])
        if k + 1 < len(levels_MHz):
            subset = np.flatnonzero(near[k] | near[k + 1])
            covariance = local_covariance(time_us, levels_MHz, amplitudes, decay_per_us, noise, offset_prior_MHz, subset)
            if covariance is not None:
                i, j = np.searchsorted(subset, [k, k + 1])
                gap_crb[k] = np.sqrt(abs(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j]))
    return crb, gap_crb


def clusters(gap_snr, min_gap_snr):
    """-> index arrays of sorted levels joined by adjacent gaps with gap / sigma_gap < min_gap_snr."""
    breaks = np.flatnonzero(np.asarray(gap_snr) >= min_gap_snr) + 1
    return np.split(np.arange(len(gap_snr) + 1), breaks)

"""Multiparity readout shots: decode n mod 4, flag the shots where the qubit reads f, post-select.

A qubit in f at readout time reads e in both parity readouts: the readout thresholds I only, and
f sits near e in I. So such a shot decodes as n = 3, whatever the mode holds. f sits apart from e
in Q, so these shots can be flagged with the g/e/f cloud centers of a reference measured at the
same readout phase (for example ``FStateHoldExperiment``'s ``blob_centers_i/_q``). Found on
2026-11-13 (vault ``Lab/guan/AGP/2026/11/2026-11-13``): after a Fock 3 prep in M1, about 5 % of
the shots at 0.45 mA.

Data in, numbers out; no config objects. Nothing here raises on bad data except for wrong shapes.
"""
import numpy as np

from fitting.qsim.f_hold import assign_nearest


def lanes(idata, qdata, read_num):
    """-> complex shots, shape (points, shots, read_num), from interleaved (points, shots * read_num)."""
    shots = np.asarray(idata) + 1j * np.asarray(qdata)
    return shots.reshape(shots.shape[0], -1, int(read_num))


def decode_n_mod4(first, second, threshold):
    """-> n mod 4 = b1 + 2 b2 with b = (I > threshold), as ``classify_two_parity_readouts``
    (``cond_sec_phase = -90``, e high in I)."""
    return (np.real(first) > threshold).astype(int) + 2 * (np.real(second) > threshold).astype(int)


def reads_f(shots, centers):
    """-> True where the nearest of the g/e/f centers is f."""
    return assign_nearest(shots, centers) == 2


def n_distribution(n, keep=None):
    """-> p(n mod 4) along the last axis, shape (..., 4), of the kept shots (all if keep is None).

    NaN where no shot is kept.
    """
    n = np.asarray(n)
    keep = np.ones(n.shape, bool) if keep is None else np.asarray(keep, bool)
    counts = np.stack([((n == k) & keep).sum(axis=-1) for k in range(4)], axis=-1).astype(float)
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts / total


def corrected(p, confusion_matrix):
    """-> p_true with p = p_true @ C, for C[prepared, measured]; p of shape (..., 4)."""
    p = np.asarray(p, float)
    C = np.asarray(confusion_matrix, float)
    return np.linalg.solve(C.T, p.reshape(-1, 4).T).T.reshape(p.shape)


def mean_n(p):
    """-> sum_n n p(n) along the last axis."""
    return np.asarray(p, float) @ np.arange(4)


def confusion_from_shots(n_rows, keep_rows=None):
    """-> C[prepared, measured] (4, 4) from the n mod 4 shots of prepared Fock 0..3 (one row each).

    keep_rows: the same cut as applied to the data it will correct (None: all shots).
    """
    keep_rows = [None] * 4 if keep_rows is None else keep_rows
    return np.array([n_distribution(n, keep) for n, keep in zip(n_rows, keep_rows)])


def mean_n_with_error(n, keep=None):
    """-> (mean, standard error) of the raw n mod 4 labels per point (along the first axis)."""
    n = np.asarray(n, float)
    keep = np.ones(n.shape, bool) if keep is None else np.asarray(keep, bool)
    means, errors = [], []
    for row, k in zip(n, keep):
        kept = row[k]
        means.append(kept.mean() if kept.size else np.nan)
        errors.append(kept.std() / np.sqrt(kept.size) if kept.size > 1 else np.nan)
    return np.array(means), np.array(errors)


def corrected_mean_n(n, confusion_matrix, keep=None):
    """-> corrected <n> per point (along the first axis) of n mod 4 shots (points, shots)."""
    return mean_n(corrected(n_distribution(n, keep), confusion_matrix))

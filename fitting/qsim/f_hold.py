"""Estimators for the |f,n> hold scan (``experiments/qsim/f_state_hold.py``).

Data in, numbers out: g/e/f populations from single shots, the decay of P(f) during
the hold, and the transition frequency nu_n = E(f,n) - E(g,n-1) from calibrated
frequencies. No hardware and no config objects here.

The method is the one of 2026-10-28/29 (``measurement_notebooks/guan/yoko450ua_calibration.py``,
cells 5 and 12; vault report ``Lab/guan/AGP/2026/10/2026-10-29_TLS_12GHz_report``):
each shot goes to the nearest reference-cloud center, and the populations are
corrected with the 3x3 assignment matrix of the same reference clouds.

Every function here returns NaN for what it cannot estimate and does not raise on
bad data: ``Experiment.go()`` analyzes before it saves, so an error in the analysis
would lose the shots.
"""
import numpy as np
from scipy.optimize import curve_fit

STATES = "gef"


def blob_centers(ref_shots):
    """-> (3,) complex centers: median I and median Q of each reference cloud.

    ref_shots: the complex shots of prepared g, e, f (in that order).
    """
    return np.array([np.median(np.real(s)) + 1j * np.median(np.imag(s)) for s in ref_shots])


def assign_nearest(shots, centers):
    """-> index of the nearest center (0 g, 1 e, 2 f) for each complex shot."""
    shots = np.asarray(shots)
    return np.argmin(np.abs(shots[..., None] - np.asarray(centers)), axis=-1)


def assigned_fractions(shots, centers):
    """-> (3,) fractions of the shots assigned to g, e, f."""
    idx = np.ravel(assign_nearest(shots, centers))
    return np.bincount(idx, minlength=3)[:3] / max(idx.size, 1)


def assignment_matrix(ref_shots, centers):
    """-> A[prepared, assigned] (3, 3) from the g, e, f reference clouds."""
    return np.array([assigned_fractions(s, centers) for s in ref_shots])


def corrected_populations(raw, A):
    """-> populations p with raw = A.T @ p, for raw of shape (..., 3).

    Least squares: for an invertible A this is the exact inverse. For a singular A
    (clouds on top of each other, for example zero mock data) it gives the
    least-squares answer instead of an error.
    """
    raw = np.asarray(raw, float)
    flat = raw.reshape(-1, 3).T
    solution = np.linalg.lstsq(np.asarray(A, float).T, flat, rcond=None)[0]
    return solution.T.reshape(raw.shape)


def _exp_decay(t, amplitude, rate, offset):
    return amplitude * np.exp(-rate * t) + offset


def decay_fit(t, pop):
    """Fit one hold scan of |f,n>. -> dict of floats (1/us for rates).

    t: waits (us). pop: (len(t), 3) corrected populations (g, e, f).

    Model (as on 2026-10-29): P(f)(t) = amplitude * exp(-rate * t) + offset. The
    offset holds the slow part (for example |f,0> left by earlier pulses).
    ``share_to_g`` is minus the slope of P(g) against P(f) over the same points: the
    part of the P(f) loss that appears in g. ``rate_to_g`` = rate * share_to_g.
    ``rate_err`` is the fit error from the covariance (model, scaled by the residuals).

    All values are NaN when there are fewer than 4 points, P(f) does not change, or
    the fit fails.
    """
    out = dict(rate=np.nan, rate_err=np.nan, amplitude=np.nan, offset=np.nan,
               share_to_g=np.nan, rate_to_g=np.nan)
    t, pop = np.asarray(t, float), np.asarray(pop, float)
    if len(t) < 4 or not np.all(np.isfinite(pop)):
        return out
    pf, pg = pop[:, 2], pop[:, 0]
    if np.ptp(pf) < 1e-3:
        return out

    # Several start rates: the decays of interest go from ~1 us to ~100 us.
    t_max = max(t.max(), 1e-3)
    p0_amp = float(np.clip(pf[0] - pf[-1], 1e-3, 1.2))
    p0_off = float(np.clip(pf[-1], -0.1, 1.0))
    best = None
    for rate0 in np.array([0.3, 1, 3, 10, 30]) / t_max:
        try:
            popt, pcov = curve_fit(_exp_decay, t, pf, p0=[p0_amp, np.clip(rate0, 1e-4, 5), p0_off],
                                   bounds=([0, 1e-4, -0.1], [1.2, 5, 1]))
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((_exp_decay(t, *popt) - pf) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        return out

    _, (amplitude, rate, offset), pcov = best
    rate_err = np.sqrt(pcov[1, 1]) if np.all(np.isfinite(pcov)) else np.nan
    share_to_g = -np.polyfit(pf, pg, 1)[0]
    out.update(rate=float(rate), rate_err=float(rate_err), amplitude=float(amplitude),
               offset=float(offset), share_to_g=float(share_to_g),
               rate_to_g=float(rate * share_to_g))
    return out


def transition_frequencies(f_ge, f_ef, sideband_freqs, ns):
    """-> (omega_M1, nu): nu_n = E(f,n) - E(g,n-1) in MHz for each n in ns.

    omega_M1 = f_ge + f_ef - s_0 and nu_n = s_n + 2 * omega_M1, with s_n the
    calibrated f_n-g_(n+1) sideband frequency (multiphoton table). M1 self-Kerr is
    left out. The s_n are drive frequencies, so they include the sideband drive's
    Stark shift (a few MHz; see the 10-29 report). nu is NaN for n = 0 (no photon to
    lose) and for n without a table entry.
    """
    s = np.asarray(sideband_freqs, float)
    omega = float(f_ge + f_ef - s[0])
    nu = np.array([s[n] + 2 * omega if 1 <= n < len(s) else np.nan for n in ns], float)
    return omega, nu

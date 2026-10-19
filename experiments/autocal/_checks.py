"""Fit-quality checks for the calibration steps.

The notebooks write every fitted value into the config without a check. These
functions decide whether a fit is good enough to apply. Each returns
``(ok, info)``: ``info`` is a plain dict (JSON-safe) with the numbers the
decision used, so the run record shows why a step passed or failed.

The checks are deliberately model-light: the fit value must lie inside the
scanned window (not at an edge), the fitted feature must stand out from the
fit residuals (signal-to-noise), and for oscillations the fitted frequency must
be one the scan can resolve. Thresholds are arguments with defaults that pass
the clean scans of the notebooks and reject a fit to noise.
"""
import numpy as np

from fitting.fitting import decaysin, expfunc, gaussianfunc, lorfunc


def _plain(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer, int, float)) else v) for k, v in d.items()}


def _snr(y, yfit, amp):
    """|feature amplitude| / std of the fit residuals; 0 for flat data (no feature to fit)."""
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)) or np.ptp(y) == 0:
        return 0.0
    res = np.std(y - np.asarray(yfit, float))
    return float(min(abs(amp), np.ptp(y)) / max(res, 1e-12 * np.ptp(y)))


def inside(x0, lo, hi, margin=0.1):
    """x0 in [lo, hi] with `margin` (fraction of the span) kept free at both ends."""
    lo, hi = min(lo, hi), max(lo, hi)
    m = margin * (hi - lo)
    return bool(np.isfinite(x0) and lo + m <= x0 <= hi - m)


def lorentzian(x, y, p, min_snr=5.0, max_width_frac=0.25, margin=0.05):
    """Spectroscopy peak/dip: p = [y0, yscale, x0, xscale] (fitting.fitlor)."""
    x = np.asarray(x, float)
    if p is None or not np.all(np.isfinite(p)):
        return False, dict(reason='no fit')
    y0, ys, x0, w = [float(v) for v in p]
    span = float(x.max() - x.min())
    snr = _snr(y, lorfunc(x, *p), ys)
    info = dict(x0=x0, width=abs(w), snr=snr, lo=float(x.min()), hi=float(x.max()))
    ok = inside(x0, x.min(), x.max(), margin) and snr >= min_snr and 0 < abs(w) <= max_width_frac * span
    if not ok:
        info['reason'] = ('center at the edge' if not inside(x0, x.min(), x.max(), margin) else
                          'low signal-to-noise' if snr < min_snr else 'width out of range')
    return bool(ok), _plain(info)


def gaussian(x, y, p, min_amp=0.3, min_snr=4.0, margin=0.1):
    """Error-amplification product peak: p = [y0, yscale, x0, sigma] (fitting.fitgaussian), y normalised to max 1."""
    x = np.asarray(x, float)
    if p is None or not np.all(np.isfinite(p)):
        return False, dict(reason='no fit')
    y0, ys, x0, sig = [float(v) for v in p]
    span = float(x.max() - x.min())
    snr = _snr(y, gaussianfunc(x, *p), ys)
    info = dict(x0=x0, amp=ys, sigma=abs(sig), snr=snr, lo=float(x.min()), hi=float(x.max()))
    ok = (inside(x0, x.min(), x.max(), margin) and ys >= min_amp and snr >= min_snr
          and 0.01 * span < abs(sig) < 0.6 * span)
    if not ok:
        info['reason'] = ('center at the edge' if not inside(x0, x.min(), x.max(), margin) else
                          'low contrast' if ys < min_amp else 'low signal-to-noise' if snr < min_snr else 'width out of range')
    return bool(ok), _plain(info)


def ramsey(t, y, p, ramsey_freq, min_snr=4.0, band=(0.25, 1.75)):
    """Ramsey fringe: p = [yscale, freq, phase_deg, decay, y0, x0] (fitting.decaysin). The fitted frequency must lie
    within `band` x ramsey_freq: outside it the sign of the detuning is ambiguous or the fringe is not resolved."""
    t = np.asarray(t, float)
    if p is None or not np.all(np.isfinite(p)):
        return False, dict(reason='no fit')
    p = [float(v) for v in p]
    snr = _snr(y, decaysin(t, *p), p[0])
    f = abs(p[1])
    info = dict(freq=f, ramsey_freq=float(ramsey_freq), decay=abs(p[3]), snr=snr)
    ok = band[0] * ramsey_freq <= f <= band[1] * ramsey_freq and snr >= min_snr and abs(p[3]) > 2 * (t[1] - t[0])
    if not ok:
        info['reason'] = ('fringe frequency out of band (detuning too large)' if not band[0] * ramsey_freq <= f <= band[1] * ramsey_freq
                          else 'low signal-to-noise' if snr < min_snr else 'decays within two points')
    return bool(ok), _plain(info)


def rabi(x, y, p, pi_value, min_snr=4.0, margin=0.03):
    """Amplitude / length Rabi: p = [yscale, freq, phase_deg, decay, y0, (x0)]; the pi value must lie inside the scan."""
    x = np.asarray(x, float)
    if p is None or not np.all(np.isfinite(p)) or pi_value is None or not np.isfinite(pi_value):
        return False, dict(reason='no fit')
    p = [float(v) for v in p]
    pp = p if len(p) == 6 else p + [float(x[0])]
    snr = _snr(y, decaysin(x, *pp), p[0])
    info = dict(pi=float(pi_value), snr=snr, lo=float(x.min()), hi=float(x.max()))
    ok = inside(pi_value, x.min(), x.max(), margin) and snr >= min_snr
    if not ok:
        info['reason'] = 'pi value outside the scan' if not inside(pi_value, x.min(), x.max(), margin) else 'low signal-to-noise'
    return bool(ok), _plain(info)


def decay(t, y, p, min_snr=4.0, span_frac=(0.05, 3.0)):
    """T1: p = [y0, yscale, x0, decay] (fitting.expfunc); the decay time must be resolved by the scan."""
    t = np.asarray(t, float)
    if p is None or not np.all(np.isfinite(p)):
        return False, dict(reason='no fit')
    p = [float(v) for v in p]
    snr = _snr(y, expfunc(t, *p), p[1])
    span = float(t.max() - t.min())
    info = dict(T=abs(p[3]), snr=snr, span=span)
    ok = span_frac[0] * span <= abs(p[3]) <= span_frac[1] * span and snr >= min_snr
    if not ok:
        info['reason'] = 'decay time not resolved by the scan' if snr >= min_snr else 'low signal-to-noise'
    return bool(ok), _plain(info)


def jump(new, old, max_abs=None, max_rel=None):
    """The change from the old value is within the limit (a large jump means a wrong line or a wrong fit)."""
    d = float(new) - float(old)
    ok = (max_abs is None or abs(d) <= max_abs) and (max_rel is None or old == 0 or abs(d / float(old)) <= max_rel)
    info = dict(old=float(old), new=float(new), change=d)
    if not ok:
        info['reason'] = f'change {d:+.6g} larger than the limit ({max_abs if max_abs is not None else str(max_rel * 100) + " %"})'
    return bool(ok), info


def both(*checks):
    """Combine (ok, info) pairs: ok if all are; infos merged, first failure reason kept."""
    ok, info = True, {}
    for c_ok, c_info in checks:
        reason = c_info.get('reason')
        info.update({k: v for k, v in c_info.items() if k != 'reason'})
        if not c_ok and ok:
            ok, info['reason'] = False, reason
    return ok, info

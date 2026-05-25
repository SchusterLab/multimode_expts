"""Envelope model selection for decaying-sinusoid (Ramsey-style) fits.

Composes :mod:`fitting.fitting` low-level fitters into one helper that
fits both an exponential-envelope and a Gaussian-envelope sinusoid,
scores them by SSR with a safety margin, and returns the winner along
with full audit evidence.

Also hosts two small utilities used across migrated experiments:
:func:`pad_cov_to_6` (5x5 -> 6x6 covariance padding for the no-x0 fit)
and :func:`h5_safe_data` (drop string/object-typed dict entries that
``h5py`` cannot persist).
"""

import matplotlib.pyplot as plt
import numpy as np

import fitting.fitting as fitter


def pad_cov_to_6(cov):
    """Pad a 5x5 covariance matrix to 6x6, filling new row/col with inf.

    Used when the 5-param decaysin/gaussdecaysin fit (use_x0=False) is
    consumed by code that expects the 6-param x0-included shape. The new
    diagonal entry being inf marks the x0 dimension as unconstrained.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape == (6, 6):
        return cov
    out = np.full((6, 6), np.inf)
    n = min(cov.shape[0], 5)
    out[:n, :n] = cov[:n, :n]
    return out


def fit_decaysin_with_envelope_selection(
    xdata, ydata, *, fitparams=None, use_x0=False, gauss_ssr_margin=0.05,
):
    """Fit exp and Gaussian decaying-sinusoid envelopes; pick winner by SSR.

    Parameters
    ----------
    xdata, ydata : array-like
        Data to fit. ``ydata`` is the channel signal.
    fitparams : list, optional
        Initial parameters forwarded to both fitters. Same shape for both.
    use_x0 : bool, default False
        If True, fit a free time-offset x0; if False, pin x0=0 (anchors
        the dephasing envelope at the first pi/2 pulse) and pad results
        to 6-param / 6x6 so the return shape is invariant in ``use_x0``.
    gauss_ssr_margin : float, default 0.05
        Gaussian only wins if it lowers SSR by more than this fraction
        of the exp SSR. Otherwise default to exp for stability across
        sweeps where the true envelope is borderline.

    Returns
    -------
    dict
        Always-6-param ``p`` / 6x6 ``cov`` for the winner, plus full
        evidence (``p_exp``, ``cov_exp``, ``ssr_exp``, ``p_gauss``,
        ``cov_gauss``, ``ssr_gauss``) and the ``envelope`` tag
        ('exp' or 'gauss'). Caller prefixes channel names as needed.
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    p_exp, cov_exp = fitter.fitdecaysin(
        xdata, ydata, fitparams=fitparams, use_x0=use_x0)
    p_gauss, cov_gauss = fitter.fitgaussdecaysin(
        xdata, ydata, fitparams=fitparams, use_x0=use_x0)

    if not use_x0:
        p_exp = np.append(np.asarray(p_exp, dtype=float), 0.0)
        p_gauss = np.append(np.asarray(p_gauss, dtype=float), 0.0)
        cov_exp = pad_cov_to_6(cov_exp)
        cov_gauss = pad_cov_to_6(cov_gauss)

    ssr_exp = (
        float(np.sum((fitter.decaysin(xdata, *p_exp) - ydata) ** 2))
        if isinstance(p_exp, (list, np.ndarray)) else np.inf
    )
    ssr_gauss = (
        float(np.sum((fitter.gaussdecaysin(xdata, *p_gauss) - ydata) ** 2))
        if isinstance(p_gauss, (list, np.ndarray)) else np.inf
    )

    use_gauss = (
        np.isfinite(ssr_gauss) and np.isfinite(ssr_exp)
        and ssr_gauss < (1.0 - gauss_ssr_margin) * ssr_exp
    )
    envelope = 'gauss' if use_gauss else 'exp'

    return {
        'p': p_gauss if use_gauss else p_exp,
        'cov': cov_gauss if use_gauss else cov_exp,
        'p_exp': p_exp,
        'cov_exp': cov_exp,
        'ssr_exp': ssr_exp,
        'p_gauss': p_gauss,
        'cov_gauss': cov_gauss,
        'ssr_gauss': ssr_gauss,
        'envelope': envelope,
    }


def h5_safe_data(data, *, verbose=True):
    """Return a copy of ``data`` with h5py-incompatible entries dropped.

    ``slab.add_data`` calls ``h5py.create_dataset(dtype=str(arr.dtype))``,
    which rejects unicode/bytes/object scalar dtypes. The envelope tag
    ('exp' / 'gauss') and any other string evidence triggers this; the
    tag is recomputable from ``fit_ssr_*_{exp,gauss}`` on reload, so
    silent drop is safe.
    """
    safe = {}
    skipped = []
    for k, v in data.items():
        try:
            kind = np.asarray(v).dtype.kind
        except Exception:
            skipped.append(k)
            continue
        if kind in 'USO':
            skipped.append(k)
            continue
        safe[k] = v
    if verbose and skipped:
        print(f'[h5_safe_data] dropped non-numeric keys: {skipped}')
    return safe


def plot_envelope_overlay(x, p, *, envelope='exp', label=None, env_color='0.2'):
    """Draw fit curve + matching +/- envelope on the current matplotlib axes.

    Selects ``decaysin``/``expfunc`` or ``gaussdecaysin``/``gaussenvfunc``
    based on the ``envelope`` tag emitted by
    :func:`fit_decaysin_with_envelope_selection`.
    """
    if envelope == 'gauss':
        curve = fitter.gaussdecaysin(x, *p)
        env_pos = fitter.gaussenvfunc(x, p[4], p[0], p[5], p[3])
        env_neg = fitter.gaussenvfunc(x, p[4], -p[0], p[5], p[3])
    else:
        curve = fitter.decaysin(x, *p)
        env_pos = fitter.expfunc(x, p[4], p[0], p[5], p[3])
        env_neg = fitter.expfunc(x, p[4], -p[0], p[5], p[3])
    plt.plot(x, curve, label=label)
    plt.plot(x, env_pos, color=env_color, linestyle='--')
    plt.plot(x, env_neg, color=env_color, linestyle='--')

"""
Fitting functions for qubit experiments.

All fit* functions now use lmfit internally and return FitResult objects.
These are backward-compatible: ``p, pCov = fitdecaysin(...)`` still works.

For the new named-parameter API, see fitting.models and docs/lmfit_migration.md.
"""

import numpy as np

from fitting.models import (
    PARAM_ORDER,
    DecaySineDualRailModel,
    DecaySineModel,
    ExponentialModel,
    FitResult,
    GaussianPeakModel,
    HangerS21SlopedModel,
    LorentzianModel,
    RBModel,
    SineModel,
    TwoFreqDecaySineModel,
    _apply_user_guesses,
    _ensure_within_bounds,
)

# ====================================================== #
# Utility: pick best fit across measurement channels
# ====================================================== #

def get_best_fit(data, fitfunc=None, prefixes=['fit'], check_measures=('amps', 'avgi', 'avgq'), get_best_data_params=(), override=None):
    """Compare fits between check_measures channels and pick the best one.

    If fitfunc is specified, uses R^2 to determine best fit.
    """
    fit_errs = [data[f'{prefix}_err_{check}'] for check in check_measures for prefix in prefixes]

    # fix the error matrix so "0" error is adjusted to inf
    for fit_err_check in fit_errs:
        for i, fit_err in enumerate(np.diag(fit_err_check)):
            if fit_err == 0:
                fit_err_check[i][i] = np.inf

    fits = [data[f'{prefix}_{check}'] for check in check_measures for prefix in prefixes]

    if override is not None and override in check_measures:
        i_best = np.argwhere(np.array(check_measures) == override)[0][0]
    else:
        if fitfunc is not None:
            ydata = [data[check] for check in check_measures]
            xdata = data['xpts']
            ss_res_checks = np.array([np.sum((fitfunc(xdata, *fit_check) - ydata_check)**2) for fit_check, ydata_check in zip(fits, ydata)])
            ss_tot_checks = np.array([np.sum((np.mean(ydata_check) - ydata_check)**2) for ydata_check in ydata])
            r2 = 1 - ss_res_checks / ss_tot_checks
            i_best = np.argmin(r2)
        else:
            i_best = np.argmin([np.average(np.sqrt(np.abs(np.diag(fit_err) / fit))) for fit, fit_err in zip(fits, fit_errs)])

    best_data = [fits[i_best], fit_errs[i_best]]
    best_meas = check_measures[i_best]

    for param in get_best_data_params:
        best_data.append(data[f'{param}_{best_meas}'])
    return best_data


# ====================================================== #
# Bare model functions (kept for backward-compatible evaluation)
#   Usage: fitter.decaysin(xdata, *p)
# ====================================================== #

def expfunc(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)

def expfunc1(x, *p):
    y0, yscale, decay = p
    return y0 + yscale * np.exp(-x / decay)

def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0

def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0, x0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) * np.exp(-(x - x0) / decay) + y0

def decaysin1(x, *p):
    yscale, freq, phase_deg, decay, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) * np.exp(-x / decay) + y0

def decaysin_dualrail(x, *p):
    yscale, freq, phase_deg, decay, decay_phi, y0, x0_kappa, x0_phi = p
    return (yscale * (1 + np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
            * np.exp(-(x - x0_phi) / decay_phi)) * np.exp(-(x - x0_kappa) / decay) + y0)

def twofreq_decaysin(x, *p):
    yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0 = p
    return y0 + np.exp(-x / decay0) * yscale0 * (
        (1 - yscale1) * np.sin(2 * np.pi * freq0 * x + phase_deg0 * np.pi / 180)
        + yscale1 * np.sin(2 * np.pi * freq1 * x + phase_deg1 * np.pi / 180)
    )

def gaussianfunc(x, *p):
    y0, yscale, x0, sigma = p
    return y0 + yscale * np.exp(-((x - x0) / sigma) ** 2)

def lorfunc(x, *p):
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale ** 2)

def hangerfunc(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return scale * (1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0))

def hangerS21func(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return a0 + np.abs(hangerfunc(x, *p)) - scale * (1 - Q0 / Qe)

def hangerS21func_sloped(x, *p):
    f0, Qi, Qe, phi, scale, a0, slope = p
    return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope * (x - f0)

def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))

def rb_func(depth, p, a, b):
    return a * p ** depth + b


# ====================================================== #
# RB utilities (unchanged)
# ====================================================== #

def rb_error(p, d):
    """Average error rate over all gates. d = dim of system = 2^(number of qubits)."""
    return 1 - (p + (1 - p) / d)

def rb_gate_fidelity(p_rb, p_irb, d):
    """Gate fidelity from regular RB and interleaved RB."""
    return 1 - (d - 1) * (1 - p_irb / p_rb) / d


# ====================================================== #
# Fit functions (lmfit-backed, returning FitResult)
# ====================================================== #

def _do_fit(model, data, params, independent_var='x', xdata=None, **fit_kwargs):
    """Run an lmfit fit with graceful failure handling."""
    _ensure_within_bounds(params)
    kwargs = {independent_var: xdata}
    kwargs.update(fit_kwargs)
    try:
        result = model.fit(data, params, **kwargs)
    except Exception:
        print('Warning: fit failed!')
        # Return a result with the initial guess values
        result = model.fit(data, params, **kwargs, fit_kws={'maxfev': 1})
    return result


def fitexp(xdata, ydata, fitparams=None):
    model = ExponentialModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['exp'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata, max_nfev=200000)
    return FitResult(result, PARAM_ORDER['exp'])


def fitexp1(xdata, ydata, fitparams=None):
    """Fit exponential with x0 fixed at 0 (3-parameter version)."""
    model = ExponentialModel()
    params = model.guess(ydata, x=xdata)
    params['x0'].set(value=0, vary=False)
    if fitparams is not None:
        # fitexp1 legacy: [y0, yscale, decay] — 3 params, no x0
        # Map to ExponentialModel's 4 params: y0=0, yscale=1, x0=skip, decay=2
        expanded = [
            fitparams[0] if len(fitparams) > 0 else None,
            fitparams[1] if len(fitparams) > 1 else None,
            None,  # x0 is fixed
            fitparams[2] if len(fitparams) > 2 else None,
        ]
        params = _apply_user_guesses(params, PARAM_ORDER['exp'], expanded)
        params['x0'].set(value=0, vary=False)
    result = _do_fit(model, ydata, params, xdata=xdata)
    # Return with exp1 param order for backward compat
    return FitResult(result, PARAM_ORDER['exp1'])


def fitsin(xdata, ydata, fitparams=None):
    model = SineModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['sin'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['sin'])


def fitdecaysin(xdata, ydata, fitparams=None):
    model = DecaySineModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['decaysin'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['decaysin'])


def fitdecaysin1(xdata, ydata, fitparams=None):
    """Fit decaying sine with x0 fixed at 0."""
    model = DecaySineModel()
    params = model.guess(ydata, x=xdata)
    params['x0'].set(value=0, vary=False)
    params = _apply_user_guesses(params, PARAM_ORDER['decaysin'], fitparams)
    params['x0'].set(value=0, vary=False)  # re-fix after user guesses
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['decaysin'])


def fitdecaysin_dualrail(xdata, ydata, fitparams=None):
    model = DecaySineDualRailModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['decaysin_dr'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['decaysin_dr'])


def fittwofreq_decaysin(xdata, ydata, fitparams=None):
    model = TwoFreqDecaySineModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['twofreq'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['twofreq'])


def fitlor(xdata, ydata, fitparams=None):
    model = LorentzianModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['lor'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['lor'])


def fitgaussian(xdata, ydata, fitparams=None):
    model = GaussianPeakModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['gaussian'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['gaussian'])


def fithanger(xdata, ydata, fitparams=None):
    model = HangerS21SlopedModel()
    params = model.guess(ydata, x=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['hanger'], fitparams)
    result = _do_fit(model, ydata, params, xdata=xdata)
    return FitResult(result, PARAM_ORDER['hanger'])


def fitrb(xdata, ydata, fitparams=None):
    model = RBModel()
    params = model.guess(ydata, depth=xdata)
    params = _apply_user_guesses(params, PARAM_ORDER['rb'], fitparams)
    _ensure_within_bounds(params)
    try:
        result = model.fit(ydata, params, depth=xdata, method='least_squares', max_nfev=30000)
    except Exception:
        print('Warning: fit failed!')
        result = model.fit(ydata, params, depth=xdata, fit_kws={'maxfev': 1})
    return FitResult(result, PARAM_ORDER['rb'])

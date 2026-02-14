"""
lmfit Model classes for common fitting functions in qubit experiments.

Each Model subclass provides:
    - A named-parameter model function (no more positional *p)
    - A guess() method for automatic initial parameter estimation
    - Integration with lmfit's parameter fixing, bounds, and constraints

See docs/lmfit_migration.md for the parameter name reference table.
"""

import lmfit
import numpy as np

from fitting.fit_utils import guess_freq


# ====================================================================== #
# FitResult: backward-compatible wrapper around lmfit.ModelResult
# ====================================================================== #

class FitResult:
    """
    Wraps an lmfit.ModelResult to be backward-compatible with the old
    (pOpt, pCov) return convention.

    Usage (new API)::

        result = fit_decaysin(xdata, ydata)
        result['decay']           # named parameter access
        result.stderr('decay')    # standard error
        result.lmfit_result       # full lmfit.ModelResult

    Usage (backward-compatible)::

        p, pCov = fit_decaysin(xdata, ydata)
        p[3]        # still works (decay is index 3 for decaysin)
        pCov[3][3]  # still works
    """

    def __init__(self, lmfit_result, param_order):
        self.lmfit_result = lmfit_result
        self.param_order = param_order
        self._popt = np.array([
            lmfit_result.best_values.get(name, np.nan)
            for name in param_order
        ])
        self._pcov = self._build_covariance(lmfit_result, param_order)

    def _build_covariance(self, result, param_order):
        n = len(param_order)
        pcov = np.full((n, n), np.inf)
        if result.covar is not None:
            var_names = result.var_names  # only the varied params
            for i, name_i in enumerate(param_order):
                if name_i not in var_names:
                    continue
                li = var_names.index(name_i)
                for j, name_j in enumerate(param_order):
                    if name_j not in var_names:
                        continue
                    lj = var_names.index(name_j)
                    if li < result.covar.shape[0] and lj < result.covar.shape[1]:
                        pcov[i][j] = result.covar[li][lj]
        return pcov

    # --- Named access (new API) ---

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.lmfit_result.best_values[key]
        if isinstance(key, (int, np.integer)):
            return self._popt[key]
        raise KeyError(key)

    @property
    def params(self):
        return self.lmfit_result.params

    def stderr(self, name):
        p = self.lmfit_result.params[name]
        return p.stderr if p.stderr is not None else np.inf

    @property
    def best_fit(self):
        return self.lmfit_result.best_fit

    @property
    def rsquared(self):
        if hasattr(self.lmfit_result, 'rsquared'):
            return self.lmfit_result.rsquared
        return None

    # --- Backward-compatible unpacking ---

    def __iter__(self):
        """Allows: p, pCov = fitdecaysin(...)"""
        return iter((self._popt, self._pcov))

    def __len__(self):
        return 2

    def __repr__(self):
        lines = []
        rsq = self.rsquared
        if rsq is not None:
            lines.append(f"FitResult (R² = {rsq:.6f})")
        else:
            lines.append("FitResult")
        for name in self.param_order:
            val = self.lmfit_result.best_values.get(name, np.nan)
            err = self.stderr(name)
            lines.append(f"  {name:>12s} = {val:.6g} +/- {err:.3g}")
        return "\n".join(lines)


# ====================================================================== #
# Helper: apply user-supplied positional guesses onto lmfit params
# ====================================================================== #

def _apply_user_guesses(params, param_order, fitparams):
    """Override auto-guessed values with user-supplied values from a fitparams list.

    Values of None in fitparams are skipped (keep the auto-guess).
    """
    if fitparams is None:
        return params
    for i, val in enumerate(fitparams):
        if val is not None and i < len(param_order):
            name = param_order[i]
            if name in params:
                params[name].set(value=val)
    return params


def _ensure_within_bounds(params):
    """If a parameter value is outside its min/max bounds, clip to midpoint."""
    for name, par in params.items():
        if par.min is not None and par.max is not None:
            if not (par.min < par.value < par.max):
                par.set(value=(par.min + par.max) / 2)


# ====================================================================== #
# Parameter order mappings (legacy positional index → name)
# ====================================================================== #

PARAM_ORDER = {
    'exp':         ['y0', 'yscale', 'x0', 'decay'],
    'exp1':        ['y0', 'yscale', 'decay'],
    'sin':         ['yscale', 'freq', 'phase_deg', 'y0'],
    'decaysin':    ['yscale', 'freq', 'phase_deg', 'decay', 'y0', 'x0'],
    'decaysin_dr': ['yscale', 'freq', 'phase_deg', 'decay', 'decay_phi',
                    'y0', 'x0_kappa', 'x0_phi'],
    'twofreq':     ['yscale0', 'freq0', 'phase_deg0', 'decay0',
                    'yscale1', 'freq1', 'phase_deg1', 'y0'],
    'lor':         ['y0', 'yscale', 'x0', 'xscale'],
    'gaussian':    ['y0', 'yscale', 'x0', 'sigma'],
    'hanger':      ['f0', 'Qi', 'Qe', 'phi', 'scale', 'a0', 'slope'],
    'rb':          ['p', 'a', 'b'],
}


# ====================================================================== #
# Model Classes
# ====================================================================== #

class ExponentialModel(lmfit.Model):
    """y0 + yscale * exp(-(x - x0) / decay)

    For the old expfunc1 behavior (no x0), fix x0=0:
        params['x0'].set(value=0, vary=False)
    """

    def __init__(self, *args, **kwargs):
        def exponential(x, y0, yscale, x0, decay):
            return y0 + yscale * np.exp(-(x - x0) / decay)
        super().__init__(exponential, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        params = self.make_params(
            y0=data[-1],
            yscale=data[0] - data[-1],
            x0=x[0],
            decay=(x[-1] - x[0]) / 5,
        )
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class SineModel(lmfit.Model):
    """yscale * sin(2*pi*freq*x + phase_deg*pi/180) + y0"""

    def __init__(self, *args, **kwargs):
        def sine(x, yscale, freq, phase_deg, y0):
            return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0
        super().__init__(sine, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        freq_guess, phase_rad = guess_freq(x, data)
        xspan = x[-1] - x[0]
        params = self.make_params(
            yscale=np.ptp(data) / 2,
            freq=freq_guess,
            phase_deg=phase_rad * 180 / np.pi,
            y0=np.mean(data),
        )
        params['yscale'].set(min=0.5 * np.ptp(data) / 2, max=2 * np.ptp(data) / 2)
        params['freq'].set(min=0.1 / xspan, max=50 / xspan)
        params['phase_deg'].set(min=-360, max=360)
        params['y0'].set(min=np.min(data), max=np.max(data))
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class DecaySineModel(lmfit.Model):
    """yscale * sin(2*pi*freq*x + phase_deg*pi/180) * exp(-(x-x0)/decay) + y0

    For the old decaysin1 behavior (no x0), fix x0=0:
        params['x0'].set(value=0, vary=False)
    """

    def __init__(self, *args, **kwargs):
        def decaysine(x, yscale, freq, phase_deg, decay, y0, x0):
            return (yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
                    * np.exp(-(x - x0) / decay) + y0)
        super().__init__(decaysine, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        freq_guess, phase_rad = guess_freq(x, data)
        xspan = x[-1] - x[0]
        params = self.make_params(
            yscale=np.ptp(data) / 2,
            freq=freq_guess,
            phase_deg=phase_rad * 180 / np.pi,
            decay=xspan,
            y0=np.mean(data),
            x0=x[0],
        )
        params['yscale'].set(min=0.5 * np.ptp(data) / 2, max=1.5 * np.ptp(data) / 2)
        params['freq'].set(min=0.1 / xspan, max=50 / xspan)
        params['phase_deg'].set(min=-360, max=360)
        params['decay'].set(min=0.3 * xspan)
        params['y0'].set(min=np.min(data), max=np.max(data))
        params['x0'].set(min=x[0] - xspan, max=x[-1] + xspan)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class DecaySineDualRailModel(lmfit.Model):
    """yscale * (1 + sin(...) * exp(-(x-x0_phi)/decay_phi)) * exp(-(x-x0_kappa)/decay) + y0

    Dual-rail model with separate kappa (amplitude) and phi (dephasing) decay channels.
    """

    def __init__(self, *args, **kwargs):
        def decaysine_dualrail(x, yscale, freq, phase_deg, decay, decay_phi,
                               y0, x0_kappa, x0_phi):
            return (yscale
                    * (1 + np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
                       * np.exp(-(x - x0_phi) / decay_phi))
                    * np.exp(-(x - x0_kappa) / decay) + y0)
        super().__init__(decaysine_dualrail, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        freq_guess, phase_rad = guess_freq(x, data)
        xspan = x[-1] - x[0]
        params = self.make_params(
            yscale=np.ptp(data) / 2,
            freq=freq_guess,
            phase_deg=phase_rad * 180 / np.pi,
            decay=xspan,
            decay_phi=xspan,
            y0=np.mean(data),
            x0_kappa=x[0],
            x0_phi=x[0],
        )
        params['yscale'].set(min=0.5 * np.ptp(data) / 2, max=1.5 * np.ptp(data) / 2)
        params['freq'].set(min=0.1 / xspan, max=15 / xspan)
        params['phase_deg'].set(min=-360, max=360)
        params['decay'].set(min=0.3 * xspan)
        params['decay_phi'].set(min=0.3 * xspan)
        params['y0'].set(min=np.min(data), max=np.max(data))
        params['x0_kappa'].set(min=x[0] - xspan, max=x[-1] + xspan)
        params['x0_phi'].set(min=x[0] - xspan, max=x[-1] + xspan)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class TwoFreqDecaySineModel(lmfit.Model):
    """Two-frequency decaying sine: y0 + exp(-x/decay0) * yscale0 * (
        (1 - yscale1) * sin(2*pi*freq0*x + phase0) + yscale1 * sin(2*pi*freq1*x + phase1)
    )"""

    def __init__(self, *args, **kwargs):
        def twofreq_decaysine(x, yscale0, freq0, phase_deg0, decay0,
                              yscale1, freq1, phase_deg1, y0):
            return y0 + np.exp(-x / decay0) * yscale0 * (
                (1 - yscale1) * np.sin(2 * np.pi * freq0 * x + phase_deg0 * np.pi / 180)
                + yscale1 * np.sin(2 * np.pi * freq1 * x + phase_deg1 * np.pi / 180)
            )
        super().__init__(twofreq_decaysine, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        freq_guess, phase_rad = guess_freq(x, data)
        xspan = x[-1] - x[0]
        params = self.make_params(
            yscale0=np.ptp(data) / 2,
            freq0=freq_guess,
            phase_deg0=phase_rad * 180 / np.pi,
            decay0=xspan,
            yscale1=0.1,
            freq1=0.5,
            phase_deg1=0,
            y0=np.mean(data),
        )
        params['yscale0'].set(min=0.75 * np.ptp(data) / 2, max=1.25 * np.ptp(data) / 2)
        params['freq0'].set(min=0.1 / xspan, max=30 / xspan)
        params['phase_deg0'].set(min=-360, max=360)
        params['decay0'].set(min=0.1 * xspan)
        params['yscale1'].set(min=0.001, max=0.5)
        params['freq1'].set(min=0.01, max=10)
        params['phase_deg1'].set(min=-360, max=360)
        params['y0'].set(min=np.min(data), max=np.max(data))
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class LorentzianModel(lmfit.Model):
    """y0 + yscale / (1 + (x - x0)^2 / xscale^2)"""

    def __init__(self, *args, **kwargs):
        def lorentzian(x, y0, yscale, x0, xscale):
            return y0 + yscale / (1 + (x - x0) ** 2 / xscale ** 2)
        super().__init__(lorentzian, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        y0_guess = (data[0] + data[-1]) / 2
        params = self.make_params(
            y0=y0_guess,
            yscale=np.max(data) - np.min(data),
            x0=x[np.argmax(np.abs(data - y0_guess))],
            xscale=(x[-1] - x[0]) / 10,
        )
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class GaussianPeakModel(lmfit.Model):
    """y0 + yscale * exp(-((x - x0) / sigma)^2)"""

    def __init__(self, *args, **kwargs):
        def gaussian_peak(x, y0, yscale, x0, sigma):
            return y0 + yscale * np.exp(-((x - x0) / sigma) ** 2)
        super().__init__(gaussian_peak, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        params = self.make_params(
            y0=np.min(data),
            yscale=np.max(data) - np.min(data),
            x0=x[np.argmax(data)],
            sigma=(x[-1] - x[0]) / 10,
        )
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class HangerS21SlopedModel(lmfit.Model):
    """Hanger resonator S21 transmission with linear background slope."""

    def __init__(self, *args, **kwargs):
        def hanger_s21_sloped(x, f0, Qi, Qe, phi, scale, a0, slope):
            Q0 = 1 / (1 / Qi + np.real(1 / Qe))
            hanger = scale * (1 - Q0 / Qe * np.exp(1j * phi)
                              / (1 + 2j * Q0 * (x - f0) / f0))
            return a0 + np.abs(hanger) - scale * (1 - Q0 / Qe) + slope * (x - f0)
        super().__init__(hanger_s21_sloped, independent_vars=['x'], *args, **kwargs)

    def guess(self, data, x, **kwargs):
        params = self.make_params(
            f0=np.average(x),
            Qi=5000,
            Qe=1000,
            phi=0,
            scale=np.ptp(data),
            a0=np.average(data),
            slope=(data[-1] - data[0]) / (x[-1] - x[0]),
        )
        params['f0'].set(min=np.min(x), max=np.max(x))
        params['scale'].set(min=0)
        params['a0'].set(min=np.min(data), max=np.max(data))
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class RBModel(lmfit.Model):
    """a * p^depth + b  (randomized benchmarking decay)"""

    def __init__(self, *args, **kwargs):
        def rb_decay(depth, p, a, b):
            return a * p ** depth + b
        super().__init__(rb_decay, independent_vars=['depth'], *args, **kwargs)

    def guess(self, data, depth, **kwargs):
        params = self.make_params(
            p=0.9,
            a=np.max(data) - np.min(data),
            b=np.min(data),
        )
        params['p'].set(min=-1, max=0.99999)
        params['a'].set(min=-1, max=1)
        params['b'].set(min=-1, max=1)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

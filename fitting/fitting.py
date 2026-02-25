# from audioop import avg
import cmath

import numpy as np
import scipy as sp

# ====================================================== #

"""
Compare the fit between the check_measures (amps, avgi, and avgq by default) in data, using the compare_param_i-th parameter to do the comparison. Pick the best method of measurement out of the check_measures, and return the fit, fit_err, and any other get_best_data_params corresponding to that measurement.

If fitfunc is specified, uses R^2 to determine best fit.
"""
def get_best_fit(data, fitfunc=None, prefixes=['fit'], check_measures=('amps', 'avgi', 'avgq'), get_best_data_params=(), override=None):
    fit_errs = [data[f'{prefix}_err_{check}'] for check in check_measures for prefix in prefixes]

    # fix the error matrix so "0" error is adjusted to inf
    for fit_err_check in fit_errs:
        for i, fit_err in enumerate(np.diag(fit_err_check)):
            if fit_err == 0: fit_err_check[i][i] = np.inf

    fits = [data[f'{prefix}_{check}'] for check in check_measures for prefix in prefixes]

    if override is not None and override in check_measures:
        i_best = np.argwhere(np.array(check_measures) == override)[0][0]
        print(i_best)
    else:
        if fitfunc is not None:
            ydata = [data[check] for check in check_measures]  # need to figure out how to make this support multiple qubits readout
            xdata = data['xpts']

            # residual sum of squares
            ss_res_checks = np.array([np.sum((fitfunc(xdata, *fit_check) - ydata_check)**2) for fit_check, ydata_check in zip(fits, ydata)])
            # total sum of squares
            ss_tot_checks = np.array([np.sum((np.mean(ydata_check) - ydata_check)**2) for ydata_check in ydata])
            # R^2 value
            r2 = 1- ss_res_checks / ss_tot_checks
            i_best = np.argmin(r2)
            
        else:
            # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
            # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i] / fit[compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
            i_best = np.argmin([np.average(np.sqrt(np.abs(np.diag(fit_err) / fit))) for fit, fit_err in zip(fits, fit_errs)])

    best_data = [fits[i_best], fit_errs[i_best]]
    best_meas = check_measures[i_best]

    for param in get_best_data_params:
        best_data.append(data[f'{param}_{best_meas}'])
    return best_data

# ====================================================== #

def expfunc(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale*np.exp(-(x-x0)/decay)

def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0]-ydata[-1]
    if fitparams[2] is None: fitparams[2] = xdata[0]
    if fitparams[3] is None: fitparams[3] = (xdata[-1]-xdata[0])/5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams, maxfev=200000)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

def expfunc_y0fixed(x, y0, *p):
    yscale, x0, decay = p
    return y0 + yscale*np.exp(-(x-x0)/decay)

def fitexp_y0fixed(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    #if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[0] = ydata[0]-ydata[-1]
    if fitparams[2] is None: fitparams[1] = xdata[0]
    if fitparams[3] is None: fitparams[2] = (xdata[-1]-xdata[0])/5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams, maxfev=200000)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

def twofreq_decaysin(x, *p):
    yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0 = p
    return y0 + np.exp(-x / decay0) * yscale0 * (
        (1 - yscale1) * np.sin(2 * np.pi * freq0 * x + phase_deg0 * np.pi / 180)
        + yscale1 * np.sin(2 * np.pi * freq1 * x + phase_deg1 * np.pi / 180)
    )


def fittwofreq_decaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 10
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale0
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq0
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg0
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)  # exp decay
    if fitparams[4] is None:
        fitparams[4] = 0.1  # yscale1
    if fitparams[5] is None:
        fitparams[5] = 0.5  # MHz
    if fitparams[6] is None:
        fitparams[6] = 0  # phase_deg1
    if fitparams[7] is None:
        fitparams[7] = np.mean(ydata)  # y0
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.1 * (max(xdata) - min(xdata)),
            0.001,
            0.01,
            -360,
            np.min(ydata),
        ],
        [1.25 * fitparams[0], 30 / (max(xdata) - min(xdata)), 360, np.inf, 0.5, 10, 360, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(twofreq_decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov

def expfunc1(x, *p):
    y0, yscale, decay = p
    return y0 + yscale*np.exp(x/decay/-1)

def fitexp1(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*3
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0]-ydata[-1]
    if fitparams[2] is None: fitparams[2] = (xdata[-1]-xdata[0])/5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc1, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def lorfunc(x, *p):
    y0, yscale, x0, xscale = p
    return y0 + yscale/(1+(x-x0)**2/xscale**2)

def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1])/2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata)-min(xdata))/10
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(lorfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) + y0

def fitsin(xdata, ydata, fitparams=None, fixed_freq=None):
    if fitparams is None: fitparams = [None]*4
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    max_ind = np.argmax(np.abs(fourier[1:])) + 1
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq if fixed_freq is None else fixed_freq
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=np.mean(ydata)

    if fixed_freq is not None:
        # 3-parameter fit with frequency fixed
        reduced_params = [fitparams[0], fitparams[2], fitparams[3]]
        bounds = (
            [0.5*fitparams[0], -360, np.min(ydata)],
            [2*fitparams[0], 360, np.max(ydata)]
        )
        for i, param in enumerate(reduced_params):
            if not (bounds[0][i] < param < bounds[1][i]):
                reduced_params[i] = np.mean((bounds[0][i], bounds[1][i]))
                print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {reduced_params[i]}')
        pOpt_reduced = reduced_params
        pCov_reduced = np.full(shape=(3, 3), fill_value=np.inf)
        def sinfunc_fixed(x, yscale, phase_deg, y0):
            return sinfunc(x, yscale, fixed_freq, phase_deg, y0)
        try:
            pOpt_reduced, pCov_reduced = sp.optimize.curve_fit(
                sinfunc_fixed, xdata, ydata, p0=reduced_params, bounds=bounds)
        except RuntimeError:
            print('Warning: fit failed!')
        # Reconstruct full 4-element arrays for backward compatibility
        pOpt = np.array([pOpt_reduced[0], fixed_freq, pOpt_reduced[1], pOpt_reduced[2]])
        pCov = np.zeros((4, 4))
        for i_r, i_f in enumerate([0, 2, 3]):
            for j_r, j_f in enumerate([0, 2, 3]):
                pCov[i_f, j_f] = pCov_reduced[i_r, j_r]
        return pOpt, pCov

    bounds = (
        [0.5*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, np.min(ydata)],
        [2*fitparams[0], 10/(max(xdata)-min(xdata)), 360, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError:
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0, x0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-(x-x0)/decay) + y0

def decaysin1(x, *p):
    yscale, freq, phase_deg, decay, y0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-x/decay) + y0

def decaysin_dualrail(x, *p):
    yscale, freq, phase_deg, decay, decay_phi, y0, x0_kappa, x0_phi = p
    return yscale* (1 + np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-(x-x0_phi)/decay_phi)) * np.exp(-(x-x0_kappa)/decay)  + y0

def fitdecaysin_dualrail(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*8
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=max(xdata) - min(xdata)
    if fitparams[5] is None: fitparams[5]=np.mean(ydata)
    if fitparams[6] is None: fitparams[6]=xdata[0]
    if fitparams[7] is None: fitparams[7]=xdata[0]
    bounds = (
        [0.5*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.3*(max(xdata)-min(xdata)), 0.3*(max(xdata)-min(xdata)), np.min(ydata), xdata[0]-(xdata[-1]-xdata[0]), xdata[0]-(xdata[-1]-xdata[0])],
        [1.5*fitparams[0], 15/(max(xdata)-min(xdata)), 360, np.inf, np.inf, np.max(ydata), xdata[-1]+(xdata[-1]-xdata[0]), xdata[-1]+(xdata[-1]-xdata[0])]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin_dualrail, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

def fitdecaysin1(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*6
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    if fitparams[5] is None: fitparams[5]=xdata[0]

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

def _clamp_to_bounds(fitparams, bounds, verbose=True):
    """Push any out-of-bounds initial params to the midpoint of their bounds."""
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            old = param
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            if verbose:
                print(f'Attempted to init fitparam {i} to {old}, '
                      f'which is out of bounds {bounds[0][i]} to {bounds[1][i]}. '
                      f'Instead init to {fitparams[i]}')
    return fitparams


def _residual_ss(model, xdata, ydata, params):
    """Sum of squared residuals."""
    return np.sum((model(xdata, *params) - ydata) ** 2)


def fitdecaysin(xdata, ydata, fitparams=None, use_x0=True):
    """Fit decaying sinusoid with multi-start for robustness.

    Tries the primary initial guess first, then retries with shifted phases
    (0, 90, 180, 270 offsets) if the primary fit fails or gives poor results.
    If use_x0 model fails entirely, falls back to the simpler 5-param model.
    """
    from fitting.fit_utils import guess_decaysin_params

    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    n_params = 6 if use_x0 else 5
    model = decaysin if use_x0 else decaysin1

    if fitparams is None:
        fitparams = [None] * n_params

    # --- Auto-guess any None params using improved estimator ---
    has_nones = any(p is None for p in fitparams)
    if has_nones:
        amp_g, freq_g, phase_g, decay_g, offset_g = guess_decaysin_params(
            xdata, ydata)
        if fitparams[0] is None: fitparams[0] = amp_g
        if fitparams[1] is None: fitparams[1] = freq_g
        if fitparams[2] is None: fitparams[2] = phase_g
        if fitparams[3] is None: fitparams[3] = decay_g
        if fitparams[4] is None: fitparams[4] = offset_g
        if use_x0 and fitparams[5] is None:
            fitparams[5] = xdata[0]

    xrange = max(xdata) - min(xdata)
    ymin, ymax = np.min(ydata), np.max(ydata)
    yptp = ymax - ymin
    if yptp == 0:
        yptp = 1e-10  # avoid zero-width bounds for flat data

    if use_x0:
        bounds = (
            [0, 0.1/xrange, -360, 0.1*xrange, ymin - 0.1*yptp,
             xdata[0] - xrange],
            [1.5*yptp, 50/xrange, 360, np.inf, ymax + 0.1*yptp,
             xdata[-1] + xrange]
        )
    else:
        bounds = (
            [0, 0.1/xrange, -360, 0.1*xrange, ymin - 0.1*yptp],
            [1.5*yptp, 50/xrange, 360, np.inf, ymax + 0.1*yptp]
        )

    fitparams = list(fitparams)
    fitparams = _clamp_to_bounds(fitparams, bounds)

    # --- Multi-start: try primary guess + phase-shifted variants ---
    phase_offsets = [0, 90, 180, 270]
    best_pOpt = list(fitparams)
    best_pCov = np.full((n_params, n_params), np.inf)
    best_resid = np.inf

    for dph in phase_offsets:
        p0 = list(fitparams)
        p0[2] = fitparams[2] + dph
        # wrap phase into [-360, 360]
        while p0[2] > 360: p0[2] -= 720
        while p0[2] < -360: p0[2] += 720
        p0 = _clamp_to_bounds(p0, bounds, verbose=False)
        try:
            pOpt, pCov = sp.optimize.curve_fit(
                model, xdata, ydata, p0=p0, bounds=bounds, maxfev=20000)
            resid = _residual_ss(model, xdata, ydata, pOpt)
            if resid < best_resid:
                best_resid = resid
                best_pOpt = pOpt
                best_pCov = pCov
        except (RuntimeError, ValueError):
            continue

    # --- Fallback: try 5-param model if 6-param failed entirely ---
    if use_x0 and best_resid == np.inf:
        try:
            p0_5 = list(fitparams[:5])
            bounds_5 = (list(bounds[0][:5]), list(bounds[1][:5]))
            p0_5 = _clamp_to_bounds(p0_5, bounds_5, verbose=False)
            pOpt5, pCov5 = sp.optimize.curve_fit(
                decaysin1, xdata, ydata, p0=p0_5, bounds=bounds_5,
                maxfev=20000)
            # pad to 6-param format for compatibility
            best_pOpt = list(pOpt5) + [xdata[0]]
            best_pCov_5 = np.full((6, 6), np.inf)
            best_pCov_5[:5, :5] = pCov5
            best_pCov = best_pCov_5
            best_resid = _residual_ss(decaysin1, xdata, ydata, pOpt5)
        except (RuntimeError, ValueError):
            pass

    if best_resid == np.inf:
        print('Warning: fit failed (all starts)!')

    return best_pOpt, best_pCov




def gaussianfunc(x, *p):
    y0, yscale, x0, sigma = p
    return y0 + yscale * np.exp(-((x - x0) / sigma) ** 2)

def fitgaussian(xdata, ydata, fitparams=None, periodic=True):
    """Fit a Gaussian to data.

    Args:
        periodic: If True, handle wrap-around by rolling ydata so the peak
            is centered before fitting, then shifting x0 back. Use this when
            the x-axis is periodic (e.g. phase in degrees) and the peak may
            straddle the sweep boundary.
    """
    if fitparams is None: fitparams = [None]*4

    n = len(ydata)
    x_shift = 0.0
    dx = (xdata[-1] - xdata[0]) / (n - 1) if n > 1 else 1.0

    if periodic:
        # Roll ydata so the maximum is near the center of the array
        i_max = np.argmax(ydata)
        shift = n // 2 - i_max
        ydata = np.roll(ydata, shift)
        x_shift = shift * dx

    if fitparams[0] is None: fitparams[0]=np.min(ydata)
    if fitparams[1] is None: fitparams[1]=np.max(ydata)-np.min(ydata)
    if fitparams[2] is None: fitparams[2]=xdata[np.argmax(ydata)]
    if fitparams[3] is None: fitparams[3]= (max(xdata)-min(xdata))/10

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(gaussianfunc, xdata, ydata, p0=fitparams)
        if periodic and x_shift != 0:
            pOpt = pOpt.copy()
            pOpt[2] -= x_shift
            # Wrap x0 back into the original x range
            x_period = xdata[-1] - xdata[0] + dx
            pOpt[2] = xdata[0] + (pOpt[2] - xdata[0]) % x_period
    except RuntimeError:
        print('Warning: fit failed!')
    return pOpt, pCov

# ====================================================== #
    
def hangerfunc(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

def hangerS21func(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return a0 + np.abs(hangerfunc(x, *p)) - scale*(1-Q0/Qe)

def hangerS21func_sloped(x, *p):
    f0, Qi, Qe, phi, scale, a0, slope = p
    return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope*(x-f0)

def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))

def fithanger(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*7
    if fitparams[0] is None: fitparams[0]=np.average(xdata)
    if fitparams[1] is None: fitparams[1]=5000
    if fitparams[2] is None: fitparams[2]=1000
    if fitparams[3] is None: fitparams[3]=0
    if fitparams[4] is None: fitparams[4]=max(ydata)-min(ydata)
    if fitparams[5] is None: fitparams[5]=np.average(ydata)
    if fitparams[6] is None: fitparams[6]=(ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])

    print(fitparams)

    # bounds = (
    #     [np.min(xdata), -1e9, -1e9, -2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))/10, -np.max(np.abs(ydata))],
    #     [np.max(xdata), 1e9, 1e9, 2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))*10, np.max(np.abs(ydata))]
    #     )
    bounds = (
        [np.min(xdata), -np.inf, -np.inf, -np.inf, 0, min(ydata), -np.inf],
        [np.max(xdata), np.inf, np.inf, np.inf, np.inf, max(ydata), np.inf],
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=fitparams) #, bounds=bounds)
        print(pOpt)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def rb_func(depth, p, a, b):
    return a*p**depth + b

# Gives the average error rate over all gates in sequence
def rb_error(p, d): # d = dim of system = 2^(number of qubits)
    return 1 - (p + (1-p)/d)

# Run both regular RB and interleaved RB to calculate this
def rb_gate_fidelity(p_rb, p_irb, d):
    return 1 - (d-1)*(1-p_irb/p_rb) / d

def fitrb(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*3
    if fitparams[0] is None: fitparams[0]=0.9
    if fitparams[1] is None: fitparams[1]=np.max(ydata) - np.min(ydata)
    if fitparams[2] is None: fitparams[2]=np.min(ydata)
    bounds = (
        [-1, -1,-1],
        [0.99999, 1, 1]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    # try:
    pOpt, pCov = sp.optimize.curve_fit(rb_func, xdata, ydata, p0=fitparams, method='trf', max_nfev=30000) #, bounds=bounds)
        # return pOpt, pCov
    # except RuntimeError: 
    #     print('Warning: fit failed!')
    #     # return 0, 0
    return pOpt, pCov

def wigner_purity_calculation(wigner_expt,
                              cutoff = 10,
                              return_rho = False,
                              return_sweep_param = False,
                              plot_purity = False
                              ):
    
    import qutip as qt
    import matplotlib.pyplot as plt
    if isinstance(wigner_expt, list):
        return_list = []
        for _ind_wig in wigner_expt:
            return_list.append(wigner_purity_calculation(_ind_wig,
                                                         cutoff = cutoff,
                                                         return_rho = return_rho,
                                                         return_sweep_param = return_sweep_param,
                                                         plot_purity = plot_purity))
        return return_list
    ideal_state = (qt.coherent(cutoff, 1.0)).unit()
    # wigner_expt.analyze_wigner(cutoff=cutoff, debug=True)
    wigner_expt.analyze(rotate=True, 
                        initial_state=ideal_state, 
                        mode_state_num=cutoff, 
                        station=None, 
                        save_fig=False, 
                        cutoff=cutoff, 
                        debug=True)
    # wigner_expt.display(rotate=True, initial_state=ideal_state, mode_state_num=cutoff, station=None, save_fig=False)
    wigner_expt.display()
    plt.clf()
    outer_param_array = wigner_expt.outer_params
    inner_param_array = wigner_expt.inner_params
    _n_outer = len(outer_param_array)
    _n_inner = 1 if wigner_expt.inner_param == 'dummy' else len(inner_param_array)
    purity_array = np.zeros((_n_outer,_n_inner), dtype = complex)
    rho_array = wigner_expt.data["wigner_outputs"]['rho']
    for i in range(_n_outer):
        for j in range(_n_inner):
            rho = rho_array[i][j]
            purity_array[i,j] = np.trace(np.matmul(rho, rho))
    # purity_list.append(purity_array)
    plt.close('all')
    if plot_purity == True:
        if _n_inner != 1:
            fig, ax = plt.subplots()
            c = ax.pcolor(*np.meshgrid(outer_param_array,inner_param_array), 
                          purity_array)
            ax.set_xlabel(f'{wigner_expt.cfg.expt.swept_params[0]}')
            ax.set_ylabel(f'{wigner_expt.cfg.expt.swept_params[1]}')
            fig.colorbar(c,
                         label = "Purity")
        else:
            fig, ax = plt.subplots()
            ax.scatter(outer_param_array,
                       purity_array)
            ax.set_xlabel(f'{wigner_expt.cfg.expt.swept_params[0]}')
            ax.set_ylabel("Purity")
        plt.show()
        
    if return_rho == True:
        if return_sweep_param == True:
            return outer_param_array, inner_param_array, purity_array, rho_array
        else:
            return purity_array, rho_array
    else:
        return purity_array

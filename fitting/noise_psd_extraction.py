"""
CPMG noise-spectrum extraction (Bylander et al., Nat. Phys. 7, 565 (2011)).

Given CPMG decay curves at different pulse numbers N, this module:
  1. fits each decay to extract the coherence integral chi_N(tau),
  2. computes the CPMG filter function g_N(omega, tau) numerically,
  3. inverts the narrow-filter approximation (Bylander Eq. 26)
         S(omega') = chi_N(tau) / (dwdlambda^2 * tau^2 * g_N * Delta_omega)
     to produce one S(omega') point per (N, tau).

Conventions
-----------
- All times in microseconds. omega is in rad/us. f = omega / (2*pi) is in MHz.
- chi_N(tau) is dimensionless (it lives in the exponent of the coherence
  decay). For an exponential envelope exp(-tau/T2): chi_N = tau/T2. For
  Gaussian exp(-(tau/T2)^2): chi_N = (tau/T2)^2.

Reference: Bylander et al., arXiv:1101.4707, Eq. 18-26.
"""

import numpy as np
import matplotlib.pyplot as plt

import fitting.fitting as fitter
from fitting.decaysin_analysis import fit_decaysin_with_envelope_selection


# ----- filter function ---------------------------------------------------- #

def cpmg_pulse_positions(N):
    """Standard CPMG pulse positions delta_j = (2j-1)/(2N) for j=1..N.

    These are the fractions of the total free-evolution time tau where each
    pi pulse is centered (Bylander 2011 Eq. 18 layout).
    """
    if N < 1:
        return np.array([])
    return (2 * np.arange(1, N + 1) - 1) / (2 * N)


def cpmg_filter_function(omega, tau, N, tau_pi=0.0):
    """CPMG filter function g_N(omega, tau) per Bylander Eq. 21-22.

    g_N(omega, tau) = |y_N(omega, tau)|^2 / (omega * tau)^2

    Parameters
    ----------
    omega : array-like, [rad/us]
    tau : float, total free evolution time [us]
    N : int, number of pi pulses (N=0 returns the Ramsey filter)
    tau_pi : float, optional, pi-pulse width [us]. Default 0
            (instantaneous-pulse approximation; valid when tau_pi << tau/N).

    Returns
    -------
    g_N : ndarray, same shape as omega. Dimensionless.
    """
    omega = np.asarray(omega, dtype=float)
    if N == 0:
        # Ramsey: g_0 = sinc^2(omega tau / 2)
        wt2 = omega * tau / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(wt2 == 0, 1.0, (np.sin(wt2) / wt2) ** 2)

    deltas = cpmg_pulse_positions(N)
    wt = omega * tau
    cos_corr = np.cos(omega * tau_pi / 2.0) if tau_pi else 1.0

    sign = (-1.0) ** (1 + N)
    y = 1.0 + sign * np.exp(1j * wt)
    for j, dj in enumerate(deltas, start=1):
        y = y + 2.0 * ((-1.0) ** j) * np.exp(1j * omega * dj * tau) * cos_corr

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(omega == 0, 0.0, np.abs(y) ** 2 / wt ** 2)


def filter_peak(N, tau, *, omega_range=None, n_points=4001, tau_pi=0.0):
    """Find the main peak omega' and FWHM bandwidth Delta_omega of g_N.

    Returns (omega_peak, bandwidth), both in rad/us. For N=0 returns
    (0, pi/tau) as a rough characteristic width.
    """
    if N == 0:
        return 0.0, np.pi / tau

    if omega_range is None:
        # Main peak expected near pi * N / tau; sweep around it.
        omega_center = np.pi * N / tau
        omega_range = (omega_center / 10.0, 4.0 * omega_center)

    omegas = np.linspace(omega_range[0], omega_range[1], n_points)
    g = cpmg_filter_function(omegas, tau, N, tau_pi=tau_pi)

    i_max = int(np.argmax(g))
    omega_peak = float(omegas[i_max])
    g_max = float(g[i_max])
    half = g_max / 2.0

    # FWHM walk left and right from the peak
    i_left = i_max
    while i_left > 0 and g[i_left] > half:
        i_left -= 1
    i_right = i_max
    while i_right < len(g) - 1 and g[i_right] > half:
        i_right += 1
    bandwidth = float(omegas[i_right] - omegas[i_left])
    if bandwidth <= 0:
        bandwidth = np.pi / tau  # fallback
    return omega_peak, bandwidth


# ----- per-N decay fitting ------------------------------------------------ #

def fit_cpmg_decays(
    mother_data, *, channel='avgi', sweep_param_key='echoes_sweep',
    use_x0=False, gauss_ssr_margin=0.05,
):
    """Fit each CPMG-N decay curve from a SweepRunner mother experiment.

    Parameters
    ----------
    mother_data : dict
        SweepRunner-style data with keys 'xpts' (tau values), the swept
        parameter (e.g. 'echoes_sweep') giving the N list, and per-point
        'avgi' / 'avgq' arrays of shape (n_N, n_tau).
    channel : 'avgi' | 'avgq' | 'amps'
    sweep_param_key : str (default 'echoes_sweep')
    use_x0, gauss_ssr_margin : forwarded to fit_decaysin_with_envelope_selection.

    Returns
    -------
    dict with keys: 'N_values', 'taus', 'fits' (list of per-N fit dicts),
    'T2' (us), 'envelope' (list of 'exp'/'gauss').
    """
    N_values = np.asarray(mother_data[sweep_param_key])
    taus_raw = np.asarray(mother_data['xpts'])
    # SweepRunner appends xpts once per sweep point, so xpts may end up as
    # (n_N, n_tau). All rows are identical since the tau sweep is the same
    # per N — collapse to 1D.
    taus = taus_raw[0] if taus_raw.ndim == 2 else taus_raw
    ydata_all = np.asarray(mother_data[channel])  # (n_N, n_tau)

    fits, T2, envelopes = [], [], []
    for i, _ in enumerate(N_values):
        r = fit_decaysin_with_envelope_selection(
            taus, ydata_all[i], use_x0=use_x0,
            gauss_ssr_margin=gauss_ssr_margin)
        fits.append(r)
        p = r.get('p')
        try:
            T2.append(float(p[3]))
        except (TypeError, IndexError):
            T2.append(np.nan)
        envelopes.append(r.get('envelope', 'exp'))

    return {
        'N_values': N_values,
        'taus': taus,
        'fits': fits,
        'T2': np.asarray(T2),
        'envelope': envelopes,
    }


def filter_cpmg_data(mother_data, fit_summary, *,
                     exclude_N=None, keep_N=None,
                     sweep_param_key='echoes_sweep'):
    """Drop selected N values from the SweepRunner mother_data and the matching
    fit_summary in lockstep, so downstream calls (extract_psd, plot_cpmg_decays,
    plot_T2_vs_N, plot_envelope_diagnostic) all see the same filtered view.

    Usage:
        mother_man.data, fits_man = filter_cpmg_data(
            mother_man.data, fits_man, exclude_N=[0])
        # or
        m_f, f_f = filter_cpmg_data(mother_man.data, fits_man, keep_N=[1, 2, 4])

    Pass exactly one of `exclude_N` or `keep_N` (else returns input untouched).
    """
    N_values = np.asarray(fit_summary['N_values'])
    if exclude_N is not None:
        mask = ~np.isin(N_values, list(exclude_N))
    elif keep_N is not None:
        mask = np.isin(N_values, list(keep_N))
    else:
        return mother_data, fit_summary
    keep_idx = np.where(mask)[0]
    n_rows = len(N_values)

    # Filter fit_summary
    new_fit = dict(fit_summary)
    new_fit['N_values'] = N_values[mask]
    new_fit['T2']       = np.asarray(fit_summary['T2'])[mask]
    new_fit['envelope'] = [fit_summary['envelope'][i] for i in keep_idx]
    new_fit['fits']     = [fit_summary['fits'][i] for i in keep_idx]
    # taus is shared across N, leave unchanged.

    # Filter mother_data: any per-point array (first axis matches n_rows) is masked.
    new_data = {}
    for k, v in mother_data.items():
        arr = np.asarray(v) if not isinstance(v, (list, np.ndarray)) else v
        try:
            if isinstance(arr, np.ndarray) and arr.shape and arr.shape[0] == n_rows:
                new_data[k] = arr[mask]
                continue
            if isinstance(arr, list) and len(arr) == n_rows:
                new_data[k] = [arr[i] for i in keep_idx]
                continue
        except Exception:
            pass
        new_data[k] = v
    return new_data, new_fit


def chi_from_fit(fit_result, taus, *, T1=None):
    """Coherence integral chi_N(tau) from a fit result.

    The measured envelope is exp(-Gamma_1 * tau) * exp(-chi_N(tau)) (Bylander
    Eq. 23, after the tau-independent Gamma_p prefactor is absorbed into the
    fitted amplitude). If `T1` is provided, subtracts Gamma_1 * tau =
    tau/(2*T1) from chi_apparent to isolate the pure dephasing contribution.
    Otherwise returns chi_apparent (T2-only estimate that includes T1).

    Uses the envelope tag ('exp' or 'gauss') stored in the fit result.
    """
    p = fit_result['p']
    T2 = float(p[3])
    taus = np.asarray(taus, dtype=float)
    if T2 <= 0 or not np.isfinite(T2):
        return np.full_like(taus, np.nan)
    if fit_result.get('envelope', 'exp') == 'gauss':
        chi = (taus / T2) ** 2
    else:
        chi = np.abs(taus) / T2  # exponential
    if T1 is not None and T1 > 0:
        chi = chi - taus / (2.0 * T1)
        chi = np.maximum(chi, 0.0)  # clip negatives from imperfect T1 estimate
    return chi


# ----- narrow-filter S(omega) inversion ----------------------------------- #

def extract_psd_narrow_filter(
    fit_summary, *, mode='simple', mother_data=None, channel='avgi',
    sweep_param_key='echoes_sweep', T1=None, dwdlambda=1.0, tau_pi=0.0,
    tau_skip_initial=2, env_window=15, env_floor_frac=0.02,
    env_anchor='global_fit',
):
    """Bylander Eq. 26 inversion: extract S(omega') points from CPMG decays.

    mode='simple'   (default; Bylander's "starting point at a single frequency"
                    method, Fig. 2g): one point per N at the characteristic
                    tau where chi_N ~ 1, i.e. tau = T2(N). Honest accounting
                    of what the parametric envelope fit really constrains.

    mode='per_tau'  (legacy / diagnostic): one inversion per (N, tau) using
                    chi_N(tau) read from the global envelope fit. WARNING:
                    for a parametric envelope (exp or gauss) this is
                    degenerate — every point in a given N curve evaluates to
                    the same S value (collapses to a horizontal line on the
                    log-log S(omega) plot). Useful only as a sanity check.

    mode='data_driven' (full Bylander Eq. 24 spirit): pulls chi_N(tau) from
                    the raw oscillating data via a windowed amplitude fit at
                    each tau, NOT from the global envelope. Each tau gives a
                    genuinely independent S(omega') point — the "dot cloud per
                    N" in Bylander Fig. 4. Requires `mother_data` so we can
                    re-read the raw Ramsey waveform.

    Skips N=0 (Ramsey) — it probes near DC and a separate low-freq anchor is
    appropriate (use T2*).

    On the pulse-induced decay Gamma_p: Gamma_p is tau-independent at fixed N
    and is absorbed into the fitted amplitude (global fit) or the envelope
    anchor (data-driven). No two-tau ratio trick is needed for the per-curve
    sweep we do here.

    Parameters
    ----------
    fit_summary : output of fit_cpmg_decays.
    mode : 'simple' | 'per_tau' | 'data_driven'
    mother_data : SweepRunner mother_expt.data — required for mode='data_driven'.
    channel : data channel to use for the envelope extraction in data-driven mode.
    sweep_param_key : key for the N-values array in mother_data (default 'echoes_sweep').
    T1 : float, optional. Subtracts Gamma_1*tau = tau/(2*T1) from chi_apparent.
    dwdlambda : float, sensitivity prefactor (|d omega_01 / d lambda|).
    tau_pi : float, pi-pulse width [us]; passed to the filter function. Set to
             the actual echo gate duration; non-zero is significant once
             omega*tau_pi/2 approaches pi/2 (filter zero-crossing).
    tau_skip_initial : int, skip first few tau samples (small-tau numerical
                       instability + windowed amplitude fit needs a half-window).
    env_window : int (data-driven only), window size for the local amplitude fit.
    env_floor_frac : float (data-driven only), envelope values below
                     `env_floor_frac * env0` are considered noise and dropped.
    env_anchor : 'global_fit' | 'env_max' (data-driven only). 'global_fit' uses
                 the fitted amplitude p[0] as env_0; 'env_max' uses the max
                 windowed amplitude over the early-tau region.

    Returns
    -------
    dict with 1D arrays:
      'omega'    : filter peak frequency [rad/us]
      'freq_MHz' : omega / (2*pi)
      'S'        : extracted S(omega') value
      'N'        : N value for each point
      'tau'      : tau value [us]
      'mode'     : the mode used (for downstream context)
    """
    if mode not in ('simple', 'per_tau', 'data_driven'):
        raise ValueError(
            f"mode={mode!r} not in {{'simple', 'per_tau', 'data_driven'}}")

    N_values = fit_summary['N_values']
    taus_all = np.asarray(fit_summary['taus'])
    if taus_all.ndim == 2:
        taus_all = taus_all[0]
    fits = fit_summary['fits']

    if mode == 'data_driven' and mother_data is None:
        raise ValueError("mode='data_driven' requires mother_data=...")

    omegas, freqs, S_vals, N_per, tau_per = [], [], [], [], []

    for i, N in enumerate(N_values):
        N_int = int(N)
        if N_int < 1:
            continue
        fit_i = fits[i]

        if mode == 'simple':
            # One point per N at tau = T2(N). chi = 1 (exp) or 1 (gauss) at
            # the 1/e time, but Bylander's "characteristic chi at characteristic
            # tau" is chi=1 (exp envelope) or chi=1 (gauss). Use the actual
            # fitted T2 and the fit-implied chi at that tau.
            try:
                T2 = float(fit_i['p'][3])
            except (TypeError, IndexError, KeyError):
                continue
            if not np.isfinite(T2) or T2 <= 0:
                continue
            tau_ref = T2
            chi_ref = chi_from_fit(fit_i, np.array([tau_ref]), T1=T1)[0]
            if not np.isfinite(chi_ref) or chi_ref <= 0:
                continue
            omega_peak, bw = filter_peak(N_int, tau_ref, tau_pi=tau_pi)
            if bw <= 0 or omega_peak <= 0:
                continue
            g_at_peak = float(cpmg_filter_function(
                np.array([omega_peak]), tau_ref, N_int, tau_pi=tau_pi)[0])
            denom = (dwdlambda ** 2) * (tau_ref ** 2) * g_at_peak * bw
            if denom <= 0:
                continue
            omegas.append(omega_peak)
            freqs.append(omega_peak / (2.0 * np.pi))
            S_vals.append(chi_ref / denom)
            N_per.append(N_int)
            tau_per.append(float(tau_ref))
            continue

        taus = taus_all[tau_skip_initial:]

        if mode == 'per_tau':
            chi_arr = chi_from_fit(fit_i, taus, T1=T1)
        else:  # mode == 'data_driven'
            chi_arr = _chi_from_data_driven(
                fit_i, mother_data, i, channel=channel,
                sweep_param_key=sweep_param_key,
                T1=T1, window=env_window, floor_frac=env_floor_frac,
                anchor=env_anchor, tau_skip_initial=tau_skip_initial,
            )

        for tau, chi in zip(taus, chi_arr):
            if not np.isfinite(chi) or tau <= 0 or chi <= 0:
                continue
            omega_peak, bw = filter_peak(N_int, tau, tau_pi=tau_pi)
            if bw <= 0 or omega_peak <= 0:
                continue
            g_at_peak = float(cpmg_filter_function(
                np.array([omega_peak]), tau, N_int, tau_pi=tau_pi)[0])
            denom = (dwdlambda ** 2) * (tau ** 2) * g_at_peak * bw
            if denom <= 0:
                continue
            omegas.append(omega_peak)
            freqs.append(omega_peak / (2.0 * np.pi))
            S_vals.append(chi / denom)
            N_per.append(N_int)
            tau_per.append(float(tau))

    return {
        'omega': np.asarray(omegas),
        'freq_MHz': np.asarray(freqs),
        'S': np.asarray(S_vals),
        'N': np.asarray(N_per),
        'tau': np.asarray(tau_per),
        'mode': mode,
    }


# ----- data-driven envelope (Bylander Eq. 24 spirit) --------------------- #

def extract_envelope_from_data(
    taus, ydata, *, ramsey_freq, offset=0.0, window=15,
):
    """Local windowed amplitude fit at each tau.

    At each tau index, fit  y(t) = A_c*cos(2 pi f t) + A_s*sin(2 pi f t)
    in a window of `window` surrounding samples (f = ramsey_freq is fixed
    from the global fit). Returns envelope |env(tau)| = sqrt(A_c^2 + A_s^2).

    Window edges (first/last window//2 samples) get NaN.

    Parameters
    ----------
    taus, ydata : 1D arrays (same length).
    ramsey_freq : float [MHz], the virtual detuning used in the experiment
                  (read from cfg or the global fit's freq parameter).
    offset : float, y0 from the global fit. Subtracted before the fit.
    window : int (odd preferred), number of samples in the local fit window.

    Returns
    -------
    env : 1D array of same length as taus, with NaN at window edges and
          anywhere the local fit is degenerate.
    """
    taus = np.asarray(taus, dtype=float)
    y = np.asarray(ydata, dtype=float) - float(offset)
    n = len(taus)
    env = np.full(n, np.nan)
    half = max(1, int(window) // 2)
    if n < 2 * half + 2:
        return env
    two_pi_f = 2.0 * np.pi * float(ramsey_freq)
    for i in range(half, n - half):
        sl = slice(i - half, i + half + 1)
        x = taus[sl]
        yw = y[sl]
        C = np.cos(two_pi_f * x)
        S = np.sin(two_pi_f * x)
        M = np.column_stack([C, S])
        try:
            beta, *_ = np.linalg.lstsq(M, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        env[i] = float(np.sqrt(beta[0] ** 2 + beta[1] ** 2))
    return env


def _chi_from_data_driven(
    fit_result, mother_data, n_idx, *, channel='avgi',
    sweep_param_key='echoes_sweep', T1=None, window=15, floor_frac=0.02,
    anchor='global_fit', tau_skip_initial=2,
):
    """Compute chi_N(tau) for a single N from raw oscillating data.

    Pulls the raw avgi/avgq waveform for sweep-index n_idx out of mother_data,
    extracts the local Ramsey amplitude env(tau) via extract_envelope_from_data,
    anchors it (env_0), then returns:
        chi_apparent(tau) = -log(env(tau)/env_0)
        chi_N(tau)        = chi_apparent(tau) - tau/(2 T1)    [if T1 is given]

    Returns array of len(taus[tau_skip_initial:]) with NaN where the envelope
    is below `floor_frac * env_0` (noise floor) or the windowed fit failed.
    """
    p = fit_result.get('p')
    if not isinstance(p, (list, np.ndarray)):
        return np.array([])
    try:
        yscale = float(p[0])
        f_ramsey = float(p[1])
        y0 = float(p[4])
    except (IndexError, TypeError):
        return np.array([])

    taus_raw = np.asarray(mother_data['xpts'])
    taus = taus_raw[n_idx] if taus_raw.ndim == 2 else taus_raw
    y = np.asarray(mother_data[channel])[n_idx]

    env = extract_envelope_from_data(
        taus, y, ramsey_freq=f_ramsey, offset=y0, window=window,
    )

    # Anchor env_0 (envelope amplitude extrapolated to tau ~ 0).
    if anchor == 'global_fit':
        env_0 = abs(yscale)
    elif anchor == 'env_max':
        finite = env[np.isfinite(env)]
        env_0 = float(np.nanmax(finite[:max(1, len(finite) // 5)])) if finite.size else np.nan
    else:
        raise ValueError(f"anchor={anchor!r}")
    if not np.isfinite(env_0) or env_0 <= 0:
        return np.full(len(taus[tau_skip_initial:]), np.nan)

    floor = floor_frac * env_0
    # Slice to match the rest of the pipeline (skips the initial points).
    env_tail = env[tau_skip_initial:]
    tau_tail = taus[tau_skip_initial:]

    chi_apparent = np.where(
        np.isfinite(env_tail) & (env_tail > floor),
        -np.log(np.clip(env_tail, floor, env_0) / env_0),
        np.nan,
    )
    if T1 is not None and T1 > 0:
        chi_apparent = chi_apparent - tau_tail / (2.0 * float(T1))
    return chi_apparent


# ----- plotting helpers --------------------------------------------------- #

def plot_envelope_diagnostic(mother_data, fit_summary, *, channel='avgi',
                             sweep_param_key='echoes_sweep',
                             window=15, ncols=2, figsize_per=(5.5, 3.0)):
    """Per-N panel: raw data, global-fit envelope, and data-driven envelope.

    Useful for sanity-checking whether the data-driven envelope adds new
    information (frequency-dependent S(omega)) or just tracks the global
    fit (essentially white dephasing within the band).
    """
    N_values = np.asarray(mother_data[sweep_param_key])
    xpts_raw = np.asarray(mother_data['xpts'])
    taus = xpts_raw[0] if xpts_raw.ndim == 2 else xpts_raw
    ydata_all = np.asarray(mother_data[channel])
    fits = fit_summary['fits']

    n = len(N_values)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False, sharex=True,
    )
    axes_flat = axes.ravel()

    for ax, N, y, r in zip(axes_flat, N_values, ydata_all, fits):
        p = r.get('p')
        ax.plot(taus, y, 'o-', markersize=2, linewidth=0.6, alpha=0.6,
                label='data')
        if isinstance(p, (list, np.ndarray)):
            try:
                yscale, f_ramsey = float(p[0]), float(p[1])
                T2, y0 = float(p[3]), float(p[4])
            except (IndexError, TypeError):
                ax.set_title(f'N={int(N)}  (fit unusable)')
                ax.grid(alpha=0.3)
                continue

            # Global-fit envelope upper bound
            x_dense = np.linspace(taus[0], taus[-1], 400)
            if r.get('envelope') == 'gauss':
                env_global = fitter.gaussenvfunc(x_dense, y0, yscale, p[5], T2)
            else:
                env_global = fitter.expfunc(x_dense, y0, yscale, p[5], T2)
            ax.plot(x_dense, env_global, '-', color='C1', linewidth=1.4,
                    label='global-fit envelope')
            ax.plot(x_dense, 2 * y0 - env_global, '-', color='C1',
                    linewidth=1.4, alpha=0.5)

            # Data-driven envelope (windowed amplitude fit)
            env_data = extract_envelope_from_data(
                taus, y, ramsey_freq=f_ramsey, offset=y0, window=window)
            mask = np.isfinite(env_data)
            ax.plot(taus[mask], y0 + env_data[mask], 'o', color='C2',
                    markersize=3, label='data-driven envelope')
            ax.plot(taus[mask], y0 - env_data[mask], 'o', color='C2',
                    markersize=3, alpha=0.5)

            ax.set_title(f'N={int(N)}   T2_fit={T2:.2f} us')
        ax.set_ylabel(channel)
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Total free-evolution time $\\tau$ [us]')

    fig.tight_layout()
    return fig


def plot_cpmg_decays(mother_data, fit_summary, *, channel='avgi',
                     sweep_param_key='echoes_sweep', ncols=2,
                     figsize_per=(5.5, 3.0), show_fit=True):
    """One subplot per N showing the raw decay vs tau and the fitted envelope.

    Parameters
    ----------
    mother_data : dict
        SweepRunner mother data ({'xpts', '{param}_sweep', 'avgi'/'avgq'/...}).
    fit_summary : output of fit_cpmg_decays (same channel).
    channel : 'avgi' | 'avgq' | 'amps'
    sweep_param_key : key for the N-values array (default 'echoes_sweep').
    ncols : int, subplot columns.
    figsize_per : (w, h) per subplot.
    show_fit : if True, overlay the fitted decaysin + envelope.
    """
    N_values = np.asarray(mother_data[sweep_param_key])
    xpts_raw = np.asarray(mother_data['xpts'])
    taus = xpts_raw[0] if xpts_raw.ndim == 2 else xpts_raw
    ydata_all = np.asarray(mother_data[channel])  # (n_N, n_tau)
    fits = fit_summary['fits']

    n = len(N_values)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False, sharex=True,
    )
    axes_flat = axes.ravel()

    for ax, N, y, r in zip(axes_flat, N_values, ydata_all, fits):
        ax.plot(taus, y, 'o-', markersize=3, linewidth=0.8, alpha=0.7,
                label='data')
        if show_fit and isinstance(r.get('p'), (list, np.ndarray)):
            p = r['p']
            x_dense = np.linspace(taus[0], taus[-1], 400)
            try:
                if r.get('envelope') == 'gauss':
                    y_fit = fitter.gaussdecaysin(x_dense, *p)
                    env_up = fitter.gaussenvfunc(x_dense, p[4], p[0], p[5], p[3])
                else:
                    y_fit = fitter.decaysin(x_dense, *p)
                    env_up = fitter.expfunc(x_dense, p[4], p[0], p[5], p[3])
                env_lo = 2 * p[4] - env_up
                ax.plot(x_dense, y_fit, '-', linewidth=1, label='fit')
                ax.plot(x_dense, env_up, '--', color='0.4', linewidth=0.8,
                        label='envelope')
                ax.plot(x_dense, env_lo, '--', color='0.4', linewidth=0.8)
                T2_us = float(p[3])
                tag = '$T_2$ Gauss' if r.get('envelope') == 'gauss' else '$T_2$ Exp'
                ax.set_title(f'N={int(N)}   {tag}={T2_us:.2f} us')
            except Exception as exc:
                ax.set_title(f'N={int(N)}   fit failed: {exc!r}')
        else:
            ax.set_title(f'N={int(N)}   (no fit)')
        ax.set_ylabel(channel)
        ax.grid(alpha=0.3)

    # Hide any unused trailing axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel('Total free-evolution time $\\tau$ [us]')

    fig.tight_layout()
    return fig


def plot_T2_vs_N(fit_summary, ax=None, T1=None):
    """Plot fitted T2 versus N (analog of Bylander Fig. 2d)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    Ns = fit_summary['N_values']
    T2 = fit_summary['T2']
    ax.loglog(Ns, T2, 'o-', label=r'$T_2(N)$')
    if T1 is not None:
        ax.axhline(2 * T1, color='k', linestyle='--', alpha=0.5,
                   label=r'$2 T_1$ limit')
    ax.set_xlabel('Number of pi pulses, $N$')
    ax.set_ylabel(r'$T_2$ [$\mu$s]')
    ax.grid(which='both', alpha=0.3)
    ax.legend()
    return fig


def plot_psd(extracted, ax=None, style='overlay', offset_decade=0.5,
             **scatter_kwargs):
    """Plot extracted S(omega) versus frequency.

    style options:
      - 'overlay'   : single axes, all N's overlapping, colored by N
                      (Bylander Fig. 4 look). Best when curves are well
                      separated in frequency.
      - 'shifted'   : single axes, each N shifted vertically by
                      `offset_decade` decades for visual separation; legend
                      labels show the multiplicative offset.
      - 'faceted'   : one subplot per N, stacked vertically with shared
                      frequency axis. Best when curves overlap.

    `ax` is ignored for 'faceted'.
    """
    freqs = extracted['freq_MHz']
    S = extracted['S']
    Ns = extracted['N']
    if len(freqs) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, 'No PSD points extracted', transform=ax.transAxes,
                ha='center', va='center')
        return fig

    unique_Ns = sorted(np.unique(Ns).tolist())

    if style == 'faceted':
        fig, axes = plt.subplots(
            len(unique_Ns), 1, figsize=(8, 2.0 * len(unique_Ns)),
            sharex=True, squeeze=False)
        axes = axes[:, 0]
        for ax_i, N in zip(axes, unique_Ns):
            mask = Ns == N
            ax_i.loglog(freqs[mask], S[mask], 'o-', markersize=4,
                        **scatter_kwargs)
            ax_i.set_ylabel(f'N={int(N)}\n$S(\\omega)$')
            ax_i.grid(which='both', alpha=0.3)
        axes[-1].set_xlabel('Frequency [MHz]')
        fig.suptitle('CPMG noise spectroscopy')
        fig.tight_layout()
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if style == 'shifted':
        offset_factor = 10.0 ** offset_decade
        for idx, N in enumerate(unique_Ns):
            mask = Ns == N
            scale = offset_factor ** idx
            label = f'N={int(N)}' + (f' (x{scale:.2g})' if idx else '')
            ax.loglog(freqs[mask], S[mask] * scale, 'o-', label=label,
                      markersize=4, **scatter_kwargs)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.set_ylabel(r'$S(\omega)$ * offset [a.u.]')
    else:  # 'overlay'
        for N in unique_Ns:
            mask = Ns == N
            ax.loglog(freqs[mask], S[mask], 'o-', label=f'N={int(N)}',
                      markersize=4, **scatter_kwargs)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.set_ylabel(r'$S(\omega)$ [a.u.]')

    ax.set_xlabel('Frequency [MHz]')
    ax.set_title('CPMG noise spectroscopy')
    ax.grid(which='both', alpha=0.3)
    return fig

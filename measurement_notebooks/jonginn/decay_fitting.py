"""Fit amplitude decay while retaining the coherent dynamics of a return trace.

The model is |A(t)|^2 = floor + scale * |A_coherent(t)|^2 *
exp(-2 * (t / tau)^beta). Thus tau is an amplitude 1/e time, in us.
Power fitting avoids division at coherent return-amplitude zeros. A fit without
A_coherent measures apparent decay; coherent redistribution can then mimic loss.
"""

import numpy as np
from scipy.optimize import least_squares


def _aicc(residual, n_parameters):
    n = residual.size
    # A numerical residual floor keeps exact synthetic fits comparable.
    rss = max(float(residual @ residual), n * 1e-14)
    return (
        n * np.log(rss / n) + 2 * n_parameters
        + 2 * n_parameters * (n_parameters + 1) / (n - n_parameters - 1)
    )


def fit_complex_decay(
    times_us, A, coherent_A=None, *, beta=1.0,
    min_delta_aicc=6.0, max_relative_rmse=0.35,
):
    """Return a fit and diagnostics for one complex occupation-return trace.

    Pass aligned one-dimensional arrays with at least eight finite samples and
    strictly increasing, nonnegative times in us. Select any fit window before
    calling; times retain their physical zero (especially important for beta=2).
    beta=1 is exponential decay and beta=2 is Gaussian decay. Both report the
    amplitude 1/e time, although their linewidth interpretations differ.

    ``tau_us`` is NaN unless a decaying model is supported and constrained.
    ``tau_estimate_us`` always retains the optimizer's estimate for inspection.
    ``accepted`` does not establish microscopic irreversible relaxation: it
    describes a phenomenological envelope conditional on the coherent model.
    Inspect the residual and try different windows before interpreting a trend.

    Confidence intervals use the residual variance and local Jacobian. They are
    approximate, conditional on the model, and assume independent equal-variance
    power residuals; shot noise and time correlations are not modeled here.
    """
    t = np.asarray(times_us, dtype=float)
    amplitude = np.asarray(A, dtype=complex)
    if t.ndim != 1 or amplitude.ndim != 1 or amplitude.shape != t.shape:
        raise ValueError("times_us and A must be aligned one-dimensional arrays")
    if t.size < 8:
        raise ValueError("At least eight samples are required for decay fitting")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(amplitude)):
        raise ValueError("times_us and A must contain only finite samples")
    if t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times_us must be nonnegative and strictly increasing")
    if beta not in (1, 2):
        raise ValueError("beta must be 1 (exponential) or 2 (Gaussian)")

    metric = "theory_modulated" if coherent_A is not None else "apparent_power_envelope"
    coherent = np.ones(t.size, dtype=complex) if coherent_A is None else np.asarray(
        coherent_A, dtype=complex
    )
    if coherent.ndim != 1 or coherent.shape != t.shape:
        raise ValueError("coherent_A must have the same one-dimensional shape as A")
    if not np.all(np.isfinite(coherent)):
        raise ValueError("coherent_A must contain only finite samples")
    power = np.abs(amplitude) ** 2
    coherent_power = np.abs(coherent) ** 2
    power_norm = float(np.max(power))
    coherent_norm = float(np.max(coherent_power))
    if not np.isfinite(power_norm) or not np.isfinite(coherent_norm):
        raise ValueError("Squared amplitudes must be finite")
    if coherent_norm == 0:
        raise ValueError("coherent_A has no nonzero amplitude in the fit window")
    if power_norm == 0:
        raise ValueError("A has no nonzero amplitude in the fit window")
    y, c = power / power_norm, coherent_power / coherent_norm
    span = float(t[-1] - t[0])
    tau_min = float(np.min(np.diff(t))) / 4
    tau_max = 100 * float(t[-1])
    log_bounds = np.log([tau_min, tau_max])

    def decay_model(parameters):
        floor, scale, log_tau = parameters
        return floor + scale * c * np.exp(-2 * (t / np.exp(log_tau)) ** beta)

    no_decay = least_squares(
        lambda p: p[0] + p[1] * c - y,
        [0.01, 0.8], bounds=([0, 0], [1, np.inf]),
        ftol=1e-11, xtol=1e-11, gtol=1e-11,
    )
    starts = np.unique(np.clip(
        [0.15 * span, 0.5 * span, span, 3 * span, 10 * span],
        tau_min * 1.01, tau_max / 1.01,
    ))
    solutions = []
    for tau_start in starts:
        solution = least_squares(
            lambda p: decay_model(p) - y,
            [min(float(np.min(y)), 0.05), max(no_decay.x[1], 0.5), np.log(tau_start)],
            bounds=([0, 0, log_bounds[0]], [1, np.inf, log_bounds[1]]),
            max_nfev=3000, ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        solutions.append(solution)
    best = min(solutions, key=lambda item: float(item.fun @ item.fun))
    floor, scale, log_tau = best.x
    tau = float(np.exp(log_tau))
    predicted = decay_model(best.x)
    residual = y - predicted
    rss = float(residual @ residual)
    total_variance = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - rss / total_variance if total_variance > 1e-14 else np.nan
    relative_rmse = float(np.sqrt(np.mean(residual**2) / np.mean(y**2)))
    aicc, no_decay_aicc = _aicc(residual, 3), _aicc(no_decay.fun, 2)
    delta_aicc = float(no_decay_aicc - aicc)
    jac_rank = int(np.linalg.matrix_rank(best.jac))
    log_tau_se = np.nan
    if jac_rank == 3:
        covariance = np.linalg.pinv(best.jac.T @ best.jac) * rss / (t.size - 3)
        log_tau_se = float(np.sqrt(max(covariance[2, 2], 0)))
    ci_low, ci_high = np.nan, np.nan
    if np.isfinite(log_tau_se):
        ci_low, ci_high = np.exp(np.clip(
            log_tau + np.array([-1, 1]) * 1.96 * log_tau_se, -700, 700
        ))
    tau_at_boundary = bool(tau <= 1.02 * tau_min or tau >= tau_max / 1.02)
    poorly_constrained = bool(
        jac_rank < 3 or not np.isfinite(log_tau_se) or log_tau_se > 0.6
        or ci_low <= tau_min or ci_high >= tau_max
    )
    if not best.success or not no_decay.success:
        status = "optimizer_failed"
    elif delta_aicc < min_delta_aicc:
        status = "no_decay_evidence"
    elif tau_at_boundary:
        status = "tau_at_boundary"
    elif poorly_constrained:
        status = "poorly_constrained"
    elif relative_rmse > max_relative_rmse:
        status = "poor_model_fit"
    else:
        status = "accepted"
    accepted = status == "accepted"
    model_power = predicted * power_norm
    return {
        "accepted": accepted, "status": status, "metric": metric,
        "beta": float(beta), "n_points": int(t.size),
        "t_start_us": float(t[0]), "t_stop_us": float(t[-1]),
        "tau_us": tau if accepted else np.nan, "tau_estimate_us": tau,
        "tau_ci_low_us": float(ci_low), "tau_ci_high_us": float(ci_high),
        "tau_log_standard_error": log_tau_se,
        "tau_min_bound_us": tau_min, "tau_max_bound_us": tau_max,
        "tau_at_boundary": tau_at_boundary, "poorly_constrained": poorly_constrained,
        "floor_power": float(floor * power_norm),
        "scale": float(scale * power_norm / coherent_norm),
        "r_squared": float(r_squared), "relative_rmse": relative_rmse,
        "aicc": float(aicc), "no_decay_aicc": float(no_decay_aicc),
        "delta_aicc": delta_aicc,
        "model_power": model_power, "model_amplitude": np.sqrt(model_power),
        "residual_power": power - model_power,
        "no_decay_model_power": (no_decay.x[0] + no_decay.x[1] * c) * power_norm,
    }

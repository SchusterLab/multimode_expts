"""Q6: the device check of 2026-10-25 night: does an idle per Trotter pulse halve g at fixed Kerr?

Reads ``C:/experiments/261025_level_stats/claude_work/idle_check.json`` (written by
``measurement_notebooks/guan/level_stats_idle_check.py``). Per sync setting:
- [1,0,0,0,0]: two damped lines; span = sqrt((4g)^2 + d^2), weight asymmetry = d / span;
- [2,0,0,0,0]: three damped lines; the two spacings against ``fixed_n_hamiltonian`` with K and g free
  (exact, no dof), and with g fixed to the N = 1 value (K free, 1 dof).
Line spacings do not depend on the Stark frame. Writes figure q6_idle_check.png.
Run: pixi run python analysis_notebooks/level_stats/labbook/q6_idle_check.py
"""
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from common import FIGURES
from theory.qsim.central_boson import fixed_n_hamiltonian

RECORD = "C:/experiments/261025_level_stats/claude_work/idle_check.json"


def damped_lines(params, t):
    out = np.zeros_like(t, dtype=complex)
    for k in range(len(params) // 4):
        f, gamma, re, im = params[4 * k:4 * k + 4]
        out += (re + 1j * im) * np.exp((-2j * np.pi * f - gamma) * t)
    return out


def load(path):
    with h5py.File(path, "r") as h5:
        timing = json.loads(h5.attrs["derived_params"])
        A = np.asarray(h5["complex_return"]).ravel()
        cycles = np.asarray(h5["cycles"]).ravel()
    t = cycles * float(timing["floquet_cycle_us"])
    return t, A / A[0], timing


def fit_lines(t, a, count):
    """-> lines sorted by frequency: f, df, gamma, weight (start: the largest zero-padded FFT peaks)."""
    dt = t[1] - t[0]
    freqs = -np.fft.fftshift(np.fft.fftfreq(8 * len(t), dt))  # A ~ exp(-2 pi i f t)
    power = np.abs(np.fft.fftshift(np.fft.fft(a * np.hanning(len(a)), 8 * len(t))))
    peaks = [k for k in range(1, len(power) - 1) if power[k] >= power[k - 1] and power[k] >= power[k + 1]]
    peaks = sorted(peaks, key=lambda k: -power[k])[:count]
    start = np.concatenate([[freqs[k], 0.01, 1 / count, 0.0] for k in peaks])
    fun = lambda p: np.concatenate([(damped_lines(p, t) - a).real, (damped_lines(p, t) - a).imag])
    fit = least_squares(fun, start)
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) * np.sum(fit.fun ** 2) / (len(fit.fun) - len(fit.x))
    order = np.argsort(fit.x[0::4])
    err = np.sqrt(np.diag(cov))
    return dict(f=fit.x[0::4][order], df=err[0::4][order], gamma=fit.x[1::4][order],
                weight=np.abs(fit.x[2::4] + 1j * fit.x[3::4])[order], params=fit.x,
                rms=np.sqrt(np.mean(np.abs(damped_lines(fit.x, t) - a) ** 2)), freqs=freqs, power=power)


def n2_lines(kerr_MHz, g_MHz):
    m = fixed_n_hamiltonian(2, 5, np.zeros(4), [g_MHz] * 4, kerr_MHz)
    w = m.basis_eigenstate_weights[m.fock_index[(2, 0, 0, 0, 0)]]
    return np.sort(m.energies_MHz[w > 1e-6])


def spacing_fit(lines, g_fixed_MHz=None):
    spacings, err = np.diff(lines["f"]), np.sqrt(lines["df"][1:] ** 2 + lines["df"][:-1] ** 2)
    if g_fixed_MHz is None:
        res = least_squares(lambda p: (np.diff(n2_lines(p[0], p[1])) - spacings) / err, [-0.019, 0.02])
    else:
        res = least_squares(lambda p: (np.diff(n2_lines(p[0], g_fixed_MHz)) - spacings) / err, [-0.019])
    cov = np.linalg.pinv(res.jac.T @ res.jac)
    return res.x, np.sqrt(np.diag(cov)), float(np.sum(res.fun ** 2))


if __name__ == "__main__":
    record = json.load(open(RECORD))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    summary = {}
    for col, (sync, run) in enumerate(sorted(record["runs"].items(), key=lambda x: int(x[0]))):
        t1, a1, timing = load(run["trace_files"][0])
        t2, a2, _ = load(run["trace_files"][1])
        one, two = fit_lines(t1, a1, 2), fit_lines(t2, a2, 3)
        span = one["f"][1] - one["f"][0]
        w = one["weight"] / one["weight"].sum()
        d = span * (w[0] - w[1])
        g1 = np.sqrt(span ** 2 - d ** 2) / 4
        g_err = one["df"].max() * np.sqrt(2) / 4
        (k_free, g_free), (k_free_err, g_free_err), _ = spacing_fit(two)
        (k_fixed,), (k_fixed_err,), chi2 = spacing_fit(two, g1)
        g_saved = timing["couplings_MHz"][3]
        summary[sync] = dict(cycle_us=timing["floquet_cycle_us"], g_saved_kHz=1e3 * g_saved,
                             g_N1_kHz=1e3 * g1, g_N1_err_kHz=1e3 * g_err, d_kHz=1e3 * d,
                             K_gfree_kHz=1e3 * k_free, K_gfree_err=1e3 * k_free_err, g_N2_kHz=1e3 * g_free,
                             K_gN1_kHz=1e3 * k_fixed, K_gN1_err=1e3 * k_fixed_err, chi2_gN1=chi2,
                             K_over_g=k_free / g_free, gamma_N1=one["gamma"].tolist(), gamma_N2=two["gamma"].tolist(),
                             rms=(one["rms"], two["rms"]))
        print(f"sync {sync}: cycle {timing['floquet_cycle_us']:.4f} us, g saved {1e3 * g_saved:.3f} kHz")
        print(f"  N=1: lines {np.round(1e3 * one['f'], 2)} kHz, weights {np.round(w, 3)}, g {1e3 * g1:.3f} +/- "
              f"{1e3 * g_err:.3f} kHz (g/g_saved {g1 / g_saved:.3f}), d {1e3 * d:.2f} kHz, decay {np.round(one['gamma'], 4)} /us")
        print(f"  N=2: lines {np.round(1e3 * two['f'], 2)} +/- {np.round(1e3 * two['df'], 2)} kHz, decay {np.round(two['gamma'], 4)}")
        print(f"       K, g free: K {1e3 * k_free:.2f} +/- {1e3 * k_free_err:.2f} kHz, g {1e3 * g_free:.3f} +/- "
              f"{1e3 * g_free_err:.3f} kHz, K/g {k_free / g_free:.3f};  g = N=1 value: K {1e3 * k_fixed:.2f} +/- "
              f"{1e3 * k_fixed_err:.2f} kHz (chi2 {chi2:.1f}, 1 dof)")
        for row, (t, a, lines, n) in enumerate(((t1, a1, one, 1), (t2, a2, two, 2))):
            ax = axes[row, col]
            ax.plot(lines["freqs"] * 1e3, lines["power"] / lines["power"].max(), "k-", lw=1)
            for f in lines["f"]:
                ax.axvline(1e3 * f, color="C3", ls=":")
            ax.set(xlim=(-150, 150), xlabel="frequency (kHz)", ylabel="|FFT| (norm.)",
                   title=f"[{n},0,0,0,0], sync {sync} (cycle {timing['floquet_cycle_us']:.3f} us, "
                         f"window {t.max():.0f} us)")
    fig.suptitle("Idle per Trotter pulse: N = 1 lines give g, N = 2 lines give K/g")
    fig.savefig(FIGURES / "q6_idle_check.png", dpi=120)
    json.dump(summary, open("C:/experiments/261025_level_stats/claude_work/idle_check_summary.json", "w"), indent=1)

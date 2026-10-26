"""Q11: the one-body matrix with an idle per Trotter pulse (sync 34), from the five N = 1 rows.

Rows: [1,0,0,0,0] (idle_check.json, sync 34) and [0,1,0,0,0] .. [0,0,0,0,1] (idle_onebody.json).
Joint fit of all rows: a_b(t) = s_b exp(-(gamma + 2 pi i delta_b) t) sum_k V_bk^2 exp(-2 pi i E_k t), with
(E, V) the eigensystem of the 5 x 5 one-body H: M1 onsite 0, storage onsite eps_s, couplings g_s.
Parameters: eps (4), g (4), delta_b (5; the mean is fixed to 0 so that eps keeps its meaning), gamma,
s_b complex (projected out). If the idle only adds a static onsite offset per mode, this fit is good
(chi2/point ~ 1) and g_s ~ 10.2 kHz; the detuning knob can then cancel eps.
Run: pixi run python analysis_notebooks/level_stats/labbook/q11_onebody.py
"""
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from common import FIGURES

WORK = "C:/experiments/261025_level_stats/claude_work/"


def load(path):
    with h5py.File(path, "r") as h5:
        timing = json.loads(h5.attrs["derived_params"])
        A = np.asarray(h5["complex_return"]).ravel()
        cycles = np.asarray(h5["cycles"]).ravel()
    return cycles * float(timing["floquet_cycle_us"]), A / A[0], timing


def onebody(eps, g):
    H = np.zeros((5, 5))
    H[1:, 1:] = np.diag(eps)
    H[0, 1:] = H[1:, 0] = g
    return np.linalg.eigh(H)


def model_rows(p, t):
    eps, g, delta, gamma = p[:4], p[4:8], np.r_[p[8:12], -np.sum(p[8:12])], p[12]
    E, V = onebody(eps, g)
    rows = (V ** 2) @ np.exp(np.outer(-2j * np.pi * E, t))
    return rows * np.exp((-2j * np.pi * delta[:, None] - gamma) * t[None, :])


def residuals(p, t, a, noise):
    m = model_rows(p, t)
    out = []
    for b in range(len(a)):
        s = np.vdot(m[b], a[b]) / np.vdot(m[b], m[b])  # complex scale per row, projected out
        r = (a[b] - s * m[b]) / (noise[b] / np.sqrt(2))  # noise: rms of the complex sample
        out += [r.real, r.imag]
    return np.concatenate(out)


if __name__ == "__main__":
    first = json.load(open(WORK + "idle_check.json"))["runs"]["34"]["trace_files"][0]
    others = json.load(open(WORK + "idle_onebody.json"))["runs"]["34"]["trace_files"]
    loaded = [load(f) for f in [first] + others]
    t = loaded[0][0]
    a = np.array([x[1] for x in loaded])
    noise = np.array([np.std(np.diff(row)) / np.sqrt(2) for row in a])  # upper estimate per row
    g0 = loaded[0][2]["couplings_MHz"][3]
    best = None
    rng = np.random.default_rng(0)
    for trial in range(40):
        p0 = np.r_[rng.uniform(-0.04, 0.04, 4), np.full(4, g0) * rng.uniform(0.6, 1.2, 4), np.zeros(4), 0.005]
        fit = least_squares(residuals, p0, args=(t, a, noise))
        if best is None or fit.cost < best.cost:
            best = fit
    p = best.x
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    err = np.sqrt(np.diag(cov))
    chi2_per_point = 2 * best.cost / best.fun.size
    print(f"one-body fit, sync 34: chi2/point {chi2_per_point:.2f} (noise per row is an upper estimate)")
    for i, s in enumerate((4, 5, 6, 7)):
        print(f"  S{s}: onsite {1e3 * p[i]:+.2f} +/- {1e3 * err[i]:.2f} kHz, g {1e3 * p[4 + i]:.2f} +/- "
              f"{1e3 * err[4 + i]:.2f} kHz (saved {1e3 * g0:.2f})")
    print(f"  row offsets {np.round(1e3 * np.r_[p[8:12], -np.sum(p[8:12])], 2)} kHz, decay {p[12]:.4f} /us")
    E, V = onebody(p[:4], p[4:8])
    print(f"  one-body levels {np.round(1e3 * E, 2)} kHz")
    json.dump(dict(onsite_kHz=(1e3 * p[:4]).tolist(), onsite_err_kHz=(1e3 * err[:4]).tolist(),
                   g_kHz=(1e3 * p[4:8]).tolist(), g_err_kHz=(1e3 * err[4:8]).tolist(),
                   row_offsets_kHz=(1e3 * np.r_[p[8:12], -np.sum(p[8:12])]).tolist(), decay=p[12],
                   chi2_per_point=chi2_per_point), open(WORK + "idle_onebody_fit.json", "w"), indent=1)

    m = model_rows(p, t)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), layout="constrained")
    labels = ["[1,0,0,0,0]", "[0,1,0,0,0]", "[0,0,1,0,0]", "[0,0,0,1,0]", "[0,0,0,0,1]"]
    freqs = -np.fft.fftshift(np.fft.fftfreq(8 * len(t), t[1] - t[0]))
    for b, ax in enumerate(axes):
        s = np.vdot(m[b], a[b]) / np.vdot(m[b], m[b])
        for row, style, name in ((a[b], "k-", "data"), (s * m[b], "C3--", "one-body fit")):
            spec = np.abs(np.fft.fftshift(np.fft.fft(row * np.hanning(len(t)), 8 * len(t))))
            ax.plot(1e3 * freqs, spec / spec.max(), style, lw=1, label=name)
        ax.set(xlim=(-80, 80), xlabel="frequency (kHz)", title=f"{labels[b]}, sync 34")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"One-body fit with an idle per pulse: onsite {np.round(1e3 * p[:4], 1)} kHz, "
                 f"g {np.round(1e3 * p[4:8], 1)} kHz")
    fig.savefig(FIGURES / "q11_onebody.png", dpi=120)

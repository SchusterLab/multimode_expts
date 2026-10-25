# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The decaying DC background of C_1 and three ways to remove it
#
# C_1(t) = Tr[O(t) O] / D is not connected: its infinite-time (DC) part is 0.6-0.99 of
# C(0) for S7 and about 0.25 of it for the M1 Kerr probe. The AGP kernel removes a
# constant only over an infinite window. Loss makes the constant decay, and a decaying
# constant is not removed even over an infinite window: it adds
# A rate / (2 mu (mu + rate)^2) per unit g^2 (`theory.qsim.agp.damped_agp`). Methods:
#
# - **none**: the kernel integral over the measured window (`agp`).
# - **zero-padded**: the trace decays to 0 and is integrated to infinity (model only).
# - **exponential fit**: fit A e^{-rate t} for t >= 30 us, subtract, integrate
#   (`fitting.qsim.agp.fit_exponential_background`).
# - **envelope**: multiply C_1 by e^{t/T} and integrate (jonginn, T about 200 us), which
#   is `analyze(t1_us=T)` (`agp_t1`); with T = 200 us and with T from the fit.
#
# The model is a toy: the closed-system C_1 at the recorded settings, every term damped
# at one rate (the rate fitted to the measured trace at that W), plus Gaussian noise at
# the measured `C1_err`. It tests the estimators, not the device. Record:
# `docs/log/2026-10-24_agp-dc-background.md`.

# %%
import matplotlib.pyplot as plt
import numpy as np

from experiments.job_paths import data_root
from experiments.qsim.agp import CentralBosonAGPDisorderScan
from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from theory.qsim.agp import damped_agp, damped_correlation, spectral_agp, split_dc

MU_KHZ = [5., 10.]
FIT_START_US = 30.
JONGINN_T_US = 200.
DRAWS = 100
RNG = np.random.default_rng(1)

# %% [markdown]
# ## 1. Data: S7 fine and coarse (10-17..23), and today's Kerr-direction run

# %%
root = data_root() / "agp_k44_first_scan"
manifest = lambda name: CentralBosonAGPDisorderScan.from_manifest(root / "assembled_data" / name)
scans = {
    "S7 fine (122 us)": manifest("261023_001509_CentralBosonAGPDisorderScan.yaml"),
    "S7 coarse (244 us)": CentralBosonAGPDisorderScan.from_children(
        manifest("261023_001448_CentralBosonAGPDisorderScan.yaml").children
        + manifest("261023_120247_CentralBosonAGPDisorderScan.yaml").children),
    # Still running: the W values whose 5 jobs are all done.
    "Kerr (245 us)": CentralBosonAGPDisorderScan.from_job_files(
        sorted((root / "data").glob("JOB-20261024-*_FockPopulationExperiment.h5")), direction=0),
}
for name, scan in scans.items():
    scan.analyze(mu=MU_KHZ)
    print(name, "W =", scan.data["disorder_khz"].tolist())

# %% [markdown]
# ## 2. The measured envelope: A e^{-rate t} fitted for t >= 30 us

# %%
envelopes = {}
for name, scan in scans.items():
    envelopes[name] = [fit_exponential_background(
        part.data["time_us"], part.data["C1_corrected"], part.data["C1_err"], FIT_START_US)
        for part in scan.children]

fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
for name, scan in scans.items():
    lifetimes = [fit["lifetime_us"] for fit in envelopes[name]]
    errors = [np.sqrt(fit["covariance"][1, 1]) / fit["rate_per_us"]**2 for fit in envelopes[name]]
    ax.errorbar(scan.data["disorder_khz"], lifetimes, errors, fmt="o-", ms=4, capsize=2, label=name)
ax.axhline(JONGINN_T_US, color="k", ls="--", lw=1, label="jonginn's T = 200 us")
ax.set(xlabel="Disorder RMS (kHz)", ylabel="fitted 1/rate (us)", title="Envelope of the measured C_1")
ax.legend(fontsize=8)
plt.show()

for name, scan in scans.items():
    print(name)
    for part, fit in zip(scan.children, envelopes[name]):
        dc, _ = split_dc(*part.theory_spectrum())
        print(f"   W = {part.disorder_khz:5.0f} kHz: A = {fit['amplitude']:.3f} "
              f"(closed-system DC {dc:.3f}), 1/rate = {fit['lifetime_us']:.0f} us")

# %% [markdown]
# ## 3. What the envelope compensation does to the traces
#
# Measured C_1 (grey), times e^{t/200 us} (blue), times e^{t/T_fit} (orange), and the
# closed-system C_1 (black dashed). A flat compensated trace at the closed-system DC level
# means the background is removed in shape; the AGP is still sensitive to its rate.

# %%
examples = [("S7 coarse (244 us)", 0), ("S7 coarse (244 us)", 60), ("S7 coarse (244 us)", 150),
            ("Kerr (245 us)", 20), ("Kerr (245 us)", 120)]
fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.1 * len(examples)), sharex=True,
                         constrained_layout=True)
for ax, (name, strength) in zip(axes, examples):
    scan = scans[name]
    index = int(np.argmin(abs(scan.data["disorder_khz"] - strength)))
    part, fit = scan.children[index], envelopes[name][index]
    times, trace = part.data["time_us"], part.data["C1_corrected"]
    ax.plot(times, trace, color="0.6", lw=1, label="measured")
    ax.plot(times, trace * np.exp(times / JONGINN_T_US), lw=1, label="x e^{t/200us}")
    ax.plot(times, trace * np.exp(times * fit["rate_per_us"]), lw=1,
            label=f"x e^{{t/{fit['lifetime_us']:.0f}us}}")
    ax.plot(times, part.data["C1_theory"], "k--", lw=1, label="closed system")
    ax.set_ylabel(f"{name.split(' (')[0]}\nW={part.disorder_khz:g}", fontsize=8)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
axes[-1].set_xlabel("time (us)")
plt.show()


# %% [markdown]
# ## 4. The estimators on the toy model, at the measured grids, noise and envelope rates
#
# Each point is the mean over noise draws of estimate / closed-system infinite-time AGP
# (1 is right); the bars are the spread of one draw.

# %%
def estimators(times, trace, error, cutoff_khz, coupling_mhz, fit_lifetime_us):
    """-> {method: estimate} for one trace (dimensionless g^2 ||A_mu||^2)."""
    scale = (2 * np.pi * coupling_mhz)**2
    plain = scale * agp_kernel_weights(times, cutoff_khz / 1000)
    fit = fit_exponential_background(times, trace, error, FIT_START_US)
    background = fit["amplitude"] * np.exp(-fit["rate_per_us"] * times)
    return {"none": plain @ trace,
            "exponential fit": plain @ (trace - background),
            "envelope 200 us": scale * agp_kernel_weights(times, cutoff_khz / 1000, JONGINN_T_US) @ trace,
            "envelope fitted": scale * agp_kernel_weights(times, cutoff_khz / 1000, fit_lifetime_us) @ trace}


METHODS = ["none", "zero-padded", "exponential fit", "envelope 200 us", "envelope fitted"]
model = {}
for name, scan in scans.items():
    for cutoff in MU_KHZ:
        rows = {method: [] for method in METHODS}
        for part, envelope in zip(scan.children, envelopes[name]):
            differences, weights = part.theory_spectrum()
            g = float(part.data["coupling_mhz"])
            times, error = part.data["time_us"], part.data["C1_err"]
            rate = envelope["rate_per_us"]
            target = spectral_agp(differences, weights, cutoff / 1000, g)
            clean = damped_correlation(differences, weights, times, rate)
            draws = {method: [] for method in METHODS[2:] + ["none"]}
            for _ in range(DRAWS):
                noisy = clean + RNG.normal(0, error)
                fitted = fit_exponential_background(times, noisy, error, FIT_START_US)["lifetime_us"]
                for method, value in estimators(times, noisy, error, cutoff, g, fitted).items():
                    draws[method].append(value / target)
            rows["zero-padded"].append((damped_agp(differences, weights, cutoff / 1000, g, rate) / target, 0.))
            for method, values in draws.items():
                rows[method].append((np.mean(values), np.std(values)))
        model[name, cutoff] = rows

fig, axes = plt.subplots(len(MU_KHZ), len(scans), figsize=(4.3 * len(scans), 3.4 * len(MU_KHZ)),
                         constrained_layout=True, squeeze=False)
for column, (name, scan) in enumerate(scans.items()):
    for row, cutoff in enumerate(MU_KHZ):
        ax = axes[row, column]
        for method in METHODS:
            values = np.asarray(model[name, cutoff][method])
            ax.errorbar(scan.data["disorder_khz"], values[:, 0], values[:, 1], fmt="o-", ms=3,
                        capsize=2, label=method)
        ax.axhline(1, color="k", lw=.8)
        ax.set(yscale="symlog", ylim=(-3, 300), title=f"{name}, mu = {cutoff:g} kHz",
               xlabel="Disorder RMS (kHz)")
axes[0, 0].set_ylabel("estimate / closed-system AGP")
axes[1, 0].set_ylabel("estimate / closed-system AGP")
axes[0, 0].legend(fontsize=7)
plt.show()

# %% [markdown]
# ## 5. The same estimators on the measured traces
#
# Against the closed-system infinite-time AGP (black dashed). The device is not a closed
# system, so agreement is not expected everywhere; the spread between methods shows how
# much the background choice matters.

# %%
fig, axes = plt.subplots(len(MU_KHZ), len(scans), figsize=(4.3 * len(scans), 3.4 * len(MU_KHZ)),
                         constrained_layout=True, squeeze=False)
for column, (name, scan) in enumerate(scans.items()):
    strengths = scan.data["disorder_khz"]
    for row, cutoff in enumerate(MU_KHZ):
        ax = axes[row, column]
        values = {method: [] for method in METHODS if method != "zero-padded"}
        for part, envelope in zip(scan.children, envelopes[name]):
            estimates = estimators(part.data["time_us"], part.data["C1_corrected"], part.data["C1_err"],
                                   cutoff, float(part.data["coupling_mhz"]), envelope["lifetime_us"])
            for method, value in estimates.items():
                values[method].append(value)
        for method, series in values.items():
            ax.plot(strengths, series, "o-", ms=3, label=method)
        dense = np.linspace(0, strengths.max(), 121)
        ax.plot(dense, scan.theory_agp(dense, mu=cutoff)[:, 0], "k--", lw=1, label="closed system")
        ax.set(title=f"{name}, mu = {cutoff:g} kHz", xlabel="Disorder RMS (kHz)", yscale="symlog",
               ylim=(-.05, None))
        ax.set_yscale("symlog", linthresh=1e-3)
axes[0, 0].set_ylabel(r"$g^2\|A_\mu\|^2$")
axes[1, 0].set_ylabel(r"$g^2\|A_\mu\|^2$")
axes[0, 0].legend(fontsize=7)
plt.show()

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
# # Coherence vs Kerr: a rough scaling check
#
# Question: how fast does the return amplitude die, at low, medium and high M1
# self-Kerr, and does it depend on the photons in M1? The answer sets the usable
# time window and so the frequency resolution. Order of magnitude is enough.
#
# Everything here is the library: the data set catalog, `from_manifest`, and the
# coherent theory each Spectrum already computes (`data.spectrum.theory_A`). The
# timing is each file's own `derived_params`; the theory Kerr is each
# realization's recorded `self_kerr_kHz`. Nothing is re-derived in this notebook.
#
# Two measures, both per diagonal trace, no trace dropped:
#
# 1. **Theory-relative** (spectroscopy data): fit `|A(t)| = c |A_theory(t)| exp(-t/T) + floor`.
#    `c` and `floor` are linear, `T` is a 1D scan. A wrong model also lowers the
#    overlap, so `T` here is a lower bound on the coherence time.
# 2. **Model-free** (closed forward/backward pairs, StarkCal): ideally |A| stays
#    1, so `|A(t)| / |A(0)| = exp(-t/T)` is decoherence plus imperfect closing. The
#    September calibration sets reach only 13 us; a run to 100-200 us gives `T`
#    directly.
#
# Background: `docs/log/2026-10-06_issue5-d72-and-loaders.md`. It replaces
# jonginn's `decay_investigation.ipynb` (git keeps it).

# %%
import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.job_paths import data_root
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_saved import saved_parameters

CATALOG = yaml.safe_load(open("../../configs/datasets/mbr_datasets.yaml", encoding="utf-8"))["datasets"]

# Same g (15 kHz), three Kerr values; then the g = 30 kHz pair.
DATA_SETS = ["sep02to04_pairs_K3p6_g15", "sep02_pairs_K20_g15", "sep01_pairs_K44_g15",
             "sep05_pairs_K3p6_g30", "sep07_pairs_K52p3_g29p2"]
T_GRID_US = np.logspace(0.5, 3.5, 300)

# %% [markdown]
# ## 1. Theory-relative decay of the diagonal returns

# %%
def fit_envelope(t, measured, theory):
    """-> T (us) of measured ~ c * theory * exp(-t / T) + floor, by a scan over T."""
    errors = []
    for T in T_GRID_US:
        basis = np.column_stack([theory * np.exp(-t / T), np.ones_like(t)])
        coefficients, residual, *_ = np.linalg.lstsq(basis, measured, rcond=None)
        errors.append(residual[0] if residual.size else np.inf)
    return T_GRID_US[int(np.argmin(errors))]


rows = []  # one per diagonal trace
for name in DATA_SETS:
    ensemble = MBRDisorderEnsembleExperiment.from_manifest(data_root() / CATALOG[name]["manifest"])
    for record, part in zip(ensemble.realizations, ensemble.children):
        kerr_MHz = 1e-3 * float(record["self_kerr_kHz"])
        data = part.analyze(phase_frame="manual_kerr", manual_kerr_MHz=kerr_MHz)
        t = data.spectrum.time_us
        for occupation, measured, theory in zip(data.reconstruction.occupations,
                                                data.reconstruction.A_norm, data.spectrum.theory_A):
            measured, theory = np.abs(measured), np.abs(theory / theory[0])
            rows.append(dict(data_set=name, K_kHz=abs(1e3 * kerr_MHz), n_M1=occupation[0],
                             T_us=fit_envelope(t, measured, theory), t=t, measured=measured,
                             theory=theory))

# %%
print(f"{'data set':26s} {'|K| kHz':>8s}  median T (us) by n_M1 [traces]")
for name in DATA_SETS:
    mine = [r for r in rows if r["data_set"] == name]
    by_n = {n: [r["T_us"] for r in mine if r["n_M1"] == n] for n in sorted({r["n_M1"] for r in mine})}
    print(f"{name:26s} {np.median([r['K_kHz'] for r in mine]):8.1f}  "
          + "  ".join(f"{n}: {np.median(v):5.0f} [{len(v)}]" for n, v in by_n.items()))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
for n in range(4):
    mine = [r for r in rows if r["n_M1"] == n]
    if mine:
        axes[0].scatter([r["K_kHz"] for r in mine], [r["T_us"] for r in mine], s=10, alpha=0.4,
                        label=f"n_M1={n}")
axes[0].set(xlabel="|K| (kHz, recorded)", ylabel="T (us), theory-relative", yscale="log",
            title="Each dot is one diagonal trace")
axes[0].legend()
example = max((r for r in rows if r["n_M1"] == 2), key=lambda r: r["K_kHz"])
axes[1].plot(example["t"], example["measured"], label="measured |A|")
axes[1].plot(example["t"], example["theory"], ":", label="coherent theory")
axes[1].plot(example["t"], example["theory"] * np.exp(-example["t"] / example["T_us"]), "--",
             label=f"theory x exp(-t/{example['T_us']:.0f} us)")
axes[1].set(xlabel="time (us)", ylabel="|A| / |A(0)|", title=f"{example['data_set']}, one n_M1=2 trace")
axes[1].legend()

# %% [markdown]
# ## 2. Model-free decay of the closed pairs (StarkCal)
#
# The calibration set of each data set above. Replace `CLOSED_PAIR_SETS` with the
# manifests of a longer closed-pair run when there is one.

# %%
CLOSED_PAIR_SETS = {name: CATALOG[name]["calibration_manifest"] for name in DATA_SETS[:3]}

fig, axes = plt.subplots(1, len(CLOSED_PAIR_SETS), figsize=(5 * len(CLOSED_PAIR_SETS), 4),
                         constrained_layout=True, sharey=True)
for ax, (name, manifest) in zip(axes, CLOSED_PAIR_SETS.items()):
    calibration = MBRCalibrationSetExperiment.from_manifest(data_root() / manifest)
    cycle_us = saved_parameters(calibration.children).hardware.floquet_cycle_us
    for child in calibration.children:
        data = child.analyze()
        a = np.abs(data["complex_return"])
        ax.plot(np.asarray(data["physical_cycles"]) * cycle_us, a / a[0],
                color=f"C{child.cfg.expt.spectroscopy_occupations[0]}", lw=0.8, alpha=0.6)
    ax.set(title=name, xlabel="time (us)", ylim=(0, 1.3))
axes[0].set_ylabel("|A| / |A(0)|  (color: n_M1)")

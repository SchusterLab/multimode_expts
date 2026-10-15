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
# # The Sep 10 ensemble in pictures: the three views per fitter, and the form factor
#
# guan's three questions per fitter (2026-10-05): (1) are the poles on the peaks of the per-row
# FFT; (2) does the stick diagram after the row sum look like the model's; (3) does the
# histogram of the gap ratio r have the model's shape. Plus the spectral form factor, which
# needs no fitter. Code: `fitting.qsim.poles.views`. The fits are the cached ones of the survey
# (`survey_sep10.py`, folder `SURVEY` below); nothing is fitted here.
#
# The page and its figures go to the lab vault (`VAULT` below), where guan reads and comments.
# Run as a script to rebuild the page: `pixi run python analysis_notebooks/pole_finding/views_sep10.py`.

# %%
import pickle
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.job_paths import data_root
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.mbr_disorder import disorder_direction
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles.registry import data_set
from fitting.qsim.poles.views import (MODEL_COLOR, INK, display_gap_ratio_cdfs, display_gap_ratio_histograms,
                                      display_row_heatmaps, display_sticks, pooled_ratios)

LABEL = "sep10_full_K3p6_g29p2"
SURVEY = data_root() / "260818_qsim_spectroscopy" / "derived_data" / "pole_finding" / "sep10_survey"
VAULT = Path("G:/Shared drives/SLab/Multimode/Lab/guan/qsim_analysis/pole_finding")
FIGURES = VAULT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
PREFIX = "2026-10-05_sep10"
#: The fitters shown, in panel order; T1 is a model fit (its levels are the model's eigenvalues).
FITTERS = ("A", "B", "C", "T1T3", "T1")
MODEL_FIT = ("T1",)
REFERENCE_DRAWS = 1000

spectra = load_spectra(data_set(LABEL), data_root())


def cached_fit(spectrum, name):
    with open(SURVEY / f"fit_{spectrum.label.replace('/', '_')}_{name}.pkl", "rb") as file:
        result = pickle.load(file)[0]
    return result.pole_fit if hasattr(result, "pole_fit") else result


fits = [{name: cached_fit(s, name) for name in FITTERS} for s in spectra]


def save(fig, name):
    fig.savefig(FIGURES / f"{PREFIX}_{name}.png", dpi=110)
    plt.close(fig)
    return f"figures/{PREFIX}_{name}.png"


# %% [markdown]
# ## The model's gap-ratio distribution at this point, from many draws
#
# The same g, K and disorder strength as the measured realizations (onsite vector of norm
# 100 kHz, the model's delta; `docs/log/2026-09-29_pole-finding-t0.md`), random directions.

# %%
p = spectra[0].model_parameters
strength_MHz = float(np.linalg.norm(p["detunings_MHz"]))
reference = []
for seed in range(10_000, 10_000 + REFERENCE_DRAWS):
    detunings = -strength_MHz * disorder_direction(seed, p["mode_count"] - 1)
    model = fixed_n_hamiltonian(p["photon_number"], p["mode_count"], detunings, p["couplings_MHz"], p["kerr_MHz"])
    reference.append(model.energies_MHz)
reference_ratios = pooled_ratios(reference)
print(f"strength {1e3 * strength_MHz:.1f} kHz, g {1e3 * p['couplings_MHz'][0]:.2f} kHz, K {1e3 * p['kerr_MHz']:.2f} kHz;"
      f" {len(reference_ratios)} reference ratios, <r> {reference_ratios.mean():.3f}")

# %% [markdown]
# ## The figures

# %%
heatmaps, sticks = [], []
for s, f in zip(spectra, fits):
    r = s.label.split("/")[-1]
    heatmaps.append(save(display_row_heatmaps(s, f, title=f"realization {r}: per-row |FFT| and the poles each fitter puts there"),
                         f"heatmap_r{r}"))
    sticks.append(save(display_sticks(s, f, title=f"realization {r}: pole weights after the row sum (up) and the model's (down)"),
                       f"sticks_r{r}"))
found = {name: [f[name].frequencies_MHz for f in fits] for name in FITTERS}
model_sets = [s.levels_MHz for s in spectra]
histograms = save(display_gap_ratio_histograms(found, model_sets, reference_ratios, model_fit=MODEL_FIT,
                                               title="gap ratio r, 9 realizations pooled, 10 % of the levels cut at each edge"),
                  "r_histograms")
cdfs = save(display_gap_ratio_cdfs(found, model_sets, reference_ratios, model_fit=MODEL_FIT,
                                   title="the same ratios as cumulative distributions"), "r_cdfs")

with open(SURVEY / "form_factor.pkl", "rb") as file:
    ff = pickle.load(file)
t, D = ff["time_us"], ff["dimension"]
decay = float(np.median([cached_fit(s, "T1").decays_per_us.mean() for s in spectra]))
fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
ax.semilogy(t, ff["sff_model"] / D ** 2, color=MODEL_COLOR, lw=1.6, label="model levels of the 9 draws, no decay")
ax.semilogy(t, ff["sff_model"] / D ** 2 * np.exp(-2 * decay * t), color="#2a78d6", lw=1.6, ls="--",
            label=f"the same with the fitted decay (T2 {1 / decay:.0f} us)")
ax.semilogy(t, ff["sff_normalized"], color=INK, lw=0.9, marker=".", ms=3, label="measured, 9 realizations")
ax.axhline(1 / D, color=INK, lw=0.8, ls=":", label="1/D")
ax.set(xlabel="t (us)", ylabel="<|Tr U(t)|^2> / D^2", ylim=(1e-4, 1.5))
ax.grid(alpha=0.25)
ax.legend(fontsize=8, frameon=False)
form_factor = save(fig, "form_factor")

# the two survey figures of the recap, copied
repo_figures = Path(__file__).resolve().parents[2] / "docs" / "qsim" / "figures" / "pole_finding" \
    if "__file__" in globals() else Path("docs/qsim/figures/pole_finding")
for name in ("sep10_found_per_fitter", "sep10_row_offsets", "sep10_t3_merge_scan", "sep10_form_factor"):
    shutil.copy(repo_figures / f"{name}.png", FIGURES / f"{name}.png")

# %% [markdown]
# ## The page

# %%
def callout(title, image):
    return f"> [!example]- {title}\n> ![]({image})\n\n"      # a blank line: touching callouts merge


commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
model_r = np.mean(pooled_ratios(model_sets))
page = f"""---
date: 2026-10-05
data_set: {LABEL}
generated_by: analysis_notebooks/pole_finding/views_sep10.py at {commit}
---

# Sep 10 ensemble: the three views per fitter

9 disorder realizations x the complete N=3 basis (35 rows), g 29.2 kHz, K {1e3 * p['kerr_MHz']:.2f} kHz,
disorder 100 kHz (norm of the onsite vector), 200 us window, 5.0 kHz FFT bins.
The fits are the survey's of 2026-10-02 (nothing new is fitted here).
Fitters: **A** the current Matrix Pencil; **B** joint pencil; **C** joint pencil and one offset per
row; **T1T3** the sparse fit started from the Hamiltonian fit's offsets; **T1** the Hamiltonian fit
(its levels are the model's eigenvalues, so its r statistics are a model check, not a measurement).
What each method does: [[pole_finding_recap]].

> [!tip] How to comment
> Write a callout anywhere, for example `> [!guan] the rings miss the peak at -120 kHz in row 00012`.
> Claude finds every `[!guan]` callout in this folder at the next session.

## 1. Are the poles on the peaks?

Each panel: the |FFT| of every row (gray, each row scaled to its own maximum), and the fitter's
poles as rings where that row sees them (the pole frequency plus the row's offset), ring area
proportional to the row's amplitude. A ring off a peak is a false or shifted pole. A peak without
a ring is a missed level.

![]({heatmaps[0]})

{"".join(callout(f"Realization {k}", image) for k, image in enumerate(heatmaps) if k)}
## 2. Does the stick diagram look like the model's?

Each panel: one fitter's pole weights after the sum over the 35 rows, up, at the frequency the
rows see; the model's levels, down, in gray. For a complete basis every model weight is 1.

![]({sticks[0]})

{"".join(callout(f"Realization {k}", image) for k, image in enumerate(sticks) if k)}
## 3. Does the r histogram have the model's shape?

All found poles of each fitter, 9 realizations pooled, 10 % of the levels cut at each edge. The
first panel is the model's levels of the same 9 draws; the black step line in every panel is the
model at the same point from {REFERENCE_DRAWS} draws (<r> {reference_ratios.mean():.3f}; the 9 draws give
{model_r:.3f}). Poisson dashed, GOE dotted. The model itself is between the two, so the shape,
not only the first bin, is the test.

![]({histograms})

The same ratios as cumulative distributions:

![]({cdfs})

## 4. The spectral form factor (no fitter)

From the 35 diagonal returns of each realization: $\\langle |\\mathrm{{Tr}}\\,U(t)|^2 \\rangle / D^2$.
It follows the model to about 40 us, then falls 10-20 times below it. The decay explains only part
of that. The row offsets (the next figure) dephase the trace sum.

![]({form_factor})

## 5. Why: the row offsets, and what each fitter finds

The offsets that the Hamiltonian fit (T1, dots) and C (crosses) find per row, in all 9
realizations, and a fit with one shift per photon per mode (5 numbers, dashed):

![](figures/sep10_row_offsets.png)

Levels found per fitter, against what is resolvable (Cramér-Rao) and what the data hold:

![](figures/sep10_found_per_fitter.png)
"""
target = VAULT / f"{PREFIX}_views.md"
if target.exists() and any(tag in target.read_text(encoding="utf-8") for tag in ("[!guan]", "[!claude]")):
    target = VAULT / f"{PREFIX}_views_rebuilt.md"           # never overwrite a page with comments on it
    print("the page has comments; the rebuilt page goes to", target.name)
target.write_text(page, encoding="utf-8")

# the recap of every method, copied with its figures (the repo copy is the record)
recap = (repo_figures.parents[1] / "pole_finding_recap.md").read_text(encoding="utf-8")
recap = recap.replace("figures/pole_finding/", "figures/")
for name in ("sep10_r0_fit_T1T3", "sep10_r0_fit_A", "sep10_r0_fit_C"):
    shutil.copy(repo_figures / f"{name}.png", FIGURES / f"{name}.png")
header = (f"> [!info] A copy of `docs/qsim/pole_finding_recap.md` (repo, commit {commit}), for reading and"
          " comments. The repo copy is the record.\n\n")
(VAULT / "pole_finding_recap.md").write_text(header + recap, encoding="utf-8")
print("written:", sorted(x.name for x in VAULT.iterdir()))

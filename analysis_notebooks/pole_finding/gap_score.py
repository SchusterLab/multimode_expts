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
# # The gap score on the fixed synthetic set (T0)
#
# Plan `docs/qsim/pole_finding_explore.md`, T0. The set and the cached fits are made by
# `tools/pole_fixed_set.py` (`build`, then `fit C`, `fit F`); this notebook only scores them
# (`fitting.qsim.poles.gap_score`). Per case: gaps found (both ends matched to adjacent poles,
# after the one common shift, from the fit's and the true row offsets), small gaps (under half the mean) found, false poles; against the
# Cramér-Rao bound for complex and for real amplitudes: gaps resolvable (at least 4 errors), and
# how many of those were found. P(r < 0.25) pooled over the 5 draws of a condition.

# %%
import pandas as pd

from experiments.job_paths import data_root
from fitting.qsim.poles.fixed_set import load_fits, load_set, set_folder
from fitting.qsim.poles.gap_score import gauge_shift, pooled_small_gap_fraction, score_gaps

pd.set_option("display.width", 250, "display.max_columns", 40)
FOLDER = set_folder(data_root())
cases = load_set(FOLDER / "set.h5")
fits = {name: load_fits(FOLDER / f"fits_{name}.h5", cases) for name in ("C", "F")}
print({name: len(cached) for name, cached in fits.items()}, "of", len(cases), "cases fitted")

# %%
CONDITION = ["point", "rows", "T2_us", "offset_kHz"]

rows, scores = [], {}
for case in cases:
    c = case.condition
    labels = dict(point=c.point, rows=c.partial_rows or 35, T2_us=round(1 / c.decay_per_us),
                  offset_kHz=1e3 * c.offset_sigma_MHz)
    candidates = {"truth": (case.levels_MHz, case.offsets_MHz, 0.)}
    candidates |= {name: (cached[c.key][0].frequencies_MHz, cached[c.key][0].row_offsets_MHz, cached[c.key][1])
                   for name, cached in fits.items() if c.key in cached}
    for name, (found_MHz, offsets_MHz, seconds) in candidates.items():
        result = score_gaps(found_MHz, case.levels_MHz, case.sampling_MHz, case.gap_bounds_MHz,
                            gauge_shift(offsets_MHz, case.offsets_MHz))
        scores.setdefault((name, *labels.values()), []).append(result)
        rows.append(dict(fitter=name, **labels, draw=c.draw, **result.summary(), seconds=seconds))
table = pd.DataFrame(rows)

# %% [markdown]
# Mean over the draws. `resolvable_*` is the bound's count (the same for every fitter);
# `found_of_resolvable_*` is what a fitter got of those.

# %%
columns = ["gaps_found", "small_gaps", "small_found", "false_poles", "resolvable_complex", "resolvable_real",
           "found_of_resolvable_real", "small_resolvable_real", "median_abs_z_real", "seconds"]
table.groupby(CONDITION + ["fitter"])[columns].mean().round(1)

# %% [markdown]
# P(r < 0.25) pooled over the draws: the levels ("true") and each fitter's poles.

# %%
pd.DataFrame([dict(zip(["fitter"] + CONDITION, key), draws=len(group),
                   **dict(zip(["P_true", "P_found"], pooled_small_gap_fraction(group))))
              for key, group in scores.items() if key[0] != "truth"]).round(3)

# %% [markdown]
# All conditions together, per fitter.

# %%
table.groupby("fitter")[columns].mean().round(2)

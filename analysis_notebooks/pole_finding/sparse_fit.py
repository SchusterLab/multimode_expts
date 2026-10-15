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
# # T3: the convex sparse fit on the fixed synthetic set
#
# Plan `docs/qsim/pole_finding_explore.md`, T3; the fitter is `fitting.qsim.poles.sparse_fit`
# (a non-negative group lasso on a fine grid, cvxpy on a working set, then merge / drop of close
# grid poles). Fit caches in the set folder, made by the batch cell at the end (resumes):
#
# - `fits_T3.h5`: T3 with C's row offsets (C's cached fit as the start), all 80 cases;
# - `fits_T3oracle.h5`: T3 with the true row offsets (an upper line, not a fitter);
# - `fits_T3F.h5`: F's add / drop loop (`pursuit.fit`) started from T3's fit;
# - `lambda_curve_T3.jsonl`: the lambda curve (draw 0, 10 rows, the 8 conditions).
#
# Scored as in `gap_score.py` (T0).

# %%
import json

import matplotlib.pyplot as plt
import pandas as pd

from experiments.job_paths import data_root
from fitting.qsim.poles.fixed_set import load_fits, load_set, set_folder
from fitting.qsim.poles.gap_score import gauge_shift, pooled_small_gap_fraction, score_gaps

pd.set_option("display.width", 250, "display.max_columns", 40)
FOLDER = set_folder(data_root())
FITTERS = ("C", "F", "T3", "T3F", "T3oracle")
cases = load_set(FOLDER / "set.h5")
fits = {name: load_fits(FOLDER / f"fits_{name}.h5", cases) for name in FITTERS}
print({name: len(cached) for name, cached in fits.items()}, "of", len(cases), "cases fitted")

# %%
CONDITION = ["point", "rows", "T2_us", "offset_kHz"]

rows, scores = [], {}
for case in cases:
    c = case.condition
    labels = dict(point=c.point, rows=c.partial_rows or 35, T2_us=round(1 / c.decay_per_us),
                  offset_kHz=1e3 * c.offset_sigma_MHz)
    for name, cached in fits.items():
        if c.key not in cached:
            continue
        fit, seconds = cached[c.key]
        result = score_gaps(fit.frequencies_MHz, case.levels_MHz, case.sampling_MHz, case.gap_bounds_MHz,
                            gauge_shift(fit.row_offsets_MHz, case.offsets_MHz))
        scores.setdefault((name, *labels.values()), []).append(result)
        rows.append(dict(fitter=name, **labels, draw=c.draw, **result.summary(), seconds=seconds))
table = pd.DataFrame(rows)
columns = ["poles", "gaps_found", "small_found", "false_poles", "resolvable_real", "found_of_resolvable_real",
           "small_resolvable_real", "median_abs_z_real", "seconds"]

# %% [markdown]
# ## The subset of T0's F run: draw 0, 10 rows
#
# C, F (from C), T3 (C's offsets), T3F (F from T3), and T3 with the true offsets. T3F's time is
# F's part only (T3's comes on top, as C's does for F).

# %%
subset = table[(table.draw == 0) & (table.rows == 10)]
subset.pivot_table(index=CONDITION, columns="fitter", values=["found_of_resolvable_real", "false_poles"]).round(1)

# %%
subset.groupby("fitter")[columns].mean().round(2)

# %% [markdown]
# ## All draws: C, T3 and the oracle (mean over the 5 draws)

# %%
table[table.fitter.isin(["C", "T3", "T3oracle"])].groupby(CONDITION + ["fitter"])[columns].mean().round(1)

# %% [markdown]
# P(r < 0.25) pooled over the draws each fitter has (F, T3F: draw 0 only; T3oracle: draw 0 at 35
# rows), with the true value of the same draws (all 5 draws: 0.184 August, 0.352 September).

# %%
pd.DataFrame([dict(zip(["fitter"] + CONDITION, key), draws=len(group),
                   **dict(zip(["P_true", "P_found"], pooled_small_gap_fraction(group))))
              for key, group in scores.items()]).round(3).pivot_table(
    index=CONDITION, columns="fitter", values=["P_true", "P_found"])

# %% [markdown]
# ## The lambda curve
#
# lambda^2 (the chi^2 a grid point must buy) against false poles and gaps found of the
# resolvable, summed over the 8 conditions (draw 0, 10 rows). `merge_chi2` 25: the default
# (merge / drop of close grid poles after the lasso); 0: the lasso's grid poles alone.

# %%
curve = pd.DataFrame(map(json.loads, (FOLDER / "lambda_curve_T3.jsonl").read_text().splitlines()))
curve["merge_chi2"] = curve.get("merge_chi2", 25.).fillna(25.)
summed = curve.groupby(["merge_chi2", "offsets", "lambda_chi2"])[["found_of_resolvable_real", "false_poles",
                                                                  "poles", "seconds"]].sum()
summed

# %%
fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for (merge, offsets), group in summed.groupby(level=[0, 1]):
    group = group.droplevel([0, 1])
    color = "#2a78d6" if offsets == "C" else "#eb6834"
    style = "-" if merge else ":"
    ax.plot(group.false_poles, group.found_of_resolvable_real, style, color=color, marker="o", lw=2, ms=8,
            label=f"{offsets} offsets, merge {'on' if merge else 'off'}")
    for lam, row in group.iterrows():
        ax.annotate(f"{lam:g}", (row.false_poles, row.found_of_resolvable_real), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color="0.3")
ax.set(xlabel="false poles (sum)", ylabel="gaps found of resolvable (sum)",
       title="T3 on 8 cases: lambda^2 (labels)")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## The batch (off by default; each call stops in 9 minutes, run it again to resume)

# %%
RUN = None      # e.g. ("T3", None, None), ("T3oracle", [0], [10]), ("T3F", [0], [10])

if RUN:
    import time

    from fitting.qsim.poles import pursuit, sparse_fit
    from fitting.qsim.poles.fixed_set import save_fit

    name, draws, row_counts = RUN
    path = FOLDER / f"fits_{name}.h5"
    done = load_fits(path, cases)
    settings = pursuit.PursuitSettings() if name == "T3F" else sparse_fit.SparseFitSettings()
    chosen = sorted((case for case in cases if (draws is None or case.condition.draw in draws)
                     and (row_counts is None or (case.condition.partial_rows or 35) in row_counts)),
                    key=lambda case: case.condition.partial_rows is None)
    deadline = time.time() + 530
    for case in chosen:
        key = case.condition.key
        if key in done or time.time() > deadline:
            continue
        start = time.perf_counter()
        if name == "T3F":
            fit = pursuit.fit(case.A, case.time_us, settings, start=fits["T3"][key][0])
        else:
            fit = sparse_fit.fit(case.A, case.time_us, settings, start=fits["C"][key][0],
                                 offsets_MHz=case.offsets_MHz if name == "T3oracle" else None)
        save_fit(path, case, fit, time.perf_counter() - start, settings)
        print(key, len(fit.frequencies_MHz), "poles", flush=True)

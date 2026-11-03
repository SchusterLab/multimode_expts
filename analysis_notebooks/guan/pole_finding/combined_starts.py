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
# # Combined starts: T3 from T1's offsets, against every fitter (fixed set, draw 0, 10 rows)
#
# T3 (convex sparse fit) is limited by its row offsets (C's); T1 (Hamiltonian fit) finds the
# offsets to 0.02-0.05 kHz on synthetic data. `T1T3` = `sparse_fit.fit(..., start=<T1's fit>)`,
# cached as `fits_T1T3.h5`. Per case: resolvable gaps found / false poles and P(r < 0.25) found.
# Log: `docs/log/2026-09-29_pole-finding-t1t3.md`. T1 itself is model-derived (the synthetic
# data come from the same model): a best case, not a fitter result.

# %%
import time

from experiments.job_paths import data_root
from fitting.qsim.poles import sparse_fit
from fitting.qsim.poles.fixed_set import load_fits, load_set, save_fit, set_folder
from fitting.qsim.poles.gap_score import gauge_shift, score_gaps

FOLDER = set_folder(data_root())
cases = load_set(FOLDER / "set.h5")
subset = [case for case in cases if case.condition.draw == 0 and case.condition.partial_rows == 10]
fits = {name: load_fits(FOLDER / f"fits_{name}.h5", cases) for name in ("C", "F", "T1", "T3", "T3F", "T1F", "T3oracle")}

# %%
settings = sparse_fit.SparseFitSettings()
done = load_fits(FOLDER / "fits_T1T3.h5", cases)
for case in subset:
    key = case.condition.key
    if key not in done:
        start = time.perf_counter()
        result = sparse_fit.fit(case.A, case.time_us, settings, start=fits["T1"][key][0])
        save_fit(FOLDER / "fits_T1T3.h5", case, result, time.perf_counter() - start, settings)
        done[key] = (result, time.perf_counter() - start)
fits["T1T3"] = done

# %%
print("case".ljust(44), "resolvable", *[name.rjust(10) for name in fits], "  P true")
for case in subset:
    key, cells = case.condition.key, []
    for name, cached in fits.items():
        if key not in cached:
            cells.append("-")
            continue
        fit = cached[key][0]
        s = score_gaps(fit.frequencies_MHz, case.levels_MHz, case.sampling_MHz, case.gap_bounds_MHz,
                       gauge_shift(fit.row_offsets_MHz, case.offsets_MHz)).summary()
        cells.append(f"{s['found_of_resolvable_real']}/{s['false_poles']} {s['P_found']:.2f}")
    print(key[:44].ljust(44), str(s["resolvable_real"]).rjust(10), *[cell.rjust(10) for cell in cells], f"  {s['P_true']:.2f}")

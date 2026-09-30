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
# # T1: a fit of the Hamiltonian
#
# Plan `docs/qsim/pole_finding_explore.md`, T1; code `fitting.qsim.poles.hamiltonian_fit`. The
# detunings, couplings, Kerr, one decay and the row offsets fitted to all rows at once; the levels
# are the model's eigenvalues, the amplitudes s_b |<b|lambda>|^2 ("T1") or free and >= 0 ("T1free").
#
# **Caveat (guan): the levels of a model fit carry the model's statistics.** Their P(r < 0.25) is
# not a measurement; T1 is a model check (real data) and a start for a free fit (F from T1: "T1F").
#
# 1. The fixed set (T0): T1 on every case, from the true parameters perturbed by the size of the
#    real model error (detunings and Kerr 1 kHz, couplings 5 %); the recovery; the gap score
#    against C and F. T1free and T1F on the subset F has (draw 0, 10 rows).
# 2. Real data: august_N3 and august_disorder, from the recorded parameters.
#
# The fits are cached next to the set (`fits_T1.h5`, `fits_T1free.h5`, `fits_T1F.h5`) and resume;
# each fitting cell stops after `BUDGET_S` (run the file again to go on). `T1_STAGES` (environment)
# picks the stages when run as a script, e.g. `T1_STAGES=fit,score`.

# %%
import json
import os
import time

import h5py
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from fitting.qsim.mbr_disorder import disorder_direction
from fitting.qsim.poles import hamiltonian_fit as hf
from fitting.qsim.poles import pursuit
from fitting.qsim.poles.fixed_set import DIRECTION_SEED, POINTS, load_fits, load_set, save_fit, set_folder
from fitting.qsim.poles.gap_score import gauge_shift, pooled_small_gap_fraction, score_gaps

pd.set_option("display.width", 250, "display.max_columns", 40)
STAGES = os.environ.get("T1_STAGES", "fit,free,F,score,real").split(",")
BUDGET_S = 480
FOLDER = set_folder(data_root())
cases = load_set(FOLDER / "set.h5")
SUBSET = [case for case in cases if case.condition.draw == 0 and case.condition.partial_rows == 10]
SETTINGS = {"T1": hf.HamiltonianFitSettings(), "T1free": hf.HamiltonianFitSettings(amplitudes="free", starts=4)}
#: The stand-in for the recorded values: the truth off by the real model error (1-2 kHz in levels).
ERROR = dict(detuning_MHz=1e-3, coupling=0.05, kerr_MHz=1e-3)


def truth_of(case):
    return hf.true_parameters(POINTS[case.condition.point], disorder_direction(DIRECTION_SEED + case.condition.draw, 4))


def start_of(case):
    return hf.perturbed(truth_of(case), **ERROR, seed=1000 + case.condition.draw)


def fill_cache(name, todo, run):
    """Fit the cases of ``todo`` not yet in ``fits_<name>.h5`` (``run(case) -> (fit, extra attrs)``),
    each saved when done, until ``BUDGET_S`` is spent."""
    path, begin = FOLDER / f"fits_{name}.h5", time.time()
    done = load_fits(path, cases)
    for case in todo:
        if case.condition.key in done:
            continue
        if time.time() - begin > BUDGET_S:
            print(name, "budget spent; run again")
            return
        t0 = time.time()
        fit, attrs = run(case)
        save_fit(path, case, fit, time.time() - t0, SETTINGS.get(name, pursuit.PursuitSettings()))
        with h5py.File(path, "a") as file:
            file[case.condition.key].attrs.update(attrs)
        print(name, case.condition.key, f"{time.time() - t0:.0f} s", flush=True)
    print(name, "complete")


def run_hamiltonian(name):
    def run(case):
        result = hf.fit_hamiltonian(case.A, case.time_us, case.occupations, start_of(case), SETTINGS[name])
        return result.pole_fit, dict(hamiltonian=json.dumps(dict(
            parameters=result.parameters.model_dump(), errors=result.errors.model_dump(), decay_per_us=result.decay_per_us,
            chi2=result.chi2, dof=result.dof, start_chi2=result.start_chi2.tolist())))
    return run


def hamiltonian_attrs(name):
    """-> {case key: the stored HamiltonianFit summary} of a cache."""
    with h5py.File(FOLDER / f"fits_{name}.h5", "r") as file:
        return {key: json.loads(group.attrs["hamiltonian"]) for key, group in file.items() if "hamiltonian" in group.attrs}


# %% [markdown]
# ## Fit the fixed set (T1 on all 80 cases; T1free and T1F on the subset)

# %%
if "fit" in STAGES:
    fill_cache("T1", cases, run_hamiltonian("T1"))
if "free" in STAGES:
    fill_cache("T1free", SUBSET, run_hamiltonian("T1free"))
if "F" in STAGES:
    T1 = load_fits(FOLDER / "fits_T1.h5", cases)
    fill_cache("T1F", [c for c in SUBSET if c.condition.key in T1],
               lambda case: (pursuit.fit(case.A, case.time_us, pursuit.PursuitSettings(),
                                         start=T1[case.condition.key][0]), {}))

# %% [markdown]
# ## Recovery: fitted minus true parameters, over their errors
#
# Per condition (5 draws): the rms of z = (fit - truth) / error over the 9 parameters (1 if the
# errors are right), the largest |z|, the rms error of detunings, couplings and Kerr (kHz), the
# found T2, the reduced chi^2, and how many of the 12 starts reach the best chi^2 (within 1).

# %%
CONDITION = ["point", "rows", "T2_us", "offset_kHz"]


def labels_of(case):
    c = case.condition
    return dict(point=c.point, rows=c.partial_rows or 35, T2_us=round(1 / c.decay_per_us), offset_kHz=1e3 * c.offset_sigma_MHz)


if "score" in STAGES:
    rows = []
    for name in ("T1", "T1free"):
        stored = hamiltonian_attrs(name)
        for case in cases:
            if case.condition.key not in stored:
                continue
            h = stored[case.condition.key]
            fitted = hf.ModelParameters(**h["parameters"]).vector
            errors, truth = hf.ModelParameters(**h["errors"]).vector, truth_of(case).vector
            z = (fitted - truth) / errors
            chi2 = np.asarray(h["start_chi2"])
            rows.append(dict(fitter=name, **labels_of(case), draw=case.condition.draw, z_rms=np.sqrt(np.mean(z ** 2)),
                             z_max=np.abs(z).max(), detuning_err_kHz=1e3 * np.sqrt(np.mean((fitted - truth)[:4] ** 2)),
                             coupling_err_kHz=1e3 * np.sqrt(np.mean((fitted - truth)[4:8] ** 2)),
                             kerr_err_kHz=1e3 * abs(fitted - truth)[8], T2_found=1 / h["decay_per_us"],
                             reduced_chi2=h["chi2"] / h["dof"], starts_at_best=int(np.sum(chi2 < chi2.min() + 1))))
    recovery = pd.DataFrame(rows)
    if len(recovery):
        print(recovery.groupby(["fitter", "point", "rows", "T2_us", "offset_kHz"])[
            ["z_rms", "z_max", "detuning_err_kHz", "coupling_err_kHz", "kerr_err_kHz", "T2_found", "reduced_chi2",
             "starts_at_best"]].mean().round(2))

# %% [markdown]
# ## The gap score: T1 against C and F (model-derived levels: a check, not the answer)

# %%
if "score" in STAGES:
    fits = {name: load_fits(FOLDER / f"fits_{name}.h5", cases) for name in ("C", "F", "T1", "T1free", "T1F")}
    print({name: len(cached) for name, cached in fits.items()}, "of", len(cases), "cases fitted")
    table, scores = [], {}
    for case in cases:
        for name, cached in fits.items():
            if case.condition.key not in cached:
                continue
            fit, seconds = cached[case.condition.key]
            result = score_gaps(fit.frequencies_MHz, case.levels_MHz, case.sampling_MHz, case.gap_bounds_MHz,
                                gauge_shift(fit.row_offsets_MHz, case.offsets_MHz))
            scores.setdefault((name, *labels_of(case).values()), []).append(result)
            table.append(dict(fitter=name, **labels_of(case), draw=case.condition.draw, **result.summary(), seconds=seconds))
    table = pd.DataFrame(table)
    columns = ["poles", "gaps_found", "resolvable_real", "found_of_resolvable_real", "small_found", "small_resolvable_real",
               "false_poles", "median_abs_z_real", "seconds"]
    print("All cases (5 draws; C, T1):")
    print(table[table.fitter.isin(["C", "T1"])].groupby(CONDITION + ["fitter"])[columns].mean().round(1))
    print("The subset (draw 0, 10 rows):")
    subset = table[(table.draw == 0) & (table.rows == 10)]
    print(subset.pivot_table(index=CONDITION, columns="fitter", values="found_of_resolvable_real").round(0))
    print(subset.pivot_table(index=CONDITION, columns="fitter", values="false_poles").round(0))
    print(subset.pivot_table(index=CONDITION, columns="fitter", values="median_abs_z_real").round(2))
    print(subset.pivot_table(index=CONDITION, columns="fitter", values="seconds").round(0))
    print("P(r < 0.25), pooled over the draws:")
    pooled = pd.DataFrame([dict(zip(["fitter"] + CONDITION, key), draws=len(group),
                                **dict(zip(["P_true", "P_found"], pooled_small_gap_fraction(group))))
                           for key, group in scores.items()])
    print(pooled.pivot_table(index=CONDITION, columns="fitter", values="P_found").round(3).join(
        pooled[pooled.fitter == "C"].set_index(CONDITION)["P_true"].rename("true (5 draws)").round(3)))
    print("P(r < 0.25) on the subset (draw 0 only):")
    subset_P = pd.DataFrame([dict(zip(["fitter"] + CONDITION, key), **dict(zip(
        ["P_true", "P_found"], pooled_small_gap_fraction(group[:1]))))          # cases are in draw order
        for key, group in scores.items() if key[2] == 10])
    print(subset_P.pivot_table(index=CONDITION, columns="fitter", values="P_found").round(3).join(
        subset_P[subset_P.fitter == "C"].set_index(CONDITION)["P_true"].rename("true (draw 0)").round(3)))

# %% [markdown]
# ## Real data: the model check
#
# From the recorded parameters (12 starts, spread 1 kHz / 5 % / 1 kHz). Per spectrum: the fitted
# parameters and their shift from the recorded ones (kHz, with errors), T2, reduced chi^2 (1 if
# the residual is noise; the Fisher errors are not scaled by it), and the in-band residual over the
# out-of-band noise (`diagnosis.residual_excess`, mean over the rows; the band around the recorded
# model's levels). The level shifts: fitted model levels minus recorded model levels.

# %%
if "real" in STAGES:
    from experiments.qsim.pole_data import load_spectra
    from fitting.qsim.poles.diagnosis import residual_excess
    from fitting.qsim.poles.registry import data_set

    spectra = [s for label in ("august_N3", "august_disorder") for s in load_spectra(data_set(label), data_root())]
    real_rows, real_fits = [], {}
    for spectrum in spectra:
        recorded = spectrum.model_parameters
        start = hf.ModelParameters(detunings_MHz=recorded["detunings_MHz"], couplings_MHz=recorded["couplings_MHz"],
                                   kerr_MHz=recorded["kerr_MHz"])
        for name in ("T1", "T1free"):
            t0 = time.time()
            result = hf.fit_hamiltonian(spectrum.A, spectrum.time_us, spectrum.occupations, start, SETTINGS[name])
            real_fits[spectrum.label, name] = result
            shift = 1e3 * (result.parameters.vector - start.vector)
            levels = np.sort(hf.model_of(spectrum.occupations).eigen(result.parameters.vector)[0])
            recorded_levels = np.sort(hf.model_of(spectrum.occupations).eigen(start.vector)[0])
            excess = residual_excess(result.pole_fit, spectrum)
            real_rows.append(dict(spectrum=spectrum.label, fitter=name, seconds=time.time() - t0,
                                  **{f"d{k}": 1e3 * v for k, v in enumerate(result.parameters.detunings_MHz, 1)},
                                  **{f"g{k}": 1e3 * v for k, v in enumerate(result.parameters.couplings_MHz, 1)},
                                  K=1e3 * result.parameters.kerr_MHz,
                                  shift_max_kHz=np.abs(shift).max(), shift_z_max=np.abs(shift / (1e3 * result.errors.vector)).max(),
                                  error_kHz=np.median(1e3 * result.errors.vector), T2_us=1 / result.decay_per_us,
                                  reduced_chi2=result.reduced_chi2, excess_mean=excess.mean(), excess_max=excess.max(),
                                  level_shift_max_kHz=1e3 * np.abs(levels - recorded_levels).max(),
                                  starts_at_best=int(np.sum(result.start_chi2 < result.chi2 + 1))))
            print(spectrum.label, name, "recorded kHz", np.round(1e3 * start.vector, 2))
            print(spectrum.label, name, "fitted   kHz", np.round(1e3 * result.parameters.vector, 2))
            print(spectrum.label, name, "error    kHz", np.round(1e3 * result.errors.vector, 2), flush=True)
    real = pd.DataFrame(real_rows)
    print(real.round(2).to_string())

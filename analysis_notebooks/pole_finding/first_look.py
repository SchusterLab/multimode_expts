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
# # Pole finding: first look at the real data
#
# Fitters A (the current code, campaign settings), B (joint pencil) and E (FFT) on the
# registry's data sets, each in the frame its analysis notebook uses
# (`analysis_notebooks/202609_qsim_migration/mbr.py`, `mbr_disorder.py`). A coarse look
# before benchmarks 3 and 4 (docs/qsim/pole_finding.md); findings in
# `docs/log/2026-09-28_pole-finding-phase1.md`.
#
# Checks that need no exact model: the fit residual, the pole weights (integer on a
# complete basis; at most 1 where the model has no degeneracy on a partial one) and the
# agreement of A and B. Against the model: the match within max(level tolerance, 0.3 kHz)
# (the rebuilt theory differs from the recorded one by up to 0.3 kHz), next to random poles
# of the same count, and P(r < 0.25).

# %%
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.mbr_disorder_ensemble import DEFAULT_MATRIX_PENCIL, MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from fitting.qsim.matrix_pencil import settings_from_options
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles import fft_peaks, joint_pencil, per_row_reconciled
from fitting.qsim.poles.matching import level_tolerances, match_poles
from fitting.qsim.poles.pole_fit import normalize_to_initial_return, poles_from
from fitting.qsim.poles.statistics import small_gap_ratio_fraction
from fitting.qsim.poles.synthetic import distinct_levels

pd.set_option("display.width", 220)
MODEL_ERROR_MHz = 0.3e-3
# A as the 7-1 ensemble analysis ran it (the notebook's 0.5-bin tolerances), cap D = 35.
A_SETTINGS = settings_from_options(dict(
    DEFAULT_MATRIX_PENCIL, mpm_track_frequency_tolerance_bins=0.5, mpm_merge_frequency_tolerance_bins=0.5,
    mpm_dedup_frequency_tolerance_bins=0.5, mpm_requested_max_modes=35))
FITTERS = {"A": (per_row_reconciled.fit, A_SETTINGS),
           "B": (joint_pencil.fit, joint_pencil.JointPencilSettings()),
           "E": (fft_peaks.fit, fft_peaks.FFTPeakSettings())}
AUG_BRANCHES = {(2, 1, 0, 0, 0): 1, (2, 0, 1, 0, 0): 1, (1, 1, 0, 1, 0): 1, (1, 1, 0, 0, 1): 1,
                (1, 0, 1, 1, 0): 1, (1, 0, 1, 0, 1): 1}

# %% [markdown]
# ## The spectra, in their notebooks' frames

# %%
root = data_root()
spectra = {}  # label -> (analyzed data, model Kerr in MHz)
july = MBRSpectrumExperiment.from_manifest(
    root / "260526_qsim_darkmode/assembled_data/260924_163508_MBRSpectrumExperiment.yaml")
spectra["july_N3"] = july.analyze(phase_frame="manual_kerr", manual_kerr_MHz=-19.756e-3, cycle_branches={},
                                  legacy=True, spectrum_method="fft")
august = MBRSpectrumExperiment.from_manifest(
    root / "260526_qsim_darkmode/assembled_data/260924_163516_MBRSpectrumExperiment.yaml")
spectra["august_N3"] = august.analyze(phase_frame="manual_kerr", manual_kerr_MHz=-10.5e-3,
                                      cycle_branches=AUG_BRANCHES, spectrum_method="fft")
august_disorder = MBRDisorderEnsembleExperiment.from_manifest(
    root / "260526_qsim_darkmode/assembled_data/260924_195547_MBRDisorderEnsembleExperiment.yaml")
for record, part in zip(august_disorder.realizations, august_disorder.children):
    spectra[f"august_disorder/{record['realization']}"] = part.analyze(
        phase_frame="manual_kerr", manual_kerr_MHz=1e-3 * float(record["self_kerr_kHz"]),
        cycle_branches={o: AUG_BRANCHES.get(o, 0) for o in part.occupations}, spectrum_method="fft")
disorder_71 = MBRDisorderEnsembleExperiment.from_manifest(
    root / "260818_qsim_spectroscopy/assembled_data/260924_195637_MBRDisorderEnsembleExperiment.yaml")
for record, part in zip(disorder_71.realizations, disorder_71.children):
    kept = [c for c in part.children if c.initial_occupation != (0, 3, 0, 0, 0)]
    part = MBRSpectrumExperiment.from_children(kept, calibration=part.calibration, notes=part.notes)
    spectra[f"disorder_71/{record['realization']}"] = part.analyze(phase_frame="as_acquired", spectrum_method="fft")
    spectra[f"disorder_71/{record['realization']}"].model_kerr_MHz = 1e-3 * float(record["self_kerr_kHz"])
print(len(spectra), "spectra")


# %% [markdown]
# ## Every fitter on every spectrum

# %%
def bin_of(data):
    """-> the FFT bin of a spectrum's time grid, 1 / (N dt)."""
    time_us = np.asarray(data.spectrum.time_us)
    return 1 / (len(time_us) * (time_us[1] - time_us[0]))


def model_levels(data, bin_MHz):
    """-> distinct model levels, multiplicities, and the measured rows' weights on them."""
    kerr_MHz = data.get("model_kerr_MHz", data.spectrum.physical_kerr_MHz)
    model = fixed_n_hamiltonian(3, 5, data.detunings, data.hardware.couplings_MHz, kerr_MHz)
    rows = [model.fock_index[tuple(o)] for o in data.reconstruction.occupations]
    return distinct_levels(model.energies_MHz, model.basis_eigenstate_weights[rows], 1e-3 * bin_MHz)


def fit_residual(fit, A, time_us):
    """-> |a - fit| / |a| over all rows, a = A / A(0)."""
    a = normalize_to_initial_return(A)
    z = poles_from(fit.frequencies_MHz, fit.decays_per_us, time_us[1] - time_us[0])
    return float(np.linalg.norm(a - fit.amplitudes @ z[:, None] ** np.arange(len(time_us))) / np.linalg.norm(a))


rng = np.random.default_rng(0)
rows, fits = [], {}
for label, data in spectra.items():
    A, time_us = np.asarray(data.reconstruction.A), np.asarray(data.spectrum.time_us)
    dt_us = time_us[1] - time_us[0]
    bin_MHz = 1 / (len(time_us) * dt_us)
    levels, multiplicities, weights = model_levels(data, bin_MHz)
    complete = len(A) == 35
    tolerance = np.maximum(level_tolerances(levels, bin_MHz), MODEL_ERROR_MHz)
    for name, (fit, settings) in FITTERS.items():
        f = fits[label, name] = fit(A, time_us, settings)
        match = match_poles(f.frequencies_MHz, levels, tolerance, 1 / dt_us)
        random = np.mean([len(match_poles(rng.uniform(levels.min(), levels.max(), len(f.frequencies_MHz)),
                                          levels, tolerance, 1 / dt_us).level_index) for _ in range(50)])
        heavy = f.weights > (1.5 if complete else 1.2)
        rows.append(dict(
            set=label.split("/")[0], spectrum=label, fitter=name, rows=len(A), samples=len(time_us),
            bin_kHz=1e3 * bin_MHz, levels=len(levels), found=len(f.frequencies_MHz),
            matched=len(match.level_index), random=random, residual=fit_residual(f, A, time_us),
            weight_sum=f.weights.sum(),
            non_integer=np.mean(np.abs(f.weights - np.round(f.weights))) if complete else np.nan,
            small_weight=int(np.sum(np.abs(f.weights) < 0.3)),
            heavy=int(np.sum(heavy)) if not complete else np.nan))
table = pd.DataFrame(rows)
table.groupby(["set", "fitter"])[["samples", "bin_kHz", "levels", "found", "matched", "random", "residual",
                                  "weight_sum", "non_integer", "small_weight", "heavy"]].mean().round(2)


# %% [markdown]
# ## A against B: the poles they share
#
# A pole of B is shared if A has one within a quarter bin (one to one).

# %%
def shared_fraction(first, second, bin_MHz, dt_us):
    match = match_poles(first.frequencies_MHz, second.frequencies_MHz, 0.25 * bin_MHz, 1 / dt_us)
    return len(match.level_index) / max(len(second.frequencies_MHz), 1)


agreement = []
for label, data in spectra.items():
    time_us = np.asarray(data.spectrum.time_us)
    dt_us = time_us[1] - time_us[0]
    bin_MHz = 1 / (len(time_us) * dt_us)
    agreement.append(dict(set=label.split("/")[0], B_poles_found_by_A=shared_fraction(
        fits[label, "A"], fits[label, "B"], bin_MHz, dt_us), A_poles_found_by_B=shared_fraction(
        fits[label, "B"], fits[label, "A"], bin_MHz, dt_us)))
pd.DataFrame(agreement).groupby("set").mean().round(2)

# %% [markdown]
# ## P(r < 0.25), pooled over each ensemble's realizations

# %%
statistic = []
for ensemble in ("august_disorder", "disorder_71"):
    labels = [label for label in spectra if label.startswith(ensemble)]
    true = small_gap_ratio_fraction([model_levels(spectra[l], bin_of(spectra[l]))[0] for l in labels])
    statistic.append(dict(set=ensemble, model=true, **{name: small_gap_ratio_fraction(
        [fits[l, name].frequencies_MHz for l in labels]) for name in FITTERS}))
pd.DataFrame(statistic).round(2)

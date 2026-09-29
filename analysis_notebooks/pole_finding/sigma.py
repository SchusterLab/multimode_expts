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
# # The row-to-row offset sigma on the real data
#
# docs/qsim/pole_finding.md section 6: the floor (Stark-calibration slope errors,
# `calibration_sigma`) and the upper bound (per-row offset against the model,
# `model_offset_sigma`). Benchmark 2 found sigma = 0.05 bin harmless and 0.25 bin clearly
# harmful to B; the answer decides whether fitter C (one offset per row group) is needed.

# %%
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles.offsets import calibration_sigma, model_offset_sigma, model_returns
from fitting.qsim.poles.pole_fit import normalize_to_initial_return

pd.set_option("display.width", 220)
BRANCHES = {(2, 1, 0, 0, 0): 1, (2, 0, 1, 0, 0): 1, (1, 1, 0, 1, 0): 1, (1, 1, 0, 0, 1): 1,
            (1, 0, 1, 1, 0): 1, (1, 0, 1, 0, 1): 1}
root = data_root()
august = MBRSpectrumExperiment.from_manifest(
    root / "260526_qsim_darkmode/assembled_data/260924_163516_MBRSpectrumExperiment.yaml")
august_disorder = MBRDisorderEnsembleExperiment.from_manifest(
    root / "260526_qsim_darkmode/assembled_data/260924_195547_MBRDisorderEnsembleExperiment.yaml")
disorder_71 = MBRDisorderEnsembleExperiment.from_manifest(
    root / "260818_qsim_spectroscopy/assembled_data/260924_195637_MBRDisorderEnsembleExperiment.yaml")

# %% [markdown]
# ## Floor: the Stark-calibration slope errors

# %%
floors = []
for label, calibration, bin_kHz in (("august", august.calibration, 2.30), ("7-1", disorder_71.calibration, 12.22)):
    data = calibration.analyze()
    sigma_kHz = 1e3 * calibration_sigma(data.phase_error, data.hardware.floquet_cycle_us)
    floors.append(dict(calibration=label, occupations=len(sigma_kHz), median_kHz=np.median(sigma_kHz),
                       max_kHz=np.max(sigma_kHz), median_bins=np.median(sigma_kHz) / bin_kHz,
                       max_bins=np.max(sigma_kHz) / bin_kHz))
pd.DataFrame(floors).round(4)

# %% [markdown]
# ## Upper bound: one offset per row against the model

# %%
spectra = {"august_N3": august.analyze(phase_frame="manual_kerr", manual_kerr_MHz=-10.5e-3,
                                       cycle_branches=BRANCHES, spectrum_method="fft")}
for record, part in zip(august_disorder.realizations, august_disorder.children):
    spectra[f"august_disorder/{record['realization']}"] = part.analyze(
        phase_frame="manual_kerr", manual_kerr_MHz=1e-3 * float(record["self_kerr_kHz"]),
        cycle_branches={o: BRANCHES.get(o, 0) for o in part.occupations}, spectrum_method="fft")
for record, part in zip(disorder_71.realizations, disorder_71.children):
    kept = [c for c in part.children if c.initial_occupation != (0, 3, 0, 0, 0)]
    part = MBRSpectrumExperiment.from_children(kept, calibration=part.calibration, notes=part.notes)
    spectra[f"disorder_71/{record['realization']}"] = part.analyze(phase_frame="as_acquired", spectrum_method="fft")
    spectra[f"disorder_71/{record['realization']}"].model_kerr_MHz = 1e-3 * float(record["self_kerr_kHz"])

bounds = []
for label, data in spectra.items():
    time_us = np.asarray(data.spectrum.time_us)
    bin_kHz = 1e3 / (len(time_us) * (time_us[1] - time_us[0]))
    kerr_MHz = data.get("model_kerr_MHz", data.spectrum.physical_kerr_MHz)
    model = fixed_n_hamiltonian(3, 5, data.detunings, data.hardware.couplings_MHz, kerr_MHz)
    rows = [model.fock_index[tuple(o)] for o in data.reconstruction.occupations]
    m = model_returns(model.basis_eigenstate_weights[rows], model.energies_MHz, time_us)
    sigma_MHz, offsets = model_offset_sigma(normalize_to_initial_return(np.asarray(data.reconstruction.A)), m, time_us)
    bounds.append(dict(set=label.split("/")[0], rows=len(offsets), no_match=int(np.sum(np.isnan(offsets))),
                       sigma_kHz=1e3 * sigma_MHz, mean_offset_kHz=1e3 * np.nanmean(offsets),
                       sigma_bins=1e3 * sigma_MHz / bin_kHz))
pd.DataFrame(bounds).groupby("set").agg(spectra=("rows", "size"), rows=("rows", "mean"), no_match=("no_match", "sum"),
                                        sigma_kHz=("sigma_kHz", "mean"), mean_offset_kHz=("mean_offset_kHz", "mean"),
                                        sigma_bins=("sigma_bins", "mean")).round(3)

# %% [markdown]
# ## What this sigma costs B (synthetic, August conditions)
#
# As in `tune_b.py`: 20 draws at the August point, 2 seeds, SNR 100, decay 0.01 per us, with
# row offsets of width 0, the floor (0.5 kHz) and the upper bound (1.06 kHz).

# %%
from fitting.qsim.poles.bench_summaries import scores_table, small_gap_table
from fitting.qsim.poles.benchmarks import FITTERS, run_nonideal_bench
from fitting.qsim.poles.synthetic import Hardware, Nonideal, sample_phase_diagram

august_grid = Hardware(coupling_MHz=8.615e-3, dt_us=1.4509, samples=300, partial_rows=10)
points = sample_phase_diagram([-1.22], [5.80], draws=20, hardware=august_grid, seed=100)
cost = run_nonideal_bench(points, [Nonideal(snr=100, decay_per_us=0.01, offset_sigma_MHz=s) for s in (0., 0.5e-3, 1.06e-3)],
                          seeds=2, hardware=august_grid, fitters={"B": FITTERS["B"]})
gaps = small_gap_table(cost)
(scores_table(cost).groupby("offset_sigma_MHz")[["rank", "matched", "resolved", "false_poles"]].mean()
 .join(gaps.set_index("offset_sigma_MHz")[["found", "true"]].rename(columns=dict(found="I_found", true="I_true")))
 .round(2))

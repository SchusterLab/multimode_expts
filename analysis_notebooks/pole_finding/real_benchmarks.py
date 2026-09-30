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
# # Benchmarks 3 and 4: real data
#
# docs/qsim/pole_finding.md 5.3 and 5.4 on the registry
# (`analysis_notebooks/pole_finding/registry.yaml`; july_N3 left out, it is aliased). Fitter A
# with the settings of the 7-1 analysis notebook (`ANALYSIS_SETTINGS`), fitter B with its
# defaults. `lambda_eff` for the d flag and the re-merging comes from benchmark 2 at each
# disorder set's own conditions (B, SNR 100, decay 0.01 per us, row offsets at the floor of
# `sigma.py`). About 1 min.

# %%
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.poles import joint_pencil, per_row_reconciled
from fitting.qsim.poles.bench_summaries import lambda_eff, resolution_curve
from fitting.qsim.poles.benchmarks import run_nonideal_bench
from fitting.qsim.poles.real_benchmarks import run_model_bench, run_self_consistency_bench
from fitting.qsim.poles.registry import load_registry
from fitting.qsim.poles.synthetic import Hardware, Nonideal, sample_phase_diagram

pd.set_option("display.width", 220)
FITTERS = {"A": (per_row_reconciled.fit, per_row_reconciled.ANALYSIS_SETTINGS),
           "B": (joint_pencil.fit, joint_pencil.JointPencilSettings())}
spectra = [s for d in load_registry() if d.label != "july_N3" for s in load_spectra(d, data_root())]
print(len(spectra), "spectra")

# %% [markdown]
# ## lambda_eff of B at each set's conditions (benchmark 2)

# %%
CONDITIONS = {  # hardware, (K/g, delta/g), row-offset floor
    "diagonal_disorder_71": (Hardware(partial_rows=9), (-2.9, 3.3), 0.89e-3),
    "august_disorder": (Hardware(coupling_MHz=8.615e-3, dt_us=1.4509, samples=300, partial_rows=10), (-1.22, 5.80), 0.50e-3),
}
lambdas = {}
for label, (hardware, (kerr, disorder), sigma) in CONDITIONS.items():
    points = sample_phase_diagram([kerr], [disorder], draws=20, hardware=hardware, seed=100)
    result = run_nonideal_bench(points, [Nonideal(snr=100, decay_per_us=0.01, offset_sigma_MHz=sigma)], seeds=2,
                                hardware=hardware, fitters={"B": FITTERS["B"]})
    lambdas[label] = float(lambda_eff(resolution_curve(result, hardware.bin_MHz)).iloc[0])
lambdas

# %% [markdown]
# ## Benchmark 3: no model
#
# d = lambda_eff / Delta' (flag above 0.25); P(r < 0.25) of the found poles, and after
# re-merging at 1.25 and 1.5 lambda_eff (a steep change: at the resolution limit). Complete
# basis: weights per multiplet against the multiplicities; partial: poles heavier than 1.2.
# august_N3 has delta = 0 (degenerate levels), so its P(r < 0.25) says nothing.

# %%
benchmark_3 = pd.concat([run_self_consistency_bench([s for s in spectra if s.data_set == label], FITTERS,
                                                    lambda_eff_bins=lambdas.get(label, 1.0))
                         for label in dict.fromkeys(s.data_set for s in spectra)])
benchmark_3.groupby(["data_set", "fitter"]).mean(numeric_only=True).round(3)

# %% [markdown]
# ## Benchmark 4: against the model (validation only)
#
# Match within max(level tolerance, 0.3 kHz) against random poles of the same count; the
# median frequency and weight errors of the matched poles; P(r < 0.25) against the model's.

# %%
benchmark_4 = run_model_bench(spectra, FITTERS)
benchmark_4.groupby(["data_set", "fitter"]).mean(numeric_only=True).round(3)

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
# # Why each level is missed
#
# `docs/qsim/pole_finding.md` 5.5 (`fitting.qsim.poles.diagnosis.diagnose_levels`): one row
# per model level, the checks in order: in the measured rows? resolvable at all (Cramér-Rao)?
# held by the data (C's refinement started at the model)? found by the fitter? A missed
# level's cause is its first failed check; "search" if it passed all three.
#
# A, B and C on august_N3, the first august_disorder realization and the first 7-1
# realization. About 5 min, almost all C.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.poles import joint_pencil, joint_refined, per_row_reconciled
from fitting.qsim.poles.diagnosis import cause_counts, diagnose_levels, residual_excess
from fitting.qsim.poles.pole_plots import display_causes
from fitting.qsim.poles.registry import data_set

pd.set_option("display.width", 220)
FITTERS = {"A": (per_row_reconciled.fit, per_row_reconciled.ANALYSIS_SETTINGS),
           "B": (joint_pencil.fit, joint_pencil.JointPencilSettings()),
           "C": (joint_refined.fit, joint_refined.JointRefinedSettings())}
COLUMNS = ["level_kHz", "weight", "separation_kHz", "crb_kHz", "gap_snr", "cluster", "oracle_shift_kHz",
           "oracle_weight", "drop_chi2", *[f"cause_{name}" for name in FITTERS]]

# %%
spectra = [load_spectra(data_set(label), data_root())[0] for label in ("august_N3", "august_disorder", "diagonal_disorder_71")]
results = {}
for s in spectra:
    fits = {name: fit(s.A, s.time_us, settings) for name, (fit, settings) in FITTERS.items()}
    results[s.label] = (s, fits, *diagnose_levels(s, fits))


def show(label):
    s, fits, table, oracle = results[label]
    display_causes(table, list(FITTERS), title=label)
    plt.show()
    excess = pd.DataFrame({name: residual_excess(f, s) for name, f in {**fits, "oracle": oracle}.items()})
    return cause_counts(table, FITTERS), excess.describe().loc[["mean", "max"]].round(2), table[COLUMNS].round(2)


# %% [markdown]
# ## august_N3

# %%
counts, excess, table = show("august_N3")
counts

# %%
excess

# %%
table

# %% [markdown]
# ## august_disorder, first realization

# %%
counts, excess, table = show("august_disorder/0")
counts

# %%
excess

# %%
table

# %% [markdown]
# ## 7-1, first realization

# %%
counts, excess, table = show("diagonal_disorder_71/0")
counts

# %%
excess

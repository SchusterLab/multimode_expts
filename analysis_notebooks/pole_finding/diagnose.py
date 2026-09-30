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
# # Where does each fitter miss a level?
#
# One figure per data set and fitter (`fitting.qsim.poles.pole_plots.display_pole_fit`):
# per-row FFT of the data and of the residual, with the found poles (cyan circles, at the
# row's own offset for C) and the model levels (white ticks, where the row has weight). A
# level that a fitter misses but the data hold is a peak in the residual FFT of its rows;
# a level that is not in the data leaves the residual flat there.
#
# Fitters A, B and C on august_N3, the first august_disorder realization and the first 7-1
# realization. C takes minutes on august_N3.

# %%
import matplotlib.pyplot as plt

from experiments.job_paths import data_root
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.poles import joint_pencil, joint_refined, per_row_reconciled
from fitting.qsim.poles.pole_plots import display_pole_fit
from fitting.qsim.poles.registry import data_set

FITTERS = {"A": (per_row_reconciled.fit, per_row_reconciled.ANALYSIS_SETTINGS),
           "B": (joint_pencil.fit, joint_pencil.JointPencilSettings()),
           "C": (joint_refined.fit, joint_refined.JointRefinedSettings())}

# %%
spectra = [load_spectra(data_set(label), data_root())[0] for label in ("august_N3", "august_disorder", "diagonal_disorder_71")]
fits = {(s.label, name): fit(s.A, s.time_us, settings) for s in spectra for name, (fit, settings) in FITTERS.items()}


def show(spectrum, name):
    display_pole_fit(fits[spectrum.label, name], spectrum.A, spectrum.time_us, spectrum.levels_MHz,
                     spectrum.row_weights, spectrum.occupations, title=f"{spectrum.label}, fitter {name}:")
    plt.show()


# %% [markdown]
# ## august_N3 (complete basis, 150 samples)

# %%
for name in FITTERS:
    show(spectra[0], name)

# %% [markdown]
# ## august_disorder, first realization (10 rows, 300 samples)

# %%
for name in FITTERS:
    show(spectra[1], name)

# %% [markdown]
# ## 7-1, first realization (100 samples)

# %%
for name in FITTERS:
    show(spectra[2], name)

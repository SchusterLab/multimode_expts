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
# # What would a better measurement resolve?
#
# `docs/qsim/pole_finding.md` 5.6 (`fitting.qsim.poles.design`): the Cramér-Rao bound at the
# august_disorder/0 model (`260924_195535_MBRSpectrumExperiment`), for designs we could choose:
# the 10 measured rows or all 35 occupations; what the fit may assume of the amplitudes (free
# complex, real, real with whole-number sums per level); the row offsets (free with a prior, or
# a shift per photon per mode plus a random part); the decay. The noise per sample is the
# measured one (the median of the 10 rows, about 0.075); "equal time" scales it with the number
# of rows, as the shots per row fall. About 20 min.

# %%
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.poles.design import Design, design_summary
from fitting.qsim.poles.diagnosis import row_noise
from fitting.qsim.poles.pole_fit import normalize_to_initial_return
from fitting.qsim.poles.registry import data_set

pd.set_option("display.width", 220)
spectrum = load_spectra(data_set("august_disorder"), data_root())[0]
a = normalize_to_initial_return(spectrum.A)
noise = float(np.median(row_noise(a, spectrum.dt_us, spectrum.levels_MHz, 4.)))
measured = [spectrum.model_occupations.index(o) for o in spectrum.occupations]
ROWS = {"10 measured": measured, "35 complete": list(range(len(spectrum.model_occupations)))}


def summary(rows, time_factor, design):
    """rows: key of ROWS; time_factor: total time over that of the 10-row measurement."""
    index = ROWS[rows]
    weights = spectrum.model_row_weights[index]
    kept = weights.sum(axis=0) >= 0.1
    shots = time_factor * 10 / len(index)                       # shots per row, relative to now
    return dict(rows=rows, time=time_factor, levels=int(kept.sum()), **design.model_dump(include={"amplitudes", "offsets", "offset_prior_MHz", "decay_per_us"}),
                **design_summary(spectrum.time_us, spectrum.levels_MHz[kept], weights[:, kept],
                                 np.array([spectrum.model_occupations[i] for i in index]),
                                 np.full(len(index), noise / np.sqrt(shots)), spectrum.bin_MHz, design))


# %%
cases = []
for decay in (0.005, 0.01):
    for rows, time_factor in (("10 measured", 1), ("10 measured", 3.5), ("35 complete", 1), ("35 complete", 3.5)):
        for amplitudes in ("complex", "real", "real_sums"):
            if amplitudes == "real_sums" and rows != "35 complete":
                continue
            for offsets, prior in (("free", 0.8e-3), ("free", 0.5e-3), ("per_photon", 0.5e-3), ("free", 0.25e-3), ("free", 0.1e-3)):
                cases.append((rows, time_factor, Design(amplitudes=amplitudes, offsets=offsets, offset_prior_MHz=prior, decay_per_us=decay)))
table = pd.DataFrame([summary(*case) for case in cases])
table["offset_prior_kHz"] = 1e3 * table.pop("offset_prior_MHz")

# %%
table.round(2)

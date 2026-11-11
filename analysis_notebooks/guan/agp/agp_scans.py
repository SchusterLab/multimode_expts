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
# # AGP disorder scans: look at new data, analyze it, compare with the older scans
#
# The daily entry point. The data sets are in `analysis_notebooks/datasets/agp_scans.yaml`, one entry per
# scan; a new data set is one new entry. All logic is in `experiments/qsim/agp.py`.
#
# Estimator: `agp_connected` integrates C_1 minus a fitted decaying background
# A e^{-t/tau} (fit from `BACKGROUND_US`; lab book Q5). Its error bars are shot and
# readout-calibration noise only. Dashed lines: the closed system, infinite time, on each scan's
# recorded pattern, couplings and Kerr.

# %%
from experiments.qsim.agp import display_scans, list_scans, load_scan, load_scans

MU = [5., 10., 20.]     # kHz; below about 6-12 kHz the window is too short (lab book Q7)
BACKGROUND_US = 30.

# %% [markdown]
# ## What is in a new folder
# `part_of`: a scan whose jobs are all in another scan of the folder (a subset).

# %%
list_scans(r"C:\experiments\261027AGPlowKerr")

# %% [markdown]
# ## One scan: settings, analysis, plots

# %%
new = load_scan("s7_high_w")    # or a manifest path, also a C:\experiments\... one
print(new.summary())
new.analyze(mu=MU, background_start_us=BACKGROUND_US)
new.display_agp(mu=[10., 20.])
new.display_correlations()
new.display_spectrum(mu=[10., 20.], background_start_us=BACKGROUND_US)

# %% [markdown]
# ## Against the older S7 scans

# %%
scans = load_scans("s7_fine", "s7_coarse", "s7_high_w")
for scan in scans.values():
    scan.analyze(mu=MU, background_start_us=BACKGROUND_US)
display_scans(scans, mu=[10., 20.])
display_scans(scans, mu=[10., 20.], yscale="log")

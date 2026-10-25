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
# # AGP against disorder: measured, T1-compensated and closed-system theory
#
# The example for the AGP analysis. It loads saved disorder scans, integrates the
# measured two-time correlation C_1(t) with the AGP kernel for several cutoffs mu,
# and compares with the closed-system theory at the recorded couplings, detunings
# and Kerr. The notebook only sets parameters and calls the classes; all logic is
# in the library:
#
# | Layer | Where |
# |---|---|
# | job, scan assembly, `analyze()`, plots | `experiments/qsim/agp.py` |
# | measured kernel integral, calibration errors (data -> numbers) | `fitting/qsim/agp.py` |
# | closed-system C_1(t) and AGP (parameters -> predictions) | `theory/qsim/agp.py`, `theory/qsim/central_boson.py` |
#
# Units: mu is a cyclic frequency in kHz. `agp` is dimensionless, g^2 ||A_mu||^2,
# with g the mean recorded M1-storage coupling. The measured value is the kernel
# integral over the measured window only, with no fit and no decay correction;
# `agp_t1` multiplies C_1 by e^{t/T1} of the probed mode (local loss only). The
# theory is for comparison and never changes a measured value:
# `agp_theory_window` is the closed system over the same window as the data,
# `agp_theory` the infinite-time spectral sum.
#
# It reproduces jonginn's 10-23 notebook
# (`agp_storage7_mu1_to10_t1_theory_ENTIERLYCODEX.ipynb`, in the
# `multimode_thermalization_sim` repo) to 1e-12 (`docs/log/2026-10-24_theory-package-agp.md`).
# Off the measurement PC, set `MULTIMODE_DATA_ROOT` to the mounted data tree.

# %%
import matplotlib.pyplot as plt
import numpy as np

from experiments.job_paths import data_root
from experiments.qsim.agp import CentralBosonAGPDisorderScan

# %% [markdown]
# ## 1. The data: N = 3, prepared and read S7 (direction S7)
#
# The 10-17..19 scans, converted on 10-23 (`configs/datasets/fock_population_datasets.yaml`).
# "fine": cycles 0-399 by 1 (122 us), W = 0-150 kHz. "coarse": cycles 0-795 by 5
# (244 us); its W = 0 part is the 10-23 acquisition, a scan of its own.

# %%
ASSEMBLED = data_root() / "agp_k44_first_scan" / "assembled_data"
fine = CentralBosonAGPDisorderScan.from_manifest(
    ASSEMBLED / "261023_001509_CentralBosonAGPDisorderScan.yaml")
coarse = CentralBosonAGPDisorderScan.from_children(
    CentralBosonAGPDisorderScan.from_manifest(
        ASSEMBLED / "261023_001448_CentralBosonAGPDisorderScan.yaml").children
    + CentralBosonAGPDisorderScan.from_manifest(
        ASSEMBLED / "261023_120247_CentralBosonAGPDisorderScan.yaml").children,
    notes="agp_n3_s7_cycles0to795step5 with the 10-23 W = 0 part")
scans = dict(fine=fine, coarse=coarse)

# %% [markdown]
# ## 2. Settings
#
# The cutoffs and the S7 T1 (jonginn's value, from
# `261015_calibration_after_recooldown/data/JOB-20261019-00129_QsimExperiment.h5`).
# `KERR_MHZ = None` uses the jobs' device Kerr for the theory.

# %%
MU_KHZ = [1., 2., 3., 5., 10.]
T1_US, T1_ERR_US = 635.2072768807943, 11.989547372768955
KERR_MHZ = None

for name, scan in scans.items():
    data = scan.analyze(mu=MU_KHZ, t1_us=T1_US, t1_err_us=T1_ERR_US, kerr_mhz=KERR_MHZ)
    part = scan.children[0].data
    print(f"{name}: W = {data['disorder_khz'].tolist()} kHz, window {data['time_us'][-1]:.1f} us, "
          f"g = {1e3 * float(part['coupling_mhz']):.2f} kHz, "
          f"theory Kerr = {1e3 * float(part['theory_kerr_mhz']):.2f} kHz")

# %% [markdown]
# ## 3. AGP against disorder
#
# Circles: measured. Squares: local T1 compensated. Crosses: the closed system over
# the same window. Dashed: the closed system, infinite time. The errors are shot
# noise and the shared readout-calibration error (and the T1 error for the squares);
# not drift, and not the error of the local-T1 approximation.

# %%
for name, scan in scans.items():
    fig, axes = scan.display(mu=[1., 3., 10.])
    fig.suptitle(f"{name}: {scan.data['time_us'][-1]:.0f} us window")
    plt.show()

# %% [markdown]
# ## 4. The time traces against theory
#
# The measured C_1(t) (readout corrected, shot error band) and the closed-system C_1(t).

# %%
fig, axes = fine.display_traces()
plt.show()

# %%
fig, ax = fine.display_correlation(moment=1)
plt.show()

# %% [markdown]
# ## 5. Play: window bias and the regulator
#
# For the closed system, the ratio of the same-window value to the infinite-time
# value shows how much the window truncates at each mu (1 means no bias). It
# needs mu T >> 1; small mu needs long windows.

# %%
fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
for name, scan in scans.items():
    data = scan.data
    row = int(np.argmin(data["disorder_khz"]))  # W = 0
    ax.plot(data["agp_mu"], data["agp_theory_window"][row] / data["agp_theory"][row], "o-",
            label=f"{name}, T = {data['time_us'][-1]:.0f} us")
ax.axhline(1, color="gray", lw=.8)
ax.set(xlabel="mu (kHz)", ylabel="window / infinite time", yscale="log",
       title="Closed system, W = 0")
ax.legend()
plt.show()

# %% [markdown]
# The theory with another Kerr: the measured values do not change.

# %%
other = fine.analyze(mu=MU_KHZ, t1_us=T1_US, t1_err_us=T1_ERR_US, kerr_mhz=-0.044)
fig, axes = fine.display(mu=[3.])
axes[0].set_title("mu = 3 kHz, K = -44 kHz")
plt.show()

# %% [markdown]
# ## 6. Save (optional)
#
# `save()` writes a new scan layer (manifest + small HDF5 with the AGP maps) beside
# the data; the parts and jobs are not changed. Off by default.

# %%
SAVE = False
if SAVE:
    fine.analyze(mu=MU_KHZ, t1_us=T1_US, t1_err_us=T1_ERR_US, kerr_mhz=KERR_MHZ)
    print(fine.save(notes="AGP mu = 1-10 kHz, local T1 S7, theory at the device Kerr"))

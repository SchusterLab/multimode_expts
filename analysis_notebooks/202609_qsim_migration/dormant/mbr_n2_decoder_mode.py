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
# # N=2 spectroscopy from saved job files (decoder-mode data)
#
# **DORMANT.** Moved from `analysis_notebooks/202609_qsim_migration/mbr.py` on 2026-09-26 (MBR
# redesign step 8C), without changes except imports. Why: `docs/qsim/mbr_step8_plan.md`,
# decision 3 (guan): all 40 July N=2 files were taken in the removed 'decoder' phase-correction
# mode (mode unset, which meant 'decoder'; nonzero `decoder_phase_matrix`, applied per pulse at
# acquisition). The file list is `n2_spectroscopy_files.yml` beside this notebook. Not
# maintained; may break when live code changes. If it breaks, add a note here and do not fix it.

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.qsim.deprecated import mbr_n2_spectroscopy as n2

NOTEBOOK_DIR = Path.cwd()

# %% [markdown]
# # 1. N=2 Hamiltonian spectroscopy from saved jobs
#
# Each file already contains both preparation phases
# $\theta=(0,180)^\circ$. The complex return is reconstructed as
#
# $$
# Q(\phi)=P_e(0,\phi)-P_e(180^\circ,\phi),\qquad
# A_{\mathbf n}=Q(0)+iQ(90^\circ).
# $$
#
# The decoder phase matrix saved in each HDF5 file was already applied during
# acquisition and is not applied again here. The last cell applies only the
# overall occupation-dependent shifts entered explicitly below.

# %%
# Acquisition constants for this dataset. One ordered Floquet cycle
# represents `hamiltonian_dt_us` of logical Hamiltonian time.
hamiltonian_dt_us = 0.08
physical_cycle_us = 0.4130059523809524
floquet_couplings_MHz = np.array([
    0.078125, 0.078125, 0.078125, 0.078125,
])
zero_padding = 8
energy_limit_MHz = 1.25

spectroscopy_fnames = n2.load_file_catalog(
    NOTEBOOK_DIR / "n2_spectroscopy_files.yml"
)

# %% [markdown]
# ## Reconstruct the complex return
#
# Files are placed by their saved analyzer phase rather than by list position.
# Only the reconstructed traces and saved acquisition metadata are retained in
# memory; the large shot arrays in each HDF5 file are released immediately.

# %%
# `kerr_override_MHz` reproduces a hard-coded line in cell 170, which read the
# saved M1 Kerr and then overwrote it with -10e-3. Pass None to use the value
# actually stored in the files.
n2_record = n2.reconstruct_complex_return(
    spectroscopy_fnames,
    kerr_override_MHz=-10e-3,
)
A_raw = n2_record["A_raw"]
occupation_strings = n2_record["occupation_strings"]
cycles = n2_record["cycles"]
N = n2_record["N"]
mode_labels = n2_record["mode_labels"]
swap_stors = n2_record["swap_stors"]
physical_kerr_MHz = n2_record["physical_kerr_MHz"]

print("N =", N)
print("A_raw shape =", A_raw.shape)
for occupation, amplitude in zip(occupation_strings, A_raw):
    print(occupation, "A(0) =", amplitude[0])

# %% [markdown]
# ## Manual occupation-resolved rigid-shift correction and DOS
#
# This does not infer a correction from the M1 row and does not fit a
# measured trace to theory. Enter one overall spectral shift for every
# occupation string in `occupation_energy_shift_MHz` below.
#
# The value is `desired peak position - measured peak position`, in MHz. A
# positive value moves the whole spectrum to the right and a negative value
# moves it to the left. The correction changes only the phase ramp of
# $A_{\mathbf n}(t)$; it does not fit, stretch, or rescale the spectrum.
#
# The target Hamiltonian uses the signed Kerr stored in the HDF5
# configuration, as in the preceding saved-job analysis cell.

# %%
n2_grid = n2.fft_grid_and_raw_dos(
    cycles=cycles,
    A_raw=A_raw,
    occupation_strings=occupation_strings,
    hamiltonian_dt_us=hamiltonian_dt_us,
    zero_padding=zero_padding,
)

n2_theory = n2.build_fixed_n_theory(
    N=N,
    mode_labels=mode_labels,
    swap_stors=swap_stors,
    floquet_couplings_MHz=floquet_couplings_MHz,
    physical_kerr_MHz=physical_kerr_MHz,
    physical_cycle_us=physical_cycle_us,
    hamiltonian_dt_us=hamiltonian_dt_us,
    occupation_strings=occupation_strings,
    grid=n2_grid,
)

# %%
# The one genuinely hand-entered input of this workspace. Example: if a
# measured peak sits at +0.45 MHz and should be at 0 MHz, enter -0.45.
one_fock = -0.45
two_fock = -2.17

occupation_energy_shift_MHz = {
    (2, 0, 0, 0, 0): two_fock,
    (1, 1, 0, 0, 0): one_fock,
    (1, 0, 1, 0, 0): one_fock,
    (1, 0, 0, 1, 0): one_fock,
    (1, 0, 0, 0, 1): one_fock,
    (0, 2, 0, 0, 0): two_fock,
    (0, 1, 1, 0, 0): one_fock,
    (0, 1, 0, 1, 0): one_fock,
    (0, 1, 0, 0, 1): one_fock,
    (0, 0, 2, 0, 0): two_fock,
    (0, 0, 1, 1, 0): one_fock,
    (0, 0, 1, 0, 1): one_fock,
    (0, 0, 0, 2, 0): two_fock,
    (0, 0, 0, 1, 1): one_fock,
    (0, 0, 0, 0, 2): two_fock,
}

n2_shifted = n2.apply_rigid_shifts(
    A_raw=A_raw,
    occupation_strings=occupation_strings,
    occupation_energy_shift_MHz=occupation_energy_shift_MHz,
    hamiltonian_dt_us=hamiltonian_dt_us,
    grid=n2_grid,
    theory=n2_theory,
)

# %%
n2.plot_shifted_spectra(
    occupation_strings=occupation_strings,
    mode_labels=mode_labels,
    N=N,
    cycles=cycles,
    energy_limit_MHz=energy_limit_MHz,
    grid=n2_grid,
    theory=n2_theory,
    shifted=n2_shifted,
    physical_kerr_MHz=physical_kerr_MHz,
)


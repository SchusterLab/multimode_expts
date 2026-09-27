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
# # Report replots
#
# **DORMANT.** Moved from `analysis_notebooks/202609_qsim_migration/mbr.py` on 2026-09-26 (MBR
# redesign step 8C), without changes except imports. Why: `docs/qsim/mbr_step8_plan.md`,
# decision 3 (guan): report figures for one report. Not maintained; may break when live code changes. If it breaks,
# add a note here and do not fix it.

# %%
import numpy as np
import matplotlib.pyplot as plt

from experiments.job_paths import data_root
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.deprecated import mbr_replot as replot


# %% [markdown]
# # 4. Report replots
#
# Source cells 205-211. The `replot_*` settings that cell 206 kept at notebook
# scope are now one `ReplotConfig`; its defaults are those values, so building
# it with no arguments reproduces the original figures.
#
# The source also had a sector keyed 4. Its jobs are a second complete N=1
# set (under the Floquet config of the July N=2 set), not N=4; the check that
# would have caught that was commented out. Dropped for now; `analyze_sector`
# checks the photon number again.

# %%
replot_config = replot.ReplotConfig(
    manual_kerr_MHz=-19.756e-3,
    cycle_branches={1: {}, 2: {}, 3: {}},
    fft_window='raw',
    zero_padding=1,
    overview_figsize=(15, 12.3),
    save_dpi=300,
    legend_fontsize=9,
    suptitle_fontsize=14,
    overview_legend_ncols=3,
)

# Dataset choice: one saved spectrum per photon number (July 2026 sets,
# tests/data/mbr_datasets.json), with its calibration set linked.
replot_manifests = {
    1: data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_163501_MBRSpectrumExperiment.yaml",
    2: data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_163502_MBRSpectrumExperiment.yaml",
    3: data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_163508_MBRSpectrumExperiment.yaml",
}

# N=2 has one occupation on a different time grid, so only its FFT rows can
# join the report spectrum -- not its time traces.
replot_N2_supplement_manifest = data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_163503_MBRSpectrumExperiment.yaml"
replot_N2_supplement_occupation = (0, 0, 0, 0, 2)

# %%
replot_runs = replot.load_and_analyze_sectors(
    config=replot_config,
    manifests=replot_manifests,
    n2_supplement_manifest=replot_N2_supplement_manifest,
    n2_supplement_occupation=replot_N2_supplement_occupation,
)

# %% [markdown]
# ### Export one report subplot as its own figure
#
# `panel_names_by_kind` was cell 210's own settings block and stays here.

# %%
replot_panel_names_by_kind = {
    'mpm': (
        'measured_map',
        'theory_map',
        'measured_dos',
        'theory_dos',
    ),
    'fft': (
        'measured_map',
        'theory_map',
        'measured_dos',
        'theory_dos',
        'local_dos',
    ),
}

replot_single_panel_requests = [
    (3, 'mpm', 'measured_map'),
]

replot_single_panel_figures = {}
for replot_N, replot_plot_kind, replot_panel in replot_single_panel_requests:
    replot_single_panel_figures[
        (replot_N, replot_plot_kind, replot_panel)
    ] = replot.plot_single_spectroscopy_panel(
        replot_N,
        replot_plot_kind,
        replot_panel,
        runs=replot_runs,
        panel_names_by_kind=replot_panel_names_by_kind,
        figsize=replot_config.single_panel_figsize,
        figure_dpi=replot_config.figure_dpi,
        legend_fontsize=replot_config.legend_fontsize,
        save_dpi=replot_config.save_dpi,
    )
plt.show()

# %% [markdown]
# ### Time traces for the report
#
# Edit only this list to choose which traces are shown. The N=2 supplemental
# occupation is valid here despite its different grid.

# %%
replot_trace_requests = [
    (1, (1, 0, 0, 0, 0)),
    # (2, (2, 0, 0, 0, 0)),
    # (3, (3, 0, 0, 0, 0)),
    # (2, (0, 0, 0, 0, 2)),
]
replot_normalize_traces = True

replot_time_trace_figure = replot.plot_time_traces(
    replot_trace_requests,
    runs=replot_runs,
    normalized=replot_normalize_traces,
    figsize=replot_config.trace_figsize,
    figure_dpi=replot_config.figure_dpi,
    legend_fontsize=replot_config.legend_fontsize,
    title_fontsize=replot_config.suptitle_fontsize,
    save_dpi=replot_config.save_dpi,
)
plt.show()


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
# # Pole finding for MBR spectra: benchmark report
#
# The design is `docs/qsim/pole_finding.md`; this report follows its sections. Run headless
# with `pixi run pole-report` (`tools/pole_report.py`), which sets the run below and writes
# this notebook as HTML next to its results. Run interactively, it uses the defaults.

# %%
import json
import os
from pathlib import Path

import numpy as np
from IPython.display import Markdown, display

from fitting.qsim.poles.bench_io import method_markdown, save_results
from fitting.qsim.poles.bench_plots import plot_ideal_errors, plot_resolution, plot_small_gap
from fitting.qsim.poles.bench_summaries import lambda_eff, resolution_curve, scores_table, small_gap_table
from fitting.qsim.poles.benchmarks import FITTERS, run_ideal_bench, run_nonideal_bench
from fitting.qsim.poles.matching import display_match, level_tolerances, match_poles
from fitting.qsim.poles.synthetic import Hardware, Nonideal, sample_phase_diagram, synthetic_returns

run = json.loads(os.environ.get("POLE_REPORT_RUN", "{}"))
fitters = {name: FITTERS[name] for name in run.get("fitters", ["A", "B", "E"])}
small = run.get("size", "small") == "small"
output_dir = Path(run["output_dir"]) if "output_dir" in run else None
hardware = Hardware()
print("fitters:", list(fitters), "| size:", run.get("size", "small"), "| output:", output_dir)

# %% [markdown]
# ## Method
#
# Each fitter as it ran: its module docstring, its `fit`, and its settings.

# %%
display(Markdown(method_markdown(fitters)))

# %% [markdown]
# ## Synthetic cases
#
# The phase diagram of Kerr `K / g` and disorder `delta / g`, a few disorder draws per point,
# on the measured grid of the 7-1 campaign (100 samples at 0.818 us; one bin 12.2 kHz).

# %%
kerr_over_g = [0., -1., -3., -10.] if small else np.linspace(0., -10., 6)
disorder_over_g = [0., 0.3, 1., 3.3, 10.] if small else np.geomspace(0.1, 10., 9)
points = sample_phase_diagram(kerr_over_g, disorder_over_g, draws=2 if small else 5)
print(len(points), "points")

# %% [markdown]
# ## Benchmark 1: ideal data
#
# No noise, no offsets. On the measured grid (100 samples) most of the plane is ill conditioned:
# 35 levels in about 25 bins cannot all be told apart even on clean data. The same plane on
# 400 samples checks the code itself; there fitter B must reach 1e-6 bin wherever the
# conditioning is at least 1e-2 (`tests/test_pole_bench_ideal.py`).

# %%
ideal = {samples: run_ideal_bench(points, Hardware(samples=samples), fitters) for samples in (100, 400)}
for samples, result in ideal.items():
    plot_ideal_errors(scores_table(result)).axes[0].set_title(f"Benchmark 1, {samples} samples")

# %%
for samples, result in ideal.items():
    table = scores_table(result)
    display(Markdown(f"**{samples} samples**"))
    display(table.groupby(["fitter", "disorder_over_g"])[["levels", "matched", "resolved", "false_poles"]]
            .mean().unstack("fitter").round(1))

# %% [markdown]
# One case at the campaign's own point (`K / g = -2.9`, `delta / g = 3.3`), noise-free, as
# each fitter sees it.

# %%
reference = synthetic_returns(sample_phase_diagram([-2.9], [3.3], 1)[0], hardware)
for name, (fit, settings) in fitters.items():
    found = fit(reference.A, reference.time_us, settings).frequencies_MHz
    levels = reference.truth.levels_MHz
    match = match_poles(found, levels, level_tolerances(levels, hardware.bin_MHz), 1 / hardware.dt_us)
    display_match(match, found, reference.truth.levels_MHz, hardware.bin_MHz,
                  title=f"fitter {name}: {len(match.level_index)} of {len(reference.truth.levels_MHz)} "
                        f"matched, {len(match.false_poles)} false")

# %% [markdown]
# ## Benchmark 2: data with non-idealities
#
# Decay per us, white noise at a given SNR, and a frequency offset per row of width sigma.
# Each level is matched within min(0.25 bin, 1/4 of its nearest separation), so that poles
# spread at random match only a few percent; a level is *resolved* when it and its nearest
# neighbour are both matched. `lambda_eff` is the separation resolved in half the cases, in
# bins of the measured grid (12.2 kHz) for every window length. The small-gap-ratio
# statistic is `I(r0) = P(r < r0)`, `r0 = 0.25`, pooled over draws and seeds, against the
# true levels' own.
#
# Two questions, kept apart:
#
# - **2a. How good is each method?** Longer windows (same dt) without decay, and with the
#   decay of T2 = 100 us, which caps what a longer window adds. SNR 100.
# - **2b. What does the measured grid allow?** 100 samples, decay 0.01 per us; SNR and
#   sigma swept. This gives `lambda_eff` for benchmark 3.

# %%
window_points = sample_phase_diagram(kerr_over_g, disorder_over_g, draws=1 if small else 5)
decays = [Nonideal(snr=100), Nonideal(snr=100, decay_per_us=0.01)]
windows = [run_nonideal_bench(window_points, decays, seeds=2 if small else 10,
                              hardware=Hardware(samples=samples), fitters=fitters, name=f"window_{samples}")
           for samples in (100, 200, 400)]
window_curve = resolution_curve(windows, hardware.bin_MHz)
plot_resolution(window_curve);
display(lambda_eff(window_curve).unstack("fitter"))

# %%
display(scores_table(windows).groupby(["fitter", "samples", "decay_per_us"])
        [["levels", "found", "matched", "resolved", "false_poles"]].mean().round(1).unstack("fitter"))
window_gaps = small_gap_table(windows)
display(window_gaps.assign(bias=window_gaps.found - window_gaps.true)
        .groupby(["fitter", "samples", "decay_per_us"]).bias.agg(["mean", "std"]).round(3).unstack("fitter"))

# %%
snrs = [30, 100, 1000] if small else [10, 30, 100, 300, 1000]
sigmas_bins = [0.05, 0.25] if small else [0.02, 0.05, 0.1, 0.25, 0.5]
conditions = ([Nonideal(snr=snr, decay_per_us=0.01) for snr in snrs]
              + [Nonideal(snr=100, decay_per_us=0.01, offset_sigma_MHz=s * hardware.bin_MHz) for s in sigmas_bins])
nonideal = run_nonideal_bench(points, conditions, seeds=3 if small else 20, hardware=hardware, fitters=fitters)

# %%
curve = resolution_curve(nonideal, hardware.bin_MHz)
plot_resolution(curve);
display(lambda_eff(curve).unstack("fitter"))

# %%
table = scores_table(nonideal)
display(table.groupby(["fitter", "snr", "offset_sigma_MHz"])[["levels", "found", "matched", "resolved",
                                                             "false_poles"]].mean().round(1).unstack("fitter"))

# %%
gaps = small_gap_table(nonideal)
plot_small_gap(gaps);
display(gaps.assign(bias=gaps.found - gaps.true).groupby(["fitter", "snr", "offset_sigma_MHz"])
        .bias.agg(["mean", "std"]).round(3).unstack("fitter"))

# %% [markdown]
# ## Benchmarks 3 and 4: real data
#
# In `analysis_notebooks/pole_finding/real_benchmarks.py` (about 1 min; reads the registry,
# measures each set's lambda_eff with benchmark 2 at its own conditions).

# %% [markdown]
# ## Summary and results file

# %%
if output_dir is not None:
    path = save_results(output_dir / "results.h5", [*ideal.values(), *windows, nonideal], run=run)
    print("saved", path)

"""The report's figures. Each takes a summary table (``bench_summaries``) and returns the figure.

One fixed color per fitter (the validated reference categorical palette), so a fitter
keeps its color in every figure.
"""
import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.poles.bench_summaries import CONDITION

FITTER_COLORS = {"A": "#2a78d6", "B": "#eb6834", "E": "#1baf7a", "C": "#eda100", "D": "#e87ba4"}


def condition_label(row):
    """-> '100 samples, SNR 30, decay 0.01/us, sigma 1.2 kHz' for a table row."""
    return (f"{row.samples:g} samples, SNR {row.snr:g}, decay {row.decay_per_us:g}/us, "
            f"sigma {1e3 * row.offset_sigma_MHz:.2g} kHz")


def plot_ideal_errors(table, ax=None):
    """Benchmark 1: largest frequency error against the grid's conditioning, per fitter."""
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    for fitter, rows in table.groupby("fitter"):
        errors = np.maximum(rows.max_error_bins, 1e-16)
        ax.loglog(rows.conditioning, errors, "o", ms=6, mfc="none", color=FITTER_COLORS[fitter],
                  label=f"{fitter}: {int((rows.matched == rows.levels).sum())}/{len(rows)} all matched")
    ax.axhline(1e-6, color="0.5", lw=0.8, ls="--")
    ax.set(xlabel="Vandermonde conditioning of the true levels", ylabel="max |error| (bins)",
           title="Benchmark 1: noise-free returns")
    ax.legend(fontsize=8)
    return ax.figure


def plot_resolution(curve, unit="bins of the measured grid"):
    """Benchmark 2: P(resolved) against nearest-neighbour separation, one panel per condition."""
    conditions = curve[CONDITION].drop_duplicates().reset_index(drop=True)
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.2 * len(conditions), 3), sharey=True,
                             squeeze=False)
    for ax, (_, condition) in zip(axes[0], conditions.iterrows()):
        rows = curve[(curve[CONDITION] == condition.values).all(axis=1)]
        for fitter, group in rows.groupby("fitter"):
            centers = [interval.left + min(interval.length, 0.25) / 2 for interval in group.bin]
            ax.plot(centers, group.probability, "o-", lw=2, ms=5, color=FITTER_COLORS[fitter], label=fitter)
        ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
        ax.set(xlabel=f"separation ({unit})", title=condition_label(condition), ylim=(-0.05, 1.05))
        ax.title.set_fontsize(8)
    axes[0, 0].set_ylabel("P(resolved)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_small_gap(table):
    """Benchmark 2: I(r0) of the found poles against the true one, per fitter."""
    fitters = sorted(table.fitter.unique())
    fig, axes = plt.subplots(1, len(fitters), figsize=(3.2 * len(fitters), 3.2), squeeze=False)
    for ax, fitter in zip(axes[0], fitters):
        rows = table[table.fitter == fitter]
        ax.plot(rows.true, rows.found, "o", ms=6, mfc="none", color=FITTER_COLORS[fitter])
        top = np.nanmax([rows.true.max(), rows.found.max(), 0.1]) * 1.1
        ax.plot([0, top], [0, top], color="0.5", lw=0.8)
        ax.set(xlim=(0, top), ylim=(0, top), xlabel="I(r0), true levels", title=f"fitter {fitter}")
    axes[0, 0].set_ylabel("I(r0), found poles")
    fig.tight_layout()
    return fig

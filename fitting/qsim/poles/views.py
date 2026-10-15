"""The three views guan reads a fitter by (2026-10-05), for any data set and any fitters.

1. ``display_row_heatmaps``: per fitter, the |FFT| of every row (Fock state x frequency), with
   the fitter's poles marked where each row sees them (``E + delta_b``). Question: are the
   poles exactly on the peaks?
2. ``display_sticks``: per fitter, the pole weights after summing over rows (up), against the
   model's level weights (down). Question: does the stick diagram look like the model's?
3. ``display_gap_ratio_histograms``: per fitter, the histogram of the adjacent gap ratio r of
   the found poles, pooled over spectra, against the model's distribution and the Poisson and
   GOE curves; and ``display_gap_ratio_cdfs``, all fitters on one axis. Question: is the
   overall shape (not only the small-r bin) the model's?

Colors: one fixed color per fitter name (``FITTER_COLORS``), gray for the model. Plotting only.
"""
import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.mbr_spectrum import windowed_fft
from fitting.qsim.poles.diagnosis import seen_frequencies
from fitting.qsim.poles.pole_fit import normalize_to_initial_return, sample_time
from fitting.qsim.poles.statistics import bulk_gap_ratios

#: One color per fitter, in the reference palette's order; the same in every figure.
FITTER_COLORS = {"T1T3": "#2a78d6", "C": "#eb6834", "B": "#1baf7a", "A": "#eda100", "T1": "#e87ba4",
                 "T1free": "#4a3aa7", "T1F": "#008300", "F": "#e34948"}
MODEL_COLOR = "#7a7a7a"
INK = "#2b2b2b"
#: Amplitudes |c_b,lambda| below this are not marked in a row (as ``pole_plots``).
MARK_THRESHOLD = 0.05


def poisson_density(r):
    """-> P(r) of the min / max gap ratio for uncorrelated levels: 2 / (1 + r)^2 on [0, 1]."""
    return 2 / (1 + r) ** 2


def goe_density(r):
    """-> P(r) of the min / max gap ratio for the GOE surmise (Atas et al. 2013), on [0, 1]:
    2 x (27 / 8) (r + r^2) / (1 + r + r^2)^(5/2)."""
    return 2 * 27 / 8 * (r + r ** 2) / (1 + r + r ** 2) ** 2.5


def color_of(name):
    return FITTER_COLORS.get(name, INK)


def energy_range_kHz(levels_MHz, margin=0.08):
    low, high = 1e3 * np.min(levels_MHz), 1e3 * np.max(levels_MHz)
    pad = margin * (high - low)
    return low - pad, high + pad


def display_row_heatmaps(spectrum, fits, zero_padding=4, title=""):
    """-> figure: one panel per fitter, rows (Fock states) x frequency |FFT| (Hann, each row
    scaled to its own maximum), the fitter's poles as rings at ``E + delta_b`` in each row where
    ``|c_b| > MARK_THRESHOLD``, ring area ~ |c_b|. ``fits``: name -> PoleFit."""
    dt_us = sample_time(spectrum.time_us)
    a = normalize_to_initial_return(np.asarray(spectrum.A, dtype=complex))
    energy_kHz = 1e3 * np.fft.fftshift(np.fft.fftfreq(zero_padding * len(spectrum.time_us), d=dt_us))
    image = windowed_fft(a, "hann", zero_padding)
    image = image / image.max(axis=1, keepdims=True)
    rows = np.arange(len(a))
    labels = ["".join(map(str, o)) for o in spectrum.occupations] if spectrum.occupations else [str(b) for b in rows]
    fig, axes = plt.subplots(1, len(fits), figsize=(3.6 * len(fits) + 0.8, 0.22 * len(rows) + 1.6), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    x_range = energy_range_kHz(spectrum.levels_MHz)
    for ax, (name, fit) in zip(axes, fits.items()):
        ax.imshow(image, origin="lower", aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
                  extent=[energy_kHz[0], energy_kHz[-1], -0.5, len(rows) - 0.5])
        offsets = np.zeros(len(rows)) if fit.row_offsets_MHz is None else fit.row_offsets_MHz
        b, k = np.nonzero(np.abs(fit.amplitudes) > MARK_THRESHOLD)
        ax.scatter(1e3 * (fit.frequencies_MHz[k] + offsets[b]), b, s=70 * np.minimum(np.abs(fit.amplitudes[b, k]), 1.2),
                   facecolors="none", edgecolors=color_of(name), linewidths=1.3)
        ax.set_xlim(x_range)
        ax.set_title(f"{name}: {len(fit.frequencies_MHz)} poles", color=INK, fontsize=11)
        ax.set_xlabel("E/h (kHz)", color=INK)
        ax.tick_params(colors=INK, labelsize=8)
    axes[0].set_yticks(rows, labels, fontsize=7, family="monospace")
    axes[0].set_ylabel("initial Fock state (M1 S1 S2 S3 S4)", color=INK)
    fig.suptitle(title or spectrum.label, color=INK)
    return fig


def display_sticks(spectrum, fits, title=""):
    """-> figure: one panel per fitter, its pole weights w = Re sum_b c_b up (at the frequency
    the rows see, ``diagnosis.seen_frequencies``), the model's level weights (summed over the
    measured rows) down, in gray."""
    model_weights = np.asarray(spectrum.row_weights).sum(axis=0)
    fig, axes = plt.subplots(len(fits), 1, figsize=(11, 1.25 * len(fits) + 0.9), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (name, fit) in zip(axes, fits.items()):
        ax.vlines(1e3 * spectrum.levels_MHz, 0, -model_weights, color=MODEL_COLOR, lw=2)
        ax.vlines(1e3 * seen_frequencies(fit), 0, fit.weights, color=color_of(name), lw=2)
        ax.axhline(0, color="#d0d0d0", lw=0.8)
        top = max(1.2, float(np.nanmax(fit.weights)) * 1.05)
        ax.set_ylim(-1.2 * model_weights.max(), top)
        ax.text(0.005, 0.92, f"{name} ({len(fit.frequencies_MHz)} poles) up, model down", transform=ax.transAxes,
                va="top", fontsize=9, color=INK)
        ax.tick_params(colors=INK, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlim(energy_range_kHz(spectrum.levels_MHz))
    axes[-1].set_xlabel("E/h (kHz)", color=INK)
    fig.supylabel("pole weight", color=INK, fontsize=10)
    fig.suptitle(title or spectrum.label, color=INK)
    return fig


def pooled_ratios(level_sets_MHz, edge_fraction=0.10):
    """-> the bulk gap ratios of each level set, pooled."""
    parts = [bulk_gap_ratios(levels, edge_fraction) for levels in level_sets_MHz]
    return np.concatenate(parts) if parts else np.array([])


def display_gap_ratio_histograms(found_level_sets, model_level_sets, reference_ratios=None, bins=10,
                                 model_fit=(), title=""):
    """-> figure: small multiples, the model's own levels first, then one panel per fitter.

    ``found_level_sets``: name -> list of pole frequency arrays (one per spectrum);
    ``model_level_sets``: the model's levels of the same spectra; ``reference_ratios``: gap
    ratios of many more model draws at the same point (the smooth line). Fitters in
    ``model_fit`` (T1, T1free) take their levels from a model fit: marked, they carry the
    model's statistics.
    """
    edges = np.linspace(0, 1, bins + 1)
    r = np.linspace(0, 1, 201)
    panels = {"model, same draws": model_level_sets, **found_level_sets}
    columns = min(3, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.1 * rows), sharex=True, sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, sets) in zip(axes, panels.items()):
        ratios = pooled_ratios(sets)
        color = MODEL_COLOR if name.startswith("model") else color_of(name)
        ax.hist(ratios, bins=edges, density=True, color=color, alpha=0.85, edgecolor="white", linewidth=2)
        if reference_ratios is not None:
            density, _ = np.histogram(reference_ratios, bins=edges, density=True)
            ax.step(edges, np.r_[density, density[-1]], where="post", color=INK, lw=2, label="model, many draws")
        ax.plot(r, poisson_density(r), color=INK, lw=1, ls="--", label="Poisson")
        ax.plot(r, goe_density(r), color=INK, lw=1, ls=":", label="GOE")
        note = " (model fit)" if name in model_fit else ""
        ax.set_title(f"{name}{note}: {len(ratios)} ratios, <r> {np.mean(ratios):.3f}" if len(ratios) else name,
                     fontsize=10, color=INK)
        ax.tick_params(colors=INK, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.supxlabel("gap ratio r = min(s_n, s_n+1) / max(s_n, s_n+1)", color=INK, fontsize=10)
    fig.supylabel("probability density", color=INK, fontsize=10)
    fig.suptitle(title, color=INK)
    return fig


def display_gap_ratio_cdfs(found_level_sets, model_level_sets, reference_ratios=None, model_fit=(), title=""):
    """-> figure: the cumulative distribution of r for every fitter and the model, on one axis
    (less noisy than a histogram with a few hundred ratios), with the Poisson and GOE curves."""
    r = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    poisson = 2 * r / (1 + r)                                   # integral of poisson_density
    goe = np.cumsum(goe_density(r)) * (r[1] - r[0])
    ax.plot(r, poisson, color=INK, lw=1, ls="--", label="Poisson")
    ax.plot(r, goe / goe[-1], color=INK, lw=1, ls=":", label="GOE")
    if reference_ratios is not None:
        x = np.sort(reference_ratios)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=INK, lw=2.2, label="model, many draws")
    x = np.sort(pooled_ratios(model_level_sets))
    ax.step(x, np.arange(1, len(x) + 1) / len(x), color=MODEL_COLOR, lw=2, where="post", label="model, same draws")
    for name, sets in found_level_sets.items():
        x = np.sort(pooled_ratios(sets))
        if len(x):
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color_of(name), lw=2,
                    ls="-" if name not in model_fit else (0, (4, 2)), label=name + (" (model fit)" if name in model_fit else ""))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("gap ratio r", color=INK)
    ax.set_ylabel("fraction of ratios below r", color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(title, color=INK, fontsize=10)
    return fig

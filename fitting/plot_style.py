"""One plot style for lab figures (vault PNGs, notebook output), so that plotting code keeps only
the data logic.

Use it around the code that makes a figure:

    from fitting import plot_style

    with plot_style.context():
        fig, ax = plt.subplots()
        ax.plot(t, y, color=plot_style.STATE_COLORS["f"])
        plot_style.suptitle(fig, "title")

``context()`` applies ``RC`` only inside the block (``matplotlib.style.context``): surface, ink,
quiet axes and grid, font sizes, and the categorical color cycle. It is not applied on import, so
other code keeps its own settings. Note: ``fitting/fit_display.py`` sets ``font.size`` 14 and
``figure.figsize`` globally on import (and the flat ``experiments`` namespace imports it); inside
the context, ``RC`` wins.

Colors: ``CATEGORICAL`` is the validated reference palette of the dataviz guidance (8 slots, in
this order; checked for color-vision deficiency on the light surface). Fixed meanings take fixed
slots, so that one entity has one color in every figure: ``STATE_COLORS`` (g, e, f) and
``MODE_COLORS`` (M1, S1..S7). Text and axes use the ink colors, not series colors. Slot 3 (aqua)
is below 3:1 contrast on the surface: give it a marker shape or a direct label, as ``STATE_MARKERS``
does.
"""
import matplotlib.pyplot as plt
from cycler import cycler

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STATE_COLORS = dict(zip("gef", CATEGORICAL))
STATE_MARKERS = dict(g="o", e="s", f="^")
MODE_COLORS = dict(zip(["M1"] + [f"S{k}" for k in range(1, 8)], CATEGORICAL))

RC = {
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.titlesize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "errorbar.capsize": 3,
}


def context():
    """-> a context manager that applies ``RC`` to the figures made inside it."""
    return plt.style.context(RC)


def suptitle(fig, text):
    """Left-aligned figure title (the figure title of every lab figure)."""
    return fig.suptitle(text, x=0.01, ha="left")

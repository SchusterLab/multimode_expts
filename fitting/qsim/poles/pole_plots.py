"""One figure for the fit of any fitter: where each pole sits in the data, and what is left.

Any fitter's ``PoleFit`` goes in, so the fitters can be put side by side on one data set. The
panel to read first is the residual: a level the fitter missed but the data hold shows as a
peak in the residual FFT of the rows it lives in; a level that is not in the data leaves the
residual flat there.

Plotting only; the FFT is the one every spectrum here uses (``windowed_fft``).
"""
import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.mbr_spectrum import windowed_fft
from fitting.qsim.poles.pole_fit import normalize_to_initial_return, sample_time

#: Amplitudes |c_b,lambda| (and model weights) below this are not marked in a row.
MARK_THRESHOLD = 0.05


def display_pole_fit(fit, A, time_us, levels_MHz=None, row_weights=None, row_labels=None,
                     fft_window="hann", zero_padding=8, energy_range_MHz=None, title=""):
    """-> figure: per-row FFT of the data and of the residual, with the found poles and the
    model levels; the row sum of both; the residual of each row.

    ``levels_MHz`` and ``row_weights`` (row x level) are the model's (optional). A found pole is
    marked in row b at ``E + delta_b`` (the row's offset, if the fitter found one), sized by
    ``|c_b|``; a model level in row b where its weight is above ``MARK_THRESHOLD``.
    ``energy_range_MHz`` (low, high): the x range; default the span of the model levels plus a
    quarter of it on each side (``default_range``); pass the grid's ends to see the whole band.
    """
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(np.asarray(A, dtype=complex))
    fitted = fit.returns(time_us)
    energy_MHz = np.fft.fftshift(np.fft.fftfreq(zero_padding * len(time_us), d=dt_us))
    data_fft, residual_fft = (windowed_fft(x, fft_window, zero_padding) for x in (a, a - fitted))
    rows = np.arange(len(a))
    labels = [str(label) for label in row_labels] if row_labels is not None else [str(b) for b in rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), height_ratios=(2, 1), width_ratios=(1, 1),
                             constrained_layout=True)
    for ax, local, name in ((axes[0, 0], data_fft, "data"), (axes[0, 1], residual_fft, "residual")):
        image = ax.imshow(local, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
                          extent=[energy_MHz[0], energy_MHz[-1], -0.5, len(rows) - 0.5], vmin=0)
        fig.colorbar(image, ax=ax, label="|FFT|")
        mark_rows(ax, fit, levels_MHz, row_weights)
        ax.set(xlabel="energy E/h (MHz)", title=f"{name}, per row")
        ax.set_yticks(rows, labels, fontsize=7)
    axes[0, 0].legend(fontsize=8, loc="upper right")

    display_row_sums(axes[1, 0], energy_MHz, data_fft, windowed_fft(fitted, fft_window, zero_padding),
                     residual_fft, fit, levels_MHz, row_weights)
    residuals = np.linalg.norm(a - fitted, axis=1) / np.linalg.norm(a, axis=1)
    axes[1, 1].bar(rows, residuals, color="0.4")
    axes[1, 1].set_xticks(rows, labels, rotation=90, fontsize=7)
    axes[1, 1].set(ylabel="|a_b - fit| / |a_b|", title="residual per row")
    for ax in axes[:, 0].tolist() + [axes[0, 1]]:
        ax.set_xlim(energy_range_MHz or default_range(fit, levels_MHz, energy_MHz))
    fig.suptitle(f"{title}  {len(fit.frequencies_MHz)} poles; relative residual "
                 f"{np.linalg.norm(a - fitted) / np.linalg.norm(a):.3f}; frequencies modulo {1 / dt_us:.4g} MHz")
    return fig


def default_range(fit, levels_MHz, energy_MHz):
    """-> the span of the model levels (or, without a model, of the poles heavier than
    ``MARK_THRESHOLD``) plus a quarter of it each side, within the grid."""
    heavy = fit.frequencies_MHz[np.abs(fit.weights) > MARK_THRESHOLD]
    marked = np.asarray(levels_MHz) if levels_MHz is not None else heavy
    if len(marked) == 0:
        return energy_MHz[0], energy_MHz[-1]
    low, high = marked.min(), marked.max()
    margin = 0.25 * (high - low) + 2 * (energy_MHz[1] - energy_MHz[0])
    return max(low - margin, energy_MHz[0]), min(high + margin, energy_MHz[-1])


def mark_rows(ax, fit, levels_MHz, row_weights):
    """Found poles (cyan circles, area ~ |c_b|) and model levels (white ticks) in each row."""
    rows = np.arange(fit.amplitudes.shape[0])
    offsets = np.zeros(len(rows)) if fit.row_offsets_MHz is None else fit.row_offsets_MHz
    b, k = np.nonzero(np.abs(fit.amplitudes) > MARK_THRESHOLD)
    ax.scatter(fit.frequencies_MHz[k] + offsets[b], b, s=60 * np.minimum(np.abs(fit.amplitudes[b, k]), 1.5),
               facecolors="none", edgecolors="cyan", linewidths=0.9, label="found poles (size |c_b|)")
    if levels_MHz is not None:
        b, k = np.nonzero(np.asarray(row_weights) > MARK_THRESHOLD)
        ax.scatter(np.asarray(levels_MHz)[k], b, marker="|", s=80, color="white", linewidths=1.2,
                   label=f"model levels (weight > {MARK_THRESHOLD})")


def display_row_sums(ax, energy_MHz, data_fft, fitted_fft, residual_fft, fit, levels_MHz, row_weights):
    """Sum over rows of each |FFT| (so row offsets cannot cancel), found pole weights up,
    model weights down."""
    ax.plot(energy_MHz, data_fft.sum(axis=0), color="black", lw=1.4, label="data")
    ax.plot(energy_MHz, fitted_fft.sum(axis=0), color="tab:blue", ls="--", lw=1.2, label="fit")
    ax.plot(energy_MHz, residual_fft.sum(axis=0), color="tab:red", lw=1.2, label="residual")
    top = data_fft.sum(axis=0).max()
    scale = top / max(np.max(np.abs(fit.weights)), 1)
    ax.vlines(fit.frequencies_MHz, 0, scale * fit.weights, color="tab:blue", alpha=0.7, label="found pole weights")
    if levels_MHz is not None:
        ax.vlines(levels_MHz, 0, -scale * np.asarray(row_weights).sum(axis=0), color="tab:orange", alpha=0.8,
                  label="model weights (down)")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="energy E/h (MHz)", ylabel="sum over rows of |FFT|; weights (scaled)", title="sum over rows")
    ax.legend(fontsize=8, ncols=2)

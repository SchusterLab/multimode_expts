"""Q1: what the exact model has to show, before any measurement (perfect resolution).

Gap ratios of the N = 3 and N = 4 sectors over (K/g, W/g), W = RMS storage onsite spread.
Writes cache/q1_model_stats.npz and the figures q1_*.png to the vault.
Run: pixi run python analysis_notebooks/level_stats/labbook/q1_model_stats.py   (about 1 min)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (CACHE, FIGURES, GOE_MEAN_R, POISSON_MEAN_R, disorder_shape, gap_ratios,
                    goe_r_density, poisson_r_density, spectrum)

KERRS = np.array([0.0, -0.13, -0.5, -0.9, -1.8, -2.5, -4.0])
DISORDERS = np.array([0.05, 0.25, 0.5, 1.0, 1.7, 3.0, 5.0, 8.0])
DRAWS = 1000
BINS = np.linspace(0, 1, 11)


def run():
    out = {}
    for n in (3, 4):
        mean_r = np.zeros((len(KERRS), len(DISORDERS)))
        small = np.zeros_like(mean_r)
        tiny = np.zeros_like(mean_r)
        small_sd = np.zeros_like(mean_r)
        hist = np.zeros((len(KERRS), len(DISORDERS), len(BINS) - 1))
        for i, k in enumerate(KERRS):
            for j, w in enumerate(DISORDERS):
                rng = np.random.default_rng(1000 * n + 17)  # same shapes at every (K, W): paired cells
                per_real = []
                for _ in range(DRAWS):
                    levels, _ = spectrum(n, k, w, disorder_shape(rng))
                    per_real.append(gap_ratios(levels))
                r = np.concatenate(per_real)
                mean_r[i, j] = r.mean()
                small[i, j] = np.mean(r < 0.25)
                tiny[i, j] = np.mean(r < 0.1)
                small_sd[i, j] = np.std([np.mean(x < 0.25) for x in per_real])
                hist[i, j] = np.histogram(r, BINS, density=True)[0]
        out[n] = dict(mean_r=mean_r, small=small, tiny=tiny, small_sd=small_sd, hist=hist,
                      ratios_per_real=len(per_real[0]))
        print(f"N={n}: bulk ratios per realization {len(per_real[0])}")
        for i, k in enumerate(KERRS):
            print(f"  K/g={k:+.2f}  <r>: " + " ".join(f"{x:.3f}" for x in mean_r[i])
                  + " | P(r<.25): " + " ".join(f"{x:.2f}" for x in small[i]))
    np.savez(CACHE / "q1_model_stats.npz", kerrs=KERRS, disorders=DISORDERS, bins=BINS,
             **{f"N{n}_{key}": val for n, d in out.items() for key, val in d.items()})
    return out


def figures(out):
    for n, d in out.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for ax, key, title, ref in ((axes[0], "mean_r", r"$\langle r\rangle$", (POISSON_MEAN_R, GOE_MEAN_R)),
                                    (axes[1], "small", r"$P(r<0.25)$", (0.40, 0.12))):
            im = ax.imshow(d[key], origin="lower", cmap="viridis", aspect="auto")
            for i in range(len(KERRS)):
                for j in range(len(DISORDERS)):
                    ax.text(j, i, f"{d[key][i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="w" if d[key][i, j] < np.mean(d[key]) else "k")
            ax.set_xticks(range(len(DISORDERS)), [f"{w:g}" for w in DISORDERS])
            ax.set_yticks(range(len(KERRS)), [f"{k:g}" for k in KERRS])
            ax.set_xlabel("disorder W / g (RMS onsite)")
            ax.set_ylabel("Kerr K / g")
            ax.set_title(f"{title}, N = {n} (Poisson {ref[0]:.2f}, GOE {ref[1]:.2f})")
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"Exact model, perfect resolution, {DRAWS} disorder draws per cell, 10 % edges cut")
        fig.savefig(FIGURES / f"q1_model_maps_N{n}.png", dpi=130)
        plt.close(fig)

    # Histograms along the two cuts that matter.
    r = np.linspace(0, 1, 200)
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True, constrained_layout=True)
    cuts = [("Kerr axis at W/g = 1.7", [(i, 4) for i in (0, 3, 4, 6)]),
            ("disorder axis at K/g = -0.9", [(3, j) for j in (0, 2, 4, 6)])]
    for row, (label, cells) in enumerate(cuts):
        for ax, (i, j) in zip(axes[row], cells):
            for n, ls in ((3, "-"), (4, "--")):
                ax.step(centers, out[n]["hist"][i, j], where="mid", ls=ls, lw=2, label=f"N = {n}")
            ax.plot(r, poisson_r_density(r), "k:", lw=1, label="Poisson")
            ax.plot(r, goe_r_density(r), "k-.", lw=1, label="GOE")
            ax.set_title(f"K/g = {KERRS[i]:g}, W/g = {DISORDERS[j]:g}")
            ax.set_xlabel("r")
        axes[row, 0].set_ylabel(f"P(r)\n{label}")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(FIGURES / "q1_model_histograms.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    figures(run())

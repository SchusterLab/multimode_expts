# %% [markdown]
# # Slide figure: how we read the photon number of one mode (multiparity readout)
#
# Schematic only, no data. It follows the code:
# `experiments/qsim/qsim_base.py` `QsimProgram.multi_parity_readout` (the AGP scans of 10-17..11-07)
# and `experiments/qsim/fock_population_sequential.py` `multiparity` (the same two bits, one mode
# after the other). Settings as recorded in the AGP jobs: `phase_second_pulse` 180,
# `cond_sec_phase` -90, `parity_fast` False, revival time 1.41 us (JOB-20261023-00011).
# Confusion diagonal from JOB-20261024-00008 (`docs/log/2026-10-24_kerr-n123.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

INK, INK2, MUTED = "#14213D", "#4A5568", "#6A7179"
SURFACE, PANEL = "#FBFBF8", "#F2F0EA"
HOLD = "#EAF1FB"
N_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # n = 0..3, validated categorical slots 1-4

plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 13, "text.color": INK,
                     "mathtext.fontset": "dejavusans"})

fig = plt.figure(figsize=(13.3, 6.8), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13.3)
ax.set_ylim(0, 6.8)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, x1, y0, y1, text, fill="white", edge=INK2, dashed=False, size=13, weight="normal"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec=edge, lw=1.4, ls="--" if dashed else "-", zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size,
            weight=weight, zorder=4)


def arrow(p0, p1, color=INK2, rad=0.0, lw=1.6, style="-|>", ls="-", mut=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=5,
                                 shrinkA=0, shrinkB=0))


# %% (a) pulse sequence
ax.text(0.15, 6.6, "(a)  Pulse sequence for one mode", fontsize=16, weight="semibold", va="center")
Y_M, Y_Q = 5.75, 4.45       # row centres: hub mode M1, qubit
ax.text(0.15, Y_M, "M1", fontsize=14, weight="semibold", va="center")
ax.text(0.15, Y_Q, "qubit", fontsize=14, weight="semibold", va="center")
ax.plot([0.9, 13.15], [Y_Q, Y_Q], color=MUTED, lw=1.2, zorder=1)

box(0.9, 2.55, Y_M - 0.33, Y_M + 0.33, "Fock prep +\nFloquet evolution", fill=PANEL)
box(2.7, 3.85, Y_M - 0.33, Y_M + 0.33, "swap\nS$_k$ → M1", fill=PANEL)
ax.add_patch(FancyBboxPatch((4.0, Y_M - 0.22), 6.8, 0.44, boxstyle="round,pad=0,rounding_size=0.06",
                            fc=HOLD, ec="none", zorder=2))
ax.text(7.4, Y_M, "n photons stay in M1: each one shifts the qubit frequency by χ",
        ha="center", va="center", fontsize=13, color=INK, zorder=4)
box(11.0, 13.15, Y_M - 0.33, Y_M + 0.33, "next mode: swap\nS$_j$ → M1, repeat", fill=PANEL, size=12)

PH = 0.26                   # half height of a qubit pulse box
box(4.1, 4.6, Y_Q - PH, Y_Q + PH, "π/2")
box(5.8, 6.3, Y_Q - PH, Y_Q + PH, "π/2")
box(6.45, 7.3, Y_Q - PH, Y_Q + PH, "read", fill=INK, edge=INK, weight="semibold")
ax.texts[-1].set_color(SURFACE)
box(7.45, 7.95, Y_Q - PH, Y_Q + PH, "π", dashed=True)
box(8.2, 8.7, Y_Q - PH, Y_Q + PH, "π/2")
box(9.4, 9.9, Y_Q - PH, Y_Q + PH, "π/2", edge=N_COLOR[1])
box(10.05, 10.9, Y_Q - PH, Y_Q + PH, "read", fill=INK, edge=INK, weight="semibold")
ax.texts[-1].set_color(SURFACE)
ax.text(5.2, Y_Q + 0.12, r"wait $t_\pi$", ha="center", va="bottom", fontsize=13)
ax.text(5.2, Y_Q - 0.12, "1.4 µs", ha="center", va="top", fontsize=12, color=MUTED)
ax.text(9.05, Y_Q + 0.12, r"$t_\pi/2$", ha="center", va="bottom", fontsize=13)
ax.text(7.7, Y_Q + PH + 0.04, "if e", ha="center", va="bottom", fontsize=12, color=INK2)

for x0, x1, label in [(4.1, 7.3, "bit 0: parity of n"), (8.2, 10.9, "bit 1")]:
    yb = Y_Q + 0.48
    ax.plot([x0, x0, x1, x1], [yb - 0.08, yb, yb, yb - 0.08], color=INK2, lw=1.2)
    ax.text((x0 + x1) / 2, yb + 0.08, label, ha="center", va="bottom", fontsize=14, weight="semibold")

# real-time feedback from the first read
arrow((6.875, Y_Q - PH), (7.6, Y_Q - PH - 0.02), rad=0.6, color=N_COLOR[1])
arrow((6.875, Y_Q - PH), (9.65, Y_Q - PH - 0.02), rad=0.25, color=N_COLOR[1])
ax.text(8.25, 3.5, "feedback on the FPGA: reset to g if e, and phase 180° → 90° if bit 0 = e",
        ha="center", va="center", fontsize=12, color=INK2)

# %% (b) phase wheels
ax.text(0.15, 3.2, "(b)  Qubit phase before the last π/2: right ends in g, left in e",
        fontsize=16, weight="semibold", va="center")
R = 0.78
CY = 1.55
WHEELS = [
    (1.35, r"after $t_\pi$: phase n·π", {0: 0, 1: 180, 2: 0, 3: 180}, {}),
    (3.75, r"after $t_\pi/2$: phase n·π/2", {0: 0, 1: 90, 2: 180, 3: 270}, {}),
    (6.15, "bit 0 = e: frame −90°", {0: 0, 1: 0, 2: 180, 3: 180}, {1: 90, 3: 270}),
]
for cx, title, angles, ghosts in WHEELS:
    ax.add_patch(Circle((cx, CY), R, fc="white", ec=MUTED, lw=1.2, zorder=2))
    ax.plot([cx - R, cx + R], [CY, CY], color="#D5D8DC", lw=0.8, zorder=2)
    ax.text(cx, CY + R + 0.2, title, ha="center", va="bottom", fontsize=13)
    ax.text(cx + R + 0.08, CY, "g", ha="left", va="center", fontsize=13, weight="semibold")
    ax.text(cx - R - 0.08, CY, "e", ha="right", va="center", fontsize=13, weight="semibold")
    for n, ang in ghosts.items():
        th = np.deg2rad(ang)
        ax.plot([cx, cx + 0.82 * R * np.cos(th)], [CY, CY + 0.82 * R * np.sin(th)],
                color=N_COLOR[n], lw=1.4, ls=(0, (3, 2)), zorder=3, alpha=0.6)
        start = (cx + 0.9 * R * np.cos(th - 0.2), CY + 0.9 * R * np.sin(th - 0.2))
        end = (cx + 0.9 * R * np.cos(th - np.pi / 2 + 0.3), CY + 0.9 * R * np.sin(th - np.pi / 2 + 0.3))
        arrow(start, end, color=MUTED, rad=-0.35, lw=1.2, mut=11)
    for n, ang in angles.items():
        shared = [m for m, a in angles.items() if a == ang]
        side = 0 if len(shared) == 1 else (1 if n == shared[0] else -1)
        th = np.deg2rad(ang + 6 * side)
        tip = (cx + 0.82 * R * np.cos(th), CY + 0.82 * R * np.sin(th))
        arrow((cx, CY), tip, color=N_COLOR[n], lw=2.6, mut=16)
        tl = np.deg2rad(ang + (30 * side if side else 24))   # each arrow's number sits beside it
        ax.text(cx + 0.6 * R * np.cos(tl), CY + 0.6 * R * np.sin(tl), str(n), ha="center",
                va="center", fontsize=13, weight="semibold", zorder=6)
arrow((3.75 + R + 0.35, CY), (6.15 - R - 0.35, CY), color=MUTED, lw=1.4)
ax.text(1.35, CY - R - 0.22, "even → g, odd → e", ha="center", va="top", fontsize=13)
ax.text(4.95, CY - R - 0.22, "n = 0, 1 → g;  n = 2, 3 → e", ha="center", va="top", fontsize=13)

# %% (c) decoding and the facts behind it
ax.text(7.7, 3.2, "(c)  Two bits give n", fontsize=16, weight="semibold", va="center")
cols = [7.95, 8.85, 9.75]
ax.text(cols[0], 2.8, "n", ha="center", fontsize=13, weight="semibold", color=INK2)
ax.text(cols[1], 2.8, "bit 0", ha="center", fontsize=13, weight="semibold", color=INK2)
ax.text(cols[2], 2.8, "bit 1", ha="center", fontsize=13, weight="semibold", color=INK2)
for n in range(4):
    y = 2.45 - 0.32 * n
    ax.add_patch(Circle((cols[0] - 0.25, y + 0.07), 0.07, fc=N_COLOR[n], ec="none"))
    ax.text(cols[0], y, str(n), ha="center", fontsize=14, weight="semibold")
    ax.text(cols[1], y, "e" if n % 2 else "g", ha="center", fontsize=14)
    ax.text(cols[2], y, "e" if n >= 2 else "g", ha="center", fontsize=14)
ax.text(10.35, 2.3, "n = bit 0 + 2 · bit 1  (mod 4)\nN = 3 photons: mod 4 is n itself",
        ha="left", va="center", fontsize=13, linespacing=1.5)
facts = [
    "QND: χ a†a σ$_z$ commutes with n; reading the\nqubit does not change the photon number.",
    "Confusion matrix (10-24): diagonal 0.93, 0.87, 0.85,\n0.83; populations corrected by its inverse.",
]
for i, text in enumerate(facts):
    ax.text(7.7, 1.05 - 0.62 * i, text, ha="left", va="center", fontsize=13, color=INK2, linespacing=1.35)

out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "slide_multiparity_readout.png", dpi=200, facecolor=SURFACE)
fig.savefig(out / "slide_multiparity_readout.svg", facecolor=SURFACE)

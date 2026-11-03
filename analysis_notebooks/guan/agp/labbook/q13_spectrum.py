"""Q13: the spectral function S(f) of the measured C_1, the quantity that the AGP filter integrates.

``CentralBosonAGPDisorderScan.display_spectrum``: the cosine transform of C_1 over its window
(Hann), per W, next to the closed system over the same window and its weights in bins of 1/T,
with the AGP filters of MU_KHZ. Once as measured, once after the exponential background fit
(Q5) for t >= FIT_START_US. Figures go to ``figures/``. Run from this folder.
"""
# %%
from pathlib import Path
import warnings

import matplotlib

try:
    get_ipython()
except NameError:  # a plain script: no screen, draw off-screen
    matplotlib.use("Agg")

from load_scans import load_scans

MU_KHZ = [5., 10.]
FIT_START_US = 30.
MAX_KHZ = 120.
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# %%
scans = load_scans()
for scan in scans.values():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=MU_KHZ)

# %%
for name, scan in scans.items():
    slug = name.lower().replace(" ", "_")
    for suffix, start in (("", None), ("_bg", FIT_START_US)):
        fig, _ = scan.display_spectrum(max_khz=MAX_KHZ, background_start_us=start)
        fig.suptitle(f"{name}: " + fig.get_suptitle(), fontsize=10)
        fig.savefig(FIGURES / f"q13_spectrum_{slug}{suffix}.png", dpi=110)
        print(FIGURES / f"q13_spectrum_{slug}{suffix}.png")

"""Q13: the spectral function S(f) of the measured C_1, the quantity that the AGP filter integrates.

``CentralBosonAGPDisorderScan.display_spectrum``: the cosine transform of C_1 over its window
(Hann), per W, next to the closed system over the same window and its weights in bins of 1/T,
with the AGP filters of MU_KHZ. Once as measured, once after the exponential background fit
(Q5) for t >= FIT_START_US. Figures go to ``figures/``. Runs as a script from any folder, or
cell by cell in a REPL (no ``__file__`` needed: the folder is found from the checkout).
"""
# %%
from pathlib import Path
import sys
import warnings

import matplotlib

import experiments

try:
    get_ipython()
except NameError:  # a plain script: no screen, draw off-screen
    matplotlib.use("Agg")

HERE = Path(experiments.__file__).resolve().parents[1] / "analysis_notebooks" / "agp" / "labbook"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from load_scans import load_scans  # noqa: E402

MU_KHZ = [5., 10.]
FIT_START_US = 30.
MAX_KHZ = 120.
FIGURES = HERE / "figures"
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

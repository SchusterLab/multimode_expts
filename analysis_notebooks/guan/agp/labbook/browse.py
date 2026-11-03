"""Browse the AGP data sets in assembled_data: list the manifests, pick one by key, analyze, plot."""
# %%
import yaml

from experiments.job_paths import data_root
from experiments.qsim.agp import CentralBosonAGPDisorderScan

ASSEMBLED = data_root() / "agp_k44_first_scan" / "assembled_data"

# %% every disorder scan manifest: stem, direction, W (kHz), number of cycles, notes
for path in sorted(ASSEMBLED.glob("*_CentralBosonAGPDisorderScan.yaml")):
    manifest = yaml.safe_load(path.read_text())
    p = manifest["parameters"]
    print(path.stem[:13], p["direction"], p["disorder_khz"], len(p["cycles"]), manifest["notes"])

# %%
SCANS = {
    "S7_fine": "261023_001509",     # W = 0-150, 122 us
    "S7_coarse": "261027_163118",   # W = 0-150, 244 us (backfilled: 120247 + 001448)
    "Kerr": "261025_010957",        # M1, W = 20-200, 245 us
    "Kerr_W0": "261024_003051",     # M1, W = 0
}

# %%
scan = CentralBosonAGPDisorderScan.from_manifest(
    ASSEMBLED / f"{SCANS['S7_fine']}_CentralBosonAGPDisorderScan.yaml")
scan.analyze(mu=[5., 10.])
scan.display()
scan.display_spectrum(max_khz=120.)


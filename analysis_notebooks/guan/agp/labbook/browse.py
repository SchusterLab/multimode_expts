"""Browse the AGP data sets in assembled_data: list the manifests, pick one by key, analyze, plot."""
# %%
from experiments.data_paths import locate
from experiments.qsim.agp import list_scans, load_scan, registered_scans

# %% every disorder scan manifest: name, direction, W (kHz), cycles, notes
list_scans(locate("agp_k44_first_scan"))

# %% the registered data sets (analysis_notebooks/datasets/agp_scans.yaml): s7_fine, s7_coarse, kerr, kerr_w0, ...
{key: entry["label"] for key, entry in registered_scans().items()}

# %%
scan = load_scan("s7_fine")
scan.analyze(mu=[5., 10.])
scan.display()
scan.display_spectrum(max_khz=120.)


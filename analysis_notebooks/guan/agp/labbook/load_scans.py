"""Lab book 2026-10-25: load every AGP data set once (shared by the q*.py scripts)."""
from pathlib import Path
import warnings

import numpy as np

from experiments.job_paths import data_root
from experiments.qsim.agp import CentralBosonAGPDisorderScan, load_scan

ROOT = data_root() / "agp_k44_first_scan"
VAULT = Path(r"G:\Shared drives\SLab\Multimode\Lab\guan\AGP\labbook_figures")


def load_scans(kerr=True):
    """-> {name: scan}: S7 fine (122 us), S7 coarse (244 us), Kerr direction (245 us)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scans = {
            "S7 fine": load_scan("s7_fine"),
            "S7 coarse": load_scan("s7_coarse"),
        }
        if kerr:
            scans["Kerr"] = CentralBosonAGPDisorderScan.from_job_files(
                sorted((ROOT / "data").glob("JOB-20261024-*_FockPopulationExperiment.h5")),
                direction=0)
    return scans

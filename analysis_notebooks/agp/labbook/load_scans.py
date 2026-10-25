"""Lab book 2026-10-25: load every AGP data set once (shared by the q*.py scripts)."""
from pathlib import Path
import warnings

import numpy as np

from experiments.job_paths import data_root
from experiments.qsim.agp import CentralBosonAGPDisorderScan

ROOT = data_root() / "agp_k44_first_scan"
VAULT = Path(r"G:\Shared drives\SLab\Multimode\Lab\guan\AGP\labbook_figures")


def _manifest(name):
    return CentralBosonAGPDisorderScan.from_manifest(ROOT / "assembled_data" / name)


def load_scans(kerr=True):
    """-> {name: scan}: S7 fine (122 us), S7 coarse (244 us), Kerr direction (245 us)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scans = {
            "S7 fine": _manifest("261023_001509_CentralBosonAGPDisorderScan.yaml"),
            "S7 coarse": CentralBosonAGPDisorderScan.from_children(
                _manifest("261023_001448_CentralBosonAGPDisorderScan.yaml").children
                + _manifest("261023_120247_CentralBosonAGPDisorderScan.yaml").children),
        }
        if kerr:
            scans["Kerr"] = CentralBosonAGPDisorderScan.from_job_files(
                sorted((ROOT / "data").glob("JOB-20261024-*_FockPopulationExperiment.h5")),
                direction=0)
    return scans

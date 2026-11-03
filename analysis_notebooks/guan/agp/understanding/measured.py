"""The measured AGP data sets as arrays, for the scripts in this folder. Loaded from the job files each time."""
import sys
import warnings

import numpy as np

sys.path.insert(0, "../labbook")
from experiments.qsim.agp import CentralBosonAGPDisorderScan  # noqa: E402
from load_scans import ROOT, load_scans  # noqa: E402

# The Kerr-direction W = 0 run (10-24 00:30, lab book Q8) has its own manifest; load_scans leaves it out.
# It used the configuration with f0-g1 57 kHz off (lab book caveat).
KERR_W0 = "261024_003051_CentralBosonAGPDisorderScan.yaml"


def scan_arrays(scan):
    """-> dict of arrays of one scan, analyzed at mu = 10 kHz: C_1, errors, W, couplings, Kerr, pattern."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    ref = scan.children[1] if len(scan.children) > 1 else scan.children[0]
    try:
        pattern = np.asarray(scan.disorder_pattern())
    except Exception:  # a W = 0 run alone has no pattern
        pattern = np.full(4, np.nan)
    return dict(
        times=np.asarray(scan.data["time_us"]), data=np.asarray(scan.data["C1_corrected"]),
        err=np.asarray(scan.data["C1_err"]), pattern=pattern,
        W=np.array([p.disorder_khz for p in scan.children]),
        g=np.array([float(p.data["coupling_mhz"]) for p in scan.children]),
        couplings=np.asarray(ref.recorded_couplings_mhz()), kerr=float(ref.recorded_kerr_mhz()))


def measured_scans():
    """-> {name: scan_arrays}: S7 fine, S7 coarse, Kerr, Kerr W=0."""
    scans = load_scans()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scans["Kerr W=0"] = CentralBosonAGPDisorderScan.from_manifest(ROOT / "assembled_data" / KERR_W0)
    return {name: scan_arrays(scan) for name, scan in scans.items()}

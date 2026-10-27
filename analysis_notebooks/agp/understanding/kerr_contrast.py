"""Does the measured AGP at the resolvable mu tell the interacting model from free bosons?

Measured: exp-fit estimate (lab book Q5) at mu = 10, 20 kHz, error = shot noise + 0.13 x the
decaying-DC term (Q5 systematic). Theory: closed system on the scans' disorder pattern with the
recorded K, with guan's dressed K = -19 kHz, and with K = 0 (free bosons, integrable).
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "labbook"))
from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from experiments.qsim.agp import CentralBosonAGPDisorderScan
from load_scans import ROOT, load_scans
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path(__file__).parent / "figures"
MU = [10., 20.]
SYSTEMATIC = 0.13
PROBE = {"S7 fine": (4, "number"), "S7 coarse": (4, "number"), "Kerr": (0, "kerr"), "Kerr W=0": (0, "kerr")}
C = ["#2a78d6", "#eb6834", "#1baf7a"]

CACHE = Path(__file__).parent / "figures" / "scans_cache.npz"


# The Kerr-direction W = 0 run (10-24 00:30, lab book Q8) has its own manifest; load_scans leaves it out.
# It used the configuration with f0-g1 57 kHz off (lab book caveat).
KERR_W0 = "261024_003051_CentralBosonAGPDisorderScan.yaml"


def _arrays(scan):
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


def cached_scans():
    """-> {name: dict of arrays}; the SMB load takes ~2 min, so keep a local copy."""
    out = np.load(CACHE, allow_pickle=True)["scans"].item() if CACHE.exists() else {}
    if not out:
        out = {name: _arrays(scan) for name, scan in load_scans().items()}
    if "Kerr W=0" not in out:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["Kerr W=0"] = _arrays(CentralBosonAGPDisorderScan.from_manifest(ROOT / "assembled_data" / KERR_W0))
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, scans=np.array(out, dtype=object))
    return out


scans = cached_scans()
Ws = np.arange(0, 200.1, 1.)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for col, probe in enumerate(["S7", "Kerr"]):
    names = [n for n in scans if n.startswith(probe)]
    pattern, couplings, kerr = (scans[names[0]][k] for k in ("pattern", "couplings", "kerr"))
    g = float(np.mean(couplings))
    mode, kind = PROBE[names[0]]
    for row, mu in enumerate(MU):
        a = ax[row, col]
        for K, c, lab in [(kerr, C[1], f"K = {1e3*kerr:.1f} kHz (recorded)"), (-0.019, C[1], "K = −19 kHz (dressed)"), (0., C[0], "K = 0 (free bosons)")]:
            y = []
            for W in Ws:
                m = fixed_n_hamiltonian(3, 5, 1e-3 * W * pattern, couplings, K)
                y.append(spectral_agp(*spectral_weights(m, probe_values(m.fock_basis, mode, kind)), mu / 1e3, g))
            a.semilogy(Ws, y, color=c, lw=1.8, ls="--" if "dressed" in lab else "-", label=lab)
        for name, mk in zip(names, ["o", "s"]):  # Kerr: 10-24 scan, then the W = 0 run
            scan = scans[name]
            times = scan["times"]
            for i, W_part in enumerate(scan["W"]):
                gp = scan["g"][i]
                data, err = scan["data"][i], scan["err"][i]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = fit_exponential_background(times, data, err, 30.)
                trace = data - fit["amplitude"] * np.exp(-fit["rate_per_us"] * times)
                kern = (2 * np.pi * gp)**2 * agp_kernel_weights(times, mu / 1e3)
                val, noise = kern @ trace, np.sqrt(kern**2 @ err**2)
                mu_ang, gam = 2 * np.pi * mu / 1e3, fit["rate_per_us"]
                dc_term = (2 * np.pi * gp)**2 * fit["amplitude"] * gam / (2 * mu_ang * (mu_ang + gam)**2)
                tot = np.hypot(noise, SYSTEMATIC * dc_term)
                ok = tot < 0.3 * abs(val)
                a.errorbar(W_part, abs(val), yerr=tot, fmt=mk, color="k", mfc="k" if ok else "none",
                           ms=5, capsize=2, lw=1, label=name if i == 0 else None)
        a.set(title=f"{probe} probe, μ = {mu:g} kHz", ylim=(1e-4, 1))
        if row == 1: a.set_xlabel("W (kHz)")
        if col == 0: a.set_ylabel(r"$g^2\|A_\mu\|^2$")
ax[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
ax[0, 1].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.suptitle("Measured AGP (filled: total error < 30 %) vs closed-system model with and without Kerr", fontsize=9)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "kerr_contrast.png", dpi=140)

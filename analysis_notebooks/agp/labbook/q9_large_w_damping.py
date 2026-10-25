"""Q9: what is the extra damping at large W? One global constant per hypothesis, all W > 0 parts at once.

On top of the measured loss and M1 dephasing 0.0106 /us, storage m gets a dephasing
    H0  none
    Hc  gamma_m = c                    (uniform; q3b showed it spoils small W)
    H1  gamma_m = c |delta_m| / 100 kHz   (grows linearly with the storage's own detuning)
    H2  gamma_m = c (delta_m / 100 kHz)^2  (grows quadratically)
and, as an upper bound on what any storage dephasing can do, a free gamma_s per W (uniform).
The same fit on three independent data sets (S7 fine 10-17, S7 coarse 10-17, Kerr 10-24).
If H1 or H2 reaches the per-W bound with one constant, and c agrees between the data sets, the
extra damping is a dephasing that grows with detuning.
"""
import warnings

import numpy as np
from scipy.optimize import least_squares

from load_scans import load_scans
from open_models import LOSS, MODEL
from theory.qsim.agp import probe_values
from theory.qsim.central_boson import fixed_n_hamiltonian
from theory.qsim.open_system import sector_correlation

HYPOTHESES = {
    "Hc uniform": lambda c, d: np.full(4, c),
    "H1 |delta|": lambda c, d: c * np.abs(d) / 0.1,
    "H2 delta^2": lambda c, d: c * (d / 0.1)**2,
}


def part_c1(part, storage_dephasing):
    couplings, kerr = part.recorded_couplings_mhz(), part.recorded_kerr_mhz()
    sector = fixed_n_hamiltonian(3, 5, part.detunings, couplings, kerr).fock_basis
    probe = lambda b: probe_values(b, part.mode_index, part.probes[0][1])
    return sector_correlation(MODEL, sector, probe, part.data["time_us"], couplings, part.detunings,
                              kerr, LOSS, np.r_[0.0106, storage_dephasing])[0]


def residuals(parts, rule, c):
    return np.concatenate([(part_c1(p, rule(c, np.asarray(p.detunings))) - p.data["C1_corrected"])
                           / p.data["C1_err"] for p in parts])


scans = load_scans()
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    parts = [p for p in scan.children if p.disorder_khz > 0]
    per_point = lambda r: np.mean(r**2)
    base = [per_point(residuals([p], HYPOTHESES["Hc uniform"], 0.)) for p in parts]
    print(f"== {name}: W = {[p.disorder_khz for p in parts]}")
    print(f"   H0 none          chi2/pt per W {np.round(base, 1)}  total {np.mean(base):.2f}")
    for label, rule in HYPOTHESES.items():
        res = least_squares(lambda x: residuals(parts, rule, x[0]), x0=[0.005], bounds=([0], [1]),
                            diff_step=1e-3)
        each = [per_point(residuals([p], rule, res.x[0])) for p in parts]
        error = np.sqrt(np.linalg.pinv(res.jac.T @ res.jac)[0, 0] * np.mean(res.fun**2))
        print(f"   {label:16s} c = {res.x[0]:.4f} +/- {error:.4f} /us  chi2/pt per W {np.round(each, 1)}"
              f"  total {np.mean(each):.2f}")
    free = []
    for p in parts:
        res = least_squares(lambda x: residuals([p], HYPOTHESES["Hc uniform"], x[0]), x0=[0.005],
                            bounds=([0], [1]), diff_step=1e-3)
        free.append((res.x[0], per_point(res.fun)))
    print(f"   per-W uniform    gamma_s {np.round([f[0] for f in free], 4)}  chi2/pt per W "
          f"{np.round([f[1] for f in free], 1)}  total {np.mean([f[1] for f in free]):.2f}")

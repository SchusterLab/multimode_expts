"""Q11: does the connected correlator make S7 at large W measurable? Equal measurement time.

Scheme A (now): the 15 starts with n_7 >= 1, with 35/15 times the shots (same total time as B).
Scheme B: all 35 starts; m(t) = (1/D) sum_n <n|n_7(t)|n> and C_c = C_1 - Vbar m(t), Vbar = Tr V / D.
In the closed system m(t) = Vbar, so C_c = C_1 - Vbar^2: same AGP, smaller DC. With loss, m(t) also
removes the decay of that trivial part.
Noise model: the same error sigma per start and time in both schemes before the time scaling,
sigma = D C1_err / sqrt(sum V_n^2) from the measured C1_err of each part. Open model (Q3), 40 draws.
Estimators: exp fit and envelope + tail (Q5).
"""
import warnings

import numpy as np

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import load_scans
from open_models import LOSS, MODEL
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights
from theory.qsim.central_boson import fixed_n_hamiltonian

MU_KHZ = np.array([5., 10., 20., 40.])
DRAWS = 40
RNG = np.random.default_rng(11)


def estimates(times, trace, error, g):
    """-> {method: array over MU_KHZ}."""
    scale = (2 * np.pi * g)**2
    fit = fit_exponential_background(times, trace, error, 30.)
    tau = fit["lifetime_us"]
    background = fit["amplitude"] * np.exp(-times / tau)
    level = np.mean((trace * np.exp(times / tau))[times >= 0.9 * times[-1]])
    out = {"exp fit": [], "envelope+tail": []}
    for mu in MU_KHZ:
        a = 2 * np.pi * mu / 1e3
        out["exp fit"].append(scale * agp_kernel_weights(times, mu / 1e3) @ (trace - background))
        out["envelope+tail"].append(scale * (agp_kernel_weights(times, mu / 1e3, tau) @ trace
                                             - level * times[-1] * np.exp(-a * times[-1]) / (2 * a)))
    return {k: np.array(v) for k, v in out.items()}


scans = load_scans(kerr=False)
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    times = scan.data["time_us"]
    print(f"== {name}: median estimate/target [16-84 %], mu = {MU_KHZ.tolist()} kHz")
    for row, part in enumerate(scan.children):
        if part.disorder_khz < 60:
            continue
        couplings, kerr = part.recorded_couplings_mhz(), part.recorded_kerr_mhz()
        closed = fixed_n_hamiltonian(3, 5, part.detunings, couplings, kerr)
        values = probe_values(closed.fock_basis, 4)
        D, vbar = len(values), values.mean()
        g = float(np.mean(couplings))
        target = spectral_agp(*spectral_weights(closed, values), MU_KHZ / 1e3, g)
        starts = [tuple(s) for s in closed.fock_basis]
        traces = MODEL.populations(starts, times, couplings, part.detunings, kerr, LOSS,
                                   np.r_[0.0106, np.zeros(4)]) @ probe_values(MODEL.numbers, 4)
        c1 = values @ traces / D
        cc = (values - vbar) @ traces / D
        sigma = D * scan.data["C1_err"][row] / np.sqrt(np.sum(values**2))
        err_a = np.sqrt(np.sum(values**2)) * sigma / D / np.sqrt(35 / 15)
        err_b = np.sqrt(np.sum((values - vbar)**2)) * sigma / D
        result = {}
        for scheme, clean, err in (("A: 15 starts, 2.3x shots", c1, err_a), ("B: 35 starts, connected", cc, err_b)):
            draws = {"exp fit": [], "envelope+tail": []}
            for _ in range(DRAWS):
                for k, v in estimates(times, clean + RNG.normal(0, err), err, g).items():
                    draws[k].append(v / target)
            result[scheme] = {k: np.percentile(v, [16, 50, 84], axis=0) for k, v in draws.items()}
        dc_closed = np.sum(np.abs(np.asarray(closed.states).T @ (values[:, None] * np.asarray(closed.states)))
                           .diagonal()**2) / D
        print(f"  W={part.disorder_khz:4.0f}  closed DC: C_1 {dc_closed:.2f}, C_c {dc_closed - vbar**2:.2f}")
        for scheme, methods in result.items():
            for method, (lo, mid, hi) in methods.items():
                print(f"     {scheme:26s} {method:14s} " + " ".join(f"{b:6.2f}[{a:5.2f},{c:5.2f}]" for a, b, c in zip(lo, mid, hi)))

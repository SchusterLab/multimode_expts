"""Q10: why does the background subtraction leave 13 %? Exact split of the open-model C_1.

C_1 = C_DC + C_osc in the open model: C_DC starts from the eigenbasis-diagonal part of each prepared
state (the time average of the closed system, coherences kept inside degenerate blocks), C_osc from
the rest. Then, per part and mu:
  DC term          (2 pi g)^2 K @ C_DC                 (what must be removed)
  shape error 1    K @ (C_DC - best single exp fitted to C_DC itself, t >= 30 us)
  shape error 2    K @ (C_DC - best double exp fitted to C_DC itself, t >= 30 us)
  fit error        K @ (C_DC - single exp fitted to the full C_1, t >= 30 us)  (what exp fit does)
  osc bias         K @ C_osc - target                  (line broadening; no background method fixes it)
All relative to the closed-system target. No noise (pure model).
"""
import warnings

import numpy as np
from scipy.optimize import curve_fit

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import load_scans
from open_models import LOSS, MODEL
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights
from theory.qsim.central_boson import fixed_n_hamiltonian

MU_KHZ = [5., 10., 20.]
START_US = 30.


def dc_split(part):
    """-> (times, C_DC, C_full, target per mu, g) for one part, open model loss + M1 deph 0.0106."""
    times = part.data["time_us"]
    couplings, kerr = part.recorded_couplings_mhz(), part.recorded_kerr_mhz()
    closed = fixed_n_hamiltonian(3, 5, part.detunings, couplings, kerr)
    probe = lambda b: probe_values(b, part.mode_index, part.probes[0][1])
    values = probe(closed.fock_basis)
    rows = np.array([MODEL.index[tuple(s)] for s in closed.fock_basis])
    states, energies = np.asarray(closed.states), np.asarray(closed.energies_MHz)
    same = np.isclose(energies[:, None], energies[None, :], atol=1e-9)  # degenerate blocks
    starts = np.where(values != 0)[0]
    full = np.zeros((len(starts), MODEL.dim, MODEL.dim))
    dc = np.zeros_like(full)
    for k, n in enumerate(starts):
        amplitudes = states[n]  # <n|E_i>
        rho_eigen = np.outer(amplitudes, amplitudes) * same  # time-averaged, in the eigenbasis
        sector_rho = states @ rho_eigen @ states.T
        dc[k][np.ix_(rows, rows)] = sector_rho
        full[k][rows[n], rows[n]] = 1
    dephasing = np.r_[0.0106, np.zeros(4)]
    observable = probe(MODEL.numbers)
    weights = values[starts] / len(values)
    traces_dc = MODEL.populations_from_density(dc, times, couplings, part.detunings, kerr, LOSS, dephasing) @ observable
    traces_full = MODEL.populations_from_density(full, times, couplings, part.detunings, kerr, LOSS, dephasing) @ observable
    g = float(np.mean(couplings))
    target = spectral_agp(*spectral_weights(closed, values), np.array(MU_KHZ) / 1e3, g)
    return times, weights @ traces_dc, weights @ traces_full, target, g


def double_exp(times, values, start):
    keep = times >= start
    single = fit_exponential_background(times, values, None, start)
    a, r = single["amplitude"], single["rate_per_us"]
    model = lambda t, a1, r1, a2, r2: a1 * np.exp(-r1 * t) + a2 * np.exp(-r2 * t)
    p, _ = curve_fit(model, times[keep], values[keep], p0=(a / 2, 2 * r, a / 2, r / 2), maxfev=50000)
    return model(times, *p)


scans = load_scans()
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    print(f"== {name}: rows mu = {MU_KHZ} kHz; all / target")
    print("   W   | DC term | shape err 1exp | shape err 2exp | exp-fit err (full trace) | osc bias")
    for part in scan.children:
        if part.disorder_khz == 0:
            continue  # W = 0 is degenerate and has the residual detunings: see Q3b
        times, c_dc, c_full, target, g = dc_split(part)
        scale = (2 * np.pi * g)**2
        kernel = np.array([scale * agp_kernel_weights(times, m / 1e3) for m in MU_KHZ])
        single_dc = fit_exponential_background(times, c_dc, None, START_US)
        exp1 = single_dc["amplitude"] * np.exp(-single_dc["rate_per_us"] * times)
        exp2 = double_exp(times, c_dc, START_US)
        fit_full = fit_exponential_background(times, c_full, None, START_US)
        bg_full = fit_full["amplitude"] * np.exp(-fit_full["rate_per_us"] * times)
        cols = [kernel @ c_dc, kernel @ (c_dc - exp1), kernel @ (c_dc - exp2), kernel @ (c_dc - bg_full),
                kernel @ (c_full - c_dc) - target]
        fmt = lambda v: " ".join(f"{x / t:7.3f}" for x, t in zip(v, target))
        print(f" {part.disorder_khz:5.0f} | " + " | ".join(fmt(v) for v in cols)
              + f"   [tau DC {1 / single_dc['rate_per_us']:.0f} us, tau full {fit_full['lifetime_us']:.0f} us]")

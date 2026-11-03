"""Q2: the short-time sum rule. C(t) = C0 - M2 t^2 / 2 + M4 t^4 / 24 - ...

M2 = (1/D) Tr([H,V]^dag [H,V]) = (1/D) sum_ab |H_ab|^2 (V_a - V_b)^2 (Fock basis, V diagonal):
only the hopping terms enter, so M2 does not depend on W or K. For V = n_S7 only g_7 enters.
Fit C0 + a t^2 + b t^4 + c t^6 to the first points of each measured trace; M2 = -2 a.
The fit window is chosen on the model: the same fit to the closed-system trace must return M2
within 2 % at every W.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from load_scans import VAULT, load_scans
from theory.qsim.agp import probe_values, two_time_correlation
from theory.qsim.central_boson import fixed_n_hamiltonian


def exact_m2(part):
    """-> M2 in (rad/us)^2 from the recorded couplings, and its derivative structure."""
    model = fixed_n_hamiltonian(part.photon_sector_number, 1 + len(part.swap_stors), part.detunings,
                                part.recorded_couplings_mhz(), part.recorded_kerr_mhz())
    v = probe_values(model.fock_basis, part.mode_index, part.probes[0][1])
    h = np.asarray(model.hamiltonian_MHz) * 2 * np.pi
    return float(np.sum(np.abs(h)**2 * (v[:, None] - v[None, :])**2) / len(v))


def fit_m2(times, values, errors, t_max, order=3):
    """-> (M2, sigma) from a weighted fit in powers of t^2 up to t^(2 order), t <= t_max."""
    keep = times <= t_max
    x = times[keep]**2
    design = np.column_stack([x**k for k in range(order + 1)])
    w = 1 / errors[keep]
    coef, *_ = np.linalg.lstsq(design * w[:, None], values[keep] * w, rcond=None)
    cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
    return -2 * coef[1], 2 * np.sqrt(cov[1, 1])


scans = load_scans()
results = {}
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    times = scan.data["time_us"]
    if times[1] > 1:  # the coarse grid has too few early points
        continue
    # Choose t_max on the model: the largest window that recovers M2 within 2 % at every W.
    windows = np.arange(2., 12.01, .5)
    good = []
    for t_max in windows:
        ok = True
        for part in scan.children:
            diff, w = part.theory_spectrum()
            fine = np.arange(0, t_max + 1e-9, times[1])
            theory = two_time_correlation(diff, w, fine)
            m2, _ = fit_m2(fine, theory, np.full(len(fine), 0.01), t_max)
            ok &= abs(m2 / exact_m2(part) - 1) < 0.02
        good.append(ok)
    t_max = windows[np.where(good)[0].max()]
    rows = []
    for row, part in enumerate(scan.children):
        m2_exact = exact_m2(part)
        m2, sigma = fit_m2(times, scan.data["C1_corrected"][row], scan.data["C1_err"][row], t_max)
        rows.append((part.disorder_khz, m2 / m2_exact, sigma / m2_exact))
    results[name] = (t_max, np.array(rows))
    print(f"{name}: t_max = {t_max} us ({int(t_max / times[1])} points)")
    for strength, ratio, err in rows:
        print(f"   W = {strength:5.0f}: M2 / M2(recorded g) = {ratio:.3f} +/- {err:.3f}")
    r = np.array(rows)
    mean = np.sum(r[:, 1] / r[:, 2]**2) / np.sum(1 / r[:, 2]**2)
    err = 1 / np.sqrt(np.sum(1 / r[:, 2]**2))
    chi2 = np.sum((r[:, 1] - mean)**2 / r[:, 2]**2) / (len(r) - 1)
    print(f"   weighted mean {mean:.3f} +/- {err:.3f} (chi2/dof {chi2:.2f})"
          f" -> g_eff/g_saved = {np.sqrt(mean):.3f} +/- {err / 2 / np.sqrt(mean):.3f}")

fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
for name, (t_max, r) in results.items():
    ax.errorbar(r[:, 0] + (name == "Kerr") * 2, r[:, 1], r[:, 2], fmt="o", capsize=3,
                label=f"{name} (fit t <= {t_max:g} us)")
ax.axhline(1, color="k", lw=.8, label="recorded g")
ax.axhline(1.04**2, color="gray", ls="--", lw=.8, label="(1.04)^2: guan 10-24 MBR, N = 3")
ax.set(xlabel="W (kHz)", ylabel=r"$M_2^{meas}/M_2^{rec} = (g_{eff}/g_{saved})^2$",
       title="Q2 [measured vs exact identity]: short-time curvature of C_1")
ax.legend(fontsize=7)
fig.savefig(VAULT / "q2_sum_rule.png", dpi=120)

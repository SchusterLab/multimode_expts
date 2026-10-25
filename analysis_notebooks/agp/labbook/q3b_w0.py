"""Q3b: why does C_1 decay fast at W = 0? Two hypotheses, fitted to the W = 0 parts only.

(a) static residual detunings delta (4 free, kHz): they lift the degeneracy of the dark-mode states.
(b) storage dephasing gamma_s (1 free, 1/us, the same on all storages).
Both on top of the measured loss and M1 dephasing 0.0106 /us. Then the fitted delta of the fine scan
(10-17..19 data) is added to the W = 30 and 40 kHz parts of the same scan, as a prediction.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fitting.qsim.agp import fit_exponential_background
from load_scans import VAULT, load_scans
from open_models import LOSS, MODEL, part_model
from theory.qsim.agp import probe_values
from theory.qsim.central_boson import fixed_n_hamiltonian
from theory.qsim.open_system import sector_correlation

M1_DEPH = 0.0106


def model_c1(part, extra_detunings_mhz=0., storage_dephasing=0.):
    times = part.data["time_us"]
    couplings, kerr = part.recorded_couplings_mhz(), part.recorded_kerr_mhz()
    detunings = np.asarray(part.detunings) + extra_detunings_mhz
    sector = fixed_n_hamiltonian(3, 5, detunings, couplings, kerr).fock_basis
    probe = lambda basis: probe_values(basis, part.mode_index, part.probes[0][1])
    dephasing = np.r_[M1_DEPH, np.full(4, storage_dephasing)]
    return sector_correlation(MODEL, sector, probe, times, couplings, detunings, kerr, LOSS, dephasing)[0]


scans = load_scans(kerr=False)
for scan in scans.values():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
fits = {}
for name, scan in scans.items():
    part = scan.children[0]
    assert part.disorder_khz == 0
    data, error = part.data["C1_corrected"], part.data["C1_err"]
    res_a = least_squares(lambda d: (model_c1(part, 1e-3 * d) - data) / error, x0=[2., -2., 1., -1.],
                          diff_step=1e-3)
    res_b = least_squares(lambda r: (model_c1(part, 0., r[0]) - data) / error, x0=[0.005],
                          bounds=([0], [1]), diff_step=1e-3)
    chi_a, chi_b = np.mean(res_a.fun**2), np.mean(res_b.fun**2)
    cov = np.linalg.pinv(res_a.jac.T @ res_a.jac) * chi_a
    fits[name] = dict(delta=res_a.x, delta_err=np.sqrt(np.diag(cov)), chi_a=chi_a, gamma=res_b.x[0], chi_b=chi_b)
    print(f"{name} W=0: (a) delta = {np.round(res_a.x, 2)} +/- {np.round(np.sqrt(np.diag(cov)), 2)} kHz "
          f"(RMS {np.sqrt(np.mean(res_a.x**2)):.2f}), chi2/pt {chi_a:.2f} | "
          f"(b) gamma_s = {res_b.x[0]:.4f} /us, chi2/pt {chi_b:.2f}")
    fits[name]["curves"] = (part.data["time_us"], data, error, model_c1(part), model_c1(part, 1e-3 * res_a.x),
                            model_c1(part, 0., res_b.x[0]))

# Prediction: the fine-scan delta added to its W = 30, 40 parts (same days); and gamma_s likewise.
fine = scans["S7 fine"]
print("prediction on the fine scan's W > 0 parts (chi2/pt: none / + delta / + gamma_s):")
for part in fine.children[1:]:
    data, error = part.data["C1_corrected"], part.data["C1_err"]
    base = model_c1(part)
    with_delta = model_c1(part, 1e-3 * fits["S7 fine"]["delta"])
    with_gamma = model_c1(part, 0., fits["S7 fine"]["gamma"])
    chis = [np.mean(((c - data) / error)**2) for c in (base, with_delta, with_gamma)]
    lifes = [fit_exponential_background(part.data["time_us"], c, error, 30.)["lifetime_us"]
             for c in (data, base, with_delta, with_gamma)]
    print(f"   W={part.disorder_khz:4.0f}: chi2/pt {chis[0]:5.1f} / {chis[1]:5.1f} / {chis[2]:5.1f};"
          f" 1/rate data {lifes[0]:.0f}, model {lifes[1]:.0f} / {lifes[2]:.0f} / {lifes[3]:.0f} us")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
for ax, (name, fit) in zip(axes, fits.items()):
    t, data, error, base, a, b = fit["curves"]
    ax.fill_between(t, data - error, data + error, color="0.6", alpha=.5, lw=0, label="measured C_1")
    ax.plot(t, base, "C1", lw=1, label="loss + M1 dephasing (no free parameter)")
    ax.plot(t, a, "C0", lw=1.3, label=f"+ static detunings, RMS {np.sqrt(np.mean(fit['delta']**2)):.1f} kHz (chi2/pt {fit['chi_a']:.1f})")
    ax.plot(t, b, "C2--", lw=1, label=f"+ storage dephasing {fit['gamma']:.4f}/us (chi2/pt {fit['chi_b']:.1f})")
    ax.set(title=f"{name}, W = 0", xlabel="time (us)", ylabel="C_1")
    ax.legend(fontsize=7)
fig.suptitle("Q3b [model vs measured]: the fast W = 0 decay")
fig.savefig(VAULT / "q3b_w0.png", dpi=120)

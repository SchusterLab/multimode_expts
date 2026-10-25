"""Q8: is the W = 0 residual-detuning explanation consistent? The Kerr-direction W = 0 run (10-24 00:30).

Same fits as q3b_w0.py on an independent run with another probe: (a) four static detunings,
(b) one storage dephasing rate, on top of the measured loss and M1 dephasing 0.0106 /us.
With equal couplings the Kerr probe is symmetric in the storages: only the set of detunings is
fixed, not which storage has which. Several starts guard against local minima.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from experiments.qsim.agp import CentralBosonAGPDisorderScan
from load_scans import ROOT, VAULT
from open_models import LOSS, MODEL
from theory.qsim.agp import probe_values
from theory.qsim.central_boson import fixed_n_hamiltonian
from theory.qsim.open_system import sector_correlation

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scan = CentralBosonAGPDisorderScan.from_manifest(ROOT / "assembled_data" / "261024_003051_CentralBosonAGPDisorderScan.yaml")
    scan.analyze(mu=[10.])
part = scan.children[0]
times, data, error = part.data["time_us"], part.data["C1_corrected"], part.data["C1_err"]
couplings, kerr = part.recorded_couplings_mhz(), part.recorded_kerr_mhz()
print(f"Kerr W=0: {len(times)} points to {times[-1]:.0f} us, couplings {1e3 * couplings} kHz, K {1e3 * kerr:.1f} kHz")


def model_c1(delta_khz=np.zeros(4), storage_dephasing=0.):
    detunings = 1e-3 * np.asarray(delta_khz)
    sector = fixed_n_hamiltonian(3, 5, detunings, couplings, kerr).fock_basis
    dephasing = np.r_[0.0106, np.full(4, storage_dephasing)]
    return sector_correlation(MODEL, sector, lambda b: probe_values(b, 0, "kerr"), times, couplings,
                              detunings, kerr, LOSS, dephasing)[0]


base = model_c1()
print(f"no extra term: chi2/pt {np.mean(((base - data) / error)**2):.1f}")
best = None
for start in ([2., -2., 1., -1.], [4., 1., -3., 0.5], [-1., 3., 2., -4.], [1., 1., -1., -1.]):
    res = least_squares(lambda d: (model_c1(d) - data) / error, x0=start, diff_step=1e-3)
    chi = np.mean(res.fun**2)
    print(f"  start {start}: delta = {np.round(res.x, 2)} kHz, chi2/pt {chi:.2f}")
    if best is None or chi < best[1]:
        best = (res.x, chi)
res_b = least_squares(lambda r: (model_c1(storage_dephasing=r[0]) - data) / error, x0=[0.005],
                      bounds=([0], [1]), diff_step=1e-3)
chi_b = np.mean(res_b.fun**2)
delta = best[0]
print(f"(a) best: sorted delta = {np.sort(delta).round(2)} kHz (RMS {np.sqrt(np.mean(delta**2)):.2f}, "
      f"mean {np.mean(delta):.2f}), chi2/pt {best[1]:.2f}")
print(f"(b) storage dephasing {res_b.x[0]:.4f} /us, chi2/pt {chi_b:.2f}")

fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
ax.fill_between(times, data - error, data + error, color="0.6", alpha=.5, lw=0, label="measured C_1 (Kerr probe)")
ax.plot(times, base, "C1", lw=1, label="loss + M1 dephasing (no free parameter)")
ax.plot(times, model_c1(delta), "C0", lw=1.3,
        label=f"+ static detunings, RMS {np.sqrt(np.mean(delta**2)):.1f} kHz (chi2/pt {best[1]:.1f})")
ax.plot(times, model_c1(storage_dephasing=res_b.x[0]), "C2--", lw=1,
        label=f"+ storage dephasing {res_b.x[0]:.4f}/us (chi2/pt {chi_b:.1f})")
ax.set(xlabel="time (us)", ylabel="C_1", title="Q8 [model vs measured]: Kerr direction, W = 0 (10-24 00:30)")
ax.legend(fontsize=7)
fig.savefig(VAULT / "q8_w0_kerr.png", dpi=120)

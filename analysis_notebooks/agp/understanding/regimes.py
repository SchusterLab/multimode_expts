import numpy as np
from theory.qsim.agp import probe_values, spectral_weights, spectral_agp, split_dc
from theory.qsim.central_boson import fixed_n_hamiltonian
g = 0.02036; K = -0.0271; p = np.array([-0.71, 0.19, -1.03, 1.55])
PROBES = {"n_S7": (4, "number"), "Kerr": (0, "kerr")}
def spec(W_khz, kerr, extra=None):
    det = 1e-3 * W_khz * p + (0 if extra is None else extra)
    m = fixed_n_hamiltonian(3, 5, det, [g]*4, kerr)
    out = {}
    for name, (mode, kind) in PROBES.items():
        d, w = spectral_weights(m, probe_values(m.fock_basis, mode, kind))
        out[name] = (d, w)
    return m, out
def lines(d, w, tol=1e-9):
    # merge into distinct positive frequencies
    sel = (d > 1e-9) & (w > tol)
    f = np.round(d[sel]*1e6, 1)  # Hz
    u, inv = np.unique(f, return_inverse=True)
    return u/1e3, np.bincount(inv, weights=2*w[sel])  # kHz, weight (both signs)
# single-particle frequencies for K=0, W=60
for W in [60]:
    h = np.zeros((5,5)); h[0,1:] = h[1:,0] = g; h[np.arange(1,5), np.arange(1,5)] = -1e-3*W*p
    e = np.linalg.eigvalsh(h)*1e3
    print("single-particle levels (kHz), K=0 W=60:", e.round(2))
    print("distinct |e_k-e_l|:", np.unique(np.round(np.abs(e[:,None]-e[None,:])[np.triu_indices(5,1)],2)))
for label, W, kerr in [("linear K=0 W=60", 60, 0.), ("superint W=0 K=-27", 0, K), ("island W=60 K=-27", 60, K)]:
    m, out = spec(W, kerr)
    print("\n==", label, " mean spacing (kHz):", round(1e3*np.mean(np.diff(m.energies_MHz)), 1))
    for name, (d, w) in out.items():
        dc, _ = split_dc(d, w, 1e-9)
        f, ww = lines(d, w)
        print(f"  {name}: C0={w.sum():.3f} DC frac={dc/w.sum():.3f}  #lines(w>1e-9)={len(f)}  #lines(>1e-4 C0)={np.sum(ww>1e-4*w.sum())}")
        if len(f) < 15: print("     lines kHz:", f.round(2), " w:", ww.round(4))
# W=0: how much DC weight is fragile (moves to finite freq for W->0+)?
for name in PROBES:
    _, o0 = spec(0, K); _, o1 = spec(1e-4, K)
    dc0 = split_dc(*o0[name], 1e-9)[0]; dc1 = split_dc(*o1[name], 1e-12)[0]
    print(f"W=0 {name}: DC at W=0 {dc0:.4f}, DC at W->0+ {dc1:.4f}: fragile weight {dc0-dc1:.4f} of C0={o0[name][1].sum():.3f}")

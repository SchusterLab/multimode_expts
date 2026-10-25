"""Shared for Q3-Q5: the open-system C_1 of a scan part under named damping models (cached)."""
from pathlib import Path

import numpy as np

from theory.qsim.agp import probe_values
from theory.qsim.central_boson import fixed_n_hamiltonian
from theory.qsim.open_system import OpenCentralBoson, sector_correlation

CACHE = Path(__file__).parent / "cache"
# T1 (us): M1 215 (10-21 report, |2> parity), S4 1239 (10-21), S7 635 (jonginn 10-19); S5, S6 not
# measured: 1000 us assumed. Dephasing: D[sqrt(gamma) n_M1]; 0.0106 = 2 (1/T2* - 1/2T1) idle Ramsey
# 10-21; 0.015 = guan's 10-24 N=1 walk fit (Floquet frame).
LOSS = np.array([1 / 215, 1 / 1239, 1 / 1000, 1 / 1000, 1 / 635])
DAMPING = {
    "closed": (0 * LOSS, 0.),
    "loss": (LOSS, 0.),
    "loss + M1 deph 0.0106": (LOSS, 0.0106),
    "loss + M1 deph 0.015": (LOSS, 0.015),
}
HAMILTONIANS = {"recorded": (1.0, None), "dressed": (1.04, -0.019)}  # g scale, K (MHz)
MODEL = OpenCentralBoson(3, 4)


def part_model(part, damping, hamiltonian="recorded", times=None):
    """-> (C_1(t), traces) of one CentralBosonAGPExperiment under a named damping and Hamiltonian."""
    times = np.asarray(part.data["time_us"] if times is None else times)
    loss, dephasing = DAMPING[damping] if isinstance(damping, str) else damping
    scale, kerr = HAMILTONIANS[hamiltonian]
    kerr = part.recorded_kerr_mhz() if kerr is None else kerr
    couplings = scale * part.recorded_couplings_mhz()
    key = (f"{part.direction}_{part.disorder_khz:g}_{len(times)}_{times[-1]:.3f}_{damping}_{hamiltonian}"
           .replace(" ", "").replace("+", "p"))
    path = CACHE / f"{key}.npz"
    if isinstance(damping, str) and path.exists():
        cached = np.load(path)
        return cached["c"], cached["traces"]
    sector = fixed_n_hamiltonian(part.photon_sector_number, 1 + len(part.swap_stors), part.detunings,
                                 couplings, kerr).fock_basis
    probe = lambda basis: probe_values(basis, part.mode_index, part.probes[0][1])
    dephasings = np.zeros(MODEL.mode_count)
    dephasings[0] = dephasing
    c, traces = sector_correlation(MODEL, sector, probe, times, couplings, part.detunings, kerr,
                                   loss, dephasings)
    if isinstance(damping, str):
        CACHE.mkdir(exist_ok=True)
        np.savez(path, c=c, traces=traces)
    return c, traces

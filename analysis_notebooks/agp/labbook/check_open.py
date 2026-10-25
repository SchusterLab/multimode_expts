import time
import numpy as np
from theory.qsim.open_system import OpenCentralBoson, sector_correlation
from theory.qsim.agp import probe_values, spectral_weights, two_time_correlation
from theory.qsim.central_boson import fixed_n_hamiltonian
g = [0.0204]*4; det = [0.03*x for x in [-0.7, 0.19, -1.03, 1.55]]; K = -0.027
t0 = time.time(); m = OpenCentralBoson(3, 4); print("build", time.time()-t0, m.dim, len(m.keep))
times = np.arange(400) * 0.30692
model = fixed_n_hamiltonian(3, 5, det, g, K)
probe = lambda b: probe_values(b, 4)
t0 = time.time()
c, tr = sector_correlation(m, model.fock_basis, probe, times, g, det, K)
print("closed eval", time.time()-t0)
ref = two_time_correlation(*spectral_weights(model, probe(model.fock_basis)), times)
print("max diff closed", np.abs(c - ref).max())
t0 = time.time()
c2, _ = sector_correlation(m, model.fock_basis, probe, times, g, det, K, [1/215, 1/1240, 1/1000, 1/1000, 1/635], [0.0106, 0, 0, 0, 0])
print("open eval", time.time()-t0, c2[[0, 100, 399]])

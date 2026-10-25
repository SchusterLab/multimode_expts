"""The central-boson model with loss and dephasing (Lindblad), for Fock-state starts.

Same Hamiltonian and conventions as ``theory.qsim.central_boson`` (cyclic MHz,
onsite storage energy = -detuning, Kerr (K/2) n(n-1), mode 0 = M1), on all
states with at most ``max_photons`` photons, so that loss can move a state to
lower photon numbers. Per mode m:

    loss       D[sqrt(kappa_m) a_m],   kappa_m = 1/T1_m
    dephasing  D[sqrt(gamma_m) n_m],   a 0-1 coherence decays at gamma_m / 2

Rates are in 1/us. Only the blocks of rho with equal photon numbers on both
sides are kept: H and both dissipators keep them closed, and a Fock start
lives there. The populations of the basis states are returned, so any
observable diagonal in the Fock basis is one dot product away.

From guan's ``WalkModel`` (``measurement_notebooks/guan/kerr_n123_consistency.py``,
2026-10-24), generalized to rates per mode, on 2026-10-25.
"""
from itertools import product

import numpy as np
from scipy import sparse
from scipy.linalg import expm


class OpenCentralBoson:
    """Fock basis with at most ``max_photons`` photons in M1 and ``storage_count`` storages."""

    def __init__(self, max_photons, storage_count):
        self.mode_count = 1 + storage_count
        self.states = [state for state in product(range(max_photons + 1), repeat=self.mode_count)
                       if sum(state) <= max_photons]
        self.index = {state: k for k, state in enumerate(self.states)}
        self.dim = len(self.states)
        self.numbers = np.asarray(self.states, dtype=float)
        self.lowering = []
        for mode in range(self.mode_count):
            operator = np.zeros((self.dim, self.dim))
            for k, state in enumerate(self.states):
                if state[mode]:
                    lower = list(state)
                    lower[mode] -= 1
                    operator[self.index[tuple(lower)], k] = np.sqrt(state[mode])
            self.lowering.append(operator)
        totals = self.numbers.sum(axis=1)
        # vec(rho) column-stacked: element (i, j) at j * dim + i.
        self.keep = np.array([j * self.dim + i for j in range(self.dim) for i in range(self.dim)
                              if totals[i] == totals[j]])
        position = {value: p for p, value in enumerate(self.keep)}
        self.diagonal = np.array([position[i * self.dim + i] for i in range(self.dim)])

    def hamiltonian_MHz(self, couplings_MHz, detunings, kerr_MHz):
        """-> H (cyclic MHz) in this basis, as ``fixed_n_hamiltonian`` builds it per sector."""
        n_hub = self.numbers[:, 0]
        diagonal = 0.5 * kerr_MHz * n_hub * (n_hub - 1) - self.numbers[:, 1:] @ np.asarray(detunings, float)
        h = np.diag(diagonal)
        for mode, coupling in enumerate(couplings_MHz, start=1):
            hop = self.lowering[0].T @ self.lowering[mode]
            h = h + coupling * (hop + hop.T)
        return h

    def liouvillian(self, couplings_MHz, detunings, kerr_MHz, loss_per_us=0., dephasing_per_us=0.):
        """-> L (rad/us) on the kept blocks; ``loss_per_us``, ``dephasing_per_us``: scalar or per mode."""
        h = sparse.csr_matrix(2 * np.pi * self.hamiltonian_MHz(couplings_MHz, detunings, kerr_MHz))
        eye = sparse.identity(self.dim, format="csr")
        kron = sparse.kron
        full = -1j * (kron(eye, h) - kron(h.T, eye))  # vec(A rho B) = (B^T kron A) vec(rho)
        losses = np.broadcast_to(np.asarray(loss_per_us, float), (self.mode_count,))
        dephasings = np.broadcast_to(np.asarray(dephasing_per_us, float), (self.mode_count,))
        for mode in range(self.mode_count):
            for rate, jump in ((losses[mode], sparse.csr_matrix(self.lowering[mode])),
                               (dephasings[mode], sparse.diags(self.numbers[:, mode]).tocsr())):
                if rate:
                    product_ = jump.T @ jump
                    full = full + rate * (kron(jump, jump) - 0.5 * kron(eye, product_)
                                          - 0.5 * kron(product_.T, eye))
        return sparse.csr_matrix(full)[self.keep][:, self.keep].toarray()

    def populations(self, occupations, time_us, couplings_MHz, detunings, kerr_MHz,
                    loss_per_us=0., dephasing_per_us=0.):
        """-> basis-state populations (starts, times, basis) for Fock starts ``occupations``.

        ``time_us`` must be evenly spaced from 0 (the measured Floquet grids are).
        """
        times = np.asarray(time_us, dtype=float)
        steps = np.diff(times)
        if times[0] != 0 or (len(times) > 1 and not np.allclose(steps, steps[0])):
            raise ValueError("time_us must start at 0 and be evenly spaced")
        generator = self.liouvillian(couplings_MHz, detunings, kerr_MHz, loss_per_us, dephasing_per_us)
        step = expm(generator * steps[0]) if len(times) > 1 else None
        vectors = np.zeros((len(self.keep), len(occupations)), dtype=complex)
        for column, occupation in enumerate(occupations):
            vectors[self.diagonal[self.index[tuple(occupation)]], column] = 1
        out = np.empty((len(occupations), len(times), self.dim))
        for t in range(len(times)):
            out[:, t] = np.real(vectors[self.diagonal]).T
            if step is not None and t + 1 < len(times):
                vectors = step @ vectors
        return out


def sector_correlation(model, sector_states, probe, time_us, couplings_MHz, detunings, kerr_MHz,
                       loss_per_us=0., dephasing_per_us=0.):
    """-> (C(t), traces): C(t) = (1/D) sum_n V_n <n|V(t)|n> over the fixed-N ``sector_states``.

    ``probe`` maps a (basis, modes) occupation array to the diagonal of V
    (e.g. ``lambda b: theory.qsim.agp.probe_values(b, 4)``). Starts with V_n = 0
    are skipped, as in the experiment. ``traces`` (starts, times) are <n|V(t)|n>.
    """
    sector = np.asarray(sector_states, dtype=float)
    start_values = probe(sector)
    measured = [tuple(int(x) for x in state) for state, value in zip(sector, start_values) if value]
    pops = model.populations(measured, time_us, couplings_MHz, detunings, kerr_MHz,
                             loss_per_us, dephasing_per_us)
    traces = pops @ probe(model.numbers)
    weights = start_values[start_values != 0] / len(sector)
    return weights @ traces, traces

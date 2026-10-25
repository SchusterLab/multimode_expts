"""theory.qsim.open_system: the Lindblad central boson against the closed system and simple limits."""
import numpy as np

from theory.qsim.agp import probe_values, spectral_weights, two_time_correlation
from theory.qsim.central_boson import fixed_n_hamiltonian
from theory.qsim.open_system import OpenCentralBoson, sector_correlation

COUPLINGS, DETUNINGS, KERR = [.02, .021], [.03, -.01], -.027


def test_hamiltonian_blocks_match_fixed_n_hamiltonian():
    model = OpenCentralBoson(2, 2)
    h = model.hamiltonian_MHz(COUPLINGS, DETUNINGS, KERR)
    reference = fixed_n_hamiltonian(2, 3, DETUNINGS, COUPLINGS, KERR)
    rows = [model.index[tuple(state)] for state in reference.fock_basis]
    np.testing.assert_allclose(h[np.ix_(rows, rows)], reference.hamiltonian_MHz, atol=1e-15)


def test_closed_correlation_matches_the_spectral_sum():
    model = OpenCentralBoson(2, 2)
    reference = fixed_n_hamiltonian(2, 3, DETUNINGS, COUPLINGS, KERR)
    times = np.arange(60) * .5
    for mode, probe in [(2, "number"), (0, "kerr")]:
        values = lambda basis: probe_values(basis, mode, probe)
        correlation, traces = sector_correlation(model, reference.fock_basis, values, times,
                                                 COUPLINGS, DETUNINGS, KERR)
        expected = two_time_correlation(*spectral_weights(reference, values(reference.fock_basis)), times)
        np.testing.assert_allclose(correlation, expected, atol=1e-12)
        assert traces.shape == (int(np.count_nonzero(values(reference.fock_basis))), len(times))


def test_loss_and_dephasing_limits():
    model = OpenCentralBoson(2, 2)
    times = np.arange(11) * 10.
    # No coupling: a storage photon decays at its own rate and dephasing does nothing to populations.
    pops = model.populations([(0, 0, 2)], times, [0., 0.], DETUNINGS, KERR,
                             loss_per_us=[0., 0., 1 / 100.], dephasing_per_us=[.05, .05, .05])
    np.testing.assert_allclose(pops[0] @ model.numbers[:, 2], 2 * np.exp(-times / 100.), rtol=1e-10)
    np.testing.assert_allclose(pops.sum(axis=-1), 1, atol=1e-12)
    # With coupling and loss, the total photon number only goes down and the trace stays 1.
    pops = model.populations([(1, 1, 0)], times, COUPLINGS, DETUNINGS, KERR,
                             loss_per_us=[1 / 200., 1 / 1000., 1 / 600.], dephasing_per_us=[.01, 0, 0])
    total = pops[0] @ model.numbers.sum(axis=1)
    assert np.all(np.diff(total) < 0)
    np.testing.assert_allclose(pops.sum(axis=-1), 1, atol=1e-12)

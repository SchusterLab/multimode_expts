"""theory.qsim.central_boson: closed-system photon-number traces from Fock states."""
import numpy as np

from theory.qsim.central_boson import fixed_n_hamiltonian, fock_number_traces


def test_one_photon_on_resonance_swaps_as_cos_squared():
    g = .02
    model = fixed_n_hamiltonian(1, 2, [0.], [g], -.027)
    time_us = np.linspace(0., 60., 61)
    traces = fock_number_traces(model, [[1, 0]], 0, time_us)
    np.testing.assert_allclose(traces[0], np.cos(2 * np.pi * g * time_us) ** 2, atol=1e-12)


def test_traces_start_at_the_occupation_and_conserve_photon_number():
    model = fixed_n_hamiltonian(3, 5, [-.14, .04, -.21, .31], [.0204] * 4, -.027)
    states = [[3, 0, 0, 0, 0], [0, 0, 1, 0, 2], [1, 1, 1, 0, 0]]
    time_us = np.linspace(0., 80., 41)
    per_mode = np.stack([fock_number_traces(model, states, mode, time_us) for mode in range(5)])
    np.testing.assert_allclose(per_mode[:, :, 0].T, states, atol=1e-12)
    np.testing.assert_allclose(per_mode.sum(axis=0), 3., atol=1e-10)

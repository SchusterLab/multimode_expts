"""theory.qsim.agp: spectral weights, the closed-system correlation and the spectral AGP."""
import numpy as np

from theory.qsim.agp import probe_values, spectral_agp, spectral_weights, two_time_correlation
from theory.qsim.central_boson import fixed_n_hamiltonian


def test_weights_give_c0_and_a_real_even_correlation():
    model = fixed_n_hamiltonian(3, 5, [.03, -.01, .02, -.04], [.02] * 4, -.027)
    for mode, probe in [(4, "number"), (4, "number_squared"), (0, "kerr")]:
        values = probe_values(model.fock_basis, mode, probe)
        differences, weights = spectral_weights(model, values)
        np.testing.assert_allclose(weights.sum(), np.mean(values**2))
        times = np.array([-3., 0., 3.])
        correlation = two_time_correlation(differences, weights, times)
        np.testing.assert_allclose(correlation[1], np.mean(values**2))
        np.testing.assert_allclose(correlation[0], correlation[2])


def test_spectral_agp_is_a_lorentzian_filter_and_keeps_scalars():
    differences = np.array([[0., .01], [-.01, 0.]])
    weights = np.array([[.5, .25], [.25, .5]])
    cutoff, coupling = .004, .02
    expected = coupling**2 * 2 * .25 * .01**2 / (.01**2 + cutoff**2)**2
    assert isinstance(spectral_agp(differences, weights, cutoff, coupling), float)
    np.testing.assert_allclose(spectral_agp(differences, weights, cutoff, coupling), expected)
    np.testing.assert_allclose(spectral_agp(differences, weights, [cutoff, cutoff], coupling),
                               [expected] * 2)

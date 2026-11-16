"""Closed-system predictions for the exact disorder table used in a scan."""
import numpy as np

from theory.qsim.agp import (probe_values, spectral_agp, spectral_weights,
                             two_time_correlation, window_agp)
from theory.qsim.central_boson import fixed_n_hamiltonian


def ensemble_prediction(photon_number, detunings_mhz, couplings_mhz, kerr_mhz,
                        mode_index, time_us, mu_khz, probe="number"):
    """Average existing model predictions over the supplied pulse detunings.

    Rows are realizations and columns are storages, in the acquisition order.
    Pulse detunings are passed directly to ``fixed_n_hamiltonian``, which
    already implements their onsite-energy sign. Cutoffs are cyclic kHz.
    ``agp_theory_window`` is the continuous-time finite-window integral;
    ``agp_theory`` is the infinite-time spectral sum.
    """
    detunings_mhz = np.asarray(detunings_mhz, dtype=float)
    couplings_mhz = np.asarray(couplings_mhz, dtype=float)
    times = np.asarray(time_us, dtype=float)
    cutoffs_mhz = np.asarray(mu_khz, dtype=float).reshape(-1) / 1000
    coupling_mhz = float(np.mean(np.abs(couplings_mhz)))
    correlation = np.zeros(times.shape)
    finite_window = np.zeros(cutoffs_mhz.shape)
    spectral = np.zeros(cutoffs_mhz.shape)
    for detunings in detunings_mhz:
        model = fixed_n_hamiltonian(photon_number, len(couplings_mhz) + 1,
                                    detunings, couplings_mhz, kerr_mhz)
        values = probe_values(model.fock_basis, mode_index, probe)
        differences, weights = spectral_weights(model, values)
        correlation += two_time_correlation(differences, weights, times)
        finite_window += window_agp(differences, weights, cutoffs_mhz,
                                    coupling_mhz, times[-1])
        spectral += spectral_agp(differences, weights, cutoffs_mhz, coupling_mhz)
    realization_count = len(detunings_mhz)
    return dict(C1_theory=correlation / realization_count,
                agp_theory_window=finite_window / realization_count,
                agp_theory=spectral / realization_count,
                theory_kerr_mhz=np.asarray(kerr_mhz))

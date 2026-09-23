"""Spectrum construction for many-body Ramsey reconstructions.

Extracted verbatim from ``EncodingHamiltonianSpectroscopyExperiment`` (spec
section 7.5). Pure numerics; behaviour unchanged, pinned by
``tests/test_mbr_analysis_golden.py``.

:func:`analyze_spectrum` currently does two jobs the spec separates: it builds
the fixed-photon-number Fock basis, assembles and diagonalizes the Hamiltonian,
and derives theoretical amplitudes (spec's ``mbr_hamiltonian.py``), *and* it
windows, pads and FFTs the measured traces (spec's ``mbr_spectrum.py``).

Splitting it is a genuine refactor rather than a move, so it is deliberately
left for its own commit. The golden baseline covers this function through the
FFT path, so that split will be verifiable numerically.

TODO(spec 7.5): separate basis/Hamiltonian construction into
``fitting/qsim/mbr_hamiltonian.py``.
"""
import numpy as np

from pydantic import BaseModel, ConfigDict
from typing import Any


from slab import AttrDict
from mbr_hamiltonian import construct_hamiltonian
from mbr_fft import calculate_ldos_from_propagator
class SpectrumDataSet(BaseModel):
    
    time_us: Any
    energy_MHz: Any
    measured_local: Any
    theory_local: Any
    measured: np.ndarray | list
    theory: Any
    theory_A: Any
    energies_MHz: Any
    fock_basis: Any
    basis_eigenstate_weights: Any
    eigenstate_weights: Any
    spectral_weights: Any
    physical_kerr_MHz: Any
    complete_basis: Any
    energy_limit_MHz: Any
    fft_window: Any
    zero_padding: Any
    fft_resolution_MHz: Any
    
    

def ldos_weights(spectrum):
    """Local density-of-states weights, summed within each degenerate multiplet.

    Returns ``(energies_MHz, weights)`` with one row per measured row and one
    column per *distinct* eigenenergy.

    Why the summation order matters. ``rho_i(E) = sum_a |<i|E_a>|^2 delta(E-E_a)``
    is only well defined once the sum over a degenerate multiplet is taken
    *before* the modulus. Individual eigenvectors inside a multiplet are an
    arbitrary basis choice -- ``eigh`` picks one, and a different LAPACK picks
    another -- but the subspace projector ``P_lambda`` is not arbitrary. So bin
    the signed weights and take the modulus of the bin,
    ``|<f|P_lambda|b>|``, rather than binning the moduli.

    For diagonal rows (final occupation == initial) every term is
    ``|<b|E_a>|^2 >= 0``, nothing can cancel, and the two orders agree
    identically. That is why the diagonal-only code predating the off-diagonal
    generalization (2026-08-24) was correct, and why fixing the order changes
    no diagonal result.

    Merged spectra carry no ``spectral_weights``: :func:`merge_spectra` rebuilds
    theory from the diagonal probabilities in ``basis_eigenstate_weights``. Those
    are already non-negative, so binning them directly is the same quantity.
    """
    energies_MHz, energy_indices = np.unique(
        np.round(np.asarray(spectrum.energies_MHz), 10), return_inverse=True)
    n_bins = len(energies_MHz)

    if "spectral_weights" in spectrum:
        weights = np.asarray(spectrum.spectral_weights)
    else:
        weights = np.asarray(spectrum.eigenstate_weights)
        if np.any(weights < -1e-12):
            raise ValueError(
                "spectrum has no spectral_weights and its eigenstate_weights are "
                "not non-negative, so the degenerate-multiplet sum is ambiguous")

    binned = np.empty((weights.shape[0], n_bins))
    for row, row_weights in enumerate(weights):
        real = np.bincount(energy_indices, weights=np.real(row_weights), minlength=n_bins)
        imag = np.bincount(energy_indices, weights=np.imag(row_weights), minlength=n_bins)
        binned[row] = np.abs(real + 1j * imag)
    return energies_MHz, binned


def analyze_spectrum(reconstruction, 
                     photon_number, 
                     detunings, 
                     couplings_MHz,
                     floquet_cycle_us, 
                     physical_kerr_MHz, 
                     fft_window="raw", 
                     zero_padding=1):
    """
    Build the fixed-N Hamiltonian, LDOS weights, and measured/theory spectra.

    Returns SpectrumDataSet

    energies_MHz is the list of energy eigenvalue of the hamiltonian.
    theory_local is the fft result expected from theory, and 
    measured_local is the measured fft result.
    """
    return_dataset = SpectrumDataSet()
    
    cycles = reconstruction.cycles
    A = reconstruction.A
    final_occupations = reconstruction.get("final_occupations", reconstruction.occupations)
    detunings = np.asarray(detunings)
    return_dataset.physical_kerr_MHz = float(physical_kerr_MHz)
    
    return_dataset.time_us = cycles * floquet_cycle_us
    sample_time_us = return_dataset.time_us[1] - return_dataset.time_us[0]

    return_dataset.fft_window = fft_window
    return_dataset.zero_padding = zero_padding
    return_dataset.fft_resolution = 1/ (len(cycles) * sample_time_us)
    
    if sample_time_us <= 0. or not np.allclose(np.diff(return_dataset.time_us), sample_time_us):
        raise ValueError("spectroscopy time samples are not uniform")
    # One Trotter step is one complete pulse+sync Floquet cycle.
    # couplings_MHz already includes each pulse's share of that cycle:
    # g = 1/(4*pi_frac*T_cycle). Always-on self-Kerr enters without scaling.


    mode_count = len(reconstruction.occupations[0])
    H_MHz, fock_index_map = construct_hamiltonian(photon_number,
                                                  mode_count,
                                                  detunings,
                                                  return_dataset.physical_kerr_MHz,
                                                  couplings_MHz)
    return_dataset.fock_basis = [state for _,state in enumerate(fock_index_map)]
    
    
    return_dataset.energies_MHz, states = np.linalg.eigh(H_MHz) #(f: fock idx, k: eig idx )
    return_dataset.basis_eigenstate_weights = np.abs(states) ** 2
    #For each occupations for the experiment, calculate its index in the basis
    #that is used for the matrix setup
    basis_rows = [fock_index_map[tuple(occupation)] for occupation in reconstruction.occupations]
    final_rows = [fock_index_map[tuple(occupation)] for occupation in final_occupations]
    #Pick rows in the eigenstate matrix
    return_dataset.spectral_weights = states[final_rows] * states[basis_rows].conj()
    return_dataset.eigenstate_weights = np.abs(return_dataset.spectral_weights)
    #List of "Theory phase", which is the list of e^{-i2 * pi * f_{eigen} t}
    theory_phase = np.exp(-2j * np.pi * np.outer(return_dataset.energies_MHz, return_dataset.time_us))
    #Do the matrix multiplication, which will give sum_n <n|U|n> 
    #as a function of time
    return_dataset.theory_A = return_dataset.spectral_weights @ theory_phase


    #############################################################
    #######   FFT of measured dataset   #########################
    #############################################################
    return_dataset.measured_local, return_dataset.energy_MHz = calculate_ldos_from_propagator(A,
                                                                   cycles,
                                                                   sample_time_us,
                                                                   fft_window,
                                                                   zero_padding)
    return_dataset.theory_local, _  =  calculate_ldos_from_propagator(return_dataset.theory_A,
                                                          cycles,
                                                          sample_time_us,
                                                          fft_window,
                                                          zero_padding)
    
    normalized_fft_ldos = np.zeros(np.shape(return_dataset.measured_local))
    
    diagonal = np.asarray([tuple(initial) == tuple(final) for initial, final in zip(reconstruction.occupations, final_occupations)])
    for index, trace in enumerate(reversed(return_dataset.measured_local)):
        if tuple(reconstruction.occupations[index]) == final_occupations[index] and (A[index, 0] > 0):
            normalized_fft_ldos[index, :] = trace/A[index, 0] 
            continue
        normalized_fft_ldos[index, :] = trace
    
    #summing over all basis to get DOS
    return_dataset.measured = np.sum(normalized_fft_ldos, axis=0)
    return_dataset.theory = np.sum(return_dataset.theory_local, axis=0)
    
    if np.max(return_dataset.theory) > 0.:
        return_dataset.theory_local *= np.max(return_dataset.measured) / np.max(return_dataset.theory)
        return_dataset.theory = np.sum(return_dataset.theory_local, axis=0)
    
    return_dataset.complete_basis = np.all(diagonal) and set(map(tuple, reconstruction.occupations)) == set(map(tuple, return_dataset.fock_basis))
    return_dataset.energy_limit_MHz = min(np.max(np.abs(return_dataset.energy_MHz)), max(0.6, 1.2 * np.max(np.abs(return_dataset.energies_MHz))))

    return return_dataset

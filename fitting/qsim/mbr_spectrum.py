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

from itertools import product

import numpy as np

from slab import AttrDict


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

    Returns AttrDict with:
        - time_us=time_us, 
        - energy_MHz=energy_MHz, 
        - measured_local=measured_local, 
        - theory_local=theory_local,
        - measured=measured, 
        - theory=theory, 
        - energies_MHz=energies_MHz,
        - fock_basis=fock_basis,
        - basis_eigenstate_weights=np.abs(states) ** 2,
        - eigenstate_weights=eigenstate_weights, 
        - physical_kerr_MHz=physical_kerr_MHz,
        - complete_basis=complete_basis, 
        - energy_limit_MHz=energy_limit_MHz,
        - fft_window=fft_window, 
        - zero_padding=zero_padding,
        - fft_resolution_MHz=1. / (len(cycles) * sample_time_us),

    energies_MHz is the list of energy eigenvalue of the hamiltonian.
    theory_local is the fft result expected from theory, and 
    measured_local is the measured fft result.
    """
    cycles = reconstruction.cycles
    A = reconstruction.A
    final_occupations = reconstruction.get("final_occupations", reconstruction.occupations)
    detunings = np.asarray(detunings)
    physical_kerr_MHz = float(physical_kerr_MHz)
    if not np.isfinite(physical_kerr_MHz):
        raise ValueError("physical_kerr_MHz must be finite")
    if len(cycles) < 2:
        raise ValueError("spectroscopy requires at least two cycle points")
    time_us = cycles * floquet_cycle_us
    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError("spectroscopy time samples are not uniform")

    if fft_window is None:
        fft_window = "raw"
    windows = {"raw": np.ones, "hann": np.hanning,
               "hamming": np.hamming, "blackman": np.blackman}
    if fft_window not in windows:
        raise ValueError("fft_window must be 'raw', 'hann', 'hamming', or 'blackman'")
    window = windows[fft_window](len(cycles))
    if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
        raise ValueError("zero_padding must be an integer >= 1")
    if np.sum(window) <= 0.:
        raise ValueError(f"{fft_window} window needs more cycle points")

    n_fft = zero_padding * len(cycles)
    energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time_us))
    # One Trotter step is one complete pulse+sync Floquet cycle.
    # couplings_MHz already includes each pulse's share of that cycle:
    # g = 1/(4*pi_frac*T_cycle). Always-on self-Kerr enters without scaling.



    #Here, the Hamiltonian is directly calculated as a matrix in a Fock basis
    #First, product makes the all possible product states within photon_number
    #and then those are conditionally stored in fock_basis if the number = photon number
    mode_count = len(reconstruction.occupations[0])
    fock_basis = [
        list(occupation) for occupation in product(range(photon_number + 1), repeat=mode_count)
        if sum(occupation) == photon_number
    ]
    #Storing index of each fock basis
    fock_index = {tuple(occupation): index for index, occupation in enumerate(fock_basis)}
    #Making Hamiltonian matrix in a fock basis
    H_MHz = np.zeros((len(fock_basis), len(fock_basis)))
    # The pulse program adds detuning to the positive storage-M1 sideband, so the rotating-frame onsite energy is -detuning.
    onsite_MHz = np.concatenate(([0.], -detunings))
    # updating Hamiltonian indices by estimating
    # <n_i|H_{diag}|n_j> = \delta_{ij}(delta_i n_i+Kerr/2*n_M*(n_M-1) 
    #Specifically, the algorithm is
    #   1. Multiply self Kerr times n_M1
    #   2. Multiply onsize detuning times n_i
    # <n_i|H_{coupling}|n_j>  = g \delta_{n_M+1  n_i-1}\sqrt{n_M+1 n_i}+
    #                           g \delta_{n_M-1  n_i+1}\sqrt{n_M   n_i+1}
    #Specifically, the algorithm is
    #For each column occupation,
    #   1. Loop the iteraction on storage mode index i
    #   2. Find the state with n_M increased by 1 and n_i decreased by 1 
    #      using fock_index dictionary
    #   3. Add g * \sqrt{n_M+1 n_i}
    #   4. Do the same for the state iwth n_M-1 and n_i+1
    for column, occupation in enumerate(fock_basis):
        n_M1 = occupation[0]
        H_MHz[column, column] = np.dot(onsite_MHz, occupation) + 0.5 * physical_kerr_MHz * n_M1 * (n_M1 - 1)
        for mode_index, coupling_MHz in enumerate(couplings_MHz, start=1):
            if n_M1 == 0:
                continue
            final_occupation = occupation.copy()
            final_occupation[0] -= 1
            final_occupation[mode_index] += 1
            row = fock_index[tuple(final_occupation)]
            matrix_element = coupling_MHz * np.sqrt(n_M1 * (occupation[mode_index] + 1))
            H_MHz[row, column] += matrix_element
            H_MHz[column, row] += matrix_element
    #Using np.linalg.eigh, get the eigenvalue of the Hamiltonian Matrix
    #Returns matrix with the index of (f, k), where k being eigenstate index
    #And f being fock state index.
    #So each column is an eigen state in a fock basis
    energies_MHz, states = np.linalg.eigh(H_MHz)
    #For each occupations for the experiment, calculate its index in the basis
    #that is used for the matrix setup
    basis_rows = [fock_index[tuple(occupation)] for occupation in reconstruction.occupations]
    final_rows = [fock_index[tuple(occupation)] for occupation in final_occupations]
    #Pick rows in the eigenstate matrix
    spectral_weights = states[final_rows] * states[basis_rows].conj()
    eigenstate_weights = np.abs(spectral_weights)
    #List of "Theory phase", which is the list of e^{-i2 * pi * f_{eigen} t}
    theory_phase = np.exp(-2j * np.pi * np.outer(energies_MHz, time_us))
    #Do the matrix multiplication, which will give sum_n <n|U|n> 
    #as a function of time
    theory_A = spectral_weights @ theory_phase


    #############################################################
    #######   FFT of measured dataset   #########################
    #############################################################

    fft_scale = n_fft / np.sum(window)
    measured_local = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(A * window, n=n_fft, axis=1), axes=1))
    diagonal = np.asarray([tuple(initial) == tuple(final) for initial, final in zip(reconstruction.occupations, final_occupations)])
    fft_normalization = np.where(diagonal, np.maximum(np.abs(A[:, 0]), 1e-12), 1.)
    measured_local /= fft_normalization[:, None]
    theory_local = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(theory_A * window, n=n_fft, axis=1), axes=1))
    measured = np.sum(measured_local, axis=0)
    theory = np.sum(theory_local, axis=0)
    if np.max(theory) > 0.:
        theory_local *= np.max(measured) / np.max(theory)
        theory = np.sum(theory_local, axis=0)
    complete_basis = np.all(diagonal) and set(map(tuple, reconstruction.occupations)) == set(map(tuple, fock_basis))
    energy_limit_MHz = min(np.max(np.abs(energy_MHz)), max(0.6, 1.2 * np.max(np.abs(energies_MHz))))

    return AttrDict(dict(
        time_us=time_us, 
        energy_MHz=energy_MHz, 
        measured_local=measured_local, 
        theory_local=theory_local,
        measured=measured, 
        theory=theory, 
        theory_A=theory_A,
        energies_MHz=energies_MHz,
        fock_basis=fock_basis,
        basis_eigenstate_weights=np.abs(states) ** 2,
        eigenstate_weights=eigenstate_weights, 
        spectral_weights=spectral_weights,
        fft_normalization=fft_normalization,
        physical_kerr_MHz=physical_kerr_MHz,
        complete_basis=complete_basis, 
        energy_limit_MHz=energy_limit_MHz,
        fft_window=fft_window, 
        zero_padding=zero_padding,
        fft_resolution_MHz=1. / (len(cycles) * sample_time_us),
    ))

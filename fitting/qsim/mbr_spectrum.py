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

The basis and Hamiltonian construction is now
:func:`fitting.qsim.mbr_hamiltonian.fixed_n_hamiltonian` (MBR redesign step 7b),
so theory can be computed without a reconstruction.
"""

import numpy as np

from slab import AttrDict

from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian


#: The FFT windows a spectrum can be computed with, by name.
FFT_WINDOWS = {"raw": np.ones, "hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}


def windowed_fft(traces, fft_window, zero_padding):
    """|FFT| along the last axis, as every spectrum here is computed: window,
    zero padding, inverse-FFT sign (so E > 0 for exp(-2 pi i E t)), and
    scaling to the window sum."""
    sample_count = np.shape(traces)[-1]
    window = FFT_WINDOWS[fft_window](sample_count)
    n_fft = zero_padding * sample_count
    fft_scale = n_fft / np.sum(window)
    return fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(traces * window, n=n_fft, axis=-1), axes=-1))


def local_spectrum(traces, spectrum):
    """-> the local spectra of ``traces`` (row x time) on ``spectrum``'s grid,
    normalized per row as ``spectrum.measured_local`` is. Used for fitted
    returns (Matrix Pencil), so they compare with the measured ones."""
    return (windowed_fft(traces, spectrum.fft_window, spectrum.zero_padding)
            / np.asarray(spectrum.fft_normalization)[:, None])


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
    if fft_window not in FFT_WINDOWS:
        raise ValueError("fft_window must be 'raw', 'hann', 'hamming', or 'blackman'")
    window = FFT_WINDOWS[fft_window](len(cycles))
    if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
        raise ValueError("zero_padding must be an integer >= 1")
    if np.sum(window) <= 0.:
        raise ValueError(f"{fft_window} window needs more cycle points")

    n_fft = zero_padding * len(cycles)
    energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time_us))
    # One Trotter step is one complete pulse+sync Floquet cycle.
    # couplings_MHz already includes each pulse's share of that cycle:
    # g = 1/(4*pi_frac*T_cycle). Always-on self-Kerr enters without scaling.



    mode_count = len(reconstruction.occupations[0])
    hamiltonian = fixed_n_hamiltonian(photon_number, mode_count, detunings,
                                      couplings_MHz, physical_kerr_MHz)
    fock_basis = hamiltonian.fock_basis
    fock_index = hamiltonian.fock_index
    energies_MHz, states = hamiltonian.energies_MHz, hamiltonian.states
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

    measured_local = windowed_fft(A, fft_window, zero_padding)
    diagonal = np.asarray([tuple(initial) == tuple(final) for initial, final in zip(reconstruction.occupations, final_occupations)])
    fft_normalization = np.where(diagonal, np.maximum(np.abs(A[:, 0]), 1e-12), 1.)
    measured_local /= fft_normalization[:, None]
    theory_local = windowed_fft(theory_A, fft_window, zero_padding)
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


def coherent_trace_spectrum(reconstruction, spectrum, scale_theory=True):
    """|FFT| of the coherent normalized trace sum_n A_n(t) / A_n(0), measured and theory.

    Uses the spectrum's own window and energy grid. On a complete basis this
    is the FFT of Tr U(t) (the same Z(t) as the SFF). The theory trace is built
    from ``spectrum.eigenstate_weights`` and scaled to the measured peak if
    ``scale_theory``. Needs the complex time traces, so a merged
    (spectrum-only) data set is refused.

    -> AttrDict: ``energy_MHz``, ``A_normalized``, ``measured_trace``,
    ``theory_trace``, ``measured``, ``theory``, ``theory_unscaled``,
    ``theory_scale``, ``fft_window``, ``n_fft``.
    """
    A = np.asarray(reconstruction.A, dtype=complex)
    time_us = np.asarray(spectrum.time_us, dtype=float)
    energy_MHz = np.asarray(spectrum.energy_MHz, dtype=float)
    if A.ndim != 2 or time_us.ndim != 1 or A.shape[1] != len(time_us):
        raise ValueError("reconstruction.A must have shape (occupation, time point)")
    if len(time_us) < 2 or not np.isclose(time_us[0], 0.0):
        raise ValueError("A/A(0) requires a time grid beginning at zero")
    if np.any(np.abs(A[:, 0]) < 1e-12):
        raise ValueError("at least one occupation has zero return amplitude at t=0")
    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0.0 or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError("coherent trace FFT requires a common uniform time grid")

    window_name = spectrum.get("fft_window", "raw") or "raw"
    if window_name not in FFT_WINDOWS:
        raise ValueError(f"unsupported FFT window: {window_name!r}")
    window = FFT_WINDOWS[window_name](len(time_us))
    if np.sum(window) <= 0.0:
        raise ValueError(f"{window_name} window has zero coherent gain")
    n_fft = len(energy_MHz)
    expected_energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time_us))
    if energy_MHz.shape != expected_energy_MHz.shape or not np.allclose(energy_MHz, expected_energy_MHz):
        raise ValueError("saved FFT energy grid does not match the time grid")
    fft_scale = n_fft / np.sum(window)

    def spectrum_of(trace):
        return fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(trace * window, n=n_fft)))

    A_normalized = A / A[:, :1]
    measured_trace = np.sum(A_normalized, axis=0)
    measured = spectrum_of(measured_trace)

    eigenenergies_MHz = np.asarray(spectrum.energies_MHz, dtype=float)
    eigenstate_weights = np.asarray(spectrum.eigenstate_weights, dtype=float)
    if eigenstate_weights.shape != (A.shape[0], len(eigenenergies_MHz)):
        raise ValueError("theory eigenstate weights do not match occupations and energies")
    theory_A = eigenstate_weights @ np.exp(-2j * np.pi * np.outer(eigenenergies_MHz, time_us))
    if np.any(np.abs(theory_A[:, 0]) < 1e-12):
        raise ValueError("at least one theoretical return is zero at t=0")
    theory_trace = np.sum(theory_A / theory_A[:, :1], axis=0)
    theory_unscaled = spectrum_of(theory_trace)
    theory_scale = 1.0
    if scale_theory and np.max(theory_unscaled) > 0.0:
        theory_scale = np.max(measured) / np.max(theory_unscaled)
    return AttrDict(dict(
        energy_MHz=energy_MHz, A_normalized=A_normalized,
        measured_trace=measured_trace, theory_trace=theory_trace,
        measured=measured, theory=theory_scale * theory_unscaled,
        theory_unscaled=theory_unscaled, theory_scale=theory_scale,
        fft_window=window_name, n_fft=n_fft,
    ))

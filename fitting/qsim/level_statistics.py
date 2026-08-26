"""Spectrum merging, level statistics and the spectral form factor.

Extracted verbatim from ``EncodingHamiltonianSpectroscopyExperiment`` (spec
section 7.5). Pure numerics: arrays and settings in, ``AttrDict`` out, no
Experiment, station or file access. Behaviour is unchanged.

- :func:`merge_spectra` -- join separately analyzed spectra onto one common
  energy grid. The level-3 ensemble input of spec section 8.
- :func:`analyze_level_statistics` -- DOS peaks, spacings and gap ratios.
- :func:`analyze_sff` -- spectral form factor from the reconstruction.

``data`` is the analyzed result mapping (``reconstruction`` and ``spectrum``
present); the owning Experiment supplies ``self.data`` by default.
"""

import numpy as np
from scipy.signal import find_peaks

from slab import AttrDict


def merge_spectra(data):
    """
    This method is to join occupation rows from separately analyzed spectra on 
    one common energy grid.

    Each input must already contain ``reconstruction`` and ``spectrum`` 
    from ``analyze(stage='spectrum')``. The photon number, mode order, phase frame, 
    and FFT window must agree, but the record length, zero padding, and energy grids 
    may differ. A source Hamiltonian may differ from the first source only when its
    largest eigenenergy change is smaller than the worse of their physical FFT 
    resolutions. The first source defines the Hamiltonian used for all theory and 
    LDOS rows. Measured FFT magnitudes are linearly interpolated only over the shared 
    Nyquist range. Interpolation aligns coordinates; it does not improve a row's 
    physical resolution, which is retained in ``row_fft_resolution_MHz``.

    The returned object is spectrum-only because its rows do not share one complex time
    grid. It can be passed to ``display`` and to complete-basis level-statistics 
    analysis, but not to SFF or occupation time-trace displays.
    """
    sources = list(data) if isinstance(data, (list, tuple)) else [data]
    sources = [source.data if not isinstance(source, dict) and hasattr(source, "data") else source for source in sources]
    if len(sources) < 2:
        raise ValueError("merge_spectra requires at least two analyzed spectra")
    for source in sources:
        if "reconstruction" not in source or "spectrum" not in source:
            raise ValueError("every merge source must contain reconstruction and spectrum")
        if source.get("spectrum_only", False):
            raise ValueError("merge_spectra does not accept an already merged spectrum")

    reference = sources[0]
    reference_energies_MHz = np.asarray(reference.spectrum.energies_MHz)
    if "fock_basis" not in reference.spectrum or "basis_eigenstate_weights" not in reference.spectrum:
        raise ValueError("reanalyze the first spectrum with the current code before merging")
    reference_fock_basis = [tuple(occupation) for occupation in reference.spectrum.fock_basis]
    reference_basis_eigenstate_weights = np.asarray(reference.spectrum.basis_eigenstate_weights)
    if reference_basis_eigenstate_weights.shape != (len(reference_fock_basis), len(reference_energies_MHz)):
        raise ValueError("the first spectrum has inconsistent full-basis eigenstate weights")
    reference_weights_by_occupation = dict(zip(reference_fock_basis, reference_basis_eigenstate_weights))
    occupations = []
    measured_rows = []
    theory_rows = []
    row_fft_resolution_MHz = []
    row_physical_kerr_MHz = []
    hamiltonian_mismatch_MHz = []
    energy_steps_MHz = []
    energy_minima_MHz = []
    energy_maxima_MHz = []
    windows = {"raw": np.ones, 
               "hann": np.hanning, 
               "hamming": np.hamming, 
               "blackman": np.blackman}
    # Collect the spectrum and its resolution information
    for source in sources:
        spectrum = source.spectrum
        if source.photon_number != reference.photon_number or list(source.mode_labels) != list(reference.mode_labels):
            raise ValueError("merged spectra must use the same photon number and mode order")
        if source.phase_frame != reference.phase_frame:
            raise ValueError("merged spectra must use the same physical phase frame")
        if spectrum.fft_window != reference.spectrum.fft_window:
            raise ValueError("merged spectra must use the same FFT window")
        source_energies_MHz = np.asarray(spectrum.energies_MHz)
        if source_energies_MHz.shape != reference_energies_MHz.shape:
            raise ValueError("merged spectra describe different Hilbert-space dimensions")
        energy_mismatch_MHz = float(np.max(np.abs(source_energies_MHz - reference_energies_MHz)))
        allowed_mismatch_MHz = max(float(spectrum.fft_resolution_MHz), float(reference.spectrum.fft_resolution_MHz))
        if energy_mismatch_MHz > allowed_mismatch_MHz:
            raise ValueError(f"source Hamiltonian differs by {energy_mismatch_MHz:.6g} MHz, larger than the {allowed_mismatch_MHz:.6g} MHz FFT resolution")
        hamiltonian_mismatch_MHz.append(energy_mismatch_MHz)

        source_occupations = [tuple(occupation) for occupation in source.reconstruction.occupations]
        if set(occupations).intersection(source_occupations):
            raise ValueError("merged spectra contain duplicate occupations")
        occupations.extend(source_occupations)
        energy_MHz = np.asarray(spectrum.energy_MHz)
        energy_steps_MHz.append(float(np.median(np.diff(energy_MHz))))
        energy_minima_MHz.append(float(energy_MHz[0]))
        energy_maxima_MHz.append(float(energy_MHz[-1]))
        row_fft_resolution_MHz.extend([float(spectrum.fft_resolution_MHz)] * len(source_occupations))
        row_physical_kerr_MHz.extend([float(spectrum.physical_kerr_MHz)] * len(source_occupations))
    # Choose the coarsest binning and range
    energy_step_MHz = max(energy_steps_MHz)
    energy_min_MHz = max(energy_minima_MHz)
    energy_max_MHz = min(energy_maxima_MHz)


    first_bin = int(np.ceil(energy_min_MHz / energy_step_MHz - 1e-10))
    last_bin = int(np.floor(energy_max_MHz / energy_step_MHz + 1e-10))
    energy_MHz = np.arange(first_bin, last_bin + 1, dtype=float) * energy_step_MHz
    if len(energy_MHz) < 2:
        raise ValueError("merged spectra have no usable common energy range")
    if np.any(reference_energies_MHz < energy_MHz[0]) or np.any(reference_energies_MHz > energy_MHz[-1]):
        raise ValueError("Hamiltonian eigenenergies lie outside the shared Nyquist range")
    eigenstate_weights = np.asarray([reference_weights_by_occupation[occupation] for occupation in occupations])

    for source in sources:
        spectrum = source.spectrum
        source_energy_MHz = np.asarray(spectrum.energy_MHz)
        measured_rows.extend([np.interp(energy_MHz, source_energy_MHz, row) for row in spectrum.measured_local])

        time_us = np.asarray(spectrum.time_us)
        window = windows[spectrum.fft_window](len(time_us))
        theory_phase = np.exp(-2j * np.pi * np.outer(reference_energies_MHz, time_us))
        source_weights = np.asarray([reference_weights_by_occupation[tuple(occupation)] for occupation in source.reconstruction.occupations])
        theory_A = source_weights @ theory_phase
        fourier_kernel = np.exp(2j * np.pi * np.outer(time_us, energy_MHz))
        theory_rows.extend(np.abs((theory_A * window) @ fourier_kernel) / np.sum(window))

    measured_local = np.asarray(measured_rows)
    theory_local = np.asarray(theory_rows)
    measured = np.sum(measured_local, axis=0)
    theory = np.sum(theory_local, axis=0)
    if np.max(theory) > 0.:
        theory_local *= np.max(measured) / np.max(theory)
        theory = np.sum(theory_local, axis=0)

    complete_basis = set(occupations) == set(reference_fock_basis)
    row_fft_resolution_MHz = np.asarray(row_fft_resolution_MHz)
    row_physical_kerr_MHz = np.asarray(row_physical_kerr_MHz)
    energy_limit_MHz = min(np.max(np.abs(energy_MHz)), max(0.6, 1.2 * np.max(np.abs(reference_energies_MHz))))
    spectrum = AttrDict(dict(
        energy_MHz=energy_MHz, 
        measured_local=measured_local, 
        theory_local=theory_local,
        measured=measured, 
        theory=theory, 
        energies_MHz=reference_energies_MHz,
        fock_basis=[list(occupation) for occupation in reference_fock_basis], 
        basis_eigenstate_weights=reference_basis_eigenstate_weights,
        eigenstate_weights=eigenstate_weights, 
        physical_kerr_MHz=float(reference.spectrum.physical_kerr_MHz),
        complete_basis=complete_basis, 
        energy_limit_MHz=energy_limit_MHz,
        fft_window=reference.spectrum.fft_window, 
        zero_padding=None,
        fft_resolution_MHz=float(np.max(row_fft_resolution_MHz)), 
        row_fft_resolution_MHz=row_fft_resolution_MHz,
        mixed_resolution=not np.allclose(row_fft_resolution_MHz, row_fft_resolution_MHz[0]),
        row_physical_kerr_MHz=row_physical_kerr_MHz,
        hamiltonian_mismatch_MHz=float(np.max(hamiltonian_mismatch_MHz)),
        mixed_hamiltonian=not np.isclose(np.max(hamiltonian_mismatch_MHz), 0.), 
        energy_grid_step_MHz=energy_step_MHz,
    ))
    return AttrDict(dict(
        reconstruction=AttrDict(dict(occupations=occupations)), 
        spectrum=spectrum,
        hardware=reference.hardware, 
        photon_number=reference.photon_number,
        detunings=reference.detunings, 
        mode_labels=reference.mode_labels,
        phase_frame=reference.phase_frame,
        source_phase_frames=[source.get("source_phase_frame", source.phase_frame) for source in sources],
        spectrum_only=True, # To disable SFF analysis
    ))



def analyze_level_statistics(data,
                             peak_prominence=None,
                             peak_prominence_fraction=None,
                             minimum_peak_distance_MHz=None,
                             energy_limit_MHz=None):
    """Analyze measured DOS peaks and level spacings without imposing multiplicities.

    ``spectrum.measured`` is the sum of the occupation-resolved FFT magnitudes. Its
    normalization gives an ideal isolated one-level line height one, so each detected
    peak height is rounded independently to estimate its multiplicity. The rounded
    values are never rescaled or adjusted to make their sum equal the Hilbert-space
    dimension D; disagreement with D is retained as a measurement diagnostic.

    Exact degeneracy is determined only from the Hamiltonian eigenenergies. If an exact
    degeneracy is present, the result retains the raw zero gaps, the defined ratios, and
    the number of undefined 0/0 ratios. If the Hamiltonian is nondegenerate, experimental
    gap ratios are calculated only when all D measured peaks are resolved.
    """
    if "spectrum" not in data:
        raise ValueError("level statistics requires analyzed spectroscopy data")
    spectrum = data.spectrum
    if not spectrum.complete_basis:
        raise ValueError("level statistics requires the complete fixed-N occupation basis")
    window_resolution_factors = {"raw": 1., "hann": 2., "hamming": 2., "blackman": 3.}
    default_prominence_fractions = {"raw": 0.05, "hann": 0.05, "hamming": 0.05, "blackman": 0.05}
    if spectrum.fft_window not in window_resolution_factors:
        raise ValueError(f"unknown FFT window {spectrum.fft_window}")
    if peak_prominence_fraction is None:
        peak_prominence_fraction = default_prominence_fractions[spectrum.fft_window]
    if not np.isfinite(peak_prominence_fraction) or peak_prominence_fraction < 0. or peak_prominence_fraction >= 1.:
        raise ValueError("peak_prominence_fraction must be finite and in [0, 1)")

    energy_MHz = np.asarray(spectrum.energy_MHz, dtype=float)
    measured_DOS = np.asarray(spectrum.measured, dtype=float)
    theory_DOS = np.asarray(spectrum.theory, dtype=float)
    if energy_MHz.ndim != 1 or measured_DOS.shape != energy_MHz.shape or theory_DOS.shape != energy_MHz.shape:
        raise ValueError("measured DOS, theory DOS, and FFT energy axis must be one-dimensional and have the same shape")
    if not np.all(np.isfinite(energy_MHz)) or not np.all(np.isfinite(measured_DOS)) or not np.all(np.isfinite(theory_DOS)):
        raise ValueError("measured DOS, theory DOS, and FFT energy axis must be finite")

    if energy_limit_MHz is None:
        energy_limit_MHz = min(float(np.max(np.abs(energy_MHz))), float(spectrum.get("energy_limit_MHz", np.max(np.abs(energy_MHz)))))
    if not np.isfinite(energy_limit_MHz) or energy_limit_MHz <= 0.:
        raise ValueError("energy_limit_MHz must be finite and positive")
    inside_energy_limit = np.abs(energy_MHz) <= energy_limit_MHz
    energy_MHz = energy_MHz[inside_energy_limit]
    measured_DOS = measured_DOS[inside_energy_limit]
    theory_DOS = theory_DOS[inside_energy_limit]
    if len(energy_MHz) < 3:
        raise ValueError("the selected energy range has fewer than three FFT bins")

    energy_bin_widths_MHz = np.diff(energy_MHz)
    energy_step_MHz = float(energy_bin_widths_MHz[0])
    if energy_step_MHz <= 0. or not np.allclose(energy_bin_widths_MHz, energy_step_MHz):
        raise ValueError("FFT energy grid must be uniformly increasing")

    measured_DOS_range = float(np.max(measured_DOS) - np.min(measured_DOS))
    if measured_DOS_range <= 0.:
        raise ValueError("measured DOS has no peak contrast")
    if peak_prominence is None:
        peak_prominence = peak_prominence_fraction * measured_DOS_range
    elif not np.isfinite(peak_prominence) or peak_prominence < 0.:
        raise ValueError("peak_prominence must be finite and nonnegative")
    else:
        peak_prominence = float(peak_prominence)
        peak_prominence_fraction = peak_prominence / measured_DOS_range

    row_fft_resolution_MHz = spectrum.get("row_fft_resolution_MHz", [spectrum.fft_resolution_MHz])
    fft_resolution_MHz = float(np.max(row_fft_resolution_MHz))
    window_resolution_factor = window_resolution_factors[spectrum.fft_window]
    effective_resolution_MHz = window_resolution_factor * fft_resolution_MHz
    if minimum_peak_distance_MHz is None:
        minimum_peak_distance_MHz = effective_resolution_MHz
    if not np.isfinite(minimum_peak_distance_MHz) or minimum_peak_distance_MHz <= 0.:
        raise ValueError("minimum_peak_distance_MHz must be finite and positive")
    minimum_peak_distance_bins = max(1, int(np.ceil(minimum_peak_distance_MHz / energy_step_MHz - 1e-10)))

    peak_indices, peak_properties = find_peaks(
        measured_DOS,
        prominence=peak_prominence,
        distance=minimum_peak_distance_bins,
    )
    peak_prominences = peak_properties["prominences"]
    peak_energies_MHz = energy_MHz[peak_indices]
    peak_heights = measured_DOS[peak_indices]

    basis_dimension = len(data.reconstruction.occupations)
    detected_peak_count = len(peak_energies_MHz)
    if detected_peak_count == 0:
        raise ValueError("no experimental DOS peaks were detected; reduce peak_prominence")

    # The FFT normalization makes an isolated one-level peak height one. Round each measured DOS peak independently and leave any mismatch with D visible.
    rounded_peak_multiplicities = np.floor(peak_heights + 0.5).astype(int)
    rounded_multiplicity_sum = int(np.sum(rounded_peak_multiplicities))
    multiplicities_are_positive = bool(np.all(rounded_peak_multiplicities >= 1))
    multiplicity_sum_matches_D = rounded_multiplicity_sum == basis_dimension

    # Determine exact Hamiltonian degeneracies from eigenenergies rather than from finite-resolution measured peaks.
    theory_energies_MHz = np.sort(np.asarray(spectrum.energies_MHz, dtype=float))
    degeneracy_tolerance_MHz = 1e-10
    theory_raw_gaps_MHz = np.diff(theory_energies_MHz)
    theory_zero_gap_mask = np.isclose(theory_raw_gaps_MHz, 0., rtol=0., atol=degeneracy_tolerance_MHz)
    theory_raw_gaps_MHz[theory_zero_gap_mask] = 0.
    has_exact_degeneracy = bool(np.any(theory_zero_gap_mask))
    theory_distinct_energies_MHz = []
    theory_multiplicities = []
    for energy_index, energy_MHz_value in enumerate(theory_energies_MHz):
        if energy_index == 0 or not theory_zero_gap_mask[energy_index - 1]:
            theory_distinct_energies_MHz.append(float(energy_MHz_value))
            theory_multiplicities.append(1)
        else:
            theory_multiplicities[-1] += 1
    theory_distinct_energies_MHz = np.asarray(theory_distinct_energies_MHz)
    theory_multiplicities = np.asarray(theory_multiplicities, dtype=int)

    detected_peak_gaps_MHz = np.diff(peak_energies_MHz)
    inferred_degenerate_energies_MHz = np.asarray([], dtype=float)
    inferred_degenerate_gaps_MHz = np.asarray([], dtype=float)
    inferred_degenerate_gap_ratios = []
    inferred_undefined_gap_ratios = 0
    if multiplicities_are_positive:
        inferred_degenerate_energies_MHz = np.repeat(peak_energies_MHz, rounded_peak_multiplicities)
        inferred_degenerate_gaps_MHz = np.diff(inferred_degenerate_energies_MHz)
        for left_gap_MHz, right_gap_MHz in zip(inferred_degenerate_gaps_MHz[:-1], inferred_degenerate_gaps_MHz[1:]):
            smaller_gap_MHz = min(left_gap_MHz, right_gap_MHz)
            larger_gap_MHz = max(left_gap_MHz, right_gap_MHz)
            if larger_gap_MHz == 0.:
                inferred_undefined_gap_ratios += 1
            else:
                inferred_degenerate_gap_ratios.append(smaller_gap_MHz / larger_gap_MHz)
    inferred_degenerate_gap_ratios = np.asarray(inferred_degenerate_gap_ratios, dtype=float)

    full_spectrum_inside_selected_energy_range = bool(np.all(np.abs(theory_energies_MHz) <= energy_limit_MHz + degeneracy_tolerance_MHz))
    all_D_peaks_resolved = detected_peak_count == basis_dimension
    mixed_hamiltonian = bool(spectrum.get("mixed_hamiltonian", False))
    gap_ratio_available = not has_exact_degeneracy and all_D_peaks_resolved and basis_dimension >= 3 and not mixed_hamiltonian and full_spectrum_inside_selected_energy_range
    if has_exact_degeneracy:
        gap_ratio_unavailable_reason = "the exact Hamiltonian contains degenerate levels"
    elif basis_dimension < 3:
        gap_ratio_unavailable_reason = "at least three levels are required"
    elif mixed_hamiltonian:
        gap_ratio_unavailable_reason = "the merged rows use different Hamiltonians"
    elif not full_spectrum_inside_selected_energy_range:
        gap_ratio_unavailable_reason = "the selected energy range does not contain the full spectrum"
    elif not all_D_peaks_resolved:
        gap_ratio_unavailable_reason = f"{detected_peak_count} of {basis_dimension} measured peaks are resolved"
    else:
        gap_ratio_unavailable_reason = None

    experimental_level_energies_MHz = np.asarray([], dtype=float)
    gaps_MHz = np.asarray([], dtype=float)
    gap_ratios = []
    if gap_ratio_available:
        experimental_level_energies_MHz = np.asarray(peak_energies_MHz, dtype=float)
        gaps_MHz = np.diff(experimental_level_energies_MHz)
        for left_gap_MHz, right_gap_MHz in zip(gaps_MHz[:-1], gaps_MHz[1:]):
            smaller_gap_MHz = min(left_gap_MHz, right_gap_MHz)
            larger_gap_MHz = max(left_gap_MHz, right_gap_MHz)
            gap_ratios.append(smaller_gap_MHz / larger_gap_MHz)
    gap_ratios = np.asarray(gap_ratios, dtype=float)

    theory_gap_ratios = []
    theory_undefined_gap_ratios = 0
    for left_gap_MHz, right_gap_MHz in zip(theory_raw_gaps_MHz[:-1], theory_raw_gaps_MHz[1:]):
        smaller_gap_MHz = min(left_gap_MHz, right_gap_MHz)
        larger_gap_MHz = max(left_gap_MHz, right_gap_MHz)
        if larger_gap_MHz == 0.:
            theory_undefined_gap_ratios += 1
        else:
            theory_gap_ratios.append(smaller_gap_MHz / larger_gap_MHz)
    theory_gap_ratios = np.asarray(theory_gap_ratios, dtype=float)

    if "photon_number" in data:
        photon_number = data.photon_number
    else:
        photon_number = sum(data.reconstruction.occupations[0])
    return AttrDict(dict(
        energy_MHz=energy_MHz,
        measured_DOS=measured_DOS,
        theory_DOS=theory_DOS,
        peak_indices=peak_indices,
        peak_energies_MHz=peak_energies_MHz,
        peak_heights=peak_heights,
        peak_prominences=peak_prominences,
        multiplicities=rounded_peak_multiplicities,
        rounded_peak_multiplicities=rounded_peak_multiplicities,
        rounded_multiplicity_sum=rounded_multiplicity_sum,
        multiplicities_are_positive=multiplicities_are_positive,
        multiplicity_sum_matches_D=multiplicity_sum_matches_D,
        experimental_level_energies_MHz=experimental_level_energies_MHz,
        gaps_MHz=gaps_MHz,
        gap_ratios=gap_ratios,
        detected_peak_gaps_MHz=detected_peak_gaps_MHz,
        inferred_degenerate_energies_MHz=inferred_degenerate_energies_MHz,
        inferred_degenerate_gaps_MHz=inferred_degenerate_gaps_MHz,
        inferred_degenerate_gap_ratios=inferred_degenerate_gap_ratios,
        inferred_undefined_gap_ratios=inferred_undefined_gap_ratios,
        theory_energies_MHz=theory_energies_MHz,
        theory_distinct_energies_MHz=theory_distinct_energies_MHz,
        theory_multiplicities=theory_multiplicities,
        theory_gaps_MHz=theory_raw_gaps_MHz,
        theory_raw_gaps_MHz=theory_raw_gaps_MHz,
        theory_zero_gap_mask=theory_zero_gap_mask,
        theory_gap_ratios=theory_gap_ratios,
        theory_undefined_gap_ratios=theory_undefined_gap_ratios,
        has_exact_degeneracy=has_exact_degeneracy,
        degeneracy_tolerance_MHz=degeneracy_tolerance_MHz,
        basis_dimension=basis_dimension,
        detected_peak_count=detected_peak_count,
        gap_ratio_count=len(gap_ratios),
        all_D_peaks_resolved=all_D_peaks_resolved,
        gap_ratio_available=gap_ratio_available,
        gap_ratio_unavailable_reason=gap_ratio_unavailable_reason,
        sufficient_for_gap_ratios=gap_ratio_available,
        full_spectrum_inside_selected_energy_range=full_spectrum_inside_selected_energy_range,
        photon_number=photon_number,
        peak_prominence=peak_prominence,
        peak_prominence_fraction=float(peak_prominence_fraction),
        minimum_peak_distance_MHz=float(minimum_peak_distance_MHz),
        energy_limit_MHz=float(energy_limit_MHz),
        fft_resolution_MHz=fft_resolution_MHz,
        effective_resolution_MHz=effective_resolution_MHz,
        window_resolution_factor=window_resolution_factor,
        fft_window=spectrum.fft_window,
        mixed_resolution=bool(spectrum.get("mixed_resolution", False)),
        raw_window_peak_ambiguity=spectrum.fft_window == "raw",
        poisson_mean=2. * np.log(2.) - 1.,
        goe_mean=4. - 2. * np.sqrt(3.),
    ))



def analyze_sff(data, row_normalize=True):
    """
    Analyzes SFF based. Should be run after running `analyze_spectrum`.
    This dependency can be lifted in fugure. 
    """
    if "reconstruction" not in data or "spectrum" not in data:
        raise ValueError("SFF analysis requires analyzed spectroscopy data")
    if data.get("spectrum_only", False):
        raise ValueError("SFF requires one common complex time grid and is unavailable for merged spectra")
    reconstruction = data.reconstruction
    spectrum = data.spectrum
    if not spectrum.complete_basis:
        raise ValueError("SFF requires the complete fixed-N occupation basis; this data gives only a projected trace")

    cycles = np.asarray(reconstruction.cycles)
    A = np.asarray(reconstruction.A, dtype=complex).copy()
    if A.ndim != 2 or A.shape[1] != len(spectrum.time_us):
        raise ValueError("measured return and spectroscopy time dimensions differ")
    if row_normalize:
        if len(cycles) == 0 or not np.isclose(cycles[0], 0.):
            raise ValueError("row normalization requires the zero-cycle point")
        if np.any(np.abs(A[:, 0]) < 1e-12):
            raise ValueError("at least one occupation has zero return at t=0")
        A /= A[:, :1]

    energies_MHz = np.asarray(spectrum.energies_MHz)
    dimension = len(energies_MHz)
    if dimension == 0 or A.shape[0] != dimension:
        raise ValueError("measured basis dimension and theory Hilbert-space dimension differ")
    time_us = np.asarray(spectrum.time_us)
    Z_exp = np.sum(A, axis=0)
    Z_theory = np.sum(np.exp(-2j * np.pi * np.outer(energies_MHz, time_us)), axis=0)
    SFF_exp = np.abs(Z_exp / dimension) ** 2
    SFF_theory = np.abs(Z_theory / dimension) ** 2
    degenerate_pairs = np.isclose(energies_MHz[:, None], energies_MHz[None, :], rtol=0., atol=1e-10)
    plateau_reference = np.count_nonzero(degenerate_pairs) / dimension ** 2
    return AttrDict(dict(
        time_us=time_us, 
        A=A, 
        Z_exp=Z_exp,
        Z_theory=Z_theory,
        SFF_exp=SFF_exp, 
        SFF_theory=SFF_theory, 
        dimension=dimension,
        plateau_reference=plateau_reference, 
        nondegenerate_reference=1. / dimension,
        row_normalized=bool(row_normalize), 
        photon_number=data.get("photon_number", sum(reconstruction.occupations[0])),
        phase_frame=data.get("phase_frame", "as_acquired"), 
        physical_kerr_MHz=spectrum.physical_kerr_MHz,
    ))


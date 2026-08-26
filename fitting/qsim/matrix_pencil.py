"""Matrix-Pencil analysis for many-body Ramsey reconstructions.

Extracted verbatim from ``EncodingHamiltonianSpectroscopyExperiment`` (spec
section 7.5). These are pure numerics: they take arrays and settings and return
an ``AttrDict``, touching no Experiment, station or file. Behaviour is
unchanged -- ``tests/test_mbr_analysis_golden.py`` pins it.

- :func:`analyze_matrix_pencil` -- shared poles across all occupations.
- :func:`analyze_matrix_pencil_trace` -- one complex time trace, the per-row
  step the above runs inside.
- :func:`refit_occupation` -- refit one row using only that row's own
  candidate poles, before the cross-row merge.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from slab import AttrDict


def analyze_matrix_pencil(reconstruction,
                          spectrum,
                          requested_max_modes=None,
                          pencil_length=None,
                          minimum_consecutive_ranks=3,
                          minimum_supporting_rows=1,
                          track_frequency_tolerance_bins=1.5,
                          merge_frequency_tolerance_bins=None,
                          dedup_frequency_tolerance_bins=None,
                          track_decay_tolerance_per_us=None,
                          dedup_decay_tolerance_per_us=None,
                          match_decay=True,
                          numerical_floor=1e-10,
                          noise_singular_value_factor=2.858,
                          minimum_pole_radius=0.2,
                          maximum_pole_radius=1.05,
                          require_early_start=True,
                          rank_sweep_extra=None,
                          clip_growth=True,
                          least_squares_rcond=None,
                          store_rank_sweeps=False):
    """
    Find shared damped-exponential poles independently in each occupation.

    The measured return is modeled as

        A_i[n] = sum_m c_im z_m**n,
        z_m = exp((-gamma_m - 2j*pi*f_m)*dt).

    Every occupation receives its own Hankel rank sweep. Poles are tracked
    between adjacent assumed ranks, stable rowwise candidates are merged by
    frequency across occupations, and the merged frequencies/decays are held
    fixed while every ``c_im`` is refit for every occupation by least squares,
    including occupations that did not initially detect that pole.
    No Hamiltonian eigenenergy or theory spectrum is used for pole selection.

    The returned ``reconstructed_local`` is the finite-time FFT of the fitted
    complex returns, evaluated with the same window, padding, sign convention,
    and measured-A(0) normalization as ``spectrum.measured_local``. The
    continuous pole weights are returned separately as ``pole_DOS_weights``.
    For a complete basis their ideal value is the trace of each spectral
    projector, so a degenerate pole has its multiplicity as its DOS weight.
    The DOS uses the linear sum of A_i(0)-normalized complex amplitudes; the
    imaginary part and the sum of amplitude magnitudes are kept as diagnostics.

    Defaults reproduce the exploratory notebook algorithm. They are analysis
    choices rather than calibrated confidence levels. ``rank_sweep_extra=None``
    keeps the original full algebraic sweep. Setting it to 2 tests only through
    ``estimated_signal_rank + 2``, which is sufficient for a three-rank
    persistence requirement. ``match_decay=False`` tracks poles by frequency
    only and leaves decay as a fitted-pole diagnostic.
    """

    A = np.asarray(reconstruction.A, dtype=complex)
    time_us = np.asarray(spectrum.time_us, dtype=float)
    occupations = [tuple(occupation) for occupation in reconstruction.occupations]
    final_occupations = [tuple(occupation) for occupation in reconstruction.get(
        "final_occupations", occupations)]
    diagonal = np.asarray([initial == final for initial, final in zip(occupations, final_occupations)])
    if requested_max_modes is None:
        requested_max_modes = len(spectrum.fock_basis)



    ################################################################################
    ####   A Bunch of safety checks           ######################################
    ################################################################################
    if A.ndim != 2 or A.shape[0] != len(occupations) or A.shape[1] != len(time_us):
        raise ValueError("reconstruction.A must have shape (occupation, time point)")
    if len(time_us) < 5:
        raise ValueError("Matrix Pencil requires at least five time points")
    if not np.isclose(time_us[0], 0.):
        raise ValueError("Matrix Pencil DOS reconstruction requires the zero-time point")
    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError("Matrix Pencil requires uniformly spaced time points")
    if not isinstance(requested_max_modes, (int, np.integer)) or isinstance(requested_max_modes, (bool, np.bool_)) or requested_max_modes < 1:
        raise ValueError("requested_max_modes must be a positive integer")
    if not isinstance(minimum_consecutive_ranks, (int, np.integer)) or isinstance(minimum_consecutive_ranks, (bool, np.bool_)) or minimum_consecutive_ranks < 1:
        raise ValueError("minimum_consecutive_ranks must be a positive integer")
    if not isinstance(minimum_supporting_rows, (int, np.integer)) or isinstance(minimum_supporting_rows, (bool, np.bool_)) or minimum_supporting_rows < 1:
        raise ValueError("minimum_supporting_rows must be a positive integer")
    if pencil_length is None:
        pencil_length = len(time_us) // 2
    if not isinstance(pencil_length, (int, np.integer)) or isinstance(pencil_length, (bool, np.bool_)) or pencil_length < 1 or pencil_length >= len(time_us):
        raise ValueError("pencil_length must be an integer between 1 and sample_count - 1")
    maximum_algebraic_rank = min(requested_max_modes, pencil_length, len(time_us) - pencil_length)
    if maximum_algebraic_rank < minimum_consecutive_ranks:
        raise ValueError(f"this Matrix-Pencil configuration permits at most {maximum_algebraic_rank} ranks, fewer than minimum_consecutive_ranks={minimum_consecutive_ranks}")
    if rank_sweep_extra is not None and (not isinstance(rank_sweep_extra, (int, np.integer)) or isinstance(rank_sweep_extra, (bool, np.bool_)) or rank_sweep_extra < 0):
        raise ValueError("rank_sweep_extra must be None or a nonnegative integer")
    if not np.isfinite(track_frequency_tolerance_bins) or track_frequency_tolerance_bins <= 0.:
        raise ValueError("track_frequency_tolerance_bins must be finite and positive")
    if merge_frequency_tolerance_bins is None:
        merge_frequency_tolerance_bins = track_frequency_tolerance_bins
    if dedup_frequency_tolerance_bins is None:
        dedup_frequency_tolerance_bins = track_frequency_tolerance_bins
    if not np.isfinite(merge_frequency_tolerance_bins) or merge_frequency_tolerance_bins <= 0.:
        raise ValueError("merge_frequency_tolerance_bins must be finite and positive")
    if not np.isfinite(dedup_frequency_tolerance_bins) or dedup_frequency_tolerance_bins <= 0.:
        raise ValueError("dedup_frequency_tolerance_bins must be finite and positive")
    if not np.isfinite(numerical_floor) or numerical_floor <= 0.:
        raise ValueError("numerical_floor must be finite and positive")
    row_normalization = np.asarray([A[row, 0] if diagonal[row] else 1. for row in range(len(A))])[:, None]
    if np.any(np.abs(row_normalization) <= numerical_floor):
        raise ValueError("Matrix Pencil DOS reconstruction requires nonzero A_i(0) for every occupation")
    normalized_A = A / row_normalization
    if not np.isfinite(noise_singular_value_factor) or noise_singular_value_factor <= 0.:
        raise ValueError("noise_singular_value_factor must be finite and positive")
    if not np.isfinite(minimum_pole_radius) or not np.isfinite(maximum_pole_radius) or minimum_pole_radius <= 0. or maximum_pole_radius <= minimum_pole_radius:
        raise ValueError("pole radii must satisfy 0 < minimum_pole_radius < maximum_pole_radius")
    if least_squares_rcond is not None and (not np.isfinite(least_squares_rcond) or least_squares_rcond < 0.):
        raise ValueError("least_squares_rcond must be None or finite and nonnegative")

    sample_count = len(time_us)
    sampling_frequency_MHz = 1. / sample_time_us
    nyquist_MHz = 0.5 * sampling_frequency_MHz
    fft_resolution_MHz = 1. / (sample_count * sample_time_us)
    track_frequency_tolerance_MHz = track_frequency_tolerance_bins * fft_resolution_MHz
    merge_frequency_tolerance_MHz = merge_frequency_tolerance_bins * fft_resolution_MHz
    dedup_frequency_tolerance_MHz = dedup_frequency_tolerance_bins * fft_resolution_MHz
    if track_decay_tolerance_per_us is None:
        track_decay_tolerance_per_us = 2. * np.pi * track_frequency_tolerance_MHz
    if dedup_decay_tolerance_per_us is None:
        dedup_decay_tolerance_per_us = track_decay_tolerance_per_us
    if not np.isfinite(track_decay_tolerance_per_us) or track_decay_tolerance_per_us <= 0.:
        raise ValueError("track_decay_tolerance_per_us must be finite and positive")
    if not np.isfinite(dedup_decay_tolerance_per_us) or dedup_decay_tolerance_per_us <= 0.:
        raise ValueError("dedup_decay_tolerance_per_us must be finite and positive")
    ################################################################################
    ####   Safety checks end                  ######################################
    ################################################################################

    def wrap_frequency(frequency_MHz):
        """
        Nested function to place estimated frequency within the first Nyquist zone.
        This is done by first add, and then get the quotient value, and then
        subtract the added value again.
        """
        return (np.asarray(frequency_MHz) + nyquist_MHz) % sampling_frequency_MHz - nyquist_MHz

    def frequency_distance(first_MHz, second_MHz):
        """
        Calculate thre frequency diffrence between first_MHz and second MHz.
        The first_MHz can be a list or an array, when the return is the 
        list of difference. 
        The function is used to compare the estimated frequency 
        from \Sigma_r^{-1} U_r^\dagger H_1 V_r and 
        \Sigma_{r-1}^{-1} U_{r-1}^\dagger H_1 V_{r-1}
        """
        return np.abs(wrap_frequency(np.asarray(first_MHz) - second_MHz))

    def circular_frequency_center(frequencies_MHz, weights):
        """
        Convert angle in to a circle, which is basically a quotient space on R1 with
        the sampling frequency. This is to avoid the pathologial average due to finite
        sampling frequency; e.g., For 100MHz f_s, -49 MHz and 49 MHz gives 0.

        The basic algorithm is first convert frequency into phase,
        calculate the vector on IQ plane, average the vector and then
        return the angle of the averaged vector.
        """
        frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
        weights = np.asarray(weights, dtype=float)
        phases = 2. * np.pi * frequencies_MHz / sampling_frequency_MHz
        weighted_vector = np.sum(weights * np.exp(1j * phases))
        if np.abs(weighted_vector) < np.finfo(float).eps: 
            # If the vectors are all compensated to give zero, which is highly unlikely,
            # then gives the median of frequencies as an alternative.
            return float(wrap_frequency(np.median(frequencies_MHz)))
        return float(wrap_frequency(np.angle(weighted_vector) * sampling_frequency_MHz / (2. * np.pi)))

    row_candidates = []
    row_diagnostics = []
    #--- A. Row MPM iteraction initiation--------------------------------------------------
    for row_index, row in enumerate(normalized_A):
        trace_analysis = analyze_matrix_pencil_trace(
            row,
            time_us,
            requested_max_modes=requested_max_modes,
            pencil_length=pencil_length,
            minimum_consecutive_ranks=minimum_consecutive_ranks,
            track_frequency_tolerance_bins=track_frequency_tolerance_bins,
            dedup_frequency_tolerance_bins=dedup_frequency_tolerance_bins,
            track_decay_tolerance_per_us=track_decay_tolerance_per_us,
            dedup_decay_tolerance_per_us=dedup_decay_tolerance_per_us,
            match_decay=match_decay,
            numerical_floor=numerical_floor,
            noise_singular_value_factor=noise_singular_value_factor,
            minimum_pole_radius=minimum_pole_radius,
            maximum_pole_radius=maximum_pole_radius,
            require_early_start=require_early_start,
            rank_sweep_extra=rank_sweep_extra,
            clip_growth=clip_growth,
            least_squares_rcond=least_squares_rcond,
            store_rank_sweeps=store_rank_sweeps)

        for candidate in trace_analysis.candidates:
            candidate.row_index = row_index
            candidate.occupation = occupations[row_index]
            row_candidates.append(candidate)
        diagnostic = trace_analysis.diagnostic
        diagnostic.occupation = occupations[row_index]
        row_diagnostics.append(diagnostic)
    #--- B. Sortitng and MPM iteraction initiation--------------------------------------------------
    #--- 1. Order row_candidates in the order of confidence and then merge them
    #       when the frequency distance is smaller than merge_frequency_tolerance_MHz
    ordered_candidates = sorted(row_candidates, 
                                key=lambda candidate: -candidate.confidence)
    clusters = []
    for candidate in ordered_candidates:
        compatible_clusters = []
        for cluster_index, cluster in enumerate(clusters):
            existing_rows = {member.row_index for member in cluster.members}
            if candidate.row_index in existing_rows:
                continue
            distance_MHz = frequency_distance(candidate.frequency_MHz, 
                                              cluster.frequency_MHz)
            if distance_MHz <= merge_frequency_tolerance_MHz:
                compatible_clusters.append((distance_MHz, cluster_index))
        if not compatible_clusters:
            clusters.append(AttrDict(dict(frequency_MHz=candidate.frequency_MHz, 
                                          members=[candidate])))
            continue
        _, nearest_cluster_index = min(compatible_clusters)
        cluster = clusters[nearest_cluster_index]
        cluster.members.append(candidate)
        cluster.frequency_MHz = circular_frequency_center([member.frequency_MHz for member in cluster.members], 
                                                          [member.confidence for member in cluster.members])

    #--- 2. Discard candidates which appeared less than `minimum_supporting_rows`; 
    #       e.g. if if `minimum_supporting_rows` = 2 and the pole has appeard in only one row, it is discarded
    merged_candidates = []
    for cluster in clusters:
        members = cluster.members
        supporting_rows = sorted({member.row_index for member in members})
        if len(supporting_rows) < minimum_supporting_rows:
            continue
        member_weights = np.asarray([member.confidence for member in members])
        frequency_MHz = circular_frequency_center([member.frequency_MHz for member in members], member_weights)
        frequency_scatter_MHz = float(np.max([frequency_distance(member.frequency_MHz, frequency_MHz) for member in members]))
        decay_values = np.asarray([member.decay_per_us for member in members])
        raw_decay_per_us = float(np.median(decay_values))
        decay_per_us = max(0., raw_decay_per_us) if clip_growth else raw_decay_per_us
        rank_spans = np.asarray([member.rank_span for member in members])
        confidence = len(supporting_rows) * np.median(rank_spans) / (1. + frequency_scatter_MHz / merge_frequency_tolerance_MHz)
        merged_candidates.append(AttrDict(dict(frequency_MHz=frequency_MHz,
                                               raw_decay_per_us=raw_decay_per_us,
                                               decay_per_us=decay_per_us,
                                               implied_growth=raw_decay_per_us < 0.,
                                               supporting_rows=supporting_rows,
                                               supporting_occupations=[occupations[row] for row in supporting_rows],
                                               median_rank_span=float(np.median(rank_spans)),
                                               frequency_scatter_MHz=frequency_scatter_MHz,
                                               decay_scatter_per_us=float(np.max(np.abs(decay_values - raw_decay_per_us))),
                                               confidence=float(confidence),
                                               members=members)))

    #--- 3. Select upto `requested_max_modes`
    merged_candidates.sort(key=lambda candidate: (-candidate.confidence, -len(candidate.supporting_rows), candidate.frequency_scatter_MHz))
    selected_candidates = merged_candidates[:min(requested_max_modes, sample_count - 1)]
    selected_candidates.sort(key=lambda candidate: candidate.frequency_MHz)
    if not selected_candidates:
        raise RuntimeError("no stable rowwise Matrix-Pencil candidates were found")


    #--- 4. Extract amplitude of each pole using lstsq, which estimates x for A x = b with least square difference.
    selected_frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected_candidates])
    selected_decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected_candidates])
    shared_poles = np.exp((-selected_decay_per_us - 2j * np.pi * selected_frequencies_MHz) * sample_time_us) # The list of poles; [z_1, z_2, ..., z_K]

    sample_index = np.arange(sample_count) #The arange upto N-1, where N is the number of time samples
    design = shared_poles[None, :] ** sample_index[:, None] #broadcast z into row, time into column, to make (transposed) vandermonde matrix. Shape: [N, K]
    normalized_amplitudes = np.zeros((len(occupations), len(selected_candidates)), dtype=complex)
    normalized_fitted_return = np.zeros_like(normalized_A)

    for row_index, normalized_row in enumerate(normalized_A):
        row_amplitudes, _, _, _ = np.linalg.lstsq(design, 
                                                  normalized_row,
                                                  rcond=least_squares_rcond) #Finds design * x = normalized_row. The solution is the amplitude for each row.
        normalized_amplitudes[row_index] = row_amplitudes
        normalized_fitted_return[row_index] = design @ row_amplitudes #Finds design * row_amp = fitted row


    amplitudes = normalized_amplitudes * row_normalization
    fitted_return = normalized_fitted_return * row_normalization
    residual = A - fitted_return
    residual_norm_by_row = np.linalg.norm(residual, axis=1)
    signal_norm_by_row = np.linalg.norm(A, axis=1)
    relative_residual_by_row = residual_norm_by_row / np.maximum(signal_norm_by_row, np.finfo(float).eps)
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(A))

    #--- Below are redoing fft; but I think the below can be deleted. The dependency got too complicated, so i am leaving it as is.
    fft_window = spectrum.get("fft_window", "raw")
    windows = {"raw": np.ones, 
               "hann": np.hanning, 
               "hamming": np.hamming, 
               "blackman": np.blackman}
    if fft_window not in windows:
        raise ValueError("Matrix-Pencil display requires a supported spectrum.fft_window")
    zero_padding = spectrum.get("zero_padding", None)
    if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
        raise ValueError("Matrix Pencil requires an unmerged spectrum with integer zero_padding")
    n_fft = zero_padding * sample_count
    energy_MHz = np.asarray(spectrum.energy_MHz)
    expected_energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time_us))
    if energy_MHz.shape != expected_energy_MHz.shape or not np.allclose(energy_MHz, expected_energy_MHz):
        raise ValueError("Matrix Pencil requires the original uniform FFT energy grid")
    window = windows[fft_window](sample_count)
    fft_scale = n_fft / np.sum(window)
    normalization = np.asarray(spectrum.fft_normalization)[:, None]
    reconstructed_local = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(fitted_return * window, n=n_fft, axis=1), axes=1))
    reconstructed_local /= normalization
    reconstructed = np.sum(reconstructed_local, axis=0)
    spectral_amplitudes = np.where(diagonal[:, None], normalized_amplitudes, amplitudes)
    pole_local_weights = np.real(spectral_amplitudes)
    pole_complex_DOS_weights = np.sum(spectral_amplitudes, axis=0)
    pole_DOS_weights = np.real(pole_complex_DOS_weights)
    pole_DOS_imaginary_weights = np.imag(pole_complex_DOS_weights)
    pole_local_magnitude_weights = np.abs(spectral_amplitudes)
    pole_amplitude_magnitude_sums = np.sum(pole_local_magnitude_weights, axis=0)
    row_weight_sums = np.sum(pole_local_weights, axis=1)
    total_DOS_weight = float(np.sum(pole_DOS_weights))
    coherent_trace_weights = np.abs(pole_complex_DOS_weights)
    supporting_row_counts = np.asarray([len(candidate.supporting_rows) for candidate in selected_candidates])

    settings = AttrDict(dict(requested_max_modes=int(requested_max_modes),
                             pencil_length=int(pencil_length),
                             minimum_consecutive_ranks=int(minimum_consecutive_ranks),
                             minimum_supporting_rows=int(minimum_supporting_rows),
                             track_frequency_tolerance_bins=float(track_frequency_tolerance_bins),
                             merge_frequency_tolerance_bins=float(merge_frequency_tolerance_bins),
                             dedup_frequency_tolerance_bins=float(dedup_frequency_tolerance_bins),
                             track_frequency_tolerance_MHz=float(track_frequency_tolerance_MHz),
                             merge_frequency_tolerance_MHz=float(merge_frequency_tolerance_MHz),
                             dedup_frequency_tolerance_MHz=float(dedup_frequency_tolerance_MHz),
                             track_decay_tolerance_per_us=float(track_decay_tolerance_per_us),
                             dedup_decay_tolerance_per_us=float(dedup_decay_tolerance_per_us),
                             match_decay=bool(match_decay),
                             numerical_floor=float(numerical_floor),
                             noise_singular_value_factor=float(noise_singular_value_factor),
                             minimum_pole_radius=float(minimum_pole_radius),
                             maximum_pole_radius=float(maximum_pole_radius),
                             require_early_start=bool(require_early_start),
                             rank_sweep_extra=rank_sweep_extra,
                             clip_growth=bool(clip_growth),
                             least_squares_rcond=least_squares_rcond,
                             fft_window=fft_window,
                             zero_padding=int(zero_padding)))
    modes = AttrDict(dict(frequencies_MHz=selected_frequencies_MHz,
                          decay_per_us=selected_decay_per_us,
                          poles=shared_poles,
                          supporting_row_counts=supporting_row_counts,
                          supporting_rows=[candidate.supporting_rows for candidate in selected_candidates],
                          supporting_occupations=[candidate.supporting_occupations for candidate in selected_candidates],
                          local_complex_amplitudes=spectral_amplitudes,
                          local_weights=pole_local_weights,
                          local_magnitude_weights=pole_local_magnitude_weights,
                          complex_DOS_weights=pole_complex_DOS_weights,
                          DOS_weights=pole_DOS_weights,
                          DOS_imaginary_weights=pole_DOS_imaginary_weights,
                          amplitude_magnitude_sums=pole_amplitude_magnitude_sums,
                          row_weight_sums=row_weight_sums,
                          total_DOS_weight=total_DOS_weight,
                          coherent_trace_weights=coherent_trace_weights))
    fit = AttrDict(dict(amplitudes=amplitudes,
                        normalized_amplitudes=normalized_amplitudes,
                        fitted_return=fitted_return,
                        normalized_fitted_return=normalized_fitted_return,
                        residual=residual,
                        relative_residual=relative_residual,
                        relative_residual_by_row=relative_residual_by_row,
                        design_condition_number=float(np.linalg.cond(design))))
    spectra = AttrDict(dict(energy_MHz=energy_MHz,
                            measured_local=np.asarray(spectrum.measured_local),
                            reconstructed_local=reconstructed_local,
                            measured=np.asarray(spectrum.measured),
                            reconstructed=reconstructed,
                            complete_basis=bool(spectrum.complete_basis)))
    return AttrDict(dict(method="matrix_pencil",
                         occupations=occupations,
                         row_normalization=row_normalization[:, 0],
                         settings=settings,
                         sampling=AttrDict(dict(time_us=time_us,
                                                sample_time_us=sample_time_us,
                                                sampling_frequency_MHz=sampling_frequency_MHz,
                                                nyquist_MHz=nyquist_MHz,
                                                fft_resolution_MHz=fft_resolution_MHz,
                                                frequency_branch_note=f"frequencies are principal aliases modulo {sampling_frequency_MHz:.6g} MHz")),
                         modes=modes,
                         fit=fit,
                         spectra=spectra,
                         candidates=AttrDict(dict(per_row=row_candidates,
                                                  merged=merged_candidates,
                                                  selected=selected_candidates)),
                         row_diagnostics=row_diagnostics,
                         selected_frequencies_MHz=selected_frequencies_MHz,
                         selected_decay_per_us=selected_decay_per_us,
                         selected_candidates=selected_candidates,
                         amplitudes=amplitudes,
                         normalized_amplitudes=normalized_amplitudes,
                         fitted_return=fitted_return,
                         normalized_fitted_return=normalized_fitted_return,
                         residual=residual,
                         relative_residual=relative_residual,
                         relative_residual_by_row=relative_residual_by_row,
                         design_condition_number=float(np.linalg.cond(design)),
                         energy_MHz=energy_MHz,
                         reconstructed_local=reconstructed_local,
                         reconstructed=reconstructed,
                         pole_local_weights=pole_local_weights,
                         pole_local_magnitude_weights=pole_local_magnitude_weights,
                         pole_complex_DOS_weights=pole_complex_DOS_weights,
                         pole_DOS_weights=pole_DOS_weights,
                         pole_DOS_imaginary_weights=pole_DOS_imaginary_weights,
                         pole_amplitude_magnitude_sums=pole_amplitude_magnitude_sums,
                         row_weight_sums=row_weight_sums,
                         total_DOS_weight=total_DOS_weight,
                         complete_basis=bool(spectrum.complete_basis)))



def analyze_matrix_pencil_trace(trace,
                                time_us,
                                requested_max_modes=None,
                                pencil_length=None,
                                minimum_consecutive_ranks=3,
                                track_frequency_tolerance_bins=1.5,
                                dedup_frequency_tolerance_bins=None,
                                track_decay_tolerance_per_us=None,
                                dedup_decay_tolerance_per_us=None,
                                match_decay=True,
                                numerical_floor=1e-10,
                                noise_singular_value_factor=2.858,
                                minimum_pole_radius=0.2,
                                maximum_pole_radius=1.05,
                                require_early_start=True,
                                rank_sweep_extra=None,
                                clip_growth=True,
                                least_squares_rcond=None,
                                store_rank_sweeps=False):
    """
    Apply the rowwise Matrix-Pencil analysis to one complex time trace.

    This is the independent analysis performed inside each iteration of
    :meth:`analyze_matrix_pencil`: Hankel construction, SVD and rank sweep,
    pole-history tracking, stability filtering, within-trace deduplication,
    and a final least-squares amplitude fit. It requires only one complex
    trace and uniformly spaced time points; it does not require a spectrum,
    reconstruction, experiment, or calculated Hamiltonian.

    The input is divided by ``trace[0]`` for conditioning. The returned
    ``normalized_amplitudes`` fit that normalized trace, while
    ``amplitudes`` and ``fitted_return`` are restored to the original
    input scale. Therefore, for

        trace = sum_i A_i(t) / A_i(0),

    the ideal real parts of ``amplitudes`` are the delta-functional DOS
    weights, including degeneracy multiplicities.
    """

    trace = np.asarray(trace, dtype=complex)
    time_us = np.asarray(time_us, dtype=float)
    if trace.ndim != 1 or time_us.ndim != 1 or len(trace) != len(time_us):
        raise ValueError("trace and time_us must be one-dimensional arrays of equal length")
    if len(time_us) < 5:
        raise ValueError("Matrix Pencil requires at least five time points")
    if not np.isclose(time_us[0], 0.):
        raise ValueError("Matrix Pencil trace analysis requires the zero-time point")
    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError("Matrix Pencil requires uniformly spaced time points")
    if not np.isfinite(numerical_floor) or numerical_floor <= 0.:
        raise ValueError("numerical_floor must be finite and positive")
    initial_return = trace[0]
    if np.abs(initial_return) <= numerical_floor:
        initial_return = 1.
    normalized_return = trace / initial_return

    if pencil_length is None:
        pencil_length = len(time_us) // 2
    if not isinstance(pencil_length, (int, np.integer)) or isinstance(pencil_length, (bool, np.bool_)) or pencil_length < 1 or pencil_length >= len(time_us):
        raise ValueError("pencil_length must be an integer between 1 and sample_count - 1")
    if requested_max_modes is None:
        requested_max_modes = min(pencil_length, len(time_us) - pencil_length)
    if not isinstance(requested_max_modes, (int, np.integer)) or isinstance(requested_max_modes, (bool, np.bool_)) or requested_max_modes < 1:
        raise ValueError("requested_max_modes must be a positive integer")
    if not isinstance(minimum_consecutive_ranks, (int, np.integer)) or isinstance(minimum_consecutive_ranks, (bool, np.bool_)) or minimum_consecutive_ranks < 1:
        raise ValueError("minimum_consecutive_ranks must be a positive integer")
    maximum_algebraic_rank = min(requested_max_modes, pencil_length, len(time_us) - pencil_length)
    if maximum_algebraic_rank < minimum_consecutive_ranks:
        raise ValueError(f"this Matrix-Pencil configuration permits at most {maximum_algebraic_rank} ranks, fewer than minimum_consecutive_ranks={minimum_consecutive_ranks}")
    if rank_sweep_extra is not None and (not isinstance(rank_sweep_extra, (int, np.integer)) or isinstance(rank_sweep_extra, (bool, np.bool_)) or rank_sweep_extra < 0):
        raise ValueError("rank_sweep_extra must be None or a nonnegative integer")
    if not np.isfinite(track_frequency_tolerance_bins) or track_frequency_tolerance_bins <= 0.:
        raise ValueError("track_frequency_tolerance_bins must be finite and positive")
    if dedup_frequency_tolerance_bins is None:
        dedup_frequency_tolerance_bins = track_frequency_tolerance_bins
    if not np.isfinite(dedup_frequency_tolerance_bins) or dedup_frequency_tolerance_bins <= 0.:
        raise ValueError("dedup_frequency_tolerance_bins must be finite and positive")
    if not np.isfinite(noise_singular_value_factor) or noise_singular_value_factor <= 0.:
        raise ValueError("noise_singular_value_factor must be finite and positive")
    if not np.isfinite(minimum_pole_radius) or not np.isfinite(maximum_pole_radius) or minimum_pole_radius <= 0. or maximum_pole_radius <= minimum_pole_radius:
        raise ValueError("pole radii must satisfy 0 < minimum_pole_radius < maximum_pole_radius")
    if least_squares_rcond is not None and (not np.isfinite(least_squares_rcond) or least_squares_rcond < 0.):
        raise ValueError("least_squares_rcond must be None or finite and nonnegative")

    sample_count = len(time_us)
    sampling_frequency_MHz = 1. / sample_time_us
    nyquist_MHz = 0.5 * sampling_frequency_MHz
    fft_resolution_MHz = 1. / (sample_count * sample_time_us)
    track_frequency_tolerance_MHz = track_frequency_tolerance_bins * fft_resolution_MHz
    dedup_frequency_tolerance_MHz = dedup_frequency_tolerance_bins * fft_resolution_MHz
    if track_decay_tolerance_per_us is None:
        track_decay_tolerance_per_us = 2. * np.pi * track_frequency_tolerance_MHz
    if dedup_decay_tolerance_per_us is None:
        dedup_decay_tolerance_per_us = track_decay_tolerance_per_us
    if not np.isfinite(track_decay_tolerance_per_us) or track_decay_tolerance_per_us <= 0.:
        raise ValueError("track_decay_tolerance_per_us must be finite and positive")
    if not np.isfinite(dedup_decay_tolerance_per_us) or dedup_decay_tolerance_per_us <= 0.:
        raise ValueError("dedup_decay_tolerance_per_us must be finite and positive")

    def wrap_frequency(frequency_MHz):
        return (np.asarray(frequency_MHz) + nyquist_MHz) % sampling_frequency_MHz - nyquist_MHz

    def frequency_distance(first_MHz, second_MHz):
        return np.abs(wrap_frequency(np.asarray(first_MHz) - second_MHz))

    def circular_frequency_center(frequencies_MHz, weights):
        frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
        weights = np.asarray(weights, dtype=float)
        phases = 2. * np.pi * frequencies_MHz / sampling_frequency_MHz
        weighted_vector = np.sum(weights * np.exp(1j * phases))
        if np.abs(weighted_vector) < np.finfo(float).eps:
            return float(wrap_frequency(np.median(frequencies_MHz)))
        return float(wrap_frequency(np.angle(weighted_vector) * sampling_frequency_MHz / (2. * np.pi)))

    row_candidates = []
    row_diagnostics = []
    occupations = [("trace",)]
    #--- A. Row MPM iteraction initiation--------------------------------------------------
    for row_index, row in enumerate([normalized_return]):
        row_norm = np.linalg.norm(row)
        if row_norm <= numerical_floor: 
        #Disregard the row if there is no peak; the criteria is numerical_floor, which is adhoc
            row_diagnostics.append(AttrDict(dict(
                occupation=occupations[row_index],
                estimated_signal_rank=0,
                maximum_rank=0, 
                singular_values=np.array([]),
                candidates=[])))
            continue


        scaled_row = row / row_norm

        #--- 1. Hankel Matrix Formation-----------------------------------------------------
        windows = sliding_window_view(scaled_row, pencil_length + 1) # primitive for Hankel 
        unshifted = windows[:, :-1] # H_0 by discarding the last columns
        shifted = windows[:, 1:] # H_1 by discarding the first columns

        #--- 2. Thin SVD of unshifted (H_0) matrix.
        #--- singular_values is 1d array. others are 2d array matrix.
        #--- Thesholding can be done in a different way.
        left_vectors, singular_values, right_vectors_h = np.linalg.svd(unshifted, 
                                                                       full_matrices=False)
        relative_singular_values = singular_values / singular_values[0]       
        singular_value_threshold = noise_singular_value_factor * np.median(singular_values) #as we do not know the singular value of noise, it is replaced by median value.

        #--- 3. Prepare iterative MPM after setting up signal rank, numerical rank.
        estimated_signal_rank = max(1, int(np.count_nonzero(singular_values > singular_value_threshold)))
        numerical_rank = int(np.count_nonzero(relative_singular_values > numerical_floor))
        maximum_rank = min(requested_max_modes, unshifted.shape[0], unshifted.shape[1], numerical_rank)
        if rank_sweep_extra is not None:
            maximum_rank = min(maximum_rank, estimated_signal_rank + rank_sweep_extra)

        #--- 4. Execute MPM by sweeping rank from 1 to maximum_rank
        rank_solutions = []
        for rank in range(1, maximum_rank + 1):
            left = left_vectors[:, :rank]
            right = right_vectors_h[:rank].conj().T
            shifted_reduced = left.conj().T @ shifted @ right
            # np.linalg.solve(A, B) does A^{-1} B
            reduced_pencil = np.linalg.solve(np.diag(singular_values[:rank]), shifted_reduced)
            poles = np.linalg.eigvals(reduced_pencil)
            pole_radii = np.abs(poles)
            # I am not really sure if this masking is necessary. There are already tons of
            # other validation steps, including signal rank and things like that
            # plus, minimum/maximum_pole_radius is somewhat ad hoc
            valid = np.isfinite(poles) & np.isfinite(pole_radii) & (pole_radii >= minimum_pole_radius) & (pole_radii <= maximum_pole_radius)
            poles = poles[valid]
            pole_radii = pole_radii[valid]
            frequencies_MHz = -np.angle(poles) / (2. * np.pi * sample_time_us)
            decay_per_us = -np.log(pole_radii) / sample_time_us
            order = np.argsort(frequencies_MHz) # sort in an ascending order
            rank_solutions.append(AttrDict(dict(rank=rank,
                                                frequencies_MHz=frequencies_MHz[order],
                                                decay_per_us=decay_per_us[order],
                                                pole_radii=pole_radii[order])))
        #--- 5. Classification of poles; the poles are grouped when they are within tolerance
        pole_histories = []
        first_solution = rank_solutions[0]
        for pole_index, frequency_MHz in enumerate(first_solution.frequencies_MHz):
            pole_histories.append(AttrDict(dict(ranks=[first_solution.rank],
                                                 frequencies_MHz=[float(frequency_MHz)],
                                                 decay_per_us=[float(first_solution.decay_per_us[pole_index])],
                                                 pole_radii=[float(first_solution.pole_radii[pole_index])])))

        for solution in rank_solutions[1:]: #Note: solution is AttrDict
            current_rank = solution.rank
            previous_rank = current_rank - 1

            # Record every compatible (previous history, current pole) pair and its distance.
            candidate_matches = []
            for pole_history in pole_histories:
                # This loop is to pick poles that can be thought of as the same pole
                # found previously. If there is a pole that is "same", the AttrDict of
                # distance, pole_history, and index is stored.
                # The reason of collecting candidate_matches first is to consider
                # all possible combinations of distance and find the best track
                last_rank_where_pole_was_found = pole_history.ranks[-1]
                if last_rank_where_pole_was_found != previous_rank:
                    continue

                previous_frequency_MHz = pole_history.frequencies_MHz[-1]
                previous_decay_per_us = pole_history.decay_per_us[-1]
                for current_pole_index, current_frequency_MHz in enumerate(solution.frequencies_MHz):
                    # This loop is to pick pole index of the frequency
                    # that appeared previously.
                    frequency_difference_MHz = frequency_distance(current_frequency_MHz, 
                                                                  previous_frequency_MHz)
                    decay_difference_per_us = np.abs(solution.decay_per_us[current_pole_index] - previous_decay_per_us)
                    frequency_is_close = frequency_difference_MHz <= track_frequency_tolerance_MHz
                    decay_is_close = decay_difference_per_us <= track_decay_tolerance_per_us
                    if not frequency_is_close or (match_decay and not decay_is_close):
                        continue

                    match_distance = frequency_difference_MHz / track_frequency_tolerance_MHz
                    if match_decay:
                        match_distance += decay_difference_per_us / track_decay_tolerance_per_us
                    candidate_matches.append(AttrDict(dict(distance=match_distance,
                                                           pole_history=pole_history,
                                                           current_pole_index=current_pole_index)))

            candidate_matches.sort(key=lambda element: element.distance) #sort candidates in the order of match_distance
            assigned_current_pole_indices = set()
            for candidate_match in candidate_matches:
                pole_history = candidate_match.pole_history #Note: dictionary variable name also acts as a reference in C++
                current_pole_index = candidate_match.current_pole_index

                history_already_extended = pole_history.ranks[-1] == current_rank
                pole_already_assigned = current_pole_index in assigned_current_pole_indices

                if history_already_extended or pole_already_assigned:
                    continue
                pole_history.ranks.append(current_rank)
                pole_history.frequencies_MHz.append(float(solution.frequencies_MHz[current_pole_index]))
                pole_history.decay_per_us.append(float(solution.decay_per_us[current_pole_index]))
                pole_history.pole_radii.append(float(solution.pole_radii[current_pole_index]))
                assigned_current_pole_indices.add(current_pole_index)

            # A current pole that matched no existing history starts a new history at this rank.
            for current_pole_index, current_frequency_MHz in enumerate(solution.frequencies_MHz):
                if current_pole_index in assigned_current_pole_indices:
                    continue
                pole_histories.append(AttrDict(dict(ranks=[current_rank],
                                                     frequencies_MHz=[float(current_frequency_MHz)],
                                                     decay_per_us=[float(solution.decay_per_us[current_pole_index])],
                                                     pole_radii=[float(solution.pole_radii[current_pole_index])])))

        #--- 6. Filter poles that did not appear consecutively or appeared after signal rank; this is to rule out poles from noise
        candidates = []
        for pole_history in pole_histories:
            rank_span = len(pole_history.ranks)
            #below only retains poles with minimum_consecutive ranks
            if rank_span < minimum_consecutive_ranks:
                continue
            #below only retains poles appears before signal rank
            if require_early_start and pole_history.ranks[0] > estimated_signal_rank:
                continue
            frequencies_MHz = np.asarray(pole_history.frequencies_MHz)
            decay_per_us = np.asarray(pole_history.decay_per_us)
            frequency_MHz = circular_frequency_center(frequencies_MHz, 
                                                      np.ones(rank_span))
            frequency_scatter_MHz = float(np.max(frequency_distance(frequencies_MHz, 
                                                                    frequency_MHz)))
            median_decay_per_us = float(np.median(decay_per_us))
            decay_scatter_per_us = float(np.max(np.abs(decay_per_us - median_decay_per_us)))
            confidence_denominator = 1. + frequency_scatter_MHz / track_frequency_tolerance_MHz
            if match_decay:
                confidence_denominator += decay_scatter_per_us / track_decay_tolerance_per_us
            candidates.append(AttrDict(dict(row_index=row_index,
                                            occupation=occupations[row_index],
                                            frequency_MHz=frequency_MHz,
                                            decay_per_us=median_decay_per_us,
                                            pole_radius=float(np.median(pole_history.pole_radii)),
                                            first_rank=pole_history.ranks[0],
                                            last_rank=pole_history.ranks[-1],
                                            rank_span=rank_span,
                                            frequency_scatter_MHz=frequency_scatter_MHz,
                                            decay_scatter_per_us=decay_scatter_per_us,
                                            confidence=float(rank_span / confidence_denominator))))


        candidates.sort(key=lambda candidate: (-candidate.rank_span, #to sort large values first
                                               candidate.frequency_scatter_MHz, 
                                               candidate.decay_scatter_per_us))

        #--- 7. Deduplicate frequency poles from candidates; the criteria is dedup_frequency_tolerance_MHz, which is identical to the track_frequency_tolerance_MHz by defaultt
        unique_candidates = []
        for candidate in candidates:
            # This loop is to discard duplicated peaked
            duplicate = False
            for existing in unique_candidates:
                duplicate = frequency_distance(candidate.frequency_MHz, 
                                               existing.frequency_MHz) <= dedup_frequency_tolerance_MHz
                if match_decay:
                    duplicate = duplicate and np.abs(candidate.decay_per_us - existing.decay_per_us) <= dedup_decay_tolerance_per_us
                if duplicate:
                    break
            if not duplicate:
                unique_candidates.append(candidate)
        row_candidates.extend(unique_candidates)
        diagnostic = AttrDict(dict(occupation=occupations[row_index],
                                   estimated_signal_rank=estimated_signal_rank,
                                   maximum_rank=maximum_rank,
                                   singular_values=singular_values,
                                   relative_singular_values=relative_singular_values,
                                   candidates=unique_candidates))
        if store_rank_sweeps:
            diagnostic.rank_solutions = rank_solutions
            diagnostic.tracks = pole_histories
        row_diagnostics.append(diagnostic)

    selected_candidates = row_candidates[:min(requested_max_modes, sample_count - 1)]
    selected_candidates.sort(key=lambda candidate: candidate.frequency_MHz)
    selected_frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected_candidates])
    raw_decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected_candidates])
    selected_decay_per_us = np.maximum(raw_decay_per_us, 0.) if clip_growth else raw_decay_per_us.copy()
    shared_poles = np.exp((-selected_decay_per_us - 2j * np.pi * selected_frequencies_MHz) * sample_time_us)
    sample_index = np.arange(sample_count)
    design = shared_poles[None, :] ** sample_index[:, None]

    if len(selected_candidates):
        normalized_amplitudes, _, _, _ = np.linalg.lstsq(design,
                                                         normalized_return,
                                                         rcond=least_squares_rcond)
        normalized_fitted_return = design @ normalized_amplitudes
        design_condition_number = float(np.linalg.cond(design))
    else:
        normalized_amplitudes = np.array([], dtype=complex)
        normalized_fitted_return = np.zeros_like(normalized_return)
        design_condition_number = np.nan
    amplitudes = normalized_amplitudes * initial_return
    fitted_return = normalized_fitted_return * initial_return
    residual = trace - fitted_return
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(trace))
    diagnostic = row_diagnostics[0]

    return AttrDict(dict(method="matrix_pencil_trace",
                         trace=trace,
                         time_us=time_us,
                         initial_return=initial_return,
                         normalized_return=normalized_return,
                         candidates=row_candidates,
                         selected_candidates=selected_candidates,
                         selected_frequencies_MHz=selected_frequencies_MHz,
                         selected_raw_decay_per_us=raw_decay_per_us,
                         selected_decay_per_us=selected_decay_per_us,
                         poles=shared_poles,
                         normalized_amplitudes=normalized_amplitudes,
                         amplitudes=amplitudes,
                         DOS_weights=np.real(amplitudes),
                         DOS_imaginary_weights=np.imag(amplitudes),
                         amplitude_magnitudes=np.abs(amplitudes),
                         normalized_fitted_return=normalized_fitted_return,
                         fitted_return=fitted_return,
                         residual=residual,
                         relative_residual=relative_residual,
                         design_condition_number=design_condition_number,
                         diagnostic=diagnostic,
                         sampling=AttrDict(dict(sample_time_us=sample_time_us,
                                                sampling_frequency_MHz=sampling_frequency_MHz,
                                                nyquist_MHz=nyquist_MHz,
                                                fft_resolution_MHz=fft_resolution_MHz,
                                                frequency_branch_note=f"frequencies are principal aliases modulo {sampling_frequency_MHz:.6g} MHz")),
                         settings=AttrDict(dict(requested_max_modes=int(requested_max_modes),
                                                pencil_length=int(pencil_length),
                                                minimum_consecutive_ranks=int(minimum_consecutive_ranks),
                                                track_frequency_tolerance_bins=float(track_frequency_tolerance_bins),
                                                dedup_frequency_tolerance_bins=float(dedup_frequency_tolerance_bins),
                                                track_frequency_tolerance_MHz=float(track_frequency_tolerance_MHz),
                                                dedup_frequency_tolerance_MHz=float(dedup_frequency_tolerance_MHz),
                                                track_decay_tolerance_per_us=float(track_decay_tolerance_per_us),
                                                dedup_decay_tolerance_per_us=float(dedup_decay_tolerance_per_us),
                                                match_decay=bool(match_decay),
                                                numerical_floor=float(numerical_floor),
                                                noise_singular_value_factor=float(noise_singular_value_factor),
                                                minimum_pole_radius=float(minimum_pole_radius),
                                                maximum_pole_radius=float(maximum_pole_radius),
                                                require_early_start=bool(require_early_start),
                                                rank_sweep_extra=rank_sweep_extra,
                                                clip_growth=bool(clip_growth),
                                                least_squares_rcond=least_squares_rcond))))




def refit_occupation(occupation,
                     data,
                     matrix_pencil=None,
                     least_squares_rcond=None):
    """
    Refit one occupation using only the poles detected independently in that row.

    This is different from taking one row of the global Matrix-Pencil fit. The
    global fit uses every shared pole selected after candidates from all rows are
    merged. This wrapper instead uses ``row_diagnostics[row].candidates`` and
    therefore shows the Matrix-Pencil result supported by the selected occupation
    before the cross-row merge.
    """
    if "reconstruction" not in data or "spectrum" not in data:
        raise ValueError("occupation Matrix-Pencil analysis requires analyzed spectroscopy data")
    if data.get("spectrum_only", False):
        raise ValueError("occupation Matrix Pencil requires the original occupation time traces")
    if matrix_pencil is None:
        matrix_pencil = data.get("matrix_pencil", None)
    if matrix_pencil is None:
        raise ValueError("Matrix-Pencil analysis is unavailable; analyze with spectrum_method='matrix_pencil'")

    occupations = [tuple(value) for value in data.reconstruction.occupations]
    if [tuple(value) for value in matrix_pencil.occupations] != occupations:
        raise ValueError("Matrix-Pencil occupations do not match the spectroscopy reconstruction")
    if isinstance(occupation, (int, np.integer)):
        row = int(occupation)
        if row < 0 or row >= len(occupations):
            raise IndexError("occupation row is outside the spectroscopy data")
    else:
        occupation = tuple(occupation)
        if occupation not in occupations:
            raise ValueError(f"{occupation} is not in the spectroscopy data")
        row = occupations.index(occupation)
    occupation = occupations[row]

    diagnostic = matrix_pencil.row_diagnostics[row]
    row_candidates = sorted(list(diagnostic.candidates), key=lambda candidate: candidate.frequency_MHz)
    frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in row_candidates], dtype=float)
    raw_decay_per_us = np.asarray([candidate.decay_per_us for candidate in row_candidates], dtype=float)
    if matrix_pencil.settings.clip_growth:
        decay_per_us = np.maximum(raw_decay_per_us, 0.)
    else:
        decay_per_us = raw_decay_per_us.copy()

    time_us = np.asarray(matrix_pencil.sampling.time_us, dtype=float)
    sample_time_us = float(matrix_pencil.sampling.sample_time_us)
    measured_return = np.asarray(data.reconstruction.A[row], dtype=complex)
    initial_return = matrix_pencil.row_normalization[row]
    normalized_return = measured_return / initial_return
    sample_index = np.arange(len(time_us))
    poles = np.exp((-decay_per_us - 2j * np.pi * frequencies_MHz) * sample_time_us)
    design = poles[None, :] ** sample_index[:, None]
    if least_squares_rcond is None:
        least_squares_rcond = matrix_pencil.settings.least_squares_rcond
    if len(row_candidates):
        normalized_amplitudes, _, _, _ = np.linalg.lstsq(design, normalized_return, rcond=least_squares_rcond)
        normalized_fitted_return = design @ normalized_amplitudes
        design_condition_number = float(np.linalg.cond(design))
    else:
        normalized_amplitudes = np.array([], dtype=complex)
        normalized_fitted_return = np.zeros_like(normalized_return)
        design_condition_number = np.nan
    amplitudes = normalized_amplitudes * initial_return
    fitted_return = normalized_fitted_return * initial_return
    residual = measured_return - fitted_return
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(measured_return))

    windows = {"raw": np.ones, "hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}
    fft_window = matrix_pencil.settings.fft_window
    zero_padding = matrix_pencil.settings.zero_padding
    window = windows[fft_window](len(time_us))
    n_fft = zero_padding * len(time_us)
    fft_scale = n_fft / np.sum(window)
    reconstructed_spectrum = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(fitted_return * window, n=n_fft))) / data.spectrum.fft_normalization[row]
    is_diagonal = occupation == tuple(data.reconstruction.final_occupations[row])
    spectral_amplitudes = normalized_amplitudes if is_diagonal else amplitudes

    return AttrDict(dict(method="matrix_pencil_occupation",
                         row_index=row,
                         occupation=occupation,
                         diagnostic=diagnostic,
                         candidates=row_candidates,
                         time_us=time_us,
                         measured_return=measured_return,
                         normalized_return=normalized_return,
                         fitted_return=fitted_return,
                         normalized_fitted_return=normalized_fitted_return,
                         residual=residual,
                         relative_residual=relative_residual,
                         frequencies_MHz=frequencies_MHz,
                         raw_decay_per_us=raw_decay_per_us,
                         decay_per_us=decay_per_us,
                         poles=poles,
                         normalized_amplitudes=normalized_amplitudes,
                         local_weights=np.real(spectral_amplitudes),
                         local_magnitude_weights=np.abs(spectral_amplitudes),
                         design_condition_number=design_condition_number,
                         energy_MHz=np.asarray(data.spectrum.energy_MHz),
                         measured_spectrum=np.asarray(data.spectrum.measured_local[row]),
                         reconstructed_spectrum=reconstructed_spectrum))


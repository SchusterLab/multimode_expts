"""Matrix-Pencil analysis for many-body Ramsey reconstructions.

Pure numerics: arrays and settings in, an ``AttrDict`` out; no Experiment,
station or file. ``tests/test_mbr_analysis_golden.py`` pins the results.

The measured return of occupation row ``i`` is modeled as a sum of damped
exponentials with poles shared by all rows:

    A_i[n] = sum_m c_im z_m**n,    z_m = exp((-gamma_m - 2j*pi*f_m) * dt).

:func:`analyze_matrix_pencil` finds the poles and amplitudes in five steps:

1. Each row, alone (:func:`analyze_matrix_pencil_trace`, :func:`_row_candidates`):
   a. Hankel pencil of the row and its SVD; the signal rank is estimated
      from the singular values (:func:`_rank_sweep`);
   b. the poles at every assumed rank ``1 .. maximum_rank`` (same function);
   c. each pole followed from one rank to the next (:func:`_track_poles`);
   d. only poles stable over ``minimum_consecutive_ranks`` ranks kept
      (:func:`_stable_candidates`), without duplicates (:func:`_deduplicate`).
2. The row candidates clustered by frequency across rows (:func:`_cluster_across_rows`).
3. Each cluster scored; clusters with too few rows rejected (:func:`_score_clusters`).
4. The best ``requested_max_modes`` clusters are the shared poles. With the
   poles fixed, every row's amplitudes are refit by least squares, also for
   rows that did not detect a pole.
5. The FFT of the fitted returns on the spectrum's grid (:func:`_windowed_spectrum`).

No Hamiltonian or theory spectrum is used to select poles.

:func:`refit_occupation` refits one row with only its own step-1 candidates.

Frequencies are principal aliases, modulo the sampling frequency
(:class:`_FrequencyCircle`).
"""

import inspect

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from slab import AttrDict


_WINDOWS = {"raw": np.ones,
            "hann": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman}

# The occupation label a single trace gets before a caller relabels it.
_TRACE_LABEL = ("trace",)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _FrequencyCircle:
    """Frequency arithmetic modulo the sampling frequency.

    A sampled pole only defines its frequency modulo ``1/dt``. Distances and
    averages are therefore taken on that circle: for a 100 MHz sampling rate,
    -49 MHz and +49 MHz are 2 MHz apart, and their average is 50 MHz, not 0.
    """

    def __init__(self, sample_time_us, sample_count):
        self.sample_time_us = sample_time_us
        self.sampling_frequency_MHz = 1. / sample_time_us
        self.nyquist_MHz = 0.5 * self.sampling_frequency_MHz
        self.fft_resolution_MHz = 1. / (sample_count * sample_time_us)

    def wrap(self, frequency_MHz):
        """-> the alias in [-nyquist, nyquist)."""
        return (np.asarray(frequency_MHz) + self.nyquist_MHz) % self.sampling_frequency_MHz - self.nyquist_MHz

    def distance(self, first_MHz, second_MHz):
        return np.abs(self.wrap(np.asarray(first_MHz) - second_MHz))

    def center(self, frequencies_MHz, weights):
        """Weighted mean on the circle: average the unit vectors, take the angle.

        If the vectors cancel (unlikely), the median is used instead.
        """
        frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
        weights = np.asarray(weights, dtype=float)
        phases = 2. * np.pi * frequencies_MHz / self.sampling_frequency_MHz
        weighted_vector = np.sum(weights * np.exp(1j * phases))
        if np.abs(weighted_vector) < np.finfo(float).eps:
            return float(self.wrap(np.median(frequencies_MHz)))
        return float(self.wrap(np.angle(weighted_vector) * self.sampling_frequency_MHz / (2. * np.pi)))

    def sampling(self, **extra):
        return AttrDict(dict(**extra,
                             sample_time_us=self.sample_time_us,
                             sampling_frequency_MHz=self.sampling_frequency_MHz,
                             nyquist_MHz=self.nyquist_MHz,
                             fft_resolution_MHz=self.fft_resolution_MHz,
                             frequency_branch_note=(f"frequencies are principal aliases modulo "
                                                    f"{self.sampling_frequency_MHz:.6g} MHz")))


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_positive_int(value):
    return _is_int(value) and value >= 1


def _require_positive(name, value):
    if not np.isfinite(value) or value <= 0.:
        raise ValueError(f"{name} must be finite and positive")


def _check_time_grid(time_us, zero_time_use):
    """-> the sample time. The grid must start at 0, be uniform, and have 5+ points."""
    if len(time_us) < 5:
        raise ValueError("Matrix Pencil requires at least five time points")
    if not np.isclose(time_us[0], 0.):
        raise ValueError(f"Matrix Pencil {zero_time_use} requires the zero-time point")
    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError("Matrix Pencil requires uniformly spaced time points")
    return sample_time_us


def _poles(frequencies_MHz, decay_per_us, sample_time_us):
    """z = exp((-gamma - 2j pi f) dt)."""
    return np.exp((-decay_per_us - 2j * np.pi * frequencies_MHz) * sample_time_us)


def _vandermonde(poles, sample_count):
    """design[n, m] = z_m**n, so that A[n] = design @ c."""
    return poles[None, :] ** np.arange(sample_count)[:, None]


def _fit_one_row(poles, normalized_row, rcond):
    """-> (amplitudes, fitted row, design condition number); empty if no poles."""
    design = _vandermonde(poles, len(normalized_row))
    if len(poles):
        amplitudes, _, _, _ = np.linalg.lstsq(design, normalized_row, rcond=rcond)
        return amplitudes, design @ amplitudes, float(np.linalg.cond(design))
    return np.array([], dtype=complex), np.zeros_like(normalized_row), np.nan


def _windowed_spectrum(fitted_return, fft_window, zero_padding):
    """|FFT| of the fitted return(s) along the last axis, as the spectrum computes it:
    window, zero padding, inverse-FFT sign, and scaling to the window sum."""
    sample_count = fitted_return.shape[-1]
    window = _WINDOWS[fft_window](sample_count)
    n_fft = zero_padding * sample_count
    fft_scale = n_fft / np.sum(window)
    return fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(fitted_return * window, n=n_fft, axis=-1), axes=-1))


# ---------------------------------------------------------------------------
# Step 1: one row
# ---------------------------------------------------------------------------

def _row_settings(time_us, sample_time_us, requested_max_modes, pencil_length,
                  minimum_consecutive_ranks, track_frequency_tolerance_bins,
                  dedup_frequency_tolerance_bins, dedup_frequency_tolerance_MHz,
                  track_decay_tolerance_per_us, dedup_decay_tolerance_per_us,
                  match_decay, numerical_floor, noise_singular_value_factor,
                  minimum_pole_radius, maximum_pole_radius, require_early_start,
                  rank_sweep_extra, clip_growth, least_squares_rcond, store_rank_sweeps):
    """Check the per-row settings and resolve the tolerances to MHz and 1/us.

    ``requested_max_modes=None`` means as many as the pencil allows. The
    tolerances default to the tracking tolerance: frequencies in FFT bins,
    decays as ``2 pi`` times the frequency tolerance.
    """
    sample_count = len(time_us)
    if pencil_length is None:
        pencil_length = sample_count // 2
    if not _is_positive_int(pencil_length) or pencil_length >= sample_count:
        raise ValueError("pencil_length must be an integer between 1 and sample_count - 1")
    if requested_max_modes is None:
        requested_max_modes = min(pencil_length, sample_count - pencil_length)
    if not _is_positive_int(requested_max_modes):
        raise ValueError("requested_max_modes must be a positive integer")
    if not _is_positive_int(minimum_consecutive_ranks):
        raise ValueError("minimum_consecutive_ranks must be a positive integer")
    maximum_algebraic_rank = min(requested_max_modes, pencil_length, sample_count - pencil_length)
    if maximum_algebraic_rank < minimum_consecutive_ranks:
        raise ValueError(f"this Matrix-Pencil configuration permits at most {maximum_algebraic_rank} ranks, "
                         f"fewer than minimum_consecutive_ranks={minimum_consecutive_ranks}")
    if rank_sweep_extra is not None and not (_is_int(rank_sweep_extra) and rank_sweep_extra >= 0):
        raise ValueError("rank_sweep_extra must be None or a nonnegative integer")
    _require_positive("track_frequency_tolerance_bins", track_frequency_tolerance_bins)
    if dedup_frequency_tolerance_MHz is None:
        if dedup_frequency_tolerance_bins is None:
            dedup_frequency_tolerance_bins = track_frequency_tolerance_bins
        _require_positive("dedup_frequency_tolerance_bins", dedup_frequency_tolerance_bins)
    else:
        _require_positive("dedup_frequency_tolerance_MHz", dedup_frequency_tolerance_MHz)
    _require_positive("numerical_floor", numerical_floor)
    _require_positive("noise_singular_value_factor", noise_singular_value_factor)
    if not np.isfinite(minimum_pole_radius) or not np.isfinite(maximum_pole_radius) \
            or minimum_pole_radius <= 0. or maximum_pole_radius <= minimum_pole_radius:
        raise ValueError("pole radii must satisfy 0 < minimum_pole_radius < maximum_pole_radius")
    if least_squares_rcond is not None and (not np.isfinite(least_squares_rcond) or least_squares_rcond < 0.):
        raise ValueError("least_squares_rcond must be None or finite and nonnegative")

    circle = _FrequencyCircle(sample_time_us, sample_count)
    track_frequency_tolerance_MHz = track_frequency_tolerance_bins * circle.fft_resolution_MHz
    if dedup_frequency_tolerance_MHz is None:
        dedup_frequency_tolerance_MHz = dedup_frequency_tolerance_bins * circle.fft_resolution_MHz
    else:
        dedup_frequency_tolerance_bins = dedup_frequency_tolerance_MHz / circle.fft_resolution_MHz
    if track_decay_tolerance_per_us is None:
        track_decay_tolerance_per_us = 2. * np.pi * track_frequency_tolerance_MHz
    if dedup_decay_tolerance_per_us is None:
        dedup_decay_tolerance_per_us = track_decay_tolerance_per_us
    _require_positive("track_decay_tolerance_per_us", track_decay_tolerance_per_us)
    _require_positive("dedup_decay_tolerance_per_us", dedup_decay_tolerance_per_us)

    return AttrDict(dict(
        circle=circle,
        sample_count=sample_count,
        requested_max_modes=requested_max_modes,
        pencil_length=pencil_length,
        minimum_consecutive_ranks=minimum_consecutive_ranks,
        track_frequency_tolerance_bins=track_frequency_tolerance_bins,
        track_frequency_tolerance_MHz=track_frequency_tolerance_MHz,
        dedup_frequency_tolerance_bins=dedup_frequency_tolerance_bins,
        dedup_frequency_tolerance_MHz=dedup_frequency_tolerance_MHz,
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
        store_rank_sweeps=store_rank_sweeps,
    ))


def _rank_sweep(scaled_row, s):
    """1a-b. The poles of the pencil at every assumed rank.

    The Hankel matrices H_0 and H_1 (H_0 shifted by one sample) come from one
    sliding window. With the thin SVD H_0 = U S V^h, the rank-r poles are the
    eigenvalues of S_r^-1 U_r^h H_1 V_r. The signal rank is the number of
    singular values above ``noise_singular_value_factor`` times their median
    (the median stands in for the unknown noise level). Poles outside
    ``[minimum_pole_radius, maximum_pole_radius]`` are dropped.
    """
    windows = sliding_window_view(scaled_row, s.pencil_length + 1)
    unshifted = windows[:, :-1]
    shifted = windows[:, 1:]
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(unshifted, full_matrices=False)
    relative_singular_values = singular_values / singular_values[0]
    singular_value_threshold = s.noise_singular_value_factor * np.median(singular_values)

    estimated_signal_rank = max(1, int(np.count_nonzero(singular_values > singular_value_threshold)))
    numerical_rank = int(np.count_nonzero(relative_singular_values > s.numerical_floor))
    maximum_rank = min(s.requested_max_modes, unshifted.shape[0], unshifted.shape[1], numerical_rank)
    if s.rank_sweep_extra is not None:
        maximum_rank = min(maximum_rank, estimated_signal_rank + s.rank_sweep_extra)

    solutions = []
    for rank in range(1, maximum_rank + 1):
        left = left_vectors[:, :rank]
        right = right_vectors_h[:rank].conj().T
        shifted_reduced = left.conj().T @ shifted @ right
        reduced_pencil = np.linalg.solve(np.diag(singular_values[:rank]), shifted_reduced)
        poles = np.linalg.eigvals(reduced_pencil)
        pole_radii = np.abs(poles)
        valid = (np.isfinite(poles) & np.isfinite(pole_radii)
                 & (pole_radii >= s.minimum_pole_radius) & (pole_radii <= s.maximum_pole_radius))
        poles = poles[valid]
        pole_radii = pole_radii[valid]
        frequencies_MHz = -np.angle(poles) / (2. * np.pi * s.circle.sample_time_us)
        decay_per_us = -np.log(pole_radii) / s.circle.sample_time_us
        order = np.argsort(frequencies_MHz)
        solutions.append(AttrDict(dict(rank=rank,
                                       frequencies_MHz=frequencies_MHz[order],
                                       decay_per_us=decay_per_us[order],
                                       pole_radii=pole_radii[order])))
    return AttrDict(dict(solutions=solutions,
                         singular_values=singular_values,
                         relative_singular_values=relative_singular_values,
                         estimated_signal_rank=estimated_signal_rank,
                         maximum_rank=maximum_rank))


def _new_history(solution, pole_index):
    return AttrDict(dict(ranks=[solution.rank],
                         frequencies_MHz=[float(solution.frequencies_MHz[pole_index])],
                         decay_per_us=[float(solution.decay_per_us[pole_index])],
                         pole_radii=[float(solution.pole_radii[pole_index])]))


def _track_poles(solutions, s):
    """1c. Follow each pole from rank r-1 to rank r.

    A history that reached rank r-1 may take one rank-r pole within the
    tracking tolerance (in frequency, and in decay if ``match_decay``). All
    compatible pairs are assigned closest first, one pole per history. A
    rank-r pole that joins no history starts a new one.
    """
    histories = [_new_history(solutions[0], index) for index in range(len(solutions[0].frequencies_MHz))]

    for solution in solutions[1:]:
        current_rank = solution.rank
        matches = []
        for history in histories:
            if history.ranks[-1] != current_rank - 1:
                continue
            previous_frequency_MHz = history.frequencies_MHz[-1]
            previous_decay_per_us = history.decay_per_us[-1]
            for pole_index, frequency_MHz in enumerate(solution.frequencies_MHz):
                frequency_difference_MHz = s.circle.distance(frequency_MHz, previous_frequency_MHz)
                decay_difference_per_us = np.abs(solution.decay_per_us[pole_index] - previous_decay_per_us)
                frequency_is_close = frequency_difference_MHz <= s.track_frequency_tolerance_MHz
                decay_is_close = decay_difference_per_us <= s.track_decay_tolerance_per_us
                if not frequency_is_close or (s.match_decay and not decay_is_close):
                    continue
                distance = frequency_difference_MHz / s.track_frequency_tolerance_MHz
                if s.match_decay:
                    distance += decay_difference_per_us / s.track_decay_tolerance_per_us
                matches.append(AttrDict(dict(distance=distance, history=history, pole_index=pole_index)))

        matches.sort(key=lambda match: match.distance)
        assigned = set()
        for match in matches:
            history = match.history
            if history.ranks[-1] == current_rank or match.pole_index in assigned:
                continue
            history.ranks.append(current_rank)
            history.frequencies_MHz.append(float(solution.frequencies_MHz[match.pole_index]))
            history.decay_per_us.append(float(solution.decay_per_us[match.pole_index]))
            history.pole_radii.append(float(solution.pole_radii[match.pole_index]))
            assigned.add(match.pole_index)

        for pole_index in range(len(solution.frequencies_MHz)):
            if pole_index not in assigned:
                histories.append(_new_history(solution, pole_index))
    return histories


def _stable_candidates(histories, estimated_signal_rank, s):
    """1d. The histories that last ``minimum_consecutive_ranks`` ranks, as candidates.

    With ``require_early_start``, a history must also start at or below the
    signal rank; later ones fit noise. A candidate's frequency is the circular
    mean of its history, its decay the median. Its confidence is the rank span
    divided by one plus the scatter in tolerance units. Sorted longest span
    first.
    """
    candidates = []
    for history in histories:
        rank_span = len(history.ranks)
        if rank_span < s.minimum_consecutive_ranks:
            continue
        if s.require_early_start and history.ranks[0] > estimated_signal_rank:
            continue
        frequencies_MHz = np.asarray(history.frequencies_MHz)
        decay_per_us = np.asarray(history.decay_per_us)
        frequency_MHz = s.circle.center(frequencies_MHz, np.ones(rank_span))
        frequency_scatter_MHz = float(np.max(s.circle.distance(frequencies_MHz, frequency_MHz)))
        median_decay_per_us = float(np.median(decay_per_us))
        decay_scatter_per_us = float(np.max(np.abs(decay_per_us - median_decay_per_us)))
        confidence_denominator = 1. + frequency_scatter_MHz / s.track_frequency_tolerance_MHz
        if s.match_decay:
            confidence_denominator += decay_scatter_per_us / s.track_decay_tolerance_per_us
        candidates.append(AttrDict(dict(row_index=0,
                                        occupation=_TRACE_LABEL,
                                        frequency_MHz=frequency_MHz,
                                        decay_per_us=median_decay_per_us,
                                        implied_growth=median_decay_per_us < 0.,
                                        pole_radius=float(np.median(history.pole_radii)),
                                        first_rank=history.ranks[0],
                                        last_rank=history.ranks[-1],
                                        rank_span=rank_span,
                                        frequency_scatter_MHz=frequency_scatter_MHz,
                                        decay_scatter_per_us=decay_scatter_per_us,
                                        confidence=float(rank_span / confidence_denominator))))
    candidates.sort(key=lambda candidate: (-candidate.rank_span,
                                           candidate.frequency_scatter_MHz,
                                           candidate.decay_scatter_per_us))
    return candidates


def _deduplicate(candidates, s):
    """1e. Drop a candidate within the dedup tolerance of a better one (decay
    too, if ``match_decay``; growth clipped to 0 first if ``clip_growth``)."""
    def clipped(decay_per_us):
        return max(0., decay_per_us) if s.clip_growth else decay_per_us

    unique = []
    for candidate in candidates:
        duplicate = False
        for existing in unique:
            duplicate = (s.circle.distance(candidate.frequency_MHz, existing.frequency_MHz)
                         <= s.dedup_frequency_tolerance_MHz)
            if s.match_decay:
                duplicate = duplicate and (abs(clipped(candidate.decay_per_us) - clipped(existing.decay_per_us))
                                           <= s.dedup_decay_tolerance_per_us)
            if duplicate:
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def _row_candidates(normalized_row, s):
    """1a-e for one normalized row -> (candidates, diagnostic).

    A row with norm at or below ``numerical_floor`` gives no candidates.
    """
    row_norm = np.linalg.norm(normalized_row)
    if row_norm <= s.numerical_floor:
        return [], AttrDict(dict(occupation=_TRACE_LABEL,
                                 estimated_signal_rank=0,
                                 maximum_rank=0,
                                 singular_values=np.array([]),
                                 candidates=[]))

    sweep = _rank_sweep(normalized_row / row_norm, s)
    histories = _track_poles(sweep.solutions, s)
    candidates = _deduplicate(_stable_candidates(histories, sweep.estimated_signal_rank, s), s)
    diagnostic = AttrDict(dict(occupation=_TRACE_LABEL,
                               estimated_signal_rank=sweep.estimated_signal_rank,
                               maximum_rank=sweep.maximum_rank,
                               singular_values=sweep.singular_values,
                               relative_singular_values=sweep.relative_singular_values,
                               candidates=candidates))
    if s.store_rank_sweeps:
        diagnostic.rank_solutions = sweep.solutions
        diagnostic.tracks = histories
    return candidates, diagnostic


def _row_settings_record(s):
    """The resolved per-row settings, as stored in a result."""
    return dict(requested_max_modes=int(s.requested_max_modes),
                pencil_length=int(s.pencil_length),
                minimum_consecutive_ranks=int(s.minimum_consecutive_ranks),
                track_frequency_tolerance_bins=float(s.track_frequency_tolerance_bins),
                dedup_frequency_tolerance_bins=float(s.dedup_frequency_tolerance_bins),
                track_frequency_tolerance_MHz=float(s.track_frequency_tolerance_MHz),
                dedup_frequency_tolerance_MHz=float(s.dedup_frequency_tolerance_MHz),
                track_decay_tolerance_per_us=float(s.track_decay_tolerance_per_us),
                dedup_decay_tolerance_per_us=float(s.dedup_decay_tolerance_per_us),
                match_decay=bool(s.match_decay),
                numerical_floor=float(s.numerical_floor),
                noise_singular_value_factor=float(s.noise_singular_value_factor),
                minimum_pole_radius=float(s.minimum_pole_radius),
                maximum_pole_radius=float(s.maximum_pole_radius),
                require_early_start=bool(s.require_early_start),
                rank_sweep_extra=s.rank_sweep_extra,
                clip_growth=bool(s.clip_growth),
                least_squares_rcond=s.least_squares_rcond)


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
                                store_rank_sweeps=False,
                                dedup_frequency_tolerance_MHz=None):
    """Step 1 on one complex time trace, then a least-squares amplitude fit.

    Needs only the trace and a uniform time grid that starts at 0.

    The trace is divided by ``trace[0]`` for conditioning (by 1 if that is
    below ``numerical_floor``). ``normalized_amplitudes`` fit the normalized
    trace; ``amplitudes`` and ``fitted_return`` are on the input scale. So for
    ``trace = sum_i A_i(t) / A_i(0)`` the ideal real parts of ``amplitudes``
    are the DOS weights, degeneracies included.
    """
    trace = np.asarray(trace, dtype=complex)
    time_us = np.asarray(time_us, dtype=float)
    if trace.ndim != 1 or time_us.ndim != 1 or len(trace) != len(time_us):
        raise ValueError("trace and time_us must be one-dimensional arrays of equal length")
    sample_time_us = _check_time_grid(time_us, "trace analysis")
    _require_positive("numerical_floor", numerical_floor)
    s = _row_settings(time_us, sample_time_us, requested_max_modes, pencil_length,
                      minimum_consecutive_ranks, track_frequency_tolerance_bins,
                      dedup_frequency_tolerance_bins, dedup_frequency_tolerance_MHz,
                      track_decay_tolerance_per_us, dedup_decay_tolerance_per_us,
                      match_decay, numerical_floor, noise_singular_value_factor,
                      minimum_pole_radius, maximum_pole_radius, require_early_start,
                      rank_sweep_extra, clip_growth, least_squares_rcond, store_rank_sweeps)

    initial_return = trace[0]
    if np.abs(initial_return) <= numerical_floor:
        initial_return = 1.
    normalized_return = trace / initial_return

    candidates, diagnostic = _row_candidates(normalized_return, s)
    diagnostic.raw_candidates = list(candidates)
    diagnostic.candidates = candidates

    selected_candidates = candidates[:min(s.requested_max_modes, s.sample_count - 1)]
    selected_candidates.sort(key=lambda candidate: candidate.frequency_MHz)
    selected_frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected_candidates])
    raw_decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected_candidates])
    selected_decay_per_us = np.maximum(raw_decay_per_us, 0.) if clip_growth else raw_decay_per_us.copy()
    poles = _poles(selected_frequencies_MHz, selected_decay_per_us, sample_time_us)
    normalized_amplitudes, normalized_fitted_return, design_condition_number = _fit_one_row(
        poles, normalized_return, least_squares_rcond)

    amplitudes = normalized_amplitudes * initial_return
    fitted_return = normalized_fitted_return * initial_return
    residual = trace - fitted_return
    return AttrDict(dict(method="matrix_pencil_trace",
                         trace=trace,
                         time_us=time_us,
                         initial_return=initial_return,
                         normalized_return=normalized_return,
                         raw_candidates=list(candidates),
                         candidates=candidates,
                         rejected_candidates=[],
                         selected_candidates=selected_candidates,
                         selected_frequencies_MHz=selected_frequencies_MHz,
                         selected_raw_decay_per_us=raw_decay_per_us,
                         selected_decay_per_us=selected_decay_per_us,
                         poles=poles,
                         normalized_amplitudes=normalized_amplitudes,
                         amplitudes=amplitudes,
                         DOS_weights=np.real(amplitudes),
                         DOS_imaginary_weights=np.imag(amplitudes),
                         amplitude_magnitudes=np.abs(amplitudes),
                         normalized_fitted_return=normalized_fitted_return,
                         fitted_return=fitted_return,
                         residual=residual,
                         relative_residual=float(np.linalg.norm(residual) / np.linalg.norm(trace)),
                         design_condition_number=design_condition_number,
                         diagnostic=diagnostic,
                         sampling=s.circle.sampling(),
                         settings=AttrDict(_row_settings_record(s))))


# ---------------------------------------------------------------------------
# Steps 2-3: across rows
# ---------------------------------------------------------------------------

class _MergeTolerance:
    """How far a row candidate may be from a cluster's frequency.

    Two modes:

    - FFT bins (default): a fixed ``merge_frequency_tolerance_bins`` bins.
    - Calibration standard error (``row_frequency_standard_errors_MHz``
      given): each row's frequency is uncertain by the standard error of the
      phase calibration of its final occupation. Rows with the same final
      occupation share that error. For a row of group g, the tolerance is
      ``sigma`` times the standard deviation of (row frequency - cluster
      frequency), where the cluster frequency is a weighted mean over the
      groups, with a ``floor``.
    """

    def __init__(self, rows, final_occupations, fft_resolution_MHz, merge_frequency_tolerance_bins,
                 row_frequency_standard_errors_MHz, sigma, floor_MHz):
        self.calibration = row_frequency_standard_errors_MHz is not None
        self.sigma = sigma
        self.floor_MHz = floor_MHz
        if self.calibration:
            row_se_MHz = np.asarray(row_frequency_standard_errors_MHz, dtype=float)
            if row_se_MHz.shape != (rows,):
                raise ValueError("row_frequency_standard_errors_MHz must contain one value per reconstruction row")
            if np.any(row_se_MHz < 0.) or not np.all(np.isfinite(row_se_MHz)):
                raise ValueError("row frequency standard errors must be finite and nonnegative")
            self.group_by_row = list(final_occupations)
            self.group_se_MHz = {}
            for group, standard_error_MHz in zip(self.group_by_row, row_se_MHz):
                if group in self.group_se_MHz and not np.isclose(self.group_se_MHz[group], standard_error_MHz,
                                                                  rtol=1e-12, atol=0.):
                    raise ValueError("rows with the same final occupation must use the same calibration "
                                     "frequency standard error")
                self.group_se_MHz[group] = float(standard_error_MHz)
            _require_positive("merge_frequency_tolerance_sigma", sigma)
            _require_positive("merge_frequency_tolerance_floor_MHz", floor_MHz)
            self.row_se_MHz = row_se_MHz
            self.bins = np.nan
            self.fixed_MHz = np.nan
        else:
            self.row_se_MHz = None
            self.group_by_row = None
            _require_positive("merge_frequency_tolerance_bins", merge_frequency_tolerance_bins)
            self.bins = merge_frequency_tolerance_bins
            self.fixed_MHz = merge_frequency_tolerance_bins * fft_resolution_MHz

    def for_row(self, row_index, cluster):
        if not self.calibration:
            return self.fixed_MHz
        group = self.group_by_row[row_index]
        own_weight = cluster.calibration_group_weights.get(group, 0.)
        variance_MHz2 = (
            (1. - own_weight) ** 2 * self.group_se_MHz[group] ** 2
            + np.sum([(weight * self.group_se_MHz[other_group]) ** 2
                      for other_group, weight in cluster.calibration_group_weights.items()
                      if other_group != group])
        )
        return max(self.floor_MHz, self.sigma * np.sqrt(variance_MHz2))

    def update(self, cluster):
        """The cluster's weight per calibration group, and its frequency standard error."""
        if not self.calibration:
            cluster.calibration_group_weights = None
            cluster.frequency_standard_error_MHz = np.nan
            return
        normalized_weights = cluster.member_weights / np.sum(cluster.member_weights)
        group_weights = {}
        for member, weight in zip(cluster.members, normalized_weights):
            group = self.group_by_row[member.row_index]
            group_weights[group] = group_weights.get(group, 0.) + float(weight)
        cluster.calibration_group_weights = group_weights
        cluster.frequency_standard_error_MHz = float(np.sqrt(np.sum([
            (weight * self.group_se_MHz[group]) ** 2 for group, weight in group_weights.items()
        ])))


def _cluster_across_rows(candidates, circle, tolerance):
    """2. Most confident first, each candidate joins the nearest compatible
    cluster (in tolerance units) that has no candidate of its row yet, or
    starts one. A cluster's frequency is the confidence-weighted circular
    mean of its members, updated after each join."""
    clusters = []
    for candidate in sorted(candidates, key=lambda candidate: -candidate.confidence):
        compatible = []
        for cluster_index, cluster in enumerate(clusters):
            if candidate.row_index in {member.row_index for member in cluster.members}:
                continue
            distance_MHz = circle.distance(candidate.frequency_MHz, cluster.frequency_MHz)
            tolerance_MHz = tolerance.for_row(candidate.row_index, cluster)
            if distance_MHz <= tolerance_MHz:
                compatible.append((distance_MHz / tolerance_MHz, distance_MHz, cluster_index))

        if compatible:
            cluster = clusters[min(compatible)[2]]
        else:
            cluster = AttrDict({"members": []})
            clusters.append(cluster)
        cluster.members.append(candidate)
        cluster.member_weights = np.asarray([max(member.confidence, np.finfo(float).eps)
                                             for member in cluster.members])
        cluster.frequency_MHz = circle.center([member.frequency_MHz for member in cluster.members],
                                              cluster.member_weights)
        tolerance.update(cluster)
    return clusters


def _score_clusters(clusters, circle, tolerance, minimum_supporting_rows, clip_growth, occupations):
    """3. -> (merged candidates, rejected clusters).

    A cluster needs ``minimum_supporting_rows`` rows. Its decay is the median
    of its members (clipped at 0 if ``clip_growth``). Its score is
    rows x median rank span / (1 + largest member distance in tolerance units).
    """
    merged = []
    rejected = []
    for cluster in clusters:
        members = cluster.members
        supporting_rows = sorted({member.row_index for member in members})
        if len(supporting_rows) < minimum_supporting_rows:
            cluster.rejection_reason = "fewer than minimum_supporting_rows"
            rejected.append(cluster)
            continue
        frequency_MHz = circle.center([member.frequency_MHz for member in members], cluster.member_weights)
        distances_MHz = [circle.distance(member.frequency_MHz, frequency_MHz) for member in members]
        frequency_scatter_MHz = float(np.max(distances_MHz))
        decay_values = np.asarray([member.decay_per_us for member in members])
        raw_decay_per_us = float(np.median(decay_values))
        decay_per_us = max(0., raw_decay_per_us) if clip_growth else raw_decay_per_us
        rank_spans = np.asarray([member.rank_span for member in members])
        merge_tolerances_MHz = np.asarray([tolerance.for_row(member.row_index, cluster) for member in members])
        normalized_frequency_scatter = float(np.max([distance_MHz / tolerance_MHz for distance_MHz, tolerance_MHz
                                                     in zip(distances_MHz, merge_tolerances_MHz)]))
        rank_confidence = float(len(supporting_rows) * np.median(rank_spans) / (1. + normalized_frequency_scatter))
        merged.append(AttrDict({
            "frequency_MHz": frequency_MHz,
            "raw_decay_per_us": raw_decay_per_us,
            "decay_per_us": decay_per_us,
            "implied_growth": raw_decay_per_us < 0.,
            "supporting_rows": supporting_rows,
            "supporting_occupations": [occupations[row] for row in supporting_rows],
            "median_rank_span": float(np.median(rank_spans)),
            "frequency_scatter_MHz": frequency_scatter_MHz,
            "normalized_frequency_scatter": normalized_frequency_scatter,
            "frequency_standard_error_MHz": cluster.frequency_standard_error_MHz,
            "merge_tolerances_MHz": merge_tolerances_MHz,
            "decay_scatter_per_us": float(np.max(np.abs(decay_values - raw_decay_per_us))),
            "rank_confidence": rank_confidence,
            "selection_score": rank_confidence,
            "confidence": rank_confidence,
            "members": members,
        }))
    return merged, rejected


# ---------------------------------------------------------------------------
# The whole analysis
# ---------------------------------------------------------------------------

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
                          store_rank_sweeps=False,
                          dedup_frequency_tolerance_MHz=None,
                          row_frequency_standard_errors_MHz=None,
                          merge_frequency_tolerance_sigma=3.0,
                          merge_frequency_tolerance_floor_MHz=1e-4):
    """Shared damped-exponential poles of all occupation rows (module docstring).

    - ``reconstruction``: ``A`` (occupation x time), ``occupations`` and
      optionally ``final_occupations``. Diagonal rows (initial = final) are
      normalized to ``A_i(0)``; off-diagonal rows are not.
    - ``spectrum``: the time grid, the Fock basis (``requested_max_modes``
      defaults to its size), and the FFT settings and grid that step 5 must
      match.

    ``reconstructed_local`` is the finite-time FFT of the fitted returns, with
    the same window, padding, sign and A(0) normalization as
    ``spectrum.measured_local``. The pole weights are separate:
    ``pole_DOS_weights`` is the real part of the summed A(0)-normalized
    amplitudes. For a complete basis its ideal value is the trace of each
    spectral projector, so a degenerate pole weighs its multiplicity. The
    imaginary part and the sum of magnitudes are diagnostics.

    The defaults reproduce the exploratory notebook algorithm; they are
    analysis choices, not calibrated confidence levels.
    ``rank_sweep_extra=None`` sweeps every algebraic rank; 2 stops at the
    signal rank + 2, enough for a three-rank persistence requirement.
    ``match_decay=False`` tracks by frequency only.
    Merging across rows is by FFT bins, or by the calibration standard errors
    if ``row_frequency_standard_errors_MHz`` is given (:class:`_MergeTolerance`).
    """
    A = np.asarray(reconstruction.A, dtype=complex)
    time_us = np.asarray(spectrum.time_us, dtype=float)
    occupations = [tuple(occupation) for occupation in reconstruction.occupations]
    final_occupations = [tuple(occupation) for occupation in
                         reconstruction.get("final_occupations", reconstruction.occupations)]
    diagonal = np.asarray([initial == final for initial, final in zip(occupations, final_occupations)])
    if requested_max_modes is None:
        requested_max_modes = len(spectrum.fock_basis)

    if A.ndim != 2 or A.shape[0] != len(occupations) or A.shape[1] != len(time_us):
        raise ValueError("reconstruction.A must have shape (occupation, time point)")
    sample_time_us = _check_time_grid(time_us, "DOS reconstruction")
    if not _is_positive_int(requested_max_modes):
        raise ValueError("requested_max_modes must be a positive integer")
    if not _is_positive_int(minimum_supporting_rows):
        raise ValueError("minimum_supporting_rows must be a positive integer")
    s = _row_settings(time_us, sample_time_us, requested_max_modes, pencil_length,
                      minimum_consecutive_ranks, track_frequency_tolerance_bins,
                      dedup_frequency_tolerance_bins, dedup_frequency_tolerance_MHz,
                      track_decay_tolerance_per_us, dedup_decay_tolerance_per_us,
                      match_decay, numerical_floor, noise_singular_value_factor,
                      minimum_pole_radius, maximum_pole_radius, require_early_start,
                      rank_sweep_extra, clip_growth, least_squares_rcond, store_rank_sweeps)
    circle = s.circle
    if merge_frequency_tolerance_bins is None and row_frequency_standard_errors_MHz is None:
        merge_frequency_tolerance_bins = track_frequency_tolerance_bins
    tolerance = _MergeTolerance(len(A), final_occupations, circle.fft_resolution_MHz,
                                merge_frequency_tolerance_bins, row_frequency_standard_errors_MHz,
                                merge_frequency_tolerance_sigma, merge_frequency_tolerance_floor_MHz)

    row_normalization = np.asarray([A[row, 0] if diagonal[row] else 1. for row in range(len(A))])[:, None]
    if np.any(np.abs(row_normalization) <= numerical_floor):
        raise ValueError("Matrix Pencil DOS reconstruction requires nonzero A_i(0) for every occupation")
    normalized_A = A / row_normalization

    # 1. Each row alone.
    row_candidates = []
    row_diagnostics = []
    for row_index, row in enumerate(normalized_A):
        trace_analysis = analyze_matrix_pencil_trace(
            row, time_us,
            requested_max_modes=s.requested_max_modes,
            pencil_length=s.pencil_length,
            minimum_consecutive_ranks=s.minimum_consecutive_ranks,
            track_frequency_tolerance_bins=s.track_frequency_tolerance_bins,
            dedup_frequency_tolerance_bins=s.dedup_frequency_tolerance_bins,
            dedup_frequency_tolerance_MHz=s.dedup_frequency_tolerance_MHz,
            track_decay_tolerance_per_us=s.track_decay_tolerance_per_us,
            dedup_decay_tolerance_per_us=s.dedup_decay_tolerance_per_us,
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
        for candidate in trace_analysis.raw_candidates:
            candidate.row_index = row_index
            candidate.occupation = occupations[row_index]
            row_candidates.append(candidate)
        diagnostic = trace_analysis.diagnostic
        diagnostic.occupation = occupations[row_index]
        row_diagnostics.append(diagnostic)

    # 2-3. Across rows.
    clusters = _cluster_across_rows(row_candidates, circle, tolerance)
    merged_candidates, rejected_clusters = _score_clusters(
        clusters, circle, tolerance, minimum_supporting_rows, clip_growth, occupations)

    # 4. Select the shared poles and refit every row's amplitudes.
    merged_candidates.sort(key=lambda candidate: (
        -candidate.selection_score,
        -candidate.rank_confidence,
        -len(candidate.supporting_rows),
        candidate.frequency_scatter_MHz,
    ))
    selected_candidates = merged_candidates[:min(s.requested_max_modes, s.sample_count - 1)]
    selected_candidates.sort(key=lambda candidate: candidate.frequency_MHz)
    if not selected_candidates:
        if row_candidates and rejected_clusters:
            raise RuntimeError("all stable Matrix-Pencil candidates failed minimum_supporting_rows")
        raise RuntimeError("no stable rowwise Matrix-Pencil candidates were found")

    selected_frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected_candidates])
    selected_decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected_candidates])
    shared_poles = _poles(selected_frequencies_MHz, selected_decay_per_us, sample_time_us)
    design = _vandermonde(shared_poles, s.sample_count)
    normalized_amplitudes = np.zeros((len(occupations), len(selected_candidates)), dtype=complex)
    normalized_fitted_return = np.zeros_like(normalized_A)
    for row_index, normalized_row in enumerate(normalized_A):
        row_amplitudes, _, _, _ = np.linalg.lstsq(design, normalized_row, rcond=least_squares_rcond)
        normalized_amplitudes[row_index] = row_amplitudes
        normalized_fitted_return[row_index] = design @ row_amplitudes
    design_condition_number = float(np.linalg.cond(design))

    amplitudes = normalized_amplitudes * row_normalization
    fitted_return = normalized_fitted_return * row_normalization
    residual = A - fitted_return
    relative_residual_by_row = (np.linalg.norm(residual, axis=1)
                                / np.maximum(np.linalg.norm(A, axis=1), np.finfo(float).eps))
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(A))

    # 5. The FFT of the fits, on the spectrum's own grid.
    fft_window = spectrum.get("fft_window", "raw")
    if fft_window not in _WINDOWS:
        raise ValueError("Matrix-Pencil display requires a supported spectrum.fft_window")
    zero_padding = spectrum.get("zero_padding", None)
    if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
        raise ValueError("Matrix Pencil requires an unmerged spectrum with integer zero_padding")
    energy_MHz = np.asarray(spectrum.energy_MHz)
    expected_energy_MHz = np.fft.fftshift(np.fft.fftfreq(zero_padding * s.sample_count, d=sample_time_us))
    if energy_MHz.shape != expected_energy_MHz.shape or not np.allclose(energy_MHz, expected_energy_MHz):
        raise ValueError("Matrix Pencil requires the original uniform FFT energy grid")
    reconstructed_local = _windowed_spectrum(fitted_return, fft_window, zero_padding)
    reconstructed_local /= np.asarray(spectrum.fft_normalization)[:, None]
    reconstructed = np.sum(reconstructed_local, axis=0)

    # Pole weights: diagonal rows in A(0) units, off-diagonal rows as measured.
    spectral_amplitudes = np.where(diagonal[:, None], normalized_amplitudes, amplitudes)
    pole_local_weights = np.real(spectral_amplitudes)
    pole_complex_DOS_weights = np.sum(spectral_amplitudes, axis=0)
    pole_DOS_weights = np.real(pole_complex_DOS_weights)
    pole_DOS_imaginary_weights = np.imag(pole_complex_DOS_weights)
    pole_local_magnitude_weights = np.abs(spectral_amplitudes)
    pole_amplitude_magnitude_sums = np.sum(pole_local_magnitude_weights, axis=0)
    row_weight_sums = np.sum(pole_local_weights, axis=1)
    total_DOS_weight = float(np.sum(pole_DOS_weights))

    settings = AttrDict(dict(
        _row_settings_record(s),
        minimum_supporting_rows=int(minimum_supporting_rows),
        merge_frequency_tolerance_bins=float(tolerance.bins),
        merge_frequency_tolerance_MHz=float(tolerance.fixed_MHz),
        merge_frequency_tolerance_mode="calibration_standard_error" if tolerance.calibration else "fft_bins",
        row_frequency_standard_errors_MHz=None if tolerance.row_se_MHz is None else tolerance.row_se_MHz.copy(),
        row_frequency_error_groups=None if tolerance.group_by_row is None else list(tolerance.group_by_row),
        merge_frequency_tolerance_sigma=float(merge_frequency_tolerance_sigma),
        merge_frequency_tolerance_floor_MHz=float(merge_frequency_tolerance_floor_MHz),
        fft_window=fft_window,
        zero_padding=int(zero_padding)))
    modes = AttrDict(dict(
        frequencies_MHz=selected_frequencies_MHz,
        decay_per_us=selected_decay_per_us,
        poles=shared_poles,
        frequency_standard_errors_MHz=np.asarray([c.frequency_standard_error_MHz for c in selected_candidates]),
        selection_scores=np.asarray([c.selection_score for c in selected_candidates]),
        supporting_row_counts=np.asarray([len(c.supporting_rows) for c in selected_candidates]),
        supporting_rows=[c.supporting_rows for c in selected_candidates],
        supporting_occupations=[c.supporting_occupations for c in selected_candidates],
        local_complex_amplitudes=spectral_amplitudes,
        local_weights=pole_local_weights,
        local_magnitude_weights=pole_local_magnitude_weights,
        complex_DOS_weights=pole_complex_DOS_weights,
        DOS_weights=pole_DOS_weights,
        DOS_imaginary_weights=pole_DOS_imaginary_weights,
        amplitude_magnitude_sums=pole_amplitude_magnitude_sums,
        row_weight_sums=row_weight_sums,
        total_DOS_weight=total_DOS_weight,
        coherent_trace_weights=np.abs(pole_complex_DOS_weights)))
    fit = AttrDict(dict(amplitudes=amplitudes,
                        normalized_amplitudes=normalized_amplitudes,
                        fitted_return=fitted_return,
                        normalized_fitted_return=normalized_fitted_return,
                        residual=residual,
                        relative_residual=relative_residual,
                        relative_residual_by_row=relative_residual_by_row,
                        design_condition_number=design_condition_number))
    spectra = AttrDict(dict(energy_MHz=energy_MHz,
                            measured_local=np.asarray(spectrum.measured_local),
                            reconstructed_local=reconstructed_local,
                            measured=np.asarray(spectrum.measured),
                            reconstructed=reconstructed,
                            complete_basis=bool(spectrum.complete_basis)))
    candidate_summary = AttrDict({
        "raw_per_row": row_candidates,
        "per_row": list(row_candidates),
        "rejected_per_row": [],
        "clusters": clusters,
        "rejected_clusters": rejected_clusters,
        "merged": merged_candidates,
        "selected": selected_candidates,
    })
    # The flat keys repeat modes/fit/spectra; older callers read them.
    return AttrDict(dict(method="matrix_pencil",
                         occupations=occupations,
                         row_normalization=row_normalization[:, 0],
                         settings=settings,
                         sampling=circle.sampling(time_us=time_us),
                         modes=modes,
                         fit=fit,
                         spectra=spectra,
                         candidates=candidate_summary,
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
                         design_condition_number=design_condition_number,
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


def refit_occupation(occupation,
                     data,
                     matrix_pencil=None,
                     least_squares_rcond=None):
    """Refit one occupation with only the poles its own row detected (step 1).

    Not the same as one row of the global fit, which uses every shared pole
    selected after the cross-row merge. This shows what the selected
    occupation supports on its own. ``occupation`` is a row index or an
    occupation tuple.
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
    measured_return = np.asarray(data.reconstruction.A[row], dtype=complex)
    initial_return = matrix_pencil.row_normalization[row]
    normalized_return = measured_return / initial_return
    poles = _poles(frequencies_MHz, decay_per_us, float(matrix_pencil.sampling.sample_time_us))
    if least_squares_rcond is None:
        least_squares_rcond = matrix_pencil.settings.least_squares_rcond
    normalized_amplitudes, normalized_fitted_return, design_condition_number = _fit_one_row(
        poles, normalized_return, least_squares_rcond)
    amplitudes = normalized_amplitudes * initial_return
    fitted_return = normalized_fitted_return * initial_return
    residual = measured_return - fitted_return

    reconstructed_spectrum = (_windowed_spectrum(fitted_return, matrix_pencil.settings.fft_window,
                                                 matrix_pencil.settings.zero_padding)
                              / data.spectrum.fft_normalization[row])
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
                         relative_residual=float(np.linalg.norm(residual) / np.linalg.norm(measured_return)),
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


# ``analyze(mpm_...=...)`` options of the spectrum class, checked by name.
OPTION_PREFIX = "mpm_"


OPTION_NAMES = frozenset(
    inspect.signature(analyze_matrix_pencil)
    .parameters) - {"reconstruction", "spectrum"}


def strip_option_prefix(options):
    """Strip the ``mpm_`` prefix; reject anything not a real option.

    The old ``kwargs.get("mpm_...")`` chain silently ignored a typo, so a
    mis-spelled tolerance looked like it worked and quietly did nothing.
    """
    stripped = {}
    for name, value in options.items():
        bare = name.removeprefix(OPTION_PREFIX)
        if bare not in OPTION_NAMES:
            raise TypeError(
                f"analyze() got an unexpected keyword argument {name!r}. "
                "Matrix-Pencil options are "
                + ", ".join(sorted(OPTION_PREFIX + n
                                   for n in OPTION_NAMES)))
        stripped[bare] = value
    return stripped

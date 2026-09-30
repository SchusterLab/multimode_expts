"""Matrix-Pencil analysis for many-body Ramsey reconstructions.

Pure numerics: arrays and a :class:`MatrixPencilSettings` in, an ``AttrDict``
out; no Experiment, station, file or FFT. Pinned by
``tests/test_matrix_pencil_regression.py`` (saved real data) and
``tests/test_matrix_pencil_synthetic.py`` (known poles).

The measured return of occupation row ``i`` is modeled as a sum of damped
exponentials with poles shared by all rows:

    A_i[n] = sum_m c_im z_m**n,    z_m = exp((-gamma_m - 2j*pi*f_m) * dt).

:func:`analyze_matrix_pencil` finds the poles and amplitudes in four steps:

1. Each row, alone (:func:`_row_candidates`):
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

No Hamiltonian or theory spectrum is used to select poles. The spectrum of
the fitted returns is the caller's (``fitting.qsim.mbr_spectrum.local_spectrum``).

:func:`refit_row` refits one row with only its own step-1 candidates.

Frequencies are principal aliases, modulo the sampling frequency
(:class:`_FrequencyCircle`).
"""

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict, Field, model_validator

from slab import AttrDict


_PositiveInt = Annotated[int, Field(ge=1)]
_Positive = Annotated[float, Field(gt=0., allow_inf_nan=False)]


class MatrixPencilSettings(BaseModel):
    """The analysis choices. The defaults reproduce the exploratory notebook
    algorithm; they are not calibrated confidence levels.

    Tolerances given as ``None`` follow the tracking tolerance: frequencies in
    FFT bins, decays as ``2 pi`` times the frequency tolerance in MHz.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Most shared poles kept, and the highest rank swept. None: as many as
    #: the pencil allows.
    requested_max_modes: _PositiveInt | None = None
    #: Hankel pencil length L. None: half the samples.
    pencil_length: _PositiveInt | None = None
    #: A row pole must persist over this many consecutive ranks.
    minimum_consecutive_ranks: _PositiveInt = 3
    #: A shared pole must be found in this many rows.
    minimum_supporting_rows: _PositiveInt = 1
    track_frequency_tolerance_bins: _Positive = 1.5
    merge_frequency_tolerance_bins: _Positive | None = None
    dedup_frequency_tolerance_bins: _Positive | None = None
    track_decay_tolerance_per_us: _Positive | None = None
    dedup_decay_tolerance_per_us: _Positive | None = None
    #: Track and deduplicate by decay as well as frequency.
    match_decay: bool = True
    #: Relative singular values below this are numerically zero.
    numerical_floor: _Positive = 1e-10
    #: Signal rank: singular values above this factor times their median.
    noise_singular_value_factor: _Positive = 2.858
    minimum_pole_radius: _Positive = 0.2
    maximum_pole_radius: _Positive = 1.05
    #: A row pole must first appear at or below the estimated signal rank.
    require_early_start: bool = True
    #: Stop the rank sweep at the signal rank plus this. None: sweep all.
    rank_sweep_extra: Annotated[int, Field(ge=0)] | None = None
    #: Clip growing poles (negative decay) to zero decay.
    clip_growth: bool = True
    least_squares_rcond: Annotated[float, Field(ge=0., allow_inf_nan=False)] | None = None
    #: Keep every rank's poles and the pole tracks in the row diagnostics.
    store_rank_sweeps: bool = False
    #: Merging by calibration standard errors (when the caller gives them):
    #: tolerance = sigma x standard error, at least the floor.
    merge_frequency_tolerance_sigma: _Positive = 3.0
    merge_frequency_tolerance_floor_MHz: _Positive = 1e-4

    @model_validator(mode="after")
    def _radii_ordered(self):
        if self.maximum_pole_radius <= self.minimum_pole_radius:
            raise ValueError("maximum_pole_radius must exceed minimum_pole_radius")
        return self


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

    def sampling(self, time_us):
        return AttrDict(dict(time_us=time_us,
                             sample_time_us=self.sample_time_us,
                             sampling_frequency_MHz=self.sampling_frequency_MHz,
                             nyquist_MHz=self.nyquist_MHz,
                             fft_resolution_MHz=self.fft_resolution_MHz))


@dataclass(frozen=True)
class _Resolved:
    """The settings with every tolerance in MHz and 1/us, for one time grid."""
    settings: MatrixPencilSettings
    circle: _FrequencyCircle
    sample_count: int
    requested_max_modes: int
    pencil_length: int
    track_MHz: float
    dedup_MHz: float
    track_decay_per_us: float
    dedup_decay_per_us: float

    @classmethod
    def build(cls, settings, time_us):
        s = settings
        sample_count = len(time_us)
        circle = _FrequencyCircle(time_us[1] - time_us[0], sample_count)
        pencil_length = s.pencil_length or sample_count // 2
        if pencil_length >= sample_count:
            raise ValueError(f"pencil_length must be below the sample count {sample_count}")
        maximum_algebraic_rank = min(pencil_length, sample_count - pencil_length)
        requested_max_modes = s.requested_max_modes or maximum_algebraic_rank
        if min(requested_max_modes, maximum_algebraic_rank) < s.minimum_consecutive_ranks:
            raise ValueError(f"this Matrix-Pencil configuration permits at most "
                             f"{min(requested_max_modes, maximum_algebraic_rank)} ranks, fewer than "
                             f"minimum_consecutive_ranks={s.minimum_consecutive_ranks}")
        track_MHz = s.track_frequency_tolerance_bins * circle.fft_resolution_MHz
        dedup_bins = s.dedup_frequency_tolerance_bins or s.track_frequency_tolerance_bins
        track_decay_per_us = s.track_decay_tolerance_per_us or 2. * np.pi * track_MHz
        return cls(settings=s, circle=circle, sample_count=sample_count,
                   requested_max_modes=requested_max_modes, pencil_length=pencil_length,
                   track_MHz=track_MHz,
                   dedup_MHz=dedup_bins * circle.fft_resolution_MHz,
                   track_decay_per_us=track_decay_per_us,
                   dedup_decay_per_us=s.dedup_decay_tolerance_per_us or track_decay_per_us)


def _check_time_grid(time_us):
    """The grid must start at 0, be uniform, and have 5+ points."""
    if time_us.ndim != 1 or len(time_us) < 5:
        raise ValueError("Matrix Pencil requires at least five time points")
    if not np.isclose(time_us[0], 0.):
        raise ValueError("Matrix Pencil requires the zero-time point")
    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError("Matrix Pencil requires uniformly spaced time points")


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


# ---------------------------------------------------------------------------
# Step 1: one row
# ---------------------------------------------------------------------------

@dataclass
class RowCandidate:
    """A pole one row supports: a pole track that lasted long enough."""
    frequency_MHz: float
    decay_per_us: float
    pole_radius: float
    first_rank: int
    last_rank: int
    rank_span: int
    frequency_scatter_MHz: float
    decay_scatter_per_us: float
    confidence: float
    row_index: int = 0
    occupation: tuple = ()


@dataclass
class _Track:
    """One pole followed over consecutive ranks."""
    ranks: list = field(default_factory=list)
    frequencies_MHz: list = field(default_factory=list)
    decay_per_us: list = field(default_factory=list)
    pole_radii: list = field(default_factory=list)

    def add(self, solution, pole_index):
        self.ranks.append(solution["rank"])
        self.frequencies_MHz.append(float(solution["frequencies_MHz"][pole_index]))
        self.decay_per_us.append(float(solution["decay_per_us"][pole_index]))
        self.pole_radii.append(float(solution["pole_radii"][pole_index]))
        return self


def _rank_sweep(scaled_row, r):
    """1a-b. The poles of the pencil at every assumed rank.

    The Hankel matrices H_0 and H_1 (H_0 shifted by one sample) come from one
    sliding window. With the thin SVD H_0 = U S V^h, the rank-r poles are the
    eigenvalues of S_r^-1 U_r^h H_1 V_r. The signal rank is the number of
    singular values above ``noise_singular_value_factor`` times their median
    (the median stands in for the unknown noise level). Poles outside
    ``[minimum_pole_radius, maximum_pole_radius]`` are dropped.
    """
    s = r.settings
    windows = sliding_window_view(scaled_row, r.pencil_length + 1)
    unshifted = windows[:, :-1]
    shifted = windows[:, 1:]
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(unshifted, full_matrices=False)
    relative_singular_values = singular_values / singular_values[0]
    singular_value_threshold = s.noise_singular_value_factor * np.median(singular_values)

    estimated_signal_rank = max(1, int(np.count_nonzero(singular_values > singular_value_threshold)))
    numerical_rank = int(np.count_nonzero(relative_singular_values > s.numerical_floor))
    maximum_rank = min(r.requested_max_modes, unshifted.shape[0], unshifted.shape[1], numerical_rank)
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
        frequencies_MHz = -np.angle(poles) / (2. * np.pi * r.circle.sample_time_us)
        decay_per_us = -np.log(pole_radii) / r.circle.sample_time_us
        order = np.argsort(frequencies_MHz)
        solutions.append(dict(rank=rank,
                              frequencies_MHz=frequencies_MHz[order],
                              decay_per_us=decay_per_us[order],
                              pole_radii=pole_radii[order]))
    return AttrDict(dict(solutions=solutions,
                         singular_values=singular_values,
                         relative_singular_values=relative_singular_values,
                         estimated_signal_rank=estimated_signal_rank,
                         maximum_rank=maximum_rank))


def _track_poles(solutions, r):
    """1c. Follow each pole from rank r-1 to rank r.

    A track that reached rank r-1 may take one rank-r pole within the
    tracking tolerance (in frequency, and in decay if ``match_decay``). All
    compatible pairs are assigned closest first, one pole per track. A
    rank-r pole that joins no track starts a new one.
    """
    match_decay = r.settings.match_decay
    tracks = [_Track().add(solutions[0], index) for index in range(len(solutions[0]["frequencies_MHz"]))]

    for solution in solutions[1:]:
        current_rank = solution["rank"]
        frequencies_MHz = solution["frequencies_MHz"]
        decays_per_us = solution["decay_per_us"]
        matches = []
        for track in tracks:
            if track.ranks[-1] != current_rank - 1:
                continue
            frequency_differences_MHz = r.circle.distance(frequencies_MHz, track.frequencies_MHz[-1])
            decay_differences_per_us = np.abs(decays_per_us - track.decay_per_us[-1])
            for pole_index in range(len(frequencies_MHz)):
                frequency_difference_MHz = frequency_differences_MHz[pole_index]
                decay_difference_per_us = decay_differences_per_us[pole_index]
                if frequency_difference_MHz > r.track_MHz:
                    continue
                if match_decay and decay_difference_per_us > r.track_decay_per_us:
                    continue
                distance = frequency_difference_MHz / r.track_MHz
                if match_decay:
                    distance += decay_difference_per_us / r.track_decay_per_us
                matches.append((distance, track, pole_index))

        matches.sort(key=lambda match: match[0])
        assigned = set()
        for _, track, pole_index in matches:
            if track.ranks[-1] == current_rank or pole_index in assigned:
                continue
            track.add(solution, pole_index)
            assigned.add(pole_index)

        for pole_index in range(len(frequencies_MHz)):
            if pole_index not in assigned:
                tracks.append(_Track().add(solution, pole_index))
    return tracks


def _stable_candidates(tracks, estimated_signal_rank, r):
    """1d. The tracks that last ``minimum_consecutive_ranks`` ranks, as candidates.

    With ``require_early_start``, a track must also start at or below the
    signal rank; later ones fit noise. A candidate's frequency is the circular
    mean of its track, its decay the median. Its confidence is the rank span
    divided by one plus the scatter in tolerance units. Sorted longest span
    first.
    """
    s = r.settings
    candidates = []
    for track in tracks:
        rank_span = len(track.ranks)
        if rank_span < s.minimum_consecutive_ranks:
            continue
        if s.require_early_start and track.ranks[0] > estimated_signal_rank:
            continue
        frequencies_MHz = np.asarray(track.frequencies_MHz)
        decay_per_us = np.asarray(track.decay_per_us)
        frequency_MHz = r.circle.center(frequencies_MHz, np.ones(rank_span))
        frequency_scatter_MHz = float(np.max(r.circle.distance(frequencies_MHz, frequency_MHz)))
        median_decay_per_us = float(np.median(decay_per_us))
        decay_scatter_per_us = float(np.max(np.abs(decay_per_us - median_decay_per_us)))
        confidence_denominator = 1. + frequency_scatter_MHz / r.track_MHz
        if s.match_decay:
            confidence_denominator += decay_scatter_per_us / r.track_decay_per_us
        candidates.append(RowCandidate(frequency_MHz=frequency_MHz,
                                       decay_per_us=median_decay_per_us,
                                       pole_radius=float(np.median(track.pole_radii)),
                                       first_rank=track.ranks[0],
                                       last_rank=track.ranks[-1],
                                       rank_span=rank_span,
                                       frequency_scatter_MHz=frequency_scatter_MHz,
                                       decay_scatter_per_us=decay_scatter_per_us,
                                       confidence=float(rank_span / confidence_denominator)))
    candidates.sort(key=lambda candidate: (-candidate.rank_span,
                                           candidate.frequency_scatter_MHz,
                                           candidate.decay_scatter_per_us))
    return candidates


def _deduplicate(candidates, r):
    """1e. Drop a candidate within the dedup tolerance of a better one (decay
    too, if ``match_decay``; growth clipped to 0 first if ``clip_growth``)."""
    s = r.settings

    def clipped(decay_per_us):
        return max(0., decay_per_us) if s.clip_growth else decay_per_us

    unique = []
    for candidate in candidates:
        duplicate = False
        for existing in unique:
            duplicate = (r.circle.distance(candidate.frequency_MHz, existing.frequency_MHz)
                         <= r.dedup_MHz)
            if s.match_decay:
                duplicate = duplicate and (abs(clipped(candidate.decay_per_us) - clipped(existing.decay_per_us))
                                           <= r.dedup_decay_per_us)
            if duplicate:
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def _row_candidates(row, r):
    """1a-e for one row -> (candidates, diagnostic).

    The row is scaled by its first sample, then to unit norm, for
    conditioning. A row with norm at or below ``numerical_floor`` gives no
    candidates.
    """
    floor = r.settings.numerical_floor
    if np.abs(row[0]) > floor:
        row = row / row[0]
    row_norm = np.linalg.norm(row)
    if row_norm <= floor:
        return [], AttrDict(dict(estimated_signal_rank=0,
                                 maximum_rank=0,
                                 singular_values=np.array([]),
                                 candidates=[]))

    sweep = _rank_sweep(row / row_norm, r)
    tracks = _track_poles(sweep.solutions, r)
    candidates = _deduplicate(_stable_candidates(tracks, sweep.estimated_signal_rank, r), r)
    diagnostic = AttrDict(dict(estimated_signal_rank=sweep.estimated_signal_rank,
                               maximum_rank=sweep.maximum_rank,
                               singular_values=sweep.singular_values,
                               relative_singular_values=sweep.relative_singular_values,
                               candidates=candidates))
    if r.settings.store_rank_sweeps:
        diagnostic.rank_solutions = sweep.solutions
        diagnostic.tracks = tracks
    return candidates, diagnostic


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

    def __init__(self, settings, final_occupations, fft_resolution_MHz, row_frequency_standard_errors_MHz):
        self.calibration = row_frequency_standard_errors_MHz is not None
        self.sigma = settings.merge_frequency_tolerance_sigma
        self.floor_MHz = settings.merge_frequency_tolerance_floor_MHz
        if self.calibration:
            row_se_MHz = np.asarray(row_frequency_standard_errors_MHz, dtype=float)
            if row_se_MHz.shape != (len(final_occupations),):
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
            self.fixed_MHz = np.nan
        else:
            bins = settings.merge_frequency_tolerance_bins or settings.track_frequency_tolerance_bins
            self.fixed_MHz = bins * fft_resolution_MHz

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


@dataclass
class _Cluster:
    members: list = field(default_factory=list)
    member_weights: np.ndarray | None = None
    frequency_MHz: float = np.nan
    calibration_group_weights: dict | None = None
    frequency_standard_error_MHz: float = np.nan
    rejection_reason: str = ""


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
            cluster = _Cluster()
            clusters.append(cluster)
        cluster.members.append(candidate)
        cluster.member_weights = np.asarray([max(member.confidence, np.finfo(float).eps)
                                             for member in cluster.members])
        cluster.frequency_MHz = circle.center([member.frequency_MHz for member in cluster.members],
                                              cluster.member_weights)
        tolerance.update(cluster)
    return clusters


def _score_clusters(clusters, circle, tolerance, minimum_supporting_rows, clip_growth):
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
        decay_values = np.asarray([member.decay_per_us for member in members])
        raw_decay_per_us = float(np.median(decay_values))
        rank_spans = np.asarray([member.rank_span for member in members])
        merge_tolerances_MHz = np.asarray([tolerance.for_row(member.row_index, cluster) for member in members])
        normalized_frequency_scatter = float(np.max([distance_MHz / tolerance_MHz for distance_MHz, tolerance_MHz
                                                     in zip(distances_MHz, merge_tolerances_MHz)]))
        merged.append(AttrDict({
            "frequency_MHz": frequency_MHz,
            "raw_decay_per_us": raw_decay_per_us,
            "decay_per_us": max(0., raw_decay_per_us) if clip_growth else raw_decay_per_us,
            "supporting_rows": supporting_rows,
            "median_rank_span": float(np.median(rank_spans)),
            "frequency_scatter_MHz": float(np.max(distances_MHz)),
            "normalized_frequency_scatter": normalized_frequency_scatter,
            "frequency_standard_error_MHz": cluster.frequency_standard_error_MHz,
            "merge_tolerances_MHz": merge_tolerances_MHz,
            "decay_scatter_per_us": float(np.max(np.abs(decay_values - raw_decay_per_us))),
            "score": float(len(supporting_rows) * np.median(rank_spans) / (1. + normalized_frequency_scatter)),
            "members": members,
        }))
    return merged, rejected


# ---------------------------------------------------------------------------
# The whole analysis
# ---------------------------------------------------------------------------

def analyze_matrix_pencil(reconstruction, time_us, settings=None,
                          row_frequency_standard_errors_MHz=None):
    """Shared damped-exponential poles of all occupation rows (module docstring).

    - ``reconstruction``: ``A`` (row x time), ``occupations`` and optionally
      ``final_occupations``. Diagonal rows (initial = final) are normalized
      to ``A_i(0)``; off-diagonal rows are not.
    - ``time_us``: uniform, starting at 0.
    - ``settings``: a :class:`MatrixPencilSettings`; the defaults if None.
    - ``row_frequency_standard_errors_MHz``: one per row. If given, rows are
      merged by these calibration errors instead of FFT bins
      (:class:`_MergeTolerance`).

    ``modes.DOS_weights`` is the real part of the summed A(0)-normalized
    amplitudes. For a complete basis its ideal value is the trace of each
    spectral projector, so a degenerate pole weighs its multiplicity.
    """
    settings = settings or MatrixPencilSettings()
    A = np.asarray(reconstruction.A, dtype=complex)
    time_us = np.asarray(time_us, dtype=float)
    occupations = [tuple(occupation) for occupation in reconstruction.occupations]
    final_occupations = [tuple(occupation) for occupation in
                         reconstruction.get("final_occupations", reconstruction.occupations)]
    diagonal = np.asarray([initial == final for initial, final in zip(occupations, final_occupations)])
    if A.ndim != 2 or A.shape[0] != len(occupations) or A.shape[1] != len(time_us):
        raise ValueError("reconstruction.A must have shape (occupation, time point)")
    _check_time_grid(time_us)
    r = _Resolved.build(settings, time_us)
    circle = r.circle
    tolerance = _MergeTolerance(settings, final_occupations, circle.fft_resolution_MHz,
                                row_frequency_standard_errors_MHz)

    row_normalization = np.where(diagonal, A[:, 0], 1.)[:, None]
    if np.any(np.abs(row_normalization) <= settings.numerical_floor):
        raise ValueError("Matrix Pencil requires nonzero A_i(0) for every diagonal row")
    normalized_A = A / row_normalization

    # 1. Each row alone.
    row_candidates = []
    row_diagnostics = []
    for row_index, row in enumerate(normalized_A):
        candidates, diagnostic = _row_candidates(row, r)
        for candidate in candidates:
            candidate.row_index = row_index
            candidate.occupation = occupations[row_index]
        diagnostic.occupation = occupations[row_index]
        row_candidates.extend(candidates)
        row_diagnostics.append(diagnostic)

    # 2-3. Across rows.
    clusters = _cluster_across_rows(row_candidates, circle, tolerance)
    merged, rejected_clusters = _score_clusters(
        clusters, circle, tolerance, settings.minimum_supporting_rows, settings.clip_growth)

    # 4. Select the shared poles and refit every row's amplitudes.
    merged.sort(key=lambda candidate: (
        -candidate.score,
        -len(candidate.supporting_rows),
        candidate.frequency_scatter_MHz,
    ))
    selected = sorted(merged[:min(r.requested_max_modes, r.sample_count - 1)],
                      key=lambda candidate: candidate.frequency_MHz)
    if not selected:
        if row_candidates and rejected_clusters:
            raise RuntimeError("all stable Matrix-Pencil candidates failed minimum_supporting_rows")
        raise RuntimeError("no stable rowwise Matrix-Pencil candidates were found")

    frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected])
    decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected])
    shared_poles = _poles(frequencies_MHz, decay_per_us, circle.sample_time_us)
    design = _vandermonde(shared_poles, r.sample_count)
    normalized_amplitudes = np.zeros((len(occupations), len(selected)), dtype=complex)
    normalized_fitted_return = np.zeros_like(normalized_A)
    for row_index, normalized_row in enumerate(normalized_A):
        row_amplitudes, _, _, _ = np.linalg.lstsq(design, normalized_row, rcond=settings.least_squares_rcond)
        normalized_amplitudes[row_index] = row_amplitudes
        normalized_fitted_return[row_index] = design @ row_amplitudes

    amplitudes = normalized_amplitudes * row_normalization
    fitted_return = normalized_fitted_return * row_normalization
    residual = A - fitted_return

    # Pole weights: diagonal rows in A(0) units, off-diagonal rows as measured.
    local_complex_amplitudes = np.where(diagonal[:, None], normalized_amplitudes, amplitudes)
    complex_DOS_weights = np.sum(local_complex_amplitudes, axis=0)
    DOS_weights = np.real(complex_DOS_weights)

    return AttrDict(dict(
        method="matrix_pencil",
        occupations=occupations,
        row_normalization=row_normalization[:, 0],
        settings=AttrDict(dict(settings.model_dump(),
                               merge_frequency_tolerance_mode=("calibration_standard_error" if tolerance.calibration
                                                               else "fft_bins"),
                               row_frequency_standard_errors_MHz=row_frequency_standard_errors_MHz)),
        # The tolerances actually used, after the None defaults are resolved.
        resolved=AttrDict(dict(
            requested_max_modes=r.requested_max_modes,
            pencil_length=r.pencil_length,
            track_frequency_tolerance_MHz=r.track_MHz,
            dedup_frequency_tolerance_MHz=r.dedup_MHz,
            merge_frequency_tolerance_MHz=tolerance.fixed_MHz,
            track_decay_tolerance_per_us=r.track_decay_per_us,
            dedup_decay_tolerance_per_us=r.dedup_decay_per_us)),
        sampling=circle.sampling(time_us),
        modes=AttrDict(dict(
            frequencies_MHz=frequencies_MHz,
            decay_per_us=decay_per_us,
            poles=shared_poles,
            frequency_standard_errors_MHz=np.asarray([c.frequency_standard_error_MHz for c in selected]),
            supporting_rows=[c.supporting_rows for c in selected],
            supporting_row_counts=np.asarray([len(c.supporting_rows) for c in selected]),
            local_complex_amplitudes=local_complex_amplitudes,
            local_weights=np.real(local_complex_amplitudes),
            complex_DOS_weights=complex_DOS_weights,
            DOS_weights=DOS_weights,
            total_DOS_weight=float(np.sum(DOS_weights)))),
        fit=AttrDict(dict(
            amplitudes=amplitudes,
            normalized_amplitudes=normalized_amplitudes,
            fitted_return=fitted_return,
            residual=residual,
            relative_residual=float(np.linalg.norm(residual) / np.linalg.norm(A)),
            relative_residual_by_row=(np.linalg.norm(residual, axis=1)
                                      / np.maximum(np.linalg.norm(A, axis=1), np.finfo(float).eps)),
            design_condition_number=float(np.linalg.cond(design)))),
        candidates=AttrDict(dict(
            per_row=row_candidates,
            clusters=clusters,
            rejected_clusters=rejected_clusters,
            merged=merged,
            selected=selected)),
        row_diagnostics=row_diagnostics,
    ))


def refit_row(result, row_trace, row):
    """Refit one row with only the poles its own step 1 found.

    Not the same as that row of the global fit, which uses every shared pole
    selected after the cross-row merge: this shows what the row supports on
    its own. ``result`` is the :func:`analyze_matrix_pencil` result the row
    belongs to; ``row_trace`` is the row's measured return.
    """
    settings = result.settings
    candidates = sorted(result.row_diagnostics[row].candidates, key=lambda candidate: candidate.frequency_MHz)
    frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in candidates], dtype=float)
    raw_decay_per_us = np.asarray([candidate.decay_per_us for candidate in candidates], dtype=float)
    decay_per_us = np.maximum(raw_decay_per_us, 0.) if settings.clip_growth else raw_decay_per_us.copy()

    measured_return = np.asarray(row_trace, dtype=complex)
    initial_return = result.row_normalization[row]
    normalized_return = measured_return / initial_return
    poles = _poles(frequencies_MHz, decay_per_us, float(result.sampling.sample_time_us))
    normalized_amplitudes, normalized_fitted_return, design_condition_number = _fit_one_row(
        poles, normalized_return, settings.least_squares_rcond)
    fitted_return = normalized_fitted_return * initial_return
    residual = measured_return - fitted_return
    return AttrDict(dict(
        row_index=row,
        occupation=result.occupations[row],
        diagnostic=result.row_diagnostics[row],
        candidates=candidates,
        time_us=np.asarray(result.sampling.time_us, dtype=float),
        measured_return=measured_return,
        fitted_return=fitted_return,
        residual=residual,
        relative_residual=float(np.linalg.norm(residual) / np.linalg.norm(measured_return)),
        frequencies_MHz=frequencies_MHz,
        raw_decay_per_us=raw_decay_per_us,
        decay_per_us=decay_per_us,
        poles=poles,
        normalized_amplitudes=normalized_amplitudes,
        amplitudes=normalized_amplitudes * initial_return,
        design_condition_number=design_condition_number,
    ))


#: ``analyze(mpm_...=...)`` options of the spectrum classes carry this prefix.
OPTION_PREFIX = "mpm_"


def settings_from_options(options, **defaults):
    """``{'mpm_track_frequency_tolerance_bins': 0.5, ...}`` -> settings.

    ``defaults`` apply where ``options`` do not set a value. A misspelled
    option raises (pydantic's ``extra='forbid'``).
    """
    stripped = {name.removeprefix(OPTION_PREFIX): value for name, value in options.items()}
    return MatrixPencilSettings(**{**defaults, **stripped})

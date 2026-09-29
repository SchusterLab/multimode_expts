"""Why a level is missed: one row per model level, the checks in order (spec 5.5).

1. In the rows: the model weight of the level in the measured rows (``min_weight``).
2. Resolvable: the gaps to both neighbours are at least ``min_gap_snr`` times their
   Cramer-Rao errors (``fitting.qsim.poles.resolution``), from the model's amplitudes, each
   row's noise (measured out of band), the decay, and free row offsets with C's prior. No
   unbiased method does better.
3. Held by the data: the refinement of C started at the model levels (``oracle_fit``,
   ``oracle_starts``); the level keeps its weight, stays in place, and dropping it costs
   chi^2 (``drop_one_chi2``).
4. Found by a fitter: its poles matched to the oracle positions (the model's where not held).

A missed level's cause is its first failed check; "search" if it passed all three and the
fitter still missed it. Pure numerics.
"""
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fitting.qsim.poles.joint_refined import refine_with_group_offsets, remove_offsets, variable_projection
from fitting.qsim.poles.matching import alias_distance, level_tolerances, match_poles
from fitting.qsim.poles.pole_fit import PoleFit, normalize_to_initial_return
from fitting.qsim.poles.real_benchmarks import MODEL_ERROR_MHz
from fitting.qsim.poles.resolution import clusters, cramer_rao_bounds

CHECKS = ("in_rows", "resolvable", "held")
#: The oracle merges its closest start poles until their pole matrix is below this condition.
ORACLE_MAX_CONDITION = 1e8
CAUSES = {"in_rows": "not in rows", "resolvable": "unresolvable", "held": "not in data"}


class DiagnosisSettings(BaseModel):
    """The thresholds of the checks."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: A level is in the rows if its summed model weight there is at least this.
    min_weight: Annotated[float, Field(ge=0.)] = 0.1
    #: Resolvable if the gap to the nearest level is at least this many Cramer-Rao errors.
    min_gap_snr: Annotated[float, Field(gt=0.)] = 4.
    #: Held if the oracle keeps at least this fraction of the model weight ...
    held_weight_fraction: Annotated[float, Field(ge=0.)] = 0.5
    #: ... and dropping the level costs at least this chi^2 (noise units).
    held_chi2: Annotated[float, Field(ge=0.)] = 25.
    #: Row offsets: Gaussian prior width (MHz), as fitter C.
    offset_prior_MHz: Annotated[float, Field(gt=0.)] = 0.5e-3
    #: Levels within this many bins enter a level's Cramer-Rao bound.
    crb_window_bins: Annotated[float, Field(gt=0.)] = 3.
    #: Decay of every level at the oracle's start (per us; T2 about 100 us). The Cramer-Rao
    #: bounds use the oracle's median decay instead.
    decay_per_us: Annotated[float, Field(ge=0.)] = 0.01
    #: Frequencies farther than this many bins from every level are out of band (Hann window).
    band_margin_bins: Annotated[float, Field(gt=0.)] = 4.


def hann_spectrum(a, dt_us):
    """-> (E, X_b(E)) with X = sum_n w_n a_n e^{+2 pi i E t_n} / sqrt(sum w^2), w Hann:
    white noise of variance s^2 per sample has E|X|^2 = s^2 in every bin."""
    window = np.hanning(a.shape[1])
    X = np.fft.ifft(a * window, axis=1) * a.shape[1] / np.sqrt(np.sum(window ** 2))
    return np.fft.fftfreq(a.shape[1], d=dt_us), X


def in_band(frequencies_MHz, levels_MHz, bin_MHz, margin_bins, dt_us):
    """-> True where a frequency is within ``margin_bins`` of a level (alias-wrapped)."""
    distance = np.abs(alias_distance(frequencies_MHz, levels_MHz, 1 / dt_us))
    return distance.min(axis=1) <= margin_bins * bin_MHz


def row_noise(a, dt_us, levels_MHz, margin_bins):
    """-> s_b, each row's noise per sample: the rms of X_b(E) out of band."""
    frequencies, X = hann_spectrum(a, dt_us)
    band = in_band(frequencies, levels_MHz, 1 / (a.shape[1] * dt_us), margin_bins, dt_us)
    return np.sqrt(np.mean(np.abs(X[:, ~band]) ** 2, axis=1))


def residual_excess(fit, spectrum, settings=DiagnosisSettings()):
    """-> per row, mean |X_residual|^2 in band / s_b^2: 1 if the residual there is only noise."""
    a = normalize_to_initial_return(spectrum.A)
    noise = row_noise(a, spectrum.dt_us, spectrum.levels_MHz, settings.band_margin_bins)
    frequencies, R = hann_spectrum(a - fit.returns(spectrum.time_us), spectrum.dt_us)
    band = in_band(frequencies, spectrum.levels_MHz, spectrum.bin_MHz, settings.band_margin_bins, spectrum.dt_us)
    return np.mean(np.abs(R[:, band]) ** 2, axis=1) / noise ** 2


def oracle_starts(levels_MHz, weights, time_us, decay_per_us):
    """-> index groups of sorted levels: one start pole each, at the weighted mean. The closest
    adjacent groups are merged until V_{n,g} = exp((-gamma - 2 pi i E_g) t_n) has a condition
    below ``ORACLE_MAX_CONDITION`` (numerics only; resolvability is ``cramer_rao_bounds``)."""
    groups = [[k] for k in range(len(levels_MHz))]
    while True:
        centers = np.array([np.average(levels_MHz[g], weights=weights[g]) for g in groups])
        V = np.exp(np.outer(time_us, -decay_per_us - 2j * np.pi * centers))
        if len(groups) == 1 or np.linalg.cond(V) <= ORACLE_MAX_CONDITION:
            return groups, centers
        k = int(np.argmin(np.diff(centers)))
        groups[k:k + 2] = [groups[k] + groups[k + 1]]


def seen_frequencies(fit):
    """-> E_lambda + sum_b |c_b,lambda| delta_b / sum_b |c_b,lambda|: where the rows that hold a
    pole see it. With row offsets (C) a level held by a few rows moves with their offsets, so
    fitters compare in this frame; without offsets it is E."""
    if fit.row_offsets_MHz is None:
        return fit.frequencies_MHz
    weights = np.abs(fit.amplitudes)
    return fit.frequencies_MHz + fit.row_offsets_MHz @ weights / weights.sum(axis=0)


def oracle_fit(a, time_us, levels_MHz, settings=DiagnosisSettings()):
    """-> C's refinement started at ``levels_MHz``: a PoleFit in their order (not sorted)."""
    rows = np.arange(len(a))
    E, gamma, offsets, errors = refine_with_group_offsets(
        a, time_us, rows, np.asarray(levels_MHz, dtype=float), np.full(len(levels_MHz), settings.decay_per_us),
        np.zeros(len(a)), settings.offset_prior_MHz)
    amplitudes = variable_projection(remove_offsets(a, time_us, offsets), time_us, E, gamma)[1]
    return PoleFit(E, gamma, amplitudes, len(E), errors, offsets)


def drop_one_chi2(a, time_us, fit, noise):
    """-> per pole, chi^2 without it minus chi^2 with it, the other poles and offsets fixed and
    the amplitudes refitted: chi^2 = sum_b |r_b|^2 / s_b^2. An upper bound on its significance."""
    corrected = remove_offsets(a, time_us, fit.row_offsets_MHz)

    def chi2(keep):
        r = variable_projection(corrected, time_us, fit.frequencies_MHz[keep], fit.decays_per_us[keep])[0]
        return np.sum(np.abs(r) ** 2 / noise[:, None] ** 2)

    everything = np.ones(len(fit.frequencies_MHz), dtype=bool)
    return np.array([chi2(everything & (np.arange(len(everything)) != k)) for k in range(len(everything))]) - chi2(everything)


def diagnose_levels(spectrum, fits, settings=DiagnosisSettings()):
    """-> (DataFrame with one row per model level, the oracle PoleFit). ``fits``: name -> PoleFit.

    Columns: the model level and weight; ``crb_kHz``; ``gap_snr``, the smaller of its two
    adjacent gaps over its Cramer-Rao error; ``cluster``, the number of levels it cannot be told
    from; the oracle's frequency shift, weight and drop-one chi^2 (levels with their own oracle
    pole, ``oracle_starts``); each check; per fitter ``found_X`` and ``cause_X``. Frequencies are
    compared where the rows see them (``seen_frequencies``). Levels sorted, as ``RealSpectrum``
    keeps them.
    """
    a = normalize_to_initial_return(spectrum.A)
    weight = spectrum.row_weights.sum(axis=0)
    table = pd.DataFrame(dict(level_kHz=1e3 * spectrum.levels_MHz, multiplicity=spectrum.multiplicities, weight=weight))
    table["in_rows"] = weight >= settings.min_weight
    kept = np.flatnonzero(table.in_rows)
    levels, amplitudes = spectrum.levels_MHz[kept], spectrum.row_weights[:, kept]
    noise = row_noise(a, spectrum.dt_us, levels, settings.band_margin_bins)
    starts, centers = oracle_starts(levels, weight[kept], spectrum.time_us, settings.decay_per_us)
    oracle = oracle_fit(a, spectrum.time_us, centers, settings)
    # the bounds at the decay the data show (the oracle's median), not the assumed start
    decay = max(float(np.median(oracle.decays_per_us)), 0.)
    crb, gap_crb = cramer_rao_bounds(spectrum.time_us, levels, amplitudes, decay, noise,
                                     settings.offset_prior_MHz, settings.crb_window_bins * spectrum.bin_MHz)
    pair_snr = np.diff(levels) / gap_crb
    groups = clusters(pair_snr, settings.min_gap_snr)
    columns = dict(separation_kHz=1e3 * np.minimum(np.r_[np.inf, np.diff(levels)], np.r_[np.diff(levels), np.inf]),
                   crb_kHz=1e3 * crb, gap_snr=np.minimum(np.r_[np.inf, pair_snr], np.r_[pair_snr, np.inf]),
                   cluster=np.concatenate([np.full(len(g), len(g)) for g in groups]))
    for name, values in columns.items():
        table.loc[kept, name] = values
    single = np.array([len(g) == 1 for g in starts])
    alone = np.array([g[0] for g in starts if len(g) == 1], dtype=int)
    shift = alias_distance(seen_frequencies(oracle), centers, 1 / spectrum.dt_us)[np.arange(len(centers)), np.arange(len(centers))]
    for name, values in dict(oracle_shift_kHz=1e3 * shift, oracle_weight=oracle.weights,
                             drop_chi2=drop_one_chi2(a, spectrum.time_us, oracle, noise)).items():
        table.loc[kept[alone], name] = values[single]
    table["resolvable"] = table.in_rows & (table.cluster == 1)
    # held: keeps its weight, stays nearer its own place than a neighbour, and costs chi^2 to drop
    table["held"] = (table.resolvable & (table.oracle_weight >= settings.held_weight_fraction * table.weight)
                     & (np.abs(table.oracle_shift_kHz) <= 0.5 * table.separation_kHz)
                     & (table.drop_chi2 >= settings.held_chi2))
    reference = spectrum.levels_MHz.copy()
    held = table.held.to_numpy()
    reference[held] += 1e-3 * table.oracle_shift_kHz.to_numpy()[held]
    tolerance = np.maximum(level_tolerances(spectrum.levels_MHz, spectrum.bin_MHz), MODEL_ERROR_MHz)
    for name, fit in fits.items():
        match = match_poles(seen_frequencies(fit), reference, tolerance, 1 / spectrum.dt_us)
        table[f"found_{name}"] = np.isin(np.arange(len(reference)), match.level_index)
        table[f"cause_{name}"] = [cause(row, name) for _, row in table.iterrows()]
    return table, oracle


def cause(row, fitter):
    """-> 'found', the first failed check's cause, or 'search' (all checks passed, still missed)."""
    if row[f"found_{fitter}"]:
        return "found"
    return next((CAUSES[check] for check in CHECKS if not row[check]), "search")


def cause_counts(table, fitters):
    """-> levels per cause (rows) and fitter (columns)."""
    return pd.DataFrame({name: table[f"cause_{name}"].value_counts() for name in fitters}).fillna(0).astype(int)

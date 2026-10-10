"""Photon-count correlations and regularized AGP estimates, without a station.

Readout labels assume cond_sec_phase=-90, phase_second_pulse=180 and support
on photon numbers 0..3. Linear correction retains signed probabilities and
weights. Errors include finite shared calibration statistics, not unknown
preparation errors, calibration drift or an inverse of open-system dynamics.
"""
import json
from itertools import combinations_with_replacement
from pathlib import Path

import h5py
import numpy as np


def _json_attribute(handle, key, default=None):
    value = handle.attrs.get(key)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _job_config(handle):
    config = _json_attribute(handle, "config", {})
    if "expt" not in config and isinstance(config.get("config"), dict):
        config = config["config"]
    if "expt" not in config:
        raise ValueError("The raw HDF5 must contain its experiment config")
    return config


def discover_population_jobs(directory, **criteria):
    """Return sorted raw HDF5 paths matching exact cfg.expt field values."""
    matches = []
    for path in sorted(Path(directory).rglob("*.h5")):
        with h5py.File(path, "r") as handle:
            if "idata" not in handle or "config" not in handle.attrs:
                continue
            experiment = _job_config(handle)["expt"]
            if all(experiment.get(key) == value for key, value in criteria.items()):
                matches.append(str(path.resolve()))
    return matches


def _check_readout(config):
    experiment = config["expt"]
    mode = experiment.get("readout")
    if mode != "multiparity" and not experiment.get("multiparity_readout", False):
        raise ValueError("This job did not request multiparity readout")
    if not np.isclose(experiment.get("cond_sec_phase", 90) % 360, 270):
        raise ValueError("Number labels require the notebook's cond_sec_phase=-90")
    if not np.isclose(experiment.get("phase_second_pulse", 180) % 360, 180):
        raise ValueError("Number labels require phase_second_pulse=180")
    if int(config.get("read_num", 0)) < 2:
        raise ValueError("The raw config must record read_num, including herald lanes")


def _classify_shots(raw_i, config, threshold, e_is_high_i, herald_lane):
    read_num = int(config["read_num"])
    if raw_i.ndim != 2 or raw_i.shape[1] % read_num:
        raise ValueError("idata must be (sweep point, shots times read_num)")
    if not np.all(np.isfinite(raw_i)) or not np.isfinite(threshold):
        raise ValueError("Raw IQ values and the threshold must be finite")
    shots = raw_i.reshape(len(raw_i), -1, read_num)
    excited = shots > threshold if e_is_high_i else shots < threshold
    labels = excited[:, :, -2].astype(int) + 2 * excited[:, :, -1].astype(int)
    accepted = np.ones(labels.shape, dtype=bool)
    if herald_lane is not None:
        if not 0 <= herald_lane < read_num - 2:
            raise ValueError("herald_lane must precede the two final parity lanes")
        accepted = ~excited[:, :, herald_lane]
    return labels, accepted


def _probability_rows(labels, accepted):
    shot_counts = accepted.sum(axis=1)
    if np.any(shot_counts < 2):
        raise ValueError("At least two accepted shots are needed at every time")
    counts = np.column_stack([((labels == number) & accepted).sum(axis=1)
                              for number in range(4)])
    return counts, counts / shot_counts[:, None], shot_counts


def _job_times(handle, cycles, cycle_us):
    derived = _json_attribute(handle, "derived_params", {})
    if "time_us" in handle:
        return np.asarray(handle["time_us"], dtype=float).reshape(-1), "saved time_us"
    if cycle_us is not None:
        if not np.isfinite(cycle_us) or cycle_us <= 0:
            raise ValueError("Explicit cycle_us must be positive")
        return cycles * cycle_us, "explicit cycle_us override"
    if "floquet_cycle_us" in derived:
        return cycles * float(derived["floquet_cycle_us"]), "compiled derived_params"
    if np.all(cycles == 0):
        return np.zeros_like(cycles), "zero-cycle points only"
    raise ValueError("Missing compiled cycle time: supply cycle_us explicitly; no guess is made")


def load_population_job(path, cycle_us=None, threshold=None, e_is_high_i=True,
                        herald_lane=None):
    """Read one 1D raw job. Herald filtering is explicit; no mask is inferred."""
    with h5py.File(path, "r") as handle:
        config = _job_config(handle)
        _check_readout(config)
        qubit = int(config["expt"].get("qubits", [0])[0])
        if threshold is None:
            threshold = np.asarray(config["device"]["readout"]["threshold"]).reshape(-1)[qubit]
        raw_i = np.asarray(handle["idata"], dtype=float)
        if raw_i.ndim == 1:
            raw_i = raw_i[None, :]
        cycles = np.asarray(handle["xpts"], dtype=float).reshape(-1)
        times, timing_source = _job_times(handle, cycles, cycle_us)
        metadata = {key: _json_attribute(handle, key, {})
                    for key in ["derived_params", "config_versions"]}
    labels, accepted = _classify_shots(raw_i, config, threshold, e_is_high_i, herald_lane)
    counts, probabilities, shot_counts = _probability_rows(labels, accepted)
    if len(times) != len(labels) or len(cycles) != len(labels):
        raise ValueError("Only one-dimensional sweep jobs are supported")
    return dict(source=str(Path(path).resolve()), config=config, metadata=metadata,
                cycles=cycles, time_us=times, timing_source=timing_source,
                number_shots=labels, accepted_shots=accepted, counts=counts,
                probabilities=probabilities, shot_counts=shot_counts,
                threshold=float(threshold), herald_lane=herald_lane,
                e_is_high_i=bool(e_is_high_i))


def _mean_probability_covariance(probability, count):
    """Unbiased estimated covariance of a multinomial sample mean."""
    return (np.diag(probability) - np.outer(probability, probability)) / (count - 1)


def _readout_signature(job):
    experiment = job.get("config", {}).get("expt", {})
    return dict(threshold=job.get("threshold"), e_is_high_i=job.get("e_is_high_i"),
                herald_lane=job.get("herald_lane"),
                cond_sec_phase=experiment.get("cond_sec_phase", -90),
                phase_second_pulse=experiment.get("phase_second_pulse", 180),
                parity_fast=experiment.get("parity_fast", False))


def calibrate_readout(jobs, prepared_numbers):
    """C[prepared, measured]; each input must be a single zero-cycle job."""
    if len(jobs) != len(prepared_numbers):
        raise ValueError("Supply one prepared number for every calibration job")
    counts = np.zeros((4, 4), dtype=float)
    for job, number in zip(jobs, prepared_numbers):
        if number not in range(4) or len(job["counts"]) != 1 or np.any(job["time_us"] != 0):
            raise ValueError("Calibration needs single zero-time jobs prepared in 0, 1, 2, or 3")
        if _readout_signature(job) != _readout_signature(jobs[0]):
            raise ValueError("Calibration jobs must share readout and classification settings")
        counts[int(number)] += job["counts"][0]
    totals = counts.sum(axis=1)
    if np.any(totals < 2):
        raise ValueError("All four prepared Fock states need at least two shots")
    confusion = counts / totals[:, None]
    if np.linalg.matrix_rank(confusion) != 4:
        raise ValueError("The four-state confusion matrix is singular")
    covariance = np.array([_mean_probability_covariance(row, count)
                           for row, count in zip(confusion, totals)])
    return dict(counts=counts, confusion=confusion, row_covariance=covariance,
                condition_number=float(np.linalg.cond(confusion)),
                sources=[job["source"] for job in jobs],
                readout_signature=_readout_signature(jobs[0]),
                convention="C[prepared, measured]; p_measured = C.T @ p_true")


def _moment_statistics(probabilities, corrected, counts, values, calibration):
    confusion = np.eye(4) if calibration is None else calibration["confusion"]
    weights = np.linalg.solve(confusion, values)
    mean = corrected @ values
    variance = np.maximum(probabilities @ weights**2 - (probabilities @ weights)**2, 0)
    shot_variance = variance / (counts - 1)
    jacobian = -corrected[:, :, None] * weights[None, None, :]
    row_covariance = np.zeros((4, 4, 4)) if calibration is None else calibration["row_covariance"]
    calibration_covariance = _calibration_covariance(jacobian, row_covariance)
    return dict(mean=mean, raw_mean=probabilities @ values, shot_variance=shot_variance,
                calibration_jacobian=jacobian, calibration_covariance=calibration_covariance,
                standard_error=np.sqrt(shot_variance + np.diag(calibration_covariance)))


def _calibration_covariance(jacobian, row_covariance):
    covariance = np.zeros((len(jacobian), len(jacobian)))
    for prepared in range(4):
        derivative = jacobian[:, prepared, :]
        covariance += derivative @ row_covariance[prepared] @ derivative.T
    return covariance


def population_moments(job, calibration=None):
    """Number and pair moments, with shot and shared-calibration uncertainty.

    No clipping or renormalization follows the linear inversion. Errors use
    the delta method for a fixed, independent four-state calibration.
    """
    probabilities = np.asarray(job["probabilities"], dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or np.any(probabilities < 0):
        raise ValueError("Raw probabilities must have shape (time, 4) and be nonnegative")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Each raw probability row must sum to one")
    if np.any(np.asarray(job["shot_counts"]) < 2):
        raise ValueError("At least two shots are required for an uncertainty estimate")
    if calibration is not None and "readout_signature" in calibration:
        if calibration["readout_signature"] != _readout_signature(job):
            raise ValueError("Calibration and measurement readout/classification settings differ")
    confusion = np.eye(4) if calibration is None else np.asarray(calibration["confusion"])
    corrected = np.linalg.solve(confusion.T, probabilities.T).T
    result = {key: job[key] for key in ["time_us", "source", "config", "shot_counts"]}
    result.update(raw_probabilities=probabilities, corrected_probabilities=corrected,
                  calibration=calibration, metadata=job.get("metadata", {}))
    for name, values in [("number", np.arange(4.)), ("pairs", np.array([0., 0., 1., 3.]))]:
        result[name] = _moment_statistics(probabilities, corrected, job["shot_counts"],
                                           values, calibration)
    return result


def fock_basis(total_photons, mode_count):
    """Full fixed-N sector; mode zero is the hub."""
    if total_photons < 0 or mode_count < 1:
        raise ValueError("Use nonnegative total_photons and positive mode_count")
    states = combinations_with_replacement(range(mode_count), total_photons)
    return np.array([[state.count(mode) for mode in range(mode_count)] for state in states])


def probe_weights(occupations, probe="number", mode=0):
    """Diagonal values of n_mode or n_mode(n_mode-1)/2; no trace normalization."""
    occupations = np.asarray(occupations)
    if occupations.ndim != 2 or not 0 <= mode < occupations.shape[1]:
        raise ValueError("occupations must be (states, modes), with a valid mode index")
    number = occupations[:, mode].astype(float)
    if probe == "number":
        return number
    if probe == "kerr":
        return number * (number - 1) / 2
    raise ValueError("probe must be 'number' or 'kerr'")


def _check_support(occupations, total_photons, mode_count, probe, mode):
    basis = fock_basis(total_photons, mode_count)
    occupations = np.asarray(occupations)
    if occupations.ndim != 2 or occupations.shape[1] != mode_count:
        raise ValueError("Each occupation must have mode_count entries")
    supplied = [tuple(row) for row in occupations]
    if len(set(supplied)) != len(supplied) or not set(supplied) <= set(map(tuple, basis)):
        raise ValueError("Occupations must be distinct physical states in the fixed-N sector")
    values = probe_weights(basis, probe, mode)
    needed = {tuple(row) for row, value in zip(basis, values) if value != 0}
    if not needed <= set(supplied):
        raise ValueError(f"Missing {len(needed - set(supplied))} nonzero-probe Fock states")
    return basis, values


def _same_calibration(traces):
    reference = traces[0]["calibration"]
    for trace in traces[1:]:
        other = trace["calibration"]
        if (reference is None) != (other is None):
            raise ValueError("Use the same calibration for all correlation traces")
        if reference is not None:
            for key in ["confusion", "row_covariance", "sources"]:
                if not np.array_equal(reference[key], other[key]):
                    raise ValueError("Use the same calibration for all correlation traces")
    return reference


def _dynamics_signature(trace):
    """Only initial state, shot count and bookkeeping may vary within a correlation."""
    config = trace.get("config", {})
    experiment = config.get("expt", {})
    keys = ["swap_stors", "detunings", "floquet_waveform", "floquet_hardware_loop",
            "scramble_sync_cycles", "palindrome_scramble", "update_phases", "zero_floquet_gain",
            "floquet_gauss_sigma", "coupler_current", "ro_stor", "readout_route",
            "readout_swap_pulses", "postpulse", "readout", "cond_sec_phase",
            "phase_second_pulse", "parity_fast", "man_mode_no"]
    derived = trace.get("metadata", {}).get("derived_params", {})
    signature = dict(experiment={key: experiment.get(key) for key in keys},
                     timing={key: derived.get(key) for key in
                             ["floquet_cycle_us", "m1s_pi_fracs", "couplings_MHz"]},
                     device=config.get("device"), hardware=config.get("hw"))
    return json.dumps(signature, sort_keys=True)


def _check_trace_configs(traces, occupations, mode):
    reference = _dynamics_signature(traces[0])
    for trace, occupation in zip(traces, occupations):
        if _dynamics_signature(trace) != reference:
            raise ValueError("Correlation traces must share the dynamics and readout configuration")
        experiment = trace.get("config", {}).get("expt", {})
        for key in ["occupations", "initial_occupation"]:
            if key in experiment and not np.array_equal(experiment[key], occupation):
                raise ValueError("Saved occupation does not match the supplied initial state")
        if "ro_stor" in experiment:
            storages = experiment.get("swap_stors", [])
            if mode > len(storages):
                raise ValueError("Cannot resolve the requested mode from saved swap_stors")
            expected = 0 if mode == 0 else storages[mode - 1]
            if experiment["ro_stor"] != expected:
                raise ValueError("Saved readout mode does not match the requested probe mode")


def build_correlation(traces, occupations, total_photons=3, mode_count=5,
                      probe="number", mode=0):
    """C(t)=sum_b v_b <V(t)>_b / D, requiring every nonzero-weight initial state."""
    if not traces or len(traces) != len(occupations):
        raise ValueError("Supply one trace per occupation")
    if total_photons > 3:
        raise ValueError("Two-parity number readout assumes no population above three")
    basis, full_values = _check_support(occupations, total_photons, mode_count, probe, mode)
    _check_trace_configs(traces, occupations, mode)
    weights = probe_weights(occupations, probe, mode) / len(basis)
    calibration = _same_calibration(traces)
    times = np.asarray(traces[0]["time_us"])
    if any(not np.array_equal(trace["time_us"], times) for trace in traces):
        raise ValueError("All initial states must use exactly the same time grid")
    moments = [trace["number" if probe == "number" else "pairs"] for trace in traces]
    correlation = weights @ np.array([moment["mean"] for moment in moments])
    raw = weights @ np.array([moment["raw_mean"] for moment in moments])
    shot_variance = weights**2 @ np.array([moment["shot_variance"] for moment in moments])
    jacobian = sum(weight * moment["calibration_jacobian"] for weight, moment in zip(weights, moments))
    row_covariance = np.zeros((4, 4, 4)) if calibration is None else calibration["row_covariance"]
    covariance = np.diag(shot_variance) + _calibration_covariance(jacobian, row_covariance)
    return dict(time_us=times, correlation=correlation, raw_correlation=raw,
                standard_error=np.sqrt(np.maximum(np.diag(covariance), 0)), covariance=covariance,
                expected_t0=float(np.mean(full_values**2)), dimension=len(basis),
                initial_weights=weights, occupations=np.asarray(occupations), probe=probe, mode=mode,
                sources=[trace["source"] for trace in traces], calibration=calibration)


def _real_correlations(time_us, correlation):
    times = np.asarray(time_us, dtype=float)
    signal = np.asarray(correlation)
    if np.iscomplexobj(signal) and np.any(signal.imag != 0):
        raise ValueError("Photon-count correlations must be real")
    signal = np.atleast_2d(signal.real.astype(float))
    if signal.shape[1] != len(times) or not np.all(np.isfinite(signal)):
        raise ValueError("correlation must have finite shape (time,) or (channel, time)")
    return times, signal


def _fit_diagnostics(frequencies, decays, amplitudes, fitted, signal, condition):
    opposite_distance = [np.min(abs(frequencies + frequency)) for frequency in frequencies]
    scale = np.maximum(np.linalg.norm(signal, axis=1), np.finfo(float).eps)
    return dict(relative_residual=np.linalg.norm(fitted - signal, axis=1) / scale,
                imaginary_fit_fraction=np.linalg.norm(fitted.imag, axis=1) / scale,
                negative_real_weight=np.maximum(-amplitudes.real, 0).sum(axis=1),
                imaginary_weight=np.abs(amplitudes.imag).sum(axis=1),
                growing_pole_count=int(np.sum(decays < 0)),
                maximum_opposite_frequency_distance_mhz=float(max(opposite_distance, default=0)),
                design_condition_number=float(condition))


def fit_correlation_mpm(time_us, correlation, max_modes=16, settings=None):
    """Use the existing engine on real traces, retaining DC and both frequency signs.

    max_modes counts signed poles, including DC, and also caps the engine's
    rank sweep. No mean subtraction, frequency reflection or residue clipping
    is performed. The inherited engine's selection limitations still apply.
    """
    from fitting.qsim import matrix_pencil

    times, signal = _real_correlations(time_us, correlation)
    settings = settings or matrix_pencil.MatrixPencilSettings(requested_max_modes=max_modes, clip_growth=False)
    if settings.clip_growth:
        raise ValueError("Set clip_growth=False so growing poles remain visible in diagnostics")
    rows = [(index,) for index in range(len(signal))]
    reconstruction = matrix_pencil.AttrDict(dict(A=signal, occupations=rows,
                                             final_occupations=[(-1, index) for index in range(len(signal))]))
    result = matrix_pencil.analyze_matrix_pencil(reconstruction, times, settings)
    frequencies, decays = result.modes.frequencies_MHz, result.modes.decay_per_us
    amplitudes, fitted = result.fit.amplitudes, result.fit.fitted_return
    diagnostics = _fit_diagnostics(frequencies, decays, amplitudes, fitted, signal,
                                    result.fit.design_condition_number)
    return dict(time_us=times, signal=signal, frequencies_mhz=frequencies,
                decay_per_us=decays, amplitudes=amplitudes, fitted=fitted,
                diagnostics=diagnostics, settings=settings.model_dump(),
                method="existing row-candidate/shared-mode matrix-pencil engine")


def _angular_cutoff(cutoff_mhz):
    if not np.isfinite(cutoff_mhz) or cutoff_mhz <= 0:
        raise ValueError("cutoff_mhz must be a positive cyclic frequency in MHz")
    return 2 * np.pi * cutoff_mhz


def regularized_agp_from_fit(fit, cutoff_mhz, reference_mhz=None):
    """Inferred zero-decay, even-signal AGP norm; not an inverse Lindblad map.

    Frequencies in MHz mean cycles/us. Output is us^2 for a parameter in
    rad/us. reference_mhz=g/(2*pi) converts it to the norm for lambda/g.
    Decays and sine parts are omitted explicitly; signed real weights remain.
    """
    mu = _angular_cutoff(cutoff_mhz)
    omega = 2 * np.pi * np.asarray(fit["frequencies_mhz"])
    spectral_filter = omega**2 / (omega**2 + mu**2)**2
    values = np.asarray(fit["amplitudes"]).real @ spectral_filter
    units = "us^2; parameter in rad/us"
    if reference_mhz is not None:
        values *= _angular_cutoff(reference_mhz)**2
        units = "dimensionless; parameter divided by angular reference coupling"
    return dict(values=values, cutoff_mhz=float(cutoff_mhz), reference_mhz=reference_mhz,
                units=units, interpretation="zero-decay even-signal inference from fitted poles",
                spectral_filter=spectral_filter)


def finite_time_agp(time_us, correlation, cutoff_mhz, covariance=None):
    """Direct trapezoid kernel integral over the measured window; no decay correction.

    This is a truncated correlation integral, not the positive finite-window
    operator norm Q_T. Missing late-time cancellation can leave a DC bias.
    """
    times, signal = _real_correlations(time_us, correlation)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Use increasing times starting at zero")
    mu = _angular_cutoff(cutoff_mhz)
    widths = np.diff(times)
    trapezoid = np.zeros(len(times))
    trapezoid[:-1] += widths / 2
    trapezoid[1:] += widths / 2
    weights = trapezoid * np.exp(-mu * times) * (1 - mu * times) / (2 * mu)
    result = dict(values=signal @ weights, weights=weights, cutoff_mhz=float(cutoff_mhz),
                  units="us^2; parameter in rad/us", window_us=float(times[-1]),
                  interpretation="direct finite-time integral of the measured correlation")
    if covariance is not None:
        result["standard_error"] = float(np.sqrt(max(weights @ np.asarray(covariance) @ weights, 0)))
    return result


def _write_value(group, name, value):
    if isinstance(value, dict):
        child = group.create_group(name)
        child.attrs["kind"] = "dict"
        for key, item in value.items():
            _write_value(child, str(key), item)
    elif value is None:
        group.create_group(name).attrs["kind"] = "none"
    elif isinstance(value, (str, Path)):
        group.create_dataset(name, data=str(value), dtype=h5py.string_dtype("utf-8"))
    elif isinstance(value, (list, tuple)) and (not value or isinstance(value[0], (str, dict, type(None)))):
        child = group.create_group(name)
        child.attrs["kind"] = "list"
        for index, item in enumerate(value):
            _write_value(child, str(index), item)
    else:
        group.create_dataset(name, data=np.asarray(value))


def _read_value(node):
    if isinstance(node, h5py.Dataset):
        if h5py.check_string_dtype(node.dtype) is not None:
            return node.asstr()[()]
        value = node[()]
        return value.item() if np.ndim(value) == 0 else value
    kind = node.attrs.get("kind", "dict")
    if kind == "none":
        return None
    if kind == "list":
        return [_read_value(node[str(index)]) for index in range(len(node))]
    return {key: _read_value(node[key]) for key in node}


def save_analysis_h5(path, data, metadata=None, sources=()):
    """Create a separate processed file; refuse to overwrite existing raw or processed data."""
    with h5py.File(path, "x") as handle:
        handle.attrs["format"] = "agp_population_analysis_v1"
        _write_value(handle, "data", data)
        _write_value(handle, "metadata", metadata or {})
        _write_value(handle, "sources", [str(source) for source in sources])
    return str(Path(path).resolve())


def load_analysis_h5(path):
    """Reload saved arrays, metadata and sources without station/database access."""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != "agp_population_analysis_v1":
            raise ValueError("Not an AGP processed/calibration file")
        return {key: _read_value(handle[key]) for key in handle}

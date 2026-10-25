"""The AGP norm from a measured two-time correlation, without a station.

- :func:`agp_kernel_weights`: integration weights of the AGP kernel over a
  measured time grid, optionally with a local T1 compensation.
- :func:`finite_time_agp`: the kernel integral of one correlation, with its
  standard error from a covariance.
- :func:`calibration_error_factors`: the readout-calibration error of a
  corrected moment, as factors that sum like independent errors.

The closed-system theory of the same quantities is in ``theory/qsim/agp.py``.
The kernel weights and the calibration factors are from jonginn's 10-23
notebook (``agp_storage7_mu1_to10_t1_theory_ENTIERLYCODEX.ipynb``, in the
``multimode_thermalization_sim`` repo), moved here on 2026-10-24.
"""
import json
from pathlib import Path

import h5py
import numpy as np


def _angular_cutoff(cutoff_mhz):
    """-> mu in rad/us for a cyclic cutoff in MHz."""
    if not np.isfinite(cutoff_mhz) or cutoff_mhz <= 0:
        raise ValueError("cutoff_mhz must be a positive cyclic frequency in MHz")
    return 2 * np.pi * cutoff_mhz


def agp_kernel_weights(time_us, cutoff_mhz, lifetime_us=None):
    """-> weights w with ``w @ C`` the AGP kernel integral of C over the grid.

    The kernel is K(t) = e^{-mu t} (1 - mu t) / (2 mu), mu = 2 pi cutoff_mhz
    (rad/us); its cosine transform is omega^2 / (omega^2 + mu^2)^2. The
    integral of K(t) e^{t/T1} C(t) from 0 to the last time is exact for C
    linear between the samples (no trapezoid of the product).
    ``lifetime_us=T1`` multiplies C by e^{t/T1}, which removes the explicit
    local loss term -n/T1 of a probed mode, not the change of its current.

    Units: us^2, for a parameter in rad/us. ``time_us`` must start at 0 and
    increase. This is a truncated integral: a constant part of C that the
    window does not average out leaves a bias of order T e^{-mu T} / (2 mu).
    """
    times = np.asarray(time_us, dtype=float)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_us must be 1D, start at 0 and increase")
    a = _angular_cutoff(cutoff_mhz)
    b = a if lifetime_us is None else a - 1 / float(lifetime_us)
    if b <= 0:
        raise ValueError("the T1 compensation needs 1/T1 < 2 pi cutoff")
    tau = 1 / a

    # Primitives of K(t) e^{t/T1} and of t K(t) e^{t/T1}, with K = tau/2 (1 - a t) e^{-a t}.
    def primitive_0(t):
        return tau / 2 * np.exp(-b * t) * (a * t / b + a / b**2 - 1 / b)

    def primitive_1(t):
        return tau / 2 * np.exp(-b * t) * (
            a * t**2 / b + (2 * a / b**2 - 1 / b) * t + 2 * a / b**3 - 1 / b**2)

    left, right = times[:-1], times[1:]
    width = right - left
    area = primitive_0(right) - primitive_0(left)
    moment = primitive_1(right) - primitive_1(left)
    weights = np.zeros(len(times))
    weights[:-1] += (right * area - moment) / width
    weights[1:] += (moment - left * area) / width
    return weights


def finite_time_agp(time_us, correlation, cutoff_mhz, covariance=None, lifetime_us=None):
    """The AGP kernel integral of a measured correlation over its window.

    ``time_us`` and ``correlation`` are matching 1D arrays. ``covariance``
    (time, time) gives ``standard_error``. No decay correction, except the
    optional local T1 compensation (see :func:`agp_kernel_weights`).
    """
    times = np.asarray(time_us, dtype=float)
    weights = agp_kernel_weights(times, cutoff_mhz, lifetime_us)
    value = weights @ np.asarray(correlation, dtype=float)
    result = dict(values=float(value), weights=weights, cutoff_mhz=float(cutoff_mhz),
                  lifetime_us=lifetime_us, units="us^2; parameter in rad/us",
                  window_us=float(times[-1]),
                  interpretation="finite-time kernel integral of the measured correlation")
    if covariance is not None:
        variance = weights @ np.asarray(covariance) @ weights
        result["standard_error"] = float(np.sqrt(max(variance, 0)))
    return result


def calibration_error_factors(corrected_probabilities, confusion_matrix, calibration_shots,
                              values):
    """-> F (..., 4): the calibration error of ``corrected @ values``, one column per prepared state.

    ``confusion_matrix`` is C[prepared, measured], each row a multinomial mean
    of ``calibration_shots`` shots (scalar or one per row). With the weights
    w = C^-1 values, a row error dC_a moves the corrected moment by
    -p_corrected[a] (dC_a . w). Rows are independent, so the calibration
    covariance of two corrected moments x and y is ``sum_a F_x[a] F_y[a]``:
    a sum over the last axis, after any linear combination of the factors.
    """
    matrix = np.asarray(confusion_matrix, dtype=float)
    shots = np.broadcast_to(np.asarray(calibration_shots, dtype=float), (len(matrix),))
    if np.any(shots < 2):
        raise ValueError("each calibration row needs at least two shots")
    weights = np.linalg.solve(matrix, np.asarray(values, dtype=float))
    projected = np.array([weights @ ((np.diag(row) - np.outer(row, row)) / (count - 1)) @ weights
                          for row, count in zip(matrix, shots)])
    return np.asarray(corrected_probabilities, dtype=float) * np.sqrt(projected)


########################################################################
# An independent raw-shot classifier. tools/convert_fock_population_catalog.py
# checks every converted job's raw probabilities against it.
########################################################################

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

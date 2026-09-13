# -*- coding: utf-8 -*-
"""Rebuilding a spectroscopy average from fewer of its saved shots.

Why this exists
---------------
A spectrum measured with 1,000 shots per point and the same spectrum
measured with 50 answer a question the acquisition cannot: how much of a
reconstructed line is signal and how much is shot noise. Both come from one
acquisition -- the saved single-shot I/Q is re-averaged over a random subset,
so no new measurement time is spent.

The one subtlety, and it is the reason this is not two lines
-----------------------------------------------------------
The saved ``avgi``/``avgq`` and the mean of the saved raw shots need not
agree: QICK's on-board averaging may use a different ADC offset or a
different number of rounds than ``collect_shots`` reports. So a subset mean
cannot simply replace the saved average.

What is added instead is the subset's *fluctuation about the full raw mean*:

    subsampled = saved_average + mean(subset) - mean(all shots)

which reproduces the saved average exactly when every shot is selected,
whatever offset the two paths differ by. The returned metadata records that
raw-minus-saved difference per job, so a config where it is not merely an
offset is visible rather than silent.

Pre-selected acquisitions are refused: their saved average is conditioned on
herald lanes, and unconditioned sampling of the final lane cannot reproduce
it.

Lane bookkeeping stays with the caller
--------------------------------------
Which interleaved lane holds the science measurement is acquisition
knowledge, so the caller passes one lane count per job (see
``experiments.qsim.dark_base.readout_lane_count``). This module then needs
nothing from ``experiments``.
"""
from copy import copy

import numpy as np
from slab import AttrDict


def subsample_spectroscopy_shots(spectroscopy_expts,
                                 shots_per_point,
                                 readout_lanes,
                                 seed=None):
    """Re-average each saved sweep point over a random subset of its shots.

    Args:
        spectroscopy_expts: flat list of loaded spectroscopy jobs. Each needs
            saved ``idata``/``qdata`` and ``avgi``/``avgq``. Not modified:
            the subsampled averages go into shallow copies.
        shots_per_point: final-readout shots to use per saved
            ``(Floquet cycle, preparation phase)`` point. Drawn without
            replacement, so it cannot exceed the shots a point actually has.
        readout_lanes: one interleaving period per job, in the same order.
            The science measurement is that job's last lane.
        seed: makes the subset reproducible.

    Returns:
        ``(subsampled_expts, metadata)``. The metadata records the requested
        count, the seed, the available-shot range, and per job the
        raw-versus-saved averaging diagnostics described in the module
        docstring.

    Raises:
        ValueError: on a non-positive shot count, a job with no saved shots,
            a length mismatch against ``readout_lanes``, a raw length that is
            not a multiple of the lane count, a point with fewer shots than
            requested, or a pre-selected acquisition.
    """
    if isinstance(shots_per_point, (bool, np.bool_)):
        raise ValueError("shots_per_point must be a positive integer")
    if not isinstance(shots_per_point, (int, np.integer)) or shots_per_point < 1:
        raise ValueError("shots_per_point must be a positive integer")
    shots_per_point = int(shots_per_point)
    spectroscopy_expts = list(spectroscopy_expts)
    if not spectroscopy_expts:
        raise ValueError("spectroscopy_expts cannot be empty")
    readout_lanes = [int(lanes) for lanes in readout_lanes]
    if len(readout_lanes) != len(spectroscopy_expts):
        raise ValueError(
            f"got {len(readout_lanes)} readout-lane counts for "
            f"{len(spectroscopy_expts)} jobs"
        )

    rng = np.random.default_rng(seed)
    subsampled_expts = []
    job_summaries = []
    available_shots = []

    for job_index, expt in enumerate(spectroscopy_expts):
        if (expt.cfg.expt.get("active_reset", False)
                and expt.cfg.expt.get("pre_selection_reset", False)):
            raise ValueError(
                "shot subsampling does not support pre_selection_reset; "
                "the saved average is conditioned on herald readouts"
            )
        if "idata" not in expt.data or "qdata" not in expt.data:
            raise ValueError(f"spectroscopy job {job_index} has no saved single-shot IQ data")
        if "avgi" not in expt.data or "avgq" not in expt.data:
            raise ValueError(f"spectroscopy job {job_index} has no saved averaged IQ data")

        saved_avgi = np.asarray(expt.data["avgi"], dtype=float)
        saved_avgq = np.asarray(expt.data["avgq"], dtype=float)
        if saved_avgi.shape != saved_avgq.shape:
            raise ValueError(f"spectroscopy job {job_index} has mismatched avgi/avgq shapes")
        point_count = saved_avgi.size

        def point_rows(values, name):
            try:
                array = np.asarray(values)
            except ValueError:
                array = None
            if array is not None and array.ndim >= 2 and array.shape[0] == point_count:
                return [np.asarray(array[index], dtype=float).reshape(-1)
                        for index in range(point_count)]
            if point_count == 1 and array is not None and array.dtype != object:
                return [np.asarray(array, dtype=float).reshape(-1)]
            if len(values) == point_count:
                return [np.asarray(values[index], dtype=float).reshape(-1)
                        for index in range(point_count)]
            raise ValueError(
                f"spectroscopy job {job_index} has {name} that does not "
                f"match its {point_count} sweep points"
            )

        idata_rows = point_rows(expt.data["idata"], "idata")
        qdata_rows = point_rows(expt.data["qdata"], "qdata")

        read_num = readout_lanes[job_index]
        final_lane = read_num - 1

        final_i_rows = []
        final_q_rows = []
        for point_index, (idata, qdata) in enumerate(zip(idata_rows, qdata_rows)):
            if len(idata) != len(qdata):
                raise ValueError(
                    f"spectroscopy job {job_index}, point {point_index} has "
                    "different I/Q shot counts"
                )
            if len(idata) % read_num:
                raise ValueError(
                    f"spectroscopy job {job_index}, point {point_index} raw "
                    f"length {len(idata)} is not divisible by read_num={read_num}"
                )
            final_i = idata[final_lane::read_num]
            final_q = qdata[final_lane::read_num]
            if len(final_i) < shots_per_point:
                raise ValueError(
                    f"spectroscopy job {job_index}, point {point_index} has "
                    f"only {len(final_i)} final-readout shots; requested "
                    f"{shots_per_point}"
                )
            final_i_rows.append(final_i)
            final_q_rows.append(final_q)
            available_shots.append(len(final_i))

        saved_avgi_flat = saved_avgi.reshape(-1)
        saved_avgq_flat = saved_avgq.reshape(-1)
        full_i_mean = np.asarray([np.mean(values) for values in final_i_rows])
        full_q_mean = np.asarray([np.mean(values) for values in final_q_rows])
        full_raw_minus_saved_avgi = full_i_mean - saved_avgi_flat
        full_raw_minus_saved_avgq = full_q_mean - saved_avgq_flat

        sampled_avgi = np.empty(point_count, dtype=float)
        sampled_avgq = np.empty(point_count, dtype=float)
        for point_index, (final_i, final_q) in enumerate(zip(final_i_rows, final_q_rows)):
            selected_indices = rng.choice(len(final_i),
                                          size=shots_per_point,
                                          replace=False)
            sampled_avgi[point_index] = (
                saved_avgi_flat[point_index]
                + np.mean(final_i[selected_indices])
                - full_i_mean[point_index]
            )
            sampled_avgq[point_index] = (
                saved_avgq_flat[point_index]
                + np.mean(final_q[selected_indices])
                - full_q_mean[point_index]
            )

        sampled_avgi = sampled_avgi.reshape(saved_avgi.shape)
        sampled_avgq = sampled_avgq.reshape(saved_avgq.shape)
        sampled_data = AttrDict(dict(expt.data))
        sampled_data["avgi"] = sampled_avgi
        sampled_data["avgq"] = sampled_avgq
        sampled_data["amps"] = np.abs(sampled_avgi + 1j * sampled_avgq)
        sampled_data["phases"] = np.angle(sampled_avgi + 1j * sampled_avgq)
        sampled_data.pop("Pe", None)
        sampled_data.pop("return_quadrature", None)

        sampled_expt = copy(expt)
        sampled_expt.data = sampled_data
        subsampled_expts.append(sampled_expt)
        job_summaries.append(AttrDict(dict(
            job_index=job_index,
            read_num=read_num,
            point_count=point_count,
            minimum_available_shots=min(len(values) for values in final_i_rows),
            maximum_available_shots=max(len(values) for values in final_i_rows),
            median_full_raw_minus_saved_avgi=float(
                np.median(full_raw_minus_saved_avgi)
            ),
            median_full_raw_minus_saved_avgq=float(
                np.median(full_raw_minus_saved_avgq)
            ),
            maximum_full_raw_minus_saved_avgi_scatter=float(
                np.max(np.abs(
                    full_raw_minus_saved_avgi
                    - np.median(full_raw_minus_saved_avgi)
                ))
            ),
            maximum_full_raw_minus_saved_avgq_scatter=float(
                np.max(np.abs(
                    full_raw_minus_saved_avgq
                    - np.median(full_raw_minus_saved_avgq)
                ))
            ),
        )))

    metadata = AttrDict(dict(
        shots_per_point=shots_per_point,
        seed=seed,
        replace=False,
        minimum_available_shots=min(available_shots),
        maximum_available_shots=max(available_shots),
        job_summaries=job_summaries,
    ))
    return subsampled_expts, metadata

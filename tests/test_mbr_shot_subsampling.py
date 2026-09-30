# -*- coding: utf-8 -*-
"""Re-averaging a saved spectroscopy point over fewer of its shots.

The reduced-shot replay is how we tell a real reconstructed line from shot
noise, so it has to be right in a specific, checkable way: **selecting every
available shot must reproduce the saved average exactly**, whatever offset
QICK's on-board averaging and ``collect_shots`` happen to differ by. That
identity is the whole reason the implementation adds a fluctuation about the
full raw mean instead of just replacing the average, and it is what these
tests pin.

Synthetic jobs, deliberately: a real job would make the offset agreement a
coincidence of one dataset rather than a property of the code.
"""
import numpy as np
import pytest
from slab import AttrDict

from fitting.qsim.mbr_reconstruction import subsample_spectroscopy_shots


class FakeJob:
    """The shape the subsampler reads: ``.data`` and ``.cfg.expt``."""

    def __init__(self, i_lanes, q_lanes, saved_avgi, saved_avgq, **expt):
        self.data = AttrDict(dict(
            idata=[np.asarray(row, dtype=float) for row in i_lanes],
            qdata=[np.asarray(row, dtype=float) for row in q_lanes],
            avgi=np.asarray(saved_avgi, dtype=float),
            avgq=np.asarray(saved_avgq, dtype=float),
        ))
        self.cfg = AttrDict(dict(expt=AttrDict(expt)))


def _interleave(science, herald, lanes):
    """One point's raw array: ``lanes - 1`` herald reads then the science read."""
    out = []
    for value in science:
        out.extend([herald] * (lanes - 1))
        out.append(value)
    return out


# A deliberate offset between the saved average and the raw-shot mean: the
# saved value is 5.0 low, which is exactly the situation the implementation
# is built to tolerate.
OFFSET = 5.0


def _one_job(lanes=2, shots=8, points=2):
    science = [np.arange(shots, dtype=float) + 100.0 * point
               for point in range(points)]
    i_rows = [_interleave(row, -999.0, lanes) for row in science]
    q_rows = [_interleave(row + 0.5, -999.0, lanes) for row in science]
    saved_avgi = [np.mean(row) - OFFSET for row in science]
    saved_avgq = [np.mean(row + 0.5) - OFFSET for row in science]
    return FakeJob(i_rows, q_rows, saved_avgi, saved_avgq), lanes


def test_taking_every_shot_reproduces_the_saved_average():
    """The identity the offset-tolerant form exists for."""
    job, lanes = _one_job(shots=8)

    out, _ = subsample_spectroscopy_shots([job], 8, [lanes], seed=0)

    np.testing.assert_allclose(out[0].data["avgi"], job.data["avgi"])
    np.testing.assert_allclose(out[0].data["avgq"], job.data["avgq"])


def test_a_subset_stays_near_the_saved_average_not_the_raw_mean():
    """A naive implementation would return the raw mean, i.e. 5.0 too high."""
    job, lanes = _one_job(shots=8)

    out, _ = subsample_spectroscopy_shots([job], 4, [lanes], seed=1)

    saved = np.asarray(job.data["avgi"])
    got = np.asarray(out[0].data["avgi"])
    # The subset mean of 0..7 can sit at most 2 away from the full mean here,
    # so anything near saved + OFFSET is the offset having leaked in.
    assert np.all(np.abs(got - saved) < 2.5), got - saved


def test_the_seed_makes_the_subset_reproducible():
    job, lanes = _one_job()

    first, _ = subsample_spectroscopy_shots([job], 4, [lanes], seed=7)
    again, _ = subsample_spectroscopy_shots([job], 4, [lanes], seed=7)
    other, _ = subsample_spectroscopy_shots([job], 4, [lanes], seed=8)

    np.testing.assert_allclose(first[0].data["avgi"], again[0].data["avgi"])
    assert not np.allclose(first[0].data["avgi"], other[0].data["avgi"])


def test_the_original_job_is_untouched():
    job, lanes = _one_job()
    before_avgi = np.array(job.data["avgi"])
    before_raw = np.array(job.data["idata"][0])

    out, _ = subsample_spectroscopy_shots([job], 3, [lanes], seed=0)

    np.testing.assert_allclose(job.data["avgi"], before_avgi)
    np.testing.assert_allclose(job.data["idata"][0], before_raw)
    assert out[0] is not job


def test_derived_quantities_follow_the_new_average():
    job, lanes = _one_job()

    out, _ = subsample_spectroscopy_shots([job], 4, [lanes], seed=2)

    avgi = np.asarray(out[0].data["avgi"])
    avgq = np.asarray(out[0].data["avgq"])
    np.testing.assert_allclose(out[0].data["amps"], np.abs(avgi + 1j * avgq))
    np.testing.assert_allclose(out[0].data["phases"], np.angle(avgi + 1j * avgq))


def test_the_lane_count_selects_which_readout_is_sampled():
    """Three lanes: sampling must ignore the two herald reads entirely."""
    job, _ = _one_job(lanes=3, shots=8)

    out, _ = subsample_spectroscopy_shots([job], 8, [3], seed=0)

    np.testing.assert_allclose(out[0].data["avgi"], job.data["avgi"])


def test_a_wrong_lane_count_is_caught_not_absorbed():
    """The guard that makes the lane-count contract enforceable.

    With the wrong period the raw length is no longer a multiple of it, or
    the strided slice picks up herald reads. Either way it must raise rather
    than return other readouts' shots as the science average.
    """
    job, _ = _one_job(lanes=2, shots=8)

    with pytest.raises(ValueError):
        subsample_spectroscopy_shots([job], 8, [3], seed=0)


def test_lane_counts_must_match_the_job_count():
    job, lanes = _one_job()
    with pytest.raises(ValueError, match="readout-lane counts"):
        subsample_spectroscopy_shots([job, job], 4, [lanes], seed=0)


def test_pre_selected_acquisitions_are_refused():
    job, lanes = _one_job()
    job.cfg.expt["active_reset"] = True
    job.cfg.expt["pre_selection_reset"] = True

    with pytest.raises(ValueError, match="pre_selection_reset"):
        subsample_spectroscopy_shots([job], 4, [lanes], seed=0)


def test_asking_for_more_shots_than_exist_raises():
    job, lanes = _one_job(shots=8)
    with pytest.raises(ValueError, match="only 8 final-readout shots"):
        subsample_spectroscopy_shots([job], 9, [lanes], seed=0)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, np.True_])
def test_the_shot_count_must_be_a_positive_integer(bad):
    job, lanes = _one_job()
    with pytest.raises(ValueError, match="positive integer"):
        subsample_spectroscopy_shots([job], bad, [lanes], seed=0)


def test_metadata_reports_the_offset_so_it_is_not_silent():
    """The raw-minus-saved difference is the diagnostic for "is it an offset".

    Here it is exactly OFFSET at every point, so the median is OFFSET and the
    scatter about it is zero. A config where the two averaging paths differ by
    more than an offset shows up as scatter instead.
    """
    job, lanes = _one_job()

    _, meta = subsample_spectroscopy_shots([job], 4, [lanes], seed=0)

    summary = meta["job_summaries"][0]
    assert summary["median_full_raw_minus_saved_avgi"] == pytest.approx(OFFSET)
    assert summary["maximum_full_raw_minus_saved_avgi_scatter"] \
        == pytest.approx(0.0)
    assert meta["shots_per_point"] == 4
    assert meta["replace"] is False

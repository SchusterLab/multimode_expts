# -*- coding: utf-8 -*-
"""What a single-shot calibration writes into the station config.

This was a copy-pasted notebook cell in seven places and it writes the
readout settings every later experiment reads. Now that there is one copy,
these tests pin what it writes -- including the two details a re-derivation
would plausibly get wrong: the angle *accumulates* onto the existing phase,
and which of the two confusion-matrix fields is filled depends on whether
active reset was on.
"""
import numpy as np
import pytest
from slab import AttrDict

from experiments.readout_calibration import apply_singleshot_calibration


class FakeStation:
    def __init__(self):
        self.hardware_cfg = AttrDict(dict(device=AttrDict(dict(
            readout=AttrDict(dict(phase=[10.0], threshold=[0.0],
                                  threshold_list=[[0.0]], Ie=[0.0], Ig=[0.0]))))))
        self.autocalib_path = "/tmp/does-not-matter"


class FakeHistogram:
    """An acquired HistogramExperiment, already fitted."""

    def __init__(self, active_reset=False, angle=25.0):
        self.cfg = AttrDict(dict(expt=AttrDict(dict(active_reset=active_reset))))
        self.data = dict(
            fids=[0.97],
            confusion_matrix=np.array([[0.98, 0.02], [0.03, 0.97]]),
            thresholds=[3.5],
            angle=angle,
            Ie_rot=np.array([9.0, 11.0, 10.0]),
            Ig_rot=np.array([-1.0, 1.0, 0.0]),
        )
        self.analyzed_with = None

    def analyze(self, **kwargs):
        self.analyzed_with = kwargs


@pytest.fixture
def station():
    return FakeStation()


def test_it_analyzes_before_reading_the_fit(station):
    """The fields written must be the ones just fitted, not a stale pass."""
    expt = FakeHistogram()

    apply_singleshot_calibration(station, expt)

    assert expt.analyzed_with is not None, "analyze was never called"
    assert expt.analyzed_with["plot"] is False
    assert expt.analyzed_with["station"] is station
    assert expt.analyzed_with["subdir"] == station.autocalib_path


def test_the_angle_accumulates_onto_the_existing_phase(station):
    """The fit is a residual through the current rotation, not an absolute."""
    apply_singleshot_calibration(station, FakeHistogram(angle=25.0))

    assert station.hardware_cfg.device.readout.phase == [35.0]


def test_threshold_is_written_both_ways_it_is_read(station):
    apply_singleshot_calibration(station, FakeHistogram())

    readout = station.hardware_cfg.device.readout
    assert readout.threshold == [3.5]
    assert readout.threshold_list == [[3.5]]


def test_blob_centres_are_medians_of_the_rotated_shots(station):
    apply_singleshot_calibration(station, FakeHistogram())

    readout = station.hardware_cfg.device.readout
    assert readout.Ie == [10.0]
    assert readout.Ig == [0.0]


def test_active_reset_selects_which_confusion_matrix_is_filled(station):
    """The two are not interchangeable: a reset changes the error rates."""
    apply_singleshot_calibration(station, FakeHistogram(active_reset=True))

    readout = station.hardware_cfg.device.readout
    assert "confusion_matrix_with_active_reset" in readout
    assert "confusion_matrix_without_reset" not in readout


def test_without_active_reset_the_other_field_is_filled(station):
    apply_singleshot_calibration(station, FakeHistogram(active_reset=False))

    readout = station.hardware_cfg.device.readout
    assert "confusion_matrix_without_reset" in readout
    assert "confusion_matrix_with_active_reset" not in readout


def test_nothing_is_snapshotted(station):
    """Persisting a config version stays the caller's decision.

    The station fake has no snapshot methods at all, so a call to one would
    raise here rather than quietly writing a version during a test.
    """
    apply_singleshot_calibration(station, FakeHistogram())
    assert not hasattr(station, "snapshot_hardware_config")

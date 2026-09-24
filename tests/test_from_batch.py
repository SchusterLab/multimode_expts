# -*- coding: utf-8 -*-
"""`from_batch` converts an acquisition aggregate into a stage Experiment.

The seam this covers
--------------------
`BatchRunner` instantiates its `ExptClass` to hold the batch. That class is
provenance -- it is what the queue recorded and what names the HDF5 file --
so an acquisition cell must keep passing whatever it always passed
(`EncSpec`, in jonginn's notebooks). But aggregate *analysis* moved to four
stage classes, and `EncSpec.analyze(stage=...)` now raises on purpose.

Without a converter the only route from one to the other is to re-read every
job off disk. `from_batch` reuses the already-loaded children instead.

Run:  pixi run python -m pytest tests/test_from_batch.py -v
"""
import pytest
from slab import AttrDict

from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment as EncSpec,
)
from experiments.qsim.legacy_mbr import MBROrthogonalityExperiment
from experiments.qsim.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.legacy_mbr import MBRPropagatorExperiment
from experiments.qsim.legacy_mbr import MBRSpectrumExperiment

STAGE_CLASSES = [MBRPhaseCorrectionExperiment, MBRSpectrumExperiment,
                 MBRPropagatorExperiment, MBROrthogonalityExperiment]


class _Child:
    """One acquired job, as `BatchRunner` leaves it in `batch_expts`."""

    def __init__(self, tag):
        self.cfg = AttrDict(dict(expt=AttrDict(dict(tag=tag))))
        self.data = AttrDict(dict(tag=tag))


def _aggregate(station=None, n=3):
    """What `BatchRunner.execute` returns, with `EncSpec` as the ExptClass."""
    raw = EncSpec.__new__(EncSpec)
    raw.batch_expts = [_Child(i) for i in range(n)]
    raw.batch_job_ids = [f"JOB-{i:03d}" for i in range(n)]
    raw._analysis_station = station
    return raw


@pytest.mark.parametrize("StageClass", STAGE_CLASSES,
                         ids=lambda c: c.__name__)
def test_from_batch_returns_the_stage_class(StageClass):
    """Every stage inherits it, and each returns an instance of *itself*."""
    raw = _aggregate()

    expt = StageClass.from_batch(raw)

    assert type(expt) is StageClass
    assert not isinstance(raw, StageClass)      # the point of the conversion


def test_from_batch_reuses_the_loaded_children(station):
    """No re-read: the same child objects come through, same order.

    A campaign is hundreds of jobs; round-tripping them through HDF5 to
    change the wrapper's class would be the expensive way to do nothing.
    """
    raw = _aggregate(station=station)

    expt = MBRSpectrumExperiment.from_batch(raw)

    assert expt.batch_expts == raw.batch_expts
    assert all(a is b for a, b in zip(expt.batch_expts, raw.batch_expts))
    assert [child.data.tag for child in expt.batch_expts] == [0, 1, 2]


def test_from_batch_carries_job_ids_and_station(station):
    """`expt.batch_job_ids` is the line every acquisition cell prints next."""
    raw = _aggregate(station=station)

    expt = MBRPhaseCorrectionExperiment.from_batch(raw)

    assert expt.batch_job_ids == ["JOB-000", "JOB-001", "JOB-002"]
    assert expt._analysis_station is station


def test_an_explicit_station_wins(station):
    """The aggregate's station is a default, not an override."""
    raw = _aggregate(station=None)

    expt = MBRSpectrumExperiment.from_batch(raw, station=station)

    assert expt._analysis_station is station


def test_from_batch_starts_with_empty_data(station):
    """The aggregate's own `data` is the stage analysis' output slot.

    Inheriting the acquisition wrapper's `data` would make `analyze()` look
    like it had already run.
    """
    raw = _aggregate(station=station)
    raw.data = AttrDict(dict(stale="from the acquisition wrapper"))

    expt = MBRSpectrumExperiment.from_batch(raw)

    assert expt.data == {}


def test_from_batch_rejects_a_saved_job_list(station):
    """A list of files is `from_job_files`' job, and the error should say so."""
    with pytest.raises(TypeError, match="from_job_ids or from_job_files"):
        MBRSpectrumExperiment.from_batch(["JOB-001_EncSpec.h5"], station=station)


def test_from_batch_rejects_an_empty_batch(station):
    raw = _aggregate(station=station, n=0)
    with pytest.raises(ValueError, match="cannot be empty"):
        MBRSpectrumExperiment.from_batch(raw)


def test_the_stage_shim_still_refuses_the_old_call(station):
    """`from_batch` is the migration, so the thing it replaces must stay dead.

    If `analyze(stage=...)` ever started working again, every notebook would
    keep the old shape and this helper would go unused.
    """
    raw = _aggregate(station=station)
    with pytest.raises(TypeError, match="MBRSpectrumExperiment"):
        raw.analyze(stage="spectrum")

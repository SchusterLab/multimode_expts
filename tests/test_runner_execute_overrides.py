# -*- coding: utf-8 -*-
"""CharacterizationRunner.execute(overrides=...) runs one job per override dict.

This is the old `BatchRunner` path, merged into `CharacterizationRunner`
(docs/qsim/mbr_redesign.md, section 4). In queue mode at most `batch_size`
jobs wait in the queue at a time; the next group goes in when the current one
is collected. The result is a plain list of the job Experiments, in config
order, and the queue job IDs are in `runner.last_job_ids`.

These tests drive `execute()` end to end against a fake queue. The shared
`station` fixture is a mock station, and a mock station runs locally unless
the call asks for the queue, so the queue tests pass `use_queue=True`. The
fake client is not a real `JobClient`, so the mock + queue guard lets it
through.

Not tested here: anything about physics, or the real job server. The fake
`job_client` below is the whole queue.

Run:  pixi run python -m pytest tests/test_runner_execute_overrides.py -v
"""
import json

import numpy as np
import pytest
from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner, json_plain
from tests.test_characterization_runner import MockExperiment


class _FakeResult:
    """What `wait_for_completion` hands back."""

    def __init__(self, expt, status="completed", error_message=None):
        self._expt = expt
        self.status = status
        self.error_message = error_message

    def is_successful(self):
        return self.status == "completed"

    def load_expt(self):
        return self._expt


class _FakeJobClient:
    """A queue that completes every job immediately, in submission order."""

    def __init__(self, station, fail_on=None):
        self.station = station
        self.fail_on = fail_on          # job_id to report as failed
        self.submitted = []             # kwargs of every submit_job call
        self.cancelled = []
        self.in_queue = 0               # submitted but not yet collected
        self.max_in_queue = 0
        self._next_id = 1

    def submit_job(self, **kwargs):
        job_id = f"JOB-{self._next_id:03d}"
        self._next_id += 1
        self.submitted.append((job_id, kwargs))
        self.in_queue += 1
        self.max_in_queue = max(self.max_in_queue, self.in_queue)
        return job_id

    def wait_for_completion(self, job_id, **kwargs):
        self.in_queue -= 1
        if job_id == self.fail_on:
            return _FakeResult(None, status="failed", error_message="boom")
        expt = MockExperiment(path=str(self.station.data_path))
        expt.data = AttrDict(dict(job_id=job_id))
        return _FakeResult(expt)

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)


def _runner(station, **kwargs):
    client = kwargs.pop("job_client", None) or _FakeJobClient(station)
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10, start=0)),
        job_client=client,
        show=False,
        **kwargs,
    )
    return runner, client


def test_queue_returns_the_job_experiments_in_order(station):
    runner, client = _runner(station)

    expts = runner.execute(overrides=[{}, {}, {}], batch_size=2, use_queue=True,
                           log=False, show=False)

    assert len(client.submitted) == 3
    assert isinstance(expts, list)
    assert [e.data.job_id for e in expts] == ["JOB-001", "JOB-002", "JOB-003"]
    assert runner.last_job_ids == ["JOB-001", "JOB-002", "JOB-003"]


def test_batch_size_bounds_the_jobs_in_the_queue(station):
    """With batch_size=2 and 5 override dicts the groups are 2/2/1, and every job
    still comes back, in config order."""
    runner, client = _runner(station)

    expts = runner.execute(overrides=[{"reps": i} for i in range(5)],
                           batch_size=2, use_queue=True, log=False, show=False)

    assert client.max_in_queue == 2
    assert [job_id for job_id, _ in client.submitted] == [
        "JOB-001", "JOB-002", "JOB-003", "JOB-004", "JOB-005"]
    assert [e.data.job_id for e in expts] == runner.last_job_ids
    assert [kw["expt_config"]["reps"] for _, kw in client.submitted] == [0, 1, 2, 3, 4]


def test_each_config_is_merged_over_the_shared_kwargs(station):
    runner, client = _runner(station)

    runner.execute(overrides=[{"reps": 1}, {}], use_queue=True, start=7,
                   log=False, show=False)

    configs = [kw["expt_config"] for _, kw in client.submitted]
    assert configs == [dict(reps=1, start=7), dict(reps=10, start=7)]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, np.bool_(True)])
def test_rejects_bad_batch_size(station, batch_size):
    """`bool` is a subclass of `int`, so without the explicit bool rejection
    `batch_size=True` would silently mean 1."""
    runner, _ = _runner(station)
    with pytest.raises(ValueError, match="batch_size"):
        runner.execute(overrides=[{}], batch_size=batch_size, use_queue=True)


def test_queue_mode_requires_a_job_client(station):
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10)),
        job_client=None,
        show=False,
    )
    with pytest.raises(ValueError, match="job_client"):
        runner.execute(overrides=[{}], use_queue=True)


def test_rejects_empty_overrides(station):
    runner, _ = _runner(station)
    with pytest.raises(ValueError, match="overrides cannot be empty"):
        runner.execute(overrides=[], use_queue=True)


def test_records_provenance(station):
    """Every job carries the class and module it was acquired under. The
    worker re-imports by these strings."""
    runner, client = _runner(station, ExptProgram=None)

    runner.execute(overrides=[{}], batch_size=1, use_queue=True, log=False, show=False)

    _, kwargs = client.submitted[0]
    assert kwargs["experiment_class"] == "MockExperiment"
    assert kwargs["experiment_module"] == MockExperiment.__module__
    assert kwargs["program_class"] is None
    assert kwargs["program_module"] is None
    assert kwargs["user"] == station.user


def test_records_the_program_when_there_is_one(station):
    class FakeProgram:
        pass

    runner, client = _runner(station, ExptProgram=FakeProgram)

    runner.execute(overrides=[{}], batch_size=1, use_queue=True, log=False, show=False)

    _, kwargs = client.submitted[0]
    assert kwargs["program_class"] == "FakeProgram"
    assert kwargs["program_module"] == FakeProgram.__module__


def test_json_plain_converts_numpy_for_the_queue():
    plain = json_plain({
        "array": np.arange(3),
        "scalar": np.float64(2.5),
        "flag": np.bool_(True),
        "nested": [np.int64(7), {"deep": np.arange(2)}],
        "tuple": (np.int64(1), 2),
        "left_alone": "text",
    })
    assert plain["array"] == [0, 1, 2]
    assert plain["scalar"] == 2.5 and isinstance(plain["scalar"], float)
    assert plain["flag"] is True
    assert plain["nested"] == [7, {"deep": [0, 1]}]
    assert plain["tuple"] == [1, 2]
    assert plain["left_alone"] == "text"
    json.dumps(plain)          # the actual requirement


def test_a_failed_job_raises_and_cancels_the_rest(station):
    """The queue must not be left holding jobs nobody is waiting for."""
    client = _FakeJobClient(station, fail_on="JOB-002")
    runner, _ = _runner(station, job_client=client)

    with pytest.raises(RuntimeError, match="JOB-002"):
        runner.execute(overrides=[{}, {}, {}], batch_size=3, use_queue=True,
                       log=False, show=False)

    assert "JOB-003" in client.cancelled


def test_local_mode_runs_each_config(station):
    """Local runs need no job client and record no job IDs."""
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(start=0, step=60, expts=10)),
        use_queue=False,
        show=False,
    )

    expts = runner.execute(overrides=[dict(expts=4), dict(expts=6)])

    assert [len(e.data["xpts"]) for e in expts] == [4, 6]
    assert runner.last_job_ids == []


def test_mock_station_runs_locally_unless_the_call_asks(station):
    """A mock station ignores the runner's default use_queue=True."""
    runner, client = _runner(station)
    runner.default_expt_cfg = AttrDict(dict(start=0, step=60, expts=3))

    expts = runner.execute(overrides=[{}])

    assert client.submitted == []
    assert len(expts) == 1

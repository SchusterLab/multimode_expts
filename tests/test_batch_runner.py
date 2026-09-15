# -*- coding: utf-8 -*-
"""BatchRunner.execute() actually runs.

Why this file exists
--------------------
`BatchRunner` moved out of `floquet_dark_mode_readout.py` into its own module
and the move did not carry `numpy`, `copy.deepcopy` or `slab.AttrDict` with
it. Those three are used only inside `execute()`, so `import
experiments.batch_runner` stayed green and nothing in the suite noticed: the
first `NameError` would have landed on the measurement PC, mid-campaign,
after the queue had already been handed jobs.

An import-only test would not have caught it either. The check has to reach
the lines that use those names:

  * `np`       -- the `batch_size` type guard, first statement past the
                  `job_client` check, and `_plain`'s ndarray/generic branches;
  * `deepcopy` -- building the aggregate's `cfg`/`cfg.expt`;
  * `AttrDict` -- the aggregate's `cfg` and its empty `data`.

So these tests drive `execute()` end to end against a fake queue and assert
on the aggregate that comes back.

Not tested here: anything about physics, or the real job server. The fake
`job_client` below is the whole queue. `execute` needs one, by design, so
this is the only way to exercise it off-prod.

Run:  pixi run python -m pytest tests/test_batch_runner.py -v
"""
import numpy as np
import pytest
from slab import AttrDict

from experiments.batch_runner import BatchRunner

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
        self._next_id = 1

    def submit_job(self, **kwargs):
        job_id = f"JOB-{self._next_id:03d}"
        self._next_id += 1
        self.submitted.append((job_id, kwargs))
        return job_id

    def wait_for_completion(self, job_id, **kwargs):
        if job_id == self.fail_on:
            return _FakeResult(None, status="failed", error_message="boom")
        expt = MockExperiment(path=str(self.station.data_path))
        expt.data = AttrDict(dict(job_id=job_id))
        return _FakeResult(expt)

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)


def _runner(station, **kwargs):
    client = kwargs.pop("job_client", None) or _FakeJobClient(station)
    runner = BatchRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10, start=0)),
        job_client=client,
        show=False,
        **kwargs,
    )
    return runner, client


def test_execute_returns_an_aggregate(station):
    """The end-to-end path: submit, collect, build the aggregate.

    This is the test the import regression fails. Every one of `np`,
    `deepcopy` and `AttrDict` is on this path.
    """
    runner, client = _runner(station)

    aggregate = runner.execute([{}, {}, {}], batch_size=2, log=False, show=False)

    assert len(client.submitted) == 3
    assert isinstance(aggregate, MockExperiment)
    assert len(aggregate.batch_expts) == 3
    assert aggregate.batch_job_ids == ["JOB-001", "JOB-002", "JOB-003"]
    # deepcopy + AttrDict: the aggregate carries a config of its own, not a
    # reference to the station's.
    assert isinstance(aggregate.cfg, AttrDict)
    assert aggregate.cfg.expt.reps == 10
    assert aggregate.cfg.expt is not runner.default_expt_cfg
    assert aggregate.data == {}
    assert aggregate._analysis_station is station


def test_execute_respects_batch_size(station):
    """batch_size bounds how many jobs are in flight, not how many run.

    Collection happens per group, so with batch_size=2 and 5 configs the
    groups are 2/2/1 and every job still comes back, in config order.
    """
    runner, client = _runner(station)

    aggregate = runner.execute([{"reps": i} for i in range(5)],
                               batch_size=2, log=False, show=False)

    assert [job_id for job_id, _ in client.submitted] == [
        "JOB-001", "JOB-002", "JOB-003", "JOB-004", "JOB-005"]
    assert aggregate.batch_job_ids == [job_id for job_id, _ in client.submitted]
    assert [e.data.job_id for e in aggregate.batch_expts] == aggregate.batch_job_ids


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, np.bool_(True)])
def test_execute_rejects_bad_batch_size(station, batch_size):
    """The `np.bool_` arm of the guard is why `execute` needs numpy at all.

    `bool` is a subclass of `int`, so without the explicit bool rejection
    `batch_size=True` would silently mean 1.
    """
    runner, _ = _runner(station)
    with pytest.raises(ValueError, match="batch_size"):
        runner.execute([{}], batch_size=batch_size)


def test_execute_requires_a_job_client(station):
    runner = BatchRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(reps=10)),
        job_client=None,
        show=False,
    )
    with pytest.raises(ValueError, match="job_client"):
        runner.execute([{}])


def test_execute_rejects_empty_configs(station):
    runner, _ = _runner(station)
    with pytest.raises(ValueError, match="configs cannot be empty"):
        runner.execute([])


def test_execute_records_provenance(station):
    """Every job carries the class and module it was acquired under.

    This is the recording the module docstring calls the reason names in this
    tree count as provenance: the worker re-imports by these strings, so a
    moved class breaks jobs submitted before the move.
    """
    runner, client = _runner(station, ExptProgram=None)

    runner.execute([{}], batch_size=1, log=False, show=False)

    _, kwargs = client.submitted[0]
    assert kwargs["experiment_class"] == "MockExperiment"
    assert kwargs["experiment_module"] == MockExperiment.__module__
    assert kwargs["program_class"] is None
    assert kwargs["program_module"] is None
    assert kwargs["user"] == station.user


class _ProgramTakingExperiment(MockExperiment):
    """`QsimBaseExperiment` takes `program=`; plain `Experiment` does not.

    `execute` builds the aggregate down one of two branches depending on
    whether the runner has a program, so covering the program arm needs a
    class with the qsim constructor.
    """

    def __init__(self, program=None, **kwargs):
        super().__init__(**kwargs)
        self.program = program


def test_execute_records_the_program_when_there_is_one(station):
    class FakeProgram:
        pass

    client = _FakeJobClient(station)
    runner = BatchRunner(
        station=station,
        ExptClass=_ProgramTakingExperiment,
        default_expt_cfg=AttrDict(dict(reps=10)),
        ExptProgram=FakeProgram,
        job_client=client,
        show=False,
    )

    aggregate = runner.execute([{}], batch_size=1, log=False, show=False)

    _, kwargs = client.submitted[0]
    assert kwargs["program_class"] == "FakeProgram"
    assert kwargs["program_module"] == FakeProgram.__module__
    # The program arm of the aggregate construction, not just the submission.
    assert aggregate.program is FakeProgram


def test_plain_converts_only_at_the_json_boundary(station):
    """`_plain` is the other numpy user, and the queue needs real JSON types."""
    plain = BatchRunner._plain({
        "array": np.arange(3),
        "scalar": np.float64(2.5),
        "flag": np.bool_(True),
        "nested": [np.int64(7), {"deep": np.arange(2)}],
        "left_alone": "text",
    })
    assert plain["array"] == [0, 1, 2]
    assert plain["scalar"] == 2.5 and isinstance(plain["scalar"], float)
    assert plain["flag"] is True
    assert plain["nested"] == [7, {"deep": [0, 1]}]
    assert plain["left_alone"] == "text"
    import json
    json.dumps(plain)          # the actual requirement


def test_a_failed_job_raises_and_cancels_the_rest(station):
    """The queue must not be left holding jobs nobody is waiting for."""
    client = _FakeJobClient(station, fail_on="JOB-002")
    runner, _ = _runner(station, job_client=client)

    with pytest.raises(RuntimeError, match="JOB-002"):
        runner.execute([{}, {}, {}], batch_size=3, log=False, show=False)

    assert "JOB-003" in client.cancelled

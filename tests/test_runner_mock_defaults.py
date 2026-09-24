# -*- coding: utf-8 -*-
"""On a mock station, the runners' execute() only acquires and saves.

Mock data is all zeros. A fit on it gives NaN or an error, and a
postprocessor would write that result into the station config for the next
cell. So on a mock station, `execute()` of `CharacterizationRunner`,
`SweepRunner` and `BatchRunner` defaults to no analyze and no postprocess
(see `mock_run_defaults` and docs/qsim/mock_suite_plan.md, step 1).
An explicit argument from the caller still wins.

The shared `station` fixture (tests/conftest.py) reports is_mock=True.
Flip `station._is_mock` to get the non-mock defaults.

Run:  pixi run python -m pytest tests/test_runner_mock_defaults.py -v
"""
from slab import AttrDict

from experiments.batch_runner import BatchRunner
from experiments.characterization_runner import CharacterizationRunner, mock_run_defaults
from experiments.sweep_runner import SweepRunner

from tests.test_characterization_runner import MockExperiment
from tests.test_sweep_runner import MockExperiment as MockSweepExperiment


def _counting_postprocessor():
    calls = []

    def post(station, expt):
        calls.append(expt)

    return post, calls


def _char_runner(station, post, Runner=CharacterizationRunner):
    return Runner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=AttrDict(dict(start=0, step=60, expts=10)),
        postprocessor=post,
        use_queue=False,
    )


def _sweep_runner(station, post):
    return SweepRunner(
        station=station,
        ExptClass=MockSweepExperiment,
        default_expt_cfg=AttrDict(dict(start=0, step=0.1, expts=10)),
        sweep_param='freq',
        postprocessor=post,
        use_queue=False,
    )


def test_mock_run_defaults_explicit_wins():
    out = mock_run_defaults(dict(postprocess=True, go_kwargs=dict(analyze=True)), local=True)
    assert out["postprocess"] is True
    assert out["go_kwargs"]["analyze"] is True


def test_mock_run_defaults_queue_has_no_go_kwargs():
    # run() passes unknown kwargs to the preprocessor, so no go_kwargs there.
    assert mock_run_defaults({}, local=False) == dict(postprocess=False)


def test_characterization_mock_acquires_only(station):
    post, calls = _counting_postprocessor()
    expt = _char_runner(station, post).execute()
    assert "xpts" in expt.data
    assert expt._analysis is None
    assert calls == []


def test_characterization_mock_explicit_wins(station):
    post, calls = _counting_postprocessor()
    expt = _char_runner(station, post).execute(
        postprocess=True, go_kwargs=dict(analyze=True, save=False))
    assert expt._analysis is not None
    assert len(calls) == 1


def test_characterization_real_defaults_unchanged(station):
    station._is_mock = False
    post, calls = _counting_postprocessor()
    expt = _char_runner(station, post).execute(go_kwargs=dict(save=False), show=False)
    assert expt._analysis is not None
    assert len(calls) == 1


def test_sweep_mock_acquires_only(station):
    post, calls = _counting_postprocessor()
    mother = _sweep_runner(station, post).execute(
        4990, 5010, 3, incremental_save=False)
    assert len(mother.data['freq_sweep']) == 3
    assert mother._chevron_analysis is None
    assert calls == []


def test_sweep_mock_explicit_wins(station):
    post, calls = _counting_postprocessor()
    mother = _sweep_runner(station, post).execute(
        4990, 5010, 3, incremental_save=False, analyze=True, postprocess=True)
    assert mother._chevron_analysis is not None
    assert len(calls) == 1


def test_sweep_real_defaults_unchanged(station):
    station._is_mock = False
    post, calls = _counting_postprocessor()
    mother = _sweep_runner(station, post).execute(
        4990, 5010, 3, incremental_save=False)
    assert mother._chevron_analysis is not None
    assert len(calls) == 1


def test_batch_mock_acquires_only(station):
    post, calls = _counting_postprocessor()
    batch = _char_runner(station, post, Runner=BatchRunner).execute(
        [dict(), dict()], use_queue=False)
    assert len(batch.batch_expts) == 2
    assert all(e._analysis is None for e in batch.batch_expts)
    assert calls == []


def test_batch_mock_explicit_wins(station):
    post, calls = _counting_postprocessor()
    batch = _char_runner(station, post, Runner=BatchRunner).execute(
        [dict(go_kwargs=dict(analyze=True, save=False))], postprocess=True, use_queue=False)
    assert batch.batch_expts[0]._analysis is not None
    assert len(calls) == 1


def test_batch_real_defaults_unchanged(station):
    station._is_mock = False
    post, calls = _counting_postprocessor()
    batch = _char_runner(station, post, Runner=BatchRunner).execute(
        [dict(go_kwargs=dict(save=False))], use_queue=False, show=False)
    assert batch.batch_expts[0]._analysis is not None
    assert len(calls) == 1

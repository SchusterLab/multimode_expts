"""Synthetic multiparity analysis and the normal Experiment save lifecycle."""

from copy import deepcopy
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.multiparity import (
    MultiparityQsimExperiment,
    multiparity_calibration_config,
    multiparity_mean_error,
)
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment


CONFUSION = np.array([
    [.8, .1, .1, 0],
    [.1, .7, .2, 0],
    [.05, .15, .7, .1],
    [0, .1, .1, .8],
])
COUNTS = np.array([[90, 17, 55, 38], [23, 47, 60, 70]])
CALIBRATION_FIELDS = [
    'confusion_matrix', 'nmod4_mean_corrected',
    'confusion_matrix_job_id', 'confusion_matrix_source',
    *[f'p_mod{n}_corrected' for n in range(4)],
]


def _shot_rows(counts):
    rows = []
    for row_counts in counts:
        labels = np.repeat(np.arange(4), row_counts)
        lanes = np.full((len(labels), 4), 100.)
        lanes[:, -2] = 1.5 + 2 * (labels % 2) - 1
        lanes[:, -1] = 1.5 + 2 * (labels // 2) - 1
        rows.append(lanes.reshape(-1))
    return np.asarray(rows)


def _experiment(*, two_dimensional=False):
    expt = MultiparityQsimExperiment.__new__(MultiparityQsimExperiment)
    expt.cfg = AttrDict({
        'read_num': 4,
        'expt': {'qubits': [0], 'readout': 'multiparity'},
        'device': {'readout': {'threshold': [1.5]}},
    })
    counts = np.tile(COUNTS, (2, 1)) if two_dimensional else COUNTS
    idata = _shot_rows(counts)
    expt.data = {'xpts': np.array([0, 2]), 'idata': idata,
                 'qdata': np.zeros_like(idata)}
    if two_dimensional:
        expt.data['ypts'] = np.array([10, 20])
    return expt


def _calibration(matrix=CONFUSION, **metadata):
    expt = MultiParityConfusionMatrixExperiment.__new__(MultiParityConfusionMatrixExperiment)
    expt.cfg = deepcopy(_experiment().cfg)
    expt.data = {'multiparity_confusion_matrix': matrix.copy()}
    expt.__dict__.update(metadata)
    return expt


def _measured_calibration(prepared=(2, 0, 3, 1)):
    expt = _calibration()
    expt.cfg.expt.update(cond_sec_phase=-90, phase_second_pulse=180,
                         prepulse=True, postpulse=True)
    counts = np.rint(CONFUSION[np.asarray(prepared)] * 200).astype(int)
    idata = _shot_rows(counts)
    expt.data = {'xpts': np.asarray(prepared), 'idata': idata,
                 'qdata': np.zeros_like(idata)}
    expt.analyze()
    return expt


def _calibration_file(tmp_path, matrix=CONFUSION):
    fname = tmp_path / 'calibration.h5'
    with h5py.File(fname, 'w') as data:
        data['multiparity_confusion_matrix'] = matrix
    return str(fname)


@pytest.mark.parametrize('two_dimensional', [False, True])
def test_config_matrix_and_experiment_give_same_correction(two_dimensional):
    expt = _experiment(two_dimensional=two_dimensional)
    raw = expt.analyze_multiparity()
    outputs = []
    for calibration in [CONFUSION.tolist(), _calibration(),
                        multiparity_calibration_config(_calibration())]:
        expt.cfg.expt.multiparity_confusion_matrix = calibration
        result = expt.analyze_multiparity()
        for key, value in raw.items():
            if key != 'n_err':
                np.testing.assert_array_equal(result[key], value)
        expected = np.array([1.15, 2.])
        if two_dimensional:
            expected = np.tile(expected, (2, 1))
        np.testing.assert_allclose(result['nmod4_mean_corrected'], expected, atol=1e-14)
        corrected = np.stack([result[f'p_mod{n}_corrected'] for n in range(4)], axis=-1)
        raw_probability = np.stack([raw[f'p_mod{n}'] for n in range(4)], axis=-1)
        np.testing.assert_allclose(corrected @ CONFUSION, raw_probability, atol=1e-14)
        np.testing.assert_allclose(corrected @ np.arange(4), expected, atol=1e-14)
        for n in range(4):
            key = f'p_mod{n}_corrected'
            assert result[key].shape == raw[f'p_mod{n}'].shape
            np.testing.assert_array_equal(expt.data[key], result[key])
        outputs.append(result)
    assert outputs[0].keys() == outputs[1].keys()
    for output in outputs[1:]:
        assert output.keys() == outputs[0].keys()
        for key in outputs[0]:
            np.testing.assert_array_equal(output[key], outputs[0][key])


def test_object_config_becomes_serializable_and_keeps_metadata_on_repeat():
    calibration = _measured_calibration()
    calibration.job_id = 'JOB-20261021-01234'
    calibration.fname = 'calibration.h5'
    original_data = deepcopy(calibration.data)
    expt = _experiment()
    expt.cfg.expt.multiparity_confusion_matrix = calibration
    first = expt.analyze_multiparity()

    assert expt.cfg.expt.multiparity_confusion_matrix == CONFUSION.tolist()
    json.dumps(expt.cfg)
    assert first['confusion_matrix_job_id'].item().decode() == calibration.job_id
    assert first['confusion_matrix_source'].item().decode() == calibration.fname
    assert not np.shares_memory(first['confusion_matrix'],
                                calibration.data['multiparity_confusion_matrix'])
    for key in original_data:
        np.testing.assert_array_equal(calibration.data[key], original_data[key])
    second = expt.analyze_multiparity()
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(second[key], first[key])


def test_replacing_or_removing_config_clears_previous_calibration_fields():
    expt = _experiment()
    raw = expt.analyze_multiparity()
    calibration = _measured_calibration()
    calibration.job_id = 'JOB-20261021-01234'
    calibration.fname = 'calibration.h5'
    expt.cfg.expt.multiparity_confusion_matrix = calibration
    expt.analyze_multiparity()

    expt.cfg.expt.multiparity_confusion_matrix = np.eye(4).tolist()
    changed = expt.analyze_multiparity()
    for key in ['confusion_matrix_job_id', 'confusion_matrix_source']:
        assert key not in changed
        assert key not in expt.data
    np.testing.assert_allclose(changed['nmod4_mean_corrected'], raw['mean_n_mod4'])
    np.testing.assert_allclose(changed['n_err'], raw['n_err'])
    for n in range(4):
        np.testing.assert_array_equal(changed[f'p_mod{n}_corrected'], raw[f'p_mod{n}'])

    expt.cfg.expt.multiparity_confusion_matrix = None
    again = expt.analyze_multiparity()
    assert raw.keys() == again.keys()
    for key in raw:
        np.testing.assert_array_equal(again[key], raw[key])
    for key in CALIBRATION_FIELDS:
        assert key not in expt.data


@pytest.mark.parametrize('two_dimensional', [False, True])
@pytest.mark.parametrize('calibration_kind', ['none', 'matrix', 'measured', 'serialized'])
def test_mean_uncertainty_matches_shot_statistics(two_dimensional, calibration_kind):
    expt = _experiment(two_dimensional=two_dimensional)
    weights = np.arange(4, dtype=float)
    if calibration_kind == 'matrix':
        expt.cfg.expt.multiparity_confusion_matrix = CONFUSION.tolist()
    elif calibration_kind == 'measured':
        expt.cfg.expt.multiparity_confusion_matrix = _measured_calibration()
    elif calibration_kind == 'serialized':
        expt.cfg.expt.multiparity_confusion_matrix = multiparity_calibration_config(
            _measured_calibration()
        )
    if calibration_kind != 'none':
        weights = np.linalg.solve(CONFUSION, weights)
    counts = np.tile(COUNTS, (2, 1)) if two_dimensional else COUNTS
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    shots = counts.sum(axis=1)
    expected = []
    for row_counts in counts:
        shot_values = np.repeat(weights, row_counts)
        expected.append(np.std(shot_values, ddof=1) / np.sqrt(len(shot_values)))
    result = expt.analyze_multiparity()
    np.testing.assert_allclose(result['n_err'].reshape(-1), expected, rtol=1e-14)
    assert result['n_err'].shape == result['mean_n_mod4'].shape
    helper_result = multiparity_mean_error(
        probabilities, shots,
        None if calibration_kind == 'none' else CONFUSION,
    )
    np.testing.assert_allclose(helper_result, expected, rtol=1e-14)


def test_calibration_rows_follow_prepared_number():
    calibration = _measured_calibration(prepared=(2, 0, 3, 1))
    np.testing.assert_array_equal(calibration.data['multiparity_confusion_matrix'], CONFUSION)
    assert 'multiparity_confusion_matrix_covariance' not in calibration.data


def test_old_calibration_covariance_is_ignored_and_removed_on_reanalysis():
    calibration = _measured_calibration()
    calibration.data['multiparity_confusion_matrix_covariance'] = np.full((4, 4, 4), np.nan)
    expt = _experiment()
    expt.data['confusion_matrix_covariance'] = np.full((4, 4, 4), np.nan)
    expt.cfg.expt.multiparity_confusion_matrix = calibration
    result = expt.analyze()
    assert 'confusion_matrix_covariance' not in result
    np.testing.assert_allclose(
        result['n_err'], multiparity_mean_error(COUNTS / 200, [200, 200], CONFUSION),
    )
    calibration.analyze()
    assert 'multiparity_confusion_matrix_covariance' not in calibration.data


def test_uncertainty_uses_each_points_shots_and_one_shot_is_nan():
    expt = _experiment()
    labels_by_point = [np.array([0, 1, 3]), np.array([0, 1, 1, 2, 3])]
    rows = []
    for labels in labels_by_point:
        rows.append(_shot_rows([np.bincount(labels, minlength=4)])[0])
    expt.data['idata'] = rows
    expt.data['qdata'] = [np.zeros_like(row) for row in rows]
    result = expt.analyze_multiparity()
    expected = [np.std(labels, ddof=1) / np.sqrt(len(labels)) for labels in labels_by_point]
    np.testing.assert_allclose(result['n_err'], expected, rtol=1e-14)

    expt.data['idata'] = [row[:4] for row in rows]
    expt.data['qdata'] = [np.zeros_like(row) for row in expt.data['idata']]
    assert np.isnan(expt.analyze_multiparity()['n_err']).all()


def test_analyze_only_updates_data_without_implicit_io(monkeypatch):
    expt = _experiment()
    acquired_data = deepcopy(expt.data)
    expt.data = {}
    expt.cfg.expt.multiparity_confusion_matrix = _calibration()

    def unexpected_io(*args, **kwargs):
        raise AssertionError('analysis must not save or read another job')

    monkeypatch.setattr(expt, 'save_data', unexpected_io)
    monkeypatch.setattr(expt, 'datafile', unexpected_io)
    monkeypatch.setattr(MultiparityQsimExperiment, 'from_h5file', unexpected_io)
    result = expt.analyze(data=acquired_data)
    assert result is expt.data
    np.testing.assert_allclose(result['nmod4_mean_corrected'], [1.15, 2.], atol=1e-14)
    assert all(isinstance(value, np.ndarray) for value in result.values())


@pytest.mark.parametrize('input_kind', ['experiment', 'serialized', 'filename'])
def test_go_and_hdf5_reload_preserve_correction_uncertainty_and_provenance(
        tmp_path, monkeypatch, input_kind):
    calibration = _measured_calibration()
    calibration.fname = str(tmp_path / 'JOB-20261021-01234_캘리브레이션.h5')
    calibration.save_data()
    loaded_calibration = MultiParityConfusionMatrixExperiment.from_h5file(calibration.fname)

    target = _experiment()
    acquired_data = deepcopy(target.data)
    target.data = {}
    calibration_input = loaded_calibration
    if input_kind == 'serialized':
        calibration_input = json.loads(json.dumps(multiparity_calibration_config(loaded_calibration)))
    elif input_kind == 'filename':
        calibration_input = calibration.fname
    target.cfg.expt.multiparity_confusion_matrix = calibration_input
    target.fname = str(tmp_path / 'target.h5')
    monkeypatch.setattr(target, 'acquire', lambda progress=False: acquired_data)
    target.go(save=True, analyze=True)
    expected = target.data
    with h5py.File(target.fname, 'r') as handle:
        assert set(handle.keys()) == set(expected.keys())
        assert all(handle[key].dtype.kind in 'iufS' for key in handle)
        config = json.loads(handle.attrs['config'])
        saved_calibration = config['expt']['multiparity_confusion_matrix']
        if input_kind in ['serialized', 'filename']:
            assert saved_calibration['matrix'] == CONFUSION.tolist()
            assert saved_calibration['fname'] == calibration.fname
        else:
            assert saved_calibration == CONFUSION.tolist()
        np.testing.assert_array_equal(handle['confusion_matrix'][...], CONFUSION)
        assert 'confusion_matrix_job_id' not in handle
        assert handle['confusion_matrix_source'][()].decode('utf-8') == calibration.fname
        assert handle['n_err'].shape == (2,)
        assert 'confusion_matrix_covariance' not in handle
    if input_kind == 'filename':
        def unexpected_file_lookup(fname):
            raise AssertionError('saved analysis must not need the original calibration file')

        monkeypatch.setattr(
            'experiments.qsim.multiparity.multiparity_calibration_from_file',
            unexpected_file_lookup,
        )
    loaded_target = MultiparityQsimExperiment.from_h5file(target.fname)
    actual = loaded_target.analyze()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('input_kind', ['matrix', 'experiment', 'serialized', 'filename'])
@pytest.mark.parametrize('matrix', [
    np.eye(3), np.full((4, 4), np.nan), 2 * np.eye(4),
    np.array([[1.1, -.1, 0, 0], *np.eye(4)[1:]]),
])
def test_existing_matrix_validation_applies_to_all_inputs(tmp_path, matrix, input_kind):
    expt = _experiment()
    calibration = matrix.tolist()
    if input_kind == 'experiment':
        calibration = _calibration(matrix)
    elif input_kind == 'serialized':
        calibration = multiparity_calibration_config(_calibration(matrix))
    elif input_kind == 'filename':
        calibration = _calibration_file(tmp_path, matrix)
    expt.cfg.expt.multiparity_confusion_matrix = calibration
    with pytest.raises(ValueError, match='finite, nonnegative 4x4 matrix'):
        expt.analyze_multiparity()


@pytest.mark.parametrize('input_kind', ['matrix', 'experiment', 'serialized', 'filename'])
def test_singular_matrix_keeps_existing_error_message(tmp_path, input_kind):
    expt = _experiment()
    matrix = np.full((4, 4), .25)
    calibration = matrix.tolist()
    if input_kind == 'experiment':
        calibration = _calibration(matrix)
    elif input_kind == 'serialized':
        calibration = multiparity_calibration_config(_calibration(matrix))
    elif input_kind == 'filename':
        calibration = _calibration_file(tmp_path, matrix)
    expt.cfg.expt.multiparity_confusion_matrix = calibration
    with pytest.raises(ValueError, match='multiparity_confusion_matrix is singular') as error:
        expt.analyze_multiparity()
    assert isinstance(error.value.__cause__, np.linalg.LinAlgError)


def test_filename_is_saved_without_parsing_a_job_id_from_it_or_prefix():
    calibration = _calibration(
        prefix='JOB-20261021-01234', fname='JOB-20261021-01234_calibration.h5',
    )
    expt = _experiment()
    expt.cfg.expt.multiparity_confusion_matrix = calibration
    result = expt.analyze_multiparity()
    assert 'confusion_matrix_job_id' not in result
    assert result['confusion_matrix_source'].item().decode() == calibration.fname


@pytest.mark.parametrize('preprocessor_kind', ['default', 'custom'])
@pytest.mark.parametrize('input_source', ['defaults', 'override'])
def test_filename_passes_normal_runner_and_worker_analysis(
        tmp_path, preprocessor_kind, input_source):
    from experiments.characterization_runner import CharacterizationRunner, default_preprocessor

    fname = _calibration_file(tmp_path)
    submitted = {}

    class Client:
        def submit_job(self, **job):
            submitted.update(json.loads(json.dumps(job)))
            return 'JOB-20261021-01235'

    runner = CharacterizationRunner.__new__(CharacterizationRunner)
    runner.ExptClass = MultiparityQsimExperiment
    runner.program = None
    runner.station = SimpleNamespace(user='test')
    runner.default_expt_cfg = AttrDict()
    overrides = dict(frequency=20)
    if input_source == 'defaults':
        runner.default_expt_cfg['multiparity_confusion_matrix'] = fname
    else:
        overrides['multiparity_confusion_matrix'] = fname
    preprocessor_calls = []

    def custom_preprocessor(station, defaults, **kwargs):
        preprocessor_calls.append(station)
        config = default_preprocessor(station, defaults, **kwargs)
        config['derived_frequency'] = config['frequency'] + 10
        return config

    runner.preprocessor = custom_preprocessor if preprocessor_kind == 'custom' else default_preprocessor
    runner.job_client = Client()
    runner._serialize_station_config = lambda: '{}'
    job_id = runner._submit(**overrides)
    assert job_id == 'JOB-20261021-01235'
    assert submitted['expt_config']['multiparity_confusion_matrix'] == fname
    assert submitted['expt_config']['frequency'] == 20
    if preprocessor_kind == 'custom':
        assert preprocessor_calls == [runner.station]
        assert submitted['expt_config']['derived_frequency'] == 30
    else:
        assert preprocessor_calls == []
    worker_expt = _experiment()
    worker_expt.cfg.expt.update(submitted['expt_config'])
    result = worker_expt.analyze()
    np.testing.assert_allclose(result['nmod4_mean_corrected'], [1.15, 2.], atol=1e-14)
    assert 'confusion_matrix_job_id' not in result
    assert result['confusion_matrix_source'].item().decode() == fname
    assert worker_expt.cfg.expt.multiparity_confusion_matrix == dict(
        matrix=CONFUSION.tolist(), fname=fname,
    )


def test_filename_must_exist(tmp_path):
    expt = _experiment()
    expt.cfg.expt.multiparity_confusion_matrix = str(tmp_path / 'missing.h5')
    with pytest.raises(FileNotFoundError):
        expt.analyze()


def test_filename_must_contain_the_saved_calibration_matrix(tmp_path):
    fname = tmp_path / 'not_a_calibration.h5'
    with h5py.File(fname, 'w') as data:
        data['unrelated_data'] = [0, 1, 2]
    expt = _experiment()
    expt.cfg.expt.multiparity_confusion_matrix = str(fname)
    with pytest.raises(KeyError, match='multiparity_confusion_matrix'):
        expt.analyze()

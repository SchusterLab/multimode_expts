"""Complete AGP child archives and flat disorder-by-cycle datasets."""

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.agp import CentralBosonAGPDisorderScan, CentralBosonAGPExperiment


def write_job(directory, name, disorder, *, contrast=.8, corrected=True, shots=20):
    directory.mkdir(parents=True, exist_ok=True)
    fname = directory / f'{name}.h5'
    expt = CentralBosonAGPExperiment.__new__(CentralBosonAGPExperiment)
    expt.cfg = AttrDict({
        'read_num': 4,
        'expt': {'photon_sector_number': 2, 'direction': 0, 'ro_stor': 7,
                 'swap_stors': [7], 'detunings': [disorder / 1000],
                 'readout': 'multiparity', 'qubits': [0]},
        'device': {'readout': {'threshold': [0.]}},
    })
    if corrected:
        matrix = contrast * np.eye(4) + (1 - contrast) / 4
        expt.cfg.expt.multiparity_confusion_matrix = {
            'matrix': matrix.tolist(), 'fname': f'calibration_{name}.h5',
        }
    rows = []
    for point in range(6):
        labels = np.repeat(np.arange(4), [point + 1, 2, 3, shots - point - 6])
        lanes = np.full((shots, 4), 100.)
        lanes[:, -2] = 2 * (labels % 2) - 1
        lanes[:, -1] = 2 * (labels // 2) - 1
        rows.append(lanes.reshape(-1))
    expt.data = {
        'xpts': np.array([0, 2, 5]), 'ypts': np.arange(2),
        'occupations': np.array([[1, 1], [2, 0]]),
        'idata': np.array(rows), 'qdata': np.zeros_like(rows),
        'unfamiliar_numeric_data': np.array([32768, 3], dtype=np.uint16),
    }
    expt.analyze()
    external = directory / f'{name}_external.h5'
    with h5py.File(external, 'w') as saved:
        group = saved.create_group('original_group')
        group.attrs['origin'] = 'external source'
        group['values'] = np.array([4., 5.])
    with h5py.File(fname, 'w') as saved:
        for key, value in expt.data.items():
            saved[key] = value
        saved.attrs['config'] = json.dumps(expt.cfg)
        saved.attrs['derived_params'] = json.dumps({'floquet_cycle_us': 1.2})
        saved.attrs['original_attribute'] = np.array([3, 8], dtype=np.int16)
        saved['annotation'] = '원본 측정 메모'
        nested = saved.create_group('unfamiliar_group')
        nested.attrs['group_attribute'] = 'preserve this too'
        values = nested.create_dataset('complex_data', data=np.array([1 + 2j, 3 - 4j]),
                                       compression='gzip', shuffle=True)
        values.attrs['units'] = 'ADC'
        saved['soft_link'] = h5py.SoftLink('/unfamiliar_group/complex_data')
        saved['external_link'] = h5py.ExternalLink(str(external), '/original_group')
    return fname


@pytest.fixture
def jobs(tmp_path):
    directory = tmp_path / 'data'
    return [write_job(directory, 'high', 40, contrast=.6),
            write_job(directory, 'zero', 0, contrast=.9),
            write_job(directory, 'middle', 20, contrast=.8)]


def assert_h5_contents_equal(source, copied):
    assert set(source.attrs) == set(copied.attrs)
    for key in source.attrs:
        np.testing.assert_array_equal(copied.attrs[key], source.attrs[key])
    assert set(source) == set(copied)
    for key, item in source.items():
        other = copied[key]
        if isinstance(item, h5py.Group):
            assert isinstance(other, h5py.Group)
            assert_h5_contents_equal(item, other)
        else:
            assert isinstance(other, h5py.Dataset)
            assert other.dtype == item.dtype
            assert other.shape == item.shape
            assert other.compression == item.compression
            assert other.shuffle == item.shuffle
            np.testing.assert_array_equal(other[()], item[()])
            assert set(item.attrs) == set(other.attrs)
            for attribute in item.attrs:
                np.testing.assert_array_equal(other.attrs[attribute], item.attrs[attribute])


def test_collects_sorted_jobs_and_keeps_each_jobs_correction(jobs):
    scan = CentralBosonAGPDisorderScan(jobs)
    data = scan.analyze()
    assert data is scan.data
    ordered = [jobs[1], jobs[2], jobs[0]]
    assert scan.fnames == [str(path.resolve()) for path in ordered]
    np.testing.assert_array_equal(data['disorder_khz'], [0., 20., 40.])
    np.testing.assert_array_equal(data['cycles'], [0, 2, 5])
    np.testing.assert_array_equal(data['detunings'], [[0.], [.02], [.04]])
    assert data['idata'].shape == (3, 6, 80)
    assert data['p_mod0'].shape == (3, 2, 3)
    assert data['unfamiliar_numeric_data'].dtype == np.uint16
    for index, fname in enumerate(ordered):
        with h5py.File(fname, 'r') as saved:
            for key in ['idata', 'qdata', 'p_mod0', 'n_err', 'confusion_matrix',
                        'unfamiliar_numeric_data', *scan.correlation_keys,
                        *[f'{key}_corrected' for key in scan.correlation_keys]]:
                np.testing.assert_array_equal(data[key][index], saved[key][()])
    assert not np.array_equal(data['confusion_matrix'][0], data['confusion_matrix'][1])
    for key in scan.correlation_keys:
        assert data[key].shape == data[f'{key}_corrected'].shape == (3, 3)


def test_saves_all_child_contents_and_attributes_without_changing_inputs(jobs, tmp_path):
    before = [hashlib.sha256(path.read_bytes()).digest() for path in jobs]
    scan = CentralBosonAGPDisorderScan(jobs, notes='disorder sweep 전체 보관')
    output = scan.save_data(tmp_path / 'assembled.h5')
    assert output == scan.fname == str((tmp_path / 'assembled.h5').resolve())
    with h5py.File(output, 'r') as saved:
        assert 'scan' not in saved and 'assembled' not in saved and 'processed' not in saved
        assert saved.attrs['notes'] == scan.notes
        assert json.loads(saved.attrs['source_files']) == scan.fnames
        assert list(saved['children']) == ['0000', '0001', '0002']
        for key in scan.data:
            np.testing.assert_array_equal(saved[key][()], scan.data[key])
        for index, fname in enumerate(scan.fnames):
            with h5py.File(fname, 'r') as original:
                assert_h5_contents_equal(original, saved[f'children/{index:04d}'])
    assert before == [hashlib.sha256(path.read_bytes()).digest() for path in jobs]
    assert not list(tmp_path.glob('*.yaml'))


def test_bundle_can_reanalyze_display_and_resave_after_originals_are_removed(jobs, tmp_path):
    scan = CentralBosonAGPDisorderScan(jobs)
    first = scan.save_data(tmp_path / 'first.h5')
    for path in (tmp_path / 'data').glob('*.h5'):
        path.unlink()
    loaded = CentralBosonAGPDisorderScan.from_h5file(first)
    assert loaded.fnames == scan.fnames
    for key in scan.data:
        np.testing.assert_array_equal(loaded.data[key], scan.data[key])
    reanalyzed = loaded.analyze()
    for key in scan.data:
        np.testing.assert_array_equal(reanalyzed[key], scan.data[key])
    fig, ax = loaded.display(moment=2)
    assert ax.get_xlabel() == 'Floquet cycles'
    plt.close(fig)
    loaded.data['new_processed_result'] = np.arange(3.)
    second = loaded.save_data(tmp_path / 'second.h5')
    with h5py.File(first, 'r') as original, h5py.File(second, 'r') as copied:
        assert_h5_contents_equal(original['children'], copied['children'])
        np.testing.assert_array_equal(copied['new_processed_result'][()], np.arange(3.))
    # The original in-memory collector also switches to its saved snapshots.
    scan.save_data(tmp_path / 'third.h5')


def test_ragged_and_child_specific_data_is_not_lost(tmp_path):
    first = write_job(tmp_path / 'data', 'first', 0, shots=12)
    second = write_job(tmp_path / 'data', 'second', 30, shots=24)
    with h5py.File(first, 'a') as saved:
        saved['only_in_first_job'] = np.array([6, 7, 8])
    scan = CentralBosonAGPDisorderScan([first, second])
    output = scan.save_data(tmp_path / 'ragged.h5')
    assert 'idata' not in scan.data  # Different shot counts cannot form a dense stack.
    with h5py.File(output, 'r') as saved:
        assert saved['children/0000/idata'].shape == (6, 48)
        assert saved['children/0001/idata'].shape == (6, 96)
        np.testing.assert_array_equal(saved['children/0000/only_in_first_job'][()], [6, 7, 8])
        for index, fname in enumerate([first, second]):
            with h5py.File(fname, 'r') as original:
                assert_h5_contents_equal(original, saved[f'children/{index:04d}'])


@pytest.mark.parametrize('calibrated', [False, True])
def test_missing_correction_never_substitutes_raw_values(tmp_path, calibrated):
    jobs = [write_job(tmp_path / 'data', 'first', 0, corrected=calibrated),
            write_job(tmp_path / 'data', 'second', 30, corrected=False)]
    scan = CentralBosonAGPDisorderScan(jobs)
    scan.analyze()
    for key in scan.correlation_keys:
        if calibrated:
            assert np.isfinite(scan.data[f'{key}_corrected'][0]).all()
            assert np.isnan(scan.data[f'{key}_corrected'][1]).all()
        else:
            assert f'{key}_corrected' not in scan.data
    fig, ax = scan.display()
    assert ('corrected' if calibrated else 'raw') in ax.get_title()
    plt.close(fig)


@pytest.mark.parametrize('moment', [1, 2])
@pytest.mark.parametrize('raw', [False, True])
def test_display_uses_selected_stacked_array(jobs, tmp_path, moment, raw):
    scan = CentralBosonAGPDisorderScan(jobs)
    scan.analyze()
    fig, ax = scan.display(moment=moment, raw=raw)
    key = scan.correlation_keys[moment - 1] + ('' if raw else '_corrected')
    np.testing.assert_array_equal(np.asarray(ax.collections[0].get_array()).reshape(3, 3),
                                  scan.data[key])
    assert ax.get_ylabel() == 'Disorder (kHz)'
    fig.savefig(tmp_path / f'moment-{moment}-raw-{raw}.png')
    plt.close(fig)


@pytest.mark.parametrize('field, replacement', [
    ('direction', 7), ('ro_stor', 0), ('swap_stors', [4]), ('photon_sector_number', 3),
])
def test_incompatible_jobs_are_not_combined(jobs, field, replacement):
    with h5py.File(jobs[0], 'a') as saved:
        config = json.loads(saved.attrs['config'])
        config['expt'][field] = replacement
        saved.attrs['config'] = json.dumps(config)
    with pytest.raises(ValueError, match=field):
        CentralBosonAGPDisorderScan(jobs).analyze()


def test_cycle_axis_mismatch_is_not_combined(jobs):
    with h5py.File(jobs[0], 'a') as saved:
        saved['xpts'][...] = [0, 3, 5]
    with pytest.raises(ValueError, match='cycle axis'):
        CentralBosonAGPDisorderScan(jobs).analyze()


def test_recorded_disorder_overrides_rms_and_duplicate_rows_are_preserved(jobs, tmp_path):
    with h5py.File(jobs[0], 'a') as saved:
        config = json.loads(saved.attrs['config'])
        config['expt']['disorder_khz'] = 20.
        saved.attrs['config'] = json.dumps(config)
    scan = CentralBosonAGPDisorderScan(jobs)
    scan.save_data(tmp_path / 'duplicate.h5')
    np.testing.assert_array_equal(scan.data['disorder_khz'], [0, 20, 20])
    assert len(scan.fnames) == 3
    with pytest.raises(ValueError, match='one job per disorder value'):
        scan.display()


def test_default_save_directory_and_existing_file_protection(jobs):
    scan = CentralBosonAGPDisorderScan(jobs)
    first = Path(scan.save_data())
    second = Path(scan.save_data())
    assert first.parent == jobs[0].parent.parent / 'assembled_data'
    assert first != second and first.exists() and second.exists()
    original = jobs[0].read_bytes()
    with pytest.raises(FileExistsError):
        scan.save_data(jobs[0])
    assert jobs[0].read_bytes() == original

"""Run the conversion notebook on synthetic files, without any prod data."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import h5py
import jupytext
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

NOTEBOOK = Path(__file__).resolve().parents[1] / 'analysis_notebooks/jonginn/convert_legacy_agp.py'
STATES = [list(state) for state in product(range(4), repeat=5)
          if sum(state) == 3 and state[-1] > 0]
MATRICES = {
    'coarse': .8 * np.eye(4) + .05,
    'fine': .6 * np.eye(4) + .1,
    'fine_zero': .9 * np.eye(4) + .025,
}


@pytest.fixture
def notebook():
    # Execute definition cells for validation; notebooks are not library imports.
    cells = jupytext.read(NOTEBOOK).cells
    namespace = {'__name__': 'legacy_conversion_test'}
    for cell in cells:
        if 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, str(NOTEBOOK), 'exec'), namespace)
    namespace['display'] = lambda value: None
    return namespace, cells


def write_job(fname, occupation, cycles, confusion, threshold=.2, calibration=False,
              phase=-90, old_flags=False):
    """Forty shots per point, with known measured and corrected distributions."""
    true_labels = ([occupation[0]] if calibration else
                   [(occupation[-1] + point) % 4 for point in range(len(cycles))])
    rows = []
    for label in true_labels:
        labels = np.repeat(np.arange(4), np.rint(40 * confusion[label]).astype(int))
        lanes = np.full((40, 4), threshold + 100.)  # Reset lanes must be ignored.
        lanes[:, -2] = threshold + np.where(labels % 2, 1., -1.)
        lanes[:, -1] = threshold + np.where(labels // 2, 1., -1.)
        rows.append(lanes.ravel())
    config = {
        'read_num': 4,
        'device': {'readout': {'threshold': [threshold]}},
        'expt': {
            'qubits': [0], 'swap_stors': [4, 5, 6, 7],
            'ro_stor': 0 if calibration else 7, 'occupations': occupation,
            'readout': 'multiparity', 'prepulse': False, 'postpulse': True,
            'cond_sec_phase': phase, 'phase_second_pulse': 180,
            'detunings': [0., 0., 0., 0.], 'floquet_cycles': list(cycles),
        },
    }
    if old_flags:
        del config['expt']['readout']
        config['expt']['multiparity_readout'] = True
    fname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fname, 'x') as saved:
        saved.attrs['config'] = json.dumps(config)
        saved.attrs['unknown_attribute'] = np.array([2., 7.])
        saved.attrs['derived_params'] = json.dumps({'floquet_cycle_us': .3})
        saved.create_dataset('xpts', data=cycles)
        raw = np.asarray(rows)
        saved.create_dataset('idata', data=raw[0] if calibration else raw)
        saved.create_dataset('qdata', data=np.zeros_like(raw[0] if calibration else raw))
        saved.create_dataset('avgi', data=np.arange(len(cycles))[None, :])
        saved.create_dataset('unknown_text', data=b'keep this too')
        nested = saved.create_group('old_analysis')
        nested.attrs['description'] = 'legacy analysis'
        dataset = nested.create_dataset('trace', data=np.arange(12), compression='gzip', shuffle=True)
        dataset.attrs['units'] = 'arbitrary'


def make_calibration(notebook, directory, phase=-90):
    files = [directory / f'cal_{n}.h5' for n in [2, 0, 3, 1]]
    for fname, n in zip(files, [2, 0, 3, 1]):
        write_job(fname, [n, 0, 0, 0, 0], [0], MATRICES['fine'],
                  calibration=True, phase=phase)
    return files


def make_agp(directory, cycles=(0, 1, 2)):
    files = [directory / f'state_{n}.h5' for n in range(len(STATES))]
    for fname, occupation in zip(files, reversed(STATES)):
        write_job(fname, occupation, cycles, MATRICES['fine'])
    return files


def test_full_notebook_preserves_192_sources_and_corrects_all_17_outputs(notebook, tmp_path, monkeypatch):
    ns, cells = notebook
    groups = ns['legacy_groups'](tmp_path / 'data')
    sources = set()
    for group_index, (name, group) in enumerate(groups.items()):
        matrix = MATRICES[name]
        threshold = .1 * (group_index + 1)
        for fname, prepared in zip(group['calibration'], [2, 0, 3, 1]):
            write_job(fname, [prepared, 0, 0, 0, 0], [0], matrix, threshold,
                      calibration=True, old_flags=name == 'coarse')
            sources.add(fname.resolve())
        # Coarse and fine may have different grids; within each assembly they agree.
        cycles = [0, 5, 10] if name == 'coarse' else [0, 1, 2, 3]
        for files in group['scans'].values():
            for fname, occupation in zip(files, reversed(STATES)):
                write_job(fname, occupation, cycles, matrix, threshold,
                          old_flags=name == 'coarse')
                sources.add(fname.resolve())
    assert len(sources) == 192
    before = {fname: ns['file_sha256'](fname) for fname in sources}
    real_init = h5py.File.__init__
    output_writes = []

    def checked_file(self, name, mode='r', *args, **kwargs):
        if isinstance(name, (str, Path)):  # h5py also constructs wrappers from open handles.
            path = Path(name).resolve()
            if path in sources:
                assert mode == 'r', f'Original opened for writing: {path}'
            elif mode != 'r':
                assert mode == 'x'
                output_writes.append(path)
        real_init(self, name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py.File, '__init__', checked_file)
    ns.update(groups=groups, DIRECTION=7, CONVERTED_DIR=tmp_path / 'converted',
              ASSEMBLED_DIR=tmp_path / 'assembled')
    plots = []
    monkeypatch.setattr(ns['plt'], 'show', lambda: plots.append(True))
    for cell in cells:
        tags = cell.metadata.get('tags', [])
        if cell.cell_type == 'code' and 'definitions' not in tags and 'settings' not in tags:
            exec(compile(cell.source, str(NOTEBOOK), 'exec'), ns)
    assert len(output_writes) == 17
    assert len(plots) == 8
    assert len(ns['conversion_checks']) == 15
    for fname in sources:
        assert ns['file_sha256'](fname) == before[fname]

    for name, fname in ns['converted_calibrations'].items():
        calibration = ns['MultiParityConfusionMatrixExperiment'].from_h5file(fname)
        np.testing.assert_allclose(calibration.data['multiparity_confusion_matrix'], MATRICES[name])
        assert calibration.cfg.expt.prepulse is False
        calibration.analyze()
        np.testing.assert_allclose(calibration.data['multiparity_confusion_matrix'], MATRICES[name])

    for name, scans in ns['converted_agp'].items():
        for disorder, fname in scans.items():
            calibration_name = 'fine_zero' if disorder == 0 else name
            agp = ns['CentralBosonAGPExperiment'].from_h5file(fname)
            np.testing.assert_array_equal(agp.data['occupations'], STATES)
            cycles = agp.data['xpts']
            labels = (np.asarray(STATES)[:, -1, None] + np.arange(len(cycles))) % 4
            corrected = np.eye(4)[labels]
            measured = corrected @ MATRICES[calibration_name]
            for n in range(4):
                np.testing.assert_allclose(agp.data[f'p_mod{n}'], measured[..., n])
                np.testing.assert_allclose(agp.data[f'p_mod{n}_corrected'], corrected[..., n], atol=1e-13)
            np.testing.assert_allclose(agp.data['nmod4_mean_corrected'], labels, atol=1e-13)
            np.testing.assert_allclose(agp.data['nmod4_second_moment_corrected'], labels**2, atol=1e-13)
            initial = np.asarray(STATES)[:, -1]
            np.testing.assert_allclose(agp.data['nmod4_ensemble_averaged_correlation_corrected'],
                                       initial @ labels / 35)
            np.testing.assert_allclose(agp.data['nmod4_second_moment_ensemble_averaged_correlation_corrected'],
                                       initial**2 @ labels**2 / 35)
            assert agp.cfg.expt.multiparity_confusion_matrix['fname'] == str(ns['converted_calibrations'][calibration_name])

    for name, disorders in [('coarse', [30, 60, 90, 120, 150]), ('fine', [0, 30, 40, 60, 90, 120, 150])]:
        scan = ns['CentralBosonAGPDisorderScan'].from_h5file(ns['assembled_files'][name])
        np.testing.assert_array_equal(scan.data['disorder_khz'], disorders)
        assert scan.data['nmod4_ensemble_averaged_correlation_corrected'].shape == (len(disorders), len(scan.data['cycles']))
        with h5py.File(scan.fname, 'r') as saved:
            assert len(saved['children']) == len(disorders)
            for child in saved['children'].values():
                assert len(child['legacy_jobs']) == 15
                assert 'old_analysis/trace' in child['legacy_jobs/0000']
        previous = deepcopy(scan.data)
        scan.analyze()  # Rebuild from embedded children, not from the original paths.
        for key, value in previous.items():
            np.testing.assert_array_equal(scan.data[key], value)


def test_conversion_never_overwrites_an_input_or_existing_output(notebook, tmp_path):
    ns, _ = notebook
    files = make_calibration(ns, tmp_path / 'data')
    hashes = [ns['file_sha256'](fname) for fname in files]
    with pytest.raises(FileExistsError):
        ns['convert_calibration'](files, files[0])
    output = ns['convert_calibration'](files, tmp_path / 'calibration.h5')
    with pytest.raises(FileExistsError):
        ns['convert_calibration'](files, output)
    assert hashes == [ns['file_sha256'](fname) for fname in files]


def test_calibration_state_coverage_and_acquisition_only_phase_check(notebook, tmp_path):
    ns, _ = notebook
    files = make_calibration(ns, tmp_path / 'missing')
    with pytest.raises(ValueError, match='each of 0, 1, 2, 3'):
        ns['convert_calibration']([files[0], *files[:-1]], tmp_path / 'missing.h5')
    files = make_calibration(ns, tmp_path / 'phase', phase=90)
    # Analysis uses saved shots without repeating acquisition checks.
    output = ns['convert_calibration'](files, tmp_path / 'phase.h5')
    calibration = ns['MultiParityConfusionMatrixExperiment'].from_h5file(output)
    np.testing.assert_allclose(calibration.data['multiparity_confusion_matrix'], MATRICES['fine'])
    assert calibration.cfg.expt.cond_sec_phase == 90
    assert calibration.cfg.expt.prepulse is False
    calibration.analyze()
    # The original phase and pulse requirements still apply to new acquisition.
    calibration.cfg.expt.prepulse = True
    with pytest.raises(ValueError, match='cond_sec_phase=-90'):
        calibration.acquire()


def test_agp_rejects_missing_state_and_mismatched_cycles_or_readout(notebook, tmp_path):
    ns, _ = notebook
    calibration = ns['convert_calibration'](make_calibration(ns, tmp_path / 'cal'), tmp_path / 'cal.h5')
    files = make_agp(tmp_path / 'scan')
    with pytest.raises(ValueError, match='each selected occupation exactly once'):
        ns['convert_agp'](files[:-1], calibration, tmp_path / 'missing.h5', 30)
    # These are disposable synthetic inputs, newly created for the invalid cases.
    other = tmp_path / 'other.h5'
    write_job(other, STATES[-1], [0, 2, 3], MATRICES['fine'])
    with pytest.raises(ValueError, match='share their cycle axis'):
        ns['convert_agp']([other, *files[1:]], calibration, tmp_path / 'cycles.h5', 30)
    wrong_readout = tmp_path / 'threshold.h5'
    write_job(wrong_readout, STATES[-1], [0, 1, 2], MATRICES['fine'], threshold=.8)
    with pytest.raises(ValueError, match='different readout/classification settings'):
        ns['convert_agp']([wrong_readout, *files[1:]], calibration, tmp_path / 'threshold_out.h5', 30)


def test_assembler_checks_cycle_axes_between_disorders(notebook, tmp_path):
    ns, _ = notebook
    calibration = ns['convert_calibration'](make_calibration(ns, tmp_path / 'cal'), tmp_path / 'cal.h5')
    outputs = []
    for disorder, cycles in [(30, [0, 1, 2]), (60, [0, 2, 4])]:
        files = make_agp(tmp_path / str(disorder), cycles)
        outputs.append(ns['convert_agp'](files, calibration, tmp_path / f'W{disorder}.h5', disorder))
    with pytest.raises(ValueError, match='share the same cycle axis'):
        ns['CentralBosonAGPDisorderScan'](outputs).analyze()


def test_single_cycle_agp_keeps_its_point_axis(notebook, tmp_path):
    ns, _ = notebook
    calibration = ns['convert_calibration'](make_calibration(ns, tmp_path / 'cal'), tmp_path / 'cal.h5')
    files = make_agp(tmp_path / 'scan', cycles=[0])
    output = ns['convert_agp'](files, calibration, tmp_path / 'scan.h5', 0)
    assert ns['verify_converted'](output)['reanalysis'] == 'same'
    agp = ns['CentralBosonAGPExperiment'].from_h5file(output)
    assert agp.data['nmod4_mean_corrected'].shape == (15, 1)


@pytest.mark.parametrize('prepulse,postpulse', [(False, True), (True, False), (True, True)])
def test_calibration_acquisition_keeps_existing_pulse_requirements(notebook, monkeypatch, prepulse, postpulse):
    ns, _ = notebook
    from experiments.qsim.qsim_base import QsimExperiment
    cls = ns['MultiParityConfusionMatrixExperiment']
    calibration = cls.__new__(cls)
    calibration.cfg = ns['AttrDict']({'expt': {'readout': 'multiparity', 'cond_sec_phase': -90,
                                             'prepulse': prepulse, 'postpulse': postpulse}})
    monkeypatch.setattr(QsimExperiment, 'acquire', lambda self, **kwargs: 'acquired')
    if prepulse and postpulse:
        assert calibration.acquire() == 'acquired'
    else:
        with pytest.raises(ValueError, match='prepulse and postpulses'):
            calibration.acquire()

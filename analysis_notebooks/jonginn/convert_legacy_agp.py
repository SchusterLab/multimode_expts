# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Legacy AGP conversion — 원본은 읽기만 합니다
#
# pippin의 repo 환경에서 Jupytext notebook으로 열고 위에서부터 실행합니다.
# SSH, station, runner, worker, 측정 장비는 사용하지 않습니다.
#
# 1. 지정한 legacy 파일을 읽고 목록과 원본 SHA-256을 확인합니다.
# 2. 준비 상태별 네 JOB을 새 confusion-matrix 파일 하나로 변환합니다.
# 3. occupation별 15 JOB을 disorder별 새 CentralBosonAGPExperiment로 변환합니다.
# 4. coarse/fine 각각을 assemble하고 color map을 표시합니다.
#
# 원본 HDF5는 항상 `r`, 결과 HDF5는 항상 `x`로 엽니다. 기존 파일에는
# 덮어쓰지 않습니다. 모든 원본 dataset/group/attribute는 새 파일의
# `legacy_jobs/0000`, ... 안에 보존합니다. 새 분석 결과는 최상위에 저장합니다.
# calibration은 각 측정에 연결된 변환본을 사용합니다. AGP 적분, T1 보정,
# matrix-pencil fitting, calibration 통계 오차 전파는 추가하지 않습니다.

# %% tags=["definitions"]
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from slab import AttrDict
from slab.experiment import NpEncoder

from experiments.qsim.agp import CentralBosonAGPDisorderScan, CentralBosonAGPExperiment
from experiments.qsim.multiparity import MultiparityQsimExperiment
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.qsim_base import readout_lane_count, saved_readout_mode


# %% [markdown]
# ## 첨부 목록의 파일 연결
#
# 날짜와 번호는 제공된 목록을 명시적으로 옮겼습니다. 디렉터리 검색이나
# 파일 순서로 준비 상태·occupation·calibration을 추측하지 않습니다.
# Fine W=90은 10/17과 10/18에 걸쳐 있습니다. Fine W=0은 10/19의 별도
# calibration을 사용하지만 최종 fine scan에는 다른 W들과 함께 들어갑니다.

# %% tags=["definitions"]
def job_files(raw_dir, day, numbers):
    return [Path(raw_dir) / f'JOB-{day}-{number:05d}_QsimExperiment.h5'
            for number in numbers]


def legacy_groups(raw_dir):
    return {
        'coarse': {
            'assembly': 'coarse',
            'calibration': job_files(raw_dir, '20261017', [272, 273, 274, 275]),
            'scans': {
                30: job_files(raw_dir, '20261017', [292, 293, 294, 295, 296, 297, 298, 299,
                                                    300, 301, 302, 303, 304, 305, 306]),
                60: job_files(raw_dir, '20261017', [307, 308, 309, 310, 311, 312, 313, 314,
                                                    315, 316, 317, 318, 319, 320, 321]),
                90: job_files(raw_dir, '20261017', [322, 323, 324, 325, 326, 327, 328, 329,
                                                    330, 331, 332, 333, 334, 335, 336]),
                120: job_files(raw_dir, '20261017', [337, 338, 339, 340, 341, 342, 343, 344,
                                                     345, 346, 347, 348, 349, 350, 351]),
                150: job_files(raw_dir, '20261017', [352, 353, 354, 355, 356, 357, 358, 359,
                                                     360, 361, 362, 363, 364, 365, 366]),
            },
        },
        'fine': {
            'assembly': 'fine',
            'calibration': job_files(raw_dir, '20261017', [369, 370, 371, 372]),
            'scans': {
                30: job_files(raw_dir, '20261017', [373, 374, 375, 376, 377, 378, 379, 380,
                                                    381, 382, 383, 384, 385, 386, 387]),
                40: job_files(raw_dir, '20261017', [388, 389, 390, 391, 392, 393, 394, 395,
                                                    396, 397, 398, 399, 400, 401, 402]),
                60: job_files(raw_dir, '20261017', [403, 404, 405, 406, 407, 408, 409, 410,
                                                    411, 412, 413, 414, 415, 416, 417]),
                90: (job_files(raw_dir, '20261017', [418, 419, 420])
                     + job_files(raw_dir, '20261018', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
                120: job_files(raw_dir, '20261018', [13, 14, 15, 16, 17, 18, 19, 20,
                                                     21, 22, 23, 24, 25, 26, 27]),
                150: job_files(raw_dir, '20261018', [28, 29, 30, 31, 32, 33, 34, 35,
                                                     36, 37, 38, 39, 40, 41, 42]),
            },
        },
        'fine_zero': {
            'assembly': 'fine',
            'calibration': job_files(raw_dir, '20261019', [9, 10, 11, 12]),
            'scans': {
                0: job_files(raw_dir, '20261019', [23, 24, 25, 26, 27, 28, 29, 30,
                                                   31, 32, 33, 34, 35, 36, 37]),
            },
        },
    }


# %% [markdown]
# ## 읽기, 배열 배치와 새 파일 저장
#
# 다음 함수들은 메모리에서 분석용 배열의 순서와 모양을 맞춥니다.
# hardware/pulse 설정은 바꾸지 않습니다. 새 cfg의 occupation/sweep 축은
# 합쳐진 배열의 구조를 나타내며, 각 원본 cfg는 archive에 그대로 남습니다.

# %% tags=["definitions"]
def read_legacy_job(fname):
    """Read a legacy job; normalize only in-memory point/shot array shapes."""
    fname = Path(fname).resolve()
    with h5py.File(fname, 'r') as saved:
        config = json.loads(saved.attrs['config'])
        if 'expt' not in config and isinstance(config.get('config'), dict):
            config = config['config']
        config = AttrDict(config)
        data = {key: item[()] for key, item in saved.items()
                if isinstance(item, h5py.Dataset)}
        attrs = dict(saved.attrs)
    if 'read_num' not in config:
        config.read_num = readout_lane_count(config)
    data['xpts'] = np.asarray(data['xpts']).reshape(-1)
    for key in ['idata', 'qdata']:
        if np.ndim(data[key]) == 1:
            data[key] = data[key][None, :]
        if np.ndim(data[key]) != 2 or len(data[key]) != len(data['xpts']):
            raise ValueError(f'{fname}: {key} must have one shot row per cycle')
    return {'fname': fname, 'config': config, 'data': data, 'attrs': attrs}


def readout_signature(config):
    """Classification settings shared by the rows of one analysis/calibration."""
    expt = config.expt
    qubit = expt.qubits[0]
    threshold = np.asarray(config.device.readout.threshold).reshape(-1)[qubit]
    return (config.read_num, qubit, float(threshold), saved_readout_mode(expt),
            expt.get('cond_sec_phase', 90), expt.get('phase_second_pulse', 180))


def check_readout_settings(jobs, calibration_config=None):
    reference = readout_signature(calibration_config or jobs[0]['config'])
    for job in jobs:
        if readout_signature(job['config']) != reference:
            raise ValueError(f"{job['fname']}: different readout/classification settings")


def stack_legacy_data(jobs, point_shape):
    """Stack compatible numeric data; preserve everything else in legacy_jobs."""
    data = {}
    common = set.intersection(*(set(job['data']) for job in jobs))
    excluded = {'xpts', 'ypts', 'idata', 'qdata', 'occupations'}
    for key in sorted(common - excluded):
        values = [np.asarray(job['data'][key]) for job in jobs]
        if all(value.dtype.kind in 'biufc' and value.shape == values[0].shape
               for value in values):
            stacked = np.stack(values)
            if values[0].size == len(jobs[0]['data']['xpts']):
                stacked = stacked.reshape(point_shape)
            data[key] = stacked
    for key in ['idata', 'qdata']:
        data[key] = np.concatenate([job['data'][key] for job in jobs], axis=0)
    return data


def write_converted(fname, experiment, jobs):
    """Create a new HDF5; copy complete source roots opened read-only."""
    fname = Path(fname).resolve()
    fname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fname, 'x') as saved:
        saved.attrs['config'] = json.dumps(experiment.cfg, cls=NpEncoder)
        saved.attrs['job_class'] = type(experiment).__name__
        saved.attrs['job_module'] = type(experiment).__module__
        saved.attrs['converted_from'] = json.dumps([str(job['fname']) for job in jobs])
        # Only carry a common recorded timing value; never infer physical time.
        timing = jobs[0]['attrs'].get('derived_params')
        if timing is not None and all(job['attrs'].get('derived_params') == timing for job in jobs):
            saved.attrs['derived_params'] = timing
        for key, value in experiment.data.items():
            saved.create_dataset(key, data=value)
        archive = saved.create_group('legacy_jobs')
        for index, job in enumerate(jobs):
            with h5py.File(job['fname'], 'r') as source:
                source.copy('/', archive, name=f'{index:04d}', expand_soft=True,
                            expand_external=True, expand_refs=True)
    experiment.fname = str(fname)
    return fname


# %% [markdown]
# ## Calibration과 AGP 변환
#
# Calibration의 행은 `occupations[0]`으로 결정합니다. AGP의 행은 저장된
# occupation을 현재 `generate_fock_strings()`의 순서에 맞춥니다.
# 입력 파일 순서가 새 클래스의 순서와 달라도 올바른 상태로 연결됩니다.

# %% tags=["definitions"]
def convert_calibration(fnames, output_fname):
    jobs = [read_legacy_job(fname) for fname in fnames]
    prepared = [job['config'].expt.occupations[0] for job in jobs]
    if sorted(prepared) != [0, 1, 2, 3]:
        raise ValueError('calibration needs exactly one job prepared in each of 0, 1, 2, 3')
    jobs = [job for _, job in sorted(zip(prepared, jobs), key=lambda pair: pair[0])]
    for job in jobs:
        if (not np.array_equal(job['data']['xpts'], [0])
                or any(job['config'].expt.occupations[1:])):
            raise ValueError(f"{job['fname']}: calibration must be zero-cycle M1 preparation")
    check_readout_settings(jobs)
    calibration = MultiParityConfusionMatrixExperiment.__new__(MultiParityConfusionMatrixExperiment)
    calibration.cfg = deepcopy(jobs[0]['config'])
    calibration.cfg.expt.pop('occupations', None)
    calibration.cfg.expt.swept_params = ['init_man_fock_state']
    calibration.cfg.expt.init_man_fock_states = ['0', '1', '2', '3']
    calibration.cfg.expt.multiparity_confusion_matrix = None
    calibration.data = stack_legacy_data(jobs, (4,))
    calibration.data['xpts'] = np.arange(4)
    calibration.data['occupations'] = np.asarray([job['config'].expt.occupations for job in jobs])
    calibration.analyze()
    return write_converted(output_fname, calibration, jobs)


def convert_agp(fnames, calibration_fname, output_fname, disorder_khz, direction=7):
    jobs = [read_legacy_job(fname) for fname in fnames]
    calibration = MultiParityConfusionMatrixExperiment.from_h5file(calibration_fname)
    check_readout_settings(jobs, calibration.cfg)
    agp = CentralBosonAGPExperiment.__new__(CentralBosonAGPExperiment)
    agp.cfg = deepcopy(jobs[0]['config'])
    agp.cfg.expt.photon_sector_number = sum(agp.cfg.expt.occupations)
    agp.cfg.expt.direction = direction
    expected = agp.generate_fock_strings()
    by_occupation = {tuple(job['config'].expt.occupations): job for job in jobs}
    if len(by_occupation) != len(jobs) or set(by_occupation) != set(map(tuple, expected)):
        raise ValueError('AGP inputs must cover each selected occupation exactly once')
    jobs = [by_occupation[tuple(occupation)] for occupation in expected]
    cycles = jobs[0]['data']['xpts']
    for job in jobs:
        if not np.array_equal(job['data']['xpts'], cycles):
            raise ValueError('occupation jobs within one AGP must share their cycle axis')
        for key in ['swap_stors', 'ro_stor', 'detunings']:
            if job['config'].expt[key] != agp.cfg.expt[key]:
                raise ValueError(f"{job['fname']}: different {key} within one AGP")
    agp.cfg.expt.pop('occupations', None)
    agp.cfg.expt.swept_params = ['occupations', 'floquet_cycle']
    agp.cfg.expt.floquet_cycles = cycles.tolist()
    agp.cfg.expt.disorder_khz = float(disorder_khz)
    agp.cfg.expt.multiparity_confusion_matrix = str(Path(calibration_fname).resolve())
    agp.data = stack_legacy_data(jobs, (len(jobs), len(cycles)))
    agp.data.update(occupations=np.asarray(expected), xpts=cycles.copy(), ypts=np.arange(len(jobs)))
    agp.analyze()
    return write_converted(output_fname, agp, jobs)


# %% [markdown]
# ## 원본 보존과 분석 결과 검증
#
# 원본을 하나씩 분류한 확률과 변환 후 확률을 비교합니다. 다시 불러와 분석한
# 결과도 같아야 하며, 모든 원본 dataset/group/attribute가 archive에 있어야 합니다.
# 과거 n_err에 calibration 통계 오차가 포함되어 있다면 현재의 shot-only n_err와
# 같을 필요는 없습니다. 과거 값도 archive에 보존됩니다.

# %% tags=["definitions"]
def assert_same_hdf5(source, copied):
    """Compare complete groups, dataset values/properties and all attributes."""
    assert set(source.attrs) == set(copied.attrs), source.name
    for key in source.attrs:
        np.testing.assert_array_equal(source.attrs[key], copied.attrs[key])
    if isinstance(source, h5py.Dataset):
        assert source.shape == copied.shape and source.dtype == copied.dtype, source.name
        assert source.compression == copied.compression and source.shuffle == copied.shuffle
        np.testing.assert_array_equal(source[()], copied[()])
    else:
        assert set(source) == set(copied), source.name
        for key in source:
            assert_same_hdf5(source[key], copied[key])


def verify_converted(fname):
    classes = {cls.__name__: cls for cls in
               [MultiParityConfusionMatrixExperiment, CentralBosonAGPExperiment]}
    with h5py.File(fname, 'r') as saved:
        cls = classes[saved.attrs['job_class']]
        expected = {key: item[()] for key, item in saved.items()
                    if isinstance(item, h5py.Dataset)}
        originals = json.loads(saved.attrs['converted_from'])
        for index, source_fname in enumerate(originals):
            with h5py.File(source_fname, 'r') as source:
                assert_same_hdf5(source['/'], saved[f'legacy_jobs/{index:04d}'])

    # Classify each original job separately to check the state/point ordering.
    for index, source_fname in enumerate(originals):
        job = read_legacy_job(source_fname)
        raw = MultiparityQsimExperiment.__new__(MultiparityQsimExperiment)
        raw.cfg = job['config']
        raw.cfg.expt.multiparity_confusion_matrix = None
        raw.data = {key: job['data'][key] for key in ['xpts', 'idata', 'qdata']}
        raw.analyze()
        probability = np.stack([raw.data[f'p_mod{n}'] for n in range(4)], axis=-1)
        for n in range(4):
            original_probability = probability[0, n] if cls is MultiParityConfusionMatrixExperiment else probability[:, n]
            np.testing.assert_array_equal(expected[f'p_mod{n}'][index], original_probability)
        if cls is MultiParityConfusionMatrixExperiment:
            np.testing.assert_array_equal(expected['multiparity_confusion_matrix'][index], probability[0])
        else:
            corrected = np.linalg.solve(expected['confusion_matrix'].T, probability.T).T
            for n in range(4):
                np.testing.assert_allclose(expected[f'p_mod{n}_corrected'][index], corrected[:, n],
                                           rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(expected['nmod4_mean_corrected'][index], corrected @ np.arange(4),
                                       rtol=1e-13, atol=1e-13)
    reloaded = cls.from_h5file(fname)
    reloaded.analyze()
    for key, value in expected.items():
        if np.asarray(value).dtype.kind in 'fc':
            np.testing.assert_allclose(reloaded.data[key], value, rtol=1e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(reloaded.data[key], value)
    return {'file': str(fname), 'legacy_jobs_preserved': len(originals), 'reanalysis': 'same'}


def file_sha256(fname):
    with open(fname, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def inspect_inputs(groups):
    rows = []
    for name, group in groups.items():
        entries = [('calibration', None, group['calibration'])]
        entries += [('AGP', disorder, files) for disorder, files in group['scans'].items()]
        for kind, disorder, files in entries:
            for fname in files:
                job = read_legacy_job(fname)
                cycles = job['data']['xpts']
                rows.append({'group': name, 'kind': kind, 'W_khz': disorder,
                             'occupation': job['config'].expt.occupations,
                             'ro_stor': job['config'].expt.ro_stor,
                             'cycle_count': len(cycles), 'cycle_first': cycles[0],
                             'cycle_last': cycles[-1], 'read_num': job['config'].read_num,
                             'threshold': readout_signature(job['config'])[2],
                             'file': str(job['fname'])})
    return pd.DataFrame(rows)


# %% [markdown]
# ## 실행 설정
#
# 여기부터는 pippin의 원본 파일이 필요합니다. 실행마다 새 출력 폴더를
# 사용합니다. 같은 이름의 파일이 있으면 덮어쓰지 않고 중단합니다.
# 기존 data 디렉터리는 입력으로만 사용합니다.

# %% tags=["settings"]
EXPERIMENT_ROOT = Path(r'C:\experiments\agp_k44_first_scan')
RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
CONVERTED_DIR = EXPERIMENT_ROOT / 'converted_data' / f'{RUN_TAG}_agp_legacy'
ASSEMBLED_DIR = EXPERIMENT_ROOT / 'assembled_data' / f'{RUN_TAG}_agp_legacy'
DIRECTION = 7
groups = legacy_groups(EXPERIMENT_ROOT / 'data')

# %% tags=["inspect"]
inventory = inspect_inputs(groups)
display(inventory)
source_hashes = {fname: file_sha256(fname) for fname in inventory['file']}
print(f'{len(source_hashes)} original files; read-only inputs')
print('Converted output:', CONVERTED_DIR)
print('Assembled output:', ASSEMBLED_DIR)

# %% tags=["convert_calibrations"]
converted_calibrations = {}
for name, group in groups.items():
    converted_calibrations[name] = convert_calibration(
        group['calibration'], CONVERTED_DIR / f'confusion_{name}.h5')
    print(name, converted_calibrations[name])

# %% tags=["convert_agp"]
converted_agp = {'coarse': {}, 'fine': {}}
for name, group in groups.items():
    for disorder, files in group['scans'].items():
        output = CONVERTED_DIR / f"agp_{group['assembly']}_W{disorder:03d}_S{DIRECTION}.h5"
        converted_agp[group['assembly']][disorder] = convert_agp(
            files, converted_calibrations[name], output, disorder, direction=DIRECTION)
        print(group['assembly'], disorder, output)

# %% tags=["verify_converted"]
converted_files = list(converted_calibrations.values())
converted_files += [fname for scans in converted_agp.values() for fname in scans.values()]
conversion_checks = [verify_converted(fname) for fname in converted_files]
display(pd.DataFrame(conversion_checks))

# %% [markdown]
# ## Assembled data와 color map
#
# Disorder 사이의 cycle 축 일치 검사는 CentralBosonAGPDisorderScan이 담당합니다.
# Fine W=0의 calibration은 별도로 유지합니다. 각 새 AGP의 전체 내용이
# assembled HDF5의 children에 들어가며, 그 안의 legacy_jobs도 보존됩니다.

# %% tags=["assemble"]
assembled_files = {}
for name, scans in converted_agp.items():
    scan = CentralBosonAGPDisorderScan(list(scans.values()), notes=f'Legacy AGP conversion: {name}')
    scan.analyze()
    assembled_files[name] = scan.save_data(ASSEMBLED_DIR / f'{name}.h5')
    restored = CentralBosonAGPDisorderScan.from_h5file(assembled_files[name])
    for key, value in scan.data.items():
        np.testing.assert_array_equal(restored.data[key], value)
    print(name, restored.data['disorder_khz'], assembled_files[name])

# %% tags=["verify_originals"]
for fname, before in source_hashes.items():
    assert file_sha256(fname) == before, f'Original file changed: {fname}'
print(f'{len(source_hashes)} original files unchanged; 3 calibrations + 12 AGP + 2 assembled files')

# %% tags=["display"]
for name, fname in assembled_files.items():
    scan = CentralBosonAGPDisorderScan.from_h5file(fname)
    for moment in (1, 2):
        for raw in (False, True):
            fig, ax = scan.display(moment=moment, raw=raw)
            ax.set_title(f'{name}: {ax.get_title()}')
            plt.show()
            plt.close(fig)

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
# # Legacy AGP conversion: read-only source files
#
# Open this file as a Jupytext notebook in the repo environment on pippin
# and run the cells from top to bottom. No SSH, station, runner, worker,
# or measurement hardware is used.
#
# 1. Read the listed legacy files, inspect the inventory, and hash the sources.
# 2. Convert four prepared-state JOBs into one new confusion-matrix file.
# 3. Convert 15 occupation JOBs per disorder into one CentralBosonAGPExperiment.
# 4. Assemble coarse and fine scans separately and display their color maps.
#
# Source HDF5 files are always opened with `r`; output files use `x` and never
# overwrite existing files. Combined shots and analysis results are stored at
# the root. Original filenames are recorded in the `converted_from` attribute;
# original JOB contents are not copied into a separate archive. Each scan uses
# its associated converted calibration. Assembled files keep shots only in children.
# No AGP integral, T1 correction, matrix-pencil fitting, or calibration
# statistical error propagation is added.

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
# ## File associations from the supplied catalog
#
# Dates and JOB numbers are listed explicitly from the supplied catalog.
# Prepared states, occupations, and calibrations are not inferred from
# directory searches or file order. Fine W=90 spans Oct17 and Oct18.
# Fine W=0 uses a separate Oct19 calibration and joins the other fine scans
# in the final assembled file.

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
# ## Reading, arranging arrays, and writing new files
#
# These functions reorder and reshape analysis arrays in memory without
# changing hardware or pulse settings. The new cfg's occupation and sweep
# axes describe the combined layout. Original per-JOB configs and other legacy
# contents remain in the source files listed in `converted_from`.

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
    """Combine shots and compatible numeric data from the source jobs."""
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
    """Create a new HDF5 with combined data and the original filenames."""
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
    experiment.fname = str(fname)
    return fname


# %% [markdown]
# ## Calibration and AGP conversion
#
# Calibration rows are identified by `occupations[0]`. AGP rows are matched
# by their saved occupations to the current `generate_fock_strings()` order.
# Input file order may differ from the new class's state order.

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
# ## Verifying source preservation and analysis results
#
# Compare combined shots and probabilities with the individual source files.
# Reloading and reanalysis must give the same results. A legacy n_err that includes
# calibration uncertainty need not match the current shot-only n_err.
# Legacy analysis values remain in the original files.

# %% tags=["definitions"]
def verify_converted(fname):
    classes = {cls.__name__: cls for cls in
               [MultiParityConfusionMatrixExperiment, CentralBosonAGPExperiment]}
    with h5py.File(fname, 'r') as saved:
        cls = classes[saved.attrs['job_class']]
        expected = {key: item[()] for key, item in saved.items()
                    if isinstance(item, h5py.Dataset)}
        originals = json.loads(saved.attrs['converted_from'])

    # Classify each original job separately to check the state/point ordering.
    for index, source_fname in enumerate(originals):
        job = read_legacy_job(source_fname)
        point_count = len(job['data']['xpts'])
        start = index * point_count
        for key in ['idata', 'qdata']:
            np.testing.assert_array_equal(expected[key][start:start + point_count], job['data'][key])
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
    return {'file': str(fname), 'source_files': len(originals), 'shots': 'same', 'reanalysis': 'same'}


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
# ## Run settings
#
# The remaining cells require the original files on pippin. Each run uses new
# output directories. An existing output filename stops conversion instead of
# being overwritten. The original data directory is used only as an input.

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
# ## Assembled data and color maps
#
# CentralBosonAGPDisorderScan checks that cycle axes agree across disorder values.
# Fine W=0 retains its separate calibration. Each converted AGP is copied in full
# into the assembled HDF5's children group. Shots are stored only in those children;
# the assembled root contains axes and results without duplicate idata/qdata.

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

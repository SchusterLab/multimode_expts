"""Reprocess saved disorder spectroscopy from HDF5 only.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
256-276 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr_disorder.py`.

This section reads nothing from the job queue. Cell 261 is why it is worth its
own module: it is a 248-line **notebook-local reimplementation of the HDF5
loading layer** -- `SavedSpectroscopyExperiment`, `read_h5_header`, `load_h5`,
`read_shots`, `saved_parameters`, `saved_floquet_timing`, `common_grid_jobs`
and `make_plain`. That is a parallel loader to the library's own
`EncodingHamiltonianSpectroscopyExperiment.from_h5file`, and to
`experiments/floquet_timing.resolve_floquet_timing`.

TODO: those loading paths are still not reconciled here, but the decision has
been made and half-executed elsewhere. `experiments/saved_jobs.py` is now the
library's HDF5 loader, and it does what this module's copy does -- reads the
file, resolves the Floquet timing from provenance, never touches a station --
for every MBR stage. What remains here that `saved_jobs` does not do is the
part that motivated the copy: reading headers without loading shot arrays, and
selecting a rectangular common grid across jobs. Folding those two into
`saved_jobs` and deleting the rest is the remaining step.

The settings prefixes (`diag_*`, `mpm_*`, `spectroscopy_*`) are arguments
rather than notebook globals. The dataset manifest in cell 258 stayed in the
notebook: which jobs belong to which realization is a dataset choice.

Temporary home, per the stage-2 instructions.
"""

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.legacy_mbr import MBRSpectrumExperiment


# Local HDF5 only: no JobClient, job database, pickle, Station, or pulse compilation.
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from slab import AttrDict
from experiments.qsim.floquet_dark_mode_readout import EncodingHamiltonianSpectroscopyExperiment


def make_plain(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): make_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parameter_key(value):
    return json.dumps(make_plain(value), sort_keys=True, allow_nan=False)


def saved_parameters(expt):
    # These parameters really are in H5. Pulse-CSV values and RFSoC clocks
    # were NOT included in older H5 files, and cannot be inferred from job names.
    ecfg = expt.cfg.expt
    modes = list(map(int, ecfg.swap_stors))
    man = int(ecfg.get('man_mode_no', 1))
    kerr = np.asarray(expt.cfg.device.manipulate.kerr).reshape(-1)
    return dict(photon_number=sum(ecfg.spectroscopy_occupations),
                swap_stors=modes,
                physical_kerr_MHz=-abs(float(kerr[min(man - 1, len(kerr) - 1)])),
                man_mode_no=man,
                floquet_waveform=ecfg.get('floquet_waveform'),
                scramble_sync_cycles=int(ecfg.get('scramble_sync_cycles', 10)),
                floquet_gauss_sigma=ecfg.get('floquet_gauss_sigma'),
                use_multiphoton_swap=bool(ecfg.get('use_multiphoton_swap', False)))


def saved_floquet_timing(expt):
    # `source` explicitly distinguishes H5 metadata from a historical notebook
    # record supplied in the settings. No claim of recovering absent instructions.
    hardware = getattr(expt, 'saved_hardware', None)
    if not hardware or hardware.get('floquet_cycle_us') is None:
        return None
    cycles = expt.cfg.expt.get('offdiag_cycles', expt.cfg.expt.get('floquet_cycles', []))
    last = max(cycles) if len(cycles) else None
    cycle_us = float(hardware['floquet_cycle_us'])
    return dict(cycle_us=cycle_us, last_cycle=last,
                last_time_us=None if last is None else last * cycle_us,
                source=hardware['source'])


class SavedSpectroscopyExperiment(MBRSpectrumExperiment):
    """Notebook-only analysis adapter; H5 objects never initialize instruments."""

    @classmethod
    def _saved_parameters(cls, expts, station=None):
        first = expts[0]
        parameters = saved_parameters(first)
        metadata = getattr(first, 'saved_hardware', None)
        if not metadata or metadata.get('floquet_cycle_us') is None or metadata.get('couplings_MHz') is None:
            raise ValueError('This H5 has no archived cycle clock/couplings. Supply their historical values '
                             'in the dataset hardware settings; raw cycle-domain data are already loaded.')
        cycle_us = float(metadata['floquet_cycle_us'])
        coupling = np.asarray(metadata['couplings_MHz'], dtype=float)
        modes = parameters['swap_stors']
        if (cycle_us <= 0 or not np.isfinite(cycle_us) or coupling.shape != (len(modes),)
                or not np.isfinite(coupling).all() or np.any(coupling <= 0)):
            raise ValueError('Archived cycle time/couplings do not match the selected modes')
        detuning = cls._saved_detunings(first.cfg.expt, len(modes))
        for child in expts[1:]:
            if saved_parameters(child) != parameters:
                raise ValueError('Selected H5 jobs have different physical settings')
            if not np.array_equal(cls._saved_detunings(child.cfg.expt, len(modes)), detuning):
                raise ValueError('Selected H5 jobs have different disorder detunings')
            other = getattr(child, 'saved_hardware', None)
            if (not other or other.get('floquet_cycle_us') != cycle_us
                    or not np.array_equal(other.get('couplings_MHz'), coupling)):
                raise ValueError('Selected H5 jobs have different archived clocks/couplings')
        hardware = AttrDict(dict(floquet_cycle_us=cycle_us, couplings_MHz=coupling,
                                physical_kerr_MHz=parameters['physical_kerr_MHz'],
                                decoder_cycle_us=float(metadata.get('decoder_cycle_us', cycle_us)),
                                detunings_MHz=detuning.copy(), source=metadata['source']))
        return AttrDict(dict(swap_stors=modes, detunings=detuning,
                             mode_labels=['M1'] + [f'S{mode}' for mode in modes], hardware=hardware))


    @classmethod
    def _postprocess_reconstruction(cls,
                                    reconstruction,
                                    saved_correction,
                                    calibration,
                                    hardware,
                                    phase_frame,
                                    manual_kerr_MHz,
                                    cycle_branches,
                                    legacy):

        result = super()._postprocess_reconstruction(reconstruction,
                                                     saved_correction,
                                                     calibration,
                                                     hardware,
                                                     phase_frame,
                                                     manual_kerr_MHz,
                                                     cycle_branches,
                                                     legacy)

        actual_us = float(hardware.floquet_cycle_us)
        decoder_us = float(hardware.get('decoder_cycle_us', actual_us))

        if np.isclose(decoder_us, actual_us, rtol=1e-9, atol=1e-12):
            return result
        if result.phase_frame != 'as_acquired':
            raise ValueError('Different archived decoder/physical clocks are validated here only for '
                             "phase_frame='as_acquired', manual_kerr_MHz=None")
        if saved_correction.application_sign != -1 or saved_correction.modes != {'final_analyzer'}:
            raise ValueError('The archived decoder-clock correction requires the recorded -1 final-analyzer convention')

        # DDS detuning acts for actual_us; the decoder advanced its reference phase
        # for decoder_us. Remove only their known difference, independently of peaks/H.
        corrected = result.reconstruction  # base method already copied the raw A.
        final = np.asarray(corrected.final_occupations)
        detunings = np.asarray(hardware.detunings_MHz)
        n_man = final[:, 0]
        reference_kerr = float(hardware.physical_kerr_MHz) * n_man * (n_man - 1) / 2
        # The saved gamma also included K*choose(n_M1,2)*T_decoder.
        # Use the same independently saved K, not a value fitted to peaks.
        turns_per_cycle = (decoder_us - actual_us) * (final[:, 1:] @ detunings - reference_kerr)
        rotation = np.exp(-2j * np.pi * turns_per_cycle[:, None] * corrected.cycles[None, :])
        corrected.A *= rotation
        corrected.A_norm = np.asarray([
            row / row[0] if tuple(initial) == tuple(final_state) else row
            for row, initial, final_state in zip(corrected.A, corrected.occupations, corrected.final_occupations)
        ])
        result.decoder_clock_correction = AttrDict(dict(
            decoder_cycle_us=decoder_us, physical_cycle_us=actual_us,
            turns_per_cycle=turns_per_cycle, source=hardware.source))
        return result


def read_h5_header(path):
    # Header + dimensions only. Shot arrays are read once, after campaign selection.
    path = Path(path)
    with h5py.File(path, 'r') as handle:
        cfg = AttrDict(json.loads(handle.attrs['config']))
        ecfg = cfg.expt
        if 'n_cycle_pairs' in ecfg and 'd72_realization' not in ecfg:
            kind = 'calibration'
            outer = len(ecfg.get('n_physical_cycles', ecfg.n_cycle_pairs))
        elif 'd72_realization' in ecfg and ('offdiag_cycles' in ecfg or 'floquet_cycles' in ecfg):
            kind = 'spectroscopy'
            outer = 2 * len(ecfg.offdiag_cycles) if 'offdiag_cycles' in ecfg else len(ecfg.floquet_cycles)
        else:
            raise ValueError('not an N-photon phase calibration or disorder spectroscopy H5')
        inner = len(ecfg.get('spectroscopy_prep_phases', [0., 180.]))
        for name in ('avgi', 'avgq'):
            if name not in handle or handle[name].shape != (outer, inner):
                raise ValueError(f'incomplete {name}: expected {(outer, inner)}')
            if not np.isfinite(handle[name][...]).all():
                raise ValueError(f'non-finite values in {name}')
        for name in ('idata', 'qdata'):
            if name not in handle or handle[name].shape[0] != outer * inner:
                raise ValueError(f'incomplete {name}: expected {outer * inner} sweep points')
        hardware = None
        # Support an explicit snapshot if future files contain one. Legacy files
        # checked on Sep-07 contain neither this attribute nor compiled programs.
        if 'spectroscopy_hardware' in handle.attrs:
            hardware = json.loads(handle.attrs['spectroscopy_hardware'])
            hardware['source'] = f'H5 spectroscopy_hardware: {path.name}'
    return SimpleNamespace(cfg=cfg, fname=str(path), kind=kind,
                           job_id=path.name[:18], saved_hardware=hardware)


def load_h5(header, hardware=None, cache=None, load_shots=False):
    path = Path(header.fname)
    fingerprint = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns,
                   parameter_key(hardware), bool(load_shots))
    if cache is not None and fingerprint in cache:
        return cache[fingerprint]
    # Same __new__ + config/data reconstruction as Experiment.from_h5file,
    # but do not load gigabytes of single shots just to plot averaged spectra.
    expt = SavedSpectroscopyExperiment.__new__(SavedSpectroscopyExperiment)
    expt.fname = str(path)
    with h5py.File(path, 'r') as handle:
        expt.cfg = AttrDict(json.loads(handle.attrs['config']))
        expt.data = AttrDict({name: np.asarray(value) for name, value in handle.items()
                             if load_shots or name not in ('idata', 'qdata')})
        expt.data['attrs'] = dict(handle.attrs)
    expt.path, expt.config_file, expt.prefix = str(path.parent), str(path), path.stem
    expt.job_id = header.job_id
    expt.saved_hardware = header.saved_hardware or hardware
    if cache is not None:
        cache[fingerprint] = expt
    return expt


def read_shots(expt):
    # Only shot-noise / split-half analyses need these two large arrays.
    with h5py.File(expt.fname, 'r') as handle:
        for name in ('idata', 'qdata'):
            if name not in expt.data:
                expt.data[name] = np.asarray(handle[name])


def common_grid_jobs(headers):
    # Preserve every saved job in the manifest. Only the rectangular analysis
    # aggregate excludes orphan/short rows; raw preview can still show those jobs.
    grouped = {}
    for header in headers:
        cfg = header.cfg.expt
        initial = tuple(cfg.spectroscopy_occupations)
        final = tuple(cfg.get('offdiag_decoder_occupation', cfg.get('spectroscopy_final_occupations', initial)))
        grouped.setdefault((final, initial), []).append(header)
    valid, excluded = {}, []
    for pair, chunks in grouped.items():
        grids = {}
        for header in chunks:
            cfg = header.cfg.expt
            phases = (0., 90.) if 'offdiag_cycles' in cfg else (float(cfg.spectroscopy_analyzer_phase),)
            for phase in phases:
                grids.setdefault(phase, []).extend(cfg.get('offdiag_cycles', cfg.get('floquet_cycles', [])))
        first = tuple(sorted(grids.get(0., [])))
        second = tuple(sorted(grids.get(90., [])))
        regular = len(first) < 3 or len(set(np.diff(first))) == 1
        if not first or first != second or len(set(first)) != len(first) or first[0] != 0 or not regular:
            excluded.extend(dict(job_id=item.job_id, reason='orphan, duplicate, or unequal analyzer/cycle coverage') for item in chunks)
        else:
            valid[pair] = (first, chunks)
    if not valid:
        return [], excluded, []
    # Longest available contiguous grid starting at zero, selected without looking
    # at amplitudes, fitted peaks, or theory. Do not silently trim long rows.
    grid = max((value[0] for value in valid.values()), key=len)
    kept = []
    for row_grid, chunks in valid.values():
        if row_grid == grid:
            kept.extend(item.job_id for item in chunks)
        else:
            excluded.extend(dict(job_id=item.job_id, reason='short row: differs from longest saved cycle grid') for item in chunks)
    return sorted(kept), excluded, list(grid)


def scan_completed_spectroscopy(data_directory, selected_ids):
    """Read only the requested HDF5 files; never discover extra jobs."""
    directory = Path(data_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'H5 data directory does not exist: {directory}')
    headers = {}
    manifest = dict(data_directory=str(directory.resolve()), selected_ids=list(selected_ids),
                    missing_h5=[], excluded=[])
    for job_id in selected_ids:
        path = directory / f'{job_id}_EncodingHamiltonianSpectroscopyExperiment.h5'
        if not path.is_file():
            manifest['missing_h5'].append(job_id)
            continue
        try:
            headers[job_id] = read_h5_header(path)
        except (OSError, KeyError, ValueError, TypeError) as error:
            manifest['excluded'].append(dict(job_id=job_id, reason=str(error)))
    return manifest, headers


def classify_selected_jobs(headers, calibration_job_ids, job_ids_by_realization):
    """Keep the user's display groups; verify their saved settings and cycle coverage."""
    selection = dict(parameters=None,
                     calibration_job_ids=[],
                     job_ids_by_realization={},
                     analysis_job_ids_by_realization={},
                     realizations={},
                     issues=[])
    for display_key, requested_jobs in job_ids_by_realization.items():
        jobs = [job for job in requested_jobs if job in headers]
        if not jobs:
            selection['issues'].append(f'r={display_key}: no requested spectroscopy H5 files available')
            continue
        group = [headers[job] for job in jobs]
        parameters = [saved_parameters(header) for header in group]
        identities = [dict(saved_realization=int(header.cfg.expt.d72_realization),
                           seed=header.cfg.expt.get('d72_seed'),
                           detunings_MHz=make_plain(header.cfg.expt.get('detunings', []))) for header in group]
        if (any(header.kind != 'spectroscopy' for header in group)
                or any(value != parameters[0] for value in parameters[1:])
                or any(value != identities[0] for value in identities[1:])):
            raise ValueError(f'r={display_key} combines different saved realizations/settings')
        if selection['parameters'] is None:
            selection['parameters'] = parameters[0]
        elif selection['parameters'] != parameters[0]:
            raise ValueError('Groups have different physical settings; select their matching calibrations separately')
        kept, excluded, cycles = common_grid_jobs(group)
        cfg = group[0].cfg.expt
        available_pairs = {int(header.cfg.expt.offdiag_pair_index) for header in group
                           if 'offdiag_pair_index' in header.cfg.expt}
        missing_pairs = sorted(set(range(len(cfg.get('d72_selected_pairs', [])))) - available_pairs)
        unavailable_jobs = [job for job in requested_jobs if job not in headers]
        key = str(display_key)
        selection['job_ids_by_realization'][key] = jobs
        selection['analysis_job_ids_by_realization'][key] = kept
        selection['realizations'][key] = dict(
            identity=identities[0], display_realization=int(display_key), analysis_cycles=cycles,
            excluded_from_rectangular_analysis=excluded, missing_pair_indices=missing_pairs,
            missing_job_ids=unavailable_jobs, incomplete=bool(excluded or missing_pairs or unavailable_jobs))
    if not selection['job_ids_by_realization']:
        raise ValueError('No usable spectroscopy H5 files in selected_ids; inspect manifest for missing/rejected files')
    for job in calibration_job_ids:
        if job not in headers:
            continue
        if headers[job].kind != 'calibration' or saved_parameters(headers[job]) != selection['parameters']:
            raise ValueError(f'Calibration {job} does not match the selected physical settings')
        selection['calibration_job_ids'].append(job)
    return selection


def wrap_frequency(frequencies_MHz, sampling_frequency_MHz):
    frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
    return (
        frequencies_MHz + 0.5 * sampling_frequency_MHz
    ) % sampling_frequency_MHz - 0.5 * sampling_frequency_MHz


def match_levels(measured_MHz, theory_MHz, sampling_frequency_MHz, tolerance_MHz):
    measured_MHz = np.sort(wrap_frequency(measured_MHz, sampling_frequency_MHz))
    theory_MHz = np.sort(wrap_frequency(theory_MHz, sampling_frequency_MHz))
    distances_MHz = np.abs(measured_MHz[:, None] - theory_MHz[None, :])
    n_measured = len(measured_MHz)
    n_theory = len(theory_MHz)

    # Sequence alignment on the ordered 1D spectra. The primary objective is
    # the number of within-tolerance pairs; total frequency error is secondary.
    best_count = np.zeros((n_measured + 1, n_theory + 1), dtype=int)
    best_cost = np.zeros((n_measured + 1, n_theory + 1), dtype=float)
    choice = np.full((n_measured + 1, n_theory + 1), '', dtype=object)

    for measured_index in range(n_measured, -1, -1):
        for theory_index in range(n_theory, -1, -1):
            if measured_index == n_measured and theory_index == n_theory:
                continue
            options = []
            if measured_index < n_measured:
                options.append(
                    (
                        best_count[measured_index + 1, theory_index],
                        best_cost[measured_index + 1, theory_index],
                        1,
                        'skip_measured',
                    )
                )
            if theory_index < n_theory:
                options.append(
                    (
                        best_count[measured_index, theory_index + 1],
                        best_cost[measured_index, theory_index + 1],
                        2,
                        'skip_theory',
                    )
                )
            if (
                measured_index < n_measured
                and theory_index < n_theory
                and distances_MHz[measured_index, theory_index] <= tolerance_MHz
            ):
                options.append(
                    (
                        1 + best_count[measured_index + 1, theory_index + 1],
                        distances_MHz[measured_index, theory_index]
                        + best_cost[measured_index + 1, theory_index + 1],
                        0,
                        'match',
                    )
                )
            selected = min(options, key=lambda option: (-option[0], option[1], option[2]))
            best_count[measured_index, theory_index] = selected[0]
            best_cost[measured_index, theory_index] = selected[1]
            choice[measured_index, theory_index] = selected[3]

    pairs = []
    measured_index = 0
    theory_index = 0
    while measured_index < n_measured or theory_index < n_theory:
        action = choice[measured_index, theory_index]
        if action == 'match':
            pairs.append((measured_index, theory_index))
            measured_index += 1
            theory_index += 1
        elif action == 'skip_measured':
            measured_index += 1
        elif action == 'skip_theory':
            theory_index += 1
        else:
            raise RuntimeError('ordered level matching did not terminate')

    matched_measured = np.asarray([pair[0] for pair in pairs], dtype=int)
    matched_theory = np.asarray([pair[1] for pair in pairs], dtype=int)
    spurious = np.setdiff1d(np.arange(n_measured), matched_measured)
    missing = np.setdiff1d(np.arange(n_theory), matched_theory)
    errors_MHz = np.asarray(
        [measured_MHz[measured_index] - theory_MHz[theory_index] for measured_index, theory_index in pairs]
    )

    # For equal counts, sorted-to-sorted pairing is the 1D distribution
    # distance. It is a global error metric, not a claim of level identity.
    if n_measured == n_theory:
        ordered_errors_MHz = measured_MHz - theory_MHz
        ordered_mae_MHz = float(np.mean(np.abs(ordered_errors_MHz)))
        ordered_rmse_MHz = float(np.sqrt(np.mean(ordered_errors_MHz**2)))
        ordered_max_error_MHz = float(np.max(np.abs(ordered_errors_MHz)))
    else:
        ordered_errors_MHz = np.asarray([], dtype=float)
        ordered_mae_MHz = np.nan
        ordered_rmse_MHz = np.nan
        ordered_max_error_MHz = np.nan

    return AttrDict(
        dict(
            measured_MHz=measured_MHz,
            theory_MHz=theory_MHz,
            distances_MHz=distances_MHz,
            pairs=pairs,
            matched_measured=matched_measured,
            matched_theory=matched_theory,
            spurious=spurious,
            missing=missing,
            errors_MHz=errors_MHz,
            tolerance_MHz=float(tolerance_MHz),
            matched_mae_MHz=(float(np.mean(np.abs(errors_MHz))) if len(errors_MHz) else np.nan),
            ordered_errors_MHz=ordered_errors_MHz,
            ordered_mae_MHz=ordered_mae_MHz,
            ordered_rmse_MHz=ordered_rmse_MHz,
            ordered_max_error_MHz=ordered_max_error_MHz,
        )
    )


def build_dataset_manifest(hardware_by_job_id=None):
    """The per-dataset job manifest (cell 258).

    Returns (dataset_dumps, hardware_by_job_id).
    """
    hardware_by_job_id = {} if hardware_by_job_id is None else hardware_by_job_id

    # 2. Dataset catalog and its archived timing. Job lists are preserved.
    dataset_dumps = {


        'Sep10 K3.6 g29.2' :{
            'calibration_job_ids': [f'JOB-20260909-{job:05d}' for job in range(152, 212)] + [f'JOB-20260910-{job:05d}' for job in range(1, 11)],
            'job_ids_by_realization': {
                0: [f'JOB-20260910-{job:05d}' for job in range(31, 73)] + [f'JOB-20260911-{job:05d}' for job in range(1, 29)],
            },
        },
        'Sep07 K52.3 g29.2': {
            # qsim_experiments_highkerr_untracked.ipynb: full N=3 phase calibration.
            'calibration_job_ids': [f'JOB-20260907-{job:05d}' for job in range(199, 269)],
            'job_ids_by_realization': {
                0: [f'JOB-20260907-{job:05d}' for job in range(304, 334)],
                1: [f'JOB-20260907-{job:05d}' for job in range(334, 364)],
                2: [f'JOB-20260907-{job:05d}' for job in range(364, 394)],
                3: [f'JOB-20260907-{job:05d}' for job in range(406, 416)] + [f'JOB-20260908-{job:05d}' for job in range(1, 21)],
                4: [f'JOB-20260908-{job:05d}' for job in range(21, 51)],
                5: [f'JOB-20260908-{job:05d}' for job in range(51, 81)],
            },
        },
        'Sep05 K3.6 g30': {
            'calibration_job_ids': [f'JOB-20260905-{job:05d}' for job in range(75, 145)],
            'job_ids_by_realization': {
                0: [f'JOB-20260905-{job:05d}' for job in range(145, 160)],
                1: [f'JOB-20260905-{job:05d}' for job in range(160, 175)],
            },
        },
        'Sep01 K44 g15': {
            'calibration_job_ids': [f'JOB-20260831-{job:05d}' for job in range(479, 549)],
            'job_ids_by_realization': {
                0: [
                    'JOB-20260901-00003', 'JOB-20260901-00004', 'JOB-20260901-00005',
                    'JOB-20260901-00006', 'JOB-20260901-00007', 'JOB-20260901-00008',
                    'JOB-20260901-00009', 'JOB-20260901-00010', 'JOB-20260901-00011',
                    'JOB-20260901-00012',
                ],
                1: [
                    'JOB-20260901-00013', 'JOB-20260901-00014', 'JOB-20260901-00015',
                    'JOB-20260901-00016', 'JOB-20260901-00017', 'JOB-20260901-00018',
                    'JOB-20260901-00019', 'JOB-20260901-00020', 'JOB-20260901-00021',
                    'JOB-20260901-00022',
                ],
                2: [
                    'JOB-20260901-00023', 'JOB-20260901-00024', 'JOB-20260901-00025',
                    'JOB-20260901-00026', 'JOB-20260901-00027', 'JOB-20260901-00028',
                    'JOB-20260901-00029', 'JOB-20260901-00030', 'JOB-20260901-00031',
                    'JOB-20260901-00032',
                ],
                3: [
                    'JOB-20260901-00033', 'JOB-20260901-00034', 'JOB-20260901-00035',
                    'JOB-20260901-00036', 'JOB-20260901-00037', 'JOB-20260901-00038',
                    'JOB-20260901-00039', 'JOB-20260901-00040', 'JOB-20260901-00041',
                    'JOB-20260901-00042',
                ],
                4: [
                    'JOB-20260901-00043', 'JOB-20260901-00044', 'JOB-20260901-00045',
                    'JOB-20260901-00046', 'JOB-20260901-00047', 'JOB-20260901-00048',
                    'JOB-20260901-00049', 'JOB-20260901-00050', 'JOB-20260901-00051',
                    'JOB-20260901-00052',
                ],
            },
        },
        'Sep02 K20 g15': {
            'calibration_job_ids': [f'JOB-20260902-{job:05d}' for job in range(1, 71)],
            'job_ids_by_realization': {
                0: [
                    'JOB-20260902-00071', 'JOB-20260902-00072', 'JOB-20260902-00073',
                    'JOB-20260902-00074', 'JOB-20260902-00075', 'JOB-20260902-00076',
                    'JOB-20260902-00077', 'JOB-20260902-00078', 'JOB-20260902-00079',
                    'JOB-20260902-00080',
                ],
                1: [
                    'JOB-20260902-00081', 'JOB-20260902-00082', 'JOB-20260902-00083',
                    'JOB-20260902-00084', 'JOB-20260902-00085', 'JOB-20260902-00086',
                    'JOB-20260902-00087', 'JOB-20260902-00088', 'JOB-20260902-00089',
                    'JOB-20260902-00090',
                ],
                2: [
                    'JOB-20260902-00091', 'JOB-20260902-00092', 'JOB-20260902-00093',
                    'JOB-20260902-00094', 'JOB-20260902-00096', 'JOB-20260902-00097',
                    'JOB-20260902-00099', 'JOB-20260902-00100', 'JOB-20260902-00102',
                    'JOB-20260902-00103',
                ],
                3: [
                    'JOB-20260902-00105', 'JOB-20260902-00106', 'JOB-20260902-00108',
                    'JOB-20260902-00109', 'JOB-20260902-00111', 'JOB-20260902-00112',
                    'JOB-20260902-00114', 'JOB-20260902-00115', 'JOB-20260902-00117',
                    'JOB-20260902-00118',
                ],
                4: [
                    'JOB-20260902-00120', 'JOB-20260902-00121', 'JOB-20260902-00123',
                    'JOB-20260902-00124', 'JOB-20260902-00126', 'JOB-20260902-00127',
                    'JOB-20260902-00129', 'JOB-20260902-00130', 'JOB-20260902-00132',
                    'JOB-20260902-00133',
                ],
            },
        },
        'Sep02-04 K3.6 g15': {
            'calibration_job_ids': [f'JOB-20260902-{job:05d}' for job in range(768, 838)],
            'job_ids_by_realization': {
                0: [
                    'JOB-20260902-00840', 'JOB-20260902-00841', 'JOB-20260903-00001',
                    'JOB-20260903-00002', 'JOB-20260903-00003', 'JOB-20260903-00004',
                    'JOB-20260903-00005', 'JOB-20260903-00006', 'JOB-20260903-00007',
                    'JOB-20260903-00008', 'JOB-20260903-00009', 'JOB-20260903-00010',
                    'JOB-20260903-00011', 'JOB-20260903-00012', 'JOB-20260903-00013',
                ],
                1: [
                    'JOB-20260903-00014', 'JOB-20260903-00015', 'JOB-20260903-00016',
                    'JOB-20260903-00017', 'JOB-20260903-00018', 'JOB-20260903-00019',
                    'JOB-20260903-00020', 'JOB-20260903-00021', 'JOB-20260903-00022',
                    'JOB-20260903-00023', 'JOB-20260903-00024', 'JOB-20260903-00025',
                    'JOB-20260903-00026', 'JOB-20260903-00027', 'JOB-20260903-00028',
                ],
                2: [
                    'JOB-20260903-00029', 'JOB-20260903-00030', 'JOB-20260903-00031',
                    'JOB-20260903-00032', 'JOB-20260903-00033', 'JOB-20260903-00034',
                    'JOB-20260903-00035', 'JOB-20260903-00036', 'JOB-20260903-00037',
                    'JOB-20260903-00038', 'JOB-20260903-00039', 'JOB-20260903-00040',
                    'JOB-20260903-00041', 'JOB-20260903-00042', 'JOB-20260903-00043',
                ],
                3: [
                    'JOB-20260903-00044', 'JOB-20260903-00045', 'JOB-20260903-00046',
                    'JOB-20260903-00047', 'JOB-20260903-00048', 'JOB-20260903-00049',
                    'JOB-20260903-00050', 'JOB-20260903-00051', 'JOB-20260903-00052',
                    'JOB-20260903-00053', 'JOB-20260903-00054', 'JOB-20260903-00055',
                    'JOB-20260903-00056', 'JOB-20260903-00057', 'JOB-20260903-00058',
                ],
                4: [
                    'JOB-20260903-00059', 'JOB-20260903-00060', 'JOB-20260903-00062',
                    'JOB-20260903-00063', 'JOB-20260903-00065', 'JOB-20260903-00066',
                    'JOB-20260903-00068', 'JOB-20260903-00069', 'JOB-20260903-00071',
                    'JOB-20260903-00072', 'JOB-20260903-00074', 'JOB-20260903-00075',
                    'JOB-20260903-00077', 'JOB-20260903-00078', 'JOB-20260903-00080',
                ],
                5: [
                    'JOB-20260903-00082', 'JOB-20260903-00083', 'JOB-20260903-00085',
                    'JOB-20260903-00086', 'JOB-20260903-00088', 'JOB-20260903-00089',
                    'JOB-20260903-00091', 'JOB-20260903-00092', 'JOB-20260903-00094',
                    'JOB-20260903-00095', 'JOB-20260903-00097', 'JOB-20260903-00098',
                    'JOB-20260903-00100', 'JOB-20260903-00101', 'JOB-20260903-00103',
                ],
                6: [
                    'JOB-20260903-00105', 'JOB-20260903-00106', 'JOB-20260903-00108',
                    'JOB-20260903-00109', 'JOB-20260903-00111', 'JOB-20260903-00112',
                    'JOB-20260903-00114', 'JOB-20260903-00115', 'JOB-20260903-00117',
                    'JOB-20260903-00118', 'JOB-20260903-00120', 'JOB-20260903-00121',
                    'JOB-20260903-00123', 'JOB-20260903-00124', 'JOB-20260903-00126',
                ],
                7: [
                    'JOB-20260903-00128', 'JOB-20260903-00129', 'JOB-20260903-00131',
                    'JOB-20260903-00132', 'JOB-20260903-00134', 'JOB-20260903-00135',
                    'JOB-20260903-00137', 'JOB-20260903-00138', 'JOB-20260903-00140',
                    'JOB-20260903-00141', 'JOB-20260903-00143', 'JOB-20260903-00144',
                    'JOB-20260903-00146', 'JOB-20260903-00147', 'JOB-20260903-00148',
                ],
                8: [
                    'JOB-20260903-00149', 'JOB-20260903-00150', 'JOB-20260903-00151',
                    'JOB-20260903-00152', 'JOB-20260903-00153', 'JOB-20260903-00154',
                    'JOB-20260903-00155', 'JOB-20260903-00156', 'JOB-20260903-00157',
                    'JOB-20260903-00158', 'JOB-20260903-00159', 'JOB-20260903-00160',
                    'JOB-20260903-00161', 'JOB-20260903-00162', 'JOB-20260903-00163',
                ],
                9: [
                    'JOB-20260903-00164', 'JOB-20260903-00165', 'JOB-20260903-00166',
                    'JOB-20260903-00167', 'JOB-20260903-00168', 'JOB-20260903-00169',
                    'JOB-20260903-00170', 'JOB-20260903-00171', 'JOB-20260903-00172',
                    'JOB-20260903-00173', 'JOB-20260903-00174', 'JOB-20260903-00175',
                    'JOB-20260903-00176', 'JOB-20260903-00177', 'JOB-20260903-00178',
                ],
                10: [
                    'JOB-20260903-00179', 'JOB-20260903-00180', 'JOB-20260903-00181',
                    'JOB-20260903-00182', 'JOB-20260903-00183', 'JOB-20260903-00184',
                    'JOB-20260903-00185', 'JOB-20260903-00186', 'JOB-20260903-00187',
                    'JOB-20260903-00188', 'JOB-20260903-00189', 'JOB-20260903-00190',
                    'JOB-20260903-00191', 'JOB-20260903-00192', 'JOB-20260903-00193',
                ],
                11: [
                    'JOB-20260903-00194', 'JOB-20260903-00195', 'JOB-20260903-00196',
                    'JOB-20260903-00197', 'JOB-20260903-00198', 'JOB-20260903-00199',
                    'JOB-20260903-00200', 'JOB-20260903-00201', 'JOB-20260903-00202',
                    'JOB-20260903-00203', 'JOB-20260904-00001', 'JOB-20260904-00002',
                    'JOB-20260904-00003', 'JOB-20260904-00004', 'JOB-20260904-00005',
                ],
                12: [
                    'JOB-20260904-00006', 'JOB-20260904-00007', 'JOB-20260904-00008',
                    'JOB-20260904-00009', 'JOB-20260904-00010', 'JOB-20260904-00011',
                    'JOB-20260904-00012', 'JOB-20260904-00013', 'JOB-20260904-00014',
                    'JOB-20260904-00015', 'JOB-20260904-00016', 'JOB-20260904-00017',
                    'JOB-20260904-00018', 'JOB-20260904-00019', 'JOB-20260904-00020',
                ],
                13: [
                    'JOB-20260904-00021', 'JOB-20260904-00022', 'JOB-20260904-00023',
                    'JOB-20260904-00024', 'JOB-20260904-00025', 'JOB-20260904-00026',
                    'JOB-20260904-00027', 'JOB-20260904-00028', 'JOB-20260904-00029',
                    'JOB-20260904-00030', 'JOB-20260904-00031', 'JOB-20260904-00032',
                    'JOB-20260904-00033', 'JOB-20260904-00034', 'JOB-20260904-00035',
                ],
                14: [
                    'JOB-20260904-00036', 'JOB-20260904-00037', 'JOB-20260904-00038',
                    'JOB-20260904-00039', 'JOB-20260904-00040', 'JOB-20260904-00041',
                    'JOB-20260904-00042', 'JOB-20260904-00043', 'JOB-20260904-00044',
                    'JOB-20260904-00045', 'JOB-20260904-00046', 'JOB-20260904-00047',
                    'JOB-20260904-00048', 'JOB-20260904-00049', 'JOB-20260904-00050',
                ],
                15: [
                    'JOB-20260904-00051', 'JOB-20260904-00052', 'JOB-20260904-00053',
                    'JOB-20260904-00054', 'JOB-20260904-00055', 'JOB-20260904-00056',
                    'JOB-20260904-00057', 'JOB-20260904-00058', 'JOB-20260904-00059',
                    'JOB-20260904-00060', 'JOB-20260904-00061', 'JOB-20260904-00062',
                    'JOB-20260904-00063', 'JOB-20260904-00064', 'JOB-20260904-00065',
                ],
                16: [
                    'JOB-20260904-00066', 'JOB-20260904-00067', 'JOB-20260904-00068',
                    'JOB-20260904-00069', 'JOB-20260904-00070', 'JOB-20260904-00071',
                    'JOB-20260904-00072', 'JOB-20260904-00073', 'JOB-20260904-00074',
                    'JOB-20260904-00075', 'JOB-20260904-00076', 'JOB-20260904-00077',
                    'JOB-20260904-00078', 'JOB-20260904-00079', 'JOB-20260904-00080',
                ],
                17: [
                    'JOB-20260904-00081', 'JOB-20260904-00082', 'JOB-20260904-00083',
                    'JOB-20260904-00084', 'JOB-20260904-00085', 'JOB-20260904-00086',
                    'JOB-20260904-00087', 'JOB-20260904-00088', 'JOB-20260904-00089',
                    'JOB-20260904-00090', 'JOB-20260904-00091', 'JOB-20260904-00092',
                    'JOB-20260904-00093', 'JOB-20260904-00094', 'JOB-20260904-00095',
                ],
                18: [
                    'JOB-20260904-00096', 'JOB-20260904-00097', 'JOB-20260904-00098',
                    'JOB-20260904-00099', 'JOB-20260904-00100', 'JOB-20260904-00101',
                    'JOB-20260904-00102', 'JOB-20260904-00103', 'JOB-20260904-00104',
                    'JOB-20260904-00105', 'JOB-20260904-00106', 'JOB-20260904-00107',
                    'JOB-20260904-00108', 'JOB-20260904-00109', 'JOB-20260904-00110',
                ],
            },
        },
    }

    # Legacy H5 files omit the pulse CSV and compiled-clock metadata. These explicit
    # historical supplements are NOT reconstructed from H5 and never come from the
    # current station. Unknown campaigns must provide their own archived metadata.
    # Sep07: saved run instructions = 92 tProc ticks at 430.08 MHz; pi_frac = 40.
    # Sep05: same physical clock; its acquisition decoder used the older clock below.
    # Earlier g15: retained acquisition-time metadata from the existing analysis export.
    hardware_by_dataset = {
        'Sep07 K52.3 g29.2': dict(
            floquet_cycle_us=92 / 430.08,
            couplings_MHz=[1 / (4 * 40 * (92 / 430.08))] * 4,
            decoder_cycle_us=92 / 430.08,
            source='Sep07 archived run instructions (92 ticks / 430.08 MHz), pi_frac 40; H5 omits these fields'),
        'Sep05 K3.6 g30': dict(
            floquet_cycle_us=92 / 430.08,
            couplings_MHz=[1 / (4 * 40 * (92 / 430.08))] * 4,
            decoder_cycle_us=0.21963713369963367,
            source='Sep05 archived compiled timing and decoder-frame audit; H5 omits these fields'),
    }
    for name in ('Sep01 K44 g15', 'Sep02 K20 g15', 'Sep02-04 K3.6 g15'):
        hardware_by_dataset[name] = dict(
            floquet_cycle_us=0.41351877289377287, couplings_MHz=[0.015114186851211072] * 4,
            decoder_cycle_us=0.41351877289377287,
            source='Archived acquisition-time hardware metadata from the existing g15 analysis export; not H5')


    # Bind each supplement to verified job IDs, not to whichever label remains
    # selected when the job lists are edited. Unknown jobs are never assigned
    # a historical clock just because their Kerr or date looks similar.
    hardware_by_job_id = {}

    for name, dataset in dataset_dumps.items():
        hardware = hardware_by_dataset.get(name)
        jobs = dataset['calibration_job_ids'] + [job for group in dataset['job_ids_by_realization'].values() for job in group]
        for job in jobs:
            hardware_by_job_id[job] = hardware

    return dataset_dumps, hardware_by_job_id


def select_dataset(dataset_dumps, dataset_to_postproc):
    """Pick one dataset's calibration and per-realization job IDs (cell 259).

    Returns (calibration_job_ids, job_ids_by_realization, selected_ids).
    """
    # 3. Resolve only the chosen explicit IDs. No start/end range or automatic additions.
    selected_dataset = dataset_dumps[dataset_to_postproc]
    calibration_job_ids = list(selected_dataset['calibration_job_ids'])
    job_ids_by_realization = {key: list(jobs) for key, jobs in selected_dataset['job_ids_by_realization'].items()}
    spectroscopy_job_ids = [job for jobs in job_ids_by_realization.values() for job in jobs]
    selected_ids = calibration_job_ids + spectroscopy_job_ids
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError('The selected dataset contains a duplicate job ID')
    print(f'{dataset_to_postproc}: {len(job_ids_by_realization)} realizations, '
          f'{len(spectroscopy_job_ids)} spectroscopy jobs, {len(calibration_job_ids)} calibration jobs')

    return calibration_job_ids, job_ids_by_realization, selected_ids


def load_saved_spectroscopy(dataset_to_postproc, data_base_directory,
                            manifest_directory, calibration_job_ids,
                            job_ids_by_realization, selected_ids,
                            hardware_by_job_id, hardware_override=None,
                            load_shots=False, reload_data=True):
    """Load the saved HDF5 files (cell 264). No MPM here.

    Returns a dict of the names the later steps read, including
    `loaded_spectroscopy`, `phase_calibration` and `partial_realizations`.
    """
    raw_job_cache = {}

    # Read selected H5 headers, then mean arrays. This cell does not run MPM.
    from time import perf_counter

    if reload_data or 'h5_array_cache' not in globals():
        h5_array_cache = {}
    manifest, headers = scan_completed_spectroscopy(data_base_directory, selected_ids)
    print('Unavailable H5 files:', manifest['missing_h5'])
    for item in manifest['excluded']:
        print(f"Rejected {item['job_id']}: {item['reason']}")
    selection = classify_selected_jobs(headers, calibration_job_ids, job_ids_by_realization)
    manifest['selected_explicit_groups'] = selection
    for issue in selection['issues']:
        print(issue)

    # Keep requested lists unchanged: a missing file can be picked up on the next load.
    loaded_calibration_ids = selection['calibration_job_ids']
    loaded_job_groups = {int(key): jobs for key, jobs in selection['job_ids_by_realization'].items()}
    available_ids = loaded_calibration_ids + [job for jobs in loaded_job_groups.values() for job in jobs]
    raw_job_cache = {}
    started = perf_counter()
    for job_id in available_ids:
        hardware = hardware_override or hardware_by_job_id.get(job_id)
        raw_job_cache[job_id] = load_h5(headers[job_id], hardware, h5_array_cache, load_shots)
    active_objects = {id(expt) for expt in raw_job_cache.values()}
    h5_array_cache = {key: value for key, value in h5_array_cache.items() if id(value) in active_objects}
    print(f'{len(available_ids)} H5 files loaded in {perf_counter()-started:.1f} s; no server requests.')

    # Calibration is fitted from saved quadratures. Physical MHz additionally needs archived timing.
    phase_calibration = None
    if loaded_calibration_ids:
        children = [raw_job_cache[job] for job in loaded_calibration_ids]
        phase_calibration = MBRPhaseCorrectionExperiment._from_expts(children, job_ids=loaded_calibration_ids)
        phase_calibration.data = MBRPhaseCorrectionExperiment.analyze_calibration(children)
        phase_calibration.data.mode_labels = ['M1'] + [f'S{mode}' for mode in children[0].cfg.expt.swap_stors]
        if children[0].saved_hardware is not None:
            phase_calibration.data.hardware = SavedSpectroscopyExperiment._saved_parameters(children).hardware
        print(f'Phase calibration: {len(phase_calibration.data.occupations)} occupations; fitted from H5.')

    # All available chunks remain inspectable; only complete common-grid rows enter analysis.
    loaded_spectroscopy = {}
    available_spectroscopy = {}
    partial_realizations = {}
    for realization, all_jobs in sorted(loaded_job_groups.items()):
        info = selection['realizations'][str(realization)]
        analysis_jobs = selection['analysis_job_ids_by_realization'][str(realization)]
        children = [raw_job_cache[job] for job in all_jobs]
        available_spectroscopy[realization] = SavedSpectroscopyExperiment._from_expts(children, job_ids=all_jobs)
        partial_realizations[realization] = info
        if analysis_jobs:
            children = [raw_job_cache[job] for job in analysis_jobs]
            loaded_spectroscopy[realization] = SavedSpectroscopyExperiment._from_expts(children, job_ids=analysis_jobs)
        print(f'r={realization}: {len(all_jobs)} saved jobs; {len(analysis_jobs)} on common cycle grid; '
              f"{'PARTIAL realization' if info['incomplete'] else 'all selected pairs present'}")
        if info['identity']['saved_realization'] != realization:
            print(f"  Display r={realization}; saved realization={info['identity']['saved_realization']}, "
                  f"saved seed={info['identity']['seed']}")
        if info['missing_pair_indices']:
            print('  Unavailable pair indices:', info['missing_pair_indices'])
        for item in info['excluded_from_rectangular_analysis']:
            print(f"  {item['job_id']}: {item['reason']} (retained in available_spectroscopy/raw_job_cache)")

    loaded_dataset_name = dataset_to_postproc
    sources = sorted({expt.saved_hardware['source'] for expt in raw_job_cache.values() if expt.saved_hardware is not None})
    for source in sources:
        print('Absolute time/couplings:', source)
    if any(expt.saved_hardware is None for expt in raw_job_cache.values()):
        print('No archived clock/couplings: cycle-domain data loaded; physical-MHz analysis needs that metadata.')
    if manifest_directory is not None:
        manifest_folder = Path(manifest_directory)
        manifest_folder.mkdir(parents=True, exist_ok=True)
        manifest_name = re.sub(r'[^A-Za-z0-9_-]+', '_', dataset_to_postproc)
        manifest_path = manifest_folder / f'explicit_{manifest_name}.json'
        manifest_path.write_text(json.dumps(make_plain(manifest), indent=2), encoding='utf-8')
        print(f'Manifest: {manifest_path.resolve()}')
    print('Ready: raw data and phase calibration loaded. No spectroscopy frequency fit has run.')

    return {
        "raw_job_cache": raw_job_cache,
        "loaded_dataset_name": loaded_dataset_name,
        "loaded_spectroscopy": loaded_spectroscopy,
        "partial_realizations": partial_realizations,
        "phase_calibration": phase_calibration,
        "available_spectroscopy": available_spectroscopy,
    }


def preview_saved_traces(loaded):
    """Preview the saved calibration and disorder traces (cell 266). No MPM.

    Returns (reconfigured_expts, failures).
    """
    loaded_dataset_name = loaded["loaded_dataset_name"]
    loaded_spectroscopy = loaded["loaded_spectroscopy"]
    partial_realizations = loaded["partial_realizations"]
    phase_calibration = loaded["phase_calibration"]
    raw_job_cache = loaded["raw_job_cache"]
    preview_failures = []

    # Preview only: no MPM or frequency fitting. Measured H5 quadratures stay unchanged.
    import matplotlib.pyplot as plt

    if phase_calibration is None:
        print('No phase calibration in this selection; showing saved spectroscopy only.')
    else:
        cal = phase_calibration.data
        cal_labels = [str(tuple(state)) for state in cal.occupations]
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
        axes[0].errorbar(np.arange(len(cal_labels)), cal.phase_mod180, yerr=cal.phase_error,
                         fmt='.', capsize=2)
        axes[0].set(ylabel='phase slope (deg / physical cycle)', title='Measured phase calibration')
        if 'hardware' in cal:
            error = 1000 * np.asarray(cal.phase_error) / (360 * cal.hardware.floquet_cycle_us)
            axes[1].bar(np.arange(len(cal_labels)), error)
            axes[1].set(ylabel='slope standard error (kHz)', title='Calibration uncertainty (not peak-fit uncertainty)')
        else:
            axes[1].bar(np.arange(len(cal_labels)), cal.phase_error)
            axes[1].set(ylabel='slope standard error (deg / cycle)', title='Calibration uncertainty')
        for ax in axes:
            ax.set_xticks(np.arange(len(cal_labels)), cal_labels, rotation=90, fontsize=7)
        fig.suptitle(loaded_dataset_name)
        plt.show()

    fft_previews = {}
    reconfigured_expts = {}
    for realization, loaded in sorted(loaded_spectroscopy.items()):
        status = partial_realizations[realization]
        partial = 'PARTIAL acquisition' if status['incomplete'] else 'all selected channels acquired'
        expt = SavedSpectroscopyExperiment._from_expts(loaded.batch_expts, job_ids=loaded.batch_job_ids)
        reconfigured_expts[realization] = expt

        if loaded.batch_expts[0].saved_hardware is None:
            reconstruct = (SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy
                           if 'offdiag_cycles' in loaded.batch_expts[0].cfg.expt
                           else SavedSpectroscopyExperiment.reconstruct_spectroscopy)
            raw = reconstruct(loaded.batch_expts)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(raw.cycles, np.abs(raw.A).T)
            ax.set(xlabel='physical Floquet cycles', ylabel='|measured return|', title=f'r={realization}: {partial}')
        else:
            data = expt.analyze(
                                calibration=phase_calibration,
                                phase_frame='as_acquired',
                                spectrum_method='fft',
                                fft_window='raw',
                                zero_padding=1)
            fig = expt.display(data=data, spectrum_method='fft', level_statistics=False)
            previous_title = fig._suptitle.get_text() if fig._suptitle else ''
            fig.suptitle(f'r={realization}: {partial}\n{previous_title}')
            fft_previews[realization] = data
            print(f'r={realization}: {len(data.reconstruction.A)} traces, '
                  f'{data.spectrum.time_us[-1]:.3f} us; {partial}')
        plt.show()

    # A saved late chunk must still be visible even when its earlier chunk failed.
    for realization, status in sorted(partial_realizations.items()):
        for item in status['excluded_from_rectangular_analysis']:
            child = raw_job_cache[item['job_id']]
            if 'offdiag_cycles' not in child.cfg.expt:
                continue
            raw = SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy([child])
            fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
            ax.plot(raw.cycles, raw.A[0].real, '.-', label='Re return')
            ax.plot(raw.cycles, raw.A[0].imag, '.-', label='Im return')
            ax.set(xlabel='physical Floquet cycles', ylabel='measured return',
                   title=f"r={realization}: {item['job_id']} — saved partial trace; not pooled into level statistics")
            ax.legend()
            plt.show()

    return reconfigured_expts, preview_failures


def mpm_settings(target_level_count=35,
                 track_frequency_tolerance_bins=1.0,
                 minimum_consecutive_ranks=3,
                 minimum_supporting_rows=1,
                 merge_frequency_tolerance="calibration",
                 calibration_sigma_multiplier=3.0,
                 frequency_tolerance_floor_kHz=0.1,
                 match_tolerance_bins=1.5,
                 show_mpm_figures=False):
    """Resolve the Matrix-Pencil settings (cell 269).

    Returns (spectroscopy_mpm_options, target_level_count,
    match_tolerance_bins, show_mpm_figures).
    """
    mpm_track_frequency_tolerance_bins = track_frequency_tolerance_bins
    mpm_minimum_consecutive_ranks = minimum_consecutive_ranks
    mpm_minimum_supporting_rows = minimum_supporting_rows
    mpm_merge_frequency_tolerance = merge_frequency_tolerance
    mpm_calibration_sigma_multiplier = calibration_sigma_multiplier
    mpm_frequency_tolerance_floor_kHz = frequency_tolerance_floor_kHz

    # Stable rowwise MPM candidates are merged using calibration uncertainty.
    # Rank stability and cross-row consistency determine the selection score.
    target_level_count = 35
    mpm_track_frequency_tolerance_bins = 0.5
    mpm_minimum_consecutive_ranks = 3
    mpm_rank_sweep_extra = mpm_minimum_consecutive_ranks - 1
    mpm_minimum_supporting_rows = 1
    mpm_merge_frequency_tolerance = 'calibration'
    mpm_calibration_sigma_multiplier = 1.0
    mpm_frequency_tolerance_floor_kHz = 1e-12  # cross-row merge floor in kHz
    # Existing numeric reuse retained: dedup is in kHz, tracking is in FFT bins.
    mpm_dedup_frequency_tolerance_kHz = mpm_track_frequency_tolerance_bins

    # A theory match means an order-preserving principal-zone match within this many
    # finite-time FFT bins. Raw all-35 assignment errors are reported separately.
    match_tolerance_bins = 0.2
    show_mpm_figures = True

    # Re-run this settings cell and the MPM cell after changing a fit setting.
    # Loading, matching, and plotting do not need to rerun MPM.
    spectroscopy_mpm_options = dict(
        phase_frame='as_acquired',
        spectrum_method='mpm',
        fft_window='raw',
        zero_padding=1,
        mpm_match_decay=False,
        mpm_requested_max_modes=target_level_count,
        mpm_track_frequency_tolerance_bins=mpm_track_frequency_tolerance_bins,
        mpm_minimum_consecutive_ranks=mpm_minimum_consecutive_ranks,
        mpm_rank_sweep_extra=mpm_rank_sweep_extra,
        mpm_minimum_supporting_rows=mpm_minimum_supporting_rows,
        mpm_merge_frequency_tolerance_bins=mpm_merge_frequency_tolerance,
        mpm_calibration_sigma_multiplier=mpm_calibration_sigma_multiplier,
        mpm_merge_frequency_tolerance_floor_kHz=mpm_frequency_tolerance_floor_kHz,
        mpm_dedup_frequency_tolerance_MHz=1e-3 * mpm_dedup_frequency_tolerance_kHz,
    )

    return (
        spectroscopy_mpm_options,
        target_level_count,
        match_tolerance_bins,
        show_mpm_figures,
    )


def run_mpm(loaded, spectroscopy_mpm_options, target_level_count,
            saved_floquet_timing_fn=None):
    """Run Matrix Pencil on the loaded spectroscopy (cell 271).

    Returns `spectroscopy_records`.
    """
    loaded_dataset_name = loaded["loaded_dataset_name"]
    loaded_spectroscopy = loaded["loaded_spectroscopy"]
    phase_calibration = loaded["phase_calibration"]

    # Reuse raw jobs but make a fresh aggregate, so a new fit does not mutate an older record's expt.data.
    print(f'Analyzing: {loaded_dataset_name}')
    spectroscopy_records = {}
    for realization, loaded in sorted(loaded_spectroscopy.items()):
        started = perf_counter()
        print(f'r={realization}: running MPM...', flush=True)
        expt = SavedSpectroscopyExperiment._from_expts(loaded.batch_expts, job_ids=loaded.batch_job_ids)
        data = expt.analyze(calibration=phase_calibration, **spectroscopy_mpm_options)

        first_cfg = expt.batch_expts[0].cfg.expt
        saved_target_kerr_kHz = float(first_cfg.d72_self_kerr_kHz)
        analyzed_hardware_kerr_kHz = 1e3 * float(data.hardware.physical_kerr_MHz)
        if not np.isclose(saved_target_kerr_kHz, analyzed_hardware_kerr_kHz, rtol=0.0, atol=1e-3):
            raise RuntimeError(f'r={realization}: saved 7-2 Kerr {saved_target_kerr_kHz:.6f} kHz differs from '
                               f'as-acquired Kerr {analyzed_hardware_kerr_kHz:.6f} kHz; '
                               'reload the original phase calibration and use manual_kerr')

        mpm = data.matrix_pencil
        theory_levels_MHz = np.asarray(data.spectrum.energies_MHz, dtype=float)
        if len(theory_levels_MHz) != target_level_count:
            raise RuntimeError(f'r={realization}: expected {target_level_count} theory levels, '
                               f'got {len(theory_levels_MHz)}')
        spectroscopy_records[realization] = AttrDict(dict(
            dataset=loaded_dataset_name,
            job_ids=list(loaded.batch_job_ids), expt=expt, data=data, theory_levels_MHz=theory_levels_MHz,
            selected_pairs=[(tuple(map(int, pair[0])), tuple(map(int, pair[1])))
                            for pair in first_cfg.d72_selected_pairs],
        ))

        cycles = np.asarray(data.reconstruction.cycles)
        time_us = np.asarray(data.spectrum.time_us)
        print(f'r={realization}: cycles={cycles[0]:g}..{cycles[-1]:g}, points={len(cycles)}, '
              f'analysis time={time_us[0]:.3f}..{time_us[-1]:.3f} us, '
              f'T_F={data.hardware.floquet_cycle_us:.9f} us')
        print(f'  rowwise accepted/raw={len(mpm.candidates.per_row)}/{len(mpm.candidates.raw_per_row)}, '
              f'merged clusters={len(mpm.candidates.merged)}, selected shared poles={len(mpm.selected_frequencies_MHz)}, '
              f'residual={mpm.relative_residual:.3f}, condition={mpm.design_condition_number:.3g}, '
              f'elapsed={perf_counter() - started:.1f} s')
        # Time provenance is explicit: H5 metadata or the bound historical supplement.
        timing = saved_floquet_timing(loaded.batch_expts[0])
        spectroscopy_records[realization].saved_timing = timing
        if timing is not None:
            print('  Timing source:', timing['source'])
            print(f"  Archived timing: T_F={timing['cycle_us']:.9f} us, "
                  f"last point={cycles[-1] * timing['cycle_us']:.3f} us (scheduled, excludes preparation/readout)")
            if not np.isclose(timing['cycle_us'], data.hardware.floquet_cycle_us, rtol=1e-9, atol=1e-12):
                print('  WARNING: analysis time differs from the archived timing metadata. '
                      'Do not interpret the frequency scale until that discrepancy is resolved.')

    return spectroscopy_records


def report_theory_matches(spectroscopy_records, match_tolerance_bins=1.5,
                          ncols=3):
    """Match recovered poles to all 35 exact theory levels (cell 274)."""

    for realization, record in sorted(spectroscopy_records.items()):
        data = record.data
        mpm = data.matrix_pencil
        sampling_frequency_MHz = float(mpm.sampling.sampling_frequency_MHz)
        tolerance_MHz = match_tolerance_bins * float(data.spectrum.fft_resolution_MHz)
        match = match_levels(
            mpm.selected_frequencies_MHz, record.theory_levels_MHz, sampling_frequency_MHz, tolerance_MHz
        )
        record.match = match
        print(
            f'r={realization}: matched={len(match.pairs)}/'
            f'{len(match.theory_MHz)} within '
            f'{1e3 * tolerance_MHz:.3f} kHz; '
            f'matched MAE={1e3 * match.matched_mae_MHz:.3f} kHz; '
            f'sorted-35 MAE/RMSE/max='
            f'{1e3 * match.ordered_mae_MHz:.3f}/'
            f'{1e3 * match.ordered_rmse_MHz:.3f}/'
            f'{1e3 * match.ordered_max_error_MHz:.3f} kHz'
        )
        print('  missing theory (kHz):', np.round(1e3 * match.theory_MHz[match.missing], 3))
        print('  unmatched measured (kHz):', np.round(1e3 * match.measured_MHz[match.spurious], 3))

    ncols = 4
    nrows = int(np.ceil(len(spectroscopy_records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14 * 2, 3.8 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for axis, (realization, record) in zip(axes, sorted(spectroscopy_records.items())):
        match = record.match
        theory_kHz = 1e3 * match.theory_MHz
        measured_kHz = 1e3 * match.measured_MHz
        for measured_index, theory_index in match.pairs:
            axis.plot(
                [theory_kHz[theory_index], measured_kHz[measured_index]],
                [1.0, 0.0],
                color='tab:blue',
                alpha=0.22,
                linewidth=0.8,
            )
        axis.scatter(
            theory_kHz, np.ones(len(theory_kHz)), marker='|', s=150, color='tab:green', label='exact theory'
        )
        axis.scatter(
            measured_kHz, np.zeros(len(measured_kHz)), marker='x', s=28, color='black', label='selected MPM pole'
        )
        axis.scatter(
            theory_kHz[match.missing],
            np.ones(len(match.missing)),
            marker='|',
            s=180,
            linewidths=2.0,
            color='tab:red',
            label='unmatched theory',
        )
        axis.scatter(
            measured_kHz[match.spurious],
            np.zeros(len(match.spurious)),
            marker='x',
            s=42,
            linewidths=1.7,
            color='tab:red',
            label='unmatched MPM',
        )
        axis.set_yticks([0.0, 1.0], ['MPM', 'theory'])
        axis.set(
            xlabel='principal-zone energy E/h (kHz)',
            title=(
                f'r={realization}: {len(match.pairs)}/'
                f'{len(theory_kHz)} within tolerance; '
                f'sorted MAE={1e3 * match.ordered_mae_MHz:.2f} kHz'
            ),
        )
        axis.grid(axis='x', alpha=0.2)

    for axis in axes[len(spectroscopy_records) :]:
        axis.set_visible(False)
    axes[0].legend(fontsize=8, loc='upper left', ncols=2)
    plt.show()

    return plt.gcf()


def report_level_statistics(spectroscopy_records, partial_realizations,
                            include_partial_acquisitions=False,
                            edge_fraction=0.10, level_stat_bins=15):
    """Level statistics of the recovered spectra (cell 276)."""
    include_partial_acquisitions = False  # Missing extracted poles are NOT excluded.
    edge_fraction = 0.1
    level_stat_bins = 10

    measured_ratios_by_realization = {}
    theory_ratios_by_realization = {}

    for realization, record in sorted(spectroscopy_records.items()):
        if not include_partial_acquisitions and partial_realizations.get(realization, {}).get(
            'incomplete', False
        ):
            print(f'r={realization}: partial acquisition stays in previews, excluded from pooled statistics')
            continue
        match = record.match

        measured_trim_count = int(np.ceil(edge_fraction * len(match.measured_MHz)))
        theory_trim_count = int(np.ceil(edge_fraction * len(match.theory_MHz)))
        measured_bulk_MHz = match.measured_MHz[measured_trim_count:-measured_trim_count]
        theory_bulk_MHz = match.theory_MHz[theory_trim_count:-theory_trim_count]

        measured_gaps_MHz = np.diff(measured_bulk_MHz)
        measured_ratios = np.minimum(measured_gaps_MHz[:-1], measured_gaps_MHz[1:]) / np.maximum(
            measured_gaps_MHz[:-1], measured_gaps_MHz[1:]
        )
        measured_ratios_by_realization[realization] = measured_ratios

        theory_gaps_MHz = np.diff(theory_bulk_MHz)
        theory_ratios = np.minimum(theory_gaps_MHz[:-1], theory_gaps_MHz[1:]) / np.maximum(
            theory_gaps_MHz[:-1], theory_gaps_MHz[1:]
        )
        theory_ratios_by_realization[realization] = theory_ratios

        print(
            f'r={realization}: selected={len(match.measured_MHz)}, '
            f'missing={len(match.missing)}, '
            f'spurious={len(match.spurious)}, '
            f'bulk levels={len(measured_bulk_MHz)}/'
            f'{len(theory_bulk_MHz)}, '
            f'measured/theory ratios='
            f'{len(measured_ratios)}/{len(theory_ratios)}'
        )

    measured_pooled = np.concatenate(list(measured_ratios_by_realization.values()))
    theory_pooled = np.concatenate(list(theory_ratios_by_realization.values()))

    ratio_axis = np.linspace(0.0, 1.0, 1000)
    poisson_pdf = 2.0 / (1.0 + ratio_axis) ** 2
    goe_pdf = (27.0 / 4.0) * (ratio_axis + ratio_axis**2) / (1.0 + ratio_axis + ratio_axis**2) ** 2.5
    poisson_mean = 2.0 * np.log(2.0) - 1.0
    goe_mean = 4.0 - 2.0 * np.sqrt(3.0)
    ratio_edges = np.linspace(0.0, 1.0, level_stat_bins + 1)
    realizations = np.asarray(sorted(measured_ratios_by_realization))
    measured_means = np.asarray(
        [np.mean(measured_ratios_by_realization[realization]) for realization in realizations]
    )
    theory_means = np.asarray(
        [np.mean(theory_ratios_by_realization[realization]) for realization in realizations]
    )
    measured_mean = np.mean(measured_means)
    theory_mean = np.mean(theory_means)
    measured_sem = np.std(measured_means, ddof=1) / np.sqrt(len(measured_means))
    theory_sem = np.std(theory_means, ddof=1) / np.sqrt(len(theory_means))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

    axes[0].hist(
        measured_pooled,
        bins=ratio_edges,
        density=True,
        alpha=0.35,
        color='black',
        edgecolor='black',
        label=(f'measured MPM: n={len(measured_pooled)}, ' f'mean={np.mean(measured_pooled):.3f}'),
    )
    axes[0].hist(
        theory_pooled,
        bins=ratio_edges,
        density=True,
        histtype='step',
        linewidth=2,
        color='tab:green',
        label=(f'exact theory: n={len(theory_pooled)}, ' f'mean={np.mean(theory_pooled):.3f}'),
    )

    axes[0].plot(ratio_axis, poisson_pdf, color='tab:blue', linewidth=2, label='Poisson')
    axes[0].plot(ratio_axis, goe_pdf, color='tab:orange', linewidth=2, label='GOE')
    axes[0].set(
        xlim=(0.0, 1.0),
        xlabel=r'adjacent-gap ratio $\tilde r$',
        ylabel='probability density',
        title='pooled bulk level statistics',
    )
    axes[0].legend(fontsize=8)

    x = np.arange(len(realizations))
    axes[1].scatter(x - 0.08, measured_means, marker='x', s=65, color='black', label='measured MPM')
    axes[1].scatter(x + 0.08, theory_means, color='tab:green', label='exact theory')
    mean_x = len(realizations)
    axes[1].errorbar(mean_x - 0.08, measured_mean, yerr=measured_sem, fmt='D', capsize=4, color='black')
    axes[1].errorbar(mean_x + 0.08, theory_mean, yerr=theory_sem, fmt='D', capsize=4, color='tab:green')
    axes[1].axhline(poisson_mean, color='tab:blue', linestyle='--', label=f'Poisson mean={poisson_mean:.3f}')
    axes[1].axhline(goe_mean, color='tab:orange', linestyle='--', label=f'GOE mean={goe_mean:.3f}')
    axes[1].set_xticks(np.append(x, mean_x), [f'r={r}' for r in realizations] + ['mean'])
    axes[1].set(
        ylim=(0.0, 1.0),
        ylabel=r'mean adjacent-gap ratio $\langle\tilde r\rangle$',
        title='mean gap ratio by realization',
    )
    axes[1].legend(fontsize=8)
    plt.show()

    level_statistics = dict(
        edge_fraction=edge_fraction,
        measured_by_realization=measured_ratios_by_realization,
        theory_by_realization=theory_ratios_by_realization,
        measured_pooled=measured_pooled,
        theory_pooled=theory_pooled,
        measured_realization_means=measured_means,
        theory_realization_means=theory_means,
        measured_disorder_mean=measured_mean,
        measured_disorder_sem=measured_sem,
        theory_disorder_mean=theory_mean,
        theory_disorder_sem=theory_sem,
    )

    print(
        f'measured={measured_mean:.3f} '
        f'+/- {measured_sem:.3f} SEM; '
        f'theory={theory_mean:.3f} '
        f'+/- {theory_sem:.3f} SEM; '
        f'Poisson={poisson_mean:.3f}; '
        f'GOE={goe_mean:.3f}'
    )

    return plt.gcf()

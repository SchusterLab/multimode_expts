"""Measure central-boson populations for a selected initial-state direction.

One job fixes the dynamics and readout mode, then sweeps initial occupations
and Floquet cycles. Analysis adds number moments and sector-averaged
two-time correlations to the parent's multiparity results.
"""
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from experiments.assembled_data import ASSEMBLED_DIR, new_stem
from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.multiparity import MultiparityQsimExperiment
from itertools import product


class CentralBosonAGPExperiment(MultiparityQsimExperiment):
    """Occupation-by-cycle population scan, with ordinary multiparity analysis.

    ``cfg.expt.photon_sector_number`` fixes the total photon number.
    ``cfg.expt.direction`` selects M1 (0) or a physical ID in ``swap_stors``.
    Only states with nonzero initial occupation in that mode are generated,
    in ascending lexicographic order with columns ``[M1, *swap_stors]``.
    ``cfg.expt.floquet_cycles`` is the shared inner sweep. The inherited driver
    runs a Program for each pair. Readout still uses ``cfg.expt.ro_stor``.

    ``data['occupations']`` records the state table; ``ypts`` holds its row
    indices and ``xpts`` holds cycles. Analyzed probabilities and means have
    shape ``(initial states, cycle points)``. First- and second-moment
    ensemble-averaged correlations have shape ``(cycle points,)`` and use
    the full sector dimension, including unmeasured zero-weight states.
    Moments describe photon number when populations are confined to n <= 3.
    """

    default_program = FockPopulationProgram #Evinced just in case. Might be better to define AGPProgram?

    def generate_fock_strings(self):
        """Return fixed-N states whose selected direction is initially occupied."""
        _photon_sector = self.cfg.expt.photon_sector_number
        _modes = [0, *self.cfg.expt.swap_stors]
        _number_of_modes = len(_modes)
        _direction_index = _modes.index(self.cfg.expt.direction)
        occupations = [list(_state) for _state in product(range(_photon_sector + 1), repeat=_number_of_modes)
                       if sum(_state) == _photon_sector and _state[_direction_index] > 0]
        return occupations

    def sweep_axes(self): #overrides `sweep_axes` of `QsimExperiment`
        cycles = self.cfg.expt.floquet_cycles
        occupations = self.generate_fock_strings()
        self.cfg.expt.swept_params = ['occupations', 'floquet_cycle']
        return [('occupations', occupations), ('floquet_cycle', cycles)]

    def shape_data(self, axes, points):
        data = super().shape_data(axes, points)
        data['occupations'] = np.asarray(axes[0][1], dtype=int)
        data['ypts'] = np.arange(len(data['occupations']))
        return data

    def analyze(self, data=None, **kwargs):
        """Add <n_r^2(t)> and ensemble-averaged two-time correlations.

        The initial mode d is ``direction``; the measured mode r is ``ro_stor``.
        C_k(t) = Tr_N[n_r^k(t) n_d^k(0)] / D.
        For k=1,2, weight each state's moment by its initial occupation to
        that power and divide by the full fixed-N sector dimension D.
        Raw results use the existing keys' convention; corrected results
        have a ``_corrected`` suffix when a calibration is configured.
        """
        super().analyze(data=data, **kwargs)
        modes = [0, *self.cfg.expt.swap_stors]
        direction_index = modes.index(self.cfg.expt.direction)
        initial_n = np.asarray(self.data['occupations'])[:, direction_index]
        photons = self.cfg.expt.photon_sector_number
        # Count the complete fixed-N basis, including zero-direction states.
        dimension = 0
        for state in product(range(photons + 1), repeat=len(modes)):
            if sum(state) == photons:
                dimension += 1
        self.data['sector_dimension'] = np.asarray(dimension)

        for suffix in ['', '_corrected']:
            second_key = f'nmod4_second_moment{suffix}'
            averaged_correlation_key = f'nmod4_ensemble_averaged_correlation{suffix}'
            second_averaged_correlation_key = f'nmod4_second_moment_ensemble_averaged_correlation{suffix}'
            
            # Drop the previous names when reanalyzing an older result.
            self.data.pop(f'nmod4_correlation{suffix}', None)
            self.data.pop(f'nmod4_second_moment_correlation{suffix}', None)
            
            if f'nmod4_mean{suffix}' not in self.data:
                for key in [second_key, averaged_correlation_key,
                            second_averaged_correlation_key]:
                    self.data.pop(key, None)
                continue
            #For now, the only hard-coded part is inserting 4 in range(4) below.
            #This is the only part to edit for the higher photon number extension
            probabilities = np.stack([self.data[f'p_mod{n}{suffix}'] for n in range(4)], axis=-1) 
            
            first_moment = self.data[f'nmod4_mean{suffix}']
            n_values = np.arange(probabilities.shape[-1])
            second_moment = probabilities @ n_values**2
            self.data[second_key] = second_moment
            self.data[averaged_correlation_key] = initial_n @ first_moment / dimension
            self.data[second_averaged_correlation_key] = initial_n**2 @ second_moment / dimension

        return self.data


class CentralBosonAGPDisorderScan:
    """Collect complete AGP jobs and display their disorder-by-cycle correlations.

    Input files must already contain CentralBosonAGPExperiment's analysis.
    Each job keeps its own readout correction. No correction or AGP integral
    is calculated here. Disorder is cfg.expt.disorder_khz when recorded;
    otherwise it is the RMS of the saved storage detunings, converted to kHz.

    self.data contains flat arrays for the HDF5 viewer. Compatible numeric
    child datasets other than raw idata/qdata are stacked along the job axis.
    Raw shots are stored only in the complete child snapshots. save_data()
    copies every child's complete HDF5 contents, including nested groups and
    all attributes, into children/0000, children/0001, ... in disorder order.

    Example::

        scan = CentralBosonAGPDisorderScan(job_fnames)
        scan.analyze()
        scan.display(moment=2)  # corrected when available; raw=True selects raw
        scan.save_data()       # one new HDF5, with no YAML manifest
    """

    correlation_keys = (
        'nmod4_ensemble_averaged_correlation',
        'nmod4_second_moment_ensemble_averaged_correlation',
    )

    def __init__(self, fnames, notes=''):
        if isinstance(fnames, (str, Path)):
            fnames = [fnames]
        self.fnames = [str(Path(fname).resolve()) for fname in fnames]
        if not self.fnames:
            raise ValueError('a disorder scan needs at least one AGP job')
        self._child_sources = [(fname, '/') for fname in self.fnames]
        self.notes = notes
        self.fname = None
        self.data = {}

    def analyze(self):
        """Stack saved child results without changing or reanalyzing any job.

        Missing corrected rows are NaN, so raw and corrected values are never
        mixed. Data that cannot be stacked still remains in the child archive.
        """
        children, settings, disorder = [], [], []
        for fname, group_path in self._child_sources:
            with h5py.File(fname, 'r') as saved:
                group = saved[group_path]
                children.append({key: item[()] for key, item in group.items()
                                 if isinstance(item, h5py.Dataset)
                                 and key not in ('idata', 'qdata')})
                expt = json.loads(group.attrs['config'])['expt']
                settings.append(expt)
                detunings = np.asarray(expt['detunings'], dtype=float)
                rms_khz = 1000 * np.sqrt(np.mean(detunings**2)) if detunings.size else 0.
                disorder.append(expt.get('disorder_khz', rms_khz))

        cycles = np.asarray(children[0]['xpts'])
        common_settings = ['photon_sector_number', 'swap_stors', 'direction', 'ro_stor']
        for child, expt in zip(children, settings):
            if not np.array_equal(child['xpts'], cycles):
                raise ValueError('AGP jobs must share the same cycle axis')
            for key in common_settings:
                if expt[key] != settings[0][key]:
                    raise ValueError(f'AGP jobs have different {key}')
            for key in self.correlation_keys:
                if np.shape(child[key]) != cycles.shape:
                    raise ValueError(f'{key} must have one value per cycle')

        order = np.argsort(disorder, kind='stable')
        children = [children[index] for index in order]
        settings = [settings[index] for index in order]
        self.fnames = [self.fnames[index] for index in order]
        self._child_sources = [self._child_sources[index] for index in order]

        # Keep complete children, and stack their numeric results at the root.
        # Raw shots stay in the children without a duplicate assembled array.
        data = {}
        common_keys = set.intersection(*(set(child) for child in children))
        for key in sorted(common_keys):
            values = [np.asarray(child[key]) for child in children]
            if all(value.dtype.kind in 'biufc' and value.shape == values[0].shape
                   for value in values):
                data[key] = np.stack(values)
        for key in self.correlation_keys:
            corrected_key = f'{key}_corrected'
            if any(corrected_key in child for child in children):
                data[corrected_key] = np.stack([
                    child.get(corrected_key, np.full(cycles.shape, np.nan))
                    for child in children
                ])
        data['disorder_khz'] = np.asarray(disorder, dtype=float)[order]
        data['cycles'] = cycles.copy()
        data['detunings'] = np.asarray([expt['detunings'] for expt in settings])
        self.data = data
        return self.data

    def display(self, moment=1, raw=False, **kwargs):
        """Plot one stacked correlation: x=cycles, y=disorder in kHz.

        Corrected data is the default when present. Jobs without correction
        are blank in that map; raw=True displays every job's raw correlation.
        Additional kwargs go to matplotlib's pcolormesh.
        """
        if not self.data:
            self.analyze()
        if moment not in (1, 2):
            raise ValueError('moment must be 1 or 2')
        if np.any(np.diff(self.data['disorder_khz']) == 0):
            raise ValueError('a disorder-axis color map needs one job per disorder value')
        key = self.correlation_keys[moment - 1]
        kind = 'raw'
        if not raw and f'{key}_corrected' in self.data:
            key = f'{key}_corrected'
            kind = 'corrected'
        fig, ax = plt.subplots(figsize=(10, 5))
        mesh = ax.pcolormesh(self.data['cycles'], self.data['disorder_khz'],
                             self.data[key], shading='auto', **kwargs)
        ax.set(xlabel='Floquet cycles', ylabel='Disorder (kHz)',
               title=f'{type(self).__name__}: moment {moment} ({kind})')
        fig.colorbar(mesh, ax=ax, label=f'Ensemble-averaged correlation ({kind})')
        fig.tight_layout()
        return fig, ax

    def save_data(self, fname=None):
        """Write root arrays plus complete child snapshots to one new HDF5.

        Default directory is assembled_data beside the input jobs' data
        directory, or the current bundle's directory after loading a bundle.
        Existing files are not overwritten. Original jobs are only read.
        """
        if not self.data:
            self.analyze()
        if fname is None:
            directory = (Path(self.fname).parent if self.fname else
                         Path(self.fnames[0]).parent.parent / ASSEMBLED_DIR)
            fname = directory / f'{new_stem(type(self).__name__, directory)}.h5'
        fname = Path(fname).resolve()
        fname.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(fname, 'x') as saved:
            saved.attrs['class'] = type(self).__name__
            saved.attrs['source_files'] = json.dumps(self.fnames, ensure_ascii=False)
            saved.attrs['notes'] = self.notes
            for key, value in self.data.items():
                saved.create_dataset(key, data=value)
            archive = saved.create_group('children')
            for index, (source_file, group_path) in enumerate(self._child_sources):
                with h5py.File(source_file, 'r') as source:
                    source.copy(group_path, archive, name=f'{index:04d}',
                                expand_soft=True, expand_external=True, expand_refs=True)
        self.fname = str(fname)
        self._child_sources = [(self.fname, f'/children/{index:04d}')
                               for index in range(len(self.fnames))]
        return self.fname

    @classmethod #class method to directly instantiate the object from fname
    def from_h5file(cls, fname):
        """Load a complete scan; original jobs are not needed for analyze/save."""
        fname = str(Path(fname).resolve())
        with h5py.File(fname, 'r') as saved:
            scan = cls(json.loads(saved.attrs['source_files']), notes=saved.attrs['notes'])
            scan.data = {key: item[()] for key, item in saved.items()
                         if isinstance(item, h5py.Dataset)}
            scan._child_sources = [(fname, f'/children/{key}')
                                   for key in sorted(saved['children'])]
        scan.fname = fname
        return scan

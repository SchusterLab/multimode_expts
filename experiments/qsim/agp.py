"""Measure central-boson populations for a selected initial-state direction.

One job fixes the dynamics and readout mode, then sweeps initial occupations
and Floquet cycles. Analysis adds number moments and sector-averaged
two-time correlations to the parent's multiparity results.
"""
import numpy as np

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

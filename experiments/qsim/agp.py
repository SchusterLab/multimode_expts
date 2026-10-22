"""Measure all selected initial Fock states in one AGP population job.

One job fixes the dynamics and readout mode, then sweeps initial occupations
and Floquet cycles. Multiparity analysis and display come from the parent.
"""

import numpy as np

from experiments.qsim.fock_population import FockPopulationProgram, validate_occupations
from experiments.qsim.multiparity import MultiparityQsimExperiment
from itertools import product


class AGPExperiment(MultiparityQsimExperiment):
    """Occupation-by-cycle population scan, with ordinary multiparity analysis.

    ``cfg.expt.initial_occupations`` lists the initial Fock states, each in
    ``[M1, *swap_stors]`` order. ``cfg.expt.floquet_cycles`` is the shared
    inner sweep. The inherited driver runs a Program for each pair.

    ``data['occupations']`` records the state table; ``ypts`` holds its row
    indices and ``xpts`` holds cycles. Analyzed probabilities and means have
    shape ``(initial states, cycle points)``. This class does not calculate a
    correlation or an AGP estimate.
    """

    default_program = FockPopulationProgram
    def generate_fock_strings(self):
        _photon_sector = self.cfg.expt.photon_sector_number
        _number_of_modes = len(self.cfg.expt.swap_stors) + 1
        occupations = [_state for _state in product(np.arange(0, _photon_sector+1), _number_of_modes) if sum(_state) == _photon_sector]
        return occupations
        
        
    def sweep_axes(self):
        cycles = self.cfg.expt.floquet_cycles
        occupations = self.generate_fock_strings()
        for occupation in occupations:
            validate_occupations(occupation, self.cfg.expt.swap_stors)
        self.cfg.expt.swept_params = ['occupations', 'floquet_cycle']
        return [('occupations', occupations), ('floquet_cycle', cycles)]

    def shape_data(self, axes, points):
        data = super().shape_data(axes, points)
        data['occupations'] = np.asarray(axes[0][1], dtype=int)
        data['ypts'] = np.arange(len(data['occupations']))
        return data

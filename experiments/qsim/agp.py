"""Measure central-boson populations for a selected initial-state direction.

One job fixes the dynamics and readout mode, then sweeps initial occupations
and Floquet cycles. Multiparity analysis and display come from the parent.
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
    shape ``(initial states, cycle points)``. This class does not calculate a
    correlation or an AGP estimate.
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

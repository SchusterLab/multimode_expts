"""Measure the four rows of C[prepared number, measured multiparity label].

Use ``MultiParityConfusionMatrix`` as a CharacterizationRunner's ExptClass.
It prepares Fock 0, 1, 2, 3 in M1 using QsimProgram's existing prepulse path,
with no evolution between preparation and readout. Reset settings and shot
counts come from the caller, as for other Qsim experiments.

Set ``cfg.expt.multiparity_confusion_matrix`` to the numeric matrix, or to an
analyzed calibration Experiment for in-memory analysis, then call ``analyze()``.
"""

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.qsim_base import QsimExperiment, saved_readout_mode
from experiments.qsim.multiparity import MultiparityMixin


def check_multiparity_readout_paramter(expt_cfg):
        if saved_readout_mode(expt_cfg) != 'multiparity':
            raise ValueError("MultiParityConfusionMatrix requires readout='multiparity'")
        if (expt_cfg.get('cond_sec_phase', 90) != -90
                or expt_cfg.get('phase_second_pulse', 180) != 180):
            raise ValueError(
                'MultiParityConfusionMatrix requires cond_sec_phase=-90 '
                'and phase_second_pulse=180 for n = b0 + 2*b1'
            )
        if expt_cfg.get('multiparity_confusion_matrix') is not None:
            expt_cfg.multiparity_confusion_matrix = None
            print("multiparity_confusion_matrix has been hard reset to None")
        if not expt_cfg.get("prepulse", False) or not expt_cfg.get("postpulse", False):
            raise ValueError("multiparity readout requires prepulse and postpulses")
            
class MultiParityConfusionMatrixExperiment(MultiparityMixin, QsimExperiment):
    """One calibration job measuring Fock 0..3 with two parity readouts.

    ``analyze()`` stores a numeric 4x4 ``multiparity_confusion_matrix`` in
    ``self.data``. Row n contains P(measured label | prepared Fock n).
    """

    def acquire(self, progress=False, debug=False):
        # Runners and the queue worker replace cfg after construction. Apply
        # defaults here, before the driver asks the Program for its lane count.
        self.cfg.expt.swept_params = ['init_man_fock_state']
        self.cfg.expt.init_man_fock_states = ['0', '1', '2', '3']
        
        check_multiparity_readout_paramter(self.cfg.expt)
        return super().acquire(progress=progress, debug=debug)

    def shape_data(self, axes, points):
        data = super().shape_data(axes, points)
        data['xpts'] = np.asarray(data['xpts'], dtype=int)
        return data

    def analyze(self, data=None, **kwargs):
        """Build C from raw classifications, indexed by prepared Fock number."""
        if data is not None:
            self.data = data
        check_multiparity_readout_paramter(self.cfg.expt)
        prepared = np.asarray(self.data['xpts']).reshape(-1)
        if (len(self.data.get('ypts', [])) > 0
                or prepared.shape != (4,)
                or not np.array_equal(np.sort(prepared), np.arange(4))):
            raise ValueError(
                'MultiParityConfusionMatrix requires exactly one point '
                'for each prepared Fock number 0, 1, 2, 3 in xpts'
            )
        multiparity = self.analyze_multiparity()
        probabilities = np.stack(
            [multiparity[f'p_mod{n}'] for n in range(4)], axis=-1
        )
        if not np.isfinite(probabilities).all():
            raise ValueError('every calibration state must have nonempty readout shots')
        confusion = np.empty((4, 4))
        confusion[prepared.astype(int)] = probabilities
        self.data['multiparity_confusion_matrix'] = confusion
        self.data.pop('multiparity_confusion_matrix_covariance', None)
        return self.data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        fig, axis = plt.subplots(figsize=(4, 3))
        image = axis.imshow(data['multiparity_confusion_matrix'], vmin=0, vmax=1)
        for (row, col), value in np.ndenumerate(data['multiparity_confusion_matrix']):
            axis.text(col, row, f'{value:.2f}', ha='center', va='center',
                    color='black' if value > 0.5 else 'white', fontsize=8)
        axis.set(xlabel='Measured number', ylabel='Prepared number',
                 xticks=range(4), yticks=range(4))
        fig.colorbar(image, ax=axis, label='Probability')
        fig.tight_layout()
        plt.show()
        return fig

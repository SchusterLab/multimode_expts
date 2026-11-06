"""Calibrate each sequential multiparity readout step in one JOB.

Prepare Fock 0..3 in each target mode while the other modes retain the chosen
background occupations. Every setting runs the full sequential readout order.
The saved tensor has axes (prepared number, measured label, readout step).
Raw IQ and the actual per-setting retained shot counts remain in the same HDF5.
"""

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.fock_population import mode_name
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
    readout_layout,
)


class MultiParityConfusionMatrixSequentialExperiment(FockPopulationSequentialReadoutExperiment):
    """One prepared-occupation sweep; reuse the sequential population Program."""

    # TODO: Extend prepared states and matrix dimensions together with higher-parity pulses and decoding.
    def acquire(self, progress=False, debug=False):
        expt = self.cfg.expt
        expt.multiparity_confusion_matrix = None
        expt.multiparity_confusion_matrices = None
        expt.prepulse = False
        expt.floquet_cycle = 0
        expt.floquet_cycles = [0]
        expt.swept_params = ["occupations"]
        return super().acquire(progress=progress, debug=debug)

    def sweep_axes(self):
        expt = self.cfg.expt
        modes = readout_layout(self.cfg)["ro_stors"]
        physical_modes = [0, *expt.swap_stors]
        background = list(expt.get("background_occupations", [0] * len(physical_modes)))
        expt.background_occupations = background
        prepared = []
        for mode in modes:
            index = physical_modes.index(mode)
            for number in range(4):
                occupations = background.copy()
                occupations[index] = number
                prepared.append(occupations)
        return [("occupations", prepared)]

    def shape_data(self, axes, points):
        data = super().shape_data(axes, points)
        steps = len(data["ro_stors"])
        calibration_steps = []
        prepared_numbers = []
        for step in range(steps):
            for number in range(4):
                calibration_steps.append(step)
                prepared_numbers.append(number)
        data.update(xpts=np.arange(4 * steps),
                    prepared_occupations=np.asarray(axes[0][1], dtype=int),
                    calibration_steps=np.asarray(calibration_steps),
                    prepared_numbers=np.asarray(prepared_numbers),
                    background_occupations=np.asarray(self.cfg.expt.background_occupations))
        return data

    def analyze(self, data=None, **kwargs): 
        #TODO: 4 is hard-coded here; This should be resolved when the multiparity readout is extended to higher photon number
        self.analyze_multiparity(data)
        steps = len(self.data["ro_stors"])
        matrices = np.zeros((4, 4, steps))
        counts = np.zeros((4, steps), dtype=int)
        for step in range(steps):
            start = 4 * step
            stop = start + 4
            matrices[:, :, step] = self.data["p_mod"][start:stop, step]
            counts[:, step] = self.data["shot_counts"][start:stop, step]
        if not np.isfinite(matrices).all():
            raise ValueError("Every calibration setting needs accepted readout shots")
        self.data.update(multiparity_confusion_matrices=matrices,
                         calibration_shot_counts=counts)
        return self.data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        modes = data["ro_stors"]
        fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4), squeeze=False)
        for step, mode in enumerate(modes):
            axis = axes[0, step]
            matrix = data["multiparity_confusion_matrices"][:, :, step]
            image = axis.imshow(matrix, vmin=0, vmax=1)
            for (row, col), value in np.ndenumerate(matrix):
                axis.text(col, row, f"{value:.2f}", ha="center", va="center",
                          color="black" if value > .5 else "white")
            axis.set(title=f"{mode_name(mode)}, step {step}", xlabel="Measured label",
                     ylabel="Prepared number", xticks=range(matrix.shape[1]),
                     yticks=range(matrix.shape[0]))
            fig.colorbar(image, ax=axis, label="Probability")
        fig.tight_layout()
        return fig, axes

# -*- coding: utf-8 -*-
"""Zero-cycle encoder/decoder cross-return matrix ``M[j, i]``.

Spec sections 7.3/7.4: ``analyze(stage='orthogonality')`` on the god Experiment
becomes one aggregate Experiment owning its whole triple. The three methods
below are verbatim slices; ``analyze`` and ``display`` are new and only replace
the string dispatch with plain calls.

What it measures: how well the encoded states are distinguishable at zero
Floquet cycles. Rows are decoder occupations, columns are encoder occupations.
The raw matrix is deliberately not normalized -- ``offdiagonal_normalized_power``
carries the leakage figure ``|M_ji|^2/(|M_ii||M_jj|)``, and the display shows
raw and normalized side by side so a small diagonal cannot hide as good
orthogonality.

Usage -- ``analysis_notebooks/guan/MBR_analysis.py`` is the worked example::

    expt = MBROrthogonalityExperiment.from_job_files(paths)
    expt.analyze()
    expt.display()

The base class is still the god Experiment, which holds the loading layer
(``from_job_files``, ``_quadrature``) until it is extracted. Per spec 7.4 no new
aggregate base is invented ahead of the duplication that would justify it.
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)


class MBROrthogonalityExperiment(EncodingHamiltonianSpectroscopyExperiment):
    """Aggregate: the zero-cycle overlap matrix from one column batch."""

    @classmethod
    def reconstruct_orthogonality(cls,
                                  orthogonality_expts,
                                  occupations=None):
        """Reconstruct the zero-cycle encoder-to-decoder cross-return matrix.

        Rows are decoder occupations and columns are encoder occupations. Each
        encoder job contains outer rows ``(decoder, phi=0/90)`` and inner
        preparation phases ``theta=0/180``. With the QICK phase convention,
        ``M[j, i] = Q_0 - i Q_90``. The raw matrix is deliberately not divided
        by its zero-cycle values because those values are the diagnostic.
        """
        first_cfg = orthogonality_expts[0].cfg.expt
        swap_stors = [int(stor) for stor in first_cfg.swap_stors]
        decoder_order = [
            tuple(occupation)
            for occupation in first_cfg.orthogonality_decoder_occupations
        ]
        columns = {}
        for expt in orthogonality_expts:
            cfg = expt.cfg.expt
            encoder = tuple(cfg.spectroscopy_occupations)
            quadrature = np.asarray(
                cls._quadrature(expt), dtype=float
            ).reshape(len(decoder_order), 2)
            columns[encoder] = quadrature[:, 0] - 1j * quadrature[:, 1]

        occupation_order = decoder_order if occupations is None else [
            tuple(occupation) for occupation in occupations
        ]

        decoder_indices = [decoder_order.index(occ) for occ in occupation_order]
        matrix = np.column_stack([
            columns[occupation][decoder_indices]
            for occupation in occupation_order
        ]).astype(complex, copy=False)
        amplitude = np.abs(matrix)
        power = amplitude ** 2
        diagonal_amplitude = np.abs(np.diag(matrix))
        denominator = np.outer(diagonal_amplitude, diagonal_amplitude)
        normalized_power = power / denominator
        normalized_amplitude = matrix / np.sqrt(denominator)
        offdiagonal_normalized_power = normalized_power.copy()
        np.fill_diagonal(offdiagonal_normalized_power, 0.)
        column_leakage = np.sum(offdiagonal_normalized_power, axis=0)

        return AttrDict(dict(
            occupations=occupation_order,
            mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors],
            matrix=matrix,
            amplitude=amplitude,
            power=power,
            diagonal_amplitude=diagonal_amplitude,
            normalized_amplitude=normalized_amplitude,
            normalized_power=normalized_power,
            offdiagonal_normalized_power=offdiagonal_normalized_power,
            column_leakage=column_leakage,
            matrix_orientation="rows=decoder, columns=encoder",
        ))

    def display_orthogonality(self, 
                              data=None,
                              **kwargs):
        """Plot raw cross return and raw/normalized off-diagonal leakage."""
        data = self.data if data is None else data
        if "matrix" not in data:
            raise ValueError(
                "orthogonality display requires stage='orthogonality' data"
            )
        matrix = np.asarray(data.matrix, dtype=complex)
        labels = [str(tuple(occupation)) for occupation in data.occupations]
        size = len(labels)
        if matrix.shape != (size, size):
            raise ValueError("orthogonality matrix and labels have different sizes")

        raw_offdiagonal = np.abs(matrix).copy()
        np.fill_diagonal(raw_offdiagonal, 0.)
        panels = [
            (np.abs(matrix), r"raw $|M_{j i}|$ (diagonal contrast retained)"),
            (raw_offdiagonal, r"raw off-diagonal $|M_{j i}|$"),
            (
                np.asarray(data.offdiagonal_normalized_power, dtype=float),
                r"normalized off-diagonal $|M_{j i}|^2/(|M_{ii}||M_{jj}|)$",
            ),
        ]
        
        figsize = kwargs.get("figsize", (max(16, 1.35 * size + 10), 6))
        fig, axes = plt.subplots(
            1, 3, figsize=figsize,
            constrained_layout=True,
        )
        for axis, (values, title) in zip(axes, panels):
            image = axis.imshow(
                values, origin="upper", aspect="equal", cmap="magma", vmin=0.
            )
            axis.set_title(title)
            axis.set_xlabel("encoder occupation i")
            axis.set_ylabel("decoder occupation j")
            axis.set_xticks(np.arange(size))
            axis.set_yticks(np.arange(size))
            axis.set_xticklabels(labels, rotation=55, ha="right")
            axis.set_yticklabels(labels)
            fig.colorbar(image, ax=axis)
            if size <= 6:
                for row in range(size):
                    for column in range(size):
                        value = values[row, column]
                        text = "nan" if not np.isfinite(value) else f"{value:.3f}"
                        axis.text(
                            column, row, text,
                            ha="center", va="center", color="cyan", fontsize=8,
                        )

        finite_offdiagonal = np.asarray(
            data.offdiagonal_normalized_power, dtype=float).copy()
        np.fill_diagonal(finite_offdiagonal, np.nan)
        max_leakage = (
            float(np.nanmax(finite_offdiagonal))
            if np.any(np.isfinite(finite_offdiagonal)) else np.nan
        )
        fig.suptitle(
            "zero-cycle encoder/decoder cross return; "
            f"min diagonal |M|={np.min(data.diagonal_amplitude):.3f}; "
            f"max normalized off-diagonal power={max_leakage:.3g}"
        )
        return fig

    @staticmethod
    def orthogonality_batch(default_expt_cfg,
                            swap_stors,
                            occupations,
                            sync_cycles=10,
                            reps=300,
                            **kwargs):
        """
        Build one zero-cycle job for each encoder occupation.

        Within that job, measure every decoder occupation at analyzer phases
        0 and 90 degrees, each with preparation phases 0 and 180 degrees.
        
        ``decoder_analyzer_rows`` stores indices. specifically, modulo 2 should
        give the index for analyzer_phase, where as quotient by 2 gives
        which occupation index should be ran.
        """
        swap_stors = [int(stor) for stor in swap_stors]
        occupations = [list(occupation) for occupation in occupations]
        # decoder 0 at 0/90 deg, then decoder 1 at 0/90 deg, and so on.
        decoder_analyzer_rows = list(range(2 * len(occupations)))
        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            detunings=[0.] * len(swap_stors),
            scramble_sync_cycles=int(sync_cycles),
            floquet_cycle=0,
            floquet_hardware_loop=False,
            update_phases=False,
            palindrome_scramble=False,
            spectroscopy_phase_correction_mode=kwargs.get("correction_mode","final_analyzer"),
            final_analyzer_phase_per_cycle_deg=0.,
            orthogonality_decoder_occupations=deepcopy(occupations),
            orthogonality_analyzer_phases=[0., 90.],
            decoder_analyzer_rows=decoder_analyzer_rows,
            spectroscopy_prep_phases=[0., 180.],
            swept_params=[
                "decoder_analyzer_row",
                "spectroscopy_prep_phase",
            ],
        ))
        configs = [
            dict(spectroscopy_occupations=list(occupation))
            for occupation in occupations
        ]
        return AttrDict(dict(
            default_expt_cfg=defaults,
            configs=configs,
            occupations=deepcopy(occupations),
            points_per_job=4 * len(occupations),
            total_points=4 * len(occupations) ** 2,
        ))

    def analyze(self, data=None, occupations=None, **kwargs):
        """Reconstruct the cross-return matrix from the loaded columns.

        ``occupations`` optionally fixes the row/column order; it defaults to
        the order recorded in the jobs.
        """
        if data is not None:
            self.data = data
        self.data = self.reconstruct_orthogonality(self.batch_expts, occupations)
        return self.data

    def display(self, data=None, **kwargs):
        """Raw, raw off-diagonal, and normalized leakage panels."""
        if data is not None:
            self.data = data
        return self.display_orthogonality(self.data, **kwargs)

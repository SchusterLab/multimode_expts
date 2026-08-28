# -*- coding: utf-8 -*-
"""Raw short-time propagator matrices ``M_q[j, i]`` from encoded MBR jobs.

Spec sections 7.3/7.4: ``analyze(stage='propagator')`` on the god Experiment
becomes one aggregate Experiment that owns its whole triple. Both methods below
are verbatim slices; only :meth:`MBRPropagatorExperiment.analyze` is new, and it
replaces the string dispatch with a plain call.

Rows are decoder occupations, columns are encoder occupations. ``raw_matrices``
is what the quadratures give directly; ``matrices`` additionally undoes the
per-decoder phase correction that was applied at pulse time, so it is the
Kerr-preserving form.

The base class is still the god Experiment, which is where the loading layer
(``from_job_files``, ``_saved_parameters``, ``_quadrature``) lives until it is
extracted. Per spec 7.4 no new aggregate base is invented ahead of the
duplication that would justify it.

Usage -- see ``analysis_notebooks/guan/MBR_analysis.py`` for the worked example::

    expt = MBRPropagatorExperiment.from_job_files(paths)
    expt.analyze()
    expt.data.matrices          # (cycle, decoder, encoder)

No display yet: the god Experiment never had one for this stage, and inventing
one here would not be a move. That is the one part of the triple this class is
still missing.
"""
from copy import deepcopy

import numpy as np

from slab import AttrDict
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)


class MBRPropagatorExperiment(EncodingHamiltonianSpectroscopyExperiment):
    """Aggregate: raw propagator columns from one encoded-occupation batch."""

    @classmethod
    def reconstruct_propagator(cls,
                               propagator_expts,
                               occupations=None):
        """Reconstruct raw and Kerr-preserving ``M_q[j, i]`` matrices."""
        first_cfg = propagator_expts[0].cfg.expt
        swap_stors = [int(stor) for stor in first_cfg.swap_stors]
        cycles = [int(cycle) for cycle in first_cfg.propagator_cycles]
        decoder_order = [
            tuple(occupation)
            for occupation in first_cfg.propagator_occupations
        ]

        columns = {}
        for expt in propagator_expts:
            encoder = tuple(expt.cfg.expt.spectroscopy_occupations)
            quadrature = np.asarray(
                cls._quadrature(expt), dtype=float
            ).reshape(len(cycles), len(decoder_order), 2)
            columns[encoder] = (
                quadrature[:, :, 0] - 1j * quadrature[:, :, 1]
            )

        occupation_order = decoder_order if occupations is None else [
            tuple(occupation) for occupation in occupations
        ]
        decoder_indices = [
            decoder_order.index(occupation)
            for occupation in occupation_order
        ]
        raw_matrices = np.stack([
            columns[occupation][:, decoder_indices]
            for occupation in occupation_order
        ], axis=2).astype(complex, copy=False)
        phase_correction = np.asarray(
            first_cfg.propagator_decoder_phase_correction_deg,
            dtype=float,
        )[decoder_indices]
        matrices = raw_matrices * np.exp(
            -1j * np.deg2rad(
                np.asarray(cycles)[:, None, None]
                * phase_correction[None, :, None]
            )
        )

        return AttrDict(dict(
            cycles=np.asarray(cycles, dtype=int),
            occupations=occupation_order,
            mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors],
            raw_matrices=raw_matrices,
            matrices=matrices,
            decoder_phase_correction_deg=phase_correction,
            matrix_orientation="rows=decoder, columns=encoder",
        ))

    @staticmethod
    def propagator_batch(default_expt_cfg,
                         swap_stors,
                         occupations,
                         cycles,
                         phase_by_occupation,
                         sync_cycles=10,
                         reps=300):
        """Build one raw short-time propagator job per encoded occupation."""
        swap_stors = [int(stor) for stor in swap_stors]
        occupations = [list(occupation) for occupation in occupations]
        cycles = [int(cycle) for cycle in cycles]
        decoder_phase_correction = [
            float(phase_by_occupation[tuple(occupation)])
            for occupation in occupations
        ]

        # Every outer sweep value says exactly what is played:
        # [Floquet cycle, decoder occupation..., analyzer phase].
        cycle_decoder_analyzers = [
            [cycle, *decoder_occupation, analyzer_phase]
            for cycle in cycles
            for decoder_occupation in occupations
            for analyzer_phase in (0., 90.)
        ]

        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            detunings=[0.] * len(swap_stors),
            scramble_sync_cycles=int(sync_cycles),
            floquet_cycle=0,
            floquet_hardware_loop=False,
            update_phases=True,
            palindrome_scramble=False,
            spectroscopy_phase_correction_mode="final_analyzer",
            final_analyzer_phase_per_cycle_deg=0.,
            propagator_cycles=cycles,
            propagator_occupations=deepcopy(occupations),
            propagator_decoder_phase_correction_deg=(
                decoder_phase_correction
            ),
            cycle_decoder_analyzers=cycle_decoder_analyzers,
            spectroscopy_prep_phases=[0., 180.],
            swept_params=[
                "cycle_decoder_analyzer",
                "spectroscopy_prep_phase",
            ],
        ))
        configs = [
            dict(spectroscopy_occupations=list(occupation))
            for occupation in occupations
        ]
        points_per_job = 4 * len(cycles) * len(occupations)
        return AttrDict(dict(
            default_expt_cfg=defaults,
            configs=configs,
            cycles=cycles,
            occupations=deepcopy(occupations),
            points_per_job=points_per_job,
            total_points=points_per_job * len(occupations),
        ))

    def analyze(self, data=None, occupations=None, **kwargs):
        """Reconstruct the propagator matrices from the loaded jobs.

        ``occupations`` optionally fixes the row/column order; it defaults to
        the order recorded in the jobs.
        """
        if data is not None:
            self.data = data
        self.data = self.reconstruct_propagator(self.batch_expts, occupations)
        return self.data

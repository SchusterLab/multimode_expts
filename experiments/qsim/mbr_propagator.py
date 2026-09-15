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
from math import comb

import numpy as np
from scipy.linalg import eig as generalized_eig

from slab import AttrDict
from experiments.qsim.mbr_spectroscopy_program import (
    NPhotonHamiltonianSpectroscopyProgram,
)
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment


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

    def analyze(self, data=None, occupations=None, calibration=None,
                floquet_cycle_us=None, finite_difference_cycles=None,
                eigenphase_cycle=None):
        """Reconstruct the propagator matrices from the loaded jobs.

        ``occupations`` optionally fixes the row/column order; it defaults to
        the order recorded in the jobs.

        Passing ``calibration`` additionally runs the Hamiltonian tomography
        of :meth:`analyze_propagator_dynamics` and merges its results into
        ``self.data``, which is what the old ``analyze(stage='propagator',
        calibration=...)`` did. Without it this is reconstruction only.

        ``floquet_cycle_us`` defaults to the value resolved from the saved
        jobs; the other two knobs are passed straight through.
        """
        if data is not None:
            self.data = data
        self.data = self.reconstruct_propagator(self.batch_expts, occupations)
        if calibration is None:
            return self.data

        station = getattr(self, "_analysis_station", None)
        calibration = MBRPhaseCorrectionExperiment._calibration_data(
            calibration, station)
        if floquet_cycle_us is None:
            self.data.hardware = self._saved_parameters(
                self.batch_expts, station).hardware
            floquet_cycle_us = self.data.hardware.floquet_cycle_us
        self.data.update(self.analyze_propagator_dynamics(
            self.data, calibration, floquet_cycle_us,
            finite_difference_cycles=finite_difference_cycles,
            eigenphase_cycle=eigenphase_cycle,
            station=station))
        self.data.calibration = calibration
        self.data.floquet_cycle_us = float(floquet_cycle_us)
        return self.data


    @classmethod
    def analyze_propagator_dynamics(cls, 
                                    reconstruction, 
                                    calibration,
                                    floquet_cycle_us,
                                    finite_difference_cycles=None,
                                    eigenphase_cycle=None,
                                    station=None):
        """
        Get a short-time Hamiltonian and eigenfrequencies from full M_q.

        The method assumes <i|U(q)|j> = D_i U_q E_j. 
        There are two different methods to calculate the energy spectrum
        
        1. finite_difference

        - Calculates Hamiltonian using three different time stamps
        - H = i/2pi dU/dt \\approx i/2pi (-3 U(0) + 4 U(q)-U(2q)) / 2qT_floquet
        - This result is omitted when no [0, q, 2q] triple was acquired.
        
        2. eigen phase
        normalize endpoint contrast; generalized eigenvalues against the full
        same-batch M_0 remove fixed D and E from the spectrum.
        """
        
        calibration = MBRPhaseCorrectionExperiment._calibration_data(
            calibration, station)
        if calibration is None or "results" not in calibration:
            raise ValueError("analyzed calibration results are required")
        floquet_cycle_us = float(floquet_cycle_us)
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0.:
            raise ValueError("floquet_cycle_us must be finite and positive")

        occupations = []
        for raw_occupation in reconstruction.occupations:
            values = np.asarray(raw_occupation, dtype=float)
            if values.ndim != 1 or np.any(values < 0) or not np.all(values == np.round(values)):
                raise ValueError("occupations must be nonnegative integer states")
            occupations.append(tuple(int(value) for value in values))
        if not occupations:
            raise ValueError("occupations cannot be empty")
        photon_number = sum(occupations[0])
        mode_count = len(occupations[0])
        for occupation in occupations:
            if len(occupation) != mode_count or sum(occupation) != photon_number:
                raise ValueError("occupations must belong to one fixed-N sector")
        full_dimension = comb(photon_number + mode_count - 1, photon_number)
        if len(occupations) != full_dimension or len(set(occupations)) != full_dimension:
            raise ValueError("eigenanalysis requires the complete fixed-N basis")

        if "mode_labels" in calibration:
            if list(calibration.mode_labels) != list(reconstruction.mode_labels):
                raise ValueError("calibration and propagator mode labels differ")
        if "hardware" in calibration:
            calibration_cycle_us = calibration.hardware.get("floquet_cycle_us", None)
            if calibration_cycle_us is not None:
                if not np.isclose(float(calibration_cycle_us), floquet_cycle_us):
                    raise ValueError("calibration and propagator cycle times differ")

        cycles = np.asarray(reconstruction.cycles, dtype=int)
        if len(np.unique(cycles)) != len(cycles):
            raise ValueError("propagator cycles must be unique")
        cycle_index = {}
        for index, cycle in enumerate(cycles):
            cycle_index[int(cycle)] = index
        if 0 not in cycle_index:
            raise ValueError("propagator data needs q=0")

        dimension = len(occupations)
        matrices = np.asarray(reconstruction.matrices, dtype=complex)
        if matrices.shape != (len(cycles), dimension, dimension):
            raise ValueError("propagator matrices do not match the full basis")
        M0 = matrices[cycle_index[0]]
        condition_number = float(np.linalg.cond(M0))
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise ValueError("q=0 matrix is singular or too ill-conditioned")

        q0_by_occupation = {}
        for result in calibration.results:
            occupation = tuple(result.occupation)
            if occupation in q0_by_occupation:
                raise ValueError(f"duplicate calibration result for {occupation}")
            zero_indices = np.flatnonzero(np.asarray(result.physical_cycles) == 0)
            if len(zero_indices) != 1:
                raise ValueError(f"{occupation} calibration needs exactly one q=0 sample")
            returns = np.asarray(result.complex_return, dtype=complex)
            q0_by_occupation[occupation] = returns[zero_indices[0]]

        self_returns = []
        for occupation in occupations:
            if occupation not in q0_by_occupation:
                raise ValueError(f"calibration is missing {occupation}")
            value = complex(q0_by_occupation[occupation])
            if not np.isfinite(value) or abs(value) <= 1e-10:
                raise ValueError(f"invalid q=0 calibration return for {occupation}")
            self_returns.append(value)
        
            
        
        self_returns = np.asarray(self_returns)
        endpoint_factors = np.sqrt(self_returns)
        endpoint_denominator = endpoint_factors[:, None] * endpoint_factors[None, :]
        endpoint_normalized_matrices = matrices / endpoint_denominator[None, :, :]

        step_cycles = None
        if finite_difference_cycles is None:
            for cycle in sorted(cycle_index):
                if cycle > 0 and cycle % 2 == 0 and 2 * cycle in cycle_index:
                    step_cycles = cycle
                    break
            if step_cycles is not None:
                finite_difference_cycles = [0, step_cycles, 2 * step_cycles]
        else:
            finite_difference_cycles = [int(cycle) for cycle in finite_difference_cycles]
            if len(finite_difference_cycles) != 3:
                raise ValueError("finite_difference_cycles must be [0, s, 2*s]")
            step_cycles = finite_difference_cycles[1]

        finite_difference = None
        if finite_difference_cycles is not None:
            expected_cycles = [0, step_cycles, 2 * step_cycles]
            if (step_cycles <= 0 or step_cycles % 2 != 0
                    or finite_difference_cycles != expected_cycles):
                raise ValueError(
                    "finite_difference_cycles must be even [0, s, 2*s]"
                )
            for cycle in expected_cycles:
                if cycle not in cycle_index:
                    raise ValueError(f"propagator matrix q={cycle} is missing")

            M_step = matrices[cycle_index[step_cycles]]
            M_twostep = matrices[cycle_index[2 * step_cycles]]
            step_time_us = step_cycles * floquet_cycle_us
            derivative = (
                -3. * M0 + 4. * M_step - M_twostep
            ) / (2. * step_time_us)
            measured_generator_MHz = 1j * derivative / (2. * np.pi)
            effective_hamiltonian_MHz = np.linalg.solve(
                M0, measured_generator_MHz
            )
            fd_values = generalized_eig(
                measured_generator_MHz, M0, right=False
            )
            if not np.all(np.isfinite(fd_values)):
                raise ValueError(
                    "finite-difference eigenfrequencies are not finite"
                )
            fd_values = fd_values[np.argsort(fd_values.real)]

            predicted_twostep = M_step @ np.linalg.solve(M0, M_step)
            semigroup_residual = np.linalg.norm(
                M_twostep - predicted_twostep
            )
            semigroup_residual /= max(np.linalg.norm(M_twostep), 1e-10)
            finite_difference = AttrDict(dict(
                cycles=np.asarray(expected_cycles),
                step_cycles=step_cycles,
                step_time_us=step_time_us,
                derivative_matrix_per_us=derivative,
                access_dressed_generator_MHz=measured_generator_MHz,
                endpoint_normalized_generator_MHz=(
                    measured_generator_MHz / endpoint_denominator
                ),
                effective_hamiltonian_MHz=effective_hamiltonian_MHz,
                hamiltonian_gauge="M0^-1 K = E^-1 H E",
                complex_eigenfrequencies_MHz=fd_values,
                eigenfrequencies_MHz=fd_values.real,
                semigroup_relative_residual=float(semigroup_residual),
            ))

        if eigenphase_cycle is None:
            if step_cycles is not None:
                eigenphase_cycle = step_cycles
            else:
                even_cycles = sorted(
                    cycle for cycle in cycle_index
                    if cycle > 0 and cycle % 2 == 0
                )
                if not even_cycles:
                    raise ValueError(
                        "need an available positive even eigenphase cycle"
                    )
                eigenphase_cycle = even_cycles[0]
        eigenphase_cycle = int(eigenphase_cycle)
        if eigenphase_cycle <= 0 or eigenphase_cycle % 2 != 0 or eigenphase_cycle not in cycle_index:
            raise ValueError("eigenphase_cycle must be an available positive even cycle")
        eigenphase_time_us = eigenphase_cycle * floquet_cycle_us
        eigenphase_values = generalized_eig(
            matrices[cycle_index[eigenphase_cycle]], M0, right=False
        )
        if not np.all(np.isfinite(eigenphase_values)):
            raise ValueError("eigenphase eigenvalues are not finite")
        eigenphase_frequencies = -np.angle(eigenphase_values) / (2. * np.pi * eigenphase_time_us)
        order = np.argsort(eigenphase_frequencies)
        eigenphase_values = eigenphase_values[order]
        eigenphase_frequencies = eigenphase_frequencies[order]

        normalized_M0 = endpoint_normalized_matrices[cycle_index[0]]
        identity_residual = np.linalg.norm(normalized_M0 - np.eye(dimension))
        identity_residual /= np.sqrt(dimension)
        calibration_mismatch = np.linalg.norm(np.diag(M0) - self_returns)
        calibration_mismatch /= max(np.linalg.norm(self_returns), 1e-10)
        return AttrDict(dict(
            calibration_self_returns=self_returns,
            endpoint_factors=endpoint_factors,
            endpoint_denominator=endpoint_denominator,
            endpoint_normalized_matrices=endpoint_normalized_matrices,
            zero_cycle_condition_number=condition_number,
            endpoint_normalized_zero_cycle_identity_residual=float(identity_residual),
            calibration_diagonal_relative_mismatch=float(calibration_mismatch),
            finite_difference=finite_difference,
            eigenphase=AttrDict(dict(
                cycle=eigenphase_cycle,
                generalized_eigenvalues=eigenphase_values,
                pole_radii=np.abs(eigenphase_values),
                eigenfrequencies_MHz=eigenphase_frequencies,
                alias_period_MHz=1. / eigenphase_time_us,
                single_cycle_branch_ambiguous=(eigenphase_cycle != 1),
            )),
        ))


class EncodingPropagatorProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure one raw column of the short-time propagator."""

    def initialize(self):
        ecfg = self.cfg.expt
        cycle_decoder_analyzer = list(ecfg.cycle_decoder_analyzer)
        decoder_occupation = list(cycle_decoder_analyzer[1:-1])

        ecfg.floquet_cycle = int(cycle_decoder_analyzer[0])
        ecfg.spectroscopy_analyzer_phase = float(
            cycle_decoder_analyzer[-1])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.
        ecfg.spectroscopy_final_occupations = decoder_occupation

        super().initialize()

    def _get_inverse_pulses(self, _):
        # The parent body requests inverse(encoder); decode the selected row.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)

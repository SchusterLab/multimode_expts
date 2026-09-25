# -*- coding: utf-8 -*-
"""The four MBR aggregate classes from before the redesign.

``MBRPhaseCorrectionExperiment``, ``MBROrthogonalityExperiment``,
``MBRPropagatorExperiment`` and ``MBRSpectrumExperiment`` moved here without
changes on 2026-09-24 (``docs/qsim/mbr_redesign.md``, section 2), so that the
new classes can use these names. Code not yet ported imports them from here.
Each class keeps the docstring of the module it came from, as a comment above
it. Their pulse programs did not move.

Delete this module when nothing imports it (redesign step 6).
"""
import inspect
from copy import deepcopy
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig as generalized_eig
from slab import AttrDict

from experiments.qsim.dark_base import readout_lane_count
from experiments.qsim.deprecated.encoding_spectroscopy import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.deprecated.mbr_propagator import EncodingPropagatorProgram
from experiments.qsim.deprecated.mbr_nphoton_program import NPhotonHamiltonianSpectroscopyProgram
from experiments.qsim.utils import flatten_exp_lists
from fitting.qsim import level_statistics as level_statistics_analysis
from fitting.qsim import matrix_pencil as matrix_pencil_analysis
from fitting.qsim import mbr_spectrum as mbr_spectrum_analysis
from fitting.qsim.mbr_reconstruction import (
    subsample_spectroscopy_shots,
)


# ============================================================================
# MBRPhaseCorrectionExperiment (was experiments/qsim/mbr_phase_correction.py)
#
# Closed-cycle phase calibration and the analyzer correction built from it.
#
# Spec sections 7.3/7.4: ``analyze(stage='calibration')`` on the god Experiment
# becomes one aggregate Experiment owning its whole triple. All eight methods
# below are verbatim slices apart from two edits noted at the bottom of this
# docstring; ``analyze`` and ``display`` are new and replace the string dispatch
# with plain calls.
#
# What it measures. Each occupation is driven for a range of *entire* Floquet
# cycles at analyzer phases phi=0 and phi=90. Those two give the complex return
# ``Q_0 - i Q_90`` per cycle count; its phase slope is the accumulated phase per
# cycle. Closed pairs sit two physical cycles apart, so the slope is determined
# only modulo 180 deg/cycle -- picking a representative is what ``cycle_branches``
# does, and the unwrapping itself lives in ``fitting/qsim/mbr_phase.py``.
#
# What it produces. ``phase_mod180`` per occupation, and via
# :meth:`phase_correction_from_calibration` the ``phase_by_occupation`` map the
# spectroscopy pulse program subtracts from its final analyzer. That makes this
# class an input to the spectrum stage, which is the one real dependency between
# two stage Experiments.
#
# Usage -- ``analysis_notebooks/guan/MBR_analysis.py`` is the worked example::
#
#     calibration = MBRPhaseCorrectionExperiment.from_job_files(paths)
#     calibration.analyze()
#     calibration.display()
#     correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
#         calibration, cycle_branches={(2, 0, 0): 1})
#
# The base class is still the god Experiment, which holds the loading layer
# (``from_job_files``, ``_saved_parameters``, ``_quadrature``) and the
# ``fitting/qsim/mbr_phase.py`` aliases until they are extracted. Per spec 7.4 no
# new aggregate base is invented ahead of the duplication that would justify it.
#
# Two edits to otherwise verbatim bodies, both re-addressing a name the move
# invalidated, neither changing behaviour:
#
# - ``_calibration_data`` called ``cls.analyze(calibration, stage='calibration')``.
#   This class's ``analyze`` has no ``stage``, so it is now ``cls.analyze(...)``.
# - ``display_calibration_results`` hard-coded
#   ``EncodingHamiltonianSpectroscopyExperiment.display_cycle_phase``, which no
#   longer has that method. Now names this class.
#
# Both are declared in ``tests/test_mbr_stage_split.py`` so the AST pin still
# covers every other statement.
# ============================================================================

class MBRPhaseCorrectionExperiment(EncodingHamiltonianSpectroscopyExperiment):
    """Aggregate: phase per entire Floquet cycle from one calibration batch."""

    @classmethod
    def _calibration_data(cls, calibration, station=None):
        if calibration is None:
            return None
        from pathlib import Path
        if isinstance(calibration, (str, Path, list, tuple)):
            calibration = cls.from_job_files(calibration, station=station)
        if hasattr(calibration, "data"):
            if "phase_mod180" not in calibration.data:
                if hasattr(calibration, "batch_expts"):
                    cls.analyze(calibration)
                else:
                    raise ValueError("calibration must contain the aggregated calibration jobs")
            return calibration.data
        return AttrDict(calibration)

    @classmethod
    def phase_correction_from_calibration(cls, 
                                          calibration, 
                                          cycle_branches=0, 
                                          second_branch=False,
                                          station=None):
        """
        Prepare the phase calibration list, which is returned as `phase_by_occupation` 
        by `build_phase_correction` method.
        """
        
        calibration = cls._calibration_data(calibration, station=station)
        if calibration is None:
            raise ValueError("calibration is required")
        if "hardware" not in calibration:
            raise ValueError("calibration hardware is unavailable; analyze the calibration experiment first")
        branches = cls._cycle_branches(calibration.occupations, cycle_branches)
        if second_branch:
            if np.any(branches):
                raise ValueError("use either cycle_branches or second_branch, not both")
            branches += 1
        return cls.build_phase_correction(calibration.occupations, calibration.phase_mod180, branches, calibration.hardware.physical_kerr_MHz, calibration.hardware.floquet_cycle_us)

    @classmethod
    def analyze_cycle_phase(cls, #batch_expt is plugged in
                            phi0_expts, 
                            phi90_expts, 
                            occupation, 
                            cycle_pairs, 
                            radius_fraction=0.1,
                            unwrap_mode="pair"):
        """
        The method recieves phi0 and phi90 experiments to
            1. reconstruct Q0-iQ90
            2. Unwrap the phase using np.unwrap
            2. fit phase per physical entire cycle.
        
        Return: AttrDict containing
            - occupation: tuple(occupation), #state string tuple
            - physical_cycles: physical_cycles, #floquet cycles
            - complex_return: complex_return,
            - relative_return: relative_return,
            - return_phase: phase,
            - phase_fit: phase_fit,
            - fnames: fnames,
            - phase_per_cycle: parameters[0],
            - phase_error: np.sqrt(covariance[0, 0])}
        """
        if isinstance(phi0_expts, (list, tuple)):
            phi0_expts = list(phi0_expts)
        else:
            phi0_expts = [phi0_expts]
        if isinstance(phi90_expts, (list, tuple)):
            phi90_expts = list(phi90_expts)
        else:
            phi90_expts = [phi90_expts]
        if len(phi0_expts) != len(phi90_expts):
            raise ValueError("phi=0 and phi=90 repeat counts differ")

        if unwrap_mode not in ("pair", "odd_guide"):
            raise ValueError("unwrap_mode must be 'pair' or 'odd_guide'")
        cycle_pairs = np.asarray(cycle_pairs)
        complex_returns = []
        fnames = []

        for expt_phi0, expt_phi90 in zip(phi0_expts, phi90_expts):
            if not np.array_equal(expt_phi0.data["ypts"], cycle_pairs):
                raise ValueError("saved cycle-pair sweep changed")
            if not np.array_equal(expt_phi90.data["ypts"], cycle_pairs):
                raise ValueError("saved cycle-pair sweep changed")
            if not np.allclose(expt_phi0.data["xpts"], [0., 180.]):
                raise ValueError("saved preparation phases changed")
            if not np.allclose(expt_phi90.data["xpts"], [0., 180.]):
                raise ValueError("saved preparation phases changed")
            complex_returns.append(cls._quadrature(expt_phi0) - 1j * cls._quadrature(expt_phi90))
            for expt in [expt_phi0, expt_phi90]:
                if hasattr(expt, "fname"):
                    fname = str(expt.fname).replace("\\", "/").split("/")[-1]
                    fnames.append(fname)

        complex_returns = np.asarray(complex_returns)
        # Rows are repeated jobs; average the same cycle point across repeats.
        complex_return = complex_returns.mean(axis=0)
        magnitude = np.abs(complex_return)
        positive = magnitude[np.isfinite(magnitude) & (magnitude > 0.)]
        radius_floor = 0.
        if len(positive):
            radius_floor = radius_fraction * np.median(positive)
        valid_mask = np.isfinite(complex_return) & (magnitude > radius_floor)
        if unwrap_mode == "odd_guide":
            physical_cycles = cycle_pairs
            closed_mask = physical_cycles % 2 == 0
        else:
            physical_cycles = 2 * cycle_pairs
            closed_mask = np.ones(len(cycle_pairs), dtype=bool)
        fit_mask = valid_mask & closed_mask
        if np.count_nonzero(fit_mask) < 3:
            raise RuntimeError(f"{tuple(occupation)} has too few valid IQ points")

        phase = cls._unwrap_cycle_phase(complex_return, physical_cycles, valid_mask, closed_mask)
        parameters, covariance = np.polyfit(physical_cycles[fit_mask], 
                                            phase[fit_mask], 
                                            1, 
                                            cov=True)
        phase_fit = parameters[0] * physical_cycles + parameters[1]
        relative_return = magnitude.copy()
        if magnitude[0] > 1e-12:
            relative_return = magnitude / magnitude[0]
        return AttrDict(dict(
            occupation=tuple(occupation),
            physical_cycles=physical_cycles,
            complex_return=complex_return,
            relative_return=relative_return,
            return_phase=phase,
            phase_fit=phase_fit,
            closed_mask=closed_mask,
            unwrap_mode=unwrap_mode,
            fnames=fnames,
            phase_per_cycle=parameters[0],
            phase_error=np.sqrt(covariance[0, 0]),
        ))

    @staticmethod
    def display_cycle_phase(result, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[4., 1.])
        iq_axis = fig.add_subplot(grid[0, 0])
        relative_axis = fig.add_subplot(grid[1, 0])
        phase_axis = fig.add_subplot(grid[:, 1])

        iq_axis.plot(result.complex_return.real, result.complex_return.imag, "o--", color="black", linewidth=1.2, markersize=4, label="raw return")
        points = iq_axis.scatter(result.complex_return.real, result.complex_return.imag, c=result.physical_cycles, cmap="viridis", s=28, zorder=3)
        if result.get("unwrap_mode", "pair") == "odd_guide":
            iq_axis.scatter(result.complex_return.real[result.closed_mask], result.complex_return.imag[result.closed_mask], color="tab:red", s=30, zorder=4, label="closed pair")
        iq_limit = 1.15 * np.nanmax(np.abs(result.complex_return))
        if iq_limit <= 0.:
            iq_limit = 1.
        iq_axis.axhline(0., color="0.85")
        iq_axis.axvline(0., color="0.85")
        iq_axis.set(xlim=(-iq_limit, iq_limit), ylim=(-iq_limit, iq_limit), xlabel=r"$Q_0$", ylabel=r"$-Q_{90}$", title="raw complex return")
        iq_axis.set_aspect("equal", adjustable="box")
        iq_axis.legend()
        fig.colorbar(points, ax=iq_axis, label="number of physical entire Floquet cycles")

        relative_axis.plot(result.physical_cycles, result.relative_return, "o-")
        if result.get("unwrap_mode", "pair") == "odd_guide":
            relative_axis.plot(result.physical_cycles[result.closed_mask], result.relative_return[result.closed_mask], "o", color="tab:red")
        relative_axis.axhline(1., color="0.7")
        relative_axis.set(xlabel="number of physical entire Floquet cycles", ylabel=r"$|A|/|A(0)|$", title="relative return")

        phase_axis.plot(result.physical_cycles, result.return_phase, "o", label="measured")
        if result.get("unwrap_mode", "pair") == "odd_guide":
            phase_axis.plot(result.physical_cycles[result.closed_mask], result.return_phase[result.closed_mask], "o", color="tab:red", label="closed pair (fit)")
        phase_axis.plot(result.physical_cycles, result.phase_fit, label="fit")
        phase_axis.set(xlabel="number of physical entire Floquet cycles", ylabel="return phase (deg)")
        phase_axis.legend()

        title = f"{result.occupation}: {result.phase_per_cycle:.4f} +/- {result.phase_error:.4f} deg / cycle"
        if result.fnames:
            title += "\n" + "\n".join(result.fnames)
        fig.suptitle(title)
        return fig

    @staticmethod
    def display_calibration_results(calibration, ncols=None):
        results = calibration.results
        if len(results) == 0:
            raise ValueError("calibration results cannot be empty")
        if ncols is None:
            nrows = max(1, int(np.floor(np.sqrt(len(results)))))
            ncols = int(np.ceil(len(results) / nrows))
        elif not isinstance(ncols, (int, np.integer)) or ncols < 1:
            raise ValueError("ncols must be a positive integer")
        else:
            ncols = min(int(ncols), len(results))
            nrows = int(np.ceil(len(results) / ncols))

        fig = plt.figure(figsize=(7 * ncols, 4 * nrows), constrained_layout=True)
        subfigures = fig.subfigures(nrows, ncols, squeeze=False)
        for result, subfigure in zip(results, subfigures.flat):
            MBRPhaseCorrectionExperiment.display_cycle_phase(result, subfigure)
        for subfigure in subfigures.flat[len(results):]:
            subfigure.set_visible(False)
        return fig

    @staticmethod
    def display_calibration_summary(calibration):
        rows = np.arange(len(calibration.occupations))
        labels = [str(occupation) for occupation in calibration.occupations]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(rows) + 2)), constrained_layout=True)
        ax.errorbar(calibration.phase_mod180, rows, xerr=calibration.phase_error, fmt="o")
        ax.axvline(0., color="0.7")
        ax.set(xlabel="measured phase mod 180 (deg / entire cycle)", ylabel="occupation", title="entire-cycle calibration summary")
        ax.set_yticks(rows)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        return fig

    @classmethod
    def analyze_calibration(cls, 
                            expts, 
                            occupations=None, 
                            cycle_pairs=None, 
                            repeats=None):
        """
        The method first groups input experiments in a pair of encoding/decoding
        calibration with phi 0 and 90. For each pair, extract phase accumulation
        using `analyze_cycle_phase` and return the collective dictionary in the following form.
        
        Return: Attrdict containing
            - "occupations": occupation_order,
            - "results": results,
            - "phase_mod180": np.asarray([result.phase_per_cycle for result in results]),
            - "phase_error": np.asarray([result.phase_error for result in results])
        
        """
        
        if not expts:
            raise ValueError("calibration expts cannot be empty")
        grouped = {}
        for expt in expts:
            cfg = expt.cfg.expt
            occupation = tuple(cfg.spectroscopy_occupations)
            phi = cfg.spectroscopy_analyzer_phase
            if phi not in (0., 90.):
                raise ValueError(f"{occupation} has analyzer phase {phi}; expected 0 or 90")
            if occupation not in grouped:
                grouped[occupation] = {0.: [], 90.: []}
            grouped[occupation][phi].append(expt)

        if occupations is None:
            occupation_order = list(grouped)
        else:
            occupation_order = [tuple(occupation) for occupation in occupations]
        if len(occupation_order) != len(grouped) or set(occupation_order) != set(grouped):
            raise ValueError("calibration occupations do not match the saved configs")
        results = []
        for occupation in occupation_order:
            phi0_expts = grouped[occupation][0.]
            phi90_expts = grouped[occupation][90.]
            if not phi0_expts or len(phi0_expts) != len(phi90_expts):
                raise ValueError(f"{occupation} needs the same nonzero number of phi=0 and phi=90 jobs")
            if repeats is not None and len(phi0_expts) != repeats:
                raise ValueError(f"{occupation} has {len(phi0_expts)} repeats; expected {repeats}")
            saved_cycle_pairs = np.asarray(phi0_expts[0].cfg.expt.n_cycle_pairs)
            if cycle_pairs is not None and not np.array_equal(saved_cycle_pairs, cycle_pairs):
                raise ValueError(f"{occupation}: calibration cycles do not match the requested cycles")
            saved_unwrap_mode = phi0_expts[0].cfg.expt.get("phase_unwrap_mode", "pair")
            if saved_unwrap_mode == "odd_guide":
                saved_cycle_counts = np.asarray(phi0_expts[0].cfg.expt.n_physical_cycles)
            else:
                saved_cycle_counts = saved_cycle_pairs
            for expt in phi0_expts + phi90_expts:
                if not np.array_equal(expt.cfg.expt.n_cycle_pairs, saved_cycle_pairs):
                    raise ValueError(f"{occupation}: calibration configs use different cycles")
                if expt.cfg.expt.get("phase_unwrap_mode", "pair") != saved_unwrap_mode:
                    raise ValueError(f"{occupation}: calibration configs use different unwrap modes")
                if saved_unwrap_mode == "odd_guide" and not np.array_equal(expt.cfg.expt.n_physical_cycles, saved_cycle_counts):
                    raise ValueError(f"{occupation}: calibration configs use different physical cycles")
            results.append(cls.analyze_cycle_phase(phi0_expts,
                                                   phi90_expts,
                                                   occupation,
                                                   saved_cycle_counts,
                                                   unwrap_mode=saved_unwrap_mode,))
        return AttrDict(dict(
            occupations=occupation_order,
            results=results,
            phase_mod180=np.asarray([result.phase_per_cycle for result in results]),
            phase_error=np.asarray([result.phase_error for result in results]),
        ))

    @staticmethod
    def calibration_batch(default_expt_cfg, 
                          swap_stors, 
                          occupations, 
                          cycle_pairs,
                          sync_cycles=10, 
                          repeats=1, 
                          reps=1500,
                          unwrap_mode="pair"):
        """
        Returns dictionary of 
            - default_expt_cfg
            - list of config to be overrided in each job
            - program selected for diagonal or interleaved off-diagonal acquisition
            - repeats (usually 1)
        The list of config is then used to make and batch jobs in a chunk.
        For now, other paramters such as `update_phase`, `palindrome_scramble`, 
        `spectroscopy_prep_phases`, `floquet_hardware_loop`, `swept_params`
        are fixed.
        In a nearest term, the program should migrate to using `floquet_hardware_loop`
        
        The actual batch is done by plugging the output to the BatchRunner.
        Example:
            calibration_batch = EncSpec.calibration_batch()
            calibration_expt = calibration_runner.execute(calibration_batch.configs)
        """
        if unwrap_mode not in ("pair", "odd_guide"):
            raise ValueError("unwrap_mode must be 'pair' or 'odd_guide'")

        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=reps, 
            storage_reset=swap_stors, 
            swap_stors=swap_stors,
            n_cycle_pairs=cycle_pairs.tolist(),
            scramble_sync_cycles=sync_cycles,
            
            floquet_hardware_loop=False,
            detunings=[0.] * len(swap_stors), #detuning must be 0 for the calibration
            update_phases=True, 
            palindrome_scramble=False, 
            phase_unwrap_mode=unwrap_mode,
            spectroscopy_prep_phases=[0., 180.],
        ))
        if unwrap_mode == "odd_guide":
            physical_cycles = []
            for cycle_pair in cycle_pairs.tolist():
                physical_cycles.extend([2 * cycle_pair, 2 * cycle_pair + 1])
            defaults.update(dict(
                n_physical_cycles=physical_cycles,
                swept_params=["n_physical_cycle", "spectroscopy_prep_phase"],
            ))
        else:
            defaults.update(dict(
                swept_params=["n_cycle_pair", "spectroscopy_prep_phase"],
            ))
        configs = [
            dict(spectroscopy_occupations=occupation, 
                 spectroscopy_analyzer_phase=phi,
                 final_analyzer_phase_per_cycle_deg=0.)
            for occupation in occupations for _ in range(repeats) for phi in [0., 90.]
        ]
        return AttrDict(dict(default_expt_cfg=defaults, 
                             configs=configs, 
                             repeats=repeats))

    def analyze(self,
                data=None,
                occupations=None,
                cycle_pairs=None,
                repeats=None):
        """Fit the phase per entire cycle for every calibrated occupation.

        Attaches the saved hardware parameters and mode labels, because
        :meth:`phase_correction_from_calibration` needs the Kerr rate and the
        Floquet cycle length that go with this data.
        """
        if not hasattr(self, "batch_expts"):
            return super().analyze(data=data)
        if data is not None:
            self.data = data
        self.data = self.analyze_calibration(
            self.batch_expts, occupations, cycle_pairs, repeats=repeats)
        saved = self._saved_parameters(
            self.batch_expts, getattr(self, "_analysis_station", None))
        self.data.hardware = saved.hardware
        self.data.mode_labels = saved.mode_labels
        return self.data

    def display(self, data=None, ncols=None):
        """Per-occupation IQ/phase fits, then the phase-per-cycle summary."""
        if not hasattr(self, "batch_expts"):
            return super().display(data=data)
        if data is not None:
            self.data = data
        if "results" not in self.data:
            self.data = self.analyze_calibration(self.batch_expts)
        self.display_calibration_results(self.data, ncols)
        return self.display_calibration_summary(self.data)


# ============================================================================
# MBROrthogonalityExperiment (was experiments/qsim/mbr_orthogonality.py)
#
# Zero-cycle encoder/decoder cross-return matrix ``M[j, i]``.
#
# Spec sections 7.3/7.4: ``analyze(stage='orthogonality')`` on the god Experiment
# becomes one aggregate Experiment owning its whole triple. The three methods
# below are verbatim slices; ``analyze`` and ``display`` are new and only replace
# the string dispatch with plain calls.
#
# What it measures: how well the encoded states are distinguishable at zero
# Floquet cycles. Rows are decoder occupations, columns are encoder occupations.
# The raw matrix is deliberately not normalized -- ``offdiagonal_normalized_power``
# carries the leakage figure ``|M_ji|^2/(|M_ii||M_jj|)``, and the display shows
# raw and normalized side by side so a small diagonal cannot hide as good
# orthogonality.
#
# Usage -- ``analysis_notebooks/guan/MBR_analysis.py`` is the worked example::
#
#     expt = MBROrthogonalityExperiment.from_job_files(paths)
#     expt.analyze()
#     expt.display()
#
# The base class is still the god Experiment, which holds the loading layer
# (``from_job_files``, ``_quadrature``) until it is extracted. Per spec 7.4 no new
# aggregate base is invented ahead of the duplication that would justify it.
# ============================================================================

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

    def display_orthogonality(self, data=None, figsize=None):
        """Plot raw cross return and raw/normalized off-diagonal leakage.

        ``figsize`` defaults to a width that grows with the matrix, so a
        larger basis stays legible.
        """
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
        
        if figsize is None:
            figsize = (max(16, 1.35 * size + 10), 6)
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
                            correction_mode="final_analyzer"):
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
            spectroscopy_phase_correction_mode=correction_mode,
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

    def analyze(self, data=None, occupations=None):
        """Reconstruct the cross-return matrix from the loaded columns.

        ``occupations`` optionally fixes the row/column order; it defaults to
        the order recorded in the jobs.
        """
        if not hasattr(self, "batch_expts"):
            return super().analyze(data=data)
        if data is not None:
            self.data = data
        self.data = self.reconstruct_orthogonality(self.batch_expts, occupations)
        return self.data

    def display(self, data=None, figsize=None):
        """Raw, raw off-diagonal, and normalized leakage panels."""
        if not hasattr(self, "batch_expts"):
            return super().display(data=data)
        if data is not None:
            self.data = data
        return self.display_orthogonality(self.data, figsize=figsize)


# ============================================================================
# MBRPropagatorExperiment (was experiments/qsim/mbr_propagator.py)
#
# Raw short-time propagator matrices ``M_q[j, i]`` from encoded MBR jobs.
#
# Spec sections 7.3/7.4: ``analyze(stage='propagator')`` on the god Experiment
# becomes one aggregate Experiment that owns its whole triple. Both methods below
# are verbatim slices; only :meth:`MBRPropagatorExperiment.analyze` is new, and it
# replaces the string dispatch with a plain call.
#
# Rows are decoder occupations, columns are encoder occupations. ``raw_matrices``
# is what the quadratures give directly. ``matrices`` applies the per-decoder
# correction in analysis only when it was not already applied to the pulse.
#
# The base class is still the god Experiment, which is where the loading layer
# (``from_job_files``, ``_saved_parameters``, ``_quadrature``) lives until it is
# extracted. Per spec 7.4 no new aggregate base is invented ahead of the
# duplication that would justify it.
#
# Usage -- see ``analysis_notebooks/guan/MBR_analysis.py`` for the worked example::
#
#     expt = MBRPropagatorExperiment.from_job_files(paths)
#     expt.analyze()
#     expt.data.matrices          # (cycle, decoder, encoder)
#
# No display yet: the god Experiment never had one for this stage, and inventing
# one here would not be a move. That is the one part of the triple this class is
# still missing.
# ============================================================================

class MBRPropagatorExperiment(EncodingHamiltonianSpectroscopyExperiment):
    """Aggregate: raw propagator columns from one encoded-occupation batch."""

    @classmethod
    def reconstruct_propagator(cls,
                               propagator_expts,
                               occupations=None):
        """Reconstruct raw and Kerr-preserving ``M_q[j, i]`` matrices.

        Jobs share the cycle and decoder order. Returned correction angles
        have shape (decoder, encoder); correction locations follow encoder order.
        """
        first_cfg = propagator_expts[0].cfg.expt
        swap_stors = [int(stor) for stor in first_cfg.swap_stors]
        cycles = [int(cycle) for cycle in first_cfg.propagator_cycles]
        decoder_order = [
            tuple(occupation)
            for occupation in first_cfg.propagator_occupations
        ]

        raw_columns = {}
        columns = {}
        phase_corrections = {}
        correction_locations = {}
        for expt in propagator_expts:
            correction_location = expt.cfg.expt.get("phase_correction_location", "analysis")
            if correction_location not in ("pulse", "analysis"):
                raise ValueError("phase_correction_location must be 'pulse' or 'analysis'")
            encoder = tuple(expt.cfg.expt.spectroscopy_occupations)
            quadrature = np.asarray(
                cls._quadrature(expt), dtype=float
            ).reshape(len(cycles), len(decoder_order), 2)
            raw_columns[encoder] = (
                quadrature[:, :, 0] - 1j * quadrature[:, :, 1]
            )
            phase_correction = np.asarray(
                expt.cfg.expt.propagator_decoder_phase_correction_deg, dtype=float
            )
            columns[encoder] = raw_columns[encoder]
            if correction_location == "analysis":
                columns[encoder] = raw_columns[encoder] * np.exp(
                    -1j * np.deg2rad(
                        np.asarray(cycles)[:, None] * phase_correction[None, :]
                    )
                )
            phase_corrections[encoder] = phase_correction
            correction_locations[encoder] = correction_location

        occupation_order = decoder_order if occupations is None else [
            tuple(occupation) for occupation in occupations
        ]
        decoder_indices = [
            decoder_order.index(occupation)
            for occupation in occupation_order
        ]
        raw_matrices = np.stack([
            raw_columns[occupation][:, decoder_indices]
            for occupation in occupation_order
        ], axis=2).astype(complex, copy=False)
        matrices = np.stack([
            columns[occupation][:, decoder_indices]
            for occupation in occupation_order
        ], axis=2).astype(complex, copy=False)
        phase_correction = np.stack([
            phase_corrections[occupation][decoder_indices]
            for occupation in occupation_order
        ], axis=1)

        return AttrDict(dict(
            cycles=np.asarray(cycles, dtype=int),
            occupations=occupation_order,
            mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors],
            raw_matrices=raw_matrices,
            matrices=matrices,
            decoder_phase_correction_deg=phase_correction,
            phase_correction_location=np.asarray([
                correction_locations[occupation]
                for occupation in occupation_order
            ], dtype="S"),
            matrix_orientation="rows=decoder, columns=encoder",
        ))

    @staticmethod
    def propagator_batch(default_expt_cfg,
                         swap_stors,
                         occupations,
                         cycles,
                         phase_by_occupation,
                         sync_cycles=10,
                         reps=300,
                         phase_correction_location="pulse"):
        """Build one propagator job per encoder, with pulse or analysis correction."""
        if phase_correction_location not in ("pulse", "analysis"):
            raise ValueError("phase_correction_location must be 'pulse' or 'analysis'")
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
            phase_correction_location=phase_correction_location,
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
        if not hasattr(self, "batch_expts"):
            return super().analyze(data=data)
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


# ============================================================================
# MBRSpectrumExperiment (was experiments/qsim/mbr_spectrum.py)
#
# Many-body Ramsey spectrum: reconstruction, spectrum, and every plot of it.
#
# Spec sections 7.3/7.4: ``analyze(stage='spectrum')`` on the god Experiment
# becomes one aggregate Experiment owning its whole triple. This was the fat
# branch -- 982 lines of methods plus the two dispatch bodies -- and it is the
# path the golden baseline covers, so it moves verbatim and nothing here is
# rewritten.
#
# Not to be confused with ``fitting/qsim/mbr_spectrum.py``, imported below as
# ``mbr_spectrum_analysis``. That module is the pure numerics (window, pad, FFT,
# Hamiltonian, LDOS weights); this one is the Experiment that loads jobs, feeds
# them in, and plots what comes back. Spec 7.5 puts the two under those names
# deliberately: same subject, opposite sides of the acquire/analyze seam.
#
# The chain, in the order ``analyze`` runs it:
#
# 1. optional shot subsampling (``subsample_spectroscopy_shots``);
# 2. quadratures to a complex return ``A = Q_0 - i Q_90``
#    (``reconstruct_spectroscopy``, or ``reconstruct_pair_spectroscopy`` when the
#    jobs carry ``offdiag_cycles``);
# 3. phase-frame transformation against the calibration
#    (``_postprocess_reconstruction``, which needs
#    ``MBRPhaseCorrectionExperiment``);
# 4. spectrum, Hamiltonian and theory (``analyze_spectrum``, inherited alias);
# 5. optionally Matrix Pencil instead of the FFT peak fit.
#
# ``analyze`` takes its knobs as named parameters, so they show up in a
# notebook's ``?`` and a misspelling raises. The computation is unchanged and the
# golden baseline pins it; only the plumbing from argument to use was rewritten.
#
# Usage -- ``analysis_notebooks/guan/MBR_analysis.py`` is the worked example::
#
#     expt = MBRSpectrumExperiment.from_job_files(paths)
#     expt.analyze(calibration=calibration, cycle_branches={(3, 0, 0, 0, 0): 1})
#     expt.display()
#
# Two edits to otherwise verbatim bodies, both re-addressing a name the move
# invalidated:
#
# - ``analyze`` reached the calibration through the god module's
#   ``_stage_owner('calibration')``, a name that does not exist here. It names
#   :class:`MBRPhaseCorrectionExperiment` directly, which is the honest form: a
#   spectrum cannot be phase-corrected without a calibration, so this is a real
#   dependency between two stage Experiments rather than leftover coupling.
# - ``display_result`` hard-coded
# ``EncodingHamiltonianSpectroscopyExperiment.display_local_density_of_states``,
# which no longer has that method, so it names this class. Declared in
# ``tests/test_mbr_stage_split.py``.
# ============================================================================

_MATRIX_PENCIL_PREFIX = "mpm_"


_MATRIX_PENCIL_NAMES = frozenset(
    inspect.signature(matrix_pencil_analysis.analyze_matrix_pencil)
    .parameters) - {"reconstruction", "spectrum"}


def _matrix_pencil_options(options):
    """Strip the ``mpm_`` prefix; reject anything not a real option.

    The old ``kwargs.get("mpm_...")`` chain silently ignored a typo, so a
    mis-spelled tolerance looked like it worked and quietly did nothing.
    """
    stripped = {}
    for name, value in options.items():
        bare = name.removeprefix(_MATRIX_PENCIL_PREFIX)
        if bare not in _MATRIX_PENCIL_NAMES:
            raise TypeError(
                f"analyze() got an unexpected keyword argument {name!r}. "
                "Matrix-Pencil options are "
                + ", ".join(sorted(_MATRIX_PENCIL_PREFIX + n
                                   for n in _MATRIX_PENCIL_NAMES)))
        stripped[bare] = value
    return stripped


class MBRSpectrumExperiment(EncodingHamiltonianSpectroscopyExperiment):
    """Aggregate: one fixed-photon-number sector's spectrum from its jobs."""

    def analyze(self,
                data=None,
                occupations=None,
                calibration=None,
                cycle_branches: int | list | dict = 0,
                second_branch=False,
                phase_frame="as_acquired",
                manual_kerr_MHz=None,
                legacy=None,
                spectrum_method="fft",
                fft_window="raw",
                zero_padding=1,
                shots_per_point=None,
                shot_seed=None,
                mpm_calibration_sigma_multiplier=3.0,
                mpm_merge_frequency_tolerance_floor_kHz=0.1,
                **matrix_pencil_options):
        """Reconstruct, phase-correct, and transform one sector to a spectrum.

        Same computation as before; the knobs are now in the signature instead
        of behind ``kwargs.get``, so they are discoverable from a docstring or a
        ``?`` in a notebook, and a misspelled one raises instead of being
        silently ignored.

        - ``occupations`` fixes the reconstruction row order. Defaults to the
          order recorded in the jobs.
        - ``calibration`` supplies the analyzer phase correction: an analyzed
          :class:`MBRPhaseCorrectionExperiment`, paths to its jobs, or ``None``
          to use whatever correction was applied at pulse time.
        - ``cycle_branches`` picks the 180 deg/cycle branch per occupation --
          int, list, or ``{occupation: branch}``. ``second_branch=True`` is the
          shorthand for "add one to every branch" and cannot be combined with a
          nonzero ``cycle_branches``.
        - ``phase_frame`` selects the frame the reconstruction is transformed
          into before the spectrum is taken; ``'as_acquired'`` keeps the frame
          the data was measured in. ``manual_kerr_MHz`` overrides the Kerr rate
          used to build the correction, and ``legacy`` handles the pre-marking
          analyzer convention.
        - ``spectrum_method`` is ``'fft'`` or ``'matrix_pencil'`` (``'mpm'`` and
          ``'rowwise_matrix_pencil'`` are accepted spellings). Matrix Pencil is
          stored in ``data.matrix_pencil``; the FFT is computed either way.
        - ``shots_per_point`` subsamples the raw shots, with ``shot_seed`` for
          reproducibility. ``shot_seed`` alone is an error.
        - ``**matrix_pencil_options`` are forwarded to
          :func:`fitting.qsim.matrix_pencil.analyze_matrix_pencil`, spelled with
          the historical ``mpm_`` prefix (``mpm_pencil_length=...``). Unknown
          names raise, which the old ``kwargs.get`` chain could not do.
        """
        if not hasattr(self, "batch_expts"):
            return super().analyze(data=data)
        if data is not None:
            self.data = data
        matrix_pencil_options = _matrix_pencil_options(matrix_pencil_options)
        spectrum_method = str(spectrum_method).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
        analysis_expts = self.batch_expts
        shot_subsampling = None
        if shots_per_point is not None:
            analysis_expts, shot_subsampling = self.subsample_spectroscopy_shots(
                self.batch_expts,
                shots_per_point,
                seed=shot_seed,
            )
        elif shot_seed is not None:
            raise ValueError("shot_seed requires shots_per_point")
        saved = self._saved_parameters(analysis_expts,
                                       getattr(self, "_analysis_station", None))
        if "offdiag_cycles" in analysis_expts[0].cfg.expt:
            acquired_reconstruction = self.reconstruct_pair_spectroscopy(
                analysis_expts, occupations)
        else:
            acquired_reconstruction = self.reconstruct_spectroscopy(
                analysis_expts, occupations)
        photon_numbers = {sum(occupation) for occupation in acquired_reconstruction.occupations}
        if len(photon_numbers) != 1:
            raise ValueError("spectroscopy jobs must belong to one fixed-photon-number sector")
        photon_number = photon_numbers.pop()
        calibration_arg = calibration
        calibration = MBRPhaseCorrectionExperiment._calibration_data(
            calibration_arg, getattr(self, "_analysis_station", None))
        if second_branch:
            cycle_branches = self._cycle_branches(acquired_reconstruction.final_occupations,
                                                  cycle_branches)
            if np.any(cycle_branches):
                raise ValueError("use either cycle_branches or second_branch, not both")
            cycle_branches += 1
        saved_correction = self._saved_correction(analysis_expts)
        postprocessed = self._postprocess_reconstruction(
            acquired_reconstruction,
            saved_correction,
            calibration,
            saved.hardware,
            phase_frame,
            manual_kerr_MHz,
            cycle_branches,
            legacy)
        spectrum = self.analyze_spectrum(
            postprocessed.reconstruction,
            photon_number,
            saved.detunings,
            saved.hardware.couplings_MHz,
            saved.hardware.floquet_cycle_us,
            postprocessed.physical_kerr_MHz,
            fft_window,
            zero_padding,
        )
        self.data = AttrDict(dict(
            calibration=calibration,
            correction=saved_correction,
            saved_correction=saved_correction,
            target_correction=postprocessed.target_correction,
            acquired_reconstruction=acquired_reconstruction,
            reconstruction=postprocessed.reconstruction,
            spectrum=spectrum,
            hardware=saved.hardware,
            photon_number=photon_number,
            detunings=saved.detunings,
            mode_labels=saved.mode_labels,
            phase_frame=postprocessed.phase_frame,
            cycle_branches=postprocessed.cycle_branches,
            analyzer_phase_application_sign=postprocessed.analyzer_phase_application_sign,
            legacy_analyzer_migration=postprocessed.legacy_analyzer_migration,
            spectrum_method=spectrum_method,
        ))
        if spectrum_method == "matrix_pencil":
            merge_tolerance_bins = matrix_pencil_options.get("merge_frequency_tolerance_bins")
            row_calibration_se_MHz = None
            if isinstance(merge_tolerance_bins, str):
                if merge_tolerance_bins.lower() != "calibration":
                    raise ValueError("mpm_merge_frequency_tolerance_bins must be numeric, None, or 'calibration'")
                if calibration is None:
                    raise ValueError("calibration-derived MPM merging requires the phase calibration experiment")

                calibration_occupations = [tuple(occupation) for occupation in calibration.occupations]
                phase_slope_se = np.asarray(calibration.phase_error, dtype=float)
                if phase_slope_se.shape != (len(calibration_occupations),):
                    raise ValueError("calibration.phase_error must contain one slope standard error per occupation")

                calibration_cycle_us = float(calibration.hardware.floquet_cycle_us)
                if not np.isfinite(calibration_cycle_us) or calibration_cycle_us <= 0.:
                    raise ValueError("calibration Floquet cycle must be finite and positive")

                calibration_se_MHz = {
                    occupation: abs(float(slope_se)) / (360. * calibration_cycle_us)
                    for occupation, slope_se in zip(calibration_occupations, phase_slope_se)
                }
                reconstruction = postprocessed.reconstruction
                if "final_occupations" in reconstruction:
                    final_occupations = [tuple(occupation) for occupation in reconstruction.final_occupations]
                else:
                    final_occupations = [tuple(occupation) for occupation in reconstruction.occupations]
                missing_errors = [occupation for occupation in final_occupations if occupation not in calibration_se_MHz]
                if missing_errors:
                    raise ValueError(f"calibration is missing phase standard errors for {missing_errors}")
                row_calibration_se_MHz = np.asarray([calibration_se_MHz[occupation] for occupation in final_occupations])
                merge_tolerance_bins = None

            merge_floor_MHz = 1e-3 * mpm_merge_frequency_tolerance_floor_kHz
            matrix_pencil_options["merge_frequency_tolerance_bins"] = merge_tolerance_bins
            if row_calibration_se_MHz is not None:
                matrix_pencil_options["row_frequency_standard_errors_MHz"] = row_calibration_se_MHz
            matrix_pencil_options.setdefault("merge_frequency_tolerance_sigma", mpm_calibration_sigma_multiplier)
            matrix_pencil_options.setdefault("merge_frequency_tolerance_floor_MHz", merge_floor_MHz)
            self.data.matrix_pencil = self.analyze_matrix_pencil(
                postprocessed.reconstruction,
                spectrum,
                **matrix_pencil_options,
            )
        if shot_subsampling is not None:
            self.data.shot_subsampling = shot_subsampling
        if hasattr(calibration_arg, "batch_job_ids"):
            self.calibration_job_ids = list(calibration_arg.batch_job_ids)
        return self.data

    def display(self, data=None, occupation=None, **kwargs):
        """Spectrum panels, or one occupation's time trace when named.

        Body is the former ``display`` spectrum branch, unchanged.
        """
        if not hasattr(self, "batch_expts"):
            return super().display(data=data, **kwargs)
        if data is not None:
            self.data = data
        spectrum_method = str(kwargs.get("spectrum_method", self.data.get("spectrum_method", "fft"))).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
        if occupation is not None:
            if self.data.get("spectrum_only", False):
                raise ValueError("occupation time traces are unavailable for merged spectra with different time grids")
            if spectrum_method == "matrix_pencil":
                return self.display_matrix_pencil_occupation(data=self.data,
                                                             occupation=occupation,
                                                             show_magnitude_weights=kwargs.get("show_mpm_magnitude_weights", False))
            return self.display_occupation(self.data.reconstruction, self.data.spectrum, occupation, self.data.get("phase_frame", None), kwargs.get("ldos_weight_cutoff", 1e-3))
        if spectrum_method == "matrix_pencil":
            return self.display_matrix_pencil(data=self.data,
                                              show_poles=kwargs.get("show_mpm_poles", True))
        fig = self.display_result(self.data.reconstruction,
                                  self.data.spectrum,
                                  self.data.mode_labels)
        if self.data.spectrum.complete_basis and kwargs.get("level_statistics", True):
            self.display_level_statistics(
                data=self.data,
                peak_prominence=kwargs.get("level_peak_prominence", None),
                peak_prominence_fraction=kwargs.get("level_peak_prominence_fraction", None),
                minimum_peak_distance_MHz=kwargs.get("level_minimum_peak_distance_MHz", None),
                energy_limit_MHz=kwargs.get("level_energy_limit_MHz", None),
            )
        return fig


    @classmethod
    def subsample_spectroscopy_shots(cls,
                                     spectroscopy_expts,
                                     shots_per_point,
                                     seed=None):
        """Rebuild saved averages from fewer final-readout shots.

        Resolves each job's readout-lane count -- from the saved
        ``cfg.read_num`` where present, otherwise re-derived for jobs saved
        before that field existed -- and hands the numerics to
        :func:`fitting.qsim.mbr_reconstruction.subsample_spectroscopy_shots`,
        whose docstring explains the offset-tolerant re-averaging.

        Returns ``(subsampled_expts, metadata)``.
        """
        expts = list(flatten_exp_lists(spectroscopy_expts))
        readout_lanes = [int(expt.cfg.get("read_num", 0))
                         or readout_lane_count(expt.cfg)
                         for expt in expts]
        return subsample_spectroscopy_shots(
            expts, shots_per_point, readout_lanes, seed=seed)

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
        """
        The postprocessing got a bit complicated as the previous experiment
        did not designate `application_sign`. The current convention is 
        `application_sign` = -1, whereas previously it was +1.
        The designation of +1 to application_sign is done by setting legacy = True.
        
        
        
        """
        
        if manual_kerr_MHz is not None and phase_frame == "as_acquired":
            phase_frame = "manual_kerr"
        if phase_frame == "zero_kerr":
            if manual_kerr_MHz is not None:
                raise ValueError("zero_kerr does not take manual_kerr_MHz")
            manual_kerr_MHz = 0.
        if phase_frame not in ("as_acquired", "uncorrected", "zero_kerr", "manual_kerr"):
            raise ValueError("phase_frame must be 'as_acquired', 'uncorrected', 'zero_kerr', or 'manual_kerr'")
        occupations = reconstruction.occupations
        final_occupations = reconstruction.get("final_occupations", occupations)
        branches = cls._cycle_branches(final_occupations, cycle_branches)
        A = reconstruction.A.copy()
        target_correction = None
        application_sign = saved_correction.application_sign
        legacy_migration = False

        if phase_frame == "as_acquired":
            if legacy is not None:
                raise ValueError("legacy is only used with uncorrected/zero_kerr/manual_kerr rephasing")
            for row, branch in enumerate(branches):
                A[row] *= np.exp(-1j * np.deg2rad(180. * branch) * reconstruction.cycles)
            physical_kerr_MHz = hardware.physical_kerr_MHz
        else:
            if saved_correction.modes != {"final_analyzer"}:
                raise ValueError("uncorrected/zero_kerr/manual_kerr rephasing requires spectroscopy_phase_correction_mode='final_analyzer'")
            if application_sign is None:
                nonzero_correction = any(not np.isclose(phase, 0.) for phase in saved_correction.phase_by_occupation.values())
                if nonzero_correction and legacy is None:
                    raise ValueError("saved jobs do not record the analyzer sign; use legacy=True for old +correction jobs or legacy=False for -correction jobs")
                application_sign = 1. if legacy else -1.
                legacy_migration = bool(legacy)
            elif legacy is not None and application_sign != (1. if legacy else -1.):
                raise ValueError("legacy disagrees with the saved analyzer phase application sign")
            legacy_migration = application_sign == 1.

            if phase_frame == "uncorrected":
                if manual_kerr_MHz is not None:
                    raise ValueError("uncorrected does not take manual_kerr_MHz")
                for row, occupation in enumerate(final_occupations):
                    saved_phase = saved_correction.phase_by_occupation[tuple(occupation)]
                    A[row] *= np.exp(-1j * np.deg2rad(application_sign * saved_phase + 180. * branches[row]) * reconstruction.cycles)
                physical_kerr_MHz = hardware.physical_kerr_MHz
            else:
                if manual_kerr_MHz is None or not np.isfinite(manual_kerr_MHz):
                    raise ValueError("phase_frame='manual_kerr' requires a finite signed manual_kerr_MHz")
                if calibration is None:
                    raise ValueError("zero_kerr/manual_kerr rephasing requires calibration")
                calibration_phase = {tuple(occupation): phase for occupation, phase in zip(calibration.occupations, calibration.phase_mod180)}
                missing = [occupation for occupation in final_occupations if tuple(occupation) not in calibration_phase]
                if missing:
                    raise ValueError(f"calibration is missing occupations {missing}")
                target_correction = cls.build_phase_correction(final_occupations, [calibration_phase[tuple(occupation)] for occupation in final_occupations], branches, float(manual_kerr_MHz), hardware.floquet_cycle_us)
                for row, occupation in enumerate(final_occupations):
                    saved_phase = saved_correction.phase_by_occupation[tuple(occupation)]
                    target_phase = target_correction.phase_by_occupation[tuple(occupation)]
                    A[row] *= np.exp(-1j * np.deg2rad(application_sign * saved_phase + target_phase) * reconstruction.cycles)
                physical_kerr_MHz = float(manual_kerr_MHz)
        normalized_A = np.asarray([row / row[0] if tuple(initial) == tuple(final) else row for row, initial, final in zip(A, occupations, final_occupations)])
        return AttrDict(dict(reconstruction=AttrDict(dict(occupations=occupations, 
                                                          final_occupations=final_occupations,
                                                          cycles=reconstruction.cycles,
                                                          A=A,
                                                          A_norm=normalized_A)), 
                             target_correction=target_correction, 
                             physical_kerr_MHz=physical_kerr_MHz, 
                             phase_frame=phase_frame, 
                             cycle_branches=branches, 
                             analyzer_phase_application_sign=application_sign, 
                             legacy_analyzer_migration=legacy_migration))

    @classmethod
    def reconstruct_pair_spectroscopy(cls, spectroscopy_expts,
                                      occupations=None):
        """Reconstruct the interleaved off-diagonal acquisition path."""
        grouped = {}
        for expt in spectroscopy_expts:
            cfg = expt.cfg.expt
            initial = tuple(cfg.spectroscopy_occupations)
            final = tuple(cfg.offdiag_decoder_occupation)
            cycles = np.asarray(cfg.offdiag_cycles, dtype=int)
            quadratures = np.asarray(
                cls._quadrature(expt), dtype=float
            ).reshape(-1, 2)
            A = quadratures[:, 0] - 1j * quadratures[:, 1]
            phase = float(cfg.offdiag_decoder_phase_correction_deg)
            A *= np.exp(-1j * np.deg2rad(phase * cycles))
            grouped.setdefault((final, initial), []).append((cycles, A))

        if occupations is None:
            state_order = list(grouped)
        else:
            initial_order = [tuple(occupation) for occupation in occupations]
            state_order = [
                state for initial in initial_order
                for state in grouped if state[1] == initial
            ]

        rows = []
        expected_cycles = None
        for state in state_order:
            cycles = np.concatenate([chunk[0] for chunk in grouped[state]])
            A = np.concatenate([chunk[1] for chunk in grouped[state]])
            order = np.argsort(cycles)
            cycles, A = cycles[order], A[order]
            if expected_cycles is None:
                expected_cycles = cycles
            rows.append(A)

        initial_occupations = [state[1] for state in state_order]
        final_occupations = [state[0] for state in state_order]
        A = np.asarray(rows, dtype=complex)
        A_norm = np.asarray([
            row / row[0] if initial == final else row
            for row, initial, final in zip(
                A, initial_occupations, final_occupations
            )
        ])
        return AttrDict(dict(
            occupations=initial_occupations,
            final_occupations=final_occupations,
            cycles=expected_cycles,
            A=A,
            A_norm=A_norm,
        ))

    @classmethod
    def reconstruct_spectroscopy(cls, 
                                 spectroscopy_expts, 
                                 occupations=None):
        """
        Combine chunked spectroscopy jobs into the complex return amplitudes in
        the phase frame used during acquisition.

        For each initial occupation ``alpha``, the saved jobs must contain the
        analyzer settings ``phi=0`` and ``phi=90``. ``_quadrature`` first forms
        ``Q_phi=Pe(theta=0)-Pe(theta=180)`` from the two preparation phases.
        With the QICK convention ``Q_phi=Re[A_alpha exp(+i phi)]``, the two
        analyzer quadratures give

            ``A_alpha=Q_0-i Q_90=<alpha|U|alpha>``.

        The analyzer phase already contains any correction played by the pulse
        program. This method only reconstructs what was acquired: it does not
        undo or replace that correction, select a 180-degree phase branch,
        change the self-Kerr frame, or perform an FFT. Those operations belong
        to ``analyze(stage='spectrum')`` after this reconstruction.

        Jobs are grouped using the saved ``spectroscopy_occupations`` and
        ``spectroscopy_analyzer_phase``. Cycle chunks are concatenated and
        sorted, and every occupation and analyzer quadrature must cover the same
        non-overlapping cycle points. If ``occupations`` is supplied, it sets
        the returned row order and must contain exactly the occupations present
        in the saved jobs.

        Returns an AttrDict with ``occupations``, the common sorted ``cycles``,
        and a complex array ``A`` of shape ``(n_occupations, n_cycles)``. This
        result is called ``acquired_reconstruction`` by ``analyze`` to distinguish
        it from the reconstruction after an optional phase-frame transformation.
        """
        if not spectroscopy_expts:
            raise ValueError("spectroscopy_expts cannot be empty")
        grouped = {}
        for expt in spectroscopy_expts:
            cfg = expt.cfg.expt
            occupation = tuple(cfg.spectroscopy_occupations)
            final_occupation = tuple(cfg.get("spectroscopy_final_occupations", occupation))
            state = (final_occupation, occupation)
            phi = cfg.spectroscopy_analyzer_phase
            if phi not in (0., 90.):
                raise ValueError(f"{occupation} has analyzer phase {phi}; expected 0 or 90")
            if "floquet_cycles" not in cfg:
                raise ValueError(f"{occupation}, phi={phi}: this is not a spectroscopy job")
            if not np.allclose(expt.data["xpts"], [0., 180.]):
                raise ValueError(f"{occupation}, phi={phi}: saved preparation phases changed")
            if not np.array_equal(expt.data["ypts"], cfg.floquet_cycles):
                raise ValueError(f"{occupation}, phi={phi}: saved cycles do not match its config")
            if state not in grouped:
                grouped[state] = {0.: [], 90.: []}
            grouped[state][phi].append(expt)

        if occupations is None:
            state_order = list(grouped)
        else:
            occupation_order = [tuple(occupation) for occupation in occupations]
            state_order = [next(state for state in grouped if state[1] == occupation) for occupation in occupation_order]
        if len(state_order) != len(grouped) or set(state_order) != set(grouped):
            raise ValueError("spectroscopy occupations do not match the saved configs")
        expected_cycles = None
        rows = []

        for state in state_order:
            quadratures = []
            for phi in [0., 90.]:
                expts = grouped[state][phi]
                if not expts:
                    raise ValueError(f"{occupation} is missing phi={phi} data")
                cycles = np.concatenate([np.asarray(expt.data["ypts"]) for expt in expts])
                quadrature = np.concatenate([cls._quadrature(expt) for expt in expts])
                order = np.argsort(cycles)
                cycles = cycles[order]
                if len(np.unique(cycles)) != len(cycles):
                    raise ValueError(f"{occupation}, phi={phi}: spectroscopy cycles overlap")
                if expected_cycles is None:
                    expected_cycles = cycles
                elif not np.array_equal(cycles, expected_cycles):
                    raise ValueError(f"{occupation}, phi={phi}: spectroscopy cycles are incomplete")
                quadratures.append(quadrature[order])
            rows.append(quadratures[0] - 1j * quadratures[1])
        A = np.asarray(rows, dtype = complex)
        occupation_order = [state[1] for state in state_order]
        final_occupations = [state[0] for state in state_order]
        normalized_A = np.asarray([row / row[0] if initial == final else row for row, (final, initial) in zip(A, state_order)])
        return AttrDict(dict(occupations=occupation_order,
                             final_occupations=final_occupations,
                             cycles=expected_cycles, 
                             A= A,
                             A_norm= normalized_A))

    def analyze_matrix_pencil_occupation(self,
                                         occupation,
                                         data=None,
                                         matrix_pencil=None,
                                         least_squares_rcond=None):
        """Refit one occupation using only the poles found in that row.

        Thin wrapper: supplies ``self.data`` by default, then delegates to
        :func:`fitting.qsim.matrix_pencil.refit_occupation`. The module is
        imported under an alias so the historical ``matrix_pencil`` argument
        name survives the move.
        """
        data = self.data if data is None else data
        return matrix_pencil_analysis.refit_occupation(
            occupation,
            data,
            matrix_pencil=matrix_pencil,
            least_squares_rcond=least_squares_rcond,
        )

    def analyze_level_statistics(self,
                                 data=None,
                                 peak_prominence=None,
                                 peak_prominence_fraction=None,
                                 minimum_peak_distance_MHz=None,
                                 energy_limit_MHz=None):
        """Analyze measured DOS peaks and level spacings. See
        :func:`fitting.qsim.level_statistics.analyze_level_statistics`."""
        if data is None:
            data = self.data
        return level_statistics_analysis.analyze_level_statistics(
            data,
            peak_prominence=peak_prominence,
            peak_prominence_fraction=peak_prominence_fraction,
            minimum_peak_distance_MHz=minimum_peak_distance_MHz,
            energy_limit_MHz=energy_limit_MHz,
        )

    def analyze_sff(self, data=None, row_normalize=True):
        """Spectral form factor. See
        :func:`fitting.qsim.level_statistics.analyze_sff`."""
        data = self.data if data is None else data
        return level_statistics_analysis.analyze_sff(data, row_normalize=row_normalize)

    @staticmethod
    def display_local_density_of_states(spectrum, occupations, ax=None):
        """Plot rho_i(E) = sum_a |<i|E_a>|^2 delta(E-E_a)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        else:
            fig = ax.figure
        # Summed within each degenerate multiplet before the modulus is taken;
        # see mbr_spectrum.ldos_weights for why that order is the physics.
        ldos_energies_MHz, ldos_weights = mbr_spectrum_analysis.ldos_weights(spectrum)
        for row, weights in enumerate(ldos_weights):
            height = 0.8 * weights
            ax.hlines(row, -spectrum.energy_limit_MHz, spectrum.energy_limit_MHz, color="0.85")
            ax.vlines(ldos_energies_MHz, row, row + height, color="tab:blue")
        ax.set_yticks(np.arange(len(occupations)))
        ax.set_yticklabels([str(occupation) for occupation in occupations])
        ax.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="eigenenergy E/h (MHz)", ylabel="initial occupation", title="exact local density-of-states weights")
        return fig

    @staticmethod
    def display_occupation(reconstruction, 
                           spectrum, 
                           occupation, 
                           phase_frame=None, 
                           ldos_weight_cutoff=0,
                           axes = None,
                           figsize = None,
                           plot_abs_A = False):
        """
        The lowest-level function that displays
        - time trace
        - FFT result
        - LDOS result
        
        """
        
        if isinstance(occupation, (int, np.integer)):
            row = int(occupation)
            if row < 0 or row >= len(reconstruction.occupations):
                raise IndexError("occupation row is outside the spectroscopy data")
        else:
            occupation = tuple(occupation)
            if occupation not in reconstruction.occupations:
                raise ValueError(f"{occupation} is not in the spectroscopy data")
            row = reconstruction.occupations.index(occupation)
        occupation = tuple(reconstruction.occupations[row])
        final_occupation = tuple(reconstruction.get(
            "final_occupations", reconstruction.occupations
        )[row])
        offdiagonal = final_occupation != occupation

        measured = spectrum.measured_local[row]
        theory = spectrum.theory_local[row].copy()
        if np.max(theory) > 0.:
            theory *= np.max(measured) / np.max(theory)
        if not np.isfinite(ldos_weight_cutoff) or ldos_weight_cutoff < 0.:
            raise ValueError("ldos_weight_cutoff must be finite and nonnegative")
        ldos_energies_MHz, all_ldos_weights = mbr_spectrum_analysis.ldos_weights(spectrum)
        ldos_weights = all_ldos_weights[row]
        keep = ldos_weights >= ldos_weight_cutoff
        if axes is None:
            if figsize is None:
                figsize = (18, 4.8)
            fig, axes = plt.subplots(1, 3, 
                                     figsize=figsize, 
                                     constrained_layout=True)
        axes[0].plot(spectrum.time_us, reconstruction.A[row].real, label="Re A")
        axes[0].plot(spectrum.time_us, reconstruction.A[row].imag, label="Im A")
        if plot_abs_A:
            axes[0].plot(spectrum.time_us, np.abs(reconstruction.A[row]), "--", color="0.5", label="|A|")
        axes[0].set(xlabel="time (us)", 
                    ylabel="cross return" if offdiagonal else "return amplitude",
                    title=(
                        rf"$\langle {final_occupation}|U(t)|{occupation}\rangle$"
                        if offdiagonal else "oscillation trace"
                    ))
        axes[0].legend()

        axes[1].plot(spectrum.energy_MHz, measured, color="black", label="measured")
        axes[1].plot(spectrum.energy_MHz, theory, color="tab:orange", label="theory (shape scaled)")
        axes[1].set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude", title="off-diagonal finite-time FFT" if offdiagonal else "finite-time FFT")
        axes[1].legend()

        axes[2].vlines(ldos_energies_MHz[keep], 0., ldos_weights[keep], color="tab:blue")
        axes[2].plot(ldos_energies_MHz[keep], ldos_weights[keep], "o", color="tab:blue", markersize=4)
        axes[2].set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="eigenenergy E/h (MHz)", ylabel="spectral weight", title="exact off-diagonal spectral weights" if offdiagonal else "exact LDOS weights")
        title = (
            rf"$\langle {final_occupation}|U(t)|{occupation}\rangle$"
            if offdiagonal else str(occupation)
        )
        if phase_frame is not None:
            title += f"; frame={phase_frame}"
        if not offdiagonal:
            fig.suptitle(f"{title}; Kerr={spectrum.physical_kerr_MHz:.6g} MHz")
        return fig

    def display_occupations(self, 
                            data=None, 
                            occupations=None, 
                            ldos_weight_cutoff=1e-3,
                            spectrum_method=None,
                            show_mpm_magnitude_weights=False):
        data = self.data if data is None else data
        if "reconstruction" not in data or "spectrum" not in data:
            raise ValueError("occupation display requires analyzed spectroscopy data")
        if data.get("spectrum_only", False):
            raise ValueError("occupation time traces are unavailable for merged spectra with different time grids")
        if spectrum_method is None:
            spectrum_method = data.get("spectrum_method", "fft")
        spectrum_method = str(spectrum_method).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
        if occupations is None:
            selections = range(len(data.reconstruction.occupations))
        elif isinstance(occupations, (int, np.integer)):
            selections = [occupations]
        else:
            selections = list(occupations)
            if selections and all(np.isscalar(value) for value in selections):
                selections = [selections]

        figures = {}
        for occupation in selections:
            if isinstance(occupation, (int, np.integer)):
                row = int(occupation)
                initial = tuple(data.reconstruction.occupations[row])
                final = tuple(data.reconstruction.final_occupations[row])
                key = initial if initial == final else (final, initial)
            else:
                key = tuple(occupation)
            if spectrum_method == "matrix_pencil":
                figures[key] = self.display_matrix_pencil_occupation(data=data, occupation=occupation, show_magnitude_weights=show_mpm_magnitude_weights)
            else:
                figures[key] = self.display_occupation(data.reconstruction, data.spectrum, occupation, data.get("phase_frame", None), ldos_weight_cutoff)
        return figures

    def display_level_statistics(self,
                                 data=None,
                                 level_statistics=None,
                                 peak_prominence=None,
                                 peak_prominence_fraction=None,
                                 minimum_peak_distance_MHz=None,
                                 energy_limit_MHz=None):
        """
        Plot the measured complete-basis DOS and the spacing information justified by
        the data. Exact degeneracy gives raw gaps and all defined ratios, with Poisson/GOE
        means retained only as visual references. A nondegenerate Hamiltonian gives a
        gap-ratio plot only when all D measured levels are resolved; otherwise it gives a
        raw detected-gap diagnostic.
        
        """
        if level_statistics is None:
            level_statistics = self.analyze_level_statistics(
                data=data,
                peak_prominence=peak_prominence,
                peak_prominence_fraction=peak_prominence_fraction,
                minimum_peak_distance_MHz=minimum_peak_distance_MHz,
                energy_limit_MHz=energy_limit_MHz,
            )
        panel_count = 3 if level_statistics.has_exact_degeneracy else 2
        figure_width = 20 if level_statistics.has_exact_degeneracy else 14
        fig, axes = plt.subplots(1, panel_count, figsize=(figure_width, 5.2), constrained_layout=True)
        axes[0].plot(level_statistics.energy_MHz, level_statistics.theory_DOS, color="0.65", linewidth=1.5, alpha=0.6, label="exact H shape (scaled)")
        axes[0].plot(level_statistics.energy_MHz, level_statistics.measured_DOS, color="black", linewidth=2., label="experiment")
        axes[0].errorbar(level_statistics.peak_energies_MHz, level_statistics.peak_heights, xerr=0.5 * level_statistics.effective_resolution_MHz, fmt="o", color="tab:red", capsize=3, label="detected peaks (bar = resolution)")
        for peak_energy_MHz, peak_height, multiplicity in zip(level_statistics.peak_energies_MHz, level_statistics.peak_heights, level_statistics.multiplicities):
            axes[0].annotate(f"{peak_height:.2f} -> m~{multiplicity}", (peak_energy_MHz, peak_height), xytext=(0, 7), textcoords="offset points", ha="center", color="tab:red")
        multiplicity_title = f"m = rounded measured peak height; sum(m)={level_statistics.rounded_multiplicity_sum}, D={level_statistics.basis_dimension} (not forced)"
        axes[0].set(xlabel="energy E/h (MHz)", ylabel="summed FFT magnitude", title=f"measured DOS\n{multiplicity_title}")
        axes[0].set_ylim(bottom=0.)
        axes[0].legend()

        if level_statistics.has_exact_degeneracy:
            theory_gap_indices = np.arange(len(level_statistics.theory_raw_gaps_MHz))
            theory_zero_gap_count = int(np.count_nonzero(level_statistics.theory_zero_gap_mask))
            axes[1].scatter(theory_gap_indices, level_statistics.theory_raw_gaps_MHz, s=55, color="0.7", alpha=0.5, label=f"exact H: {theory_zero_gap_count}/{len(theory_gap_indices)} zero gaps")
            if len(level_statistics.inferred_degenerate_gaps_MHz):
                measured_gap_indices = np.arange(len(level_statistics.inferred_degenerate_gaps_MHz))
                measured_zero_gap_count = int(np.count_nonzero(np.isclose(level_statistics.inferred_degenerate_gaps_MHz, 0., rtol=0., atol=level_statistics.degeneracy_tolerance_MHz)))
                measured_label = f"experiment from rounded heights: {measured_zero_gap_count}/{len(measured_gap_indices)} zero gaps, sum(m)={level_statistics.rounded_multiplicity_sum}"
                if not level_statistics.multiplicity_sum_matches_D or level_statistics.detected_peak_count != len(level_statistics.theory_distinct_energies_MHz):
                    measured_label += " (incomplete)"
                axes[1].scatter(measured_gap_indices, level_statistics.inferred_degenerate_gaps_MHz, s=90, color="black", label=measured_label)
            else:
                axes[1].text(0.5, 0.5, "measured peak heights do not give positive multiplicities", transform=axes[1].transAxes, ha="center", va="center")
            axes[1].axhline(0., color="0.8")
            axes[1].set(xlabel="gap index", ylabel=r"adjacent gap $E_{n+1}-E_n$ (MHz)", title="raw adjacent gaps\nexact zeros retained")
            if len(level_statistics.theory_gap_ratios):
                theory_ratio_indices = np.arange(len(level_statistics.theory_gap_ratios))
                axes[2].scatter(theory_ratio_indices, level_statistics.theory_gap_ratios, s=55, color="0.7", alpha=0.5, label=f"exact H: n={len(theory_ratio_indices)}, 0/0 omitted={level_statistics.theory_undefined_gap_ratios}")
            else:
                axes[2].plot([], [], "o", color="0.7", label=f"exact H: all ratios undefined, 0/0 omitted={level_statistics.theory_undefined_gap_ratios}")
            if len(level_statistics.inferred_degenerate_gap_ratios):
                measured_ratio_indices = np.arange(len(level_statistics.inferred_degenerate_gap_ratios))
                axes[2].scatter(measured_ratio_indices, level_statistics.inferred_degenerate_gap_ratios, s=90, color="black", label=f"experiment: n={len(measured_ratio_indices)}, 0/0 omitted={level_statistics.inferred_undefined_gap_ratios}")
            else:
                axes[2].plot([], [], "o", color="black", label=f"experiment: no defined ratios, 0/0 omitted={level_statistics.inferred_undefined_gap_ratios}")
            axes[2].axhline(level_statistics.poisson_mean, color="tab:blue", linestyle="--", label=f"Poisson mean={level_statistics.poisson_mean:.3f}")
            axes[2].axhline(level_statistics.goe_mean, color="tab:orange", linestyle="--", label=f"GOE mean={level_statistics.goe_mean:.3f}")
            axes[2].set(xlabel="defined gap-ratio sample", ylabel=r"adjacent-gap ratio $\tilde r$", ylim=(-0.03, 1.03), title="defined ratios including exact zeros\nreference lines are not a fit")
            axes[2].legend()
            figure_title = f"N={level_statistics.photon_number}, D={level_statistics.basis_dimension}; exact multiplicity greater than one"
        elif not level_statistics.gap_ratio_available:
            theory_gap_indices = np.arange(len(level_statistics.theory_raw_gaps_MHz))
            detected_gap_indices = np.arange(len(level_statistics.detected_peak_gaps_MHz))
            axes[1].scatter(theory_gap_indices, level_statistics.theory_raw_gaps_MHz, s=55, color="0.7", alpha=0.5, label="exact H gaps (background)")
            if len(level_statistics.detected_peak_gaps_MHz):
                axes[1].scatter(detected_gap_indices, level_statistics.detected_peak_gaps_MHz, s=90, color="black", label="gaps between detected peaks")
            axes[1].text(0.5, 0.95, level_statistics.gap_ratio_unavailable_reason, transform=axes[1].transAxes, ha="center", va="top")
            axes[1].set(xlabel="gap index", ylabel=r"adjacent gap $E_{n+1}-E_n$ (MHz)", title="measured spectrum incomplete: raw gaps only\nPoisson/GOE comparison not used")
            figure_title = f"N={level_statistics.photon_number}, D={level_statistics.basis_dimension}; {level_statistics.detected_peak_count}/{level_statistics.basis_dimension} measured peaks detected"
        else:
            theory_ratio_indices = np.arange(len(level_statistics.theory_gap_ratios))
            experimental_ratio_indices = np.arange(len(level_statistics.gap_ratios))
            axes[1].scatter(theory_ratio_indices, level_statistics.theory_gap_ratios, s=55, color="0.7", alpha=0.5, label=f"exact H (background): mean={np.mean(level_statistics.theory_gap_ratios):.3f}")
            axes[1].scatter(experimental_ratio_indices, level_statistics.gap_ratios, s=90, color="black", label=f"experiment: mean={np.mean(level_statistics.gap_ratios):.3f}")
            axes[1].axhline(level_statistics.poisson_mean, color="tab:blue", linestyle="--", label=f"Poisson mean={level_statistics.poisson_mean:.3f}")
            axes[1].axhline(level_statistics.goe_mean, color="tab:orange", linestyle="--", label=f"GOE mean={level_statistics.goe_mean:.3f}")
            axes[1].set(xlabel="gap-ratio sample", ylabel=r"adjacent-gap ratio $\tilde r$", ylim=(-0.03, 1.03), title="adjacent-gap ratios")
            figure_title = f"N={level_statistics.photon_number}, D={level_statistics.basis_dimension}; all measured levels resolved"
        axes[1].legend()
        fig.suptitle(figure_title)
        return fig

    def display_sff(self, 
                    data=None, 
                    sff=None, 
                    row_normalize=True,
                    plot_theory_limit = False):
        if sff is None:
            sff = self.analyze_sff(data=data, row_normalize=row_normalize)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        axes[0].plot(sff.time_us, sff.SFF_exp, color="black", label="experiment")
        axes[0].plot(sff.time_us, sff.SFF_theory, color="tab:orange", label="theory")
        if plot_theory_limit:
            axes[0].axhline(sff.plateau_reference, color="0.6", linestyle=":", label="theory infinite-time average")
        axes[0].set(xlabel="time (us)", ylabel=r"$K(t)=|\mathrm{Tr}\,U(t)/D|^2$", title="spectral form factor")
        axes[0].legend()

        axes[1].semilogy(sff.time_us, np.maximum(sff.SFF_exp, 1e-12), color="black", label="experiment")
        axes[1].semilogy(sff.time_us, np.maximum(sff.SFF_theory, 1e-12), color="tab:orange", label="theory")
        if plot_theory_limit:
            axes[1].axhline(sff.plateau_reference, color="0.6", linestyle=":", label="theory infinite-time average")
        axes[1].set(xlabel="time (us)", ylabel=r"$K(t)$", title="spectral form factor (log scale)")
        axes[1].legend()
        fig.suptitle(f"N={sff.photon_number}, D={sff.dimension}; frame={sff.phase_frame}; Kerr={sff.physical_kerr_MHz:.6g} MHz")
        return fig

    def display_matrix_pencil(self,
                              data=None,
                              matrix_pencil=None,
                              show_poles=True):
        """
        Compare the measured and theoretical occupation-resolved FFTs and DOS.

        The upper panels show the measured and theoretical occupation-resolved
        finite-time FFTs. The lower-left panel compares the summed measured FFT
        with the Matrix-Pencil reconstruction and delta-function DOS. The
        lower-right panel compares the summed theoretical FFT with the exact
        Hamiltonian delta-function DOS.
        """
        data = self.data if data is None else data
        if "reconstruction" not in data or "spectrum" not in data:
            raise ValueError("Matrix-Pencil display requires analyzed spectroscopy data")
        if data.get("spectrum_only", False):
            raise ValueError("Matrix Pencil requires occupation traces on one common time grid")
        if matrix_pencil is None:
            matrix_pencil = data.get("matrix_pencil", None)
        if matrix_pencil is None:
            raise ValueError("Matrix-Pencil analysis is unavailable; analyze with spectrum_method='matrix_pencil'")

        reconstruction = data.reconstruction
        spectrum = data.spectrum
        rows = np.arange(len(reconstruction.occupations))
        labels = [
            str(initial) if tuple(initial) == tuple(final)
            else f"{tuple(final)} <- {tuple(initial)}"
            for initial, final in zip(
                reconstruction.occupations,
                reconstruction.final_occupations,
            )
        ]
        energy_MHz = np.asarray(spectrum.energy_MHz)
        measured_local = np.asarray(spectrum.measured_local)
        reconstructed_local = np.asarray(matrix_pencil.reconstructed_local)
        theory_local = np.asarray(spectrum.theory_local)
        if measured_local.shape != reconstructed_local.shape or measured_local.shape != theory_local.shape:
            raise ValueError("measured, Matrix-Pencil, and theory spectra use different grids")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        measured_axis = axes[0, 0]
        theory_axis = axes[0, 1]
        measured_DOS_axis = axes[1, 0]
        theory_DOS_axis = axes[1, 1]
        extent = [energy_MHz[0], energy_MHz[-1], -0.5, len(rows) - 0.5]
        vmax = max(np.max(measured_local), np.max(theory_local))
        for axis, local, title in zip((measured_axis, theory_axis),
                                      (measured_local, theory_local),
                                      ("measured finite-time FFT", "theory finite-time FFT")):
            image = axis.imshow(local, origin="lower", aspect="auto", interpolation="nearest", extent=extent, cmap="magma", vmin=0., vmax=vmax)
            if show_poles:
                for frequency_MHz in matrix_pencil.selected_frequencies_MHz:
                    axis.axvline(frequency_MHz, color="cyan", linewidth=0.7, alpha=0.45)
            axis.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", title=title)
            axis.set_yticks(rows)
            axis.set_yticklabels(labels)
        if show_poles:
            rowwise_frequencies_MHz = [candidate.frequency_MHz for candidate in matrix_pencil.candidates.per_row]
            rowwise_indices = [candidate.row_index for candidate in matrix_pencil.candidates.per_row]
            measured_axis.scatter(rowwise_frequencies_MHz, rowwise_indices, s=20, facecolors="none", edgecolors="cyan", linewidths=0.8, label="rowwise Matrix-Pencil poles")
            measured_axis.legend()
        measured_axis.set_ylabel(f"occupation {data.mode_labels}")
        fig.colorbar(image, ax=(measured_axis, theory_axis), label="spectral magnitude")

        measured_DOS_axis.plot(energy_MHz, spectrum.measured, color="black", linewidth=1.5, label="measured FFT sum")
        measured_DOS_axis.plot(energy_MHz, matrix_pencil.reconstructed, color="tab:blue", linestyle="--", linewidth=1.5, label="Matrix-Pencil finite-time reconstruction")
        measured_DOS_axis.vlines(matrix_pencil.selected_frequencies_MHz, 0., matrix_pencil.pole_DOS_weights, color="tab:blue", alpha=0.7, label="Matrix-Pencil linear pole DOS weights")
        measured_DOS_axis.plot(matrix_pencil.selected_frequencies_MHz, matrix_pencil.pole_DOS_weights, "o", color="tab:blue", markersize=5)
        measured_DOS_title = "measured FFT sum and Matrix-Pencil DOS" if spectrum.complete_basis else "measured projected FFT sum and Matrix-Pencil weights"
        measured_DOS_axis.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude / pole weight", title=measured_DOS_title)
        measured_DOS_axis.legend()

        exact_energies_MHz, exact_energy_indices = np.unique(np.round(np.asarray(spectrum.energies_MHz), 10), return_inverse=True)
        eigenstate_weights = np.asarray(spectrum.eigenstate_weights)
        if eigenstate_weights.ndim != 2 or eigenstate_weights.shape[1] != len(spectrum.energies_MHz):
            raise ValueError("exact eigenstate weights and energies have different dimensions")
        exact_state_weights = np.sum(eigenstate_weights, axis=0)
        exact_DOS_weights = np.bincount(exact_energy_indices, weights=exact_state_weights, minlength=len(exact_energies_MHz))
        theory_DOS_axis.plot(energy_MHz, spectrum.theory, color="tab:orange", linewidth=1.5, label="theory FFT sum")
        theory_DOS_axis.vlines(exact_energies_MHz, 0., exact_DOS_weights, color="tab:orange", alpha=0.7, label="exact Hamiltonian DOS weights")
        theory_DOS_axis.plot(exact_energies_MHz, exact_DOS_weights, "o", color="tab:orange", markersize=5)
        theory_DOS_title = "theory FFT sum and exact DOS" if spectrum.complete_basis else "theory projected FFT sum and exact spectral weights"
        theory_DOS_axis.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude / DOS weight", title=theory_DOS_title)
        theory_DOS_axis.legend()
        if show_poles:
            for frequency_MHz in matrix_pencil.selected_frequencies_MHz:
                measured_DOS_axis.axvline(frequency_MHz, color="cyan", linewidth=0.7, alpha=0.35)
                theory_DOS_axis.axvline(frequency_MHz, color="cyan", linewidth=0.7, alpha=0.35)
        fig.suptitle(f"K={len(matrix_pencil.selected_frequencies_MHz)} shared poles; global relative residual={matrix_pencil.relative_residual:.3f}; frequencies modulo fs={matrix_pencil.sampling.sampling_frequency_MHz:.6g} MHz")
        return fig

    def display_matrix_pencil_occupation(self,
                                         data=None,
                                         occupation=None,
                                         result=None,
                                         show_magnitude_weights=False):
        """Plot the independent rowwise Matrix-Pencil result for one occupation."""
        data = self.data if data is None else data
        if result is None:
            if occupation is None:
                raise ValueError("occupation must be supplied when result is None")
            result = self.analyze_matrix_pencil_occupation(occupation, data=data)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
        axes[0].plot(result.time_us, result.measured_return.real, color="tab:blue", label="Re A")
        axes[0].plot(result.time_us, result.measured_return.imag, color="tab:orange", label="Im A")
        axes[0].plot(result.time_us, result.fitted_return.real, "--", color="tab:blue", label="Re MPM fit")
        axes[0].plot(result.time_us, result.fitted_return.imag, "--", color="tab:orange", label="Im MPM fit")
        axes[0].set(xlabel="time (us)", ylabel="return amplitude", title="oscillation trace and rowwise MPM fit")
        axes[0].legend()

        axes[1].plot(result.energy_MHz, result.measured_spectrum, color="black", label="measured FFT")
        axes[1].plot(result.energy_MHz, result.reconstructed_spectrum, "--", color="tab:blue", label="rowwise MPM reconstruction")
        for frequency_index, frequency_MHz in enumerate(result.frequencies_MHz):
            label = "rowwise MPM poles" if frequency_index == 0 else None
            axes[1].axvline(frequency_MHz, color="cyan", linewidth=0.8, alpha=0.55, label=label)
        axes[1].set(xlim=(-data.spectrum.energy_limit_MHz, data.spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude", title="finite-time FFT")
        axes[1].legend()

        if len(result.frequencies_MHz):
            axes[2].vlines(result.frequencies_MHz, 0., result.local_weights, color="tab:blue")
            axes[2].plot(result.frequencies_MHz, result.local_weights, "o", color="tab:blue", label="linear local weights")
            if show_magnitude_weights:
                axes[2].plot(result.frequencies_MHz, result.local_magnitude_weights, "x", color="0.4", label="amplitude magnitudes")
        else:
            axes[2].text(0.5, 0.5, "no stable rowwise candidates", transform=axes[2].transAxes, ha="center", va="center")
        axes[2].axhline(0., color="0.8", linewidth=0.8)
        axes[2].set(xlim=(-data.spectrum.energy_limit_MHz, data.spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="local pole weight", title="individual-occupation MPM weights")
        if len(result.frequencies_MHz):
            axes[2].legend()

        frame = data.get("phase_frame", "as_acquired")
        fig.suptitle(f"{result.occupation}; rowwise poles={len(result.frequencies_MHz)}; estimated signal rank={result.diagnostic.estimated_signal_rank}; relative residual={result.relative_residual:.3f}; frame={frame}")
        return fig

    @staticmethod
    def display_result(reconstruction, 
                       spectrum, 
                       mode_labels):
        rows = np.arange(len(reconstruction.occupations))
        labels = [
            str(initial) if tuple(initial) == tuple(final)
            else f"{tuple(final)} <- {tuple(initial)}"
            for initial, final in zip(
                reconstruction.occupations,
                reconstruction.final_occupations,
            )
        ]
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

        extent = [spectrum.energy_MHz[0], spectrum.energy_MHz[-1], -0.5, len(rows) - 0.5]
        vmax = max(np.max(spectrum.measured_local), np.max(spectrum.theory_local))
        for ax, local, title in zip(axes[0], [spectrum.measured_local, spectrum.theory_local], ["experiment", "theory"]):
            image = ax.imshow(local, origin="lower", aspect="auto", interpolation="nearest", extent=extent, cmap="magma", vmin=0., vmax=vmax)
            fft_label = f"{spectrum.fft_window}, pad x{spectrum.zero_padding}"
            if spectrum.get("zero_padding", None) is None:
                fft_label = f"{spectrum.fft_window}, merged grid"
            elif spectrum.get("mixed_resolution", False):
                fft_label = f"{spectrum.fft_window}, mixed resolution"
            ax.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", title=f"{title}: {fft_label}")
            ax.set_yticks(rows)
            ax.set_yticklabels(labels)
        axes[0, 0].set_ylabel(f"occupation {mode_labels}")
        fig.colorbar(image, ax=axes[0], label="spectral magnitude")

        MBRSpectrumExperiment.display_local_density_of_states(spectrum, labels, axes[1, 0])
        axes[1, 1].plot(spectrum.energy_MHz, spectrum.measured, color="black", label="experiment")
        axes[1, 1].plot(spectrum.energy_MHz, spectrum.theory, color="tab:orange", label="theory")
        title = "projected spectrum"
        if spectrum.complete_basis:
            title = "complete-basis DOS"
        axes[1, 1].set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude", title=title)
        axes[1, 1].legend()
        resolution_label = f"FFT resolution: {spectrum.fft_resolution_MHz:.6g} MHz"
        if spectrum.get("mixed_resolution", False):
            resolution_label = f"FFT resolution range: {np.min(spectrum.row_fft_resolution_MHz):.6g}-{np.max(spectrum.row_fft_resolution_MHz):.6g} MHz"
        if spectrum.get("mixed_hamiltonian", False):
            resolution_label += f"; source-H mismatch: {spectrum.hamiltonian_mismatch_MHz:.6g} MHz"
        fig.suptitle(f"Kerr used in plotted Hamiltonian: {spectrum.physical_kerr_MHz:.6g} MHz; {resolution_label}")
        return fig

    @staticmethod
    def spectroscopy_batch(default_expt_cfg, 
                           swap_stors, 
                           occupations, 
                           cycle_chunks,
                           phase_by_occupation, 
                           detunings=None, 
                           sync_cycles=10, 
                           reps=300,
                           final_occupations=None):
        """
        Returns dictionary of 
            - default_expt_cfg
            - list of config to be overrided in each job
        The list of config is then used to make and batch jobs in a chunk.
        The actual batch is done by plugging the output to the BatchRunner.
        
        Example:
            spectroscopy_batch = EncSpec.spectroscopy_batch()
            spectroscopy_runner = BatchRunner(
                ExptProgram=spectroscopy_batch.program, ...)
            spectroscopy_expt = spectroscopy_runner.execute(spectroscopy_batch.configs)
        """
        if detunings is None:
            detunings = [0.] * len(swap_stors)
        else:
            detunings = list(detunings)
        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=reps, 
            storage_reset=swap_stors, 
            swap_stors=swap_stors,
            detunings=detunings, 
            scramble_sync_cycles=sync_cycles,
            
            floquet_hardware_loop=False,
            update_phases=True, 
            palindrome_scramble=False, 
            spectroscopy_phase_correction_mode="final_analyzer",
            spectroscopy_prep_phases=[0., 180.],
            swept_params=["floquet_cycle", "spectroscopy_prep_phase"],
        ))
        final_occupations = occupations if final_occupations is None else final_occupations
        pairs = list(zip(occupations, final_occupations))
        if any(tuple(initial) != tuple(final) for initial, final in pairs):
            defaults.update(dict(
                final_analyzer_phase_per_cycle_deg=0.,
                swept_params=[
                    "cycle_decoder_analyzer",
                    "spectroscopy_prep_phase",
                ],
            ))
            configs = [
                dict(
                    spectroscopy_occupations=list(initial),
                    offdiag_decoder_occupation=list(final),
                    offdiag_pair_index=pair_index,
                    offdiag_chunk_index=chunk_index,
                    offdiag_cycles=cycles.tolist(),
                    offdiag_decoder_phase_correction_deg=(
                        phase_by_occupation[tuple(final)]
                    ),
                    cycle_decoder_analyzers=[
                        [int(cycle), *final, phi]
                        for cycle in cycles for phi in [0., 90.]
                    ],
                )
                for pair_index, (initial, final) in enumerate(pairs)
                for chunk_index, cycles in enumerate(cycle_chunks)
            ]
            return AttrDict(dict(
                default_expt_cfg=defaults,
                configs=configs,
                program=EncodingPropagatorProgram,
            ))

        configs = [
            dict(spectroscopy_occupations=occupation,
                 spectroscopy_final_occupations=final_occupation,
                 spectroscopy_analyzer_phase=phi,
                 final_analyzer_phase_per_cycle_deg=phase_by_occupation[tuple(final_occupation)],
                 floquet_cycles=cycles.tolist())
            for occupation, final_occupation in pairs for cycles in cycle_chunks for phi in [0., 90.]
        ]
        return AttrDict(dict(default_expt_cfg=defaults, 
                             configs=configs,
                             program=NPhotonHamiltonianSpectroscopyProgram))

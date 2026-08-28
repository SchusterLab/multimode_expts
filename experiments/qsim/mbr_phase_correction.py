# -*- coding: utf-8 -*-
"""Closed-cycle phase calibration and the analyzer correction built from it.

Spec sections 7.3/7.4: ``analyze(stage='calibration')`` on the god Experiment
becomes one aggregate Experiment owning its whole triple. All eight methods
below are verbatim slices apart from two edits noted at the bottom of this
docstring; ``analyze`` and ``display`` are new and replace the string dispatch
with plain calls.

What it measures. Each occupation is driven for a range of *entire* Floquet
cycles at analyzer phases phi=0 and phi=90. Those two give the complex return
``Q_0 - i Q_90`` per cycle count; its phase slope is the accumulated phase per
cycle. Closed pairs sit two physical cycles apart, so the slope is determined
only modulo 180 deg/cycle -- picking a representative is what ``cycle_branches``
does, and the unwrapping itself lives in ``fitting/qsim/mbr_phase.py``.

What it produces. ``phase_mod180`` per occupation, and via
:meth:`phase_correction_from_calibration` the ``phase_by_occupation`` map the
spectroscopy pulse program subtracts from its final analyzer. That makes this
class an input to the spectrum stage, which is the one real dependency between
two stage Experiments.

Usage -- ``analysis_notebooks/guan/MBR_analysis.py`` is the worked example::

    calibration = MBRPhaseCorrectionExperiment.from_job_files(paths)
    calibration.analyze()
    calibration.display()
    correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
        calibration, cycle_branches={(2, 0, 0): 1})

The base class is still the god Experiment, which holds the loading layer
(``from_job_files``, ``_saved_parameters``, ``_quadrature``) and the
``fitting/qsim/mbr_phase.py`` aliases until they are extracted. Per spec 7.4 no
new aggregate base is invented ahead of the duplication that would justify it.

Two edits to otherwise verbatim bodies, both re-addressing a name the move
invalidated, neither changing behaviour:

- ``_calibration_data`` called ``cls.analyze(calibration, stage='calibration')``.
  This class's ``analyze`` has no ``stage``, so it is now ``cls.analyze(...)``.
- ``display_calibration_results`` hard-coded
  ``EncodingHamiltonianSpectroscopyExperiment.display_cycle_phase``, which no
  longer has that method. Now names this class.

Both are declared in ``tests/test_mbr_stage_split.py`` so the AST pin still
covers every other statement.
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)


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
                repeats=None,
                **kwargs):
        """Fit the phase per entire cycle for every calibrated occupation.

        Attaches the saved hardware parameters and mode labels, because
        :meth:`phase_correction_from_calibration` needs the Kerr rate and the
        Floquet cycle length that go with this data.
        """
        if data is not None:
            self.data = data
        self.data = self.analyze_calibration(
            self.batch_expts, occupations, cycle_pairs, repeats=repeats)
        saved = self._saved_parameters(
            self.batch_expts, getattr(self, "_analysis_station", None))
        self.data.hardware = saved.hardware
        self.data.mode_labels = saved.mode_labels
        return self.data

    def display(self, data=None, ncols=None, **kwargs):
        """Per-occupation IQ/phase fits, then the phase-per-cycle summary."""
        if data is not None:
            self.data = data
        if "results" not in self.data:
            self.data = self.analyze_calibration(self.batch_expts)
        self.display_calibration_results(self.data, ncols)
        return self.display_calibration_summary(self.data)

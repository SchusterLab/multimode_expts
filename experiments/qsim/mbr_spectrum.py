# -*- coding: utf-8 -*-
"""Many-body Ramsey spectrum: reconstruction, spectrum, and every plot of it.

Spec sections 7.3/7.4: ``analyze(stage='spectrum')`` on the god Experiment
becomes one aggregate Experiment owning its whole triple. This was the fat
branch -- 982 lines of methods plus the two dispatch bodies -- and it is the
path the golden baseline covers, so it moves verbatim and nothing here is
rewritten.

Not to be confused with ``fitting/qsim/mbr_spectrum.py``, imported below as
``mbr_spectrum_analysis``. That module is the pure numerics (window, pad, FFT,
Hamiltonian, LDOS weights); this one is the Experiment that loads jobs, feeds
them in, and plots what comes back. Spec 7.5 puts the two under those names
deliberately: same subject, opposite sides of the acquire/analyze seam.

The chain, in the order ``analyze`` runs it:

1. optional shot subsampling (``subsample_spectroscopy_shots``);
2. quadratures to a complex return ``A = Q_0 - i Q_90``
   (``reconstruct_spectroscopy``, or ``reconstruct_pair_spectroscopy`` when the
   jobs carry ``offdiag_cycles``);
3. phase-frame transformation against the calibration
   (``_postprocess_reconstruction``, which needs
   ``MBRPhaseCorrectionExperiment``);
4. spectrum, Hamiltonian and theory (``analyze_spectrum``, inherited alias);
5. optionally Matrix Pencil instead of the FFT peak fit.

``analyze`` still takes its knobs through ``**kwargs``, because it is the moved
branch body unchanged. Turning those into an explicit signature is the obvious
next step and is a real edit, so it gets its own commit with the golden as the
net.

Usage -- ``analysis_notebooks/guan/MBR_analysis.py`` is the worked example::

    expt = MBRSpectrumExperiment.from_job_files(paths)
    expt.analyze(calibration=calibration, cycle_branches={(3, 0, 0, 0, 0): 1})
    expt.display()

Two edits to otherwise verbatim bodies, both re-addressing a name the move
invalidated:

- ``analyze`` reached the calibration through the god module's
  ``_stage_owner('calibration')``, a name that does not exist here. It names
  :class:`MBRPhaseCorrectionExperiment` directly, which is the honest form: a
  spectrum cannot be phase-corrected without a calibration, so this is a real
  dependency between two stage Experiments rather than leftover coupling.
- ``display_result`` hard-coded
``EncodingHamiltonianSpectroscopyExperiment.display_local_density_of_states``,
which no longer has that method, so it names this class. Declared in
``tests/test_mbr_stage_split.py``.
"""
import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict
from experiments.MM_base import MMAveragerProgram
from fitting.qsim import level_statistics as level_statistics_analysis
from fitting.qsim import matrix_pencil as matrix_pencil_analysis
from fitting.qsim import mbr_spectrum as mbr_spectrum_analysis
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
    EncodingPropagatorProgram,
    NPhotonHamiltonianSpectroscopyProgram,
    flatten_exp_lists,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment


class MBRSpectrumExperiment(EncodingHamiltonianSpectroscopyExperiment):
    """Aggregate: one fixed-photon-number sector's spectrum from its jobs."""

    def analyze(self, data=None, **kwargs):
        """Reconstruct, phase-correct, and transform to a spectrum.

        Body is the former ``analyze(stage='spectrum')`` branch, unchanged.
        """
        if data is not None:
            self.data = data
        spectrum_method = str(kwargs.get("spectrum_method", "fft")).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
        analysis_expts = self.batch_expts
        shot_subsampling = None
        shots_per_point = kwargs.get("shots_per_point", None)
        if shots_per_point is not None:
            analysis_expts, shot_subsampling = self.subsample_spectroscopy_shots(
                self.batch_expts,
                shots_per_point,
                seed=kwargs.get("shot_seed", None),
            )
        elif kwargs.get("shot_seed", None) is not None:
            raise ValueError("shot_seed requires shots_per_point")
        saved = self._saved_parameters(analysis_expts, 
                                       getattr(self, "_analysis_station", None))
        if "offdiag_cycles" in analysis_expts[0].cfg.expt:
            acquired_reconstruction = self.reconstruct_pair_spectroscopy(
                analysis_expts, kwargs.get("occupations"))
        else:
            acquired_reconstruction = self.reconstruct_spectroscopy(
                analysis_expts, kwargs.get("occupations"))
        photon_numbers = {sum(occupation) for occupation in acquired_reconstruction.occupations}
        if len(photon_numbers) != 1:
            raise ValueError("spectroscopy jobs must belong to one fixed-photon-number sector")
        photon_number = photon_numbers.pop()
        calibration_arg = kwargs.get("calibration", None)
        calibration = MBRPhaseCorrectionExperiment._calibration_data(
            calibration_arg, getattr(self, "_analysis_station", None))
        cycle_branches = kwargs.get("cycle_branches", 0)
        if kwargs.get("second_branch", False):
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
            kwargs.get("phase_frame", "as_acquired"), 
            kwargs.get("manual_kerr_MHz", None), 
            cycle_branches, 
            kwargs.get("legacy", None))
        spectrum = self.analyze_spectrum(
            postprocessed.reconstruction, 
            photon_number, 
            saved.detunings,
            saved.hardware.couplings_MHz, 
            saved.hardware.floquet_cycle_us,
            postprocessed.physical_kerr_MHz,
            kwargs.get("fft_window", "raw"),
            kwargs.get("zero_padding", 1),
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
            self.data.matrix_pencil = self.analyze_matrix_pencil(
                postprocessed.reconstruction,
                spectrum,
                requested_max_modes=kwargs.get("mpm_requested_max_modes", None),
                pencil_length=kwargs.get("mpm_pencil_length", None),
                minimum_consecutive_ranks=kwargs.get("mpm_minimum_consecutive_ranks", 3),
                minimum_supporting_rows=kwargs.get("mpm_minimum_supporting_rows", 1),
                track_frequency_tolerance_bins=kwargs.get("mpm_track_frequency_tolerance_bins", 1.5),
                merge_frequency_tolerance_bins=kwargs.get("mpm_merge_frequency_tolerance_bins", None),
                dedup_frequency_tolerance_bins=kwargs.get("mpm_dedup_frequency_tolerance_bins", None),
                track_decay_tolerance_per_us=kwargs.get("mpm_track_decay_tolerance_per_us", None),
                dedup_decay_tolerance_per_us=kwargs.get("mpm_dedup_decay_tolerance_per_us", None),
                match_decay=kwargs.get("mpm_match_decay", True),
                numerical_floor=kwargs.get("mpm_numerical_floor", 1e-10),
                noise_singular_value_factor=kwargs.get("mpm_noise_singular_value_factor", 2.858), # this is based on the paper; IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
                minimum_pole_radius=kwargs.get("mpm_minimum_pole_radius", 0.2),
                maximum_pole_radius=kwargs.get("mpm_maximum_pole_radius", 1.05),
                require_early_start=kwargs.get("mpm_require_early_start", True),
                rank_sweep_extra=kwargs.get("mpm_rank_sweep_extra", None),
                clip_growth=kwargs.get("mpm_clip_growth", True),
                least_squares_rcond=kwargs.get("mpm_least_squares_rcond", None),
                store_rank_sweeps=kwargs.get("mpm_store_rank_sweeps", False),
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
        """
        Rebuild saved spectroscopy averages from fewer final-readout shots.

        Every saved sweep point contains interleaved readout lanes in ``idata``
        and ``qdata``. The final science measurement is the last lane of each
        repetition. This method draws ``shots_per_point`` paired I/Q samples
        from that lane without replacement and replaces only ``avgi``,
        ``avgq``, ``amps``, and ``phases`` in lightweight experiment copies.
        The original experiments and their raw-shot arrays are not modified.

        QICK's saved averaged values and ``collect_shots`` may use different ADC
        offsets or averaging rounds. For each point, the subset fluctuation
        relative to the complete raw-shot mean is therefore added to the saved
        average. Selecting every available shot then reproduces the saved
        ``avgi`` and ``avgq`` exactly without assuming a particular QICK offset.

        ``shots_per_point`` is the number of final-readout shots used for each
        saved ``(Floquet cycle, preparation phase)`` point. ``seed`` makes the
        random subset reproducible. Pre-selected acquisitions are rejected
        because their saved average is conditioned on herald lanes and cannot
        be reproduced by unconditioned final-lane sampling.

        Returns ``(subsampled_expts, metadata)``. The metadata records the
        requested shot count, seed, available-shot range, and raw-versus-saved
        averaging diagnostics for every child job.
        """

        if isinstance(shots_per_point, (bool, np.bool_)):
            raise ValueError("shots_per_point must be a positive integer")
        if not isinstance(shots_per_point, (int, np.integer)) or shots_per_point < 1:
            raise ValueError("shots_per_point must be a positive integer")
        shots_per_point = int(shots_per_point)
        spectroscopy_expts = list(flatten_exp_lists(spectroscopy_expts))
        if not spectroscopy_expts:
            raise ValueError("spectroscopy_expts cannot be empty")

        rng = np.random.default_rng(seed)
        subsampled_expts = []
        job_summaries = []
        available_shots = []

        for job_index, expt in enumerate(spectroscopy_expts):
            if (expt.cfg.expt.get("active_reset", False)
                    and expt.cfg.expt.get("pre_selection_reset", False)):
                raise ValueError(
                    "shot subsampling does not support pre_selection_reset; "
                    "the saved average is conditioned on herald readouts"
                )
            if "idata" not in expt.data or "qdata" not in expt.data:
                raise ValueError(f"spectroscopy job {job_index} has no saved single-shot IQ data")
            if "avgi" not in expt.data or "avgq" not in expt.data:
                raise ValueError(f"spectroscopy job {job_index} has no saved averaged IQ data")

            saved_avgi = np.asarray(expt.data["avgi"], dtype=float)
            saved_avgq = np.asarray(expt.data["avgq"], dtype=float)
            if saved_avgi.shape != saved_avgq.shape:
                raise ValueError(f"spectroscopy job {job_index} has mismatched avgi/avgq shapes")
            point_count = saved_avgi.size

            def point_rows(values, name):
                try:
                    array = np.asarray(values)
                except ValueError:
                    array = None
                if array is not None and array.ndim >= 2 and array.shape[0] == point_count:
                    return [np.asarray(array[index], dtype=float).reshape(-1)
                            for index in range(point_count)]
                if point_count == 1 and array is not None and array.dtype != object:
                    return [np.asarray(array, dtype=float).reshape(-1)]
                if len(values) == point_count:
                    return [np.asarray(values[index], dtype=float).reshape(-1)
                            for index in range(point_count)]
                raise ValueError(
                    f"spectroscopy job {job_index} has {name} that does not "
                    f"match its {point_count} sweep points"
                )

            idata_rows = point_rows(expt.data["idata"], "idata")
            qdata_rows = point_rows(expt.data["qdata"], "qdata")

            read_num = int(expt.cfg.get("read_num", 0))
            if read_num < 1:
                read_num = 1
                if expt.cfg.expt.get("parity_check", False):
                    read_num += 1
                if expt.cfg.expt.get("active_reset", False):
                    reset_params = MMAveragerProgram.get_active_reset_params(expt.cfg)
                    read_num += MMAveragerProgram.active_reset_read_num(**reset_params)
                if expt.cfg.expt.get("multiparity_readout", False):
                    read_num += 1
            final_lane = read_num - 1

            final_i_rows = []
            final_q_rows = []
            for point_index, (idata, qdata) in enumerate(zip(idata_rows, qdata_rows)):
                if len(idata) != len(qdata):
                    raise ValueError(
                        f"spectroscopy job {job_index}, point {point_index} has "
                        "different I/Q shot counts"
                    )
                if len(idata) % read_num:
                    raise ValueError(
                        f"spectroscopy job {job_index}, point {point_index} raw "
                        f"length {len(idata)} is not divisible by read_num={read_num}"
                    )
                final_i = idata[final_lane::read_num]
                final_q = qdata[final_lane::read_num]
                if len(final_i) < shots_per_point:
                    raise ValueError(
                        f"spectroscopy job {job_index}, point {point_index} has "
                        f"only {len(final_i)} final-readout shots; requested "
                        f"{shots_per_point}"
                    )
                final_i_rows.append(final_i)
                final_q_rows.append(final_q)
                available_shots.append(len(final_i))

            saved_avgi_flat = saved_avgi.reshape(-1)
            saved_avgq_flat = saved_avgq.reshape(-1)
            full_i_mean = np.asarray([np.mean(values) for values in final_i_rows])
            full_q_mean = np.asarray([np.mean(values) for values in final_q_rows])
            full_raw_minus_saved_avgi = full_i_mean - saved_avgi_flat
            full_raw_minus_saved_avgq = full_q_mean - saved_avgq_flat

            sampled_avgi = np.empty(point_count, dtype=float)
            sampled_avgq = np.empty(point_count, dtype=float)
            for point_index, (final_i, final_q) in enumerate(zip(final_i_rows, final_q_rows)):
                selected_indices = rng.choice(len(final_i),
                                              size=shots_per_point,
                                              replace=False)
                sampled_avgi[point_index] = (
                    saved_avgi_flat[point_index]
                    + np.mean(final_i[selected_indices])
                    - full_i_mean[point_index]
                )
                sampled_avgq[point_index] = (
                    saved_avgq_flat[point_index]
                    + np.mean(final_q[selected_indices])
                    - full_q_mean[point_index]
                )

            sampled_avgi = sampled_avgi.reshape(saved_avgi.shape)
            sampled_avgq = sampled_avgq.reshape(saved_avgq.shape)
            sampled_data = AttrDict(dict(expt.data))
            sampled_data["avgi"] = sampled_avgi
            sampled_data["avgq"] = sampled_avgq
            sampled_data["amps"] = np.abs(sampled_avgi + 1j * sampled_avgq)
            sampled_data["phases"] = np.angle(sampled_avgi + 1j * sampled_avgq)
            sampled_data.pop("Pe", None)
            sampled_data.pop("return_quadrature", None)

            sampled_expt = copy(expt)
            sampled_expt.data = sampled_data
            subsampled_expts.append(sampled_expt)
            job_summaries.append(AttrDict(dict(
                job_index=job_index,
                read_num=read_num,
                point_count=point_count,
                minimum_available_shots=min(len(values) for values in final_i_rows),
                maximum_available_shots=max(len(values) for values in final_i_rows),
                median_full_raw_minus_saved_avgi=float(
                    np.median(full_raw_minus_saved_avgi)
                ),
                median_full_raw_minus_saved_avgq=float(
                    np.median(full_raw_minus_saved_avgq)
                ),
                maximum_full_raw_minus_saved_avgi_scatter=float(
                    np.max(np.abs(
                        full_raw_minus_saved_avgi
                        - np.median(full_raw_minus_saved_avgi)
                    ))
                ),
                maximum_full_raw_minus_saved_avgq_scatter=float(
                    np.max(np.abs(
                        full_raw_minus_saved_avgq
                        - np.median(full_raw_minus_saved_avgq)
                    ))
                ),
            )))

        metadata = AttrDict(dict(
            shots_per_point=shots_per_point,
            seed=seed,
            replace=False,
            minimum_available_shots=min(available_shots),
            maximum_available_shots=max(available_shots),
            job_summaries=job_summaries,
        ))
        return subsampled_expts, metadata

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

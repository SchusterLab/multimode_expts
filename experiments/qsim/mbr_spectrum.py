# -*- coding: utf-8 -*-
"""A fixed-photon-number sector's spectrum from diagonal MBR time traces.

An assembled class (docs/qsim/mbr_redesign.md, sections 1, 2 and 5): one
diagonal :class:`MBRTimeTraceExperiment` per occupation. It never goes to the
worker itself.

    spectrum = MBRSpectrumExperiment(occupations, cycles=range(0, 400, 2),
                                     swap_stors=[1, 2, 3, 4], calibration=cal,
                                     cycle_branches={(3, 0, 0, 0, 0): 1})
    spectrum.acquire(runner, batch_size=10)   # runner.ExptClass is MBRTimeTraceExperiment
    spectrum.analyze(); spectrum.display()
    spectrum.save()                           # manifest YAML + assembled HDF5

    spectrum = MBRSpectrumExperiment.from_manifest(path)

``calibration`` is a saved :class:`MBRCalibrationSetExperiment`; each job
records its manifest path. With ``calibration=None`` no Stark-shift
correction is played.

The analysis chain, in the order ``analyze`` runs it:

1. optional shot subsampling (dormant; ``shots_per_point``);
2. the jobs' complex returns stacked into one reconstruction ``A[row, cycle]``;
3. phase-frame transformation
   (:func:`fitting.qsim.mbr_reconstruction.postprocess_reconstruction`);
4. spectrum, Hamiltonian and theory
   (:func:`fitting.qsim.mbr_spectrum.analyze_spectrum`);
5. optionally Matrix Pencil next to the FFT.

Steps 3 to 5 and the displays are carried over from the old
``MBRSpectrumExperiment`` (now in ``deprecated/legacy_mbr.py``) without
changes to the arithmetic.
"""
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.dark_base import readout_lane_count
from experiments.qsim.mbr_saved import saved_parameters
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from fitting.qsim import level_statistics as level_statistics_analysis
from fitting.qsim import matrix_pencil as matrix_pencil_analysis
from fitting.qsim import mbr_phase
from fitting.qsim import mbr_spectrum as mbr_spectrum_analysis
from fitting.qsim.mbr_reconstruction import (
    postprocess_reconstruction,
    subsample_spectroscopy_shots,
)


class MBRSpectrumExperiment(AssembledExperiment):
    """Diagonal time traces over the occupations of one photon-number sector."""

    child_class = MBRTimeTraceExperiment

    def __init__(self, occupations, cycles, swap_stors, calibration=None,
                 cycle_branches=0, detunings=None, sync_cycles=10, reps=300,
                 notes=""):
        """``cycle_branches`` picks the 180 deg/cycle branch of the correction
        played on the pulse at acquisition; ``analyze(cycle_branches=...)`` is
        the separate branch applied in analysis."""
        super().__init__(notes=notes)
        self.occupations = [tuple(int(n) for n in o) for o in occupations]
        self.cycles = [int(n) for n in cycles]
        self.swap_stors = [int(stor) for stor in swap_stors]
        self.calibration = calibration
        self.cycle_branches = cycle_branches
        self.detunings = (None if detunings is None
                          else [float(d) for d in detunings])
        self.sync_cycles = int(sync_cycles)
        self.reps = int(reps)

    # -- acquisition ------------------------------------------------------

    def job_overrides(self):
        """-> one diagonal TimeTrace override dict per occupation.

        The analyzer correction of each occupation comes from
        ``calibration.phase_correction(cycle_branches)``, which needs the
        calibration set saved, so every job can record where it came from.
        """
        if self.calibration is None:
            phases = {occupation: 0. for occupation in self.occupations}
            manifest = None
        else:
            if self.calibration.manifest_path is None:
                raise ValueError("save() the calibration set first, so the jobs "
                                 "can record its manifest path")
            phases = self.calibration.phase_correction(self.cycle_branches).phase_by_occupation
            manifest = self.calibration.manifest_path
        return [self.child_class.job_config(
                    occupation, occupation, self.cycles, self.swap_stors,
                    phase_per_cycle_deg=phases[occupation],
                    calibration_manifest=manifest, detunings=self.detunings,
                    sync_cycles=self.sync_cycles, reps=self.reps)
                for occupation in self.occupations]

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", calibration=None):
        """Assemble already acquired or loaded diagonal TimeTrace jobs.

        Occupations, cycles, swap modes and detunings are read from the jobs'
        cfg.expt, in job order. ``calibration`` is the set used to analyze;
        it is not read from the jobs.
        """
        children = list(children)
        if not children:
            raise ValueError("a spectrum needs at least one TimeTrace job")
        first = children[0]
        ecfg = first.cfg.expt
        for child in children:
            if child.initial_occupation != child.final_occupation:
                raise ValueError(f"{child!r} is off-diagonal; a spectrum takes diagonal traces")
            if list(child.cfg.expt.floquet_cycles) != list(ecfg.floquet_cycles):
                raise ValueError(f"{child.initial_occupation}: different Floquet cycles")
        occupations = [child.initial_occupation for child in children]
        if len(set(occupations)) != len(occupations):
            raise ValueError("each occupation must appear once")
        spectrum = cls(occupations, ecfg.floquet_cycles, ecfg.swap_stors,
                       calibration=calibration,
                       detunings=ecfg.get("detunings", None),
                       sync_cycles=int(ecfg.get("scramble_sync_cycles", 10)),
                       reps=int(ecfg.reps), notes=notes)
        spectrum.children = children
        spectrum.job_ids = list(job_ids)
        spectrum._check_children()
        return spectrum

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment

        calibration = manifest.get("calibration_manifest")
        if not calibration:
            return {}
        return dict(calibration=MBRCalibrationSetExperiment.from_manifest(calibration))

    # -- analysis ---------------------------------------------------------

    def reconstruction(self, children=None):
        """-> the jobs' returns as one acquired reconstruction.

        AttrDict with ``occupations``, ``final_occupations``, the common
        ``cycles`` and complex ``A`` / ``A_norm`` of shape
        ``(n_occupations, n_cycles)``, in the phase frame of acquisition.
        """
        children = self.children if children is None else children
        for child in children:
            if "complex_return" not in child.data:
                child.analyze()
        A = np.asarray([child.data["complex_return"] for child in children], dtype=complex)
        occupations = [child.initial_occupation for child in children]
        final_occupations = [child.final_occupation for child in children]
        return AttrDict(dict(
            occupations=occupations,
            final_occupations=final_occupations,
            cycles=np.asarray(children[0].data["cycles"]),
            A=A,
            A_norm=np.asarray([row / row[0] for row in A]),
        ))

    def analyze(self,
                phase_frame="as_acquired",
                cycle_branches=0,
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
        """Reconstruct, phase-correct, and transform the sector to a spectrum.

        - ``phase_frame``: ``'as_acquired'`` keeps the frame the data was
          measured in; ``'uncorrected'``, ``'zero_kerr'`` and ``'manual_kerr'``
          rebuild it (see
          :func:`fitting.qsim.mbr_reconstruction.postprocess_reconstruction`).
          ``manual_kerr_MHz`` sets the Kerr rate, and ``legacy`` handles jobs
          saved before the analyzer sign was recorded.
        - ``cycle_branches`` picks the 180 deg/cycle branch per occupation:
          int, list, or ``{occupation: branch}``.
        - ``spectrum_method`` is ``'fft'`` or ``'matrix_pencil'``; Matrix
          Pencil goes to ``data.matrix_pencil`` and the FFT is computed either
          way. ``mpm_*`` keyword arguments go to
          :func:`fitting.qsim.matrix_pencil.analyze_matrix_pencil`.
        - ``shots_per_point`` subsamples the raw shots, with ``shot_seed``.

        The calibration set, if any, is ``self.calibration``.
        """
        self._check_children()
        matrix_pencil_options = matrix_pencil_analysis.strip_option_prefix(matrix_pencil_options)
        spectrum_method = str(spectrum_method).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")

        analysis_children = self.children
        shot_subsampling = None
        if shots_per_point is not None:
            readout_lanes = [int(child.cfg.get("read_num", 0)) or readout_lane_count(child.cfg)
                             for child in self.children]
            sampled, shot_subsampling = subsample_spectroscopy_shots(
                self.children, shots_per_point, readout_lanes, seed=shot_seed)
            analysis_children = []
            for child in sampled:
                child = copy(child)
                child.data.pop("complex_return", None)
                child.analyze()
                analysis_children.append(child)
        elif shot_seed is not None:
            raise ValueError("shot_seed requires shots_per_point")

        saved = saved_parameters(analysis_children)
        acquired_reconstruction = self.reconstruction(analysis_children)
        photon_numbers = {sum(occupation) for occupation in acquired_reconstruction.occupations}
        if len(photon_numbers) != 1:
            raise ValueError("spectroscopy jobs must belong to one fixed-photon-number sector")
        photon_number = photon_numbers.pop()
        calibration = None
        if self.calibration is not None:
            if "phase_mod180" not in self.calibration.data:
                self.calibration.analyze()
            calibration = self.calibration.data
        saved_correction = mbr_phase.saved_correction(analysis_children)
        postprocessed = postprocess_reconstruction(
            acquired_reconstruction,
            saved_correction,
            calibration,
            saved.hardware,
            phase_frame,
            manual_kerr_MHz,
            cycle_branches,
            legacy)
        spectrum = mbr_spectrum_analysis.analyze_spectrum(
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
                final_occupations = [tuple(occupation) for occupation in reconstruction.final_occupations]
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
            self.data.matrix_pencil = matrix_pencil_analysis.analyze_matrix_pencil(
                postprocessed.reconstruction,
                spectrum,
                **matrix_pencil_options,
            )
        if shot_subsampling is not None:
            self.data.shot_subsampling = shot_subsampling
        return self.data

    def display(self, occupation=None, spectrum_method=None, level_statistics=True,
                ldos_weight_cutoff=1e-3, show_mpm_poles=True,
                show_mpm_magnitude_weights=False):
        """Spectrum panels, or one occupation's time trace when named.

        With a complete basis, the level statistics follow the spectrum
        panels unless ``level_statistics=False``.
        """
        if not self.data:
            raise ValueError("run analyze() before display()")
        if spectrum_method is None:
            spectrum_method = self.data.get("spectrum_method", "fft")
        spectrum_method = str(spectrum_method).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
        if occupation is not None:
            if spectrum_method == "matrix_pencil":
                return self.display_matrix_pencil_occupation(
                    data=self.data, occupation=occupation,
                    show_magnitude_weights=show_mpm_magnitude_weights)
            return self.display_occupation(self.data.reconstruction, self.data.spectrum,
                                           occupation, self.data.get("phase_frame", None),
                                           ldos_weight_cutoff)
        if spectrum_method == "matrix_pencil":
            return self.display_matrix_pencil(data=self.data, show_poles=show_mpm_poles)
        fig = self.display_result(self.data.reconstruction, self.data.spectrum,
                                  self.data.mode_labels)
        if self.data.spectrum.complete_basis and level_statistics:
            self.display_level_statistics(data=self.data)
        return fig

    # -- persistence ------------------------------------------------------

    def calibration_manifest(self):
        if self.calibration is None:
            return None
        return self.calibration.manifest_path

    def manifest_parameters(self):
        return dict(occupations=[list(o) for o in self.occupations],
                    cycles=self.cycles,
                    swap_stors=self.swap_stors)

    def assembled_arrays(self):
        data = self.data
        return dict(
            occupations=np.asarray(self.occupations, dtype=int),
            cycles=data.acquired_reconstruction.cycles,
            acquired_A=data.acquired_reconstruction.A,
            A=data.reconstruction.A,
            time_us=data.spectrum.time_us,
            energy_MHz=data.spectrum.energy_MHz,
            measured_local=data.spectrum.measured_local,
            measured=data.spectrum.measured,
            couplings_MHz=data.hardware.couplings_MHz,
            detunings=data.detunings,
        )

    def assembled_attrs(self):
        data = self.data
        return dict(
            phase_frame=str(data.phase_frame),
            spectrum_method=str(data.spectrum_method),
            photon_number=int(data.photon_number),
            cycle_branches=[int(b) for b in data.cycle_branches],
            floquet_cycle_us=float(data.hardware.floquet_cycle_us),
            physical_kerr_MHz=float(data.spectrum.physical_kerr_MHz),
            hardware_source=str(data.hardware.source),
            mode_labels=list(data.mode_labels),
        )

    # -- carried over from the old MBRSpectrumExperiment -------------------

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

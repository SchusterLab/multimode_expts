# -*- coding: utf-8 -*-
"""Hamiltonian tomography from M_q at q = 0, q, 2q, ...: MBRHamTomoExperiment.

An assembled class of assembled parts (docs/qsim/mbr_redesign.md, sections 2
and 5): one :class:`MBROrthogonalityExperiment` per cycle count, all over the
same occupations. Acquisition of the parts can take hours, with decisions in
between, so there is no ``acquire``; the notebook acquires and saves each
part, then combines them::

    orthos = []
    for q in (0, 20, 40):
        ortho = MBROrthogonalityExperiment(occupations, swap_stors, cycle=q,
                                           calibration=cal)
        ortho.acquire(runner); ortho.analyze(); ortho.save()
        orthos.append(ortho)
    tomo = MBRHamTomoExperiment.from_parts(orthos, calibration=cal)
    tomo.analyze(); tomo.display()
    tomo.save()                  # manifest lists the parts' manifests

    tomo = MBRHamTomoExperiment.from_manifest(path)

Without a ``calibration`` (a :class:`MBRCalibrationSetExperiment` over the
same occupations), ``analyze`` only stacks the matrices. With one, it also
runs the tomography of
:func:`fitting.qsim.mbr_propagator.analyze_propagator_dynamics`, whose
endpoint normalization needs each occupation's q = 0 self-return.
"""
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_saved import saved_parameters
from fitting.qsim.mbr_propagator import analyze_propagator_dynamics, fit_shared_step


def endpoint_calibration(calibration):
    """-> the calibration data the tomography reads, from a calibration set.

    Per occupation its return vs physical cycles; with the mode labels and
    hardware of the set. The phase fit is not needed.
    """
    for child in calibration.children:
        if "complex_return" not in child.data:
            child.analyze()
    saved = saved_parameters(calibration.children)
    return AttrDict(dict(
        mode_labels=saved.mode_labels,
        hardware=saved.hardware,
        results=[AttrDict(dict(occupation=list(child.occupation),
                               physical_cycles=np.asarray(child.data["physical_cycles"]),
                               complex_return=np.asarray(child.data["complex_return"])))
                 for child in calibration.children],
    ))


class MBRHamTomoExperiment(AssembledExperiment):
    """Orthogonality matrices M_q over cycle counts q, and the tomography."""

    child_class = MBROrthogonalityExperiment

    def __init__(self, cycles, occupations, swap_stors, calibration=None, notes=""):
        super().__init__(notes=notes)
        self.cycles = [int(q) for q in cycles]
        self.occupations = [tuple(int(n) for n in o) for o in occupations]
        self.swap_stors = [int(stor) for stor in swap_stors]
        self.calibration = calibration

    def acquire(self, runner, batch_size=10, **execute_kwargs):
        raise NotImplementedError(
            "MBRHamTomoExperiment is built with from_parts(): acquire and save one "
            "MBROrthogonalityExperiment per cycle count first")

    @classmethod
    def from_parts(cls, orthogonalities, calibration=None, notes=""):
        """Combine saved or unsaved Orthogonality objects, one per cycle count.

        They must share occupations (in the same order) and swap modes; they
        are sorted by cycle count. ``job_ids`` become all their job IDs.
        """
        parts = sorted(orthogonalities, key=lambda part: part.cycle)
        if not parts:
            raise ValueError("Hamiltonian tomography needs at least one Orthogonality")
        first = parts[0]
        for part in parts[1:]:
            if part.occupations != first.occupations:
                raise ValueError(f"q = {part.cycle}: different occupations or order")
            if part.swap_stors != first.swap_stors:
                raise ValueError(f"q = {part.cycle}: different swap modes")
        cycles = [part.cycle for part in parts]
        if len(set(cycles)) != len(cycles):
            raise ValueError(f"each cycle count must appear once; got {cycles}")
        tomo = cls(cycles, first.occupations, first.swap_stors,
                   calibration=calibration, notes=notes)
        tomo.children = parts
        tomo.job_ids = [job_id for part in parts for job_id in part.job_ids]
        tomo._check_children()
        return tomo

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", calibration=None):
        """``from_parts`` with the job IDs a manifest recorded."""
        tomo = cls.from_parts(children, calibration=calibration, notes=notes)
        tomo.job_ids = list(job_ids)
        return tomo

    @classmethod
    def plan_three_depth(cls, calibration, campaign, N=1, step=10, reps=1000, batch_size=5):
        """-> the parts of the q = [0, s, 2s] pilot, one Orthogonality per depth.

        Submits nothing; prints the workload. ``calibration`` is the saved
        N-photon calibration set: each part's jobs record its manifest and
        play its per-decoder correction. ``batch_size`` is only for the time
        estimate. Acquire each part with an ``MBROrthoColumnExperiment``
        runner, save it, then combine with :meth:`from_parts` and run
        :meth:`analyze_shared_step`.
        """
        from experiments.qsim.mbr_campaign import fixed_n_occupations

        if calibration.manifest_path is None:
            raise RuntimeError("save() the calibration set first, so the jobs can record it")
        cycles = [0, step, 2 * step]
        cycle_us = float(saved_parameters(calibration.children).hardware.floquet_cycle_us)
        occupations = fixed_n_occupations(N, len(campaign.mode_labels))
        parts = [MBROrthogonalityExperiment(occupations, campaign.modes, cycle=cycle,
                                            calibration=calibration,
                                            sync_cycles=campaign.sync_cycles, reps=reps)
                 for cycle in cycles]

        jobs = len(parts) * len(occupations)
        total_points = jobs * 4 * len(occupations)
        # 3.76 s per point at 1200 reps: the estimate of the 7-1 campaign.
        serial_minutes = total_points * 3.76 * reps / 1200.0 / 60.0
        parallel_jobs = min(batch_size, len(occupations))
        print("basis order:", [tuple(occupation) for occupation in occupations])
        print(f"Floquet cycle={cycle_us:.6f} us; q={cycles}; step time={step * cycle_us:.6f} us; "
              f"max time={2 * step * cycle_us:.6f} us")
        print(f"jobs={jobs}, points/job={4 * len(occupations)}, total points={total_points}, "
              f"program repetitions={total_points * reps:,}")
        print(f"rough time={serial_minutes:.1f} min serialized, or "
              f"{serial_minutes / parallel_jobs:.1f} min with {parallel_jobs} concurrent workers, "
              f"plus overhead")
        return parts

    # -- analysis ---------------------------------------------------------

    def reconstruction(self):
        """-> the matrices of every part, stacked as ``(cycle, decoder, encoder)``."""
        self._check_children()
        for part in self.children:
            if "matrix" not in part.data:
                part.analyze()
        return AttrDict(dict(
            cycles=np.asarray(self.cycles, dtype=int),
            occupations=list(self.occupations),
            mode_labels=list(self.children[0].data.mode_labels),
            raw_matrices=np.stack([part.data.raw_matrix for part in self.children]),
            matrices=np.stack([part.data.matrix for part in self.children]),
            analysis_phase_per_cycle_deg=np.stack(
                [part.data.analysis_phase_per_cycle_deg for part in self.children]),
            matrix_orientation="rows=decoder, columns=encoder",
        ))

    def analyze(self, floquet_cycle_us=None, finite_difference_cycles=None,
                eigenphase_cycle=None):
        """Stack the matrices; with a calibration, also run the tomography.

        ``floquet_cycle_us`` defaults to the value saved with the jobs; the
        other two go to
        :func:`fitting.qsim.mbr_propagator.analyze_propagator_dynamics`.
        """
        data = self.reconstruction()
        data.hardware = self.children[0].data.hardware
        if floquet_cycle_us is None:
            floquet_cycle_us = data.hardware.floquet_cycle_us
        data.floquet_cycle_us = float(floquet_cycle_us)
        if self.calibration is not None:
            calibration = endpoint_calibration(self.calibration)
            data.update(analyze_propagator_dynamics(
                data, calibration, floquet_cycle_us,
                finite_difference_cycles=finite_difference_cycles,
                eigenphase_cycle=eigenphase_cycle))
            data.calibration = calibration
        self.data = data
        return data

    def display(self):
        """|M_q| per cycle count; with the tomography, its eigenfrequencies too."""
        if "matrices" not in self.data:
            self.analyze()
        data = self.data
        tomography = "eigenphase" in data
        matrices = np.asarray(data.endpoint_normalized_matrices if tomography
                              else data.matrices)
        count = len(self.cycles)
        fig, axes = plt.subplots(1, count + tomography, squeeze=False,
                                 figsize=(4.5 * (count + tomography), 4.5),
                                 constrained_layout=True)
        axes = axes[0]
        labels = [str(occupation) for occupation in self.occupations]
        ticks = np.arange(len(labels)) if len(labels) <= 10 else []
        for axis, cycle, matrix in zip(axes, self.cycles, matrices):
            image = axis.imshow(np.abs(matrix), origin="upper", cmap="magma", vmin=0.)
            axis.set(title=f"q = {cycle}", xlabel="encoder i", ylabel="decoder j",
                     xticks=ticks, yticks=ticks)
            if len(ticks):
                axis.set_xticklabels(labels, rotation=55, ha="right")
                axis.set_yticklabels(labels)
            fig.colorbar(image, ax=axis)
        if tomography:
            axis = axes[-1]
            eigenphase = data.eigenphase
            axis.plot(eigenphase.eigenfrequencies_MHz, "o",
                      label=f"eigenphase, q = {eigenphase.cycle}")
            if data.finite_difference is not None:
                axis.plot(data.finite_difference.eigenfrequencies_MHz, "x",
                          label="finite difference")
            axis.set(xlabel="index", ylabel="eigenfrequency (MHz)", title="spectrum")
            axis.legend()
        fig.suptitle(r"$|M_q|$" + (" (endpoint normalized)" if tomography else "")
                     + f", {len(labels)} occupations")
        return fig

    def analyze_shared_step(self, step):
        """:meth:`analyze` at q = [0, s, 2s], plus one shared s-cycle transfer matrix.

        Fits :func:`fitting.qsim.mbr_propagator.fit_shared_step` to the raw
        and to the phase-corrected matrices. The Stark correction is played on
        the pulse, so the two agree for new jobs; they differ only for
        converted old jobs that left the correction to analysis. For N = 1
        the zero-detuning theory is the star Hamiltonian: +/- |g| (the bright
        mode) and zeros. -> ``data``, with ``data.shared_step``.
        """
        if self.calibration is None:
            raise ValueError("the tomography needs the calibration set; "
                             "pass it to MBRHamTomoExperiment.from_parts")
        if self.cycles != [0, step, 2 * step]:
            raise ValueError(f"the shared-step fit needs q=[0, {step}, {2 * step}]; got {self.cycles}")
        data = self.analyze(finite_difference_cycles=self.cycles, eigenphase_cycle=step)
        cycle_us = data.floquet_cycle_us
        theory_MHz = None
        if all(sum(occupation) == 1 for occupation in self.occupations):
            bright_MHz = np.linalg.norm(np.asarray(data.hardware.couplings_MHz, dtype=float))
            theory_MHz = np.asarray([-bright_MHz] + [0.] * (len(self.occupations) - 2) + [bright_MHz])
        data.shared_step = AttrDict(dict(
            step=int(step),
            raw=fit_shared_step(data.raw_matrices, data.cycles, step, cycle_us),
            corrected=fit_shared_step(data.matrices, data.cycles, step, cycle_us),
            theory_frequencies_MHz=theory_MHz,
        ))

        print("finite difference used:", data.finite_difference is not None)
        print(f"cond(M0)={data.zero_cycle_condition_number:.3g}")
        print(f"endpoint-normalized M0 identity residual="
              f"{data.endpoint_normalized_zero_cycle_identity_residual:.3g}")
        print(f"calibration diagonal mismatch={data.calibration_diagonal_relative_mismatch:.3g}")
        print(f"q={step} two-depth GEVP E/h (kHz):",
              np.round(1e3 * data.eigenphase.eigenfrequencies_MHz, 3))
        print("three-depth shared-step fits (lower residual is better):")
        for label, fit in (("raw", data.shared_step.raw), ("phase corrected", data.shared_step.corrected)):
            print(f"  {label}: recurrence={fit['recurrence_residual']:.3g}, "
                  f"semigroup={fit['semigroup_residual']:.3g}")
            print("    E/h (kHz):", np.round(1e3 * fit["frequencies_MHz"], 3))
            print("    |lambda|:", np.round(fit["pole_radii"], 5))
        if theory_MHz is not None:
            print("zero-detuning theory E/h (kHz):", np.round(1e3 * theory_MHz, 3))
        print(f"step-fit alias period={1e3 / (step * cycle_us):.3f} kHz; "
              f"principal zone=+/-{0.5e3 * data.eigenphase.alias_period_MHz:.3f} kHz")
        return data

    def display_shared_step(self):
        """|M_0|, |M_s|, |M_2s|; the shared-step eigenvalues on the unit circle; the
        three-depth spectrum against theory."""
        data = self.data
        if "shared_step" not in data:
            raise ValueError("run analyze_shared_step(step) first")
        step = data.shared_step.step
        index = {int(cycle): position for position, cycle in enumerate(data.cycles)}
        raw, corrected = data.shared_step.raw, data.shared_step.corrected
        labels = data.mode_labels

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), constrained_layout=True)
        for axis, cycle, title in ((axes[0], 0, r"$|M_0|$"),
                                   (axes[1], step, rf"$|M_{{{step}}}|$"),
                                   (axes[2], 2 * step, rf"$|M_{{{2 * step}}}|$")):
            image = axis.imshow(np.abs(data.matrices[index[cycle]]), origin="upper", cmap="magma", vmin=0)
            axis.set_title(title)
            axis.set_xticks(np.arange(len(labels)), labels, rotation=45)
            axis.set_yticks(np.arange(len(labels)), labels)
            axis.set_xlabel("encoder")
            axis.set_ylabel("decoder")
            fig.colorbar(image, ax=axis)

        angle = np.linspace(0, 2 * np.pi, 500)
        axes[3].plot(np.cos(angle), np.sin(angle), color="black", linestyle=":", linewidth=1)
        axes[3].scatter(raw["poles"].real, raw["poles"].imag, s=55, marker="x", label="raw")
        axes[3].scatter(corrected["poles"].real, corrected["poles"].imag, s=55,
                        facecolors="none", edgecolors="C1", label="phase corrected")
        axes[3].axhline(0, color="0.8", linewidth=0.8)
        axes[3].axvline(0, color="0.8", linewidth=0.8)
        axes[3].set_aspect("equal")
        axes[3].set(xlabel=r"$\mathrm{Re}\,\lambda$", ylabel=r"$\mathrm{Im}\,\lambda$",
                    title="shared-step eigenvalues")
        axes[3].legend(fontsize=8)

        axes[4].scatter(np.zeros(len(raw["frequencies_MHz"])), 1e3 * raw["frequencies_MHz"],
                        s=55, marker="x", label="raw fit")
        axes[4].scatter(np.ones(len(corrected["frequencies_MHz"])), 1e3 * corrected["frequencies_MHz"],
                        s=55, facecolors="none", edgecolors="C1", label="phase-corrected fit")
        theory = data.shared_step.theory_frequencies_MHz
        if theory is not None:
            axes[4].scatter(2 * np.ones(len(theory)), 1e3 * theory, marker="x", s=65,
                            color="black", label="theory")
        axes[4].set_xticks([0, 1, 2], ["raw", "corrected", "theory"])
        axes[4].set(xlim=(-0.5, 2.5), ylabel=r"$E/h$ (kHz)", title="three-depth least-squares spectrum")
        plt.show()
        return fig

    # -- persistence ------------------------------------------------------

    def _child_files(self):
        """-> the parts' manifests; every part must be saved first."""
        unsaved = [part.cycle for part in self.children if part.manifest_path is None]
        if unsaved:
            raise ValueError(f"save() the Orthogonality parts at q = {unsaved} first")
        return [part.manifest_path for part in self.children]

    @classmethod
    def _load_children(cls, manifest, timing=None):
        return [cls.child_class.from_manifest(path, timing=timing)
                for path in manifest["raw_files"]]

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment

        calibration = manifest.get("calibration_manifest")
        if not calibration:
            return {}
        return dict(calibration=MBRCalibrationSetExperiment.from_manifest(calibration))

    def calibration_manifest(self):
        if self.calibration is None:
            return None
        return self.calibration.manifest_path

    def manifest_parameters(self):
        return dict(cycles=self.cycles,
                    occupations=[list(o) for o in self.occupations],
                    swap_stors=self.swap_stors)

    def assembled_arrays(self):
        data = self.data
        arrays = dict(
            cycles=data.cycles,
            occupations=np.asarray(self.occupations, dtype=int),
            raw_matrices=data.raw_matrices,
            matrices=data.matrices,
        )
        if "eigenphase" in data:
            arrays.update(
                endpoint_normalized_matrices=data.endpoint_normalized_matrices,
                eigenphase_eigenfrequencies_MHz=data.eigenphase.eigenfrequencies_MHz,
                eigenphase_generalized_eigenvalues=data.eigenphase.generalized_eigenvalues,
            )
            if data.finite_difference is not None:
                arrays.update(
                    finite_difference_eigenfrequencies_MHz=(
                        data.finite_difference.eigenfrequencies_MHz),
                    effective_hamiltonian_MHz=data.finite_difference.effective_hamiltonian_MHz,
                )
        return arrays

    def assembled_attrs(self):
        data = self.data
        attrs = dict(
            floquet_cycle_us=float(data.floquet_cycle_us),
            hardware_source=str(data.hardware.source),
            mode_labels=list(data.mode_labels),
            matrix_orientation=str(data.matrix_orientation),
        )
        if "eigenphase" in data:
            attrs.update(eigenphase_cycle=int(data.eigenphase.cycle),
                         zero_cycle_condition_number=float(data.zero_cycle_condition_number))
        return attrs

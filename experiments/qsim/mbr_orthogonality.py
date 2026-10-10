# -*- coding: utf-8 -*-
"""The matrix M_q[decoder, initial] from ortho columns: MBROrthogonalityExperiment.

An assembled class (docs/qsim/mbr_redesign.md, sections 1, 2 and 5): one
:class:`MBROrthoColumnExperiment` per initial occupation, all at the same
number of Floquet cycles ``q``. It never goes to the worker itself. At
``q = 0`` (the default) the matrix is the encoder/decoder overlap: its
off-diagonal part is the leakage between access paths.

    ortho = MBROrthogonalityExperiment(occupations, swap_stors=[2, 3, 4, 5])
    ortho.acquire(runner, batch_size=10)   # runner.ExptClass is MBROrthoColumnExperiment
    ortho.analyze(); ortho.display()
    ortho.save()                           # manifest YAML + assembled HDF5

    ortho = MBROrthogonalityExperiment.from_manifest(path)

At ``q > 0`` a ``calibration`` (a saved :class:`MBRCalibrationSetExperiment`)
supplies each decoder's Stark-shift correction, played on the pulse.
``MBRHamTomoExperiment`` combines these matrices at q = 0, q, 2q, ...

The matrix arithmetic and the display are carried over from the old
``MBROrthogonalityExperiment`` (now in ``deprecated/legacy_mbr.py``) without
changes.
"""
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.mbr_ortho_column import MBROrthoColumnExperiment
from experiments.qsim.mbr_saved import saved_parameters


class MBROrthogonalityExperiment(AssembledExperiment):
    """Ortho columns over the initial occupations, all at one ``q``."""

    child_class = MBROrthoColumnExperiment

    def __init__(self, occupations, swap_stors, cycle=0, calibration=None,
                 cycle_branches=0, detunings=None, sync_cycles=10, reps=300,
                 notes=""):
        """``occupations`` are both the initial and the decoded states, so
        the matrix is square. ``cycle_branches`` picks the 180 deg/cycle
        branch of the calibration's correction."""
        super().__init__(notes=notes)
        self.occupations = [tuple(int(n) for n in o) for o in occupations]
        self.swap_stors = [int(stor) for stor in swap_stors]
        self.cycle = int(cycle)
        self.calibration = calibration
        self.cycle_branches = cycle_branches
        self.detunings = (None if detunings is None
                          else [float(d) for d in detunings])
        self.sync_cycles = int(sync_cycles)
        self.reps = int(reps)

    # -- acquisition ------------------------------------------------------

    def job_overrides(self):
        """-> one OrthoColumn override dict per initial occupation.

        Each decoder's analyzer correction comes from
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
        decoder_phases = [phases[occupation] for occupation in self.occupations]
        return [self.child_class.job_config(
                    initial, self.occupations, self.swap_stors, cycle=self.cycle,
                    decoder_phases_deg=decoder_phases, calibration_manifest=manifest,
                    detunings=self.detunings, sync_cycles=self.sync_cycles,
                    reps=self.reps)
                for initial in self.occupations]

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", calibration=None):
        """Assemble already acquired or loaded OrthoColumn jobs.

        The occupations are the jobs' decoder list, which must also be the
        set of their initial occupations. The jobs (and ``job_ids``, if one
        per job) are put in that order, so job ``i`` is column ``i``.
        ``calibration`` is the set the jobs were acquired with, if any.
        """
        children = list(children)
        if not children:
            raise ValueError("an orthogonality matrix needs at least one OrthoColumn job")
        first = children[0]
        occupations = first.decoder_occupations
        swap_stors = [int(stor) for stor in first.cfg.expt.swap_stors]
        for child in children:
            if child.decoder_occupations != occupations:
                raise ValueError(f"{child.initial_occupation}: different decoders")
            if child.cycle != first.cycle:
                raise ValueError(f"{child.initial_occupation}: q = {child.cycle}, "
                                 f"not {first.cycle}")
            if [int(stor) for stor in child.cfg.expt.swap_stors] != swap_stors:
                raise ValueError(f"{child.initial_occupation}: different swap modes")
        initials = [child.initial_occupation for child in children]
        if len(set(initials)) != len(initials):
            raise ValueError("each initial occupation must appear once")
        if set(initials) != set(occupations):
            raise ValueError("the initial occupations must be the decoded occupations")

        order = [initials.index(occupation) for occupation in occupations]
        job_ids = list(job_ids)
        if len(job_ids) == len(children):
            job_ids = [job_ids[i] for i in order]
        ecfg = first.cfg.expt
        ortho = cls(occupations, swap_stors, cycle=first.cycle, calibration=calibration,
                    detunings=ecfg.get("detunings", None),
                    sync_cycles=int(ecfg.get("scramble_sync_cycles", 10)),
                    reps=int(ecfg.reps), notes=notes)
        ortho.children = [children[i] for i in order]
        ortho.job_ids = job_ids
        ortho._check_children()
        return ortho

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment

        calibration = manifest.get("calibration_manifest")
        if not calibration:
            return {}
        return dict(calibration=MBRCalibrationSetExperiment.from_manifest(calibration))

    # -- analysis ---------------------------------------------------------

    def analyze(self):
        """The matrix ``M[j, i]``: rows are decoders, columns initial occupations.

        ``raw_matrix`` is the jobs' returns as acquired. ``matrix`` also
        applies any correction a converted job left to analysis
        (``analysis_phase_per_cycle_deg``); for new jobs the two are equal.
        The raw matrix is deliberately not divided by its diagonal: those
        values are the diagnostic.
        """
        self._check_children()
        for child in self.children:
            if "complex_return" not in child.data:
                child.analyze()
        raw_matrix = np.column_stack([child.data["complex_return"]
                                      for child in self.children]).astype(complex)
        correction = np.column_stack([child.analysis_phase_per_cycle_deg
                                      for child in self.children])
        matrix = raw_matrix * np.exp(-1j * np.deg2rad(self.cycle * correction))

        amplitude = np.abs(matrix)
        power = amplitude ** 2
        diagonal_amplitude = np.abs(np.diag(matrix))
        denominator = np.outer(diagonal_amplitude, diagonal_amplitude)
        normalized_power = power / denominator
        normalized_amplitude = matrix / np.sqrt(denominator)
        offdiagonal_normalized_power = normalized_power.copy()
        np.fill_diagonal(offdiagonal_normalized_power, 0.)
        column_leakage = np.sum(offdiagonal_normalized_power, axis=0)

        saved = saved_parameters(self.children)
        self.data = AttrDict(dict(
            occupations=list(self.occupations),
            mode_labels=saved.mode_labels,
            cycle=self.cycle,
            raw_matrix=raw_matrix,
            matrix=matrix,
            analysis_phase_per_cycle_deg=correction,
            amplitude=amplitude,
            power=power,
            diagonal_amplitude=diagonal_amplitude,
            normalized_amplitude=normalized_amplitude,
            normalized_power=normalized_power,
            offdiagonal_normalized_power=offdiagonal_normalized_power,
            column_leakage=column_leakage,
            matrix_orientation="rows=decoder, columns=encoder",
            hardware=saved.hardware,
        ))
        return self.data

    def display(self, figsize=None):
        """Raw |M|, raw off-diagonal |M|, and normalized off-diagonal power.

        ``figsize`` defaults to a width that grows with the matrix, so a
        larger basis stays legible.
        """
        if "matrix" not in self.data:
            self.analyze()
        data = self.data
        matrix = np.asarray(data.matrix, dtype=complex)
        labels = [str(tuple(occupation)) for occupation in data.occupations]
        size = len(labels)

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
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        for axis, (values, title) in zip(axes, panels):
            image = axis.imshow(values, origin="upper", aspect="equal", cmap="magma", vmin=0.)
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
                        axis.text(column, row, text,
                                  ha="center", va="center", color="cyan", fontsize=8)

        finite_offdiagonal = np.asarray(data.offdiagonal_normalized_power, dtype=float).copy()
        np.fill_diagonal(finite_offdiagonal, np.nan)
        max_leakage = (float(np.nanmax(finite_offdiagonal))
                       if np.any(np.isfinite(finite_offdiagonal)) else np.nan)
        fig.suptitle(
            f"q = {self.cycle} encoder/decoder cross return; "
            f"min diagonal |M|={np.min(data.diagonal_amplitude):.3f}; "
            f"max normalized off-diagonal power={max_leakage:.3g}"
        )
        return fig

    # -- persistence ------------------------------------------------------

    def calibration_manifest(self):
        if self.calibration is None:
            return None
        return self.calibration.manifest_path

    def manifest_parameters(self):
        return dict(occupations=[list(o) for o in self.occupations],
                    swap_stors=self.swap_stors,
                    cycle=self.cycle)

    def assembled_arrays(self):
        data = self.data
        return dict(
            occupations=np.asarray(self.occupations, dtype=int),
            raw_matrix=data.raw_matrix,
            matrix=data.matrix,
            analysis_phase_per_cycle_deg=data.analysis_phase_per_cycle_deg,
            normalized_power=data.normalized_power,
            column_leakage=data.column_leakage,
            couplings_MHz=data.hardware.couplings_MHz,
        )

    def assembled_attrs(self):
        data = self.data
        return dict(
            cycle=int(self.cycle),
            matrix_orientation=str(data.matrix_orientation),
            floquet_cycle_us=float(data.hardware.floquet_cycle_us),
            hardware_source=str(data.hardware.source),
            mode_labels=list(data.mode_labels),
        )

# -*- coding: utf-8 -*-
"""Level statistics over disorder realizations: MBRDisorderEnsembleExperiment.

An assembled class of assembled parts (docs/qsim/mbr_redesign.md, sections 2
and 5; docs/qsim/mbr_step7_plan.md, section 3): one
:class:`MBRSpectrumExperiment` per disorder realization, all diagonal, in one
photon-number sector. A realization is one set of pulse detunings; only the
detunings and the realization record change between parts. Acquisition takes
hours, so there is no ``acquire``; the notebook acquires and saves each part,
then combines them::

    parts = []
    for record in realizations:            # e.g. from plan_diagonal_disorder
        spectrum = MBRSpectrumExperiment(record["occupations"], cycles, swap_stors,
                                         calibration=cal,
                                         detunings=record["detunings"])
        spectrum.acquire(runner); spectrum.analyze(); spectrum.save()
        parts.append(spectrum)
    ensemble = MBRDisorderEnsembleExperiment.from_parts(parts, realizations=realizations,
                                                        calibration=cal)
    ensemble.analyze(); ensemble.display()
    ensemble.save()                        # manifest lists the parts' manifests

    ensemble = MBRDisorderEnsembleExperiment.from_manifest(path)

``analyze`` runs, per part, Matrix Pencil on the spectrum, matches the poles
to the part's theory levels, and takes adjacent-gap ratios of both; then it
pools them over realizations (:mod:`fitting.qsim.mbr_disorder`). When every
part covers the complete basis it also gives Tr U(t) and the spectral form
factor; no dataset converted so far is complete (the plan, section 3.2).
"""
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from fitting.qsim import mbr_disorder
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian

#: The Matrix Pencil settings of the 7-1 campaign's analysis cell (cell 332).
DEFAULT_MATRIX_PENCIL = dict(
    mpm_match_decay=False,
    mpm_track_frequency_tolerance_bins=1.0,
    mpm_merge_frequency_tolerance_bins=0.10,
    mpm_dedup_frequency_tolerance_bins=0.10,
    mpm_minimum_supporting_rows=1,
)


class MBRDisorderEnsembleExperiment(AssembledExperiment):
    """Spectrum parts over disorder realizations, and their level statistics."""

    child_class = MBRSpectrumExperiment

    def __init__(self, realizations, calibration=None, notes=""):
        """``realizations`` is one record (dict) per part, in part order.

        A record holds ``realization`` (its index) and whatever the plan knew:
        ``seed``, ``strength_kHz``, ``direction``, ``onsite_MHz``,
        ``selected_occupations``, ``selection_floor``, ``self_kerr_kHz``, ...
        It goes to the manifest unchanged.
        """
        super().__init__(notes=notes)
        self.realizations = [dict(record) for record in realizations]
        self.calibration = calibration

    def acquire(self, runner, batch_size=10, **execute_kwargs):
        raise NotImplementedError(
            "MBRDisorderEnsembleExperiment is built with from_parts(): acquire and "
            "save one MBRSpectrumExperiment per realization first")

    @classmethod
    def from_parts(cls, spectra, realizations=None, calibration=None, notes=""):
        """Combine Spectrum objects, one per realization.

        They must be one photon-number sector with the same swap modes and
        cycles. ``realizations`` defaults to ``{"realization": index}``
        records. A record with ``onsite_MHz`` must match its part's detunings
        (onsite = -detuning). ``calibration`` defaults to the parts' shared
        calibration set; parts with different calibration manifests are
        refused.
        """
        parts = list(spectra)
        if not parts:
            raise ValueError("a disorder ensemble needs at least one Spectrum")
        if realizations is None:
            realizations = [dict(realization=index) for index in range(len(parts))]
        realizations = [dict(record) for record in realizations]
        if len(realizations) != len(parts):
            raise ValueError(f"{len(parts)} parts but {len(realizations)} realization records")
        labels = [record.get("realization") for record in realizations]
        if None in labels or len(set(labels)) != len(labels):
            raise ValueError(f"each record needs a unique 'realization'; got {labels}")

        first = parts[0]
        sectors = {sum(occupation) for part in parts for occupation in part.occupations}
        if len(sectors) != 1:
            raise ValueError(f"parts span photon numbers {sorted(sectors)}")
        for record, part in zip(realizations, parts):
            label = record["realization"]
            if part.swap_stors != first.swap_stors:
                raise ValueError(f"r={label}: different swap modes")
            if part.cycles != first.cycles:
                raise ValueError(f"r={label}: different cycles")
            if part.detunings is None:
                raise ValueError(f"r={label}: the part records no detunings")
            if "onsite_MHz" in record and not np.allclose(
                    -np.asarray(part.detunings), np.asarray(record["onsite_MHz"])):
                raise ValueError(f"r={label}: onsite_MHz is not -detunings of its part")

        manifests = {str(part.calibration.manifest_path) for part in parts
                     if part.calibration is not None and part.calibration.manifest_path}
        if len(manifests) > 1:
            raise ValueError(f"parts use different calibration sets: {sorted(manifests)}")
        if calibration is None:
            calibration = next((part.calibration for part in parts
                                if part.calibration is not None), None)

        ensemble = cls(realizations, calibration=calibration, notes=notes)
        ensemble.children = parts
        ensemble.job_ids = [job_id for part in parts for job_id in part.job_ids]
        ensemble._check_children()
        return ensemble

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", calibration=None,
                      realizations=None):
        """``from_parts`` with the job IDs a manifest recorded."""
        ensemble = cls.from_parts(children, realizations=realizations,
                                  calibration=calibration, notes=notes)
        ensemble.job_ids = list(job_ids)
        return ensemble

    def part(self, realization):
        """-> the Spectrum part of one realization index."""
        labels = [record["realization"] for record in self.realizations]
        if realization not in labels:
            raise KeyError(f"r={realization} is not in this ensemble; available: {labels}")
        return self.children[labels.index(realization)]

    # -- analysis ---------------------------------------------------------

    def analyze(self, phase_frame="as_acquired", manual_kerr_MHz=None,
                cycle_branches=0, legacy=None, theory_kerr_MHz=None,
                match_tolerance_bins=1.5, edge_fraction=0.10,
                excluded_occupations=(), on_error="raise", **matrix_pencil_options):
        """Matrix Pencil levels per realization, matched to theory; pooled gap ratios.

        - ``phase_frame``, ``manual_kerr_MHz``, ``cycle_branches``, ``legacy``
          go to each part's ``analyze``; ``mpm_*`` options too, over
          :data:`DEFAULT_MATRIX_PENCIL` (``mpm_requested_max_modes`` defaults
          to the sector dimension).
        - Theory levels are each part's ``spectrum.energies_MHz`` (its own
          detunings, couplings and analysis Kerr). ``theory_kerr_MHz`` rebuilds
          them with another Kerr; ``"recorded"`` uses each realization
          record's ``self_kerr_kHz``.
        - ``excluded_occupations`` are left out of every part's analysis (the
          7-1 preview left out ``(0, 3, 0, 0, 0)``); the parts keep their jobs.
        - A pole matches a level within ``match_tolerance_bins`` FFT bins.
        - ``edge_fraction`` of the levels is cut at each edge before the gap
          ratios.
        - ``on_error="skip"`` records a part whose analysis fails and goes on;
          ``"raise"`` stops.
        """
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        self._check_children()
        options = dict(DEFAULT_MATRIX_PENCIL, **matrix_pencil_options)
        excluded = {tuple(int(n) for n in o) for o in excluded_occupations}
        records = []
        self._analyzed_parts = []
        for record, part in zip(self.realizations, self.children):
            label = record["realization"]
            if excluded:
                kept = [c for c in part.children if c.initial_occupation not in excluded]
                part = MBRSpectrumExperiment.from_children(
                    kept, calibration=part.calibration, notes=part.notes)
            self._analyzed_parts.append(part)
            kerr = theory_kerr_MHz
            if isinstance(kerr, str):
                if kerr != "recorded":
                    raise ValueError("theory_kerr_MHz must be a number, None or 'recorded'")
                kerr = 1e-3 * float(record["self_kerr_kHz"])
            try:
                records.append(self._analyze_part(
                    label, part, options, phase_frame, manual_kerr_MHz,
                    cycle_branches, legacy, kerr, match_tolerance_bins))
            except Exception as error:  # noqa: BLE001 -- recorded, per on_error
                if on_error == "raise":
                    raise
                records.append(AttrDict(realization=label, error=repr(error)))
                print(f"r={label}: analysis failed: {error!r}")

        analyzed = [r for r in records if "error" not in r]
        if not analyzed:
            raise RuntimeError("no disorder realization was analyzed")
        dimensions = {r.dimension for r in analyzed}
        if len(dimensions) != 1:
            raise RuntimeError(f"mixed Hilbert-space dimensions: {sorted(dimensions)}")
        dimension = dimensions.pop()
        trim = mbr_disorder.trim_count(edge_fraction, dimension)

        measured_ratios, theory_ratios, ratio_failures = {}, {}, {}
        for r in analyzed:
            r.theory_ratios = mbr_disorder.adjacent_gap_ratios(r.theory_levels_MHz, trim)
            theory_ratios[r.realization] = r.theory_ratios
            try:
                r.measured_ratios = mbr_disorder.adjacent_gap_ratios(r.poles_MHz, trim)
                measured_ratios[r.realization] = r.measured_ratios
            except ValueError as error:
                r.measured_ratios = None
                ratio_failures[r.realization] = f"{len(r.poles_MHz)} poles: {error}"

        data = AttrDict(dict(
            realizations=records,
            dimension=int(dimension),
            trim_count=int(trim),
            edge_fraction=float(edge_fraction),
            theory=mbr_disorder.pooled_statistics(theory_ratios),
            measured=(mbr_disorder.pooled_statistics(measured_ratios)
                      if measured_ratios else None),
            ratio_failures=ratio_failures,
            analysis_settings=dict(phase_frame=phase_frame, manual_kerr_MHz=manual_kerr_MHz,
                                   theory_kerr_MHz=theory_kerr_MHz,
                                   match_tolerance_bins=match_tolerance_bins,
                                   excluded_occupations=sorted(excluded),
                                   **{k: v for k, v in options.items()}),
        ))
        parts = self._analyzed_parts
        if (len(analyzed) == len(parts)
                and all(part.data.spectrum.complete_basis for part in parts)):
            data.form_factor = mbr_disorder.trace_and_form_factor(
                [part.data.reconstruction.A_norm for part in parts])
            data.form_factor.time_us = np.asarray(parts[0].data.spectrum.time_us)
        else:
            data.form_factor = None
        self.data = data
        return data

    @staticmethod
    def _analyze_part(label, part, options, phase_frame, manual_kerr_MHz,
                      cycle_branches, legacy, theory_kerr_MHz, match_tolerance_bins):
        """-> one realization's record: poles, theory levels, the match."""
        photon_number = sum(part.occupations[0])
        mode_count = len(part.occupations[0])
        hamiltonian_dimension = comb(photon_number + mode_count - 1, photon_number)
        part_options = dict(options)
        part_options.setdefault("mpm_requested_max_modes", hamiltonian_dimension)
        data = part.analyze(phase_frame=phase_frame, manual_kerr_MHz=manual_kerr_MHz,
                            cycle_branches=cycle_branches, legacy=legacy,
                            spectrum_method="matrix_pencil", **part_options)
        # The source wrapped the poles into the principal interval, then
        # commented the modulo out; they are used as fitted.
        poles_MHz = np.sort(np.asarray(data.matrix_pencil.selected_frequencies_MHz,
                                       dtype=float))
        if theory_kerr_MHz is None:
            theory_levels_MHz = np.asarray(data.spectrum.energies_MHz, dtype=float)
        else:
            theory_levels_MHz = fixed_n_hamiltonian(
                photon_number, mode_count, data.detunings,
                data.hardware.couplings_MHz, theory_kerr_MHz).energies_MHz
        theory_levels_MHz = np.sort(theory_levels_MHz)
        nyquist_MHz = 0.5 * float(data.matrix_pencil.sampling.sampling_frequency_MHz)
        if np.any(np.abs(theory_levels_MHz) > nyquist_MHz):
            raise RuntimeError(f"r={label}: theory levels lie outside Nyquist")
        match = mbr_disorder.match_levels(
            poles_MHz, theory_levels_MHz,
            match_tolerance_bins * float(data.spectrum.fft_resolution_MHz))
        print(f"r={label}: poles={len(poles_MHz)}, "
              f"matched={match.matched_count}/{len(theory_levels_MHz)}, "
              f"missing={len(match.missing_theory_MHz)}, "
              f"spurious={len(match.spurious_poles_MHz)}, "
              f"MAE={1e3 * match.mae_MHz:.3f} kHz")
        return AttrDict(dict(
            realization=label,
            dimension=int(len(theory_levels_MHz)),
            poles_MHz=poles_MHz,
            theory_levels_MHz=theory_levels_MHz,
            match=match,
            onsite_MHz=-np.asarray(data.detunings, dtype=float),
            fft_resolution_MHz=float(data.spectrum.fft_resolution_MHz),
        ))

    def display(self, gap_ratio_bins=15):
        """Pooled gap-ratio histogram, realization means; the SFF if complete."""
        if "theory" not in self.data:
            self.analyze()
        data = self.data
        panels = 3 if data.form_factor is not None else 2
        fig, axes = plt.subplots(1, panels, figsize=(6.2 * panels, 4.6),
                                 constrained_layout=True)
        edges = np.linspace(0.0, 1.0, gap_ratio_bins + 1)
        axis = axes[0]
        if data.measured is not None:
            axis.hist(data.measured.pooled, bins=edges, density=True, alpha=0.35,
                      color="black", edgecolor="black",
                      label=f"measured: n={len(data.measured.pooled)}, "
                            f"mean={np.mean(data.measured.pooled):.3f}")
        axis.hist(data.theory.pooled, bins=edges, density=True, histtype="step",
                  linewidth=2, color="tab:green",
                  label=f"theory: n={len(data.theory.pooled)}, "
                        f"mean={np.mean(data.theory.pooled):.3f}")
        ratio = np.linspace(0.0, 1.0, 1000)
        axis.plot(ratio, mbr_disorder.poisson_pdf(ratio), color="tab:blue", lw=2,
                  label="Poisson")
        axis.plot(ratio, mbr_disorder.goe_pdf(ratio), color="tab:orange", lw=2,
                  label="GOE")
        axis.set(xlim=(0, 1), xlabel=r"adjacent-gap ratio $\tilde r$",
                 ylabel="probability density",
                 title=f"pooled level statistics, D={data.dimension}, trim={data.trim_count}")
        axis.legend()

        axis = axes[1]
        labels = data.theory.realizations
        x = {label: i for i, label in enumerate(labels)}
        axis.scatter(range(len(labels)), data.theory.realization_means,
                     color="tab:green", label="theory")
        if data.measured is not None:
            axis.scatter([x[label] for label in data.measured.realizations],
                         data.measured.realization_means, color="black", marker="x",
                         s=70, label="measured")
            axis.errorbar(len(labels), data.measured.mean, yerr=data.measured.sem,
                          fmt="D", capsize=4, color="tab:red",
                          label=f"measured mean={data.measured.mean:.3f} "
                                f"+/- {data.measured.sem:.3f}")
        axis.axhline(mbr_disorder.POISSON_MEAN, color="tab:blue", ls="--",
                     label=f"Poisson mean={mbr_disorder.POISSON_MEAN:.3f}")
        axis.axhline(mbr_disorder.GOE_MEAN, color="tab:orange", ls="--",
                     label=f"GOE mean={mbr_disorder.GOE_MEAN:.3f}")
        axis.set_xticks(list(range(len(labels) + 1)),
                        [f"r={label}" for label in labels] + ["mean"])
        axis.set(ylim=(0, 1), ylabel=r"mean $\langle\tilde r\rangle$",
                 title="realization means")
        axis.legend(fontsize="small")

        if data.form_factor is not None:
            axis = axes[2]
            axis.semilogy(data.form_factor.time_us, data.form_factor.sff_normalized, "k.-")
            axis.set(xlabel="time (us)", ylabel=r"$\langle|\mathrm{Tr}\,U|^2\rangle / D^2$",
                     title=f"spectral form factor, {len(self.children)} realizations")
        return fig

    def display_levels(self, realization, match_tolerance_bins=None, xlim_kHz=None):
        """Measured poles against theory levels for one realization, matches joined.

        ``match_tolerance_bins`` re-matches at another tolerance; by default
        the match of ``analyze`` is shown.
        """
        if "theory" not in self.data:
            self.analyze()
        found = [r for r in self.data.realizations if r.realization == realization]
        if not found:
            raise KeyError(f"r={realization} is not in this ensemble")
        record = found[0]
        if "error" in record:
            raise RuntimeError(f"r={realization} analysis failed: {record.error}")
        match = record.match
        if match_tolerance_bins is not None:
            match = mbr_disorder.match_levels(
                record.poles_MHz, record.theory_levels_MHz,
                match_tolerance_bins * record.fft_resolution_MHz)
        poles_kHz = 1e3 * record.poles_MHz
        theory_kHz = 1e3 * record.theory_levels_MHz
        fig, axis = plt.subplots(figsize=(13.0, 4.2), constrained_layout=True)
        axis.vlines(poles_kHz, -0.6, -0.1, color="black", lw=1.5)
        axis.vlines(theory_kHz, 0.1, 0.6, color="tab:green", lw=1.5)
        for pole_row, theory_row in zip(match.pole_rows, match.theory_rows):
            axis.plot([poles_kHz[pole_row], theory_kHz[theory_row]], [-0.1, 0.1],
                      color="tab:gray", lw=0.8)
        if xlim_kHz is not None:
            axis.set_xlim(*xlim_kHz)
        axis.set_ylim(-0.75, 0.75)
        axis.set_yticks([-0.35, 0.35], ["experiment", "theory"])
        axis.grid(axis="x", alpha=0.2)
        axis.set(xlabel="level (kHz)",
                 title=f"r={realization}: {len(poles_kHz)} poles, matched "
                       f"{match.matched_count}/{len(theory_kHz)} within "
                       f"{1e3 * match.tolerance_MHz:.3f} kHz")
        return fig

    # -- persistence ------------------------------------------------------

    def calibration_manifest(self):
        if self.calibration is None:
            return None
        return self.calibration.manifest_path

    def manifest_parameters(self):
        return dict(realizations=self.realizations,
                    analysis_settings=self.data.get("analysis_settings", {}))

    def assembled_arrays(self):
        data = self.data
        analyzed = {r.realization: r for r in data.realizations if "error" not in r}
        labels = [record["realization"] for record in self.realizations]
        dimension = data.dimension
        width = max([dimension] + [len(r.poles_MHz) for r in analyzed.values()])
        ratio_width = max(1, dimension - 2 * data.trim_count - 2)

        def padded(values, size):
            row = np.full(size, np.nan)
            if values is not None:
                row[:len(values)] = values
            return row

        arrays = dict(
            realization=np.asarray(labels, dtype=int),
            detunings=np.asarray([part.detunings for part in self.children], dtype=float),
            levels_MHz=np.asarray([padded(analyzed[l].poles_MHz if l in analyzed else None,
                                          width) for l in labels]),
            theory_levels_MHz=np.asarray([padded(analyzed[l].theory_levels_MHz
                                                 if l in analyzed else None, dimension)
                                          for l in labels]),
            measured_gap_ratios=np.asarray([padded(analyzed[l].measured_ratios
                                                   if l in analyzed else None, ratio_width)
                                            for l in labels]),
            theory_gap_ratios=np.asarray([padded(analyzed[l].theory_ratios
                                                 if l in analyzed else None, ratio_width)
                                          for l in labels]),
        )
        if data.form_factor is not None:
            arrays.update(trace_U=data.form_factor.trace, sff=data.form_factor.sff,
                          time_us=data.form_factor.time_us)
        return arrays

    def assembled_attrs(self):
        data = self.data
        return dict(
            dimension=int(data.dimension),
            trim_count=int(data.trim_count),
            theory_mean_gap_ratio=float(data.theory.mean),
            measured_mean_gap_ratio=(float(data.measured.mean)
                                     if data.measured is not None else float("nan")),
            measured_sem_gap_ratio=(float(data.measured.sem)
                                    if data.measured is not None else float("nan")),
        )

    def _child_files(self):
        """-> the parts' manifests; every part must be saved first."""
        unsaved = [record["realization"] for record, part
                   in zip(self.realizations, self.children) if part.manifest_path is None]
        if unsaved:
            raise ValueError(f"save() the Spectrum parts of realizations {unsaved} first")
        return [part.manifest_path for part in self.children]

    @classmethod
    def _load_children(cls, manifest, timing=None):
        return [cls.child_class.from_manifest(path, timing=timing)
                for path in manifest["raw_files"]]

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment

        kwargs = dict(realizations=manifest["parameters"]["realizations"])
        calibration = manifest.get("calibration_manifest")
        if calibration:
            kwargs["calibration"] = MBRCalibrationSetExperiment.from_manifest(calibration)
        return kwargs

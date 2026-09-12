# Evidence behind the Qsim surface map

Surveyed 2026-09-11: local main dd9624f and guan 1325e4b. No notebooks were executed. Cell numbers below are one-based physical cell positions, including markdown, not execution counters. Code-line counts include comments and blank lines, exclude outputs and markdown. Section headings, code-cell references, definitions and selected implementations were inspected; this is not a complete call graph.

## Notebook entry points (source notebooks, not final destinations)

Q = `measurement_notebooks/jonginn/qsim_experiments.ipynb` (374 cells, 275 code cells, 10,100 code lines).
P = `measurement_notebooks/jonginn/data_postprocess.ipynb` (306 cells, 230 code cells, 12,990 code lines).

| Family | Q cells | P cells | Distinct work / repeated variants |
|---|---|---|---|
| Setup, loading, shared helpers | 1–6, 58–68 | 1–4 | P4 alone has 1,563 lines spanning loaders, fitting, Wigner, multiparity and disorder analysis. |
| Broadband qubit and multiphoton swap calibration | 7–57 | — | Rabi, error amplification, endpoint validation and explicit calibration updates. |
| Floquet pulse calibration | 69–122 | Helpers in 4; pulse visualization 154–166 | Frequency/gain chevrons, error amplification, Stark phase matrix; waveform variants. |
| Cooling | 123–149 | — | Guan's earlier cooling experiment, copied into this notebook; drop this section as it was never run here and original lives elsewhere. |
| Literal dark-mode work | 150–262 | 104–153 | Bare/displaced/Fock inputs, parity/multiparity/confusion correction, large-support basis transforms, disorder decay, T1, relative Ramsey/phase. |
| Debug and ordinary sideband calibration | 263–285 | 154–166 | Scratch programs and ordinary sideband chevrons; not all belong to the god module. |
| MBR phase calibration | 288–299 | 174–175, 207, 222, 243, 264–266 | Load or acquire calibration; replace selected occupation rows; phase-frame handling. |
| Encoder/decoder orthogonality | 300–303 | — | Zero-cycle cross-return matrix. |
| Spectroscopy | 304–314 | 167–197, 205–240 | Diagonal/off-diagonal returns, N=1/2/3, FFT/MPM, Kerr scan, old-frame reprocessing and report plots. Different data grids and saved formats are migration cases. |
| Propagator / Hamiltonian tomography | 315–317, 350–358 | — | Reconstruct matrices; three-depth shared-transfer fit is additional notebook-local analysis. |
| Disorder spectroscopy campaigns | 318–349 | 241–282 | Multiple occupation/channel selectors; acquisition planning/caches; per-realization spectra, matching, level statistics. |
| Saved-shot estimator explorations | — | 193–203 | Subsampling and randomized occupation/phase sampling from existing data; distinct estimators, no new measurement required. |
| Additional spectral estimators | — | 252–254, 277–306 | Block-Hankel, component reweighting, merge sensitivity, shared-frequency nonlinear fits, independent shot halves, precision probes and row-pooled selection. |
| Direct disorder SFF | 359–369 | Randomized trace/SFF exploration 198–203 is related but different | Separate visibility and hardware depth-sweep acquisition, independent replicas and disorder ensemble estimator. Main deletes its Experiment while notebook source still references it; guan preserves it provisionally. |
| Coherent-displacement Floquet Kerr | 370–374 | — | Separate displacement/closed-cycle experiment; not the spectroscopy peak-overlap Kerr scan. |
| Flux-excursion / cavity Ramsey history | — | 5–93 | Sweep plots, HDF5/pickle loading, manual Ramsey/Kerr refits, transition scans, parameter maps. |
| Wigner | — | 94–103; helpers in 4 | Reconstruction, readout correction, fidelity/purity comparison; outside the two large modules' main responsibilities. |

## What the branch has and what notebook migration still entails

- The new MBR Experiment classes are intended to own series-aware acquisition/analysis orchestration. Four aggregate owners exist on guan: `mbr_phase_correction`, `mbr_orthogonality`, `mbr_spectrum`, `mbr_propagator`. Their old notebook `stage=` entry points require migration.
- Numerical work already has homes in `fitting/qsim`; several whole measurement families have been extracted. This does not migrate notebook-local campaign selection, fitting or reports automatically.
- P261 defines `SavedSpectroscopyExperiment`, overriding historical parameters and reconstruction to address saved decoder/physical-clock differences. P262 classifies datasets and incomplete acquisitions. Reconcile those semantics with the new loading layer instead of mechanically changing the superclass/import.
- P290–291 define joint shared-pole fitting; P294–295 add precision fitting/search. No definitions with the inspected names were found in guan's `experiments/` or `fitting/`. These are additional notebook-resident implementation surface.
- The standard spectrum route and exploratory estimators need separate acceptance decisions. Their coexistence does not imply that every alternative becomes a supported library API.

## Following the notebook cells is only part of the work

The dark-mode module contains 27 classes. Eleven have no literal class-name reference in either notebook's code cells: `DarkBaseProgram`, `DarkBaseRProgram`, `HardwareFloquetDepthSweepMixin`, `DisorderSFFSequenceMixin`, `DisorderSFFDepthSweepProgram`, `EncodingStarkShiftCalibrationProgram`, `SinglePhotonFloquetSpectroscopyProgram`, `FloquetPhaseAccumulationProgram`, `ManStorScrambleProgram`, `KerrWaitProgramDark`, `SidebandStarkAmplificationModifiedProgram_newold`.

This is a lexical finding, not an unused-code list: bases and mixins are reached indirectly; dynamic dispatch, aliases, saved class names and other notebooks also matter. Even reaching a class does not exercise every configuration branch. Conversely, the notebooks reference deleted classes and implement substantial mathematics outside this module.

Use both directions: notebook workflow → dependencies, then module symbol → consumer / shared prerequisite / retained historical code / explicit retirement. Do not infer deletion from absence in these two notebooks.

## File names versus ownership

`t2_cavity_fluxexcursion.py` contains 18 classes, including slow pulses, reset/parity diagnostics, transition spectroscopy, amplitude calibration, Floquet gain chevrons and a scramble variant in addition to excursion Ramsey. A source comment near line 1086 explicitly says the dark-mode-search code should move elsewhere. Q101 calls its Floquet gain-chevron classes; they remain there on guan too.

`floquet_dark_mode_readout.py` mixes literal dark-mode transformations, common Floquet primitives, MBR acquisition, loading, reconstruction, spectral inference, statistics and plotting. MBR inherits through a dark-mode-named class but its acquisition body implements encoder → Floquet evolution → decoder → Ramsey readout. Historical project names therefore cannot define the active dependency boundary.

The source supports describing both as accumulated project work files. It does not establish why their author organized them that way.

## How to use the map

The [one-screen map](qsim_refactor_surface_map.md) records the thematic notebook destinations and extraction-then-improvement approach; implementation order remains open. The runner pattern survives, but substantial workflow and algorithm implementations sit around it in cells. Removing blanks and comment-only lines leaves 19,584 notebook lines; roughly 4,900 sit in top-level function/class definitions. Extraction must address that code as well as the large modules.

Choose one representative per core product, then cover meaningful variants: diagonal/off-diagonal; current/historical phase and timing; complete/partial grids; standard/disorder campaigns; mean arrays/raw-shot estimators. Trace shared foundations once and record subsequent differences. Treat N and dataset identity as inputs unless they expose a new contract. This is a planning inventory, not an instruction to execute all cells sequentially or to validate every historical experiment now.

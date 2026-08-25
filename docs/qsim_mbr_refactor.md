# Refactor handoff: many-body Ramsey (`experiments/qsim/floquet_dark_mode_readout.py`)

8272 loc, 31 classes. Two objects hold 67% of it: `EncodingHamiltonianSpectroscopyExperiment`
(3805) and `DarkBaseProgram` (1776). Findings below are from a read-through plus
running the analysis on the 2026-08-15..17 N=3 and disorder data.

## 1. Correctness — fix these regardless of any refactor

**Theory amplitude is silently fitted to the data.** `analyze_spectrum:5813` (copy at
`:6864`) rescales `theory_local` so its peak matches the measured peak. Both sides are
already unit-normalised at t=0 — measured by `/|A[:,0]|`, theory because `eigh` rows have
unit norm (checked numerically: deviation 9e-16). So `max(measured)/max(theory)` is a real
diagnostic — missing spectral weight, decoherence broadening, wrong Kerr — and it is
overwritten with 1 in place, unrecoverable by the caller. Report the ratio; do not apply it.

**Station fallback silently corrupts energies.** With no saved program, `_saved_parameters`
falls back to `hardware_parameters(station, …)`, which reads *today's* floquet swap CSV. For
the Aug data that gives `pi_fracs=[40,30,30,40]` where the pulses used `[40,40,40,40]`, and a
cycle time 43% off — every energy scales by 1.74, no error raised. Callers must check
`data.hardware.source == "saved program"`. Delete the fallback; make timing explicit data.

**`station` is one leaf pretending to be a dependency.** It threads through `_from_expts`,
`from_job_files`, `from_job_ids`, `_calibration_data`, `phase_correction_from_calibration`
and is used for exactly one thing: Floquet timing. Removing it deletes 5 signatures of
plumbing and the `hasattr(prog, 'm1s_pi_fracs')` duck-probe that forces analysis notebooks
to fake a program object.

**`flatten_exp_lists` shreds strings.** `flatten_exp_lists("a.h5") -> ['a','.','h','5']`;
a `Path` raises `TypeError`. Every caller needs a scalar guard, which is what the otherwise
inexplicable `isinstance(job_files, (str, Path))` in `from_job_files` is for.

**`from_job_files` pickle fallthrough.** Anything not `.h5`/`.hdf5` goes to `pickle.load`,
including a typo'd path — which then fails as an unpickling error, not "file not found".

## 2. Provenance gaps (fix in acquisition, not analysis)

- The H5 config attr drops `device.storage._ds_floquet` (`slab/experiment.py:161`), so
  Floquet timing is not reconstructable from the data file. **The `Job` table already
  records `floquet_storage_version_id`** (`job_server/models.py:75`) and
  `station._initialize_configs` already accepts `CFG-` IDs and re-attaches swap datasets.
  Missing link: stamp the four version IDs into the H5 attr (and the vault YAML) in
  `job_server/worker.py`. Small change; makes files self-describing.
- Until then timing is a hand-recorded constant with a provenance note. See
  `analysis_notebooks/guan/MBR_analysis.py`.
- Everything else survives the H5 round-trip: all 10 datasets bit-identical to the pickle,
  `cfg.expt` complete, zero differing values.
- `configs/versions/` exists only on pippin (local copy stops at Feb 2026).

## 3. Documentation gaps

- `analyze()` docstring says "one of four paths" and enumerates four; there are five
  (`None`, `calibration`, `orthogonality`, `propagator`, `spectrum`). `propagator` was added
  in c52eb42 without a docstring update.
- `_cycle_branches` documents int/list/dict input shapes and never the physics: that the
  forward/adjoint echo doubles the accumulated phase and leaves a 180° branch ambiguity
  resolved by eye. Unrecoverable from the code by anyone who wasn't there.
- `palindrome_scramble` reverses storage-mode *order* within a cycle (Trotter
  symmetrisation). `_play_closed_floquet_cycle_pairs` applies the *adjoint* (reversed order,
  +180°, an echo). Both described as "reversal"; easy to conflate.

## 4. Dead and unexercised code

Purge (no reference anywhere, 176 loc): `ManStorScrambleProgram`,
`StorageSwapPhaseAccumulationProgram`, `SidebandStarkAmplificationModifiedProgram_newold`.

Decide (reachable via batch builders / an `analyze` stage, never called from a notebook,
323 loc): `EncodingPropagatorProgram`, `EncodingStarkShiftCalibrationProgram`,
`SinglePhotonFloquetSpectroscopyProgram`, `CentralBosonLocalReturn{Program,Experiment}`.

`SidebandStarkAmplificationModifiedProgram_old` is **live** — 10 refs, used by
`Autocalibrate.ipynb`. The `_old` suffix is a lie.

## 5. Naming

The file is named for an obsolete pulse sequence (dark-mode photon-number readout); the
experiment that now dominates it is named for a state-prep detail (`Encoding…`). Neither is
guessable. Name for what is measured: **many-body Ramsey**.

Rename is cheap: no library code imports this module, `experiments/qsim/__init__.py` is
empty, and the only consumers are six notebooks. `git mv` plus a shim module that re-exports
and aliases old class names keeps every notebook working with zero search-replace. Keep
renames in commits with no behaviour change.

## 6. Target layout

Project-scoped, mirroring `experiments/qsim/`. Acquisition keeps the
one-Program-one-Experiment-per-module idiom of `experiments/single_qubit` (200–540 loc each).

    experiments/qsim/
      dark_base.py              DarkBaseProgram + DarkBaseExperiment   (~1660)
      dark_large_support.py     the 9 *large_dark* methods              (~260)
      mbr_spectroscopy.py       NPhoton… + Experiment
      mbr_cycle_phase.py        EntireFloquetCyclePhase… + Experiment
      mbr_orthogonality.py      EncodingOrthogonality… + Experiment
      mbr_phase_accumulation.py FloquetPhaseAccumulation… + Experiment
      dark_diagnostics.py       DarkT1*, ChevronR*, BroadbandGe, KerrWait  (~510)
      sideband_stark.py         SidebandStarkAmplification* (+ _old)       (~130)
      floquet_kerr.py           FloquetDisplacementKerr{Program,Experiment} (~90)

    fitting/qsim/
      matrix_pencil.py          the 3 MPM methods                          (911)
      level_statistics.py       merge_spectra, level stats, SFF             (~450)
      spectrum_fft.py           FFT estimation, after the theory split      (~170)
      mbr_reconstruct.py        reconstruct_* x4, cycle-phase unwrap, phase correction (~700)
      mbr_io.py                 HDF5 load + the validation now stranded in
                                reconstruct_spectroscopy                    (~250)

    analysis_notebooks/…        fixed-N Hamiltonian build + eigh            (~120)

`BatchRunner` (109) moves up next to `CharacterizationRunner`.

`fitting/` is currently flat; `fitting/qsim/` is a new but consistent convention. The
setuptools whitelist lists only top-level names, so a submodule under `fitting/` needs **no**
`pyproject.toml` edit — a new top-level `theory/` would.

Theory (Fock basis, `eigh`, LDOS weights) is not a fitter and may eventually be served by
the Julia numerics library, so it stays a sandbox script until a non-notebook consumer
appears. Consequence: the measured/theory comparison moves up to the caller, which is what
exposes finding 1.

**No standalone display module.** Display stays with the type it renders — per-job displays
with the job Experiment, spectrum displays with the spectrum aggregate, level-stats/SFF
displays with the ensemble. That keeps the acquisition idiom intact.

## 7. Aggregation levels

The single Experiment class currently covers four levels. Only level 0 has `acquire()`;
levels 1–3 are analysis-only aggregates and belong under `fitting/qsim/`.

| level | unit | built from |
|---|---|---|
| 0 | one HDF5 = one `(occupation, analyzer_phase)` | — |
| 1 | one matrix element `A_βα(t)` | **2 files** (φ=0, φ=90); the 0/180 prep pair is an in-file axis |
| 2 | one spectrum | N rows + a calibration aggregate + timing |
| 3 | ensemble / level statistics | M spectra (`merge_spectra`) |

Level 3 legitimately cannot support SFF or time-trace views — merged rows share no complex
time grid (`merge_spectra` docstring). That asymmetry argues levels 2 and 3 are distinct
types, not one class with a flag.

Analyzer phase is hard-restricted to `{0, 90}` (`:5622`). Keep it as a stated constraint,
not a validation accident. Note `_quadrature` mutates `expt.data`, so level-0 objects are
not freely reusable across two level-1 assemblies.

## 8. Workload

Acquisition is mostly **moving, not rewriting**. The MBR programs are already the healthy
part: `NPhoton…body` is 89 loc reading 9 cfg keys, and its five subclasses are 26–134 loc.

The rot is `DarkBaseProgram.body` — 238 loc reading **23** `cfg.expt` keys, a switchboard
serving the *legacy* dark-mode-readout experiment, which the MBR path bypasses entirely with
its own `body`. Leave it with the old experiment; retire it when that experiment retires.
Do not try to unify the two bodies.

Whole-file contract is **108 distinct `cfg.expt` keys**, untyped. Worth a dataclass or a
documented schema per experiment type, but that is a separate piece of work.

## 9. Order

1. Purge the 176 loc of dead code; decide the 323 loc of unexercised code.
2. Stamp config version IDs into the H5 attr and the vault YAML (acquisition side).
3. Fix finding 1 (report the amplitude ratio) — smallest change, biggest change in what the
   data tells you.
4. Drop `station` from the analysis path; timing becomes explicit data.
5. Extract `fitting/qsim/{matrix_pencil,level_statistics}.py` — already static methods, so
   this is a move; first code here that becomes unit-testable without a fridge.
6. Split theory out of `analyze_spectrum`.
7. Rename + shim, then drain the remaining pieces into the layout above.

Steps 1–4 are independent and each is small. Steps 5–7 are the bulk.

## 10. Open

- Acquisition-side consequences of promoting calibration to its own Experiment type
  (its cfg keys: `n_cycle_pair`, `n_physical_cycle`, `phase_unwrap_mode`,
  `zero_floquet_gain`) — not yet scoped.
- Data-directory naming: the station's `experiment_name` tracks neither job date nor vault
  project, so the Aug jobs sit under `260526_qsim_darkmode` and are logged under two vault
  projects. The vault's recorded `data_path` is the only place the subdir is written down
  (228/228 coverage for this dataset) — good enough for now; auto-managing the subdir is the
  real fix.
- Mounting pippin's repo tree would make `configs/versions/` readable off-host and remove
  the last reason to touch a pickle.

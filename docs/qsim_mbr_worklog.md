# MBR refactor worklog

Running handoff for the refactor in `docs/qsim_mbr_refactor.md`. Newest entry
last. Each entry says what landed, how to check it, and what is next, so a
session that starts cold can pick up from the bottom.

Read first: spec section 14 (working order **and its invariants**), section
13 (execution environment), appendix C (configuration traps).

## Standing facts

- Work happens on the acquisition workstation, in the `guan` worktree
  (`C:\python\multimode_expts_guan`). Spec section 13.1.
- **The offline track constructs no station, ever.** That is both the
  correctness rule (section 2.2) and the isolation rule (section 13.3): the
  worktree shares `configs/versions` and `job_server/jobs.db` with the live
  tree, so writes from here hit production.
- Run tests with `pixi run python -m pytest tests/ -q`.
- Re-bless the golden baseline only alongside a deliberate behavior change:
  `MBR_GOLDEN_BLESS=1 pixi run python -m pytest tests/test_mbr_analysis_golden.py`

## 2026-08-25 — safety net in place

**Landed**

- `experiments/job_paths.py` — `resolve_job_paths()`, the section 9.1 seam.
  Two backends: `index` (default, globs the local tree once and caches) and
  `vault` (scrapes `data_path` from the synced lab notebook, for slow remote
  mounts). Raises `JobPathError` naming the environment variable to set rather
  than skipping. Top-level module under `experiments/`, so the flattened
  exporter ignores it and it cannot collide.
- `tests/mbr_reference.py` — the reference analysis as a callable, plus
  `flatten_result()`. Holds the `_SavedJob`/`_SavedProgram` scaffolding, which
  exists only until aggregates load HDF5 themselves. Marked TODO in the module.
- `tests/test_mbr_analysis_golden.py` + `tests/data/mbr_spectrum_20260815.npz`
  — the characterization test. 4 tests, all passing.
- Spec section 14 (working order).

**Baseline contents**: 55 fields, 26 numeric arrays, 8611 elements, covering
`reconstruction.A`/`A_norm`, `acquired_reconstruction.*`, all 19 `spectrum.*`
entries, per-occupation phase corrections, and the resolved hardware block.

**Verified the net can fail**: perturbing `zero_padding` 1 to 2 flags six
fields. A golden that cannot fail is worse than none, so re-check this if the
comparison logic is ever touched.

**Behavior deliberately pinned as-is** (these are the section 2 defects; the
baseline locks them in on purpose):

- theory spectra are rescaled to match measured peaks (section 2.1);
- `_saved_parameters` adopts the first source with a program, no comparison
  (section 2.3);
- aggregate analysis writes `Pe` and `return_quadrature` back onto its sources
  (section 2.5), pinned by `test_source_mutation_is_pinned`.

**Next**: spine step 2, pure numerical extractions. Verified low-risk targets,
each with exactly one call site and no meaningful coupling to `self`:

| Method in `floquet_dark_mode_readout.py` | Target |
|---|---|
| `analyze_matrix_pencil` (already static) | `fitting/qsim/matrix_pencil.py` |
| `analyze_matrix_pencil_trace` (already static) | `fitting/qsim/matrix_pencil.py` |
| `analyze_matrix_pencil_occupation` | `fitting/qsim/matrix_pencil.py` |
| `analyze_level_statistics` | `fitting/qsim/level_statistics.py` |
| `analyze_sff` | `fitting/qsim/level_statistics.py` |

Pattern for each move: extract a pure function taking arrays, leave the method
as a thin wrapper that unpacks `self.data`, keep the golden green, one commit
per module.

## 2026-08-25 (later) — matrix pencil extracted

**Landed**: `fitting/qsim/matrix_pencil.py` (930 lines).
`floquet_dark_mode_readout.py` 8271 → 7381 lines.

- `analyze_matrix_pencil` and `analyze_matrix_pencil_trace` were already
  `@staticmethod` with pure array inputs, so they moved verbatim and stay
  reachable on the class via `staticmethod(...)` assignment.
- `analyze_matrix_pencil_occupation` keeps its signature and becomes a wrapper
  that supplies `self.data` then calls `refit_occupation`. The module is
  imported as `matrix_pencil_analysis` so the historical `matrix_pencil`
  argument name is not shadowed.

**Verification** — the golden alone was not enough, and this matters for every
later move:

The committed golden ran `spectrum_method="fft"`, which never executes the
Matrix-Pencil code. The move looked green while actually being broken:
`sliding_window_view` was imported in the god file and not carried across, so
the Matrix-Pencil path raised `NameError`.

Caught by running the Matrix-Pencil analysis on both sides — a throwaway
`git worktree add --detach <tmp> HEAD` for "before", the working tree for
"after" — and diffing flattened results: **1387 fields, 20354 elements,
bit-identical**.

So the golden now has **one baseline per spectrum method**
(`mbr_spectrum_20260815.npz`, `mbr_matrix_pencil_20260815.npz`), 5 tests.
The FFT baseline was byte-unchanged by re-blessing, which independently
confirms the move left that path alone.

**Lesson for the remaining moves**: check that the golden actually executes the
code being moved. If it does not, either extend the golden first or do the
temporary-worktree before/after diff. The worktree trick is cheap and is the
stronger check.

Also fixed: `flatten_result` raised on ragged object sequences (Matrix-Pencil
candidate lists). It now falls back to per-element recursion.

**Pre-existing, not ours**: `tests/test_branch_manager.py` has 3 failures
(`KeyError: 'hardware_config'`), present before this work started.

**Next**: `analyze_level_statistics` (line ~6906 pre-move) and `analyze_sff`
→ `fitting/qsim/level_statistics.py`. Same pattern. Check first whether the
golden reaches them — `merge_spectra` and the level-statistics path are
probably not on the `analyze(stage="spectrum")` route, in which case use the
worktree diff.

## 2026-08-25 (later still) — level statistics extracted

**Landed**: `fitting/qsim/level_statistics.py` (461 lines) holding
`merge_spectra`, `analyze_level_statistics`, `analyze_sff`.
`floquet_dark_mode_readout.py` 7386 → 6971 lines (8271 at the start of today).

Same shape as the matrix-pencil move: `merge_spectra` was already static and
moved verbatim; the other two keep their signatures and become wrappers that
supply `self.data`. Module aliased to `level_statistics_analysis`.

**Coverage gap found — worth knowing before trusting any test here.**

None of these three can run on the characterization dataset. All raise before
doing arithmetic:

- `analyze_level_statistics`: "requires the complete fixed-N occupation basis"
- `analyze_sff`: same, "this data gives only a projected trace"
- `merge_spectra`: "merged spectra contain duplicate occupations"

The Aug-15 set covers four occupations, not the complete fixed-N basis, so the
level-3 ensemble code of spec section 8 has **no test data at all**. The
before/after run confirms only that the wrappers reach the same guards with the
same messages. Finding a dataset that completes the basis, or building a
synthetic one, is a real outstanding task — this code is currently unverifiable
by execution.

**So verification moved to textual identity**: `tools/verify_moved_code.py`
compares AST-normalized statement lists between the pre-move commit and the
extracted module, erasing only the edits made on purpose (the `self`/`data`
signature change, the class-qualified internal call). For a pure move this is
stronger than a runtime check on one dataset, because it covers every branch
rather than the ones the data happens to reach.

Result: all five moved functions identical. The single reported difference is
docstring indentation inside a nested function, a `textwrap.dedent` artifact;
the executable statement matches.

Run it after any further move:

    pixi run python tools/verify_moved_code.py

It needs its `MOVED` table updated with each new function and the commit whose
god file still held it.

**Next**: Hamiltonian and fixed-photon-number basis construction →
`fitting/qsim/mbr_hamiltonian.py`. Unlike level statistics this *is* exercised
by the golden (the FFT path builds theory spectra), so expect real numerical
verification. Grep the god file for the basis/Hamiltonian builders first; spec
section 7.5 wants basis construction, Hamiltonian construction,
diagonalization and theoretical amplitudes together.

## 2026-08-25 (end of session) — spectrum extracted; step 2 mapped

**Landed**: `fitting/qsim/mbr_spectrum.py` (190 lines) holding
`analyze_spectrum`. `floquet_dark_mode_readout.py` 6972 → 6809 lines.
**8271 → 6809 across the session, no behavior change.**

`analyze_spectrum` was already static with zero `self` use. It is covered by
the golden (the FFT path runs it), and a missing `from itertools import product`
was caught by both the static import check and the golden — the first move where
the golden did its job unaided.

**Left deliberately undone**: `analyze_spectrum` does two things the spec
separates — fixed-N basis plus Hamiltonian assembly and diagonalization
(`mbr_hamiltonian.py`), and windowing/padding/FFT (`mbr_spectrum.py`).
Splitting it is a refactor, not a move, so it gets its own commit. A TODO in the
module head records it. The golden covers it, so the split is verifiable
numerically.

`tools/verify_moved_code.py` now strips docstrings recursively, so dedent
reflow no longer reports a false difference. All six moved functions:
**IDENTICAL**.

### Step 2 remaining, fully enumerated

Every pure static/no-`self` method over 20 lines in the god Experiment, with
its disposition. This is the complete step-2 worklist:

| Method | Lines | Where |
|---|---|---|
| `subsample_spectroscopy_shots` | 192 | → `fitting/qsim/mbr_reconstruction.py` |
| `build_phase_correction` | 34 | → `fitting/qsim/mbr_phase.py` |
| `_unwrap_cycle_phase` | 31 | → `fitting/qsim/mbr_phase.py` |
| `_cycle_branches` | 33 | → `fitting/qsim/mbr_phase.py` |
| `_saved_correction` | 44 | → `fitting/qsim/mbr_phase.py` |
| `analyze_spectrum` split | — | → `mbr_hamiltonian.py`, its own commit |

**Stays put, with reasons** — do not move these as part of step 2:

- `display_occupation`, `display_result`, `display_cycle_phase` (156 lines
  total): spec 7.8 keeps display on the owning Experiment.
- `spectroscopy_batch`, `calibration_batch`, `propagator_batch`,
  `orthogonality_batch` (262 lines): spec 7.7 runner territory, gated on the
  new Experiment interfaces existing first.
- `hardware_parameters` (56 lines): this *is* the section 2.2 station fallback.
  Removing it is a behavior fix, so it belongs in spine step 3, not here.

### Method for each remaining move

1. Assert exact line boundaries in a throwaway script; slice, dedent, write.
2. Wrapper on the class: `staticmethod(module.fn)`, or a `self.data`-supplying
   method. Import the module under an `_analysis` alias so argument names that
   match the module name are not shadowed.
3. Static undefined-name check on the new module (catches the missing-import
   class of bug — it has happened twice).
4. `pixi run python tools/verify_moved_code.py`, after adding the function and
   its pre-move commit to `MOVED`.
5. Golden: `pixi run python -m pytest tests/test_mbr_analysis_golden.py`.
   If the golden does not reach the moved code, say so in the commit message
   rather than implying coverage.

## 2026-08-26 — complete-basis coverage; timing resolver is real

Closes the coverage gap recorded on 2026-08-25. The level-statistics and SFF
code is now verified numerically, not just textually.

**Where the data was.** `measurement_notebooks/jonginn/data_postprocess.ipynb`
(3.3 MB, 175 code cells) — its `replot_job_ranges` cell is the authoritative
record of which jobs formed each published photon-number sector:

| N | Calibration | Spectroscopy |
|---|---|---|
| 1 | `20260722` 557–566 | `20260722` 577–595 |
| 2 | `20260722` 35–64 | `20260722` 215–244 (+ supplement 425–426 / 452–454) |
| 3 | `20260722` 683–712, `20260723` 1–40 | `20260723` 48–85, 87–149 step 2 |

**N=3 chosen.** All 140 jobs COMPLETED under one config triple
(`CFG-HW-20260717-00173`, `CFG-FL-20260722-00001`, `CFG-M1-20260722-00010`).
N=1 and N=2 spectroscopy ranges contain jobs with a *null* `program_class` and
several different config versions, which is why the notebook filters them by
program name — and, because HDF5 records no program class (spec 3.1), a
file-based loader could not reproduce that filter. N=2 additionally needs a
supplement merged in to complete its basis.

It gives `complete_basis=True`, 35 occupations, the full N=3 five-mode basis.
Both `analyze_level_statistics` and `analyze_sff` run on it.

**Verification of all three extractions, numerically at last**: N=3 analysis
run at `04cea3a` (before any extraction) versus the working tree —
**5274 numeric fields, 45453 elements, bit-identical**.

**Landed**

- `experiments/floquet_timing.py` — `resolve_floquet_timing()`, the spec 2.2
  resolver, promoted from a proof to production code. It was *needed*: the
  July sector ran at 0.4135 us per cycle against August's 0.7340, so no
  constant covers both. Reproduces the August pickle value exactly.
- `tools/export_job_provenance.py` + `tests/data/job_provenance.json` — the
  one-time read-only `jobs.db` export of spec 3.2 (148 jobs). Analysis now
  reads HDF5 + sidecar + archive, and never opens the database.
- `tests/mbr_reference.py`: `load_aggregate_resolved()` and
  `run_complete_basis_analysis()`, which reproduce the notebook's
  `replot_analyze_sector` without `JobClient` or pickles.
- Three tests, two marked `slow` (`-m "not slow"` deselects). Third baseline:
  `mbr_complete_basis_20260723.npz` (2 MB).

**Suite: 324 passed, 0 failed.** The three `test_branch_manager` failures
reported yesterday were fixed by you in `65a7415`/`4f25675`, not by anything
here.

**Next**: unchanged — `mbr_phase.py` (4 methods, 142 lines), then
`mbr_reconstruction.py` (`subsample_spectroscopy_shots`, 192 lines), then
splitting the Hamiltonian half out of `analyze_spectrum`. All three golden
paths (FFT, Matrix-Pencil, complete-basis) now guard those moves.

Worth noting for spec section 3.2: with the sidecar in place, the offline path
is now genuinely database-free. The remaining pickle/FastAPI dependency lives
only in the old notebook, not in anything under test.

## 2026-08-26 (aside) — dataset selection was loose; sectors resolved literally

Side quest from reading the notebook's `replot_job_ranges`. Not part of the
refactor spine; recorded because it changes what "the N=2 sector" means.

**How the ranges work.** Each tuple is `(date, first, last, step)` expanded by
`range()`. The fourth element is a **stride**, and N=3 uses `step=2` to skip
every other job.

**Why a stride is needed at all.** Job IDs are one global counter on a queue
shared by every user, and that is the intended design: user A measures at set
point 1, user B measures at set point 2 then refits and stores a new config
version, A goes again, and so on. Each job pins its own config, so the
interleaving is benign — I initially misread the changing `CFG-HW-*` IDs as a
possible mid-run reconfiguration of jonginn's sweep. It is not.

**What is not fine** is identifying a dataset by a numeric range over that
counter. The notebook subtracts the other user's jobs two different ad-hoc
ways: a positional stride (N=3) and a program-class filter (N=1). The stride
assumes strict alternation *and* correct phase.

Checked against the database, three of the four declared ranges disagree with
an owner-plus-program filter:

| Sector | Declared | Actually jonginn's | Foreign |
|---|---|---|---|
| N1 spectroscopy | 19 | 10 | 9 |
| N2 spectroscopy | 30 | 28 | 2 |
| N2 supplement spectroscopy | 3 | 2 | 1 |
| N3 spectroscopy | 70 | 70 | 0 |

N=3's stride was exactly right — the 31 skipped jobs are all `closed_loop` /
`closed_loop_recal`. Correct by luck rather than by construction.

**Data integrity is fine.** Every sector resolves to exactly one config
triple. Only the selection was loose, and the foreign jobs are a different
program class so they would have failed loudly rather than corrupting a
result. The risk is latent: had the other user run the same program class
concurrently, the stride would have absorbed their jobs silently.

**Landed**: `tools/resolve_sector_job_ids.py` and
`tests/data/mbr_sector_job_ids.json` — literal, verified lists filtered by
user, program class, completion and config triple. `tests/mbr_reference.py`
now reads N=3 from the JSON. The IDs are byte-identical to the declared range
for N=3, so no test result changes (324 passed).

**Worth knowing for level-3 work**: the N=2 supplement runs under a *different*
Floquet and M1 config than the N=2 main set (`CFG-FL-20260722-00001` /
`CFG-M1-20260722-00002` versus `CFG-FL-20260717-00029` /
`CFG-M1-20260717-00011`). `merge_spectra` is therefore joining data taken under
different swap calibrations. That is what it is designed to check, but it is
worth confirming deliberately when the level-3 aggregate is built.

**Follow-up for spine step 3**: apply spec 2.3 compatibility validation one
level earlier, at source *selection* rather than reconstruction — assert
agreement on `(user, experiment_class, program_class, config triple)` for a
declared set. The provenance sidecar already carries every field needed, so it
is a short function plus a test, and it would catch this class of problem
automatically.

## 2026-08-26 (aside, revised) — one record of which jobs form which dataset

Replaces the previous entry's approach. The range-resolving script was a
throwaway and has been deleted; keeping it would have made a glorified
sed/awk into a library fixture. Only its output is kept.

`tests/data/mbr_datasets.json` now records all ten datasets literally, each
with its config triple. **Interim home**: this belongs in the aggregate HDF5
manifest (spec 3.3) once aggregates can save themselves, at which point the
JSON goes away too.

### There are two campaigns, not three datasets

The three sets pointed at so far are not independent — one is a fragment, one
is a subset, one is the declaration of the other.

**July 22–23** — photon-number scaling, `replot_*` in the notebook. Three
sectors N=1/2/3 plus an N=2 supplement. This is the campaign whose ranges
interleave with `closed_loop`, so three of four declared ranges over-collect.

**August 15–17** — N=3 plus disorder, `saved_*` in the notebook. One N=3
sector (70 cal + 70 spec) and four disorder realizations of 20 each. **Every
August range is clean** — no foreign jobs anywhere, single config triple.

| Dataset | cal | spec | complete basis? |
|---|---|---|---|
| `july_N1` | 10 | 10 | — |
| `july_N2` | 30 | 28 | needs the supplement merged |
| `july_N2_supplement` | 2 | 2 | one occupation, different FL/M1 config |
| `july_N3` | 70 | 70 | **yes**, 35 occupations |
| `august_quickplot` | – | 8 | no, 4 occupations |
| `august_N3` | 70 | 70 | untested, very likely yes |
| `august_disorder_r0..r3` | – | 20 each | untested |

- The 8-file set is the notebook's `data_four_realization`: a quick-plot
  **fragment of the August campaign**, not the August N=3 sector.
- The OneNote block is `replot_job_ranges` itself — the declaration that
  `july_N3` came from, so it is not a fourth dataset.

### Open question for the next session

`august_N3` is arguably the better complete-basis fixture than `july_N3`: it
shares the August campaign and the *same* Floquet and M1 config as the
quick-plot set the golden already pins (`CFG-FL-20260814-00076`,
`CFG-M1-20260814-00121`), so one timing resolution covers both, and its four
disorder realizations give `merge_spectra` real multi-spectrum input — which
still has no coverage. `july_N3` works and is committed; switching costs a
re-bless. Not done, deliberately.

## 2026-08-26 — consolidated onto one campaign (August)

Every fixture now comes from the August 2026 campaign, under one swap
calibration (`CFG-FL-20260814-00076`, `CFG-M1-20260814-00121`). July is no
longer a fixture.

**Why.** `august_N3` completes the 35-state basis just as `july_N3` does, but
shares its configuration with the quick-plot set the golden already pins, so a
single timing resolution now covers the whole module. Its ranges are also
clean, where three of the four July ranges over-collect.

**Coverage went up, not sideways.** The August sector is pinned under three
branches, which between them cover both phase frames and both spectrum
methods — everything the old two-dataset arrangement reached, plus the
complete basis:

| Branch | phase frame | method |
|---|---|---|
| `as_acquired_fft` | as acquired | FFT |
| `as_acquired_matrix_pencil` | as acquired | Matrix Pencil |
| `manual_kerr_fft` | manual Kerr + cycle branches | FFT |

The eight-file quick-plot set stays, for two reasons: it is what
`analysis_notebooks/guan/MBR_analysis.py` actually runs, and it is the only
fixture with `calibration=None`.

11 tests in the module, 327 in the suite. `-m "not slow"` deselects the four
that touch the 140-file sector, leaving a 15 s run.

**A test earned its keep during the switch.** With both fixtures now August,
`test_timing_resolves_..._not_a_constant` started failing: its premise was that
the fixture came from a differently configured campaign, and that stopped being
true. Rewritten as `test_timing_resolver_is_not_a_constant`, which resolves one
July job and one August job directly and asserts both the values and that they
differ. July appears there as a data point, not a fixture.

**Baseline storage.** A flattened complete-basis result has ~8.5k scalar
leaves. One npz member each meant ~300 bytes of zip header per scalar: 1.5 MB
of numbers in a 4.4 MB file. Scalars now pack into a single JSON member, which
is greppable as a bonus. `tests/data` went 6.6 MB -> 3.8 MB.

Watch the packing condition: it is `ndim == 0`, not `size == 1`. Packing a
single-element *array* would return it as a 0-d scalar and report a false
shape difference — which it did, on `correction.modes` and the Matrix-Pencil
`supporting_rows` fields, until fixed.

**Still uncovered**: `merge_spectra`. The natural fixture is
`july_N2` + `july_N2_supplement`, which is what it was written for, but those
were taken under different Floquet and M1 configs — worth doing deliberately
rather than by accident.

## 2026-08-26 (correction) — the Experiment triple is the target shape

Recorded because it was proposed for violation twice, in two different
sessions, after reading the spec. The spec said the right thing in section 7.8
but stated it as a target with no reason attached, so it did not survive
contact with a plausible-sounding alternative.

**The claim that was wrong**: that `acquire`/`analyze`/`display` are separable
concerns, and that lifting the 598 lines of `display_*` out of the god
Experiment — or splitting the file along an acquisition/analysis line so the
analysis half reads alone — would improve inspectability.

**Why it is wrong.** `slab/experiment.py:170` makes the four-method lifecycle
the contract, and the runners enforce it by calling `analyze()` and
`display()` after acquisition and capturing the output to file
(`characterization_runner.py:300`/`:523`, `sweep_runner.py:230`/`:235`/
`:276`-`:289`). A measurement whose display lives elsewhere produces no
record.

More importantly it inverts the target. The shape to copy is
`experiments/single_qubit/error_amplification.py`: one Program, one
Experiment, four methods, ~180 lines, reusable mathematics delegated to
`fitting/` and the measurement-specific glue kept local. You open one file and
see the whole story of one measurement.

**So the real defect in the god Experiment** is not that display is attached.
It is that one class holds the triples of four measurements plus the aggregate
analyses behind a `stage` argument, so no triple is local. The fix is to split
the class into several Experiments, each keeping its whole triple — sections
7.3 and 7.4 — not to split the triple.

Spec section 7.8 now carries the contract, the enforcement points with file
and line, the exemplar, and the rejected alternatives. Section 14 gained an
**Invariants** block that names this first, because section 14 is what a cold
session reads first.

**Measured shape of the god class**, for planning. Class is 2341 lines
(4415-6755) of the 6810-line file; the rest is 28 acquisition classes.

| Group | Lines | Fns |
|---|---|---|
| Loading and provenance | 196 | 7 |
| Reconstruction | 462 | 6 |
| Phase | 269 | 7 |
| Calibration analysis | 168 | 2 |
| `analyze` switchboard | 249 | 1 |
| Wrappers to extracted modules | 41 | 3 |
| Display | 598 | 13 |
| Batch runners | 266 | 4 |

The HDF5-to-output spine is about 1200 lines in about 20 functions. Each
display function belongs with the measurement it displays, so the 598 lines
are distributed by the 7.3/7.4 split, not extracted.

## 2026-08-26 (later) -- seven measurement families out, verbatim

Section 7.8 says the triple is the unit of locality; section 7.6 already named
the module each family belongs in. Those two together authorize a mechanical
split, and it needs none of the pending audits, because moving a family's whole
triple into its own file decides nothing about that family.

Seven modules, 1,190 lines out. `floquet_dark_mode_readout.py` 6,810 -> 5,665.

| New module | Classes | Lines |
|---|---|---|
| `central_boson_local_return.py` | Program, Experiment, the two config validators | 368 |
| `dark_mode_multiparity_chevron.py` | `DarkBaseRProgram`, chevron Program + Experiment | 256 |
| `sideband_stark_shift_cal.py` | the three `SidebandStarkAmplificationModified` variants | 185 |
| `storage_swap_phase_cal.py` | `StorageSwapPhaseAccumulationProgram` | 105 |
| `dark_mode_t1.py` | `DarkT1Program`, `DarkT1Experiment` | 92 |
| `floquet_displacement_kerr.py` | Program + Experiment | 91 |
| `dark_mode_broadband_ge_validation.py` | `BroadbandGeValidationProgram` | 78 |

Every block is a verbatim line slice; only the module header is new. Done with
a throwaway script that sliced by line range, so re-indentation and reflow
could not creep in. `tests/test_qsim_measurement_split.py` pins each moved
definition to its AST at `c7578de` and passes (33 cases). Full suite 360 pass,
golden green -- unchanged, since none of this is the analysis spine.

### Two things the split had to solve

**The exporter.** `experiments/__init__.py` walks every file and flattens every
class it finds -- including imported ones -- into one namespace, last write
wins. So a name defined in two modules is a coin flip on filesystem order. The
test asserts no name is defined twice, and asserts the moved names are *not*
bound in the legacy module's `vars()`.

**The old addresses.** The acquisition notebooks say
`meas.qsim.floquet_dark_mode_readout.DarkT1Program`, which is attribute access
on that specific module, so it breaks the moment the class leaves. The fix is a
module-level `__getattr__` (PEP 562) mapping the 18 moved names to their new
modules.

It has to be lazy, and that is not a style preference. Each new module imports
`DarkBaseProgram` (or `DarkBaseExperiment`, or
`SidebandScrambleDarkProgramNewNew`) *from* the legacy module, because the base
classes stay behind until section 7.2 decomposes them. A top-level re-import in
the legacy module would close that cycle, and it would fail in exactly one
direction: import the new module first and the legacy module's bottom-of-file
re-import finds a half-initialized module in `sys.modules`. `__getattr__` runs
at attribute-access time instead, long after both modules are loaded.

Being invisible to `inspect.getmembers` is a feature here, not a cost: it is
what stops the exporter from binding each moved class twice.

### Judgement calls, so the next session does not relitigate them

- **`DarkBaseRProgram` moved** into the chevron module. It is named like
  infrastructure and appendix A had it as "pending consumer audit", but it has
  exactly one subclass in the whole repository. Moving it keeps the measurement
  local; leaving it behind would have kept a 70-line base in the god file for
  one caller. If a second RAverager program appears, promote it then.
- **`classify_two_parity_readouts` stayed.** It looks central-return-specific
  because that is where the naming energy went, but it is a generic two-parity
  classifier and `DarkBaseExperiment.analyze_multiparity` calls it. The two
  `configure_/validate_central_return_*` functions did move: those encode this
  protocol's conventions and nothing else calls them.
- **The three stark variants share a file.** `_old` is live, the other two are
  pending. Grouping them is not a survival decision, and the file is named for
  the measurement, not for the winner.
- **Nothing was renamed.** The names still violate section 6. Renaming touches
  notebooks and worker logs and is its own pass.
- **`dark_mode_readout.py` was not created.** The remaining eighth module in
  section 7.6 is the dark-mode half of `DarkBaseProgram`, which is a
  decomposition, not a move, and is gated on the phase-4 audit.

### What is left in the god file

5,665 lines: `DarkBaseExperiment` and `DarkBaseProgram` (the pulse base, ~1,900
lines), the five MBR acquisition programs, the legacy scramble/Kerr variants,
`BatchRunner`, and the 2,341-line god Experiment. Nothing further splits
cleanly by measurement -- the next cuts are the god Experiment (7.3/7.4) and
the pulse base (7.2), and both are decompositions.

## 2026-08-27 -- analyzer phase out to fitting/qsim/mbr_phase.py

Four of the five remaining step-2 moves, done by the standard method:
`_cycle_branches`, `build_phase_correction`, `_unwrap_cycle_phase`,
`_saved_correction` -> `fitting/qsim/mbr_phase.py` (180 lines). God file
5,666 -> 5,514. `verify_moved_code.py` reports all four IDENTICAL; full suite
396 pass, golden green (it reaches `build_phase_correction` and
`_cycle_branches` through the saved-correction path).

Every call site was internal (`cls.`/`self.`), so four class attributes
restore them:

~~~python
_cycle_branches        = staticmethod(mbr_phase_analysis.cycle_branches)
build_phase_correction = staticmethod(mbr_phase_analysis.build_phase_correction)
_unwrap_cycle_phase    = staticmethod(mbr_phase_analysis.unwrap_cycle_phase)
_saved_correction      = staticmethod(mbr_phase_analysis.saved_correction)
~~~

The notebook hits on `cycle_branches` are all the keyword argument, not the
method, so nothing outside the module had to change.

### Two judgement calls

**Underscores dropped in the new module.** `_unwrap_cycle_phase` as a
module-level name says "private to this module" while being imported across
one, which is backwards. Class attribute names are unchanged, so this is
invisible to callers. Precedent: `refit_occupation`. Bodies are still verbatim,
which is what the verify tool checks -- it compares body statements, so the
rename and dropping `cls` from `_saved_correction` (it never used it) are both
invisible to it.

**`_saved_correction` moved even though it is not mathematics.** It reads
`expt.cfg.expt`, so by the section 7.5 test -- arrays in, arrays out, no
knowledge of acquisition -- it belongs on the Experiment, not in
`fitting/qsim`. It moved anyway because it is the exact inverse of
`build_phase_correction`: one computes the correction to apply, the other
recovers the correction that was applied, and the sign and branch conventions
they must agree on are stated once at the top of the module. Splitting the pair
would put those conventions in two files. Narrowing its signature to plain
config mappings, with the `expt.cfg.expt` extraction left behind on the
Experiment, is a follow-up and is a real signature change, not a move.

### verify_moved_code.py had been red since last night

Not from this work. `merge_spectra` was moved verbatim in `c1867d6`, then
deliberately changed in `2da9cdc` (it now refuses off-diagonal merges). Its
verbatim-move claim retired the moment that fix landed, but the row stayed in
`MOVED`, so the tool reported REVIEW NEEDED regardless of input -- the exact
failure its own docstring warns about ("a tool that always says REVIEW NEEDED
is a tool nobody reads").

Moved to a new `DIVERGED` list, which prints the retirement and its reason
instead of asserting identity. Rows now also carry an optional new name, which
replaced the hardcoded `refit_occupation` special case.

### Step 2 after this

One move left: `subsample_spectroscopy_shots` (192 lines) ->
`fitting/qsim/mbr_reconstruction.py`. It is the largest single move in step 2
and the only one that creates that module.

## 2026-08-27 (later) -- the god Experiment split by analyze stage

Direction change, on request: the line count was no longer moving much per
commit, and the goal became a structure that can be opened and poked at with
data, rather than one more 30-line extraction.

### What made this cheap, and how we knew before starting

The god Experiment turned out to be **offline-only already**: `acquire` lives on
`DarkBaseExperiment`, not on it. All 2,196 lines are load -> reconstruct ->
analyze -> display, so none of it is gated on the phase-4 acquisition audit.

A call graph over all 46 methods, grouped by `analyze` stage, showed only two
cross-stage edges, and one had already evaporated:

| Group | Lines | Reaches out to |
|---|---|---|
| shared loading | ~220 | -- |
| `stage='calibration'` | ~330 | `cls.analyze(stage='calibration')` |
| `stage='orthogonality'` | ~180 | `_quadrature` |
| `stage='propagator'` | ~120 | `_quadrature` |
| `stage='spectrum'` | ~1,050 | `_quadrature`, the calibration |
| `analyze`/`display` dispatch | ~290 | dissolves into the stages |

`_postprocess_reconstruction` -> `build_phase_correction`/`_cycle_branches` was
coupling until this morning's `mbr_phase.py` extraction pointed it at
`fitting/qsim`. That is the second time the numerical extractions have paid for
themselves in structural freedom rather than in line count.

### Result

| Stage | Module | Class | Lines |
|---|---|---|---|
| propagator | `mbr_propagator.py` | `MBRPropagatorExperiment` | 171 |
| orthogonality | `mbr_orthogonality.py` | `MBROrthogonalityExperiment` | 237 |
| calibration | `mbr_phase_correction.py` | `MBRPhaseCorrectionExperiment` | 445 |
| spectrum | `mbr_spectrum.py` | `MBRSpectrumExperiment` | 1,221 |

God file 5,521 -> 3,853, and what is left is almost all pulse code. Full suite
360 -> 524 tests. Smallest stage first, so the risk ramped up rather than down;
by the time the spectrum stage moved, the mechanism had been exercised three
times.

Each class subclasses the god Experiment, which still holds the loading layer.
Per spec 7.4 no new aggregate base was invented ahead of the duplication that
would justify it.

### Two forwarding layers, both load-bearing, both transitional

`stage=` still works, delegating through a lazy `_stage_owner()`. jonginn's two
notebooks enter that way and `analysis_notebooks/guan/MBR_analysis.py` -- which
is the migration exemplar, not a compatibility surface -- is fully migrated off
it and is what to copy from.

Beyond that, **two** `__getattr__` hooks are needed, and finding out why cost a
regression and a golden failure:

1. **Metaclass `__getattr__`** on the god class. jonginn's notebook says
   `EncSpec.orthogonality_batch(...)`, which is attribute access on the *class*.
   The module-level `__getattr__` that forwards moved *classes* cannot see it,
   so the orthogonality split broke that call site silently -- nothing in the
   repo calls it and no test looked. Now 19 class-level attribute names from the
   two notebooks are pinned by test, so the next split fails loudly.

2. **Instance `__getattr__`** on the god class. The facade hands `self`, a god
   instance, to the owning class's `analyze`, whose moved body calls
   `self.reconstruct_spectroscopy` -- which lives on a subclass. The metaclass
   does not cover instance lookup. The golden caught this one immediately.

Both guard against forwarding a name the owner does not actually define: stage
classes inherit both hooks, so a blind `getattr` recurses instead of raising.

### The pin, and what it cannot do

`tests/test_mbr_stage_split.py` AST-pins every moved method to the god class at
that stage's pre-split commit, and for the spectrum stage also pins the two
dispatch *bodies* (104 and 28 statements) -- the largest moved block should not
be the one block without a net. A companion test asserts the god dispatch now
holds exactly one delegating statement per branch, so code cannot exist in both
places.

Where a name the move invalidated had to be re-addressed, the substitution is
**declared** rather than left to weaken the pin, and a test asserts each
declared edit still matches something at its pin -- an edit that no longer
applies is a hole, not a harmless leftover.

What an AST pin cannot catch is a broken *delegation*: byte-identical method,
facade that no longer reaches it or reaches it with the wrong arguments. The
golden only drives spectrum and calibration, so orthogonality and propagator
got purpose-built runtime tests: synthetic data through both the new display and
the facade, the foreign-data guard, and a monkeypatch spy pinning the propagator
call's arity and argument order.

### Deliberately not done

- **`analyze` still takes `**kwargs`.** It is the moved branch body unchanged.
  An explicit signature is what makes the spectrum stage pleasant to drive
  interactively, and it is a real edit, so it wants its own commit with the
  golden as the net. This is the highest-value next step for the stated goal.
- **The loading layer stayed on the god class.** Extracting it to
  `MBRAnalysisBase` cannot be lazy -- it is a *base class* - and it needs
  `DarkBaseExperiment`, which still lives in the god module. So it requires
  moving `DarkBaseExperiment` (plus `classify_two_parity_readouts` and
  `flatten_exp_lists`) out first. That is a real restructure of the dark-mode
  half, not a fifth mechanical stage split, and it should be decided on its own
  terms.
- **The `staticmethod` aliases to `fitting/qsim`** stayed on the god class. All
  four stage classes inherit them, and `EncSpec.analyze_spectrum` keeps
  resolving with no forwarding needed, which is strictly better during the
  transition.

## 2026-08-27 (end) -- shims dropped, explicit signature

Two follow-ups to the stage split, both reversing decisions made earlier the
same day.

### The shims are gone

Direction change, on request: a forwarding layer is a migration that never
happens, and the new API was the point of the split. Break the old call sites
loudly, once, and hand over a worked example instead.

Deleted: the `analyze(stage=...)` dispatch, the `_StageForwarding` metaclass
`__getattr__`, the instance `__getattr__`, `_MOVED_METHODS`, `_STAGE_OWNERS`,
`_stage_owner`. God file 3,853 -> 3,640.

What replaces them is a message, not a redirect. `STAGE_CLASSES` maps stage to
class and `stage=` raises `TypeError` carrying the import line and the three
calls to make. `display()` does the same when handed aggregate data, since it
sniffed `self.data` rather than dispatching on `stage`.

`MBR_analysis.py` opens with the migration reference: the stage->class table,
all 14 moved class-level methods with their new owners, and an explicit "do not
touch this" list. **The breakage list was computed against the live classes, not
written from memory** -- of the 19 names jonginn addresses, 10 still resolve and
9 moved. Worth repeating for the next such break: guessing that list would have
produced a table that is wrong in both directions.

`tests/mbr_reference.py` migrated too. It was a consumer like any other, and
the golden found every site.

### What must not move, and why it was checked

`BatchRunner` records `ExptClass.__name__` and `.__module__` into the job, and
jonginn submits with `ExptClass=EncSpec`. So the god class's **name and module
are recorded provenance**, and renaming or relocating it would break data taking
and the re-analysis of existing jobs. That is a different decision from breaking
an analysis call site, and it was left alone. Renaming it to something honest
like `MBRJobExperiment` is possible, but only alongside a provenance story.

Also confirmed: `BatchRunner` never calls the aggregate `analyze`/`display`, so
removing the facade could not affect acquisition. The remaining `analyze` on the
god class is the per-job quadrature, which *is* what the worker runs after
`acquire` -- the one branch that was ever this class's own work.

### Explicit signature

`MBRSpectrumExperiment.analyze` had 31 `kwargs.get` lookups. The 12 real knobs
are now named parameters; the 19 `mpm_*` forwards collapsed to
`**matrix_pencil_options`.

The collapse is sound for a checkable reason: every default the old block spelled
out was **identical** to `analyze_matrix_pencil`'s own, and it forwarded all of
them and nothing else. Verified against the live signature before editing, and a
test now pins that agreement so a later default change in `analyze_matrix_pencil`
cannot silently redefine what the old call site meant.

The live trap this closes: `zero_paddding=2` or `mpm_pencil_lenght=5` used to run
to completion with the knob doing nothing. Both raise now.

Retired the `analyze` row from the dispatch-body pin rather than re-blessing it.
The pin existed to prove a move; the method has since been rewritten on purpose,
so re-blessing would pin the rewrite to itself. The golden is the right net for
an intentional edit -- 11 tests across fft, matrix_pencil, complete basis and
both phase frames, green and unchanged.

### Standing after today

God file 5,521 -> 3,640, almost entirely pulse code. Suite 360 -> 479.

Open, in the order they seem worth doing:

1. The same explicit-signature treatment for the other three stages' `analyze`.
   They are much smaller and none has a `**kwargs` trap of this size.
2. `DarkBaseExperiment` (plus `classify_two_parity_readouts` and
   `flatten_exp_lists`) out of the god module -- the prerequisite for extracting
   the loading layer as `MBRAnalysisBase`, since a base class cannot resolve
   lazily.
3. `subsample_spectroscopy_shots` (194 lines) -> `fitting/qsim/`
   `mbr_reconstruction.py`, the last item on the old step-2 worklist.
4. The section 6 naming review, which now has to reckon with the god class name
   being recorded provenance.

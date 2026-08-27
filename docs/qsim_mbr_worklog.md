# MBR refactor worklog

Running handoff for the refactor in `docs/qsim_mbr_refactor.md`. Newest entry
last. Each entry says what landed, how to check it, and what is next, so a
session that starts cold can pick up from the bottom.

Read first: spec section 14 (working order), section 13 (execution
environment), appendix C (configuration traps).

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

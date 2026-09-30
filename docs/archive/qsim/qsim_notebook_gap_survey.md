# Stage 2 gap survey: the two notebooks against the target shape

> **Archived 2026-09-14; superseded as planning guidance.** Preserved from the
> working tree, including evolving and conflicting interpretations. Read the
> [current roadmap](../../qsim_refactor_surface_map.md) and
> [stage-two map](../../qsim/stage2_notebook_map.md) for current direction.

Surveyed 2026-09-14 on `guan`. Source only; no notebook was executed. Counts
are non-comment, non-blank statement lines from an AST walk of each code cell,
so they differ from the raw line counts in the
[cell inventory](../../qsim/evidence/qsim_notebook_surface_inventory.md).

Q = `measurement_notebooks/jonginn/qsim_experiments.ipynb`
P = `measurement_notebooks/jonginn/data_postprocess.ipynb`

## The target shape

Measurement, from `measurement_notebooks/guan/single_qubit_autocalibrate_v2.py`:

```python
foo_defaults = AttrDict(dict(...))                  # data, no logic
def foo_preproc(station, default_expt_cfg, **kwargs): ...
def foo_postproc(station, expt): ...
foo_runner = CharacterizationRunner(station=..., ExptClass=..., 
    default_expt_cfg=foo_defaults, preprocessor=foo_preproc, 
    postprocessor=foo_postproc, job_client=client)
foo = foo_runner.execute(span=5)
foo.display()
```

Analysis: `Expt.from_hdf5(...)`, then `.analyze()`, then `.display()`.

## Where the 19,356 lines are

| Bucket | Q | P | total | share |
|---|---|---|---|---|
| A. `def`/`class` bodies | 1,366 | 3,508 | 4,874 | 25% |
| B. literal data (catalogs, defaults) | 489 | 1,014 | 1,503 | 8% |
| C. straight-line assignments | 2,263 | 1,440 | 3,703 | 19% |
| D. control-flow blocks | 3,249 | 3,561 | 6,810 | 35% |
| E. bare calls (plot/print/analyze) | 891 | 1,227 | 2,118 | 11% |
| imports | 141 | 207 | 348 | 2% |

**Moving the utility definitions out en masse is bucket A: 25%.** Worth doing
and nearly free -- P cell 4 is 1,335 lines with **zero** free variables, so it
relocates without a single signature decision, and so does the 268-line
dataset catalog in P258. But it does not get the line count under control.
C+D+E, 65%, is the gap.

## The gap is algorithm bodies missing their `def` line

C and D are not glue. They are whole procedures written as straight-line
top-level code:

- **P172** (253 lines): FFT local-density-of-states reconstruction plus
  construction of the fixed-N target Hamiltonian. This is an `analyze` body.
- **P188** (221 lines): a complete Matrix-Pencil estimator -- block Hankel,
  SVD, rank selection, pole merging.
- **P251** (245 lines): level-comparison figure. This is a `display` body.
- **Q342** (299 lines): a disorder campaign -- channel selection, plan
  construction, submission, per-realization analysis, in one cell.

## Why this is more tractable than it looks

The worry is that top-level code is less modular than a god file's methods,
because there is no function boundary to grab. Measured, the opposite holds:
**the cell boundary is already the function boundary, and the signature is
already written in the variable names.**

Count each non-trivial cell's free variables, then subtract the names that are
already importable -- library classes and functions, and helpers defined
elsewhere in the same notebook, none of which are parameters once the cell
becomes a module function. What is left is the prospective signature:

| Real parameters | Q cells | P cells | share |
|---|---|---|---|
| 0-3 | 152 | 137 | 80% |
| 4-8 | 32 | 26 | 16% |
| 9-15 | 6 | 6 | 3% |
| 16+ | 1 | 0 | <1% |

**Four cells in five need three parameters or fewer.** The one wide cell is
Q342 at 32, and all 32 are named `d72_*` -- one campaign config object, not 32
independent globals. That prefix convention holds generally: the notebooks
already namespace each cell group's state (`d72_`, `diag_disorder_`,
`encspec_N3_`, `joint_`, `precision_`, `replot_`, `sff_`, `saved_n3_`), so the
parameter object is derivable from the prefix rather than by reading the
cell. P172's twelve free names -- `A_raw`, `N`, `cycles`, `hamiltonian_dt_us`,
`floquet_couplings_MHz`, `physical_kerr_MHz`, `mode_labels`,
`occupation_strings`, ... -- are already a well-formed `analyze` signature.

So stage 2's unit of work is "add a `def` line and a signature the author has
effectively already written", which is cheaper per line than stage 1's unit
("lift a class out of an 8.7 kLOC file").

## The spectral estimators are variants, not copies

An earlier draft of this survey said P188 reimplements
`fitting/qsim/matrix_pencil.analyze_matrix_pencil`. **That was wrong**, and the
distinction matters for the plan. What matches is the *input and knob
vocabulary* -- P188 opens with `matrix_pencil_requested_max_modes = 35` and
`matrix_pencil_numerical_floor = 1e-10` against the same `reconstruction` and
`spectrum` -- not the algorithm.

Measured two ways. Token-stream similarity across the nine MPM-shaped cells
(P188, P199, P253, P254, P255, P278, P290, P295, P299) and the library peaks at
0.25, and against the library at 0.08. Name-independent fingerprints -- the
ordered numpy/linalg call sequence, which survives renaming -- peak at 0.39.
No pair is a copy.

P188 and P253 do share the real pencil spine (`sliding_window_view` →
`vstack` → `linalg.svd` → `diag` → `linalg.eigvals` → `angle`), but P188's own
headline says what separates them: "Global-only Matrix Pencil diagnostic; the
next cell performs rowwise selection and merging." Global-only against the
library's per-row-then-merge is a deliberate variant.

So these ~3.2 kLOC are **nine estimators needing nine accept-or-retire
decisions**, not duplication to delete. Their comments make the intent legible
and suggest most are a methodological audit rather than candidate APIs: "No
Hamiltonian or expected peak count is used" (P187), "This cell does not run
MPM" (P264), "Theory is read ONLY here, after experimental selection has
finished" (P280), and a section titled "Can the measured level statistics
support a conclusion?" (P281). For a one-time check like that, the finding is
worth keeping and the code is not -- record the answer in a doc and drop the
cell. That is a physics call, not a refactor call.

## Duplication that is real

- **Redefinition within a notebook.** Q defines 12 names more than once, P
  defines 7. Most are byte-identical copies, but P cell 119 redefines six of
  cell 4's functions with *different* bodies (`_cfg_get`, `get_dark_params`,
  `stor_label`, `plot_thick_line_with_dots`, `plot_nmod4_corrected`,
  `plot_multiparity`), and Q defines `stor_label` six times in two variants. So
  which body is live depends on execution order, and an extraction has to pick
  between the variants deliberately.
- **1,503 lines are data, not code.** P258 is a 268-line catalog of job-ID
  ranges per dataset. That is a YAML file.

## Measurement is close; analysis is not

Q already has 31 cells in template-submit shape. Against the autocalibrate
template's roughly 53 lines per experiment, Q spends about 121 -- an
indicative factor of two, and the excess is campaign selection logic wrapped
around the submit, which is exactly what the surface map's area 3 asks to move
into the Experiment classes. Q's own `_defaults` / `_preproc` / `_postproc`
cells already match the template.

P is far from `from_hdf5().analyze().display()`: one submit cell, 11.5 kLOC of
analysis, and only 14 cells already in that shape. Nearly all of P's bucket
C/D is `analyze` and `display` bodies for the four MBR stage classes, plus the
exploratory estimators that need an accept/retire decision rather than a move.

## If the dormant physics is dropped

Dark mode, flux excursion, Wigner, cooling and the old sideband/debug scratch
are not under active work; MBR (`EncSpec` and relatives) is. Splitting both
notebooks by family, using the cell-inventory ranges:

| | Q | P | total |
|---|---|---|---|
| MBR and its shared prerequisites | 3,811 (45%) | 5,582 (51%) | 9,393 (48%) |
| dormant physics | 4,332 (51%) | 2,207 (20%) | 6,539 (33%) |
| undecided (estimator explorations, SFF) | 407 (5%) | 3,231 (29%) | 3,638 (19%) |

**Dropping the dormant physics removes a third.** Retiring the exploratory
estimators too would leave 9,393 lines -- under half of what is there now, and
the part that has to be correct anyway.

The split is asymmetric and that shapes the order of work. Q is half dormant,
so Q shrinks mostly by deletion. P is only a fifth dormant; its bulk is MBR
analysis and estimator explorations, so P shrinks by extraction and by
accept-or-retire decisions.

### The biggest utility cell is almost entirely dormant

P cell 4 is the 1,456-line, 71-definition cell that the "move the utils out en
masse" plan would relocate wholesale. A reachability pass over its internal
call graph, seeded from every other cell that names one of its definitions:

| | defs | lines |
|---|---|---|
| reachable from MBR cells | 7 | **98** |
| reachable only from dormant cells | 56 | 1,111 |
| reachable from nothing | 8 | 247 |

The seven that MBR needs are `load_encoding_spectroscopy`, `job_id_generator`,
`load_dark_experiments` and `state_label`, plus the three helpers they pull in
(`check_program_class`, `hdf5_path_generator`, `path_to_experiment`). Those are
the loading layer -- the same ground as
`EncodingHamiltonianSpectroscopyExperiment.from_job_ids`/`from_job_files`, which
is why the loading foundation is the first job either way.

So moving this cell to a utility module en masse would import 1,358 lines of
dormant and dead code into the library. Under a drop-the-dormant-physics
decision it is a 98-line reconciliation instead. **Prune before moving, not
after** -- for this cell the difference is 15x.

### The prefixes are section numbers

`d72_` is section **7-2**, "Occupation-constrained disorder spectroscopy";
`diag_disorder_` is 7-1, the diagonal version. The others follow
(`encspec_N3_`, `joint_`, `precision_`, `replot_`, `sff_`, `saved_n3_`). They
are positional labels, so they carry no meaning once a cell leaves the
notebook and every extracted signature needs a real name.

One useful side effect of that discipline: Q's section headings already mark
every cell "-- submits jobs" or "-- no jobs", so the author has already
labelled the acquisition/analysis split that the two target notebook shapes
need.

## "Estimator" is our word, and it hid what these cells are

The inventory's categories "saved-shot estimator explorations" and "additional
spectral estimators" are refactor-doc vocabulary. `estimator` occurs **three
times** in either notebook, all in P and all in the narrow statistical sense:
an "unbiased SFF estimator", and the two named `state_only` /
`state_theta_phi` estimators of the same `Tr U(t)`. Calling 3.2 kLOC "estimator
explorations" made a category out of a word jonginn used three times, and made
disciplined work look like scratch. What the cells are:

**P193-203 -- a shot-budget study and a sampling protocol.** Two questions
answered from saved data with no new measurement. First, how few shots per
point do we need: randomly keep fewer final-readout shots and rerun the whole
reconstruction → phase-correction → FFT → MPM chain, with calibration held at
full statistics so only spectroscopy shot noise varies. Second, can `Tr U(t)`
be built by *randomly sampling* occupations shot by shot instead of scanning
all 35 rows -- `state_only` randomizes occupation, `state_theta_phi`
randomizes occupation, theta and phi. That second one is a prototype
acquisition protocol, and it is how MBR would reach larger N where scanning
every row is not possible.

**P252-306 -- five alternative pole-selection rules and their cross-checks,**
all answering one question: is the 35-level spectrum real, or an artifact of
the rule that picked the poles?

| cells | route |
|---|---|
| 252-255 | joint block-Hankel: stack every occupation's Hankel within one realization, force rank 35 |
| 277-280 | score candidates by row-normalized component singular value instead of rank persistence |
| 281-282 | vary *only* the cross-row merge tolerance; see whether level statistics move |
| 283-303 | fit frequencies/decays/amplitudes jointly; split shots into disjoint halves, train on one, score the other |
| 304-306 | pool stable row candidates, merge tightly, keep the 35 best-scoring clusters |

**The methodology is the part worth keeping.** Nearly every cell states what it
does *not* read, which is the right guard for a level-statistics claim: "No
spectrum, Hamiltonian, or theory frequencies are passed to this function"
(P278); "Theory is read ONLY here, after experimental selection has finished"
(P280); "No Hamiltonian, eigenvectors, target levels, or target statistics are
inputs" (P294); "Only now may configured-H values enter, and ONLY as a
comparison" (P297); "Stability is necessary, not sufficient: a reproducible
extraction bias can remain" (P281). If these cells are shed, that reasoning has
to survive in writing, or the spectrum result loses its support.

## Execution counts: the one activity signal that survived

Outputs are routinely cleared, so there are no timestamps. But **42 of P's
cells and 61 of H's kept their `execution_count`**, which gives the order of
the last surviving kernel session. Read it one-sidedly: presence shows recent
activity, absence shows nothing at all.

In P, counts 107 through 129 walk straight through 278 → 280 → 282 → 288 →
290-297 → 299 → 301 → 303 → 305-306. **The pole-selection cross-checks are the
most recently active part of the notebook**, not a dormant exploration --
which reverses this survey's first reading of them. Counts 136-152 then return
to HDF5 reprocessing and rebuild the shot halves. In the flux-excursion,
Wigner and dark-mode ranges, exactly one cell has a surviving count (c6, a
sweep plot).

H's trace is the acquisition mirror: Floquet gain/phase calibration, MBR phase
calibration, orthogonality, spectroscopy, and the 7-2 disorder batches.

### One correction to the drop list

H runs dark-mode cells 152-159 at counts 65-199 -- the **"Bare"
sideband-scramble readout** (~250 lines, `sideband_scramble_preproc` with
`ExptClass=meas.QsimBaseExperiment`). That is plausibly an MBR prerequisite
check rather than the dark-mode project, so the boundary is "dark mode minus
the bare scramble readout", not the whole 150-262 range. The rest of that
range -- displace/multiparity, parity confusion, dark T1, disorder decay --
has no surviving count in either notebook.

### Notebook sizes

For the record, since it came up: Q is 0.5 MB with zero stored outputs; P is
5.7 MB with 27 figures; H is 15.3 MB with 179. The size is stored figures, and
Q -- the one with none -- is the small one.

## What this survey does not establish

Which of the nine Matrix-Pencil-shaped cells are real variants and which are
copies; whether any exploratory estimator should become a supported API; and
whether the physics in any extracted block is right. It is a shape survey.

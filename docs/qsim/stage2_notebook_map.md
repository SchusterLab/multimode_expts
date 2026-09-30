# Stage 2 execution instructions: tear the notebooks into inspectable pieces

**Deliverable: small themed notebooks and ordinary Python functions.** This is
structural decomposition, not correctness validation or final API design.
Breaking old entry points is acceptable. 
Minimize compatibility shims added/historical behavior repairs. 
These instructions supersede older notebook plans and follow-ups. 
Read the source; no historical-doc survey is required. 
**This is cut-and-paste relocation, including dormant code.**
Keep it in the working tree; delete only confirmed duplicates whose surviving copy is identified. 
Lack of a known caller does not establish duplication.

## 1. Cut by section

Sources, under `measurement_notebooks/jonginn/`:
**Q** = `qsim_experiments.ipynb`; **P** = `data_postprocess.ipynb`.
Cell numbers below are one-based physical positions, including markdown, before
editing. Use headings to confirm boundaries. The high-Kerr sibling is outside
this task; do not reconcile it or maintain it as a compatibility target.

For code generation convenience, use Jupytext-style `.py` notebooks (`# %%` cells) for output. 
Measurement entries go in `measurement_notebooks/202609_qsim_migration/`; 
analysis entries in `analysis_notebooks/202609_qsim_migration/`.

| Theme / basename | Measurement source | Analysis source |
|---|---|---|
| `multiphoton_calibration` | Q7–57, 61–68 | — |
| `floquet_calibration` | Q58–60, 69–122, bare-readout setup/check Q150–158 | — |
| `mbr` | Q286–307, 315–317 | P167–187, 189–192, 205–240; Q308–314 |
| `mbr_disorder` | Q318–349: settings, planning, submission | P241–251, 256–276; Q318–349: analysis/reports |
| `mbr_sampling` | — | P193–203 |
| `mbr_spectral_validation` | — | P188, 252–255, 277–306 |
| `mbr_tomography` | Q350–354 | Q355–358 |
| `mbr_sff` | Q359–365 | Q366–369 |
| `floquet_displacement_kerr` | Q370–374 | — |

For mixed sections, put job submission and plan preview in measurement;
put saved-data fitting/reports in analysis. Immediate post-job analysis may
remain a call in measurement. Copy necessary setup from Q1–6/P1–4; hoist its
implementation in step 2. Carry explanatory markdown with the relevant code.

**Move dormant sections into `dormant/` under the respective notebook directory:**
Q123–149 → `cooling` (this is an exception to the no deletion rule: if the contents appear to be not meaningfully modified compared to the source under measurement_notebooks/guan/qsim_experiments.py, free to delete); 
Q159–262 → `dark_mode`; Q263–285 → `debug`;
P5–93 → `flux_excursion`; P94–103 → `wigner`; P104–153 → `dark_mode`;
P154–166 → `pulse_scratch`. Carry their markdown and helpers along. 
Outputs can stay in original ipynb (which will be deleted in one collective deletion run to enable easy recovery in case they are needed). 
Keep dormant-only helpers there; copy shared helpers as needed. Park definitions
with unclear ownership in `dormant/unassigned_helpers.py`, rather than dropping
them. P204 is empty. Dormant notebooks need only relocation, not steps 2–3.
Remove the original Q/P containers only after all content has a destination
in the working tree (or an identified surviving duplicate).
Leave other users' notebooks and existing library modules alone in this pass.

## 2. Hoist definitions and bulky data

For each theme, move top-level `def`/`class` bodies into `experiments/qsim/notebook_helpers/*.py`; 
replace them with explicit imports. These are temporary homes. 
Do not first reconcile them with existing library implementations yet.
If encountering large paragraphs that are pure data catalogs, move to a yaml under the same cwd as the notebook citing them; 
keep dataset choice and small editable settings visible in the notebook.

Helpers that ideally belong in modules should also move to `experiments/qsim/notebook_helpers/`. 
The names of these helper modules should indicate their primary callers: one specific notebook/shared/a physics theme etc.
Move helpers with no active caller to the appropriate dormant notebook or `unassigned_helpers.py`.
If different definitions reuse a name, give them theme-specific names and wire
each section to its local version; do not decide which algorithm is better.

## 3. Turn long procedural blocks into functions

Work in source order, one existing commented block or cell group at a time:

1. Give the operation a descriptive name: `select_channels`, `build_plan`,
   `run_campaign`, `reconstruct_traces`, `fit_shared_frequencies`, `plot_levels`.
2. Move its body to the theme module with minimal internal edits.
3. Names read from earlier cells become arguments; names used by later cells
   become explicit returns. Imports and hoisted helpers are not arguments.
4. Group a repeated settings prefix such as `d72_*` into a plainly named
   `campaign_config` dictionary. Return a named result or dictionary for many
   outputs. Keep datasets, config and computed results separate. Do not pass
   `globals()`, copy the whole kernel namespace, or use `exec`/`%run` to hide it.
5. Replace the old block with the function call. Keep submission, fitting and
   plotting as separate calls where the source already separates them.

Keep algorithm variants separate. Only consolidate confirmed duplicates; do not redesign inheritance,
rename every internal variable, or fix physics while extracting. If a moved
function is still huge, split at its existing commented substeps. Split an
oversized module by operation (`_<theme>_planning.py`, `_<theme>_fitting.py`,
`_<theme>_plots.py`). 
Aim for inspectable notebook structures: 
a measurement job submission should essentially follow the `defaults dict -> pre/post processor -> execute()` pattern for each job;
an analysis should target the canonical `Experiment.from_h5file()`, `.analyze()`, `.display()` pattern.

## 4. Check the structure and stop

Use (based)pyright to syntax-check the new files without executing notebook
cells. Inspect imports, explicit inputs/returns and remaining long blocks.
Fix syntax errors and obvious names/imports broken by the move. 
Testing via mock mode/rerunning analysis on existing data files to confirm basic correctness is OK. 
Do not submit real experiment jobs yet.

For deeper failures, leave a short `TODO` at the relevant function and report
it. Missing scientific inputs should fail visibly, not acquire guessed defaults
or silent fallbacks. Historical loading, readout layout, numerical agreement
and hardware correctness are later theme-specific validation tasks.

**Done:** active sections are in the named entries; their definitions and long
procedures are in inspectable modules; dormant sections and helpers remain in
separate notebooks/files. Account for all source content before retiring Q/P.
Report files produced, structural checks, and known breaks. 
Fresh-kernel execution and result (analysis or ASM) equivalence are bonuses, not must haves.

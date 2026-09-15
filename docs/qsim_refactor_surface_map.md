# Qsim refactor: the working surface map

Updated 2026-09-15. **Stage 1 (library decomposition) is done. Stage 2
(notebook decomposition) has landed structurally.** Jonginn's
`qsim_experiments.ipynb` (Q) and `data_postprocess.ipynb` (P) are now sixteen
themed Jupytext entry points under
`measurement_notebooks/202609_qsim_migration/` and
`analysis_notebooks/202609_qsim_migration/`, with seventeen helper modules in
`experiments/qsim/notebook_helpers/`. Q and P themselves are retired. The
high-Kerr sibling was outside this pass and is untouched.

**What "structurally" excludes:** no notebook has been run end to end and no
job submitted. Per-theme scientific validation — numerical agreement with the
originals, readout layout, historical loading — is the next pass, and the
worklog lists the specific divergences it needs to settle. Three are worth
knowing before touching this code: the notebook-local HDF5 loader that runs
parallel to `from_h5file`, the cell-156 Floquet cycle duration that
`floquet_timing.py` documents as ~1.2% long, and `threadpoolctl` being
imported but absent from the environment.

See `docs/archive/qsim/qsim_mbr_worklog.md`, entry 2026-09-15, for what
collapsed into what and why.

## The whole surface

| Area | What belongs together | Stage-two destination |
|---|---|---|
| **Prepare and calibrate** | Broadband qubit, N-photon swaps, readout, Floquet gain/frequency/phase; bare-readout checks needed by MBR | Initial h1 sections in the measurement campaigns they enable. |
| **Four core MBR products** | Phase calibration · orthogonality · spectroscopy · propagator | A core acquisition entry point and saved-data analysis entry point; reuse the four existing Experiment owners. N and diagonal/off-diagonal returns are inputs. |
| **Disorder campaigns** | Choose channels → preview plan → acquire → reconstruct realizations → pool/report | Separate campaign acquisition and analysis notebooks. Planning, loops and reports leave the cells. |
| **Measurement and inference studies** | Reduced-shot / randomized-occupation replay; pole selection, shared-frequency fitting, independent-shot checks | Two analysis workspaces: shot/sampling studies and spectral validation. Preserve the methods and their evidence; library promotion can be decided later. |
| **Adjacent extensions** | Propagator tomography; direct disorder SFF; displacement Floquet Kerr | Separate recipes as needed. Keep their distinct measurement/analysis identities. |
| **Historical projects** | Literal dark mode, flux excursion, cooling, Wigner, debug | Move sections, outputs and their helpers into separate dormant notebooks in the working tree, outside the active path. Delete only confirmed duplicates. |

**The notebook keeps the scientific choices:** defaults, dataset selection,
small pre/post hooks, run/load, analyze, display, and interpretation. The
eventual library owns repeated planning, reconstruction, fitting and report
machinery. For this pass, temporary modules beside the notebooks are enough;
finding final homes and reconciling APIs can wait.

**Order:** relocate dormant sections and partition the rest → hoist definitions
and bulky data → wrap procedural blocks with explicit inputs and returns.

**This pass finishes at inspectability.** Keep retained algorithms distinct
and move their bodies with minimal edits. Breaking old entry points is allowed;
no compatibility shims or end-to-end repair are required. Preserve relocated
content in the working tree, including dormant code and stored outputs.
Fix extraction syntax/import mistakes; validate scientific and
runtime correctness afterward, one theme at a time.

[Self-contained execution instructions](qsim/stage2_notebook_map.md).
The cell inventory and archived findings are optional evidence, not additional
instructions or acceptance gates for this pass.

This is the current direction, not a worklog. The former use of “stage 2” for
general library API cleanup is superseded here; that cleanup follows a
concrete workflow need rather than becoming another prerequisite project.

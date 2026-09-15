# Archived: Qsim implementation follow-ups

> Superseded as stage-two instructions. Retained for later correctness work;
> these findings do not gate structural decomposition.

Carried forward at the 2026-09-14 documentation reset. These are recorded
findings to verify when implementing the [notebook plan](../../qsim/stage2_notebook_map.md),
not a new prerequisite roadmap. No fixes are claimed by this documentation pass.

## Saved-data loading and acquisition identity

The [last worklog entry](qsim_mbr_worklog.md) records decisions
to retain `EncodingHamiltonianSpectroscopyExperiment` for new MBR acquisition
provenance and use saved `program_class`, rather than the historical
`experiment_class` name, to distinguish acquisition programs. Keep analysis
ownership in the four stage classes through the existing loading/batch seam.
Do not assume one acquisition program identifies every analysis product;
verify the required stage/config distinction on the actual workflow.

Historical files have several recorded acquisition class names. Preserve
those identities and filenames. Reconcile P4's active loading helpers and
P261's `SavedSpectroscopyExperiment` with the current loader, especially saved
decoder and physical-clock semantics. Use HDF5 metadata for durable replay;
dataset catalogs locate records rather than silently supplying today's timing.

**Scope:** blocks a claimed working replay for affected files, not the
physical notebook partition. Done when representative historical identities,
timing conventions and partial grids load correctly or fail explicitly.

## Shot/readout layout

The same worklog reports disagreement among `QsimBaseExperiment.acquire`,
`readout_lane_count` and `MM_base.lane_layout`, and a fallback in analysis that
can average lanes as repetitions. Reproduce the relevant cases before fixing.
Historical jobs may lack saved `read_num`; applying today's formula alone
does not establish their actual hardware layout.

**Scope:** isolate as a correctness change with focused historical/current
layout tests. Do not silently fix it as part of a code move. This is essential
before trusting affected raw-shot replay, but does not require redesigning
all pulse/config contracts before splitting the workspaces.

## Extraction details to resolve at their workspace

- **Shared helpers:** P4 includes substantial historical code; P119 and other
  cells redefine names. Trace needed dependencies and choose variants explicitly.
- **Campaigns:** preserve selectors, calibration coverage, preview/submission
  boundaries and completion bookkeeping. Runner consolidation is a separate
  cleanup unless the extraction demonstrates it is necessary.
- **Spectral studies:** retain named algorithms and separation of inference
  from theory comparison. Similar vocabulary or low token similarity does not
  prove algorithms equivalent or distinct; inspect before consolidating.
- **H and retained notebooks:** reconcile real configuration/code differences;
  preserve stored figures. Execution counters show a surviving run order,
  not timestamps, current use, correctness or permission to retire a workflow.

The [archived gap survey](qsim_notebook_gap_survey.md) preserves
the measurements and evolving interpretations behind these points. Its
free-variable counts and proposed deletion savings are not acceptance criteria.

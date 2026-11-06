# Sequential AGP naming review

The user requested readable names and an explanation of the temporary objects,
then clarified that this follow-up should improve readability and report why
instantiation is used. No architecture change was requested in this follow-up.

- Reviewed function names in all three new source modules and the new notebook.
  Framework overrides and the existing public protocol names remain unchanged.
- Renamed `_mode_views()` to `_create_single_mode_analysis_experiments()` so its
  name identifies both the action and the objects it creates.
- Replaced opaque AGP locals with `source_job`, `readout_step`, `single_mode_input`,
  `single_mode_agp`, `source_jobs_by_occupation`, and explicit source/target field names.
  Expanded the combined result assignment and documented the skipped acquisition setup.
- Updated the existing AGP integration test's private-method reference. No numerical,
  acquisition, persistence or notebook behavior changed. Existing base files are untouched.

`FockPopulationExperiment.__new__` creates temporary processed-data inputs with
the cfg/data/metadata interface required by existing AGP. It skips acquisition
initialization. The temporary existing AGP object reuses scalar input validation,
ordering and the complete analysis. These object constructions are compatibility
choices under the unchanged existing API, not physical requirements or additional
measurements. A direct `super().analyze()` would still execute legacy scalar
population stacking on native sequential children.

Ruff passes. The existing sequential AGP integration module passes all 9 tests,
including numerical comparisons, source preservation and HDF5/manifest reload.

## Single-mode HDF5 naming follow-up

The user identified `export_mode()` and the phrase "scalar projection" as
unreadable. Renamed the method to `save_single_mode_population_h5(ro_stor, path)`
and described the actual output: one mode's populations, moments, errors and
optional correction matrix, with original measurement filename/config recorded.
The selector argument and locals now identify the readout step and saved data.
Updated the existing test call and replaced projection jargon in its names.
The new sequential AGP and notebook do not call this optional writer. File
contents and numerical behavior are unchanged; existing base modules remain
untouched. No compatibility alias or new helper was added.
The two existing raw/corrected save-and-reload cases pass; Ruff also passes.

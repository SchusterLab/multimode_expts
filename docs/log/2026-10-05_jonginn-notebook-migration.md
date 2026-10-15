# 2026-10-05: complete the newer Jonginn notebook migration

## Authority and decisions

Guan asked to finish the migration using Jonginn's explanation docs. Read the
Sept 30 inventory on `guan`, `Notebook_Program_Labeling.md` on `job_id_sorting_out`,
and Jonginn's comments on GitHub issues [7](https://github.com/SchusterLab/multimode_expts/issues/7)
and [6](https://github.com/SchusterLab/multimode_expts/issues/6).

Jonginn chose RMS (matching simulation), the full fixed-N basis rather than
visibility-selected states, and interleaved single-shot calibration. His final
correction is essential: stale readout thresholds invalidate active reset, so
postprocessing cannot replace recalibration. Sep10 Matrix Pencil tuning was
exploratory; exact grouped debug plots need not be preserved. Hardcoded timing
should be discarded; he verified `floquet_cycle_us`. High-Kerr copies, the N2
decoder-mode section, Sept05 dark-mode dataset cells and uncommitted prod scratch
edits may go. He agreed to stop editing the legacy notebooks.

## Changes

Shared implementation committed as `0349bd8` on `guan`, then brought into
`qsim-analysis` by a normal merge. The companion analysis change adds a Sep10
catalog-backed entry point and updates the numerics status. No main/prod deployment
or remote push is part of this session. The user's existing pole-finding recap
edit is preserved.

- New campaign defaults: full basis, RMS. Explicit custom occupations are checked
  against the fixed-N basis. Historical theory/norm planning is an explicit
  compatibility mode, so older baselines and existing saved onsite energies are
  not silently changed. Planning counts the actual occupations (35 per N=3
  realization), with single-shot overhead excluded from its time estimate.
- `realization_spectrum` persists each realization record in every child job
  config: seed, normalization, direction, strength, onsite energies, states and Kerr.
- Spectrum acquisition accepts `before_batch`; each batch completes before the
  next calibration/submission. The notebook defaults to recalibration before
  every occupation. Histogram calibration runs with active reset off, then uses
  the existing shared station-calibration function; invalid centers/thresholds
  raise before station mutation. Subsequent job configs hold calibration files
  and queue IDs; durable physics provenance does not depend on jobs.db.
- Preserved the existing two-Gaussian IQ fitter in `fitting/qsim/readout.py` and
  adapted final-shot refitting to current four-phase MBR jobs. It is optional,
  rejects unsuitable shots, uses analysis-only copies, and persists fit metadata
  in derived manifests. It cannot repair active reset. No raw files are overwritten.
- MBR section 2 now offers the full fixed-N orthogonality matrix for normal runs,
  retaining its short smoke check. Historical 7-1 analysis remains available.
- Deleted the five old Jonginn qsim/reprocessing/high-Kerr notebooks, the obsolete
  stage migration script, and `dormant/mbr_n2_decoder_mode.py`. Removed all notebook
  test exceptions/xfails, retaining successor checks. Unrelated dormant dark-mode
  work, deprecated modules and historical N2 file provenance are retained.
- Sep10 data were already cataloged/converted Oct 2. The analysis companion loads
  them through that catalog, displays levels/spectra/SFF and offers explicit IQ
  refitting. No copied hardcoded job ranges/timing, experimental fitter defaults,
  or duplicate debug plotting code.

Mapping of every Sept 30 leftover: `docs/qsim/notebook_migration_completion.md`;
the tentative inventory is marked superseded. Both theme status files and the
cross-theme known-failure list are updated. There are now three known deliberate
Matrix Pencil failures rather than five (the two stage-dispatch notebook failures
are removed).

## Validation

Final focused checks: **209 passed**. They cover notebook imports/names, successor
attributes, the repo-wide stage-dispatch sweep, standard single-shot station
updates, new full/custom planning and metadata, calibration ordering, an unusable
Histogram stopping before mutation, synthetic rotated-IQ recovery without raw
mutation, and mocked MBR acquisition.

Broader MBR/pole checks: **301 passed, 16 skipped, 9 xfailed, 4 failed, 15 errors**.
Every failure/error is `JobPathError` from the unmounted `C:/experiments` data tree;
mock notebook cases also skip unavailable config/server setup. This is not a
passing real-data suite. No real hardware or production resources were accessed.
Diff whitespace checks pass.

## Remaining

- Publish/review the branch changes and merge `guan` into `main` before production
  use. Check a single interleaved Histogram/TimeTrace pair on hardware and rerun
  saved-data analysis with pippin's data mounted; Sep10 notebook execution is not
  validated here.
- D72 off-diagonal calibration/conversion, calibration sections labeled not expected
  to work, the general deprecated/dormant-code cleanup, fitter selection and the
  physics audit remain separate tasks.
- Jonginn authorized discarding prod scratch edits, but this local session did not
  inspect or discard those files. Do that only after checking the current prod state.

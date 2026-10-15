# 2026-10-02: one data set catalog, and the second conversion round

Theme: analysis (`qsim-analysis`, on pippin). Session with guan (gzhwang).

## Question

jonginn pushed more job-ID lists to `job_id_sorting_out` (commits up to `91da3f7`) in answer to
issues 6 and 7. Does that give enough shape to run one round of conversion to the new HDF5
layout (four Ramsey phase combinations per file, manifest next to converted and raw data) and to
bring the new data sets into the analysis workflow?

## What was found

- The layout and the tool already existed since 2026-09-24 (`docs/qsim/mbr_redesign.md`
  section 6, `tools/migrate_mbr_jobs.py`): one job sweeps `ramsey_phase` over the four
  [prep, analyzer] pairs; the tool merges the old phase-0/phase-90 file pairs into that layout.
  Six data sets were converted then.
- jonginn's catalog (16 sections) and the JSON the code read (`tests/data/mbr_datasets.json`,
  21 entries) overlapped. All 3140 catalog job IDs resolve to files. New against the JSON:
  two N=1 sets (Jul 22, Aug 28), three orthogonality sets (Aug 19, Aug 26 x2), the Aug 24-25
  complete N=3 set and two disordered sets (Aug 19, Aug 27), all marked "quality concern" by
  jonginn, and the Sep 10-14 full-basis disorder set with 9 realizations (the JSON had 1).
- The Sep 10-14 set is diagonal, not D72: all 630 spectroscopy files have no `offdiag_cycles`
  and initial == final occupation, 35 occupations per realization (checked by script). The
  JSON label `d72_Sep10_K3p6_g29p2` was wrong.
- Its raw files record the disorder realization under names the tool did not know: `d73_*`
  (Sep 10) and flat `realization` / `seed` / `disorder_strength_kHz` / `target_onsite_MHz`
  (Sep 11-14). The files number r1-r8 as 0-7 and the Sep 11 remeasurement of r0 as a seedless
  "manual" realization. Every file carries the target onsite detunings, so the Hamiltonian is
  reconstructible.
- The provenance sidecar (`tests/data/job_provenance.json`) lacked 2148 of the catalog's jobs;
  without it the tool cannot resolve Floquet timing.

## Decisions (guan)

- The field names in the raw files are a renaming problem; a dedicated conversion block per
  data set is fine, `tools/` scripts are one-off helpers. What matters is that the converted
  files are read by the new classes. Realization indexing follows the catalog, as long as it is
  consistent.
- g and K in the headings are rough labels: g from the swap timing, K either the undriven M1
  self-Kerr or a one-trace fit. They do not block conversion; "?" where jonginn did not say.
- Promote the JSON to one YAML catalog (readability, comments) rather than keep several lists.
- The D72 off-diagonal sets stay unconverted (step 7 plan, decision 2; its questions 1-2 to
  jonginn are still open).

## What was done

- `tests/data/job_provenance.json`: exported the 2148 missing records from `jobs.db`
  (read-only, `tools/export_job_provenance.export`); all found.
- `configs/datasets/mbr_datasets.yaml`: the one catalog, 25 data sets, built by
  `tools/build_mbr_dataset_catalog.py` from jonginn's file at `91da3f7` and the JSON (read from
  commit `c3bfeda`), with the issue 6 answers applied: the 28 stale Sep 11 jobs and
  `JOB-20260830-00135` dropped, r=19 of the 7-1 campaign kept as incomplete, the four August
  realizations folded into one `august_disorder` entry, `d72_Sep10_*` renamed
  `sep10_full_K3p6_g29p2`. The script asserts every carried-over list equals the JSON's and
  that the YAML round-trips. Deleted: the JSON, `docs/job_list_and_nb_labeling/Job_list.md`,
  `docs/spectroscopy job id compilation/Only JOB IDs for Agents.md`. The human vault stays.
- `tests/mbr_reference.py` reads the YAML (`catalog()`, `dataset()`, `disorder_dataset()`);
  comments and the pole registry point at the YAML.
- `tools/convert_mbr_catalog.py`: drives `migrate_mbr_jobs` from the catalog. Its disorder
  block builds the realization record from any of the four key spellings, takes the index from
  the catalog, requires the onsite detunings to agree within a realization, and keeps what the
  files said (`file_realization`, `file_seeds`, `source_keys`). `--check` resolves files and
  timing without writing.
- Converted (pippin, `converted_data/` + `assembled_data/`, raw untouched), 12 data sets:
  `july_N1_early`, `august28_N1`, `sep10_full_K3p6_g29p2` (9 x 35 traces, 70 cal jobs),
  `september_N3`, four orthogonality sets, `august25_N3`, `august19_disorder_K44_g18`,
  `august27_disorder_K20_g15`, `august_N1_propagator`. No failures. All 20 converted manifests
  load through the new classes with the expected shapes. The Sep 10 ensemble analyzes as a
  complete basis in all 9 realizations and gives the spectral form factor for the first time
  (25-28 of 35 theory poles matched per realization, MAE 1.6-3.3 kHz, default settings).
- `analysis_notebooks/pole_finding/registry.yaml`: added `sep10_full_K3p6_g29p2` (complete,
  9 realizations) and `august25_N3` (complete, quality concern).
- Tests: the catalog readers, the disorder-ensemble module and the pole tests pass; the full
  suite result is in the status file.

## Open

- jonginn: the `K_source` column ("?" everywhere), the step 7 plan questions 1-2 (D72), and the
  log entry his status file cites (`2026-09-30_decay-notebook-readability.md`, not on his branch).
- When `job_id_sorting_out` is merged, the agents markdown comes back; delete it again.
- The 7-1 ensemble on disk still holds r0-r18; r19 (5 occupations) is in the catalog, not
  converted.

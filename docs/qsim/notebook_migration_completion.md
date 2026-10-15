Status: implemented locally, 2026-10-05; supersedes the tentative Sept 30 notebook-leftovers inventory. Hardware validation and deployment remain separate.

# Jonginn notebook migration: preserved behavior and retirements

Authority: Guan requested completing the migration from Jonginn's explanations.
[Jonginn's issue 7 answers](https://github.com/SchusterLab/multimode_expts/issues/7)
resolve the physics and deletion decisions; his later correction requires interleaved
single-shot calibration because active reset cannot be fixed in postprocessing.
`docs/job_list_and_nb_labeling/Notebook_Program_Labeling.md` remains the historical
calibration expectations list. Other calibration limitations are outside this migration.

| Item from Sept 30 inventory | Destination / decision |
|---|---|
| Q1: config versions | Notebook config blocks remain explicit dataset choices; new job HDF5s save the actual configs. No silent switch to a different hardware calibration. |
| Q2: storage table peek | Retired scratch output. |
| Q3-Q4: bare readout and calibration display | Already in Floquet calibration and MBR notebooks. |
| Q5: full-basis orthogonality | `measurement_notebooks/202609_qsim_migration/mbr.py`, section 2; smoke mode retains its short check. |
| Q6: chosen encoder/decoder pairs | Already in MBR section 4. Off-diagonal D72 still needs its separate design decisions. |
| Q7: full-basis disorder campaign | `DiagDisorderConfig` and `plan_diagonal_disorder` default to every fixed-N basis state and RMS. Explicit custom lists are supported. The measurement disorder notebook interleaves Histogram jobs. |
| Q8: theory selection / D72 | No theory-selected states in the new acquisition entry point. Explicit `state_selection='theory', normalization='norm'` only reproduces historical 7-1 plans; historical analysis remains. D72 stays dormant. |
| P1: timing | Computed from saved config with `floquet_cycle_us`; Jonginn verified it matches the old constants. No legacy hardcoded timing loaders retained. |
| P2-P3: Sept05 dark-mode cells / N2 decoder section | Retired with the old notebooks; `dormant/mbr_n2_decoder_mode.py` deleted. Other dormant dark-mode work and the historical N2 file list are retained. |
| P4: Sep10 / offline IQ refit | Sep10 was cataloged and converted Oct 2. `fitting/qsim/readout.py` preserves the two-cloud fitter; `experiments/qsim/readout_refit.py` adapts final-shot correction to current four-phase MBR jobs. Spectrum/ensemble `analyze(readout_refit=True)` is optional, raises on fit failure, and records fit provenance. Raw HDF5 is never overwritten. |
| P4: Matrix Pencil tolerances | Exploratory Sep10 tuning, not promoted to universal defaults. Fitter comparison stays in the pole-finding work. |
| P6 / R2: duplicate loaders and job lists | Replaced by `configs/datasets/mbr_datasets.yaml` and converted manifests; human plot vault retained. |
| R1: grouped debug figures | Not copied: Jonginn did not rely on their exact layout. Existing spectrum, level-match and ensemble/SFF displays are the maintained views. |
| H: high-Kerr copies | Both copies deleted; no unique logic. |

## Acquisition contract

A realization records seed, strength, normalization, direction, onsite energies,
occupations, selection mode and Kerr. `realization_spectrum` writes that record into
every TimeTrace job's config, so provenance travels with the HDF5 rather than only
the ensemble manifest or jobs.db. Estimated planning time excludes single-shot overhead.

`MBRSpectrumExperiment.acquire(before_batch=callback, batch_size=1)` completes each
batch before recalibrating and submitting the next. The single-shot runner disables
active reset for the calibration itself, then uses the shared
`experiments.readout_calibration.apply_singleshot_calibration` to update rotation,
thresholds, centers and confusion matrix. Non-finite/indistinguishable centers stop
the campaign before the next submission. Calibration HDF5 paths and queue job IDs
are saved in subsequent spectroscopy configs. Increasing the occupation interval
is an explicit notebook setting; no stability threshold has been inferred.

RMS is the convention for new acquisitions, matching Jonginn's simulation. The low-level
`disorder_direction` keeps its historical norm default for old numerical baselines;
new planning passes the convention explicitly. Past data are never rescaled.

## Retirement contract

Deleted from `measurement_notebooks/jonginn/`: `qsim_experiments.ipynb`,
`data_postprocess.ipynb`, `data_recollecting.ipynb`, and both
`qsim_experiments_highkerr_untracked*.ipynb` copies. Git history preserves the original
cells and outputs. The obsolete stage-class migration script and all notebook-test
xfail exceptions are removed; successor import/attribute checks remain.

Jonginn authorized discarding uncommitted prod scratch edits and stopping old-notebook
edits in issue 7. This local migration does not access or discard files on prod.

# 2026-10-24: `theory/` package; jonginn's AGP analysis moved into the classes (seb + Claude)

## Why

seb wanted to work with the full S7 AGP data set: compare the measured C_1(t) with theory,
calculate the AGP, change the regulator mu, and plot. No notebook in this repo did it, and
jonginn's AGP processing had no log entry. It turned out to be a Python notebook in the Julia
repo: `multimode_thermalization_sim/notebooks/jonginn/agp_storage7_mu1_to10_t1_theory_ENTIERLYCODEX.ipynb`
(commit 6b4a5bf, 10-23; "Codex generated ... It seems correct"). It read the raw job files from
hard-coded lists, did its own classification, confusion correction and Hamiltonian, and was
committed without outputs. The theory pieces kept growing in notebooks and in `fitting/`.

## Decisions (seb, 2026-10-24)

- Models get their own top-level package **`theory/`** (not `fitting/`, not a reused
  `simulation/`). Rule: `theory/` = parameters -> predictions, imports only numpy/scipy;
  `fitting/` = data -> numbers, may import `theory/`; Experiment classes call both.
  Spec: `docs/qsim/theory_package.md`. Also in `AGENTS.md`.
- The Julia repo stays out of the loop for now (cloned read-only to
  `C:\python\multimode_thermalization_sim`, not changed).
- Inline notebook logic moves into the Experiment classes; the notebook only sets parameters.
- jonginn's uncommitted edits in the `main` checkout were committed first, as on 10-22
  (c2f1e9b "jonginn nb wip"), so that the import change could merge.
- `simulation/` (RAQM-era QuTiP notebooks, nothing imported it) is deleted; the tag
  `raqm-simulation-notebooks` (local, on c2f1e9b, not pushed) marks the last commit with it.

## What was done (branch `theory`, worktree `C:\python\multimode_expts_theory`, from `main`)

- `fitting/qsim/mbr_hamiltonian.py` -> `theory/qsim/central_boson.py` (arithmetic unchanged; the
  conventions are now in its docstring). 20 import lines in 16 files, no shim; `theory` added to
  the setuptools include list.
- `theory/qsim/agp.py` (new): `probe_values`, `spectral_weights`, `two_time_correlation`,
  `spectral_agp`, from jonginn's notebook, for any probe mode (number, number^2, Kerr pairs).
- `fitting/qsim/agp.py`: `finite_time_agp` now uses `agp_kernel_weights`, jonginn's exact
  integration of the kernel against the piecewise-linear data (was a trapezoid of the product),
  with optional local T1 compensation; new `calibration_error_factors` (the shared readout-
  calibration error, delta method). The unused part (`population_moments`, `build_correlation`,
  the MPM path `fit_correlation_mpm` / `regularized_agp_from_fit`, the analysis-file I/O) is
  deleted. The raw-shot classifier `load_population_job` is kept: the converter uses it, so the
  old "used nowhere" banner was wrong. Only jonginn's
  `measurement_notebooks/jonginn/deprecated_might_be_deleted/agp_measurement_trash.py` imported
  the deleted functions; it no longer runs.
- `experiments/qsim/agp.py`:
  - `CentralBosonAGPExperiment.analyze(mu, t1_us, t1_err_us, kerr_mhz)`: `mu` may be a list
    (kHz). Adds `agp_err` (shot + calibration, the calibration file found by its recorded path or
    re-rooted on `MULTIMODE_DATA_ROOT`; a warning when not found), `agp_t1`, `agp_t1_err`,
    `C1_theory`, `agp_theory` (infinite time), `agp_theory_window` (same kernel and window as the
    data), `theory_kerr_mhz` (default: the jobs' device Kerr). One probe path for all
    directions, so the Kerr direction now has `C1_err` too.
  - `CentralBosonAGPDisorderScan`: (W, mu) maps; `theory_agp(W)` on the scan's pattern;
    `display(mu=...)` (measured, T1, same-window theory, infinite-time theory);
    `display_traces()` (C_1(t) per W with theory). **Breaking:** `agp_list` / `agp_mu_list` are
    now `agp` (W, mu) / `agp_mu` (mu,); `display()` returns `(fig, axes)`. `display()` still
    reads old files that have `agp_list`.
  - The job layer (`FockPopulationExperiment`, what the worker runs) is not changed.
- Example: `analysis_notebooks/agp/agp_disorder_scan.py` (fine and coarse S7 scans).
- Tests: `test_agp_theory.py` (new); `test_agp_kernel.py` (T1 weights against dense quadrature,
  calibration factors against a finite-difference delta method); `test_agp_experiment.py`
  (theory C_1 against a brute-force propagation on the full truncated space, spectral AGP against
  a long-window kernel integral, Kerr direction errors, T1, calibration error);
  `test_agp_disorder_scan.py`. `tests/population_jobs.py` now records couplings and a device Kerr.

## Found

- **Three AGP tests failed on `main` before this work**, not in the STATUS list: jonginn's 10-23
  changes (Kerr direction needs n_M1 >= 2; `agp_list`; the fake jobs recorded no couplings, so
  the AGP was skipped). Fixed here.
- **Parity with jonginn's notebook** (run unchanged and read-only from the scratchpad, on the
  pippin data; the coarse scan made of the 10-17 parts and the 10-23 W = 0 part, as in his
  notebook): max relative difference over W, mu = 1..10 kHz, measured and T1: `agp` 5e-13,
  `agp_err` 1e-13, `agp_t1` 8e-13, `agp_theory` 1.3e-13, `agp_t1_err` 4e-9 (finite-difference
  step).
- **The window dominates at small mu.** Closed-system theory, same kernel: the ratio
  same-window / infinite-time AGP, at W = 0:

  | mu (kHz) | 1 | 2 | 3 | 5 | 10 |
  |---|---|---|---|---|---|
  | fine (T = 122 us) | 463 | 109 | 35 | 5.5 | 1.06 |
  | coarse (T = 244 us) | 431 | 48 | 7.9 | 1.2 | 0.99 |

  At W = 150 kHz the fine ratio is 14 even at 10 kHz. C_1 is not connected: its long-time
  constant part enters the truncated integral, with weight T e^{-mu T} / (2 mu). The default
  mu = g L/D is 2.9 kHz here, so the default `agp` of the saved scans is mostly this term. This
  is arithmetic on the closed-system model, not a statement about the device.
- Measured / closed-system-same-window at mu = 10 kHz: fine 1.4-9, coarse 1.4-122, growing with
  W. The measured C_1 falls below the closed system after about 50-100 cycles at every W
  (`display_traces`). Not analyzed; no mechanism claimed.
- A new top-level package needs `pixi install` in each checkout (the editable finder maps a fixed
  package list). In the `main` checkout one `pixi run pytest` once failed with
  `No module named 'qick'`, then passed; it looked like pixi refreshing the environment that the
  worker also uses. After that, tests ran only in the worktree.

## Open

- Merge `theory` into `main` (seb's go-ahead), then `pixi install` in the `main` checkout and
  restart kernels. `experiments/__init__` imports every qsim module, and two of them now import
  `theory`, so a process that cannot import `theory` cannot load any job class. The running
  worker (started 10-23 by `job_server/start_worker.sh`: `cd` to the repo root, then
  `python -m`) has the repo root on `sys.path`, so it finds `theory/` without a restart.
  `job_server/worker.py` now also clears `theory.*` before each job, as it does `fitting.*`;
  that line takes effect at the next worker restart (until then `theory` code stays cached
  after its first import).
- jonginn: see `docs/STATUS.md`.
- guan: `WalkModel` in `measurement_notebooks/guan/kerr_n123_consistency.py` is the next
  candidate for `theory/` (open-system walks).
- Push the tag `raqm-simulation-notebooks` when the branch goes to origin.

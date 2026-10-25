# The `theory/` package: models apart from fitters

Status: current, 2026-10-24 (seb + Claude). Replaces nothing; it gives a home to model code that
was in `fitting/qsim/mbr_hamiltonian.py`, in notebooks, and in jonginn's notebook in the
`multimode_thermalization_sim` repo. Record: `docs/log/2026-10-24_theory-package-agp.md`.

## Why

`fitting/` holds estimators: measured arrays in, numbers out. Models started to grow there and in
notebooks: on 2026-10-24 there were five separate central-boson Hamiltonians (`fixed_n_hamiltonian`,
jonginn's inline one, the Julia `CentralBoson`, `central_boson_qutip_tools.py`, guan's Lindblad
walk model), with different units and sign conventions. One home, with its conventions written
once, stops that.

## The rule

| Package | Holds | Imports |
|---|---|---|
| `theory/` | models: parameters in, predictions out (Hamiltonians, closed- and open-system dynamics, correlators, spectral sums) | numpy, scipy only |
| `fitting/` | estimators: data in, parameters or numbers out (fits, kernel integrals, error propagation) | `theory/` |
| `experiments/` | jobs and assembled data sets; `analyze()` and `display()` call both | `fitting/`, `theory/` |

- A notebook sets parameters and calls an Experiment class. Logic that a notebook holds inline
  moves into the class (data flow, plots) or into `fitting/` / `theory/` (numerics).
- No compatibility shims when code moves: update the imports in one change.
- The theory never changes a measured value; `analyze()` stores it beside the data
  (`*_theory` keys).
- Large numerical studies (thermalization, level statistics at large N, Lyapunov) stay in the
  Julia repo `multimode_thermalization_sim`. It is not called from here; exchange is by files.

## Conventions

Stated in `theory/qsim/central_boson.py` and kept by every module:

- Cyclic MHz for energies, couplings and cutoffs (H/h); time in us. Experiment-level cutoffs
  `mu` are cyclic kHz.
- Onsite storage energy = minus the detuning the pulse program applies to the storage-M1 sideband.
- Kerr term (K/2) n(n-1), K the physical Kerr with the device sign.
- Basis: `itertools.product` over (M1, *storages), lexicographic, one fixed photon number.

## Contents (2026-10-24)

- `theory/qsim/central_boson.py`: `fixed_n_hamiltonian` (moved from `fitting/qsim/mbr_hamiltonian.py`,
  arithmetic unchanged).
- `theory/qsim/agp.py`: `probe_values`, `spectral_weights`, `two_time_correlation`,
  `spectral_agp` (closed system); `split_dc`, `damped_correlation`, `damped_agp` (a toy
  open system: each term only decays); `window_agp` (the kernel integral over a finite window,
  exact, no grid; 2026-10-25).
- `theory/qsim/open_system.py` (2026-10-25): `OpenCentralBoson`, the Lindblad central boson on
  all states with at most N photons (loss and dephasing per mode), for Fock or general starts;
  `sector_correlation` gives the open-system C(t). From guan's `WalkModel`. Used and checked
  against the data in the AGP lab book (`analysis_notebooks/agp/labbook/`).
- `fitting/qsim/agp.py`: `agp_kernel_weights` (exact for piecewise-linear data, optional local
  T1 compensation), `finite_time_agp`, `calibration_error_factors`, `fit_exponential_background`;
  and the raw-shot classifier
  that `tools/convert_fock_population_catalog.py` uses as a cross-check.
- `experiments/qsim/agp.py`: `CentralBosonAGPExperiment.analyze(mu, t1_us, t1_err_us, kerr_mhz)`
  and `CentralBosonAGPDisorderScan` (maps over W and mu, `theory_agp`, `display`, `display_traces`).
- Examples: `analysis_notebooks/agp/agp_disorder_scan.py`; the DC-background study
  `analysis_notebooks/agp/agp_dc_background.py`.

## Next candidates

Not moved yet; each moves when someone next works on it:

- guan's notebook still has its own `WalkModel` (`measurement_notebooks/guan/kerr_n123_consistency.py`);
  `theory.qsim.open_system.OpenCentralBoson` does the same and more (rates per mode). Switch the
  notebook when guan next works on it.
- The model parts of `fitting/qsim/poles/synthetic.py` and `fitting/qsim/mbr_disorder.py`
  (disorder realizations), if they are used without a fit.

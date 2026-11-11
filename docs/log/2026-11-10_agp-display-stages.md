# AGP display stages

guan with Claude, branch `theory`.

## What changed

- New `experiments/qsim/agp_display.py`: `AGPDisplayMixin`, used by
  `CentralBosonAGPExperiment` and `CentralBosonAGPDisorderScan`, so also by the two sequential
  classes. One method per processing stage, the same on all four classes:
  1. `display_populations(raw)`: <i|n_r(t)|i> maps (needs the parts on a scan).
  2. `display_correlations(raw, theory, style="traces"|"map", moment)`.
  3. `display_spectrum(mu, raw, theory, window, background_start_us, max_khz, yscale)`.
  4. `display_agp_mu(estimator, theory)`; the default estimator is `agp_connected` when it
     was analyzed, else `agp`.
  5. scan only: `display_agp(mu, theory)`, the AGP against W.
- `display(raw=False, theory=True, mu=None)` draws all stages that have data, in this order,
  and returns the list of figures. A stage without data raises `MissingStageData`;
  `display()` prints one line and skips it. Each stage method returns its `Figure` (before:
  some returned `(fig, axes)`).
- The panels are a grid of W (rows) by direction (columns). `_grid(key)` gives each array
  these two leading axes, so one code path draws the single-direction and the sequential
  classes. The four sequential display overrides are deleted; the sequential classes now also
  show `agp_t1`, `agp_connected`, the spectrum and the binned closed-system weights.
- `theory_spectrum` and `theory_agp` take a `direction` (default as before).
- The scan data has `direction` (single-direction classes), for the labels of arrays-only
  views.
- Removed without aliases: `display_traces`, `display_correlation`, `display_mu`, and the
  old two-panel part `display()`. Callers in notebooks and tests are changed (also jonginn's
  `agp_measurement.py`, `agp_sequential_measurement.py`, `agp_reanalysis.py`).

## Decisions (guan)

- No `stages=` argument and no kwargs routing in `display()`: that would be a small
  switchboard again (the MBR refactor removed `analyze(stage=...)`). One stage = call its
  method; everything = `display()`.
- The processing steps inside one W stay methods, not classes: a new class only when the
  children or the acquisition change.
- `display` does not call `analyze()`. Before, it did with default arguments, which could
  overwrite an analysis and failed on arrays-only views.
- `display()` is runner-compatible (no required arguments, a list of figures). A per-W
  display inside `scan.acquire` is not added.

## Open

- `CentralBosonAGPSequentialExperiment._analyze_direction` is a copy of the base `analyze`
  body. Split the base into step methods and remove the copy (agreed as the next commit).

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

## Later the same day

- `analyze` is split into step methods (commit `e239ccd`): `analyze_correlations()` (populations
  and C_k, no settings), then `_direction_curves` per direction. The sequential class overrides
  only `_collect_populations`, `_read_directions`, `_correlation_inputs` and `_store`; its copy
  `_analyze_direction` is deleted. The scan has `analyze_correlations()` too.
- Stages after guan's review: populations, `display_correlations` (C_1 and C_2 as measured, no
  theory; maps on a scan, lines for one W; separate scales; "no C_2" for Kerr), `display_traces`
  (only C_1, the AGP input, with the closed system and, when analyzed with
  `background_start_us`, the fitted decay A e^{-rate t}), spectrum, AGP(mu), AGP(W).
  `display_correlations` has no `style=`/`moment=` any more.
- `display()` on an experiment that was not analyzed runs `analyze_correlations()` and shows
  what that data allows (guan: the raw-ish layer must work without an analysis).
- Smaller figures and text (`SMALL_TEXT`). Default y scales: S(f) `symlog`, linear within the
  median shot error of S (the DC peak at f = 0 is 1000x the median of S, and S is negative by
  noise); AGP(mu) `log` (values 1e-3 to 50 on the S5 scan); AGP(W) linear, `yscale=` to change.
- The "modeled decay" is the exponential fit of `subtract_background`, not the Lindblad model of
  the lab book (`theory.qsim.open_system`); guan's correction.

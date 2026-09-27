# MBR redesign, step 9 plan: retire `experiments/qsim/notebook_helpers/`

Status: approved by guan 2026-09-26. 9A in progress. It uses `mbr_redesign.md` for the rules and
patterns, and continues `mbr_step8_plan.md`.

## 0. Decisions (guan, 2026-09-26)

1. `run_mode` and `defaults` stay in `notebook_helpers/`. They are scaffolding for testing this
   refactor. If they are promoted later, first decide whether they belong to `experiments/qsim/`
   or to top-level `experiments/` (which other projects use too).
2. The Floquet and multiphoton calibration notebooks are in scope. They are prerequisites of the
   MBR experiments and have the same problems with the same cause. The target is the pattern of
   the single-qubit autocalibrate notebooks: defaults -> pre/post hooks -> runner -> `execute`
   with kwarg overrides, visible in the notebook. Short hooks stay in the notebook. Fits go into
   `analyze` or into numerical modules.
3. The Floquet pulse is the preloaded flat-top. The synthesized 3-segment flat-top and the
   Gaussian procedures go to `dormant/` (notebook level). The pulse layer keeps its `gauss` and
   `flat_top` support for now (August data and the `august_n3` ASM golden are Gaussian).
4. `fitgaussian(periodic=True)` in `ErrorAmplificationExperiment.analyze` biases the center of a
   frequency or gain scan when the peak is near a scan edge (a synthetic 1 MHz scan: 30 kHz from
   the edge -> 26 kHz error). This is shared infra: fix it on `main`, then cherry-pick. In this
   round only an optional `periodic` keyword is added, with the default unchanged.

## 1. Steps

| Step | What |
|---|---|
| 9A | MBR helpers into the classes: `mbr_campaign` (merged into `experiments/qsim/mbr_campaign.py`, on `mbr_defaults`), `mbr_disorder_campaign` (into `mbr_disorder_ensemble.py`), `mbr_tomography` (`MBRHamTomoExperiment.plan_three_depth`, `analyze_shared_step`, `display_shared_step`; `fit_shared_step` in `fitting/qsim/mbr_propagator.py`), `mbr_n3_reprocess` (`MBRSpectrumExperiment.fit_self_kerr`, `display_peak_finders`) |
| 9B | `mbr_saved_reanalysis`: the loaders become notebook cells (canonical flow); their checks go to the classes; the coherent-trace FFT goes to `fitting/qsim/mbr_spectrum` as a display option; the trace panels merge with `display_occupation` |
| 9C | `multiphoton_calibration`: hooks inline; fits into `analyze`; sweep grids into preprocs |
| 9D | `floquet_calibration`, `floquet_bare_readout`: the same, and the notebook consolidated on the preloaded flat-top |

Nets: the ASM golden and mock acquisition; the dry runs of the measurement notebooks; the
analysis suite; for 9C and 9D the config database (the API tunnel or a copy of `jobs.db` made
on pippin; not the live `jobs.db` over SMB).

## 2. 9A: facts

- `build_campaign` on `mbr_defaults`: the final job configs differ from before only in
  `floquet_waveform` (no longer set) and `include_10cycles_buffer*` (now set). No MBR program
  reads them. `floquet_waveform` is read by the offline timing resolver for jobs without
  recorded timing; new jobs record it.
- `MBRHamTomoExperiment.analyze_shared_step` uses the parts' own Floquet cycle time; the old
  helper used the calibration set's. They are the same for jobs taken under one config.
- The N=1 theory of the shared-step display is now written for any number of modes
  (+/- |g| and zeros); the old code wrote it for 4 storage modes.

## 3. 9B: facts

- The four `load_*` functions of `mbr_saved_reanalysis` were `from_manifest` + `analyze(...)` +
  prints. They are notebook cells now. Their checks stay visible there as asserts (complete
  N=3 sector; Floquet timing recovered from the files, not from the station).
- The recorded-theory check is `MBRDisorderEnsembleExperiment.recorded_theory_mismatch_MHz()`;
  the notebook cell that asserts it keeps the `raises-exception` tag (the known 0.3 kHz
  difference).
- `coherent_normalized_trace_spectrum` -> `fitting.qsim.mbr_spectrum.coherent_trace_spectrum`
  (same arithmetic; the window table is now the module's `FFT_WINDOWS`), shown by
  `MBRSpectrumExperiment.display_coherent_trace`. `display_result` takes the aggregate panel's
  title and labels as arguments, so the old report no longer finds an axis by its title.
- `plot_occupation_trace_panels` was a weaker copy of `MBRSpectrumExperiment.display_occupation`
  (its "theory (scaled)" curve was not scaled; its `realization_idx` argument was unused).
  Removed; the notebook calls `display_occupation`.

# 2026-10-25: AGP from the time traces, overnight lab book (Claude, unattended; asked by the user on 10-24)

## Why

The user (10-24 evening) left the AGP question for a night: from the vault note
`G:\Shared drives\SLab\Multimode\Lab\guan\AGP\2026-10-24 analytical AGP from time trace.md`, choose
well-defined questions, work them as a physicist, and keep a lab book in the Obsidian vault (math,
figures, room for comments) instead of long logs. The three open choices in the note: the time
window, the cutoff mu, and what to do with the decay.

**The lab book is the main record:**
`G:\Shared drives\SLab\Multimode\Lab\guan\AGP\2026-10-25 lab book - AGP from time traces.md`
(figures in `labbook_figures\`). This log is only the short version.

## What was done (branch `theory`)

- `theory/qsim/agp.py`: `window_agp`, the exact kernel integral over a finite window (no grid).
- `theory/qsim/open_system.py` (new): `OpenCentralBoson`, the Lindblad central boson on all states
  with at most N photons, loss and dephasing per mode, Fock or density-matrix starts;
  `sector_correlation`. From guan's `WalkModel`; checked against the closed system to 6e-15.
- Tests: `tests/test_agp_theory.py` (window_agp), `tests/test_open_system.py` (new).
- Scripts, one per question: `analysis_notebooks/agp/labbook/q1_sampling.py` ... `q12_mu_curves.py`,
  shared `load_scans.py`, `open_models.py` (the damping models and the W = 0 residual detunings).
  They write the figures to the vault. Model traces are cached in `labbook/cache/` (git-ignored).
- No change to `analyze()`, the experiment classes, or anything on `main`.

## Found (labels as in the lab book: measured / model / model vs measured)

- **The decay of C_1 is M1 loss + M1 dephasing** at their measured values (T1 215 us, T2* 134 us;
  storage T1 from the 10-21 report and jonginn's S7 value), with no free parameter: chi2/point from
  150-650 (closed) to 3-40, and the envelope against W, including its non-monotonic steps.
  Two misses: W = 0, and extra damping at W >= 90 kHz that grows with detuning.
- **W = 0:** static residual detunings of 2-4 kHz lift the degeneracy of the dark-mode states and
  give the fast W = 0 decay (chi2/point 108 -> 3.6, both S7 W = 0 runs; storage dephasing fits worse).
- **The plain kernel integral is fully accounted for** by the open model (within about 10 % at
  mu = 2-40 kHz, W <= 90). Its excess over the closed-system AGP is the decaying DC term,
  ~ (2 pi g)^2 A gamma / (2 mu^3); a longer window does not remove it.
- **Estimators** (on the open model, which reproduces what they do on the data): exp fit = envelope
  + tail; residual about 0.13 x the DC term. The residual comes from slow lines (omega/2pi <~ 1/T)
  that the background fit cannot tell from a decaying constant; two exponentials do not help.
- **The measured AGP agrees with the closed-system model** where resolvable: Kerr probe within 8 % at
  all W (mu = 10, 20 kHz) and its whole mu-curve above mu_min; S7 at W <= 60 kHz. mu_min: Kerr
  5-10 kHz at all W; S7 6 / 12 / 25 kHz at W <= 30 / 60 / 90; none at W >= 120; 15 kHz at W = 0.
- The resolvable mu is about one level spacing of the 35-state sector; the S7 AGP there peaks at
  W ~ 25-30 kHz, at the knee of the participation ratio. The DC fraction of C_1 is itself a robust
  measured localization signal.
- **Large-W extra damping (Q9):** in the 10-17 S7 data it fits a storage dephasing proportional to
  delta_m^2 (one constant per run), but the constant differs 3x between the fine and coarse runs,
  and the 10-24 Kerr run wants no storage dephasing at all. Likely a property of the 10-17
  configuration, not of the scheme; test with a short S7 scan at W = 90-150 on the current config.
- **Kerr W = 0 run (Q8):** small dark-manifold effect, as predicted; detunings of RMS 4 kHz still
  preferred; fitted M1 dephasing 0.0107 /us and T1 192 us agree with the measured 0.0106 /us and
  215 us. Its oscillations live longer than any Markovian rate allows (slow M1 noise? not tested).
- **Background limit (Q10, exact DC split of the open model):** the true decaying DC is near one
  exponential; the error is the fit to the full trace, pulled by lines below ~1/T. Two exponentials
  do not help. Scan the fit start as a systematic.
- Smaller: the short-time curvature (exact sum rule, W- and K-independent) gives
  g_eff/g_saved = 0.97 +/- 0.05 (Kerr traces). The 5-cycle grid is fine for mu <= 10 kHz.
  The connected correlator (all 35 starts) gains little at equal measurement time.

## Decisions / open (for the user)

- No estimator is in `analyze()` yet. Candidate: an exp-fit background option with a fit-start scan
  and 0.13 x DC term as a systematic error.
- Suggested, not done: use the Kerr direction for AGP-vs-W; check the kHz-level residual
  detunings at W = 0; a dedicated test of the large-W extra damping.
- The 10-24 Kerr-direction scan was still running during the night (a W = 180 kHz part appeared);
  the scripts read whatever complete W values are on disk.
- Merge `theory` into `main` is still the user's go-ahead (see `2026-10-24_theory-package-agp.md`).

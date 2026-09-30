# 2026-09-28: pole finding, design

Session record; not edited after this date. The current state is `docs/STATUS.md`. The design
is `docs/qsim/pole_finding.md` (draft, reviewed by guan). The work continues on pippin, in the
`guan` worktree `C:\python\multimode_expts_guan`, because the real-data runs are large.

## Decisions (guan)

- Pole finding is treated as a benchmark of fitters, not as tuning of the current code. The
  current `fitting/qsim/matrix_pencil.py` stays frozen as fitter A (the reference); new
  fitters are written from the spec, one module each, under `fitting/qsim/poles/`.
- Four benchmarks: ideal synthetic, synthetic with non-idealities, real-data self-consistency,
  real data against the model (validation only; do not tune on it).
- The code must read top-down like a methods section (spec section 9); the entry point is a
  report notebook executed to HTML.
- The score is not `<r>`: chaos is not fully developed here, and `<r>` does not follow the
  textbook limits. Use the small-spacing statistic against the model's own prediction, and
  the Michaille-Pique ratio `d = Lambda_eff / Delta'` (PRL 82, 2083 (1999), in guan's Zotero
  library) as a flag that needs no ground truth.
- Partial-basis sets (10 of 35 rows) cannot be held to integer weights; only `w <= m`.
- Fitter C: no fixed gauge. The Stark calibration fixes the frame: each group offset has a
  zero-mean prior of width `sigma_cal`.
- Outputs go to `<experiment>/derived_data/pole_finding/<run>/` on the data tree, plus an
  Obsidian note.
- guan converts jonginn's free-form logs into manifests and registry entries (spec 8.2).

## Found

A 30-line joint pencil (all rows stacked into one Hankel matrix, one SVD, one eigenproblem;
spec 4.1) passes every synthetic case the current code fails: 4 poles without noise exact
(residual 8e-15); 4 poles at 1% noise with rank 4 found by the singular-value rule, within
0.003 bin; a pair 1 bin apart at -0.002 / 1.003 bin; a pair 0.5 bin apart in 20 of 20 seeds
(the current code: 0 of 20). Synthetic setup as in `tests/test_matrix_pencil_synthetic.py`
(64 points, 1 us, rows normalized to `A(0) = 1`). Not yet tried on real data, and not with
row-to-row offsets, which is where the joint fit is expected to be weakest.

## Next

Phase 1 of the spec, on pippin: `PoleFit`; the model-based generator; fitters A (wrapper), B,
E; benchmarks 1 and 2 on small grids; the report skeleton.

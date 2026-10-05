# 2026-10-05: the three views per fitter, and the lab vault as the reading interface

guan and Claude, on pippin. guan read the recap (`docs/qsim/pole_finding_recap.md`) and asked
for a figure-first way of working: read plots in Obsidian and comment inline, instead of long
text and number tables in the terminal.

## Decisions (guan)

- Judge a fitter by three pictures, not by scalar scores alone: (1) per-row FFT heat map with
  the poles on it (are the poles on the peaks?); (2) the stick diagram after the row sum
  against the model's; (3) the gap-ratio histogram against the model's. The scalar scores stay
  as targets for optimizers. The prize is the r figure, and its whole shape counts, not only
  P(r < 0.25): every method has a finite resolution, so the first bin always goes to 0.
- The spectral form factor is a physics goal of its own (fitter-free), not only a check.
- The big question: with our T2 and acquisition speed, is it worth going deep into the K > 0,
  delta > 0 bulk of the phase diagram (many realizations, long averaging)? The low-Kerr and the
  zero-disorder limits are expected to work.

## Done

- `fitting/qsim/poles/views.py`: the three views for any data set and fitters
  (`display_row_heatmaps`, `display_sticks`, `display_gap_ratio_histograms`,
  `display_gap_ratio_cdfs`), fixed color per fitter; `tests/test_pole_views.py`.
- `analysis_notebooks/pole_finding/views_sep10.py`: the views on the Sep 10 ensemble from the
  survey's cached fits, the model's r distribution from 1000 draws at the same point, the form
  factor. It writes the page and figures to the lab vault:
  `G:\Shared drives\SLab\Multimode\Lab\guan\qsim_analysis\pole_finding\`
  (`2026-10-05_sep10_views.md`, and a copy of the recap). Comment convention: a `> [!guan]`
  callout anywhere in that folder.
- Tests: the two slowest pole tests (C under row offsets, 51 s; the free-amplitude Hamiltonian
  recovery, 9 s) carry the existing `slow` marker. Full suite before: 223 s, of which the pole
  tests about 85 s. The survey and the views are notebooks, not tests.

## Found

- **At the Sep 10 point the model is close to Poisson**: <r> 0.415 from 1000 draws (Poisson
  0.386, GOE 0.536). A, B and C give <r> 0.50, 0.57, 0.54 and almost no ratios below 0.1: they
  merge the small gaps and would show level repulsion that the model does not have. T1T3 gives
  0.414 and the model's shape above r 0.1; its first bin is low (the resolution). T1 (a model fit)
  0.439.
- The fixed synthetic set ("T0") is synthetic: model points with the parameters of the August
  disorder set (jonginn's set 2) and the Sep 10 set (set 4), at
  `C:\experiments\260818_qsim_spectroscopy\derived_data\pole_finding\fixed_set\set.h5`. Its
  August Kerr is K/g -1.22 (K -10.5 kHz, the recorded manual Kerr); the catalog label of set 2
  says 5.7 kHz, and T1 fits -8.6 to -9.3 kHz. Open with jonginn (`K_source`).

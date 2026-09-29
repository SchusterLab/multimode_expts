# Pole finding for MBR spectra: method and benchmark

Status: draft, 2026-09-28; phases 1 and 2 implemented (`fitting/qsim/poles/`); B is the lead fitter; C is needed (section 6). Replaces nothing yet. When a fitter is chosen, this file
becomes the spec of `fitting/qsim/poles/`, and `fitting/qsim/matrix_pencil.py` (fitter A) is
retired. History and reasons: `docs/log/2026-09-27_matrix-pencil.md`.

This file is written at the level of a paper's methods section. The code follows its
structure: each numbered step names the function that does it, and a reader who knows this
file can read the code top-down (section 9).

## 1. The problem

An MBR data set gives, for each measured occupation row `b`, the complex return
`A_b(t_n)` on a uniform grid `t_n = n dt`, `n = 0 .. N-1`. The model is

    A_b(t) = sum_lambda c_{b,lambda} exp((-gamma_lambda - 2 pi i E_lambda) t)

with the same poles `(E_lambda, gamma_lambda)` for all rows. For a diagonal row the
amplitude, normalized to `A_b(0)`, is `c_{b,lambda} = <b|P_lambda|b>`, where `P_lambda`
projects on the eigenspace of `E_lambda`. The task is to find the poles and the amplitudes.
The frequencies are only known modulo `1/dt` (principal alias).

The goal is the level statistics: in particular, whether levels repel. So a fitter is judged
on two things: how accurately it finds the poles, and how much bias it puts into the
small-gap-ratio statistics.

## 2. Two failure modes, and why they matter

- **Merging.** Two levels closer than the resolution come out as one pole, near their
  weighted mean. Small spacings disappear, so the spectrum looks **more repelling** than it is.
  Michaille and Pique [PRL 82, 2083 (1999)] show that an uncorrelated (Poisson) spectrum looks
  Wigner-like when `d = Lambda / Delta'` reaches about 0.5. Here `Lambda` is the resolution and
  `Delta'` the mean spacing of the *found* levels.
- **False poles.** Noise or a model error gives poles that are not levels. The false poles
  are placed at random, so the spectrum looks **more Poisson** than it is (Berry-Robnik).

A claim of repulsion is exposed to the first mode; a claim of no repulsion to the second.
Each benchmark measures both.

## 3. Data sets

- **Complete basis:** all `D` occupations of the sector are measured (`D = 35` for N=3 on
  5 modes). Then `sum_b <b|P_lambda|b> = tr P_lambda = m_lambda`, the multiplicity: each
  pole has an integer weight, and the weights add up to `D`.
- **Partial basis:** only selected rows are measured (10 of 35 in the disorder campaigns),
  chosen so that every eigenstate has enough support in them. The pole weight
  `w_lambda = sum_{b measured} <b|P_lambda|b>` is then not an integer. The only bound that
  holds without a model is `0 <= w_lambda <= m_lambda`.

The data sets are listed in a checked-in registry (section 8.2). Each entry says complete or
partial.

## 4. Fitters

Every fitter is a pure function with the same interface:

    fit(A, time_us, settings, row_groups=None) -> PoleFit

`PoleFit` (frozen dataclass): `frequencies_MHz`, `decays_per_us`, `amplitudes` (row x pole,
normalized to `A_b(0)` for diagonal rows), `rank`, and optionally `frequency_errors_MHz`.
`row_groups` labels the rows that share a calibration (same final occupation).
Each fitter has its own pydantic settings class. Its fields are the fitter's free parameters,
and each field's docstring says which trade-off it controls.

| | Fitter | Method | Free parameters |
|---|---|---|---|
| A | per-row pencil + reconciliation | current `fitting/qsim/matrix_pencil.py`, frozen | 4 tolerances, persistence length, rank cap |
| B | joint pencil | one stacked Hankel matrix for all rows | rank rule, pencil length |
| C | joint pencil + refinement | B, then nonlinear least squares with one frequency offset per row group | B's, plus the offset prior |
| D | per-row pencil + clustering | rank per row from a singular-value rule; poles clustered once across rows | rank rule, merge tolerance |
| E | FFT peaks | peaks of the windowed FFT of the summed trace | window, peak threshold |

A and E are the references: A is what the results so far are based on, and E is what the
resolution was before MPM.

### 4.1 Fitter B: joint pencil (`fitting/qsim/poles/joint_pencil.py`)

1. Normalize the diagonal rows to their first sample (`normalize_to_initial_return`).
2. Stack the Hankel matrices of all rows, `H = [H_1; ...; H_R]`, each `(N-L) x (L+1)`. The
   rows share the right factor `z^k`, so the columns keep the shift invariance
   (`stacked_hankel_pair` -> `H0 = H[:, :-1]`, `H1 = H[:, 1:]`).
3. Choose the rank `M` from the singular values of `H0` (`choose_rank`; rule in 4.6).
4. The poles are the eigenvalues of `S_M^-1 U_M^h H1 V_M` (`shift_invariance_poles`).
5. With the poles fixed, the amplitudes of each row by linear least squares
   (`least_squares_amplitudes`).

### 4.2 Fitter C: joint + refinement (`joint_refined.py`)

1-5 as B. Then:

6. Refine all frequencies, decays and amplitudes together, plus one frequency offset
   `delta_g` per row group, by nonlinear least squares on `A` (`refine_with_group_offsets`).
   Each offset has a Gaussian prior of mean 0 and width `sigma_cal` of its group (section 6):
   the Stark calibration says the offset is 0 within its error.
7. Frequency errors from the Jacobian at the solution (`frequency_errors`).

No offset is fixed to 0. The data alone cannot tell a shift of all poles by `+c` from a shift
of all offsets by `-c`; the prior breaks this, and puts the absolute frame at the calibration's
own error, `sigma_cal / sqrt(G)` for `G` groups. Fixing one offset would treat that group's
calibration as exact and move its error to all the others. The spacings, and so the level
statistics, do not depend on the frame; only benchmark 4 sees it.

In a diagonal data set each row is its own group (its final occupation). A row's offset is
fixed by the poles it shares with other rows; for a pole seen in one row only, the prior
decides.

### 4.3 Fitter D: per-row + clustering (`per_row_clustered.py`)

1. As B.1.
2. Per row: pencil, rank by the rule of 4.6, poles (`row_poles`).
3. Cluster all row poles by frequency with one tolerance, one pole per row per cluster
   (`cluster_row_poles`); a cluster is a pole at its weighted mean frequency.
4. As B.5.

### 4.4 Fitter A: current (`per_row_reconciled.py`)

A thin wrapper around `fitting.qsim.matrix_pencil.analyze_matrix_pencil`, output converted to
`PoleFit`. Not changed. Its known defects are in the 2026-09-27 log.

### 4.5 Fitter E: FFT peaks (`fft_peaks.py`)

1. The windowed FFT of `sum_b A_b(t) / A_b(0)` (`windowed_fft`, shared with the spectrum).
2. Peaks above a threshold (`find_peaks`); amplitudes as B.5.

### 4.6 Rank rule (shared by B, C, D)

The rank sets the number of poles, so it is the main free parameter. Candidates:
the count of singular values above `k x median` (the current rule; `k = 2.858`), and an
information criterion (MDL). Benchmark 2 chooses. The rule is one function, `choose_rank`,
used by every fitter that needs one. Both rules are capped by a numerical floor (singular
values below `1e-12` of the largest are zero). Phase 1 default: MDL, pencil length `L = 2N/3`
(better than `N/2` on clean data, benchmark 1). The threshold rule undercounts when the
signal fills more than half the singular values, because the median is then a signal value.

## 5. Benchmarks

The synthetic data come from our own model, so they have the real degeneracy structure,
aliasing and weights, with exact answers.

**Generator** (`fitting/qsim/poles/synthetic.py`, `synthetic_returns`): `fixed_n_hamiltonian`
with given detunings, couplings and Kerr -> energies and eigenstates -> `A_b(t)` on the data
set's own time grid and rows -> then, if asked: decay per level, complex white noise at a
given SNR, one frequency offset per row group (Gaussian, width `sigma`). Small detuning
differences split the permutation multiplets by a chosen fraction of an FFT bin; Kerr on the
central mode shifts levels but keeps the symmetry.

**Phase-diagram sampling** (`sample_phase_diagram`). The ideal model is cheap (`D = 35`), so
the synthetic cases cover the whole plane of Kerr `K / g` and disorder strength `delta / g`
(`g` the coupling), several disorder draws per point. A draw is as in the disorder campaigns
(`fitting.qsim.mbr_disorder.disorder_direction`): a random zero-mean unit vector `u` over
the storage modes (`sum_i u_i = 0`, `sum_i u_i^2 = 1`), and onsite energies `delta_i = delta u_i`.
The plane holds the regimes that stress a fitter
in different ways: `K = 0` is linear (Poisson-like levels, but the bosonic ladder gives
equal spacings); `delta = 0` is highly degenerate; the bulk begins to show level repulsion, not
fully developed at our low photon number. The pair-separation cases above are extra, on top
of this plane.

**Decay.** One rate for all levels to start, near `1 / (100 us)` (T2 of order 100 us); the
real data may later give a better value or a per-level spread.

**Matching** (`match_poles`). One-to-one Hungarian assignment of found poles to the
*distinct* true levels (levels closer than 1e-3 bin are one level with the summed
multiplicity), with frequencies wrapped to the principal alias first. Each level has its
own tolerance, `min(0.25 bin, 1/4 of its nearest separation)` (`level_tolerances`): with a
fixed 0.25 bin, close to the mean spacing of a dense spectrum, 35 random poles over the span
"resolve" 10-20% of the levels closer than a bin; with the cap, 1-8%. The same method as `fitting.qsim.mbr_disorder.match_levels`, which does not wrap
aliases or group degenerate levels. A level is *resolved* when it and its nearest neighbour
are both matched (`resolved_levels`): a merged pair then counts as two unresolved levels. `display_match` draws found poles against true
levels, matches joined (after `MBRDisorderEnsembleExperiment.display_levels`).

### 5.1 Benchmark 1: ideal data (`run_ideal_bench`)

No noise, no offsets, over the phase-diagram sample. Pass/fail, runs as a pytest
(`tests/test_pole_bench_ideal.py`). B (and later C, D) must find every distinct level to 1e-6
of a bin and the weights to 1e-6 (`ideal_pass`), **where the time grid can tell the levels
apart**: the conditioning `s_min / s_max` of the Vandermonde matrix of the true levels on the
grid (`vandermonde_conditioning`) is at least 1e-2. Below that no fitter reaches 1e-6 in double
precision. On the measured grid (100 samples) almost no point of the plane is that well
conditioned (35 levels in about 25 bins), so the pytest runs the plane on 400 samples: it
checks the code, not the experiment. A and E are scored, not held to it: A keeps its known
defects as strict xfails (`tests/test_matrix_pencil_synthetic.py`); E is held to 0.1 bin on
separated levels (`tests/test_poles.py`).

### 5.2 Benchmark 2: data with non-idealities (`run_nonideal_bench`)

Grid over window length, SNR, decay and `sigma`, over the phase-diagram sample (which holds
the pair separations); many noise seeds per point. Two questions, kept apart (guan,
2026-09-28):

- **2a. How good is each method?** Window lengths of 100, 200 and 400 samples at the same
  `dt`, with no decay and with the decay of T2 = 100 us. The known answer is exact, so a
  fitter is judged on what it can resolve when the data allow it. In the experiment the
  traces can run long, but past about T2 they are flat, so the decay caps what a longer
  window adds.
- **2b. What does the measured grid allow?** 100 samples and decay 0.01 per us; SNR and
  `sigma` swept. This gives `Lambda_eff` for benchmark 3.

Separations are in bins of the measured grid (12.2 kHz) for every window length. Per fitter
and grid point:

- resolution probability of a pair against separation; `Lambda_eff` = the separation resolved
  in half the seeds;
- false-pole rate (poles with no level within the tolerance of the match);
- bias of frequencies and weights;
- the small-gap-ratio statistic (section 7) of the found levels, against the true one.

It chooses the rank rule and each fitter's settings, and it gives `Lambda_eff` as a function
of the conditions for benchmark 3. It also answers a question about the experiment: the
`sigma` above which the small-gap-ratio statistic cannot tell repulsion from no repulsion is a
requirement on the calibration.

### 5.3 Benchmark 3: real data, no model (`run_self_consistency_bench`)

Per data set and fitter:

- `d = Lambda_eff / Delta'`, with `Lambda_eff` from benchmark 2 at this data set's SNR, decay
  and `sigma` (section 6), and `Delta'` the mean spacing of the found poles. Flag if `d > 0.25`
  (to be set by benchmark 2);
- **complete basis:** the distance of each pole weight from the nearest integer, and the
  total weight against `D`. A weight near 2 where the model has no degeneracy is a merge; a
  fractional weight is a false pole or a weight error;
- **partial basis:** only the bound `w_lambda <= m_lambda`; a weight above 1 where no
  degeneracy is expected is a merge. Nothing is demanded of the rest;
- stability: merge the found poles again at `Lambda' = 1.25 Lambda_eff`, then `1.5`, ...
  (the barycenter rule of Michaille and Pique) and record how the small-gap-ratio statistic
  moves. A steep change says the data set is at the edge of its resolution.

### 5.4 Benchmark 4: real data against the model (`run_model_bench`)

Per data set and fitter: match the poles to the model levels (with the data set's detunings,
couplings and Kerr); matched fraction, frequency residuals, weight residuals (against
`tr P_lambda` or the projected weight for a partial basis); and the small-gap-ratio statistic
against the model's own. **Validation only**: do not tune on it, because the model's own
errors (Kerr, detunings) would enter the choice of fitter.

## 6. The row-to-row offset `sigma`

Not known directly. Three estimates:

- **Floor:** the standard error of the Stark-calibration phase slope, per final occupation
  (`MBRStarkCalExperiment`, `phase_error`), converted to frequency:
  `|slope error| / (360 deg x cycle time)` (`calibration_sigma`).
- **Upper bound:** per row, the one linear phase roll that best matches the phase-corrected
  data to the model's `A_b(t)`; the spread of these offsets (`model_offset_sigma`). Model
  errors also enter it, so it is an upper bound.
- **Ceiling:** the `sigma` at which benchmark 2 loses the small-gap-ratio statistic (5.2).

Benchmark 2 sweeps `sigma` from 0 past the ceiling; benchmark 3 uses the floor and the upper
bound of each data set to bracket its `Lambda_eff`.

## 7. The small-gap-ratio statistic

The mean `<r>` is not used as the score: chaos is not fully developed here, and the phase
diagram of `<r>` does not follow the textbook limits. Instead, the small-`r` tail of the
gap-ratio distribution, `I(r0) = P(r < r0)` (`small_gap_ratio_fraction`), with
`r_n = min(s_n, s_n+1) / max(s_n, s_n+1)` from `fitting.qsim.mbr_disorder.adjacent_gap_ratios`
(bulk levels only), pooled over disorder realizations and compared with the model's own
prediction at the same point. The gap ratio needs no unfolding. Small `r` is the part that
merging and false poles change first. The choice of `r0` is open (section 10); `0.25` to start.

## 8. Running it

### 8.1 Entry point

`analysis_notebooks/pole_finding/report.py` (jupytext), laid out as a lab report: Method,
Benchmark 1, 2, 3, 4, Summary. Each section is a few lines that call one `run_*_bench`. The
Method section is rendered from the fitters' docstrings and settings, so the text cannot drift
from what ran. Run headless: `pixi run pole-report --fitters A,B,E --size small`
(`tools/pole_report.py`; executes the notebook to HTML; `--registry` from phase 2).

### 8.2 Data set registry

`analysis_notebooks/pole_finding/registry.yaml`, format `fitting.qsim.poles.registry.DataSet`
(`load_registry`), one entry per data set: its analysis frame (`Analysis`: phase frame, manual
Kerr or "recorded", branches, legacy, excluded occupations, the model's Kerr), so that
`experiments.qsim.pole_data.load_spectra` builds every spectrum from the entry alone
(benchmarks 3 and 4: `fitting/qsim/poles/real_benchmarks.py`,
`analysis_notebooks/pole_finding/real_benchmarks.py`); manifest path relative to `data_root()`, label,
complete or partial, disorder parameters, source log reference. guan converts jonginn's logs
to manifests (with `tools/migrate_mbr_jobs.py`) and to these entries. The benchmark reads
only the registry.

### 8.3 Outputs

`<data root>/<experiment>/derived_data/pole_finding/<run>/`: the HTML report, one HDF5 of
results with provenance (commit, settings, registry entries, seeds), and cached fits keyed by
(data set, fitter, settings hash). A summary note to the Obsidian vault. The large runs go on
pippin as a plain process (not the job worker), from the `guan` checkout.

## 9. Code rules

These keep the code readable top-down, as this file is.

1. Four levels: report notebook -> `run_*_bench` -> each fitter's `fit` -> pure numerics.
   A `fit` is a list of named steps (about 10-15 lines); a numerical function is under about
   30 lines and its docstring gives its equation.
2. One module per fitter, under about 200 lines. The benchmarks do not know fitter internals;
   the fitters do not know the benchmarks.
3. Results are frozen dataclasses (`PoleFit`, `BenchResult`), not dictionaries. Diagnostics
   come from separate functions (`explain_rank_choice`), not from extra result fields.
4. Settings are pydantic models; the only input checks are there and one shape check at each
   `fit`. No defensive checks inside the numerics.
5. A free parameter is added only when a benchmark shows it matters; its docstring names the
   trade-off.
6. Every step in this file names its function, and every function named here exists. A test
   checks this (`tests/test_pole_finding_spec.py`).

## 10. Open questions

- The rank rule (4.6): threshold or MDL; to be chosen by benchmark 2.
- `r0` for the small-gap-ratio statistic (7).
- The `d` flag threshold (5.3): 0.25 to start; to be set by benchmark 2 for our fitters.
- Fitter C: how well the offsets are fixed when rows share few poles (benchmark 2, with
  weak and one-row poles).
- Partial-basis sets: whether to require a minimum model support per eigenstate before
  their poles enter the statistics.

## 11. Phases

| Phase | Where | Content |
|---|---|---|
| 0 | here | this file; `PoleFit`; the registry format |
| 1 | here | generator; fitters A, B, E; benchmarks 1 and 2 on small grids; report skeleton |
| 2 | here | benchmarks 3 and 4 on the existing manifests; the three `sigma` estimates |
| 3 | pippin | full grids and the newly converted data sets; fitters C and D |

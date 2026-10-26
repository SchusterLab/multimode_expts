# 2026-10-26: the spectral function of the measured C_1

Branch `theory`. Seb asked for it; Claude wrote it.

## What was done

- The regularized AGP norm is a filtered spectral function, so we now look at that spectral
  function directly. New `fitting.qsim.agp.cosine_transform_weights(time_us, frequency_mhz,
  window)`: a matrix M with S = M @ C. S(f) = 4 int_0^T W(t) C(t) cos(2 pi f t) dt, one-sided,
  per MHz. A term w cos(2 pi Delta t) gives a peak of area w at |Delta|, the integral of S is
  C(0), and the AGP is int S(f) g^2 f^2 / (f^2 + mu^2)^2 df. Window: Hann (default) or raw.
  Tests: `tests/test_agp_kernel.py` (sum rule, peak area and position, filter integral = the
  spectral AGP).
- New `CentralBosonAGPDisorderScan.display_spectrum(mu, window, background_start_us, max_khz,
  raw, theory, yscale)`: S(f) per W with the shot error, the closed system over the same window,
  its weights in bins of 1/T, and the AGP filters (peak 1, right axis). With
  `background_start_us`, the exponential background fit is subtracted first, and the theory
  leaves out its DC terms (`split_dc`).
- Lab book script `analysis_notebooks/agp/labbook/q13_spectrum.py`: the six figures (S7 fine,
  S7 coarse, Kerr; with and without background) go to `labbook/figures/` (git-ignored). It runs
  as a script from any folder and in a REPL (no `__file__`: the folder is found from the
  `experiments` package).

## What was found (observations, not tested)

- S7 at W = 0, 30 and 60 kHz: the measured peaks agree with the closed system (about 40, 33
  and 62 kHz).
- S7 at W = 90 and 120 kHz: the closed system has weight below about 5 kHz, where the
  mu = 5 kHz filter looks; the measured S after background subtraction has less. A possible
  cause: a level pair about 2 kHz apart has a period longer than the 244 us window, and the
  exponential fit takes it away. Not tested.
- Kerr: the peaks are at the closed-system frequencies but wider and lower. After background
  subtraction the measured S has more weight below about 5 kHz than the closed system.
- Without background subtraction, the DC peak (width about 1/T = 4 kHz) overlaps the
  mu = 5 kHz filter.

## Limits

The error band is shot error only (no calibration error, no error of the background fit).

# 2026-10-24: the decaying DC background of C_1, and three ways to remove it (seb + Claude)

Follows `2026-10-24_theory-package-agp.md`. Notebook: `analysis_notebooks/agp/agp_dc_background.py`
(on branch `theory`). All model results are a toy, labelled as such; no device mechanism is claimed.

## The question

C_1 is not connected: its infinite-time (DC) part is 0.59-0.99 of C(0) for S7 and 0.06-0.13 (of
C(0) = 0.37) for the M1 Kerr probe. seb asked whether traces that decay to 0, zero-padded to
infinity, give the infinite-time AGP. jonginn (by word of mouth to seb) multiplies the data by
e^{t/T}, T about 200 us, the overall envelope, and saw the background removed well. His code is not
on the measurement PC; the method is `analyze(t1_us=T)` (`agp_t1`).

## Model (theory.qsim.agp: split_dc, damped_correlation, damped_agp)

A decaying constant is not removed by the kernel even over an infinite window: it adds
w rate / (2 mu (mu + rate)^2) per unit g^2. So zero-padding a decayed trace removes the truncation
error but keeps a decay error of the same size or more (toy, one rate 550 us, at W = 0, mu = 10 kHz:
closed 245 us window 1.00, decaying 1.45, zero-padded 1.46; at W = 150 kHz: 1.01, 102, 102).
Longer traces do not help by themselves.

Toy test of the estimators (notebook section 4): closed-system C_1, every term damped at the rate
fitted to the measured trace at that W, noise at the measured C1_err, 100 draws; ratio to the
closed-system infinite-time AGP:

- Exponential fit (A e^{-rate t}, t >= 30 us, subtracted) and the envelope with the fitted T agree
  with each other and are within about 20% (mu = 10 kHz) and 40% (mu = 5 kHz) for S7 coarse up to
  W = 90 kHz and for Kerr at all W (read off the notebook's model figure). S7 at W >= 120 kHz: the AGP is about 1e-3 and the uncorrected bias 20-100x;
  noise and fit errors are larger than the target.
- The envelope with a fixed T = 200 us is good for Kerr (0.9-1.9 at 5 kHz, about 1 at 10 kHz) and
  wrong for S7 at W >= 30-60 kHz (over-compensated, negative).
- S7 fine (122 us): no method is right at mu = 5 kHz; the window is too short.

## Data (notebook sections 2, 3, 5)

- Fitted envelope 1/rate (t >= 30 us): S7 fine 225 -> 517 us and S7 coarse 240 -> 439 us, rising
  with W; Kerr 146-357 us, about 150-240 us except W = 40 kHz (357 us). jonginn's 200 us matches the
  Kerr run, not S7 at W >= 30 kHz.
- The fitted amplitude A matches the closed-system DC part within 0.09 at every W, most within
  0.04 (S7 and Kerr): an independent check of the DC weight.
- Kerr compensated traces sit flat at the closed-system DC level, which is what jonginn saw; their
  late revivals (closed system, near 150 and 180 us) are absent.
- Measured AGP / closed-system infinite-time AGP (computed, not read off the plots):

  | | none | exponential fit | envelope 200 us | envelope fitted |
  |---|---|---|---|---|
  | Kerr, mu = 10 kHz, W = 20-140 | 1.12-1.49 | 0.97-1.10 | 0.94-1.12 | 0.93-1.08 |
  | Kerr, mu = 5 kHz | 1.71-3.86 | 0.94-1.57 | 0.93-1.67 | 0.90-1.55 |
  | S7 coarse, mu = 10 kHz, W = 0 / 30 / 60 | 1.63 / 1.35 / 2.78 | 0.55 / 0.87 / 0.70 | 0.27 / 0.62 / -1.43 | 0.53 / 0.86 / 0.70 |
  | S7 coarse, mu = 10 kHz, W = 90 / 120 / 150 | 5.4 / 21.6 / 116 | 0.16 / 0.30 / -6.0 | -6.1 / -30 / -177 | 0.15 / 0.28 / -6.0 |

  Kerr: the three corrected methods agree with each other and with the closed system within about
  10% at mu = 10 kHz. S7: the 200 us envelope is wrong at W >= 60 kHz; the fitted methods agree
  with each other but fall to 0.15-0.3 of the closed system at W = 90-120 kHz and are negative at
  150 kHz, where the model says the estimators cannot resolve the AGP anyway.

## Decisions / open

- No method is in `analyze()` yet; seb to decide after his own look. Candidates: the fitted envelope
  per W (= `analyze(t1_us=fitted T)`) or the exponential subtraction, with the fit start scanned as a
  systematic error. Not a fixed T for all W and directions.
- Library additions (with tests): `theory.qsim.agp.split_dc / damped_correlation / damped_agp`,
  `fitting.qsim.agp.fit_exponential_background`, `CentralBosonAGPDisorderScan.from_job_files`
  (a scan still running or never saved).

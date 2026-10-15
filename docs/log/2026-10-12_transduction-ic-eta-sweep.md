# 2026-10-10..12 weekend: alternating 2x / 1x HOM pass, Ic(eta) curve (6 passes), analysis fixes (transduction)

seb (plan ok 10-09, mixing coupler set points ok 10-09 21:55) + Claude. Full record:
`C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (10-10..10-12). Cases 75-156 in `ic_debug\losssplit\`;
results `homalt.md`, `icsweep.md`, `figs\icsweep.png`. No main config changed. Shared queue with three other projects at
coupler 0.07 mA most of the time (slower, no visible effect on the data).

## Done

- Alternating 2x / 1x HOM pass (case 75): 4 + 4 H jobs interleaved, A and B at both speeds.
- Ic(eta) curve: 11 points eta = 0 (full swap), 0.1 .. 0.9, 1 (no swap) + 0.35 in passes 1/3/5; 6 shuffled passes,
  69 sets (4 inputs x 250 reps), photon-number Ramsey eta_B and vacancy per point in pass 1.

## Found

- HOM asymmetry vs swap speed, drift-free: raw P0 - P2 2x - 1x = -0.026 +/- 0.006 (4.3 sigma); excess ratio 0.54
  (1/f predicts 0.5): the asymmetry builds up during the swap at a fixed rate. The vacancy does not depend on the speed
  (-0.4 +/- 1.5 %); the 10-09 "vacancy rises with the pump gain" was a session-to-session artefact (withdrawn).
- Ic(eta), gap to the ideal through the same estimator at the Ramsey eta (6 sets per point, +/- 0.01-0.035): ~0 at full
  swap and at no swap; -0.14..-0.15 for eta 0.28-0.38 (working point); -0.10 at 0.48-0.57 and 0.89; -0.03..-0.04 at
  0.67-0.79. The measured Ic follows the ideal shape from -0.69 to +0.64. The loss is in the partial swap with the env photon
  present; preparation, the env photon in S4 and the tomography cost nothing (eta 1 gap -0.016 +/- 0.021).

## Analysis fixes (all earlier numbers affected are listed)

- The thru reference was computed at eta = P0 of the '0' output (biased up by the vacancy / decay): a false gap of up to
  ~-0.15 where Ic(eta) is steep (eta 0, 0.1, 0.2, 0.9). Now at the Ramsey eta_B (`pool/pooled.analyze_jobs(eta=...)`).
  Near 0.33 (the Ic peak) the bias is small: earlier 0.35 gaps stand.
- The logical frame drifts between sessions (up to ~150 deg over a day); pooling raw parities of different sessions gives
  a false loss. Sets are now frame-aligned before pooling (`analyze_jobs(rot=...)`). Earlier pooled results that combined
  sessions without alignment (e.g. the 0.25 mA "pooled Ic 0.083", `pool/*.md` budgets) may be biased low: to recheck.
- A cache key used the last 5 characters of job ids (they repeat daily); fixed before the final numbers.

## Open (for seb)

- The cause of the two-path loss / HOM asymmetry: a fixed-rate process during the swap with both modes occupied.
- Recheck the older pooled results with the frame alignment (analysis only).
- B2 (vacancy sign test with a wait) not run.

## Afternoon: final Ic / Fe curves and the budget (seb asked; external review run)

Report: `ic_debug\final\REPORT_2026-10-12.md` (version 2), figures `final\figs\fig1..fig8`, review
`final\REVIEW_2026-10-12.md` (an external Claude agent, read-only; its main finding confirmed independently).

- **Withdrawn (10-08/09):** the "1.5-2 % env vacancy". The formula p = 1 - P1(A)/(1 - eta_B) counts the decay during the
  swap and the B-side losses; with B and A simulated with the decay, the vacancy is -0.15 +/- 0.25 %, i.e. 0. The swap
  reference is the coherent eta_c = eta_B + 0.003..0.009.
- |2> error: coherent, on all inputs with |2>, q = 5.0 +/- 0.5 % (P1 excess of '1' / '+' / '+i' at no swap).
- **Budget (ML Ic, through the same estimator):** M1 + S4 decay and q explain 40-60 % of the gap at eta 0.28-0.58; the
  remainder is -0.04..-0.09 per point (-0.09 at the working point), interior mean -0.037 +/- 0.010 (+/- 0.015 MODEL).
- **The remainder is the two-path coherence |rho02|:** single-path |rho13| and the populations match the MODEL, and the
  |rho02| null at eta 2/3 is filled (0.044 / 0.035 vs ideal 0.014, MODEL 0.016): the stay / exchange interference of
  |2>_M1 |1>_S4 is incomplete. A link to the HOM asymmetry (which grows with the swap time) is plausible, not tested.
- Proposed next (seb to decide): '+' at eta 2/3 and 0.33 in one session with the swap frequency scanned +/- 10-20 kHz.

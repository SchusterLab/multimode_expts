# 2026-10-08/09 night: |rho01| result, channel eta scan, 2-photon swap angle (transduction)

seb (approved the night plan, asleep) + Claude. Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md`
(10-08 20:38 onward). Cases 61-67 in `ic_debug/losssplit/`; results `rho01.md`, `etascan.md` (+ `figs/etascan.png`),
`n2swap.md`; model scripts `sim_random_kerr.py`, `sim_n_dependent_swap.py` (MODEL, labelled). No main config changed.

## Done

- 6 more channel |0>+|1> jobs + vacancy (case 61); channel eta scan at eta 0.2 / 0.35 / 0.5 / 0.8 with eta and vacancy
  per point (cases 62-65); 2-photon swap angle (case 66); bare and env-only controls (case 67). All ended 04:48.
- Two short cases were added during the night by Claude (n2swap, ctrl; ~55 min in total, within the 05:30 window).

## Found (measured)

- Channel |0>+|1> (10 jobs): |rho01| 0.839 +/- 0.033 after the vacancy, |rho12| 0.981 +/- 0.011. The random cross-Kerr
  model (it predicted 1.04) is excluded at ~6 sigma.
- Eta scan (vs the ideal at the single-photon eta): eta 0.185 |rho02| .871 / |rho13| .882; 0.328 .910 / .971 (afternoon
  .853 / .985); 0.480 .896 / .891; 0.779 1.28 (the ideal is steep there) / .831; errors ~0.03-0.04. The single-path
  coherence is ideal only at the working point eta 0.33.
- |2> + partial swap: the pure-loss model fits (chi2 0.1/2, 1.8/2); eta2 - eta1 = -0.0095 +/- 0.005 (0.35), +0.002 +/- 0.008
  (0.8). The swap is linear in the N <= 2 sectors to ~1 %; the photon-number-dependent swap angle (MODEL) is excluded.
- Controls: bare '+' 0.97 / 0.94, env-only 0.98: the readout and the env photon alone are fine.
- Env vacancy over the night 0.009 +/- 0.006 (pooled), 0.018 at eta 0.35.

## Corrections

- At 23:30 Claude called eta 0.8 a "sharp test" of the vacancy; the ideal |rho02| is steep in eta there, so that point
  mostly measures the channel's effective eta.

## Open (for seb)

- The cause of the remaining channel coherence loss (two-path at eta 0.33; both types at other eta). All simple swap
  and readout models tried so far are excluded.

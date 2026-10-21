# 2026-10-21: night 10-20/21 Ic(eta) after the cooldown

Chain `losssplit/night_1020.sh` + `night_1020_extra.sh` (seb ok 10-20 17:00 / 21:45). Settings fixed all night:
`CFG-HW-20261020-00034` (g2a 2.558e-4, so s = 1), storage `CFG-M1-20261020-00010`, swap 2973 / 874.283 (10-19),
sidebands = the 10-19 values via `--sb` (f0-g1 1989.6315, f1-g2 1989.0730, lengths 0.5599 / 0.3952; chosen by the
grid + confirmation rule, see `2026-10-20_sidebands-g2a.md`). Each step its own process, single shot each, no refresh.
19:14-04:41, all steps exit 0; gates 19:30 (|2> P2 0.954) and 00:11 (0.963) passed. From 04:00 another user (guan)
shared the queue (one set 11 min longer); the planned extra 0.2 set and a bare-input check were skipped (5 am rule).

## Results (analysis `ic_debug/final/night1020_curve.py`, method of night1019_analysis.py)

| eta | eta_c | sets | Ic | ideal / MODEL d+q through | weekend Ic |
|---|---|---|---|---|---|
| 0 | 0.007 (weekend calib, no ref tonight: do not use) | 1 | -0.606 +/- 0.015 | -0.671 / -0.667 | -0.687 |
| 0.1 | 0.090 | 1 | -0.391 +/- 0.029 | -0.407 / -0.445 | -0.426 |
| 0.2 | 0.177 | 1 | -0.123 +/- 0.029 | -0.131 / -0.175 | -0.145 |
| 0.35 | 0.332 (10-19 ref) | 3 | +0.010 +/- 0.054 (+0.115 19:59, -0.068 00:40, -0.016 03:59) | +0.185 / +0.108 | +0.036 |
| 0.5 | 0.481 | 2 | -0.088 +/- 0.052 (-0.154, -0.050) | +0.008 / -0.049 | -0.082 |
| 0.65 | 0.638 | 1 | -0.234 +/- 0.029 | -0.231 / -0.246 | - |
| 0.8 | 0.801 | 1 | -0.055 +/- 0.039 | -0.009 / -0.040 | -0.035 |
| 0.9 | 0.901 | 1 | +0.158 +/- 0.048 | +0.326 / +0.278 | +0.214 |
| 1 | no swap | 1 | +0.656 +/- 0.035 | +0.668 / +0.644 | +0.641 |

## Findings

- The curve agrees with the ideal / MODEL at eta <= 0.2, 0.65, 0.8, 1; deficits at 0.35, 0.5, 0.9 as on the weekend.
  The 10-19 night (three 0.35 sets, all +0.12..+0.16) had no deficit; tonight only the first 0.35 set.
- The deficit is a loss of the 0-2 output coherence: |rho02| of '+' / '+i' at 0.35 was 0.18-0.20 on 10-19, tonight
  0.156 / 0.170 (19:59), 0.127 / 0.143 (00:40), 0.130 / 0.127 (03:59); |rho13| unchanged (0.16-0.20). The populations
  (input |0>: P1 0.66-0.71) and eta are the same in all six 0.35 sets. The input coherence is full (eta 1 set, no swap:
  |rho02| / 0.5 = 1.007 / 0.999) -> the loss is in the channel (swap), not in the preparation.
- The logical frame (arg rho02 of '+') moved ~110 deg between 10-19 (-5..-9 deg) and tonight (-115..-124 deg), and
  9 deg during the night. Hypothesis: today's photon-1 rung retune (g1-e1 -70 kHz, e1-f1 -80 kHz). A phase alone does
  not change Ic; it only shows that the setting differs.

## Next (proposed to seb, not decided)

- Interleaved 0.35 sets with the 10-19 hardware config (`CFG-HW-20261019-00062`) and tonight's (`20-00034`): does
  the loss follow the config (rungs) or the time?
- An eta reference at 0 for this swap; a bare-input check next to each low set.

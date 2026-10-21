# 2026-10-20: gain to alpha (B'), sideband N drift, |2> preparation grid, night Ic(eta) setup

Day after the cooldown recalibration (night record: `2026-10-20_cooldown-recal-night.md`). seb approved each step.
Sandbox: `C:\experiments\260601_Transduction_sandbox\` (`autocal/`, `ic_debug/losssplit/`, `ic_debug/final/`).

## Conditions

- PC crash 03:30 (bugcheck 0x3B); QICK restarted; readout IQ phase ~130 deg off -> plain single shot first in every run.
- Yoko unreachable after the crash until seb rebooted it (~14:45); jobs before that used `avoid_yoko=True` (yoko left
  at 0.30 mA). The first single shot with the yoko set again gave the same angle (+0.41 vs +0.45 deg) -> the coupler
  was at the setpoint; the avoid_yoko results stand.
- 18:04-18:08 main was checked out at HEAD~2 by another user and back; only `experiments/autocal/_manipulate.py` and
  its test differed; no effect on the jobs (Wigner / ladder code unchanged).

## Gain to alpha

- Coherent-state photon-number Ramsey (`autocal/coh_g2a.py`): 2.538e-4 (linear within 0.6 %). Fix on the way:
  `MM_base.custom_pulse` takes the DAC channel (manipulate = 3), not the channel id of its docstring.
- B' in autocal (41 gains 0-4480, 2000 reps, width fit, offset fixed; `run_g2a_test.py`, JOB-20261020-00060):
  2.558e-4 +/- 0.7 %, alpha_scale 0.965 -> `CFG-HW-20261020-00020`. Rounds > 1 fails in `ParityGainExperiment`
  with active reset (reshape error, shared code, not changed).
- Check: the 10-19 gate Wigners re-analysed vs the alpha scale give s ~ 0.975 on the old 2.641e-4 -> ~2.57e-4,
  consistent with B'. The 10-19 |2> P3 0.06 was the scale.

## Sidebands and the |2> preparation

- N drift confirmed (seb): the 10-19 stored values (20-pulse EA) were 87 / 71 kHz above the N = 1 maps of 01:00.
- Refresh 14:29 (`run_sb_n1.py`): EA 1989.503 / 1988.954; N = 1 1989.583 / 1989.040 (rows within 11 / 2 kHz) ->
  `CFG-HW-20261020-00027`, storage `CFG-M1-20261020-00010`. The EA moved 128 kHz in a day and changed side.
- Gate 15:07 with the N = 1 values failed narrowly: |2> P2 0.913 (P1 0.047).
- Photon-1 rungs (`run_rung1.py`, 15:50): g1-e1 3561.81 -> 3561.74, gain 9017 -> 9130; e1-f1 3417.83 -> 3417.75
  -> `CFG-HW-20261020-00034`.
- A/B (losssplit case `ab2`, |2> Wigner, 500 reps, relax 2500, A B C A B C): EA 0.905 / 0.913, N = 1 0.928 / 0.931,
  10-19 values 0.949 / 0.958.
- 5 x 5 grid (case `grid2`, offsets -40..+120 kHz from N = 1, new lengths, shuffled), P2:
  f0-g1 row means -40: 0.940, 0: 0.944, +40: 0.955, +80: 0.957, +120: 0.941; f1-g2 flat (column means 0.937-0.957);
  quadratic-fit rms 0.013 (= the repeat noise; the N = 1 point gave 0.947 here vs 0.93 in the A/B). The 10-19 values
  (f0-g1 +48, f1-g2 +33) are inside the optimum. Confirmation (`conf2`): grid best +80 / +120 0.943, 10-19 values
  0.953 -> chain rule (seb): the 10-19 values for the night.
- Conclusion: neither the EA nor the N = 1 centre is the preparation optimum for f0-g1 (~+60 kHz above N = 1);
  f1-g2 does not matter within this range. A final-state optimisation is needed in autocal (not done; ask seb).

## Night 10-20/21 (seb ok)

- Chain `losssplit/night_1020.sh`: Ic(eta) with the 10-19 sideband values fixed (`--sb`), each step its own process
  (single shot each, no N = 1 refresh), order 0.35, 0, 0.5, 0.2, 0.8, 1, 0.35, 0.65, 0.1, 0.9, 0.35 with eta
  references after the first set of each eta (none at 0 / 1; 0.35 reused from 10-19 case 5, same swap), gates
  before the first and the second 0.35 set; no new step after 04:00. Records `night1020_case_*.json`.
- Start gate 19:30 passed: vac 0.986, |1> 1.005, |2> 0.954. First set at 0.35: Ic +0.115 +/- 0.044 (ideal through the
  estimator +0.174, MODEL decay +0.139, decay+q +0.105; `final/night1020_quick.py`), as 10-19 (+0.140 +/- 0.027).
- Note: the 10-19 night sets used the stored EA sidebands (no refresh); the weekend sets refreshed them with N = 1
  maps before each case. A second difference between the two besides the swap gain; not tested.

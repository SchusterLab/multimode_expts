# 2026-10-29: decay check at 0.3 / 0.5 mA, records rule, workflow drafts (guan + Claude)

Physics: vault `Lab/guan/AGP/2026/10/2026-10-29_TLS_12GHz_report.md` (suspected TLS near
11.96 GHz; |f,n> -> |g,n-1>). This entry keeps only code, config and decisions.

## Configs and branches (`measurement_notebooks/guan/branches.jsonl`)

- `agp_0.30mA`: seb's `CFG-HW-20261028-00170`, `CFG-M1-20261020-00010` (floquet: main version);
  after the single shot `CFG-HW-20261029-00013`. Its g2-e2 / e2-f2 / f2-g3 entries are
  placeholders (g2-e2 +4.0 MHz, e2-f2 3421.97, f2-g3 1996.13); the |f,2> holds set g2-e2 and e2-f2
  in RAM only (cell 11).
- `agp_0.50mA`: jonginn's last ladder calibration of 10-28 (found from his job configs:
  JOB-20261028-00739..00774), `CFG-HW-20261028-00169`, `CFG-M1-20261028-00047`,
  `CFG-FL-20261027-00074`; after the single shot `CFG-HW-20261029-00015`.
- jonginn's 0.45 mA f2-g3 entry (`CFG-HW-20261028-00046`) is calibrated at gain 10000; all other
  sideband entries at 8000.

## Events

- 11:16: the QICK board Pyro server dropped connections (seb's JOB-20261029-00091..00098 failed),
  and the job server and worker were down until about 11:30 (restarted by someone else). After
  the restart the readout phase was off by ~160 deg (single shot angles 160.6 / 163.4 deg): every
  user needs a new single shot before relying on phase, threshold or active reset.

## Decisions

- guan: history and physics records go to the vault; the repo is for code. The 10-28 log entry was
  cut down to code and config on 10-29 at guan's request (its physics is now the vault Day summary).
- guan: `docs/reference/notebook_free_workflow.md` retired; drafts
  `docs/measurement_session_workflow.md` and `docs/lab_workflow_tooling_spec.md`.
- guan: the second kernel per user (analysis) is the intended way to work while the measurement
  kernel waits for the queue; no architecture change without the other users.

## Open

- A set point above 0.5 mA (nu_n below ~11961 MHz) to see the other side of the resonance.
- Tooling spec items A-D; E with seb and jonginn.

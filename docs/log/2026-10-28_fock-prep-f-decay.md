# 2026-10-28: M1 Fock prep at 0.45 mA: configs, code findings, decisions (guan + Claude)

Physics and history: vault `Lab/guan/AGP/2026/10/2026-10-28.md` (per-job notes and the "Day
summary" at the end). Short form: Fock 2/3 are limited by a decay |f,n> -> |g,n-1> (tau ~6 us
n=1, ~2.4 us n=2 at 0.45 mA; ~20x slower at 0.25 mA). This entry keeps only code and config.

## Configs and branches

- Script: `measurement_notebooks/guan/yoko450ua_calibration.py` (cells 1-10; last cell: the
  module-gap list). Data: `C:\experiments\261028_yoko450ua_calib\`.
- `agp_0.45mA` (`measurement_notebooks/guan/branches.jsonl`): from jonginn's
  `CFG-HW-20261028-00035`, `CFG-MP-20260121-00001`, `CFG-M1-20261027-00052`,
  `CFG-FL-20261027-00074`; after the ge single shot `CFG-HW-20261028-00046`.
- `agp_0.25mA`: from guan's `CFG-HW-20261026-00006`, `CFG-M1-20261023-00003`,
  `CFG-FL-20261023-00002`; after the single shot `CFG-HW-20261028-00171`.

## Code and config findings (no code changed)

- The Fock-prep pulses read only `device.multiphoton` (`MM_base.prepulse_creator2.multiphoton`).
- jonginn's `Autocalibrate.ipynb` cell 63 and `single_qubit_manualcalibrate.ipynb` cell 94
  (`lenrabi_f0g1_postproc`) write the f0-g1 length only to the ds_storage `M1` row, which no pulse
  reads. Several chevron and error-amp steps (his notebooks; `calibrate_f0g1` in
  `guan/single_qubit_autocalibrate_v2.py:2803-2832`) take their scan range or gain from that row.
- `multiphoton_calibration_v2_jonginn.ipynb`: Ramsey and amplitude Rabi drive at the bare
  f_ge / f_ef but write the entry of photon number n; `fetch_initparam_to_mp` resets all rungs to
  the n=0 values. His 0.45 / 0.5 mA tables have g1-e1 on the wrong side of g0-e0 (+0.3 MHz; the
  line is at f_ge + n*chi_ge, chi_ge < 0). Nothing checks the table against `chi_ge`.
- `tests/test_f0g1_source_of_truth.py` was deleted 2026-10-05; nothing guards the migration now.
- `HistogramExperiment` with f shots fits the e-f axis, so its gf fidelity is not g vs f.
- Board resources: three jobs failed at upload (JOB-20261028-00775, -00777, -00780, "buffer
  length must be 32768 samples or less"). Envelope memory is the total per generator; every
  prepulse gate adds its own envelope (896 samples per 35 ns pi); the standard qubit pulses take
  4992 on ch 2. QICK's compile and `MockQickSoc` do not check this. Fix specified in
  `docs/lab_workflow_tooling_spec.md` item D.

## Decisions

- guan: the decay is plausibly a TLS near ~12 GHz; little to do before a thermal cycle; all set
  points are needed.
- guan: deliver a short copyable notebook for jonginn plus the module-gap note.
- guan (2026-10-29): history and physics records go to the vault; the repo is for code. Docs:
  `docs/measurement_session_workflow.md` (draft), `docs/lab_workflow_tooling_spec.md` (draft);
  `docs/reference/notebook_free_workflow.md` retired.

## Open

- The short notebook for jonginn and the module-gap note.
- jonginn: fix the `M1`-row writes and the photon-shift sign in his calibration notebooks.
- Tooling spec items A-D for a code session; E for discussion with seb and jonginn.

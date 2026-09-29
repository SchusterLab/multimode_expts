# 2026-09-29: plan for the qsim Program and Experiment tree

## What was done

- Merge check of `guan` into `main` (`git merge-tree`, no files changed): 213 commits ahead;
  conflicts in `.gitignore` and in two jonginn notebooks that `guan` deleted and `main` changed
  (`measurement_notebooks/jonginn/data_postprocess.ipynb`, `qsim_experiments.ipynb`).
- Static call graph of the 51 Program classes in `experiments/qsim/` (for each method reached
  from `initialize`/`body`/`update`, the class that defines it), and a survey of the Experiment
  classes and their `acquire` methods. Results and the plan: `docs/qsim/program_tree_plan.md`.

## Found

- The MBR job programs do not go through `DarkBaseProgram` or `SidebandScrambleDarkProgramNewNew`
  (since step 8A1). From `QsimBaseProgram` they use only the Floquet swap setup.
- `DarkBaseProgram` is `QsimBaseProgram` with a copied, extended `body` and `initialize`. The
  features (Floquet swap setup, Floquet playback, dark-mode encoding) nest, so single
  inheritance fits; one large mixin base is not needed.
- `SidebandScrambleDarkProgramNewNew` uses nothing from its parent `SidebandScrambleProgram`.
- The sweep loop has four copies (QsimBase, DarkBase, MBRJob, Wigner) plus one in
  `SidebandAmpRabiExperiment`. The readout count has five copies that do not agree (table in the
  plan, 2.2). `post_select_pre_pulse` is counted by the Wigner Experiment, but no live Program
  plays that readout.
- `FloquetTrain._advance_storage_phase_offsets` has no callers.
- `ManipulateModePulses.man_reset` differs from `MM_base.man_reset` only by
  `dump_reset_iter_num` (default 1) and a debug print.
- The readout flags (`perform_wigner`, `parity_readout`, `multiparity_readout`,
  `slow_pi_ge_readout`) are used only in `experiments/qsim/` and the measurement notebooks;
  `fitting/` and `analysis_notebooks/` do not read them.

## Decisions (guan)

- Refactor on `guan` before the merge, so the other users restart their kernels once. The
  device runs are in local mode (`use_queue=False`) with the worker off.
- `acquire` is in scope. Class names change; the consumers are migrated, no aliases.
- One `readout` key replaces the readout booleans (guan: "a great idea"), with the rules in the
  plan, section 4.
- The 8 unused classes stay for now; retirement waits for the other users.
- `saved_jobs.select_jobs(program_class=...)` is an ad hoc filter over the job log, not in the
  acquisition path; old names there need no support.
- The plan is approved: the names as proposed (the two jonginn rows wait for jonginn); the
  DarkBase behavior where the two templates differ.
- `MM_base` changes only if the change is tiny, local, and has no effect on consumers; it gets a
  full rewrite with the v2 migration (out of scope). So `dump_reset_iter_num` does not move into
  `MM_base`: non-qsim experiments get the key too (`SidebandGeneralExperiment` in
  `multiphoton_calibration.py`). The qsim `man_reset` override moves into `QsimProgram`. All live
  configs set the key to 1, which gives the same pulses.
- `FloquetCalibrationAmplificationExperiment` (sweep or loop wrapper) is reviewed at 10E.

## Device

guan ran the measurement code on the device on 2026-09-28 in local mode: no errors, but the
data was not meaningful. The calibration steps need a manual check.

## Step 10A: the nets (later the same day)

- `tests/program_asm_golden.py` + `tests/test_program_asm_golden.py`: 75 cases (details in the
  plan, section 6). Deterministic (two runs identical); 77 tests in about 2.5 s.
- `tests/notebook_asm_golden.py`: `floquet_displacement_kerr` added. The two other notebook
  goldens came out byte-identical when regenerated.
- `tests/acquire_golden.py` + `tests/test_acquire_golden.py`: 24 cases, lane-tagged fake shots,
  one JSON golden (`tests/data/acquire_golden.json`). Deterministic.
- Full suite: 1494 passed, 3 failed, 9 xfailed, 9 xpassed. The 3 failures are in
  `tests/test_matrix_pencil_regression.py` (tolerance 1e-12; the largest relative difference is
  7e-6, in `complete_basis-disorder`). They fail every time and do not involve the new tests;
  they are for the analysis theme (`fitting/qsim/matrix_pencil`).

Found in 10A:
- The two templates compile to the same program for 13 of the 19 options, active reset
  included. They differ for 6, each a DarkBase feature that QsimBase lacks (plan 7.1).
- `ManipulateModePulses.prep_man_fock_state` (DarkBase) accepts any photon number;
  `MM_base.prep_man_fock_state` (QsimBase) only 0, 1 and the superpositions.
- `FloquetChevronProgram`, `FloquetPhaseCalProgram` and `SidebandAmpRabiProgram` set their own
  `length` on the Floquet pulse. That works only for native flat-top pulses; both pinned sets
  have arb envelopes (`preload_flattop`, `gauss`), and qick raises. So these three cannot run
  on today's Floquet calibration.
- `SidebandAmpRabiExperiment` raises unless the config sets `perform_wigner` (the other
  drivers default it); its parity lane is an active-reset lane when both are on.
- The acquire golden shows the lane errors of plan 7.3 in the values: with
  `multiparity_readout`, `QsimBaseExperiment` asks for 1 readout where `DarkBaseExperiment` asks
  for 2; with `post_select_pre_pulse`, the Wigner driver asks for 2 where the Program plays 1.
- `MM_base.lane_layout(cfg)` is the lane layout that the dual-rail and single-qubit experiments
  use. It knows `post_select_pre_pulse` (their programs play it) but not `parity_check` or
  multiparity. It is probably where the Wigner driver's count came from.
- A wrong lead, checked: `SlowPiGeLengthRabiProgram` reads `cfg.length_to_sweep` and the driver
  sets `cfg.expt.length_to_sweep`, but `MM_base_initialize` copies `cfg.expt` to the top level
  (`MM_base.py:84`), so it works.

## Step 10B: one template (later the same day)

What changed:
- `QsimBaseProgram` has DarkBase's `initialize` and `body` (without `storage_phase_matrix` and
  a stray `print("running")`), and the three manipulate-mode methods (`man_reset`,
  `prep_man_fock_state`, `multi_parity_readout`). `experiments/qsim/manipulate_mode_pulses.py`
  is deleted.
- `DarkBaseProgram` has no methods of its own (DarkModeEncoding + FloquetTrain +
  QsimBaseProgram). `DarkBaseRProgram` takes its borrowed methods from `QsimBaseProgram` and
  still plays MM_base's `man_reset` (the asymmetry `tests/test_pulse_layer_layout.py` pins).
- `MBRRamseyProgram` inherits `man_reset` from `QsimBaseProgram` (it set it explicitly).
- `FloquetTrain._advance_storage_phase_offsets` is removed (no live caller).
- The class names stay until 10F.

Checked:
- Program golden: exactly the 6 predicted `qsim_template__*` cases changed, and each is now
  byte-identical to its `dark_template__*` case (plan 7.1). The 4 cases that raised on QsimBase
  now compile. All leaf cases, the MBR ASM golden, the three notebook goldens and the acquire
  golden are unchanged.
- Full suite: 1491 passed, 2 skipped (below), 3 failed (the matrix-pencil tolerance failures
  of 10A, not related).

Effects for users of the ~25 QsimBase template leaves (none changes a config that worked
before):
- `multiparity_readout`, `slow_pi_ge_readout`, `init_man_fock_state` above 1, `do_crude_comp`,
  and `init_alpha` without Wigner now work as on the DarkBase programs. Before, the first two
  were ignored and the others raised.
- `init_man_fock_state=None` now goes to the coherent-state branch (before: an error).
- The active reset plays the qsim `man_reset`, which repeats the dump pulses
  `dump_reset_iter_num` times. All live configs set 1, which gives the same pulses.
- Until 10E, `multiparity_readout` through `QsimBaseExperiment` reads the wrong lane (that
  driver counts one readout, the program now plays two). Noted in the status file.

Deprecated code: `deprecated/mbr_sff.py` borrowed `_advance_storage_phase_offsets` in a class
body, so it no longer imports; with it `deprecated/mbr_sff_campaign.py` and
`dormant/mbr_sff.py` (SFF, to be deleted, step 7 decision 3). Following their headers and
`tests/test_deprecated_mbr.py`: a note in both module headers and the two SFF tests skipped, not
fixed.

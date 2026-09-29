# qsim Program and Experiment tree: plan (steps 10A-10G)

Status: approved by guan 2026-09-29 (decisions 0.7-0.10); 10A-10D done 2026-09-29; 8.3 is
reviewed at 10E. It continues
`mbr_step9_plan.md`; it uses `mbr_redesign.md` for the rules and patterns.

## 0. Decisions (guan, 2026-09-29)

1. Do this refactor on `guan` before the merge to `main`, so the other users restart their
   kernels only once. Device checks run in local mode (`use_queue=False`) with the worker off.
2. The `Experiment.acquire` methods are in scope: they share a contract with the Programs (the
   number of readouts in one shot).
3. Class names change where they do not say what the class does. The consumers (notebooks) are
   migrated. No aliases for old names.
4. The final readout is chosen by one key, `readout`, which replaces the booleans (section 4).
5. The 8 classes that no notebook or test uses (section 2.4) stay. Their retirement waits for
   the other users. They are only re-parented when their base changes, so that they still load.
6. `saved_jobs.select_jobs(program_class=...)` is an ad hoc filter over the job log. It is not in
   the acquisition path, and old names in it need no support.
7. The names in section 5 as proposed. The two jonginn rows wait for jonginn.
8. 7.1 as proposed: take the DarkBase behavior.
9. `MM_base` changes only if the change is tiny, local, and has no effect on consumers. It gets
   a full rewrite with the v2 migration, which is out of scope. So 7.2: `MM_base` is not changed.
10. 8.3 is reviewed when 10E starts.

## 1. Scope

In: the Programs and Experiments of `experiments/qsim/` (not `deprecated/`), the notebooks in
`measurement_notebooks/202609_qsim_migration/`, and the users' sandbox notebooks that use these
classes. Out: `MM_base` (decision 0.9), `postpulse` (used in about 45
files across `experiments/`), and the programs on QICK hardware sweeps (`SidebandRamsey`,
`SidebandChevron`, the two `CouplerRabiFreqsweep` experiments, `SidebandAmplitudeSweep`).
The old leaves that play Floquet pulses with their own loops (`SidebandScrambleProgram`,
`SidebandStarkAmplificationProgram`) are not moved onto `FloquetTrain` here: that changes their
compiled programs and needs a device check of its own.

## 2. Facts (call graph of 2026-09-29)

Static call graph from `initialize`/`body`/`update` of the 51 Program classes: for each method
reached, the class that defines it. It counts all branches, also branches that a flag turns off.

### 2.1 Which program uses which feature

| Feature | Now in | Used by |
|---|---|---|
| reset, pulse creator, active reset, readout | `MM_base` | all 51 |
| Floquet swap setup (`retrieve_swap_parameters`, `_initialize_floquet_pulses`) | `QsimBaseProgram` | about 35 |
| template `body` (reset, herald, prepulse, `core_pulses`, postpulse, readout) | `QsimBaseProgram.body` | about 25 leaves; they override only `core_pulses` (some also `initialize`) |
| the same template, copied, with more readouts (multiparity, slow pi), f-g gain compensation (`do_crude_comp`) | `DarkBaseProgram.body` | 6 leaves |
| Floquet playback and phase bookkeeping | `FloquetTrain` (mixin) | 8: the 4 MBR programs, `SidebandScrambleDarkProgramNewNew`, `DarkT1Program`, `SidebandStarkAmplificationModifiedProgram`, `FloquetDisplacementKerrProgram` |
| dark-mode load and read | `DarkModeEncoding` (mixin; calls `FloquetTrain`) | 2: `DarkT1Program`, `SidebandScrambleDarkProgramNewNew` |
| `man_reset`, `prep_man_fock_state`, `multi_parity_readout` | `ManipulateModePulses` (mixin) | through the DarkBase template; `man_reset` also MBR |

The features nest: dark-mode encoding needs the Floquet playback, which needs the Floquet swap
setup. This fits single inheritance.

### 2.2 Copies and parents that are not used

- `DarkBaseProgram` is `QsimBaseProgram` with a copied `body` and `initialize`, plus three
  mixins. The known differences are in 7.1.
- `SidebandScrambleDarkProgramNewNew` has `SidebandScrambleProgram` as a parent but uses nothing
  from it.
- The sweep loop has four copies: `QsimBaseExperiment.acquire`, `DarkBaseExperiment.acquire`,
  `MBRJobExperiment.acquire`, `QsimWignerBaseExperiment.acquire`. `SidebandAmpRabiExperiment`
  has a fifth, written for its own 2D sweep.
- The readout count has five copies, and they do not agree:

| Where | Counts | Problem |
|---|---|---|
| `readout_lane_count` (DarkBase, MBR, and `mbr_spectrum` for old jobs) | parity check, active reset, multiparity | reference |
| `QsimBaseExperiment.acquire` | parity check, active reset | no multiparity |
| `QsimWignerBaseExperiment` (`acquire` and `analyze_wigner`) | the above + `post_select_pre_pulse` | no live Program plays that readout: with the flag set, the lanes are off by one |
| `ManStorMultiparityChevronRExperiment` | active reset, multiparity | no parity check |
| `SidebandAmpRabiExperiment` | parity check, active reset | parity lane sliced at `0::read_num`, before the active-reset readouts |

### 2.3 Wigner

Wigner is a readout, not a pulse sequence. The Wigner notebooks (guan, jonginn, connie) run
`QsimWignerBaseExperiment` over `KerrWaitProgram`, `SidebandScrambleProgram` and
`CoolingSpectroscopyProgram`. `displace_man` is also used for state preparation (`init_alpha`,
`FloquetDisplacementKerrProgram`). So Wigner is a readout mode on the Program side and a child
class on the Experiment side (an extra sweep axis and its own analysis).

### 2.4 Not used by any notebook or test (they stay, decision 0.5)

`CoolingProbeProgram`, `CoolingF0g1CalProgram` (and `_CoolingBase`), `KerrStarkProgram`,
`Qsimf0g1Sepctroscopy`, `StorageSwapPhaseAccumulationProgram`,
`ManStorMultiparityChevronRProgram` (the only user of `DarkBaseRProgram`),
`SidebandStarkAmplificationModifiedProgram_newold`.

### 2.5 Recorded names

The worker loads the Experiment class by module and name, at run time. The HDF5 file name holds
the Experiment name (`JOB-<id>_<Experiment>.h5`), and `cfg.expt.QickProgramName` holds the
Program name. Nothing in the acquisition path reads old names back. Old files load by path.

## 3. Target

### 3.1 Programs

```
MMAveragerProgram (MM_base)
 └ QsimProgram          QsimBaseProgram + DarkBaseProgram in one: Floquet swap setup, ONE
    │                   template body, the readout modes (section 4), readouts_per_shot(cfg),
    │                   man_reset / prep_man_fock_state / multi_parity_readout
    ├ template leaves   override core_pulses (and initialize where they do now)
    ├ own-body leaves   BroadbandGeValidation, the Kerr family: as now, on the new base
    └ FloquetProgram          + FloquetTrain (now a class, not a mixin)
       ├ FloquetDisplacementKerrProgram, SidebandStarkAmplificationModifiedProgram
       ├ MBRRamseyProgram     replaces body (as now); the MBR job programs below it
       └ DarkModeProgram      + DarkModeEncoding (now a class, not a mixin)
          └ DarkT1Program, SidebandScrambleDarkProgramNewNew (new name in section 5)
```

`DarkBaseRProgram` (RAverager variant) takes `body` from `QsimProgram`, as it takes it from
`DarkBaseProgram` now. `ManipulateModePulses` and `dark_base.py` go away.

### 3.2 Experiments

```
slab Experiment
 └ QsimExperiment        one sweep loop over N axes (outer ... inner); readout count from
    │                    ProgramClass.readouts_per_shot(cfg); raw shots, pre-selection,
    │                    parity lanes, normalize, save, derived params
    ├ analysis-only      FloquetChevron, FloquetGainChevron, FloquetPhaseCal, SidebandStark*,
    │                    DarkT1, KerrCavityRamsey, ...: analyze/display only
    ├ MBRJobExperiment   inner axis fixed to ramsey_phase; default_program
    ├ WignerExperiment   sets readout='wigner'; adds the wigner_alpha axis and, with
    │                    pulse_correction, a phase_second_pulse axis [180, 0]; Wigner analysis
    └ loop wrappers      FloquetCalibrationAmplification (outer axis floquet_cycle, derived
                         from n_scramble_cycles), FloquetDisplacementKerr (adds floquet_cycle_us)
```

`DarkBaseExperiment` goes into `QsimExperiment`. `analyze_multiparity` and
`classify_two_parity_readouts` move next to the multiparity readout mode.
`SidebandAmpRabiExperiment.acquire` is deleted: the driver does it with
`swept_params=['detune', 'gain']`.

For 1D and 2D sweeps the data keys stay as now (`xpts`, and `ypts` for 2D). N-D sweeps also
record the axes in order. Whether the Wigner data keeps its present shape
(outer, inner, alpha[, 2], shots) is checked in 10E.

## 4. The `readout` key

One key, `cfg.expt.readout`, chooses the last step before the measurement. The sequence is:
prepulse (encode), `core_pulses`, postpulse (decode), readout step, measurement. `postpulse`
keeps its meaning: the decoding pulses. It swaps `ro_stor` into M1, and for the `'qubit'`
readout it also maps M1 to the qubit (f0-g1, then ef pi if `map_to_qubit_ge`); the other modes
read M1, so the photon stays there. The readout step plays with or without `postpulse`.
`parity_check` (a herald before the sequence) stays a separate boolean. (Decided by guan,
2026-09-29, after a first version with an `'m1'` mode whose meaning depended on `postpulse`.)

| `readout` | Readout step | Final readouts | Replaces |
|---|---|---|---|
| `'qubit'` (default) | none: the qubit is measured | 1 | no flag |
| `'parity'` | parity pulse | 1 | `parity_readout` |
| `'multiparity'` | two parity readouts | 2 | `multiparity_readout` |
| `'wigner'` | displacement by `wigner_alpha`, parity pulse | 1 | `perform_wigner` |
| `'slow_pi_ge'` | slow ge pi (number selective: M1 in vacuum or not) | 1 | `slow_pi_ge_readout` |

The MBR sequence fits the same picture (qubit pi/2 and encoding swaps, Floquet, decoding swaps
and pi/2, `'qubit'` readout), but `MBRRamseyProgram` keeps its own body: making it a template
leaf needs generic pre/post pulse lists in the template, a later step.

Rules:
1. `readout` is the only input for new jobs. The Program raises an error if it finds
   `perform_wigner`, `parity_readout`, `multiparity_readout` or `slow_pi_ge_readout` in
   `cfg.expt`, with any value (as `MBRRamseyProgram` does for `decoder_phase_matrix`). No
   translation layer on input. `post_select_pre_pulse` is refused only when true: it is an
   `MM_base` key that the dual-rail and single-qubit experiments play, and shared default dicts
   carry it as false (found in 10D).
2. The readout step plays with or without `postpulse`, as the slow pi always did. Before 10D,
   parity and Wigner with `postpulse=False` played no readout pulse, and nothing reported it.
3. `QsimProgram.readouts_per_shot(cfg)` (classmethod) = herald readouts (active reset, parity
   check) + the final readouts of the mode. The Experiment uses it and does not count by itself.
4. `MM_base.lane_layout(cfg)` is the lane layout of the dual-rail and single-qubit
   experiments (active reset, sigma_z, parity_shot, `post_select_pre_pulse`). It does not
   know `parity_check` or multiparity, and `MM_base` is not changed (0.9), so
   `readouts_per_shot` does not use it. Its active-reset count is the same function
   (`active_reset_read_num`).
5. The old booleans are read in one place only: the readers of saved configs in `qsim_base`.
   `saved_readout_mode` gives the mode that was played: from `readout`, or from the booleans for
   jobs saved before this change, with the old order (`perform_wigner` wins over the parity
   flags; `multiparity_readout` wins over `parity_readout`). `readout_lane_count` gives the
   count the driver asked for: for an old config, the old formula (one more for
   `multiparity_readout`, whatever else is set; with `perform_wigner` too, the program played
   fewer). `mbr_spectrum` and the shot subsampler use it for old jobs.
6. `MBRRamseyProgram` has its own body and its own qubit readout: it rejects any `readout`
   other than the default.
7. New HDF5 files record `readout`.

## 5. Names (proposed; to confirm)

Stem and infrastructure:

| Now | Proposed | Module |
|---|---|---|
| `QsimBaseProgram` + `DarkBaseProgram` | `QsimProgram` | `qsim_base.py` |
| `DarkBaseRProgram` | `QsimRProgram` | `qsim_base.py` |
| `FloquetTrain` (mixin) | `FloquetProgram` (done in 10C) | `floquet_train.py` |
| `DarkModeEncoding` (mixin) | `DarkModeProgram` (done in 10C) | `dark_mode_encoding.py` |
| `ManipulateModePulses` (mixin) | methods move into `QsimProgram` (done in 10B) | module deleted |
| `QsimBaseExperiment` + `DarkBaseExperiment` | `QsimExperiment` | `qsim_base.py` |
| `QsimWignerBaseExperiment` | `WignerExperiment` | `qsim_base_wigner.py` |

Leaves whose names do not say what they do:

| Now | What it does | Proposed |
|---|---|---|
| `SidebandScrambleDarkProgramNewNew` | optional load of the M1 photon into a (dark or normal) mode, Floquet scramble with phase tracking, optional read back | `DarkModeScrambleProgram` |
| `SidebandScrambleDarkProgram` (`t2_cavity_fluxexcursion`, jonginn) | `SidebandScrambleProgram` with its own `man_reset` copy | to decide with jonginn |
| `SidebandStarkAmplificationModifiedProgram` | Stark phase that Floquet pulses on B put on the ds_storage half swaps of A | `StorageSwapStarkPhaseProgram` |
| `SidebandStarkAmplificationModifiedProgram_old` (jonginn `Autocalibrate`, live) | the older version of the above, on Floquet half swaps | to decide with jonginn |

Leaves with names that say what they do keep them (`FloquetChevronProgram`, `DarkT1Program`,
`SidebandStarkAmplificationProgram`, ...).

## 6. Steps

| Step | What | Net |
|---|---|---|
| 10A | Nets first, on the code as it is now (details below) | the nets themselves |
| 10B | `QsimProgram`: merge the two templates and `initialize`; `ManipulateModePulses` methods in; all template leaves and the DarkBase leaves on it | program golden: no change, except the configs in 7.1 as decided |
| 10C | `FloquetProgram` and `DarkModeProgram` as classes in the chain; MBR, DisplacementKerr, StarkModified, DarkT1, NewNew re-parented; NewNew drops `SidebandScrambleProgram` | MBR golden, program golden: no change |
| 10D | the `readout` key; `readouts_per_shot`; old flags raise; the saved-config reader; the consumers on `guan` move to the key (the Program refuses the old flags, so this cannot wait for 10F) | program golden with the test table moved to `readout`: no change |
| 10E | `QsimExperiment` (N axes, Program-owned count); DarkBase, MBRJob, Wigner, SidebandAmpRabi, the loop wrappers | acquire net: same data, except the bug fixes in 7.3 |
| 10F | renames (section 5); consumers migrated (below); notebook dry runs | notebook golden; dry runs as in `docs/status/measurement.md` |
| 10G | device check in local mode: the calibration notebooks, then the MBR notebooks; then the merge | device |

10A, the nets (done):
- **Program golden** (`tests/program_asm_golden.py`, 75 cases): every Program whose base
  changes (the 8 unused ones too), each with a small expt config of its own, compiled on the
  pinned set `preload_current`. Plus a flag matrix of 19 options, played by both template
  bases (`qsim_template__*`, `dark_template__*`). A case marked `raises` pins its error (7
  cases). Keys name what is played, not the class, so a renamed class compares with the old
  one. The table of configs is data, so 10D can move it to `readout`.
- **Notebook golden:** `floquet_displacement_kerr` added to `tests/notebook_asm_golden.py`.
- **Acquire net** (`tests/acquire_golden.py`, 24 cases, one JSON golden): each driver in 3.2
  on the mock station; the real Programs are built, and their `acquire` and `collect_shots`
  return lane-tagged fake shots. It pins the readout count asked of each Program, the sweep
  values each Program saw, and the data keys, shapes and values (a value names its point and
  lane).

10F, the consumers:
- `202609_qsim_migration/` and `guan/`: on `guan`. The readout flags moved in 10D; the renames
  move in 10F.
- `jonginn/`, `connie/`: their latest versions are on `main`. Migrate them on the merge
  result, in one commit, from the rename table and the `readout` table; code cells only (no
  outputs). Tell them before; ask them to restart their kernels after.
- Before the merge: no queued jobs with old class names (the worker loads classes at run time).

## 7. Differences to decide

### 7.1 The two templates (10B)

| Item | `QsimBaseProgram` | `DarkBaseProgram` | Proposal |
|---|---|---|---|
| `init_man_fock_state` | branch if the key is present | branch if the value is not None | DarkBase |
| prepulse `init_stor` swap in the last prepulse branch | flat list `['storage', ...]` (looks like a bug) | nested list `[['storage', ...]]` | DarkBase; check the golden diff |
| displacement setup in `initialize` | on `perform_wigner` | on `perform_wigner` or `init_alpha` | DarkBase (it becomes `readout == 'wigner'` or `init_alpha`) |
| `storage_phase_matrix` in `initialize` | no | read if present | drop, with its only reader `FloquetTrain._advance_storage_phase_offsets`, which nothing calls (MBR rejects the key since 8A4) |

Rule: take the DarkBase behavior (the newer one). Where the golden changes, list the configs
in the log, and check them on the device in 10G.

Measured in 10A (the program golden plays both templates under 19 options): the compiled
programs are the same for 13 options, including active reset (so 7.2 holds). They differ for
6, and in each one QsimBase lacks a DarkBase feature: `multiparity_readout` and
`slow_pi_ge_readout` (QsimBase ignores them), `init_man_fock_state` above 1 and
`do_crude_comp` (QsimBase uses `MM_base.prep_man_fock_state`, which knows only 0, 1 and the
superpositions, and raises), `init_alpha` without Wigner (QsimBase does not set up the
displacement, and raises). So the 10B diff is expected in these 6 `qsim_template__*` cases
only, and each must become equal to its `dark_template__*` case.

### 7.2 `man_reset`

`ManipulateModePulses.man_reset` is `MM_base.man_reset` plus `dump_reset_iter_num` (default 1:
the same pulses) and a debug print. The MBR and DarkBase programs play it; the ~25 QsimBase
leaves play `MM_base`'s.

Decision (0.9): `MM_base` is not changed. The copy moves into `QsimProgram` and overrides
`MM_base.man_reset` for every qsim program. Moving `dump_reset_iter_num` into `MM_base` was
rejected: non-qsim experiments get the key too (`SidebandGeneralExperiment` in
`multiphoton_calibration.py`), so its meaning would change outside qsim.

Effect on the ~25 QsimBase leaves: every live config sets `dump_reset_iter_num=1`
(`notebook_helpers/defaults.py`, `mbr_campaign`, the calibration notebooks), which gives the
same pulses, so the program golden does not change. With a value above 1 they now repeat the
dump pulses, as the DarkBase and MBR programs already do.

### 7.3 Bug fixes the driver merge makes (10E)

- `QsimBaseExperiment` + `multiparity_readout`: wrong lane now; right after.
- `post_select_pre_pulse` in Wigner: refused since 10D (rule 4.1); no live Program plays it.
  Its acquire-golden case was removed in 10D; `tests/test_readout_key.py` checks the refusal.
- `SidebandAmpRabiExperiment` parity lane with active reset: right after.
- `ManStorMultiparityChevronRExperiment` + `parity_check`: right after.
- `SidebandAmpRabiExperiment` does not default `perform_wigner`, which the template reads, so
  it raises unless the config sets it (found in 10A). The one driver sets the default.

## 8. Open questions

1. Closed (decision 0.7).
2. Closed (decisions 0.8, 0.9).
3. To review at 10E (decision 0.10). Does `FloquetCalibrationAmplificationExperiment` become a plain 3-axis sweep in 10E, or does
   it stay a loop wrapper? (Same data either way; the first is less code.)

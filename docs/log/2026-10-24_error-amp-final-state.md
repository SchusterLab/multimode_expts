# 2026-10-24: error amplification gets its readout state from the pulse type (user + Claude)

## Why

The user saw that the vault plots of the 10-24 night checks (`guan/KerrCheck/2026/10/2026-10-24`,
00:00-00:06, JOB-20261023-00534..00543) had wrong fits, while the notebook fits were right.
Cause: the worker runs `go(analyze=True)`, so `ErrorAmplificationExperiment.analyze()` ran with
its defaults `state_fin='g'`, `periodic=True`. That fit went into the HDF5 files (`fit_avgi`,
`prod_avgi_fit`), and the runner drew the vault figure from it in `_collect`, before the
notebook (`kerr_n123_consistency.py` cell 4) fitted again with `state_fin='e'`. `display()`
does not fit, so a display option would not have helped. Example, f2-g3 (00534): default fit
1985.7528 MHz, `state_fin='e'` 1986.0436 MHz.

## What was done

- `experiments/single_qubit/error_amplification.py`: new `expected_final_state(expt_cfg)`. The
  tested pulses cancel when the pulse is right, so the readout state is what the pre and post
  pulses leave: qubit (ge, ef) g; man, storage, floquet e; multiphoton starting in g or e: g;
  multiphoton fN-gN+1: e. A non-empty `pre_sweep_pulse` or `post_sweep_pulse` gives None.
- `analyze(state_fin=None, periodic=None)`: argument, else `cfg.expt.state_fin`, else the rule;
  `periodic` argument, else `cfg.expt.periodic`, else False. If the state is unknown, a
  `UserWarning` and no product or fit; `analyze` now returns `data`. `display()` skips the
  product plot when there is no product.
- `tests/test_error_amplification.py` (19 tests, synthetic data).
- Not changed: the callers. Those that pass `state_fin` (autocal, floquet and multiphoton
  notebooks, `kerr_n123_consistency.py`) pass the value the rule gives.

## Decisions

- User: derive the state from the pulse type; an explicit `cfg.expt.state_fin` wins.
- The user considered one final state for all sequences. Not done: each sequence ends in the
  state that puts the leaked population in the other readout state. All-g would lose the
  contrast for f2-g3 (leak |g3>, a g2-e2 pulse does not move it) and for man; all-e would make
  ef and eN-fN read e against f.
- User: custom sweep pulses with no `state_fin` warn and skip the fit. Not raise: slab's `go()`
  runs `analyze()` before `save_data()`, so a raise in the worker would lose the data.
  Autocal ladder and storage error-amp jobs have custom sweep pulses: their worker-side fit is
  now skipped (before it used 'g': right for the qubit rungs, wrong for the sidebands and
  swaps); autocal fits again with an explicit state.

## Found

- Refit of 00534..00543 with the new defaults gives the notebook values (f2-g3 1986.0436;
  swaps S4..S7 872.3830 / 1048.5759 / 1240.3149 / 1418.3062; Floquet S4 871.9335); broadband
  unchanged (3563.6967). The HDF5 files of that night keep the old wrong `fit_avgi`; raw `avgi`
  is good.
- Old, not fixed: the Floquet runners set `qubit_start_storage`, but the program reads
  `qubit_state_start` (`error_amplification.py`, storage/floquet branch). No effect for 'g' (the
  default); a user who sets 'e' is ignored. The final state is e in both cases.
- 6 tests fail with and without this change (`test_agp_disorder_scan`, `test_agp_experiment`,
  `test_matrix_pencil_regression`); 1541 pass.
- The edit is in the `main` checkout, uncommitted. The worker clears its module cache per job,
  so it uses the new `analyze` from the next job on.

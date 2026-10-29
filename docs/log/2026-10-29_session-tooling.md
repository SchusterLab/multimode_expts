# 2026-10-29: tooling for shared-kernel sessions (spec items A-D)

guan + Claude. Spec: `docs/lab_workflow_tooling_spec.md`. guan tests the vault schema in
Obsidian and changes it if needed.

## Done

- **A. `tools/kernel_exec.py`:** `list`, `start`, `cells`, `run` (`--script/--cell`,
  `--file`, `--code`, `--timeout`, `--no-wait`), `interrupt`. REST for list, start and
  interrupt; `jupyter_client` on `kernel-<id>.json` for run. The server is the newest
  `jpserver-*.json` for the port whose token the server accepts: the runtime dir has 9 stale
  files for port 8765 from crashes, with roots `C:\python`, `C:\python\multimode_expts` and
  `C:\python\multimode_expts_guan`; the live one (2026-10-29) has root `C:\python`. Log:
  `.cache/kernel_exec/<kernel>.log`, images next to it. A script cell run sets
  `__kernel_exec_source__` in the kernel while it runs (read by `log_analysis`).
  `--no-wait` starts a detached process that sends the code and writes the log.
- **B. `experiments/vault.py`, vault schema 2:** schema is guan's decision (seb is leaving):
  "anything is fine, iterate after use". Each entry: an id (job ID, or `ANA-`/`ENT-` + date and
  time, `-2` etc. if not unique), a line with the id and an Obsidian block id (`^<id>`), `kind`
  and `entry_id` in the parameters YAML, notes between `<!-- notes:<id> -->` and
  `<!-- /notes:<id> -->`. Front matter of a new day file has `vault_schema: 2`. `add_note`,
  `withdraw_note`: temp file in the same folder + `os.replace`, again if the file changed or is
  open (Windows `PermissionError`). guan: the remaining ms gap need not be closed; raw data is in
  HDF5.
- `station.log_measurement` writes schema 2 (new kwargs `kind`, `entry_id`, `body`, `author`,
  `log_in_mock`); `station.last_vault_entry` = (day file, entry id). `station.add_note(entry_id,
  text, author=None)` finds the day file from the date in the id, in any project folder of the
  user.
- **C. `station.log_analysis(fig, title, job_ids, note, parameters, data_files, source,
  project, author)`:** an `analysis` entry with the job range, links to the job entries
  (`[[<user>/<project>/YYYY/MM/<day>#^<job>|00272]]`; to the day file only for entries without
  anchors), the source, figures, parameters (with `job_ids`), notes. Logs in mock mode too.
- **D. `experiments/board_resources.py` + `MockQickSoc(soccfg)`:** the mock fails where the board
  drivers fail: envelope memory per generator (`load_pulse_data`, with the list of envelopes on
  the channel), mux tones, `config_buf`, accumulated/decimated transfer lengths, tProc v1 pmem.
  The station gives the mock its `soccfg`. `check_program_resources(prog, soccfg)` does the
  program-level checks. No runner preflight (guan).
- Tests: `tests/test_kernel_exec.py` (5; 4 of them start a throwaway kernel on the live server,
  skipped if none), `tests/test_vault.py` (8), `tests/test_board_resources.py` (10). The 10-28
  case in the mock: probe sigma 1.2 us fails alone (as JOB-20261028-00775), 1.05 us fails only
  after the Fock 1 prefix (as -00780), 0.8 us passes with and without it.

## Found

- Full suite: 1552 passed, 6 failed, and the same 6 fail at HEAD without these changes:
  `test_agp_disorder_scan` (1), `test_agp_experiment` (2) (saved `agp`/`agp_list` keys missing),
  `test_matrix_pencil_regression` (3, known).
- `measurement_notebooks/guan/kerr_n123_consistency.py` `vault_note` finds notes by
  `### Notes` ... `---`; on schema 2 entries it would write after the closing anchor. Use
  `station.add_note` there.

## Open

- Check by hand: `os.replace` on `G:` with the Drive client. Does Obsidian on another machine
  see an edit, or a delete and a create? (Drive version history of the day file shows it.)
- Item E (one entry per job group) not started.

# 2026-10-07: the config-version archive was deleted through a worktree junction

Session with guan (gzhwang), on pippin. Branch `guan`.

## What happened

On 2026-10-06, shortly after 23:03, the agent removed the retired `qsim-analysis` worktree with
`git worktree remove --force C:\python\multimode_expts_qsim-analysis`. That worktree had
`configs\versions` as a junction to main's live archive `C:\python\multimode_expts\configs\versions`
(set up 2026-09-29, `docs/log/2026-09-29_worktree-split.md`). Git deleted the ignored contents
through the junction and emptied the archive.

The agent had checked the worktree for real work first, but with Git Bash `find`, which does not
enter junctions: it reported 0 files, and the agent took that as an empty folder instead of
checking `LinkType`. The removal of the `jonginn` worktree the same evening was not involved: its
`configs/versions` was a real folder with its own 7 files.

## Recovery

- guan restored the archive from the daily Synology backup on 2026-10-07 (about 13:31).
- Check against `jobs.db` (`config_versions`, read-only): 18,662 of 18,671 versions present.
  Missing: 9 hardware-config snapshots `CFG-HW-20261006-00108` .. `-00116`, written 22:03-23:03 on
  2026-10-06 by `JOB-20261006-00213` .. `-00229`, i.e. after the backup and before the deletion.
- They are rebuilt from each creating job's HDF5 `config` attribute: the worker writes a snapshot
  as `yaml.dump(station.hardware_cfg, default_flow_style=False, sort_keys=False,
  allow_unicode=True)` and records its SHA-256. The rebuild reproduces three existing snapshots
  (`-00105` .. `-00107`) byte for byte, and each of the 9 matches its recorded checksum.

## Prevention

`AGENTS.md`, new section "Worktrees on prod": remove the link with `cmd /c rmdir` (no `/s`) before
removing a worktree; never `git worktree remove --force`, `git clean -x`, `rm -rf` or
`Remove-Item -Recurse` while the link exists; check `LinkType` before any recursive delete.
guan chose `AGENTS.md` over agent memory, because memory does not travel to new worktrees.
The `guan` worktree (junction) and `tests-cleanup` (symbolic link) still carry such links.

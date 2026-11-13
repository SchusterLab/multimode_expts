# Worker jobs from worktrees, and the flat `experiments` namespace: findings

guan asked on 2026-11-12: (1) should the worker also run jobs from worktrees other than `main`;
(2) what to do about the import hack in `experiments/__init__.py`. This entry records what we
found and what we propose. No code was changed. No decision is made yet: see "Open
decisions" in each part. To-do items: `README.md`, "TODO".

## 1. Worker: jobs from worktrees

### How the worker is tied to `main` now

- The worker makes one station at start-up and keeps the hardware connections open
  (`job_server/worker.py:205`: Pyro name server for QICK, two yoko sockets).
- For each job it removes `experiments.*`, `fitting.*` and `theory.*` from `sys.modules` and
  imports them again. It keeps `experiments`, `experiments.dataset` and `experiments.station`
  (`job_server/worker.py:455`).
- The import goes through the pixi editable-install finder
  (`.pixi/envs/default/Lib/site-packages/__editable___multimode_expts_0_0_1_finder.py`), which
  maps each package to `C:\python\multimode_expts\...`. So the worker finds only `main`'s code.
- Each worktree has its own pixi env, with its own finder that maps to that worktree (checked
  for `autocal`, `guan`, `theory`, `transduction`).
- `jobs.db` and `configs\versions` are already shared by links (`job_server/database.py:63`).

### Problems that are there also without worktrees

1. The worker runs the files that are on disk in `main` when the job starts, also half-edited
   ones.
2. The HDF5 file gets the config version IDs (`config_versions` attribute,
   `job_server/worker.py:640`), but no code version. Only assembled data sets record `git HEAD`
   (`experiments/assembled_data.py:49`).
3. A change to `station.py` or `dataset.py` needs a worker restart. A partial reload can mix
   old and new code in one process.

### Options

| Option | How | Verdict |
|---|---|---|
| A. Switch the code in the worker process | Change the import path per job | No. The finder maps to `main`; station, `slab` and `job_server` stay from `main`; two code versions in one process. |
| B. One child process per job | The worker starts `<worktree>\.pixi\envs\default\python.exe` for the job | **Proposed.** One consistent code tree per job; a crash does not stop the worker. |
| C. Pin a commit per job | Clean checkout per commit | Not now: a commit before every submit, and one env per checkout. Possible later as a strict mode. |

### Option B: proposed design

- **Parent (always `main` code):** queue, all writes to `jobs.db`, config snapshots, output
  log capture, cancel. It holds no hardware connection.
- **Child (the job's worktree and env):** station, hardware connections, `go()`, HDF5 file,
  pickle. It writes its final configs to a JSON file; the parent snapshots them. So a branch
  with a changed `job_server/models.py` cannot write a wrong schema into the shared database.
- **New job field `code_root`.** Default: the checkout of the `experiments` package that the
  submitting kernel imported, so the worker runs the code that was tested in mock.
- **Code version in the HDF5 file:** path, branch, commit, a "dirty" flag (uncommitted
  changes) and a hash of the diff.
- **Cancel** stops the child process (today: `KeyboardInterrupt` in the worker process).

### Costs and risks

- **Start-up per job:** a new process takes 4.6 s to import the station and the experiment
  packages (measured 2026-11-12, warm, on pippin). 1.5 s of this is the import hack (part 2).
  The hardware connect time was not measured. Small for a 2-minute job; about +50% for many
  10-second jobs. If needed: keep one child alive per `code_root` for back-to-back jobs.
- **Safety code moves with the branch.** Some experiments ramp the yoko themselves
  (`experiments/single_qubit/pulse_probe_f0g1_spectroscopy.py`,
  `experiments/sequential_experiment.py`), so a branch can change ramp rates or limits.
  Proposed: the parent checks the coupler set point before it starts the child, and accepts
  only worktrees that `git worktree list` shows.
- **Worktree env not current:** after a dependency change, the worktree needs `pixi install`.
  The parent fails the job with a clear message if the env is missing.
- **Child stopped during an acquire:** not known yet what the QICK board does then. Check this.
- **Docs to change:** the rule "the worker runs only `main`" is in `AGENTS.md`,
  `docs/STATUS.md` and `docs/running_experiments.md` (item 1).

### Open decisions

1. Allow uncommitted code in the worktree? Proposed: yes, with the dirty flag and the diff
   hash in the HDF5 file. A "clean tree only" flag can come later.
2. Default `code_root` = the submitting kernel's checkout? Proposed: yes.
3. A new child for each job, or one kept alive per worktree? Proposed: a new child per job
   first; measure the cost.

## 2. The flat `experiments` namespace

### What the hack does

When the package loads, `experiments/__init__.py` imports every module of every subpackage
(not names that start with `_`) and copies every class it finds into the `experiments`
namespace. Notebooks use it as `import experiments as meas; meas.HistogramExperiment`.

### What we measured (2026-11-12, `main`)

- **351 flat names:** 311 classes from the subpackages, 40 from outside the repo (`Path`,
  `AttrDict`, `QickConfig`, `AveragerProgram`, fitting classes, ...).
- **15 class names are defined in more than one module.** The flat name gets the module that
  loads last, with no warning. On Windows the order is alphabetical; on other file systems
  `os.listdir` gives no fixed order, so another machine can get another class.

  | Name | Modules (flat name gets the last one) |
  |---|---|
  | `AmplitudeRabiChevronExperiment` | `qubit_cavity.amplitude_rabi_f0g1`, `single_qubit.amplitude_rabi`, **`single_qubit.amplitude_rabi_general`** |
  | `CavityRamseyExperiment`, `CavityRamseyProgram` | `single_qubit.t2_cavity`, **`single_qubit.t2_cavity_mod`** |
  | `ECDExperiment`, `ECDProgram` | `qubit_cavity.ecd`, **`qubit_cavity.ecd_backup`** |
  | `PulseProbeEFPowerSweepSpectroscopyExperiment` | `single_qubit.pulse_probe_ef_spectroscopy`, **`single_qubit.pulse_probe_f0g1_spectroscopy`** |
  | `PulseProbeSpectroscopyExperiment`, `PulseProbeSpectroscopyProgram` | `qubit_cavity.displace_pulse_probe`, `single_qubit.pulse_cavity_probe_readout`, **`single_qubit.pulse_probe_spectroscopy`** |
  | `PulseProbeVoltSweepSpectroscopyExperiment` | `qubit_cavity.displace_pulse_probe`, `qubit_cavity.ecd`, `qubit_cavity.ecd_backup`, `single_qubit.pulse_cavity_probe_readout`, `single_qubit.pulse_probe_spectroscopy`, **`single_qubit.pulse_probe_spectroscopy_prepulse`** |
  | `ResonatorPowerSweepSpectroscopyExperiment`, `ResonatorVoltSweepSpectroscopyExperiment` | `qubit_cavity.ecd_read_spec`, `single_qubit.flux_sweep`, **`single_qubit.resonator_spectroscopy`** |
  | `SingleBeamSplitterRBPostSelection`, `SingleBeamSplitterRBPostselectionrun` | `single_qubit.rb_BSgate_postselection`, **`single_qubit.rb_BSgate_postselection_storage`** |
  | `TestOptExperiment`, `TestOptProgram` | `qubit_cavity.test_opt`, **`qubit_cavity.test_opt_ecd`** |

- **Use:** 726 places in 77 files, 73 distinct names (attribute `meas.X` / `experiments.X`, or
  `from experiments import X`; `.ipynb_checkpoints`, `.tmp`, `.cache`, `.suite_runs` not
  counted). Almost all are in notebooks: `measurement_notebooks/jonginn` 242,
  `measurement_notebooks/guan` 162, `measurement_notebooks/QEC` 122,
  `measurement_notebooks/connie` 96, others below 35.
- **Ambiguous names in use:** `PulseProbeSpectroscopyExperiment` (14 places),
  `CavityRamseyExperiment` (2), `TestOptExperiment` (1). Foreign name in use: `ConfigType` (1).
- **Possible live bug, to check:** `measurement_notebooks/guan/multiphoton_calibration_v2 guan`
  (`.ipynb` and `.py`) uses `meas.CavityRamseyExperiment` and gets the class from
  `t2_cavity_mod.py`, not `t2_cavity.py`.
- **One broken file breaks all imports.** An error in any subpackage file stops
  `import experiments.station`, and so every kernel, test and worker start.
- **Import time:** 4.6 s with the hack, 3.0 s without it (station, both runners and two
  experiment modules, with a stub `experiments` package that skips the init).
- **Workaround in use:** `experiments/autocal` names its modules `_core`, `_qubit`, ... only to
  hide them from the hack (see its docstring).
- **Library code that uses flat names:** `experiments/sequential_experiment.py` and
  `experiments/sequential_experiment_classes.py` (`import experiments as meas` at the top).
  The other hits in `experiments/` are docstring examples.

### Interaction with part 1

Today the worker runs the package init only once, at start-up, so a broken unrelated file does
not fail the next job. With one child process per job (option B), every job runs the init:
one broken file in any subpackage would fail every job. So part 2 should come before, or with,
part 1.

### Proposal: keep the short names, find them lazily

The package defines a module-level `__getattr__` (PEP 562). The init only reads the source text
and imports nothing:

```python
# experiments/__init__.py (sketch)
import importlib, re
from pathlib import Path

_CLASS = re.compile(r"^class\s+([A-Za-z_]\w*)", re.MULTILINE)

def _scan():
    names = {}
    root = Path(__file__).parent
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and p.name[0] not in "_."):
        for f in sorted(sub.glob("*.py")):
            if f.name[0] not in "_.":
                mod = f"experiments.{sub.name}.{f.stem}"
                for name in _CLASS.findall(f.read_text(encoding="utf-8")):
                    names.setdefault(name, []).append(mod)
    return names

_NAMES = _scan()   # plus the explicit ones: MultimodeStation, the runners

def __getattr__(name):
    mods = _NAMES.get(name)
    if not mods:
        raise AttributeError(f"module 'experiments' has no attribute {name!r}")
    if len(mods) > 1:
        raise AttributeError(f"{name} is defined in {mods}: import it from one of them")
    return getattr(importlib.import_module(mods[0]), name)

def __dir__():          # tab completion
    return sorted(set(globals()) | set(_NAMES))
```

- `meas.X` and `from experiments import X` still work; 709 of the 726 places need no change.
- Only the module that defines the class is imported, at first use. A broken file affects
  only its own classes.
- An ambiguous name gives an error that lists the modules.
- Only classes defined in the repo get flat names.
- No class is stored on the package, so after the worker reloads a module, `meas.X` gives the
  new class.

### Work that comes with it

- Fix the 17 places with ambiguous names and the one `meas.ConfigType`, or first delete the
  old copies (`ecd_backup.py`, `t2_cavity_mod.py`, `test_opt*.py`, the
  `rb_BSgate_postselection*` pair). Their owners decide.
- Rule: library code (`experiments/`, `fitting/`, `tools/`, `job_server/`) uses full module
  paths; flat names are for notebooks. Fix the two `sequential_experiment*` files.
- Tests: one that fails when a new duplicate class name appears; one that fails when library
  code uses a flat name.

### Options not proposed

- Remove the flat names and use full paths: 726 places, mostly in other people's notebooks,
  and long paths.
- Re-export per subpackage (`from experiments.single_qubit import X`): still a rewrite, plus
  name lists to keep current.
- A checked-in, generated name table: more exact than the text scan, but one more file to keep
  current.

### Open decisions

1. Ambiguous name: error at use (proposed), or today's choice plus a warning?
2. Delete the old duplicate files? For their owners.

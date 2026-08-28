# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

# %%
import json
import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from slab import AttrDict
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment as ManyBodyRamsey,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment

# This file is the migration exemplar. The god Experiment is being split one
# analyze(stage=...) branch at a time into an Experiment per stage, and this
# notebook moves to each new class as it lands, so there is always one worked
# example of the current API to copy from.
#
# Migrated so far: calibration -> MBRPhaseCorrectionExperiment.
# Still on the stage= facade: spectrum. It moves when that stage does.

# Where this machine sees the two shared trees. Configs record Windows paths
# (output_root C:, vault_root G:), so off-prod every reader needs a mapping.
DATA_ROOT = Path(os.environ.get("MULTIMODE_DATA_ROOT", "/Volumes/experiments"))
VAULT_ROOT = Path(os.environ.get(
    "MULTIMODE_VAULT_ROOT",
    Path.home() / "Google Drive/Shared drives/SLab/Multimode",
))
VAULT_USER = "Jonginn"

# Floquet timing as compiled at acquisition. Not in the H5 (JSON serialization
# drops device.storage._ds_floquet) and not in the vault YAML. Recovered once
# from JOB-20260815-00009_expt.pkl -- see the appendix. Letting the station fill
# these in instead reads today's swap CSV, which is 43% off for this data.
TIMING = dict(floquet_cycle_us=0.7340315934065934, m1s_pi_fracs=[40] * 7)


def job_ids(date, first, last):
    return [f"JOB-{date}-{n:05d}" for n in range(first, last + 1)]


def vault_paths(ids):
    """job_id -> local h5 path, read from the lab-notebook YAML.

    `station.log_measurement` records a full `data_path` per run, which is the
    only place the data subdirectory is written down: it is the station's
    `experiment_name` at acquisition time and tracks neither the job date nor
    the vault project. These jobs span 260814_qsim_encspec and
    260526_qsim_darkmode, logged under EncSpec and DarkModeReadout, so glob
    across projects rather than naming one.
    """
    dates = {f"{j[4:8]}-{j[8:10]}-{j[10:12]}" for j in ids}
    found = {}
    for date in sorted(dates):
        year, month, _ = date.split("-")
        for md in VAULT_ROOT.glob(
            f"Lab/{VAULT_USER}/*/{year}/{month}/{date}.md"
        ):
            for match in re.finditer(r"data_path:\s*(.+\.h5)", md.read_text(errors="replace")):
                recorded = match.group(1).strip()
                job = re.search(r"JOB-\d{8}-\d{5}", recorded)
                if job:
                    tail = recorded.replace("\\", "/").split("experiments/", 1)[1]
                    found[job.group(0)] = DATA_ROOT / tail
    missing = [j for j in ids if j not in found]
    if missing:
        raise KeyError(f"{len(missing)} jobs absent from the vault: {missing[:5]}")
    return {j: found[j] for j in ids}


def load_h5(path, load_shots=False):
    """-> (cfg, data). Skips idata/qdata unless asked; they are ~99% of the file."""
    with h5py.File(path, "r") as handle:
        cfg = AttrDict(json.loads(handle.attrs["config"]))
        skip = () if load_shots else ("idata", "qdata")
        data = AttrDict({k: handle[k][()] for k in handle if k not in skip})
    return cfg, data


class _SavedJob:
    """Duck-types the pickled experiment for what the analysis path reads."""

    def __init__(self, job_id, cfg, data, path, prog):
        self.job_id, self.cfg, self.data = job_id, cfg, data
        self.fname = str(path)
        self.prog = prog


class _SavedProgram:
    """Supplies TIMING where `_saved_parameters` looks for a compiled program."""

    def __init__(self, floquet_cycle_us, m1s_pi_fracs):
        self._cycle_us = float(floquet_cycle_us)
        self.m1s_pi_fracs = list(m1s_pi_fracs)

    def calculate_floquet_cycle_us(self):
        return self._cycle_us


def load(ids, owner=ManyBodyRamsey, load_shots=False):
    """Wrap saved H5 jobs in the Experiment that analyzes them.

    `owner` is the stage class: the loading layer is inherited, so
    `_from_expts` returns an instance of whichever class is asked.
    """
    ids = list(ids)
    paths = vault_paths(ids)
    prog = _SavedProgram(**TIMING)
    jobs = [_SavedJob(j, *load_h5(paths[j], load_shots), paths[j], prog) for j in ids]
    return owner._from_expts(jobs, job_ids=ids)


# %% [markdown]
# ## 1. Select

# %%
SPECTROSCOPY_IDS = job_ids(20260815, 9, 16)
CALIBRATION_IDS = None

CYCLE_BRANCHES = {
    (3, 0, 0, 0, 0): 1,
    (2, 0, 0, 1, 0): 1,
}


# %% [markdown]
# ## 2. Load

# %%
expt = load(SPECTROSCOPY_IDS)
calibration = (load(CALIBRATION_IDS, MBRPhaseCorrectionExperiment)
               if CALIBRATION_IDS else None)
if calibration is not None:
    calibration.analyze()          # no stage argument: the class is the stage
    calibration.display()


# %% [markdown]
# ## 3. Analyze

# %%
data = expt.analyze(
    stage="spectrum",
    calibration=calibration,
    cycle_branches=CYCLE_BRANCHES,
    fft_window="raw",
    zero_padding=1,
    spectrum_method="fft",
)
assert expt.data.hardware.source == "saved program", expt.data.hardware.source


# %% [markdown]
# ## 4. Display

# %%
expt.display(data=data)
plt.show()


# %% [markdown]
# ## Appendix: re-derive TIMING
#
# Run only to regenerate the constants above. Reads one pickle, which is why it
# is quarantined here.

# %%
def recover_timing(job_id):
    import pathlib as _pathlib
    import pickle

    h5_path = vault_paths([job_id])[job_id]
    pkl = h5_path.parent.parent / "expt_objs" / f"{job_id}_expt.pkl"
    saved_flavour = _pathlib.WindowsPath
    _pathlib.WindowsPath = _pathlib.PureWindowsPath  # WindowsPath will not instantiate here
    try:
        with open(pkl, "rb") as handle:
            prog = pickle.load(handle).prog
    finally:
        _pathlib.WindowsPath = saved_flavour
    return dict(floquet_cycle_us=float(prog.calculate_floquet_cycle_us()),
                m1s_pi_fracs=[int(v) for v in prog.m1s_pi_fracs])


# print(recover_timing(SPECTROSCOPY_IDS[0]))

# %%

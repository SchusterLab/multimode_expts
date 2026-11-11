# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (pixi env of cwd)
#     language: python
#     name: pixi-cwd
# ---

# %% [markdown]
# # AGP remeasure (jonginn, 2026-10-29): to-do items
#
# The to-do items are in the vault note
# `Jonginn/000000 Separate Lab Notebook/20261001 AGP/20261029 AGP remeasure.md`.
# Shared kernel: session `agp_remeasure` (kernelspec `pixi-cwd`, theory worktree).
#
# Since the cleanup of 2026-11-03 (last section), the data sets of the two experiment folders
# have names (`CentralBosonAGPDisorderScan_<name>`); the old time-stamp files are in
# `assembled_data_old_20261103`.

# %%
import re
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from experiments.assembled_data import read_assembled_h5, read_manifest
from experiments.job_paths import JOB_ID_RE, data_root
from experiments.qsim.agp import (
    CentralBosonAGPDisorderScan,
    list_scans,
    load_scan,
    load_scans,
    registered_scans,
    rms_khz,
)
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationSetExperiment

FOLDER = Path(r"C:\experiments\261027AGPlowKerr")
K44 = Path(r"C:\experiments\agp_k44_first_scan")
FOLDERS = [K44, FOLDER]
OLD = "assembled_data_old_20261103"
MU = [5., 10., 20.]     # kHz, for the plots
BACKGROUND_US = 30.
NOTE = Path(r"G:\Shared drives\SLab\Multimode\Lab\Jonginn\000000 Separate Lab Notebook"
            r"\20261001 AGP\20261029 AGP remeasure.md")
VAULT_IMAGES = NOTE.parent / "images"
warnings.filterwarnings("ignore", message="Calibration file not found")


def manifest_jobs(path):
    """-> the job IDs a manifest names, itself or through its parts' manifests."""
    manifest = read_manifest(path)
    ids = set(manifest.get("job_ids") or [])
    for raw in manifest["raw_files"]:
        match = JOB_ID_RE.search(Path(raw).name)
        if match:
            ids.add(match.group(0))
        elif str(raw).endswith(".yaml"):
            ids |= manifest_jobs(Path(path).parent / Path(raw).name)
    return ids


# %% [markdown]
# ## A2: the 120 and 140 kHz data. What is assembled, and where is the gap
#
# Step 1: the scan manifests in the folder.

# %%
list_scans(FOLDER)

# %% [markdown]
# Step 2: the registered retake `s7_high_w`.

# %%
high_w = load_scan("s7_high_w")
print(high_w.summary())

# %% [markdown]
# Step 3: the population jobs of 10-29 and 10-30 that no scan uses.

# %%
used = {job_id for part in high_w.children for job_id in part.job_ids}
rows = []
for path in sorted((FOLDER / "data").glob("JOB-2026103[0]-*_FockPopulationExperiment.h5")) + \
        sorted((FOLDER / "data").glob("JOB-20261029-*_FockPopulationExperiment.h5")):
    job_id = path.name.split("_")[0]
    if job_id in used:
        continue
    expt = FockPopulationExperiment.from_h5file(path, load_shots=False).cfg.expt
    cycles = np.asarray(expt.floquet_cycles)
    rows.append(dict(job=job_id, ro_stor=expt.ro_stor, occupations=list(expt.occupations),
                     swap_stors=list(expt.swap_stors), W_khz=round(rms_khz(expt.detunings), 3),
                     cycles=f"{cycles[0]}-{cycles[-1]} ({len(cycles)})"))
pd.DataFrame(rows).sort_values("job")

# %% [markdown]
# Result (2026-11-03): 120 and 140 kHz are assembled in `s7_high_w`. 120 kHz is
# JOB-20261029-00393..00407. 140 kHz is 00408..00453. (The vault note listed the 140 kHz jobs
# under both headings; corrected there.) 00388-00392 are the 3-photon check at W = 0, not AGP
# jobs. Gap: W = 160 kHz has 3 of 15 jobs (00458, 00463, JOB-20261030-00002), because the batch
# was paused there. No new measurement for now (guan).

# %% [markdown]
# ## A4: plot all assembled data sets again, without clipping the spectrum
#
# The daily notebook plotted spectra to 250 kHz (`agp_scans.py`), `labbook/browse.py` to 120 kHz.
# Step 1: for each scan, the Nyquist frequency of its time step, and where the closed-system
# spectral weight of C_1 is: the highest frequency with weight, and the fraction above 250 kHz
# and above Nyquist (that part folds back into the measured spectrum).

# %%
scans = load_scans()
for scan in scans.values():
    scan.analyze(background_start_us=BACKGROUND_US)

rows = []
for key, scan in scans.items():
    times = np.asarray(scan.data["time_us"])
    nyquist_khz = 1000 / (2 * np.median(np.diff(times)))
    for part, (f_khz, weights) in zip(scan.children, scan.model_spectra_khz()):
        rows.append(dict(scan=key, W_khz=part.disorder_khz, dt_us=round(float(np.median(np.diff(times))), 3),
                         nyquist_khz=round(nyquist_khz, 1),
                         f_max_khz=round(float(f_khz[weights > 1e-3 * weights.max()].max()), 1),
                         above_250=round(float(weights[f_khz > 250].sum() / weights.sum()), 3),
                         above_nyquist=round(float(weights[f_khz > nyquist_khz].sum() / weights.sum()), 3)))
spectrum_table = pd.DataFrame(rows)
spectrum_table

# %% [markdown]
# Step 2: the spectra of every scan (default range: 1.3 x the highest closed-system frequency,
# at most Nyquist).

# %%
for key, scan in scans.items():
    fig = scan.display_spectrum(mu=[10., 20.])
    fig.suptitle(f"{key}: " + fig._suptitle.get_text(), fontsize=10)

# %% [markdown]
# Result (2026-11-03): `display_spectrum` now plots to 1.3 x the highest closed-system frequency
# (at most Nyquist) and marks the weight above Nyquist. The 250 kHz limit cut 88 % of the
# closed-system weight of `s7_high_w` at W = 200 kHz. `s7_coarse` at W = 150 kHz: 1.4 % folds back.
# The plots are exported to the vault at the end of this notebook.

# %% [markdown]
# ## The ultrafine S7 scan: its W label
#
# `261102_161533` labels its second part W = 20 kHz, but the detunings have an RMS of 30 kHz
# (jonginn's note: "W=0, 30"). Assembled from the same 30 job files, W is the RMS (30 kHz) and
# the data are the same. That scan was saved on 11-03 (`261103_160925`) and is now
# `s7_ultrafine` (cleanup below).

# %%
ultrafine_old = load_scan(K44 / OLD / "261102_161533_CentralBosonAGPDisorderScan.yaml")
job_ids = sorted(job_id for part in ultrafine_old.children for job_id in part.job_ids)
paths = [next((K44 / "data").glob(f"{job_id}_*.h5")) for job_id in job_ids]
ultrafine_jobs = CentralBosonAGPDisorderScan.from_job_files(paths, direction=7)
print(len(paths), "jobs;", [part.disorder_khz for part in ultrafine_old.children], "kHz labelled,",
      [part.disorder_khz for part in ultrafine_jobs.children], "kHz from the detunings")
for scan in (ultrafine_old, ultrafine_jobs):
    scan.analyze(mu=MU, background_start_us=BACKGROUND_US)
{key: bool(np.allclose(ultrafine_old.data[key], ultrafine_jobs.data[key], equal_nan=True))
 for key in ("C1", "C1_err", "C1_theory", "agp", "agp_connected", "spectrum")}

# %% [markdown]
# ## The 3-photon walk of 10-29 (JOB-20261029-00388..00392)
#
# jonginn's check before the S7 retake ("multiparity is too low for photon 3 ... run quantum walk
# with three photon"): [3,0,0,0,0], storages [4,5,6,7], read M1 and S4-S7, 50 points to 98 cycles.
# Saved on 11-03 (`261103_164201`), now `walk_n3_20261029` (cleanup below).

# %%
WALK_JOBS = [f"JOB-20261029-{n:05d}" for n in range(388, 393)]
walk_jobs = [FockPopulationExperiment.from_h5file(next((FOLDER / "data").glob(f"{job_id}_*.h5")),
                                                  load_shots=False) for job_id in WALK_JOBS]
walk = FockPopulationSetExperiment.from_children(walk_jobs, job_ids=WALK_JOBS)
walk.analyze()
walk_fig = walk.display()

# %% [markdown]
# ## Audit: is every data set assembled, registered and in the vault table?
#
# Step 1: every scan manifest in the two `assembled_data` folders. `registry`: its key in
# `agp_scans.yaml`. `vault`: named in the table of jonginn's note. `h5_ok`: its assembled HDF5
# and those of its parts exist.

# %%
note_table = NOTE.read_text(encoding="utf-8").split("### Comparing with previous round")[-1]
registry = {(data_root() / entry["manifest"]).resolve(): key for key, entry in registered_scans().items()}


def h5_ok(path):
    manifest = read_manifest(path)
    files = [Path(path).parent / manifest["assembled_h5"]]
    files += [Path(path).parent / read_manifest(Path(path).parent / Path(raw).name)["assembled_h5"]
              for raw in manifest["raw_files"] if str(raw).endswith(".yaml")]
    return all(f.exists() for f in files)


covered = {}
scan_rows = []
for folder in FOLDERS:
    for path in sorted((folder / "assembled_data").glob("*.yaml")):
        covered.setdefault(folder, set()).update(manifest_jobs(path))
    for row in list_scans(folder).itertuples():
        path = Path(row.manifest).resolve()
        scan_rows.append(dict(folder=folder.name, scan=row.scan, direction=row.direction, W_khz=row.W_khz,
                              cycles=row.cycles, jobs=row.jobs, part_of=row.part_of,
                              registry=registry.get(path, ""), vault=path.name in note_table,
                              h5_ok=h5_ok(path)))
audit = pd.DataFrame(scan_rows)
audit

# %% [markdown]
# Step 2: the job files in the two `data` folders that no manifest uses, by day and class.

# %%
unused = []
for folder in FOLDERS:
    for path in sorted((folder / "data").glob("JOB-*.h5")):
        job_id = JOB_ID_RE.search(path.name).group(0)
        if job_id not in covered.get(folder, set()):
            unused.append(dict(folder=folder.name, day=job_id[4:12], job=job_id,
                               cls=path.stem.split("_", 1)[1]))
unused = pd.DataFrame(unused)
unused.groupby(["folder", "day", "cls"]).job.agg(["count", "min", "max"])

# %% [markdown]
# Step 3: the unused population jobs, grouped by their settings (read mode, photon number,
# storages, W, cycles). A group with a full direction sector is an AGP part that no manifest has.

# %%
groups = []
for row in unused[unused.cls == "FockPopulationExperiment"].itertuples():
    folder = next(f for f in FOLDERS if f.name == row.folder)
    path = next((folder / "data").glob(f"{row.job}_*.h5"))
    expt = FockPopulationExperiment.from_h5file(path, load_shots=False).cfg.expt
    cycles = np.asarray(expt.floquet_cycles)
    groups.append(dict(folder=row.folder, day=row.day, job=row.job, ro_stor=int(expt.ro_stor),
                       N=int(sum(expt.occupations)), swap_stors=str(list(expt.swap_stors)),
                       W_khz=round(rms_khz(expt.detunings), 1),
                       cycles=f"{cycles[0]}-{cycles[-1]} ({len(cycles)})"))
groups = pd.DataFrame(groups)
(groups.groupby(["folder", "day", "ro_stor", "N", "swap_stors", "W_khz", "cycles"])
       .job.agg(["count", "min", "max"]))

# %% [markdown]
# Step 4: does loading a registered scan read the single shots? Log every dataset that
# `load_scan` + `analyze` read from an HDF5 file.

# %%
read_keys = []
original_getitem = h5py.Group.__getitem__


def logged_getitem(self, name):
    item = original_getitem(self, name)
    if isinstance(item, h5py.Dataset):
        read_keys.append((Path(self.file.filename).name, name, item.nbytes))
    return item


h5py.Group.__getitem__ = logged_getitem
try:
    start = time.perf_counter()
    check = load_scan("s7_high_w")
    check.analyze(background_start_us=BACKGROUND_US)
    seconds = time.perf_counter() - start
finally:
    h5py.Group.__getitem__ = original_getitem
log = pd.DataFrame(read_keys, columns=["file", "dataset", "nbytes"])
shots = log[log.dataset.isin(FockPopulationExperiment.SHOT_KEYS)]
job_bytes = sum(p.stat().st_size for p in (FOLDER / "data").glob("JOB-2026103[0]-*.h5")) + \
    sum(p.stat().st_size for p in (FOLDER / "data").glob("JOB-20261029-*FockPopulation*.h5"))
print(f"{seconds:.1f} s, {log.file.nunique()} files, {log.nbytes.sum() / 1e6:.1f} MB looked at, "
      f"shot datasets opened: {len(shots)} ({', '.join(shots.file.unique())}); "
      f"job files of 10-29..30 on disk: {job_bytes / 1e6:.0f} MB")

# %% [markdown]
# Step 5: which registered scans find their readout calibration (else the AGP errors are
# shot-only)?

# %%
calibration_rows = []
for key in registered_scans():
    scan = load_scan(key)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan.analyze(background_start_us=BACKGROUND_US)
    missing = sum("Calibration file not found" in str(w.message) for w in caught)
    calibrations = sorted({str(getattr(part, "calibration", None)) for part in scan.children})
    calibration_rows.append(dict(scan=key, parts=len(scan.children), parts_without_calibration=missing,
                                 calibration=", ".join(Path(c).name for c in calibrations)))
pd.DataFrame(calibration_rows)

# %% [markdown]
# Result (2026-11-03, before the cleanup):
# - 10 scan manifests in the two folders, all with their assembled HDF5s (scan and parts). 6 were
#   registered. The other 4 were subsets or replaced: 261023_001448 and 261023_120247 are in
#   261027_163118, 261030_094038 is in 261030_123324, and 261103_160925 replaces 261102_161533.
# - Loading reads no single shots: `s7_high_w` reads 10 MB of 759 MB in 0.9 s. The only shot
#   dataset that is opened is the calibration's `idata`, for its shape (the number of shots).
#   All registered scans find their calibration.
# - Job files that no manifest uses: walks and readout checks of 10-27/28 (jonginn's low-Kerr
#   calibration, in OneNote), the 160 kHz partial, 7 S6 jobs of 10-23 (question to jonginn in the
#   vault note), legacy QsimExperiment jobs of 10-10..10-21.

# %% [markdown]
# ## Cleanup (2026-11-03): every data set of the two folders again, under its name
#
# Names are now required (`save(name)`), and each W keeps its AGP(mu) and S(f) curves. The old
# files (time-stamp names) were moved to `assembled_data_old_20261103` beside `assembled_data`.
# Each data set is made again from them with the analysis below. Not made again, because another
# data set holds all of their jobs: 261023_001448 and 261023_120247 (in s7_coarse), 261030_094038
# (in s7_high_w), 261102_161533 (replaced by s7_ultrafine, which labels W = 30 correctly).
# A name that exists already is left as it is.

# %%
SCANS = {  # name: (experiment folder, old stem)
    "s7_fine": (K44, "261023_001509"),
    "s7_coarse": (K44, "261027_163118"),
    "s7_ultrafine": (K44, "261103_160925"),
    "kerr": (K44, "261025_010957"),
    "kerr_w0": (K44, "261024_003051"),
    "s7_high_w": (FOLDER, "261030_123324"),
}
WALKS = {
    "walk_n3_20261017": (K44, "261023_001438"),
    "walk_n2_20261017": (K44, "261023_001439"),
    "walk_n1_20261017": (K44, "261023_001440"),
    "walk_n3_20261029": (FOLDER, "261103_164201"),
}
ANALYSIS = dict(background_start_us=BACKGROUND_US)  # default mu axis, recorded Kerr, no T1

regenerated = {}
for name, (folder, stem) in {**SCANS, **WALKS}.items():
    cls = CentralBosonAGPDisorderScan if name in SCANS else FockPopulationSetExperiment
    target = folder / "assembled_data" / f"{cls.__name__}_{name}.yaml"
    if target.exists():
        regenerated[name] = target
        continue
    assembled = cls.from_manifest(folder / OLD / f"{stem}_{cls.__name__}.yaml")
    if name in SCANS:
        for part in assembled.children:
            part.manifest_path = None
        assembled.analyze(**ANALYSIS)
    else:
        assembled.analyze()
    notes = (assembled.notes + "; " if assembled.notes else "") + f"made again 2026-11-03 from {stem}"
    regenerated[name] = assembled.save(name, directory=folder / "assembled_data", notes=notes)
{name: path.name for name, path in regenerated.items()}

# %% [markdown]
# Check: the new files have the same jobs and the same C_1 (scans) or populations (walks) as
# the old ones, and the scans have the new curves.

# %%
checks = []
for name, (folder, stem) in {**SCANS, **WALKS}.items():
    cls = "CentralBosonAGPDisorderScan" if name in SCANS else "FockPopulationSetExperiment"
    old_arrays, _ = read_assembled_h5(folder / OLD / f"{stem}_{cls}.h5")
    new_arrays, attrs = read_assembled_h5(regenerated[name].with_suffix(".h5"))
    keys = (["disorder_khz", "C1", "C1_err", "C1_corrected", "C1_theory", "time_us"] if name in SCANS
            else ["n_mean", "n_mean_corrected", "time_us"])
    same = {key: bool(np.allclose(old_arrays[key], new_arrays[key], equal_nan=True))
            for key in keys if key in old_arrays}
    checks.append(dict(name=name, jobs=len(manifest_jobs(regenerated[name])),
                       same_jobs=manifest_jobs(folder / OLD / f"{stem}_{cls}.yaml")
                       == manifest_jobs(regenerated[name]),
                       same_arrays=all(same.values()), compared=", ".join(same),
                       curves="spectrum" in new_arrays and "agp" in new_arrays))
pd.DataFrame(checks)

# %% [markdown]
# ## The plots of every registered scan, to the vault
#
# Next to jonginn's note (`images/agp_<name>_<kind>.png`): AGP against W, AGP against mu, and
# the spectra. A file that exists already is not overwritten.

# %%
exported = {}
for key, scan in load_scans().items():
    scan.analyze(background_start_us=BACKGROUND_US)
    figures = dict(agp=scan.display_agp(mu=[10., 20.]),
                   agp_mu=scan.display_agp_mu(estimator="agp_connected"),
                   spectrum=scan.display_spectrum(mu=[10., 20.]))
    for kind, fig in figures.items():
        title = fig._suptitle.get_text() if fig._suptitle else kind
        fig.suptitle(f"{key}: {title}", fontsize=10)
        target = VAULT_IMAGES / f"agp_{key}_{kind}.png"
        if not target.exists():
            fig.savefig(target, dpi=110)
            exported.setdefault(key, []).append(target.name)
exported

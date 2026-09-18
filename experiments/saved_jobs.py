"""Load saved experiments from HDF5 alone, for offline analysis.

Why this exists
---------------
Analysis used to load a job by asking the job server for it: ``get_status`` to
find the pickle, then ``load_expt`` to unpickle a whole Experiment, compiled
Program included. That made every analysis notebook depend on a running
server, a live database and a pickle written by the same code revision -- none
of which exist on a collaborator's laptop, and none of which can be published
with a paper.

This module is the replacement. It reads the HDF5 file and nothing else:

* no ``JobClient``, no ``job_server`` import at all;
* no pickles, so no dependence on the acquisition revision;
* no ``MultimodeStation``, so no chance of substituting today's calibration
  for the historical one.

The one fact HDF5 does not carry
--------------------------------
The Floquet cycle time and the storage couplings. They were only ever
available from the compiled Program, which lives in the pickle. Three places
can supply them, tried in this order:

1. ``timing=``, historical values supplied by hand. The explicit escape hatch,
   and the only option for a file whose configs were never versioned.
2. the file's own ``derived_params`` attribute, written at acquisition. Present
   only on files written after that landed; see the module note below.
3. recomputation from the versioned config named in the provenance sidecar,
   via :func:`experiments.floquet_timing.resolve_floquet_timing`. This is
   exact, not approximate -- the configs are immutable and the arithmetic is
   the program's own.

If none of the three can supply it, loading **raises** and the message says
what to pass. It does not fall back to a placeholder: the cycle time divides
into every energy in the result, so a NaN would silently poison the whole
spectrum instead of stopping. Genuinely optional metadata may be absent, but
not this.

On ``derived_params``
---------------------
The acquisition side writes two provenance attributes beside ``config``:
``derived_params`` (this timing) and ``config_versions`` (the config version
IDs). They are attributes rather than entries in ``cfg.expt`` because
``cfg.expt`` is the *input* to a run -- the thing a notebook overrides by
hand. Timing is generated, never consumed, and a generated value parked among
the inputs eventually gets replayed into a later job as if it had been chosen.
This module already reads the attribute, so files that carry it need no
provenance sidecar; files that do not fall through to the sidecar as before.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from slab import AttrDict

from experiments.floquet_timing import TimingResolutionError, resolve_floquet_timing
from experiments.job_paths import JOB_ID_RE, job_records, resolve_job_paths

# Timing needs both of these: the couplings are 1/(4 * pi_frac * cycle_us), so
# a cycle time without the pi fracs cannot produce them.
TIMING_KEYS = ("floquet_cycle_us", "m1s_pi_fracs")

# The two large shot arrays are ~99% of a file. Only shot-noise and split-half
# analyses read them, so they are skipped by default.
SHOT_KEYS = ("idata", "qdata")


class SavedJobError(RuntimeError):
    """Raised when a saved job cannot be loaded or is missing its timing.

    A distinct type so a caller can tell "this file lacks provenance" apart
    from "this analysis is wrong".
    """


class SavedProgram:
    """Supplies Floquet timing where the analysis looks for a compiled program.

    ``_saved_parameters`` asks its children for ``calculate_floquet_cycle_us``
    and ``m1s_pi_fracs``, which only a compiled Program had. Duck-typing those
    two members is what lets the aggregate analysis run unchanged on data
    loaded from HDF5.
    """

    def __init__(self, floquet_cycle_us, m1s_pi_fracs, source="unspecified"):
        self._cycle_us = float(floquet_cycle_us)
        self.m1s_pi_fracs = list(m1s_pi_fracs)
        self.source = source

    def calculate_floquet_cycle_us(self):
        return self._cycle_us

    @property
    def couplings_MHz(self):
        """Per-mode couplings, for cross-checking against a recorded value."""
        return 1. / (4. * np.asarray(self.m1s_pi_fracs, dtype=float) * self._cycle_us)


class SavedJob:
    """One saved job: the config and data the analysis path reads.

    Stands in for the pickled Experiment. It deliberately has no ``acquire``,
    no instruments and no station -- it cannot be run, only analyzed.
    """

    def __init__(self, job_id, cfg, data, path, prog):
        self.job_id = job_id
        self.cfg = cfg
        self.data = data
        self.prog = prog
        self.fname = str(path)
        # The three names the old loaders set on a child, kept so display and
        # save paths that build a filename from them keep working.
        self.path = str(Path(path).parent)
        self.config_file = str(path)
        self.prefix = Path(path).stem

    def __repr__(self):
        return f"<SavedJob {self.job_id} {Path(self.fname).name}>"


def load_h5(path, load_shots=False):
    """-> (cfg, data) from one HDF5 file.

    ``data['attrs']`` carries the file's root attributes, matching what
    ``Experiment.from_h5file`` produces, so analysis code that reads it is
    unaffected by loading this way.
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        try:
            cfg = AttrDict(json.loads(handle.attrs["config"]))
        except KeyError:
            raise SavedJobError(
                f"{path.name} has no 'config' attribute, so it records neither "
                f"the swept parameters nor the readout calibration. It cannot "
                f"be analyzed; reconstruct it from the acquisition notebook.")
        skip = () if load_shots else SHOT_KEYS
        data = AttrDict({key: handle[key][()] for key in handle if key not in skip})
        data["attrs"] = dict(handle.attrs)
    return cfg, data


def _derived_params(data):
    """-> the file's own ``derived_params`` attribute, or None if absent."""
    raw = data.get("attrs", {}).get("derived_params")
    if raw is None:
        return None
    params = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    missing = [key for key in TIMING_KEYS if params.get(key) is None]
    if missing:
        raise SavedJobError(
            f"derived_params is present but incomplete: missing {missing}. "
            f"Either fix the file or pass timing= explicitly.")
    return params


def _check_couplings(params, prog, job_id):
    """Cross-check a recorded coupling against the one implied by the timing.

    ``derived_params`` records couplings as well as the two values they are
    computed from. Recomputing and comparing costs nothing and would catch a
    file written by a different formula than the one analysis applies.
    """
    recorded = params.get("couplings_MHz")
    if recorded is None:
        return
    recorded = np.asarray(recorded, dtype=float)
    implied = prog.couplings_MHz
    # The recorded value covers the swapped modes; the implied one covers all
    # seven. Compare only where they line up.
    if recorded.shape == implied.shape and not np.allclose(recorded, implied, rtol=1e-9):
        raise SavedJobError(
            f"{job_id}: derived_params couplings {recorded} disagree with the "
            f"couplings implied by its own cycle time and pi fracs {implied}.")


def resolve_timing(job_id, cfg, data, timing=None, record=None):
    """-> a :class:`SavedProgram` for one job, or raise saying what to supply.

    Order of preference is the module docstring's: an explicit override, then
    the file's own attribute, then recomputation from the versioned config.
    """
    if timing is not None:
        missing = [key for key in TIMING_KEYS if timing.get(key) is None]
        if missing:
            raise SavedJobError(f"timing= is missing {missing}")
        return SavedProgram(timing["floquet_cycle_us"], timing["m1s_pi_fracs"],
                            source="supplied by caller")

    params = _derived_params(data)
    if params is not None:
        prog = SavedProgram(params["floquet_cycle_us"], params["m1s_pi_fracs"],
                            source=f"H5 derived_params: {job_id}")
        _check_couplings(params, prog, job_id)
        return prog

    version_id = (record or {}).get("floquet_storage_version_id")
    if version_id:
        try:
            resolved = resolve_floquet_timing(cfg, version_id)
        except TimingResolutionError as error:
            raise SavedJobError(
                f"{job_id}: its Floquet config {version_id} could not be "
                f"resolved from the archive ({error}). Point "
                f"$MULTIMODE_CONFIG_ARCHIVE at the versions directory, or pass "
                f"the historical values as timing=.") from error
        return SavedProgram(resolved["floquet_cycle_us"], resolved["m1s_pi_fracs"],
                            source=resolved["source"])

    raise SavedJobError(
        f"{job_id}: no Floquet timing available. The file carries no "
        f"'derived_params' attribute and the provenance sidecar records no "
        f"floquet_storage_version_id for it, so the cycle time cannot be "
        f"recovered from what was saved.\n"
        f"Supply the historical values explicitly:\n"
        f"  timing=dict(floquet_cycle_us=..., m1s_pi_fracs=[...])\n"
        f"or re-export the sidecar on the acquisition workstation:\n"
        f"  pixi run python tools/export_job_provenance.py --range ... "
        f"-o tests/data/job_provenance.json")


def _flatten(values):
    """Flatten nested job-ID lists, which the notebooks build by range union."""
    if isinstance(values, (str, bytes)):
        return [values]
    flat = []
    for value in values:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            flat.append(value)
        else:
            flat.extend(_flatten(value))
    return flat


def job_id_from_path(path):
    """-> the ``JOB-YYYYMMDD-NNNNN`` in a data file's name, or its stem.

    Files are named ``{job_id}_{ExperimentClass}.h5``. A file that predates the
    job server has no ID in its name, so the stem stands in -- it is only used
    for labelling and for error messages.
    """
    match = JOB_ID_RE.search(Path(path).name)
    return match.group(0) if match else Path(path).stem


def job_ids(values):
    """-> a flat list of ``JOB-YYYYMMDD-NNNNN`` strings."""
    ids = [str(value) for value in _flatten(values)]
    if not ids:
        raise ValueError("job_ids cannot be empty")
    return ids


def select_jobs(ids, program_class=None, provenance=None, require_completed=True):
    """Split job IDs into the ones worth loading and the ones to skip.

    This replaces filtering by the unpickled ``expt.prog``'s class name. The
    sidecar records ``program_class`` and ``status`` per job, so a mixed job
    range is filtered *before* any file is opened rather than after a pickle
    is loaded and inspected.

    Args:
        ids: job IDs, already flattened.
        program_class: keep only jobs recorded under this program class. None
            keeps every job.
        provenance: the records dict; read from the sidecar when None.
        require_completed: skip jobs whose recorded status is not COMPLETED.

    Returns:
        ``(kept, skipped)`` where ``skipped`` is a list of ``(job_id, reason)``.
    """
    records = job_records() if provenance is None else provenance
    kept, skipped = [], []
    for job_id in ids:
        record = records.get(job_id)
        if record is None:
            # Absent from the sidecar is not absent from disk: a job acquired
            # since the last export still loads, it just cannot be filtered or
            # have its timing resolved. Keep it and let loading decide.
            kept.append(job_id)
            continue
        status = str(record.get("status", "")).upper()
        if require_completed and status and status != "COMPLETED":
            skipped.append((job_id, status))
            continue
        saved_class = record.get("program_class")
        if program_class is not None and saved_class != program_class:
            skipped.append((job_id, saved_class or "unknown program"))
            continue
        kept.append(job_id)
    return kept, skipped


def load_job(job_id, path=None, timing=None, provenance=None, load_shots=False):
    """-> one :class:`SavedJob`, from HDF5 and nothing else."""
    if path is None:
        path = resolve_job_paths([job_id])[job_id]
    records = job_records() if provenance is None else provenance
    cfg, data = load_h5(path, load_shots=load_shots)
    prog = resolve_timing(job_id, cfg, data, timing=timing,
                          record=records.get(job_id))
    return SavedJob(job_id, cfg, data, path, prog)


def load_aggregate(ids, owner, *, program_class=None, timing=None,
                   load_shots=False, provenance=None, analyze=False):
    """Build one aggregate Experiment for ``ids`` from HDF5 alone.

    The single entry point the analysis notebooks and helpers call. It replaces
    ``from_job_ids(client=...)`` and the three near-identical
    ``get_status``/``load_expt`` loops that used to live in the notebook
    helpers.

    Args:
        ids: job IDs; nested lists are flattened.
        owner: the stage Experiment class that reassembles them, e.g.
            ``MBRSpectrumExperiment``. The class that *named* the files is not
            necessarily the one that analyzes them.
        program_class: keep only jobs recorded under this program class.
        timing: historical ``dict(floquet_cycle_us=..., m1s_pi_fracs=[...])``
            for files that carry no provenance.
        load_shots: also read ``idata``/``qdata``, which are large.
        analyze: run ``analyze()`` on the aggregate before returning. The
            calibration stage always needs this; the spectroscopy stage takes
            analysis parameters, so it does not.

    Returns:
        the aggregate, with ``batch_job_ids`` and ``skipped_job_ids`` set.
    """
    ids = job_ids(ids)
    records = job_records(required=False) if provenance is None else provenance
    kept, skipped = select_jobs(ids, program_class=program_class,
                                provenance=records)
    if not kept:
        raise SavedJobError(
            f"none of the {len(ids)} requested jobs are usable"
            + (f" as {program_class}" if program_class else "")
            + f"; first skips: {skipped[:5]}")

    paths = resolve_job_paths(kept)
    children = [load_job(job_id, path=paths[job_id], timing=timing,
                         provenance=records, load_shots=load_shots)
                for job_id in kept]

    aggregate = owner._from_expts(children, job_ids=kept)
    aggregate.skipped_job_ids = [job_id for job_id, _ in skipped]
    aggregate.skipped_jobs = skipped
    if analyze:
        aggregate.analyze()
    return aggregate

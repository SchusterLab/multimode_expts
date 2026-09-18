"""The reference MBR analysis pipeline, as a callable.

This is the *current* behaviour of the many-body Ramsey spectrum workflow,
captured so it can be pinned. It is the executable form of
``analysis_notebooks/guan/MBR_analysis.py``: pick job IDs, load their HDF5,
run ``MBRSpectrumExperiment.analyze()``.

Not a test module (no ``test_`` prefix, so pytest does not collect it). It is
imported by ``test_mbr_analysis_golden.py`` and by the baseline generator.

The HDF5 loader now lives in the library
----------------------------------------
This module used to carry its own ``_SavedJob``/``_SavedProgram`` pair and its
own HDF5 reader, marked as scaffolding to be deleted once a real loader
landed. It has: :mod:`experiments.saved_jobs`. The functions below are thin
wrappers over it, kept because the fixtures and the golden test call them by
these names.

That the wrappers are thin is the point. These tests pin the analysis numbers
against a stored baseline, so running them through the same loader the
notebooks use is what makes the baseline evidence that the loader is right.
"""

import json
from pathlib import Path

import numpy as np

from experiments.job_paths import job_records, resolve_job_paths
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.saved_jobs import load_aggregate as _load_aggregate
from experiments.saved_jobs import load_h5

DATASETS = Path(__file__).parent / "data" / "mbr_datasets.json"


def dataset(name, kind):
    """Job IDs for one named dataset, from the recorded literal list.

    Deliberately not a numeric range. Job IDs are one global counter on a queue
    shared by every user -- they interleave by design, each job pinning its own
    config -- so a range is not an identifier for a dataset. The notebook
    subtracts the other user's jobs with a positional stride in one place and a
    program-class filter in another; three of its four July ranges pull in jobs
    belonging to someone else. (``july_N3`` and every August set are
    unaffected.)

    The lists were resolved from the job database once, by owner, program
    class, completed status and config-triple agreement. That resolver was a
    throwaway; only its output is kept. This file is the interim home -- the
    list belongs in the aggregate HDF5 manifest (spec 3.3) once aggregates can
    save themselves.
    """
    return json.loads(DATASETS.read_text())["datasets"][name][kind]


# Eight quadrature acquisitions covering four occupations, from the August
# campaign. In the notebook this is `data_four_realization`: a fragment used
# for quick plotting, NOT the August N=3 sector. It does not complete the
# fixed-N basis. The golden baseline pins it.
CHARACTERIZATION_JOB_IDS = dataset("august_quickplot", "spectroscopy")

# Analysis parameters the reference notebook uses. Kept here rather than in the
# test so the baseline generator and the test cannot disagree about them.
CHARACTERIZATION_ANALYSIS = dict(
    calibration=None,
    cycle_branches={(3, 0, 0, 0, 0): 1, (2, 0, 0, 1, 0): 1},
    fft_window="raw",
    zero_padding=1,
    spectrum_method="fft",
)

# Floquet timing as compiled at acquisition. Reconstructed exactly from
# CFG-FL-20260814-00076 + CFG-HW-20260814-00074 + configs/soccfg_snapshot.json
# (spec section 2.2); the historical pickle holds the same value because it
# computed it from the same configs. Hard-coded here only until the resolver
# replaces it, since threading the resolver in is itself a behaviour change and
# the baseline must be captured before any of those.
CHARACTERIZATION_TIMING = dict(
    floquet_cycle_us=0.7254464285714286,
    m1s_pi_fracs=[40] * 7,
)

# --------------------------------------------------------------------------
# The complete-basis dataset: August N=3
# --------------------------------------------------------------------------
#
# 70 calibration plus 70 spectroscopy jobs completing the 35-state N=3 basis,
# so level statistics, the SFF and the complete-basis branch of analyze_spectrum
# all run on it -- none of which the eight-file quick-plot set can reach.
#
# Chosen over the July N=3 sector, which also completes the basis, because it
# is the same campaign and the *same* Floquet and M1 configuration as the
# quick-plot set (CFG-FL-20260814-00076, CFG-M1-20260814-00121). One timing
# resolution therefore covers every fixture here. Its ranges are also clean:
# no other user's jobs fall inside them, unlike three of the four July ranges.
#
# Settings below are the notebook's `saved_*` values for this sector.

COMPLETE_BASIS_CALIBRATION_IDS = dataset("august_N3", "calibration")
COMPLETE_BASIS_SPECTROSCOPY_IDS = dataset("august_N3", "spectroscopy")

COMPLETE_BASIS_CYCLE_BRANCHES = {
    (2, 1, 0, 0, 0): 1,
    (2, 0, 1, 0, 0): 1,
    (1, 1, 0, 1, 0): 1,
    (1, 1, 0, 0, 1): 1,
    (1, 0, 1, 1, 0): 1,
    (1, 0, 1, 0, 1): 1,
}

# Branches worth pinning separately. The notebook runs the first and the third;
# the second exists so the Matrix-Pencil path gets complete-basis coverage too.
# Together they exercise both phase frames and both spectrum methods, which is
# everything the previous two-dataset arrangement covered plus the complete
# basis.
COMPLETE_BASIS_BRANCHES = {
    "as_acquired_fft": dict(phase_frame="as_acquired", spectrum_method="fft"),
    "as_acquired_matrix_pencil": dict(phase_frame="as_acquired",
                                      spectrum_method="matrix_pencil"),
    "manual_kerr_fft": dict(phase_frame="manual_kerr",
                            manual_kerr_MHz=-10.5e-3,
                            cycle_branches=COMPLETE_BASIS_CYCLE_BRANCHES,
                            spectrum_method="fft"),
}

COMPLETE_BASIS_ANALYSIS = dict(fft_window="raw", zero_padding=1)

PROVENANCE = Path(__file__).parent / "data" / "job_provenance.json"


def load_aggregate(job_ids=None, timing=None, owner=MBRSpectrumExperiment):
    """Build the aggregate Experiment for ``job_ids`` from HDF5 alone.

    Touches no station, no database, no vault note and no pickle, per the
    isolation rule in spec section 13.3.

    ``timing`` is pinned to :data:`CHARACTERIZATION_TIMING` by default rather
    than resolved from provenance, so this fixture keeps testing the analysis
    against a fixed cycle time even if the resolver changes.
    :func:`load_aggregate_resolved` is the one that exercises the resolver.
    """
    ids = list(CHARACTERIZATION_JOB_IDS if job_ids is None else job_ids)
    return _load_aggregate(ids, owner=owner,
                           timing=timing or CHARACTERIZATION_TIMING)


def run_reference_analysis(job_ids=None, timing=None, **overrides):
    """-> (expt, analysis_result) for the characterization workflow."""
    expt = load_aggregate(job_ids=job_ids, timing=timing)
    params = dict(CHARACTERIZATION_ANALYSIS)
    params.update(overrides)
    return expt, expt.analyze(**params)


# --------------------------------------------------------------------------
# Loading by resolved provenance, rather than a hard-coded timing constant
# --------------------------------------------------------------------------


def job_provenance():
    """-> {job_id: record} from the exported sidecar (spec section 3.2).

    Delegates to :func:`experiments.job_paths.job_records`; kept as a name
    because the golden test reads the version IDs out of it directly.
    """
    return job_records()


def load_aggregate_resolved(job_ids, owner=MBRSpectrumExperiment):
    """Build an aggregate whose Floquet timing comes from the versioned configs.

    Unlike :func:`load_aggregate` this passes no ``timing``, so the loader
    resolves each job's Floquet config from the provenance sidecar and
    recomputes the timing from the archive (spec section 2.2). That is what
    makes datasets other than the August characterization set loadable at
    all -- the July sectors were taken under a different configuration.
    """
    return _load_aggregate(list(job_ids), owner=owner)


def load_complete_basis():
    """-> (calibration_expt, spectroscopy_expt, occupations) for August N=3.

    Loads from HDF5 plus the provenance sidecar; no job server, no pickles.
    Cached per process because the two aggregates cover 140 files and every
    branch reuses them.
    """
    if not hasattr(load_complete_basis, "_cache"):
        calibration = load_aggregate_resolved(
            COMPLETE_BASIS_CALIBRATION_IDS, owner=MBRPhaseCorrectionExperiment)
        calibration.analyze()
        occupations = [tuple(map(int, o)) for o in calibration.data.occupations]
        if len(occupations) != 35 or any(sum(o) != 3 for o in occupations):
            raise RuntimeError(
                f"calibration is not the complete 35-state N=3 sector: "
                f"{len(occupations)} occupations")
        spectroscopy = load_aggregate_resolved(COMPLETE_BASIS_SPECTROSCOPY_IDS)
        load_complete_basis._cache = (calibration, spectroscopy, occupations)
    return load_complete_basis._cache


def run_complete_basis_analysis(branch="as_acquired_fft", **overrides):
    """-> (expt, analysis_result) for one branch of the August N=3 sector."""
    calibration, spectroscopy, occupations = load_complete_basis()
    params = dict(COMPLETE_BASIS_ANALYSIS,
                  calibration=calibration,
                  occupations=occupations,
                  **COMPLETE_BASIS_BRANCHES[branch])
    params.update(overrides)
    return spectroscopy, spectroscopy.analyze(**params)


# --------------------------------------------------------------------------
# Flattening, so a nested analysis result can be compared field by field
# --------------------------------------------------------------------------
#
# A single hash over the whole result would tell us "something changed" and
# nothing else. Flattening to leaf paths means a failure names the field, which
# during a code move is the entire diagnostic value.

def _is_scalar(value):
    return isinstance(value, (bool, int, float, str, np.integer, np.floating, np.bool_))


def flatten_result(value, prefix=""):
    """Nested dict/list/array -> {dotted path: ndarray or scalar}.

    Tuple keys (occupations, e.g. ``(2, 0, 0, 1, 0)``) become part of the path,
    so per-occupation phase corrections are compared individually.
    """
    flat = {}

    def emit(path, item):
        if isinstance(item, dict):
            for key, sub in item.items():
                emit(f"{path}.{key}" if path else str(key), sub)
        elif isinstance(item, np.ndarray):
            flat[path] = item
        elif isinstance(item, (list, tuple, set)):
            # Homogeneous numeric sequences are arrays in all but name; keep
            # them as one entry rather than exploding into hundreds of paths.
            # Ragged or object sequences (matrix-pencil candidate lists, say)
            # raise here rather than producing an object array, so recurse.
            try:
                as_array = np.array(sorted(item) if isinstance(item, set) else item)
            except (ValueError, TypeError):
                as_array = None
            if as_array is not None and as_array.dtype != object:
                flat[path] = as_array
            else:
                for i, sub in enumerate(item):
                    emit(f"{path}[{i}]", sub)
        elif item is None:
            flat[path] = np.array("None", dtype=object)
        elif _is_scalar(item):
            flat[path] = item
        else:
            # Anything else (an Experiment, a fit object) is not part of the
            # numerical contract; record its type so an unexpected appearance
            # is visible without pinning its internals.
            flat[path] = np.array(f"<{type(item).__name__}>", dtype=object)

    emit(prefix, value)
    return flat

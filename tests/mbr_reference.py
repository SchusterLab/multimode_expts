"""The reference MBR analysis pipeline, as a callable.

This is the *current* behaviour of the many-body Ramsey spectrum workflow,
captured so it can be pinned. It is the executable form of
``analysis_notebooks/guan/MBR_analysis.py``: pick job IDs, load their HDF5,
run ``EncodingHamiltonianSpectroscopyExperiment.analyze(stage="spectrum")``.

Not a test module (no ``test_`` prefix, so pytest does not collect it). It is
imported by ``test_mbr_analysis_golden.py`` and by the baseline generator.

Scaffolding, and known to be scaffolding
----------------------------------------
``_SavedJob`` and ``_SavedProgram`` duck-type the pickled experiment object
that the aggregate path expects to find. They exist only because raw HDF5 does
not yet carry its own provenance (section 3.1 of the refactor spec): the
analysis wants a compiled Program to ask for Floquet timing, and a file saved
today cannot supply one.

Both classes die when ``MBRSpectrumExperiment`` and its HDF5 loader land
(spec phase 2). Do not build on them.

TODO(spec 3.1): drop ``_SavedProgram`` once the timing resolver reads the
versioned configs directly, and ``_SavedJob`` once aggregates load HDF5
themselves.
"""

import json

import h5py
import numpy as np
from slab import AttrDict

from experiments.job_paths import resolve_job_paths
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)

# The August 2026 N=3 characterization set: eight quadrature acquisitions
# covering four occupations. This is the dataset the spec's appendix B evidence
# was taken from, and the one the golden baseline pins.
CHARACTERIZATION_JOB_IDS = [f"JOB-20260815-{n:05d}" for n in range(9, 17)]

# Analysis parameters the reference notebook uses. Kept here rather than in the
# test so the baseline generator and the test cannot disagree about them.
CHARACTERIZATION_ANALYSIS = dict(
    stage="spectrum",
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
    floquet_cycle_us=0.7340315934065934,
    m1s_pi_fracs=[40] * 7,
)


def load_h5(path, load_shots=False):
    """-> (cfg, data). Skips idata/qdata unless asked; they are ~99% of the file."""
    with h5py.File(path, "r") as handle:
        cfg = AttrDict(json.loads(handle.attrs["config"]))
        skip = () if load_shots else ("idata", "qdata")
        data = AttrDict({k: handle[k][()] for k in handle if k not in skip})
    return cfg, data


class _SavedProgram:
    """Supplies Floquet timing where the analysis looks for a compiled program."""

    def __init__(self, floquet_cycle_us, m1s_pi_fracs):
        self._cycle_us = float(floquet_cycle_us)
        self.m1s_pi_fracs = list(m1s_pi_fracs)

    def calculate_floquet_cycle_us(self):
        return self._cycle_us


class _SavedJob:
    """Duck-types the pickled experiment for what the analysis path reads."""

    def __init__(self, job_id, cfg, data, path, prog):
        self.job_id, self.cfg, self.data = job_id, cfg, data
        self.fname = str(path)
        self.prog = prog


def load_aggregate(job_ids=None, timing=None):
    """Build the aggregate Experiment for ``job_ids`` from HDF5 alone.

    Touches no station, no database, no vault note and no pickle, per the
    isolation rule in spec section 13.3.
    """
    ids = list(CHARACTERIZATION_JOB_IDS if job_ids is None else job_ids)
    paths = resolve_job_paths(ids)
    prog = _SavedProgram(**(timing or CHARACTERIZATION_TIMING))
    jobs = [_SavedJob(j, *load_h5(paths[j]), paths[j], prog) for j in ids]
    return EncodingHamiltonianSpectroscopyExperiment._from_expts(jobs, job_ids=ids)


def run_reference_analysis(job_ids=None, timing=None, **overrides):
    """-> (expt, analysis_result) for the characterization workflow."""
    expt = load_aggregate(job_ids=job_ids, timing=timing)
    params = dict(CHARACTERIZATION_ANALYSIS)
    params.update(overrides)
    return expt, expt.analyze(**params)


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
            as_array = np.array(sorted(item) if isinstance(item, set) else item)
            if as_array.dtype != object:
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

"""EncodingHamiltonianSpectroscopyExperiment and the old stage map -- DEPRECATED.

Moved from `experiments/qsim/floquet_dark_mode_readout.py` on 2026-09-24 (MBR
redesign step 7e), without changes except that `hardware_parameters` is now
`experiments.floquet_timing.station_floquet_hardware`. The base class of the
old MBR classes in `legacy_mbr.py`; see `docs/qsim/mbr_step7_plan.md`.

Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it.
"""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.signal import find_peaks
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.qsim import matrix_pencil as matrix_pencil_analysis
from fitting.qsim import level_statistics as level_statistics_analysis
from fitting.qsim import mbr_spectrum as mbr_spectrum_analysis
from fitting.qsim import mbr_phase as mbr_phase_analysis
from fitting.fit_display_classes import (
    GeneralFitting,
    RamseyFitting,
)
from experiments.MM_base import *
from experiments.floquet_timing import FLUX_HIGH_THRESHOLD_MHZ, station_floquet_hardware
from experiments.qsim.qsim_base import *
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display import *

from experiments.qsim.kerr import *

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.sideband_scramble import SidebandScrambleProgram
from experiments.qsim.dark_base import (
    DarkBaseExperiment,
    DarkBaseProgram,
    DarkBaseRProgram,
    classify_two_parity_readouts,
)
from experiments.qsim.floquet_phase_frame import (
    advance_floquet_offsets,
    advance_matrix_offsets,
    detuning_phase_deg,
    mod360,
)
from experiments.qsim.floquet_register_bank import (
    _play_preloaded_floquet_register_bank_entry,
    _prepare_preloaded_floquet_register_bank,
)
from experiments.qsim import mbr_saved
from experiments.qsim.utils import flatten_exp_lists

from copy import copy, deepcopy
from itertools import product

from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view

class EncodingHamiltonianSpectroscopyExperiment(DarkBaseExperiment):
    """Per-job and aggregate analysis for encoding-calibrated spectroscopy."""

    @classmethod
    def _from_expts(cls, expts, job_ids=None, station=None):
        """Collect child experiments into one aggregate for analysis.

        ``station`` is passed on to the stages that still take one for theory
        comparison. It is no longer a source of Floquet timing -- see
        ``_saved_parameters``.
        """
        
        expts = list(flatten_exp_lists(expts))
        if not expts:
            raise ValueError("experiment jobs cannot be empty")
        aggregate = cls.__new__(cls)
        aggregate.cfg = expts[0].cfg
        aggregate.data = AttrDict()
        aggregate.batch_expts = expts
        aggregate.batch_job_ids = list(job_ids or [])
        aggregate._analysis_station = station
        return aggregate

    @classmethod
    def from_job_files(cls, job_files, timing=None):
        """
        Load child HDF5 files into one analysis object.

        Takes paths, or already-loaded child experiments. ``timing`` supplies
        the Floquet cycle time for files that carry neither a
        ``derived_params`` attribute nor a provenance entry; see
        :mod:`experiments.saved_jobs`.

        Pickles are no longer accepted. They are ephemeral debugging output,
        not data: unpickling needs the acquisition revision still importable,
        and the canonical record is the HDF5 file.
        """
        from pathlib import Path

        from experiments.saved_jobs import job_id_from_path, load_job

        if isinstance(job_files, (str, Path)):
            job_files = [job_files]
        expts = []
        for job_file in flatten_exp_lists(job_files):
            if not isinstance(job_file, (str, Path)):
                expts.append(job_file)
                continue
            path = Path(job_file)
            if path.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError(
                    f"{path.name} is not an HDF5 file. Analysis loads HDF5 only; "
                    f"if this is a job pickle, load the job's .h5 instead "
                    f"(from_job_ids resolves it by job ID).")
            expts.append(load_job(job_id_from_path(path), path=path, timing=timing))
        return cls._from_expts(expts)

    @classmethod
    def from_job_ids(cls, job_ids, timing=None, program_class=None):
        """
        Load saved jobs by ID, from HDF5 alone.

        Resolves each ID to its file via
        :func:`experiments.job_paths.resolve_job_paths` and reads it with
        :mod:`experiments.saved_jobs`. No job server, no job database, no
        pickle and no station: the timing that only the compiled program used
        to carry is recovered from the file's own ``derived_params`` attribute
        or recomputed from the versioned config, and ``timing`` is the manual
        escape hatch when neither exists.

        ``program_class`` filters a mixed job range by the program class
        recorded in the provenance sidecar.
        """
        from experiments.saved_jobs import load_aggregate

        return load_aggregate(job_ids, owner=cls, timing=timing,
                              program_class=program_class)

    # Hardware parameters of saved jobs live in experiments/qsim/mbr_saved.py,
    # shared with the new MBR classes (docs/qsim/mbr_redesign.md, section 2).
    _first_scalar = staticmethod(mbr_saved.first_scalar)
    _saved_detunings = staticmethod(mbr_saved.saved_detunings)

    @classmethod
    def _saved_parameters(cls, expts, station=None):
        """Swap modes, detunings, mode labels and hardware of sister jobs.

        See :func:`experiments.qsim.mbr_saved.saved_parameters`. ``station``
        is accepted and ignored, as before: there is no station fallback.
        """
        return mbr_saved.saved_parameters(expts)

    def analyze(self, data=None, **kwargs):
        """Convert one saved job's two preparation phases to a real quadrature.

        This is the per-job analysis the worker runs after ``acquire``. It
        produces ``Q_phi`` at that job's analyzer phase and does *not* combine
        the ``phi=0`` and ``phi=90`` jobs into a complex return -- that is an
        aggregate step and belongs to a stage Experiment.

        The four aggregate stages that used to hide behind ``stage=`` are now
        separate classes; see :data:`STAGE_CLASSES` and the migration table in
        ``analysis_notebooks/guan/MBR_analysis.py``.
        """
        if "stage" in kwargs:
            raise TypeError(_stage_migration_message(kwargs["stage"]))
        if data is not None:
            self.data = data
        self._quadrature(self)
        return self.data

    @staticmethod
    def _quadrature(expt):
        if "return_quadrature" in expt.data:
            return np.asarray(expt.data["return_quadrature"])
        cycles = expt.data["ypts"]
        theta = expt.data["xpts"]
        signal = np.asarray(expt.data["avgi"]).reshape(len(cycles), len(theta))
        q = expt.cfg.expt.qubits[0]
        Ig = expt.cfg.device.readout.Ig[q]
        Ie = expt.cfg.device.readout.Ie[q]
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        expt.data["Pe"] = (signal - Ig) / (Ie - Ig)
        expt.data["return_quadrature"] = expt.data["Pe"][:, 0] - expt.data["Pe"][:, 1]
        return expt.data["return_quadrature"]

    # Analyzer-phase numerics live in fitting/qsim/mbr_phase.py (spec 7.5).
    # Wrappers keep the historical call sites and notebook usage working.
    _cycle_branches = staticmethod(mbr_phase_analysis.cycle_branches)
    build_phase_correction = staticmethod(mbr_phase_analysis.build_phase_correction)
    _unwrap_cycle_phase = staticmethod(mbr_phase_analysis.unwrap_cycle_phase)
    _saved_correction = staticmethod(mbr_phase_analysis.saved_correction)


    # Spectrum and Hamiltonian numerics live in fitting/qsim/mbr_spectrum.py
    # (spec 7.5). Wrapper keeps the historical call sites working.
    analyze_spectrum = staticmethod(mbr_spectrum_analysis.analyze_spectrum)


    # Matrix-Pencil numerics live in fitting/qsim/matrix_pencil.py (spec 7.5).
    # These wrappers keep the historical call sites and notebook usage working.
    analyze_matrix_pencil = staticmethod(matrix_pencil_analysis.analyze_matrix_pencil)
    analyze_matrix_pencil_trace = staticmethod(matrix_pencil_analysis.analyze_matrix_pencil_trace)

    # Spectrum merging, level statistics and SFF numerics live in
    # fitting/qsim/level_statistics.py (spec 7.5). Wrappers keep the historical
    # call sites, notebook usage and self.data defaulting.
    merge_spectra = staticmethod(level_statistics_analysis.merge_spectra)

    def display(self, data=None, **kwargs):
        """Per-job display, inherited. Aggregate plots live on the stage classes."""
        if data is not None:
            self.data = data
        for key, stage in (("matrix", "orthogonality"),
                           ("spectrum", "spectrum"),
                           ("phase_mod180", "calibration")):
            if key in self.data:
                raise TypeError(_stage_migration_message(stage))
        return super().display(data=self.data, **kwargs)

    hardware_parameters = staticmethod(station_floquet_hardware)


# ---------------------------------------------------------------------------
# Where the four aggregate stages went.
#
# `analyze(stage=...)` is gone. It selected between four unrelated analyses on
# one class; each is now its own Experiment with its own analyze and display.
# This map exists so the error message can name the replacement, and so
# consumers can resolve a stage programmatically. It is not a forwarding shim:
# nothing here re-exports a method under its old address.
STAGE_CLASSES = {
    "calibration": "experiments.qsim.deprecated.legacy_mbr.MBRPhaseCorrectionExperiment",
    "orthogonality": "experiments.qsim.deprecated.legacy_mbr.MBROrthogonalityExperiment",
    "propagator": "experiments.qsim.deprecated.legacy_mbr.MBRPropagatorExperiment",
    "spectrum": "experiments.qsim.deprecated.legacy_mbr.MBRSpectrumExperiment",
}


def _stage_migration_message(stage):
    """Name the replacement class, so the traceback is the migration note."""
    target = STAGE_CLASSES.get(stage)
    if target is None:
        return (f"unknown stage {stage!r}; the aggregate analyses are now "
                f"separate Experiments: {sorted(STAGE_CLASSES)}")
    module, _, name = target.rpartition(".")
    return (
        f"stage={stage!r} is gone. Use {name} instead:\n"
        f"    from {module} import {name}\n"
        f"    expt = {name}.from_job_files(paths)   # or .from_job_ids(...)\n"
        f"    expt.analyze()\n"
        f"    expt.display()\n"
        f"See analysis_notebooks/guan/MBR_analysis.py for a worked example.")

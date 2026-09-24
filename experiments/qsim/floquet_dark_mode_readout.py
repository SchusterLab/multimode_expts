# -*- coding: utf-8 -*-
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
from experiments.floquet_timing import FLUX_HIGH_THRESHOLD_MHZ
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

    @staticmethod
    def _first_scalar(value):
        """
        This is to avoid a conflict due to different data type of 
        self Kerr. Sometimes it is stored as a list, sometimes as a float...
        """
        
        values = np.asarray(value).reshape(-1)
        if len(values) == 0:
            raise ValueError("saved scalar config is empty")
        return float(values[0])

    @staticmethod
    def _saved_detunings(ecfg, mode_count):
        """
        This is to avoid a conflict due to different data type of 
        detuning. Sometimes it is stored as a list, sometimes as an np array...
        """
        
        detunings = ecfg.get("detunings", None)
        if detunings is None or detunings is False or np.asarray(detunings).size == 0:
            detunings = [0.] * mode_count
        return np.asarray(detunings, dtype=float)

    @classmethod
    def _saved_parameters(cls, expts, station=None):
        """
        Check whether all the sister expts have the same params,
        and return the params as an AttrDict.

        The Floquet timing comes from each child's ``prog``. During acquisition
        that is the live compiled program; for saved data it is the stand-in
        that :mod:`experiments.saved_jobs` attaches, carrying timing recovered
        from the file's ``derived_params`` attribute or from the versioned
        config. There is no station fallback, deliberately -- see below.
        
        Returning params are:
            - `swap_stors`
            - `detunings`
            - `mode_labels`
            - `hardware_parameters`
                - `floquet_cycles_us`
                - `couplings_MHz`
                - `physical_kerr_MHz`
        """
        
        first_cfg = expts[0].cfg
        first_expt_cfg = first_cfg.expt
        swap_stors = [int(stor) for stor in first_expt_cfg.swap_stors]
        detunings = cls._saved_detunings(first_expt_cfg, len(swap_stors))
        physical_kerr_MHz = -abs(cls._first_scalar(first_cfg.device.manipulate.kerr))
        if len(detunings) != len(swap_stors) or not np.all(np.isfinite(detunings)):
            raise ValueError("saved detunings do not match swap_stors")

        program_hardware = []
        sync_cycles = int(first_expt_cfg.get("scramble_sync_cycles", 10))
        floquet_gauss_sigma = first_expt_cfg.get("floquet_gauss_sigma", None)
        floquet_waveform = first_expt_cfg.get("floquet_waveform", None)
        for expt in expts:
            cfg = expt.cfg
            ecfg = cfg.expt
            prog = getattr(expt, "prog", None)
            if prog is not None and hasattr(prog, "calculate_floquet_cycle_us") and hasattr(prog, "m1s_pi_fracs"):
                floquet_cycle_us = float(prog.calculate_floquet_cycle_us())
                pi_fracs = np.asarray([prog.m1s_pi_fracs[stor - 1] for stor in swap_stors], dtype=float)
                couplings_MHz = 1. / (4. * pi_fracs * floquet_cycle_us)
                # `source` says where the timing came from: a live compiled
                # program during acquisition, or one of the recovered sources
                # that experiments.saved_jobs resolves offline.
                program_hardware.append((floquet_cycle_us, couplings_MHz,
                                         getattr(prog, "source", "saved program")))
                break

        if program_hardware:
            floquet_cycle_us, couplings_MHz, hardware_source = program_hardware[0]
        else:
            # Deliberately no station fallback. Asking the *current* station
            # substitutes today's calibration for the historical one, and does
            # it silently: when the swap dataset moved gauss_sigma 0.04 -> 0.02
            # us between 2026-08-14 and 08-25, that fallback returned roughly
            # half the correct cycle time and every energy with it. The cycle
            # time is not a measurement -- it is computed from immutable
            # versioned config, so it is recovered exactly or not at all.
            raise RuntimeError(
                "no Floquet timing on these children. Load them through "
                "experiments.saved_jobs (from_job_ids / from_job_files), which "
                "reads the file's own 'derived_params' attribute or recomputes "
                "the timing from the versioned config, and takes timing= for "
                "files that have neither.")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0. or not np.all(np.isfinite(couplings_MHz)) or np.min(couplings_MHz) <= 0.:
            raise ValueError("saved Floquet hardware parameters must be finite and positive")
        hardware = AttrDict(dict(floquet_cycle_us=float(floquet_cycle_us), couplings_MHz=np.asarray(couplings_MHz), physical_kerr_MHz=physical_kerr_MHz, source=hardware_source))
        return AttrDict(dict(swap_stors=swap_stors, 
                             detunings=detunings, 
                             mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors], 
                             hardware=hardware))

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

    @staticmethod
    def hardware_parameters(station, 
                            swap_stors, 
                            sync_cycles, 
                            floquet_gauss_sigma=None,
                            floquet_waveform=None):
        """
        Returns hardware related physical paramters such as
            - floquet_cycle_us: time for a single floquet cycle in a microsecond
            - couplings_MHz: an array of effective BS coupling between man and stor
            - physical_kerr_MHz: self Kerr on a central mode (manipulate)
        All the values are calculated from the config/expt_cfg input
        """
        
        if (isinstance(sync_cycles, (bool, np.bool_))
                or not isinstance(sync_cycles, (int, np.integer)) or sync_cycles < 0):
            raise ValueError("sync_cycles must be a nonnegative integer")
        ramp_sigma = station.hardware_cfg.device.manipulate.ramp_sigma
        if isinstance(ramp_sigma, (list, tuple, np.ndarray)):
            ramp_sigma = ramp_sigma[0]
        pulse_us = []
        pi_fracs = []
        cycle_tproc_cycles = 0
        for stor in swap_stors:
            pulse_name = f"M1-S{stor}"
            if station.ds_floquet.get_freq(pulse_name) < FLUX_HIGH_THRESHOLD_MHZ:
                gen_ch = station.hardware_cfg.hw.soc.dacs.flux_low.ch[0]
            else:
                gen_ch = station.hardware_cfg.hw.soc.dacs.flux_high.ch[0]
            waveform = floquet_waveform if floquet_waveform is not None else station.ds_floquet.get_waveform(pulse_name)
            # Match calculate_floquet_cycle_us: round each envelope segment
            # to its generator clock before adding the tProc sync interval.
            if waveform in ("gauss", "gaussian", "arb"):
                sigma = floquet_gauss_sigma
                if sigma is None:
                    sigma = station.ds_floquet.get_gauss_sigma(pulse_name)
                sigma_cycles = station.soccfg.us2cycles(sigma, gen_ch=gen_ch)
                pulse_cycles = sigma_cycles * station.ds_floquet.get_gauss_n_sigma(pulse_name)
            elif waveform == "preload_flattop":
                flat_cycles = station.soccfg.us2cycles(station.ds_floquet.get_len(pulse_name), gen_ch=gen_ch)
                ramp_cycles = station.soccfg.us2cycles(station.ds_floquet.get_ramp_sigma(pulse_name), gen_ch=gen_ch)
                pulse_cycles = flat_cycles + 6 * ramp_cycles
            else:
                flat_cycles = station.soccfg.us2cycles(station.ds_floquet.get_len(pulse_name), gen_ch=gen_ch)
                ramp_cycles = station.soccfg.us2cycles(ramp_sigma, gen_ch=gen_ch)
                pulse_cycles = flat_cycles + 6 * ramp_cycles
            pulse_us.append(station.soccfg.cycles2us(pulse_cycles, gen_ch=gen_ch))
            clock_ratio = float(station.soccfg["tprocs"][0]["f_time"]) / float(station.soccfg["gens"][gen_ch]["f_fabric"])
            cycle_tproc_cycles += int(pulse_cycles * clock_ratio + sync_cycles)
            pi_fracs.append(station.ds_floquet.get_pi_frac(pulse_name))

        # Match the integer synci advances, not the unquantized pulse+gap sum.
        floquet_cycle_us = station.soccfg.cycles2us(cycle_tproc_cycles)
        if not np.all(np.isfinite(pulse_us + pi_fracs)) or min(pulse_us + pi_fracs) <= 0.:
            raise ValueError("Floquet pulse lengths and pi fractions must be finite and positive")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0.:
            raise ValueError("Floquet cycle duration must be finite and positive")

        # 2 * pi * g * t_swap = pi / 2 -> g = 1/ 4/ t_swap
        # pi_frac repetitions of each pulse+sync block make a full swap:
        # g_{bare} = 1/ 4 / n_frac / (t_pulse + t_sync)
        # len(swap_stors) -> g_{eff} * T_F = g_{bare} * (t_pulse+t_sync) = 1/4/n_frac
        # So g_{eff} = 1/4/n_frac/T_F
        couplings_MHz = [1. / (4. * pi_frac * floquet_cycle_us) for pi_frac in pi_fracs]
        physical_kerr_MHz = station.hardware_cfg.device.manipulate.kerr
        if isinstance(physical_kerr_MHz, (list, tuple, np.ndarray)):
            physical_kerr_MHz = physical_kerr_MHz[0]
        physical_kerr_MHz = -abs(physical_kerr_MHz)
        if not np.isfinite(physical_kerr_MHz):
            raise ValueError("physical Kerr must be finite")
        return AttrDict(dict(floquet_cycle_us=floquet_cycle_us,
                             couplings_MHz=np.asarray(couplings_MHz),
                             physical_kerr_MHz=physical_kerr_MHz))


# ---------------------------------------------------------------------------
# Where the four aggregate stages went.
#
# `analyze(stage=...)` is gone. It selected between four unrelated analyses on
# one class; each is now its own Experiment with its own analyze and display.
# This map exists so the error message can name the replacement, and so
# consumers can resolve a stage programmatically. It is not a forwarding shim:
# nothing here re-exports a method under its old address.
STAGE_CLASSES = {
    "calibration": "experiments.qsim.legacy_mbr.MBRPhaseCorrectionExperiment",
    "orthogonality": "experiments.qsim.legacy_mbr.MBROrthogonalityExperiment",
    "propagator": "experiments.qsim.legacy_mbr.MBRPropagatorExperiment",
    "spectrum": "experiments.qsim.legacy_mbr.MBRSpectrumExperiment",
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


# ---------------------------------------------------------------------------
# Compatibility: measurement families that moved to their own modules.
#
# The acquisition notebooks address programs as
# ``meas.qsim.floquet_dark_mode_readout.<Name>``, so the old attribute has to
# keep resolving. A module-level ``__getattr__`` (PEP 562) does that lazily,
# which matters: every new module imports the still-resident base classes from
# here, so a top-level re-import would be circular.
#
# Values are absolute module paths: the retired programs sit under
# ``deprecated/``, not directly under ``experiments.qsim``.
#
# Note: ``__dir__`` advertises these names, so the flattening exporter in
# ``experiments/__init__.py`` does re-export them to the ``experiments``
# namespace. That is harmless -- it resolves to the very same class object the
# defining module exports, so the second write is idempotent.
_MOVED_TO = {
    "BroadbandGeValidationProgram": "experiments.qsim.dark_mode_broadband_ge_validation",
    "EncodingOrthogonalityProgram": "experiments.qsim.mbr_orthogonality",
    "EncodingPropagatorProgram": "experiments.qsim.mbr_propagator",
    "EncodingStarkShiftCalibrationProgram": "experiments.qsim.floquet_phase_calibration",
    "EntireFloquetCyclePhaseCalibrationProgram": "experiments.qsim.mbr_phase_correction",
    "FloquetPhaseAccumulationProgram": "experiments.qsim.floquet_phase_calibration",
    "NPhotonHamiltonianSpectroscopyProgram": "experiments.qsim.mbr_spectroscopy_program",
    "SidebandScrambleDarkProgramNewNew": "experiments.qsim.mbr_spectroscopy_program",
    "SinglePhotonFloquetSpectroscopyProgram":
        "experiments.qsim.deprecated.single_photon_spectroscopy",
    "KerrWaitProgramDark": "experiments.qsim.deprecated.dark_scramble_legacy",
    "ManStorScrambleProgram": "experiments.qsim.deprecated.dark_scramble_legacy",
    "SidebandScrambleDarkProgram": "experiments.qsim.deprecated.dark_scramble_legacy",
    "SidebandScrambleDarkProgramDebug": "experiments.qsim.deprecated.dark_scramble_legacy",
    "SidebandScrambleDarkProgramNew": "experiments.qsim.deprecated.dark_scramble_legacy",
    "DarkT1Experiment": "experiments.qsim.dark_mode_t1",
    "DisorderSFFDepthSweepProgram": "experiments.qsim.mbr_sff",
    "DisorderSFFExperiment": "experiments.qsim.mbr_sff",
    "DisorderSFFSequenceMixin": "experiments.qsim.mbr_sff",
    "HardwareFloquetDepthSweepMixin": "experiments.qsim.mbr_sff",
    "DarkT1Program": "experiments.qsim.dark_mode_t1",
    "FloquetDisplacementKerrExperiment": "experiments.qsim.floquet_displacement_kerr",
    "FloquetDisplacementKerrProgram": "experiments.qsim.floquet_displacement_kerr",
    "ManStorMultiparityChevronRExperiment": "experiments.qsim.dark_mode_multiparity_chevron",
    "ManStorMultiparityChevronRProgram": "experiments.qsim.dark_mode_multiparity_chevron",
    "SidebandStarkAmplificationModifiedProgram": "experiments.qsim.sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_newold": "experiments.qsim.sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_old": "experiments.qsim.sideband_stark_shift_cal",
    "StorageSwapPhaseAccumulationProgram": "experiments.qsim.storage_swap_phase_cal",
}


def __getattr__(name):
    module = _MOVED_TO.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(list(globals()) + list(_MOVED_TO))

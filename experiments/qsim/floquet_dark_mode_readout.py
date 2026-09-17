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
        """Collect saved jobs for analysis; station supplies missing hardware data."""
        
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
    def from_batch(cls, aggregate, station=None):
        """Re-wrap a ``BatchRunner`` aggregate as *this* stage's Experiment.

        ``BatchRunner`` returns an instance of its ``ExptClass``, which is the
        class each job was recorded under -- provenance, so acquisition cells
        must keep passing whatever they always passed. Aggregate analysis, on
        the other hand, now belongs to a stage class. This converts one to the
        other and re-reads nothing: the loaded child Experiments are reused
        as-is.

            raw = runner.execute(batch.configs, batch_size=10)
            expt = MBRPhaseCorrectionExperiment.from_batch(raw)
            expt.analyze(); expt.display()

        The job IDs and the analysis station come along, so the usual next
        line -- ``expt.batch_job_ids`` -- keeps working.
        """
        if not hasattr(aggregate, "batch_expts"):
            raise TypeError(
                "from_batch expects a BatchRunner aggregate (one with "
                f"batch_expts); got {type(aggregate).__name__}. To load saved "
                "jobs instead, use from_job_ids or from_job_files.")
        if station is None:
            station = getattr(aggregate, "_analysis_station", None)
        return cls._from_expts(aggregate.batch_expts,
                               job_ids=getattr(aggregate, "batch_job_ids", None),
                               station=station)

    @classmethod
    def from_job_files(cls, job_files, station=None):
        """
        Load child experiment pickle/H5 files into one analysis object.
        """
        import pickle
        from pathlib import Path

        if isinstance(job_files, (str, Path)):
            job_files = [job_files]
        expts = []
        for job_file in flatten_exp_lists(job_files):
            if not isinstance(job_file, (str, Path)):
                expts.append(job_file)
                continue
            path = Path(job_file)
            if path.suffix.lower() in (".h5", ".hdf5"):
                expts.append(cls.from_h5file(str(path)))
            else:
                with path.open("rb") as handle:
                    expts.append(pickle.load(handle))
        return cls._from_expts(expts, station=station)

    @classmethod
    def from_job_ids(cls, job_ids, client=None, station=None):
        """
        Load completed queue jobs without rebuilding a BatchRunner.
        If station is properly specified (along with project name and directory (which is hardcoded)),
        a list of job_ids is okay. Otherwise, a complete directory is necessary.
        """
        if isinstance(job_ids, (str, int, np.integer)):
            job_ids = [job_ids]
        job_ids = [str(job_id) for job_id in flatten_exp_lists(job_ids)]
        if not job_ids:
            raise ValueError("job_ids cannot be empty")
        if client is None:
            if station is None:
                raise ValueError("client or station is required to resolve job IDs")
            job_files = [station.expt_objs_path / f"{job_id}_expt.pkl" for job_id in job_ids]
            aggregate = cls.from_job_files(job_files, station=station)
            aggregate.batch_job_ids = job_ids
            return aggregate
        expts = []
        for job_id in job_ids:
            result = client.get_status(job_id)
            if not result.is_successful():
                raise RuntimeError(f"Job {job_id} {result.status}: {result.error_message or 'No details'}")
            expts.append(result.load_expt())
        return cls._from_expts(expts, job_ids=job_ids, station=station)

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
        If the class is called for post processing using hdf5 files,
        one should specify station with proper config as well.
        
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
                program_hardware.append((floquet_cycle_us, couplings_MHz))
                break

        if program_hardware:
            floquet_cycle_us, couplings_MHz = program_hardware[0]
            hardware_source = "saved program"
        elif station is not None:
            hardware = cls.hardware_parameters(station, 
                                               swap_stors, 
                                               sync_cycles, 
                                               floquet_gauss_sigma, 
                                               floquet_waveform)
            floquet_cycle_us, couplings_MHz = hardware.floquet_cycle_us, hardware.couplings_MHz
            hardware_source = "current station (H5 fallback)"
        else:
            raise RuntimeError("Floquet cycle time and couplings need the job pickle or station when loading H5 files")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0. or not np.all(np.isfinite(couplings_MHz)) or np.min(couplings_MHz) <= 0.:
            raise ValueError("saved Floquet hardware parameters must be finite and positive")
        hardware = AttrDict(dict(floquet_cycle_us=float(floquet_cycle_us), couplings_MHz=np.asarray(couplings_MHz), physical_kerr_MHz=physical_kerr_MHz, source=hardware_source))
        return AttrDict(dict(swap_stors=swap_stors, 
                             detunings=detunings, 
                             mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors], 
                             hardware=hardware))

    def analyze(self, data=None, **kwargs):
        """Analyze one job's preparation-phase differences.

        Four-phase spectroscopy jobs also produce ``cycles`` and the complex
        return ``A = Q_0 - i Q_90``, including the saved analysis correction.
        Legacy jobs with one analyzer phase still produce only ``Q_phi``.

        The four aggregate stages that used to hide behind ``stage=`` are now
        separate classes; see :data:`STAGE_CLASSES` and the migration table in
        ``analysis_notebooks/guan/MBR_analysis.py``.
        """
        if "stage" in kwargs:
            raise TypeError(_stage_migration_message(kwargs["stage"]))
        if data is not None:
            self.data = data
        self._quadrature(self)
        if "spectroscopy_phase_id" in self.cfg.expt.get("swept_params", []):
            self.data["cycles"] = np.asarray(self.data["ypts"])
            self.data["A"] = self._complex_return(self)
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
        if "spectroscopy_phase_id" in expt.cfg.expt.get("swept_params", []):
            phase_pairs = np.asarray(
                expt.cfg.expt.spectroscopy_phase_combinations, dtype=float
            )[np.asarray(theta, dtype=int)] 
            # the above is actually identical to spectroscopy_phase_combinations
            # but coded just in case phase_id order gets changed in future.
            quadratures = []
            for analyzer in (0., 90.):
                #logic
                # 1. np.isclose returns [[True, False], ...]
                # 2. np.all wrt axis =1 gives [True, False, ...]
                # 3. np.flatnonzero gives the index of Ture. 
                # so prep0[0] and prep180[0] is integer index
                prep0 = np.flatnonzero(np.all(
                    np.isclose(phase_pairs, [0., analyzer]), axis=1))
                prep180 = np.flatnonzero(np.all(
                    np.isclose(phase_pairs, [180., analyzer]), axis=1))
                if len(prep0) != 1 or len(prep180) != 1:
                    raise ValueError("spectroscopy needs each of the four phase combinations once")
                quadratures.append(
                    expt.data["Pe"][:, prep0[0]] - expt.data["Pe"][:, prep180[0]])
            expt.data["return_quadrature"] = np.column_stack(quadratures) #dim: [cycles, analyzer_phase]
        else:
            expt.data["return_quadrature"] = expt.data["Pe"][:, 0] - expt.data["Pe"][:, 1]
        return expt.data["return_quadrature"]

    @staticmethod
    def _complex_return(expt):
        """Reconstruct a complete four-phase job, including legacy off-diagonal jobs."""
        ecfg = expt.cfg.expt
        quadratures = np.asarray(
            EncodingHamiltonianSpectroscopyExperiment._quadrature(expt)
        ) 
        if "spectroscopy_phase_id" in ecfg.get("swept_params", []):
            cycles = np.asarray(expt.data["ypts"])
            phase = float(ecfg.get("spectroscopy_analysis_phase_per_cycle_deg", 0.))
        else:
            quadratures.reshape(-1, 2)
            cycles = np.asarray(ecfg.offdiag_cycles)
            phase = float(ecfg.offdiag_decoder_phase_correction_deg)
        A = quadratures[:, 0] - 1j * quadratures[:, 1]
        return A * np.exp(-1j * np.deg2rad(phase * cycles)) #postprocessing if phase is nonzero

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
    "calibration": "experiments.qsim.mbr_phase_correction"
                   ".MBRPhaseCorrectionExperiment",
    "orthogonality": "experiments.qsim.mbr_orthogonality"
                     ".MBROrthogonalityExperiment",
    "propagator": "experiments.qsim.mbr_propagator.MBRPropagatorExperiment",
    "spectrum": "experiments.qsim.mbr_spectrum.MBRSpectrumExperiment",
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
# Values are absolute module paths: the destinations are no longer all under
# ``experiments.qsim`` (BatchRunner is infrastructure and sits at the top
# level, and the retired programs sit under ``deprecated/``).
#
# Note: ``__dir__`` advertises these names, so the flattening exporter in
# ``experiments/__init__.py`` does re-export them to the ``experiments``
# namespace. That is harmless -- it resolves to the very same class object the
# defining module exports, so the second write is idempotent.
_MOVED_TO = {
    "BatchRunner": "experiments.batch_runner",
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

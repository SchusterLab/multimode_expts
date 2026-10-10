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

from experiments.qsim.qsim_base import QsimExperiment, QsimProgram
from experiments.qsim.sideband_scramble import SidebandScrambleProgram
from experiments.qsim.qsim_base import QsimExperiment, QsimRProgram, classify_two_parity_readouts
from experiments.qsim.dark_mode_encoding import DarkModeProgram
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
    "DarkModeScrambleProgram": "experiments.qsim.mbr_spectroscopy_program",
    "SinglePhotonFloquetSpectroscopyProgram":
        "experiments.qsim.deprecated.single_photon_spectroscopy",
    "KerrWaitProgramDark": "experiments.qsim.deprecated.dark_scramble_legacy",
    "ManStorScrambleProgram": "experiments.qsim.deprecated.dark_scramble_legacy",
    "SidebandScrambleDarkProgram": "experiments.qsim.deprecated.dark_scramble_legacy",
    "SidebandScrambleDarkProgramDebug": "experiments.qsim.deprecated.dark_scramble_legacy",
    "SidebandScrambleDarkProgramNew": "experiments.qsim.deprecated.dark_scramble_legacy",
    "DarkT1Experiment": "experiments.qsim.dark_mode_t1",
    "DarkT1Program": "experiments.qsim.dark_mode_t1",
    "FloquetDisplacementKerrExperiment": "experiments.qsim.floquet_displacement_kerr",
    "FloquetDisplacementKerrProgram": "experiments.qsim.floquet_displacement_kerr",
    "ManStorMultiparityChevronRExperiment": "experiments.qsim.dark_mode_multiparity_chevron",
    "ManStorMultiparityChevronRProgram": "experiments.qsim.dark_mode_multiparity_chevron",
    "StorageSwapStarkPhaseProgram": "experiments.qsim.sideband_stark_shift_cal",
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

#--------------------------------------------------
#------- temporary amendment for adiabatic gauge potential measurement

from numbers import Integral

from experiments.dataset import StorageManSwapDataset
from experiments.qsim.floquet_train import FloquetProgram

# Definite Fock preparation and population readout.


def validate_occupations(occupations, swap_stors):
    """Check the supported total-photon <= 3, M1 plus storage input."""
    if len(occupations) != len(swap_stors) + 1:
        raise ValueError("occupations must be [n_M1] followed by swap_stors")
    if len(set(swap_stors)) != len(swap_stors):
        raise ValueError("swap_stors must contain distinct storage modes")
    if any(isinstance(stor, bool) or not isinstance(stor, Integral)
           or stor not in range(1, 8) for stor in swap_stors):
        raise ValueError("storage modes must be in 1..7")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
           for n in occupations):
        raise ValueError("occupations must contain non-negative integers")
    if sum(occupations) > 3:
        raise ValueError("mod-4 outcomes identify photon number only for total N <= 3")


def with_storage_waits(pulses, wait_us):
    """Copy a gate list, adding the existing settling wait after each swap."""
    result = []
    for pulse in pulses:
        result.append(list(pulse))
        if pulse[0] == "storage" and wait_us > 0:
            result.append(["wait", wait_us])
    return result


class FockPopulationProgram(FloquetProgram):
    """The ordinary Qsim reset/readout template with a population experiment core.

    ``occupations`` is [n_M1, n_S...] in ``swap_stors`` order. ``ro_stor`` is
    0 for M1. ``readout_route`` is hub, direct_swap, or dump_then_swap.
    Storage routes require ``readout_swap_pulses`` (gate descriptions).
    ``use_multiphoton_swap`` selects known-N storage PREPARATION rows only.
    """

    @classmethod
    def readouts_per_shot(cls, cfg):
        # The Experiment asks for the buffer size before initialize() runs.
        if cfg.expt.get("readout") != "multiparity":
            raise ValueError("set readout='multiparity' before acquisition")
        return super().readouts_per_shot(cfg)

    def initialize(self):
        self._set_population_defaults()
        self._validate_population_config()
        expt = self.cfg.expt
        self.preparation_pulses = self.build_fock_preparation(
            expt.occupations, expt.swap_stors,
            expt.get("use_multiphoton_swap", False),
            expt.storage_pulse_wait_us)
        self.readout_pulses = with_storage_waits(
            expt.readout_swap_pulses, expt.storage_pulse_wait_us)
        super().initialize()

    def _set_population_defaults(self):
        defaults = dict(prepulse=False, postpulse=False, init_stor=0, ro_stor=0,
                        readout="multiparity", readout_route="hub",
                        readout_swap_pulses=[], parity_fast=False,
                        cond_sec_phase=-90, storage_pulse_wait_us=0.2,
                        readout_dump_mode=2)
        for name, value in defaults.items():
            self.cfg.expt.setdefault(name, value)

    def _validate_population_config(self):
        expt = self.cfg.expt
        validate_occupations(expt.occupations, expt.swap_stors)
        if expt.prepulse or expt.postpulse:
            raise ValueError("population prepulse and postpulse must be False")
        if expt.readout != "multiparity":
            raise ValueError("FockPopulationProgram requires readout='multiparity'")
        if expt.get("man_mode_no", 1) != 1:
            raise ValueError("the calibrated preparation path uses M1")
        if expt.storage_pulse_wait_us < 0:
            raise ValueError("storage_pulse_wait_us must be non-negative")
        self._validate_readout_route()

    def _validate_readout_route(self):
        expt = self.cfg.expt
        if expt.readout_route == "hub":
            if expt.ro_stor != 0 or expt.readout_swap_pulses:
                raise ValueError("hub readout needs ro_stor=0 and no transfer pulses")
        elif expt.readout_route in ("direct_swap", "dump_then_swap"):
            if expt.ro_stor not in expt.swap_stors:
                raise ValueError("storage readout ro_stor must belong to swap_stors")
            if not expt.readout_swap_pulses:
                raise ValueError("supply a fixed readout_swap_pulses sequence explicitly")
        else:
            raise ValueError("unknown readout_route: " + str(expt.readout_route))

    def build_fock_preparation(self, occupations, swap_stors,
                               use_multiphoton_swap=False,
                               storage_pulse_wait_us=0.2):
        """Load initially empty storages once, then load M1 last."""
        validate_occupations(occupations, swap_stors)
        pulses = []
        for storage, number in zip(swap_stors, occupations[1:]):
            if number == 0:
                continue
            pulses.extend(self.prep_man_fock_state(1, str(number)))
            row = f"M1-S{storage}"
            if use_multiphoton_swap:
                row = StorageManSwapDataset.multiphoton_swap_name(row, number)
            pulses.append(["storage", row, "pi", 0.0])
        pulses.extend(self.prep_man_fock_state(1, str(occupations[0])))
        return with_storage_waits(pulses, storage_pulse_wait_us)

    def _play_gate_list(self, pulses, prefix):
        if not pulses:
            return
        creator = self.get_prepulse_creator(deepcopy(pulses))
        self.sync_all()
        self.custom_pulse(self.cfg, creator.pulse, prefix=prefix)
        self.sync_all()

    def transfer_storage_for_readout(self):
        """Apply the caller's fixed transfer; never infer final photon number."""
        expt = self.cfg.expt
        if expt.readout_route == "hub":
            return
        if expt.readout_route == "dump_then_swap":
            self.man_reset(man_idx=1, dump_mode_idx=expt.readout_dump_mode,
                           chi_dressed=False)
        self._play_gate_list(self.readout_pulses, prefix="population_readout_")

    def core_pulses(self):
        self._play_gate_list(self.preparation_pulses, prefix="population_prep_")
        storages = list(self.cfg.expt.swap_stors)
        self.phase_offsets = [0.0] * len(storages)
        self.disorder_phase_offsets = [0.0] * len(storages)
        self._play_scramble_with_phase_offsets(
            self.phase_offsets, storages, self.disorder_phase_offsets)
        self.transfer_storage_for_readout()

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
    "EncodingStarkShiftCalibrationProgram": "experiments.qsim.floquet_phase_calibration",
    "FloquetPhaseAccumulationProgram": "experiments.qsim.floquet_phase_calibration",
    "SidebandScrambleDarkProgramNewNew": "experiments.qsim.mbr_spectroscopy_program",
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

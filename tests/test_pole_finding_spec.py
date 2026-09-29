"""Every function docs/qsim/pole_finding.md names exists (spec 9, rule 6).

A name is a backticked snake_case identifier. It must be defined in a module of
``fitting.qsim.poles`` or ``fitting.qsim``, unless it is a variable or field
(NOT_FUNCTIONS) or belongs to a later phase (LATER_PHASES; move it out when written).
"""
import importlib
import pkgutil
import re
from pathlib import Path

import fitting.qsim
import fitting.qsim.poles

SPEC = Path(__file__).parents[1] / "docs" / "qsim" / "pole_finding.md"

NOT_FUNCTIONS = {"row_groups", "sigma_cal", "delta_g", "decays_per_us", "phase_error"}
EXTERNAL = {"find_peaks"}  # scipy.signal
LATER_PHASES = {
    "refine_with_group_offsets", "frequency_errors",          # fitter C, phase 3
    "row_poles", "cluster_row_poles",                         # fitter D, phase 3
}


def spec_function_names():
    return set(re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", SPEC.read_text(encoding="utf-8")))


def defined_names():
    names = set()
    for package in (fitting.qsim.poles, fitting.qsim):
        for module in pkgutil.iter_modules(package.__path__):
            names |= set(vars(importlib.import_module(f"{package.__name__}.{module.name}")))
    return names


def test_every_function_the_spec_names_exists():
    missing = spec_function_names() - NOT_FUNCTIONS - EXTERNAL - LATER_PHASES - defined_names()
    assert not missing


def test_later_phase_names_are_still_in_the_spec_and_not_yet_written():
    assert LATER_PHASES <= spec_function_names()
    assert not LATER_PHASES & defined_names()

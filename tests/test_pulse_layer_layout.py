# -*- coding: utf-8 -*-
"""Which implementation each dark-mode program actually runs.

The pulse base was split into mixins, and the one thing a mixin split can get
wrong without any test noticing is *method resolution*: the program still
builds, still compiles, still acquires, and plays a different reset.

The ASM golden covers the four MBR stages, so a resolution change on their
path shows up there. These tests cover the case the golden cannot see -- a
method that only some configurations reach.
"""
import pytest

from experiments.MM_base import MM_base
from experiments.qsim.floquet_dark_mode_readout import (
    DarkBaseProgram,
    DarkBaseRProgram,
)
from experiments.qsim.manipulate_mode_pulses import ManipulateModePulses


def test_averager_base_plays_the_dark_man_reset():
    """``DarkBaseProgram`` overrides MM_base's reset, as it always has."""
    assert DarkBaseProgram.man_reset is ManipulateModePulses.man_reset


def test_raverager_base_keeps_mm_base_man_reset():
    """The asymmetry that inheriting the mixin would have destroyed.

    ``DarkBaseRProgram`` borrowed only two of the three manipulate-mode
    methods, so its ``active_reset`` path plays MM_base's ``man_reset``. If
    someone later swaps the explicit assignments for inheritance, the
    RAverager programs silently change which reset they emit -- with no
    failure anywhere else in the suite.
    """
    assert DarkBaseRProgram.man_reset is MM_base.man_reset
    assert DarkBaseRProgram.man_reset is not ManipulateModePulses.man_reset


@pytest.mark.parametrize("method", ["prep_man_fock_state",
                                    "multi_parity_readout"])
def test_both_bases_share_the_other_two(method):
    assert (getattr(DarkBaseProgram, method)
            is getattr(ManipulateModePulses, method))
    assert (getattr(DarkBaseRProgram, method)
            is getattr(ManipulateModePulses, method))

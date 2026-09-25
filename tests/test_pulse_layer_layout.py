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
from experiments.qsim.dark_base import DarkBaseProgram, DarkBaseRProgram
from experiments.qsim.manipulate_mode_pulses import ManipulateModePulses


@pytest.mark.parametrize("name", ["DarkBaseExperiment", "DarkBaseProgram",
                                  "DarkBaseRProgram",
                                  "classify_two_parity_readouts",
                                  "flatten_exp_lists"])
def test_the_old_god_module_address_still_resolves(name):
    """The acquisition notebooks address these through the god module.

    ``meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment`` is what the
    submission cells say, and saved files are named after the class, so the
    old address has to keep working after the move.
    """
    from experiments.qsim import floquet_dark_mode_readout as fdmr
    import experiments.qsim.dark_base as dark_base
    import experiments.qsim.utils as utils

    moved = getattr(fdmr, name)
    owner = utils if name == "flatten_exp_lists" else dark_base
    assert moved is getattr(owner, name)


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


def test_mbr_base_plays_the_dark_man_reset():
    """The MBR jobs left the dark-mode chain in step 8A1 but kept its reset.

    Through ``DarkBaseProgram`` they always played ``ManipulateModePulses``'s
    ``man_reset``. The ASM golden does not catch a change here: with its
    pinned configs, MM_base's ``man_reset`` compiles to the same program
    (checked by mutation, 2026-09-25). So this test is the only net.
    """
    from experiments.qsim.mbr_ramsey import MBRRamseyProgram

    assert MBRRamseyProgram.man_reset is ManipulateModePulses.man_reset


def test_mbr_base_is_not_built_on_the_dark_mode_chain():
    from experiments.qsim.mbr_ramsey import MBRJobExperiment, MBRRamseyProgram
    from experiments.qsim.dark_base import DarkBaseExperiment
    from experiments.qsim.sideband_scramble import SidebandScrambleProgram

    assert DarkBaseProgram not in MBRRamseyProgram.__mro__
    assert SidebandScrambleProgram not in MBRRamseyProgram.__mro__
    assert DarkBaseExperiment not in MBRJobExperiment.__mro__

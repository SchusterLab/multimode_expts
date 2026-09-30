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
from experiments.qsim.dark_mode_encoding import DarkModeProgram
from experiments.qsim.qsim_base import QsimProgram, QsimRProgram


@pytest.mark.parametrize("name, owner", [
    ("QsimExperiment", "experiments.qsim.qsim_base"),
    ("QsimProgram", "experiments.qsim.qsim_base"),
    ("QsimRProgram", "experiments.qsim.qsim_base"),
    ("classify_two_parity_readouts", "experiments.qsim.qsim_base"),
    ("flatten_exp_lists", "experiments.qsim.utils"),
    ("DarkModeScrambleProgram", "experiments.qsim.mbr_spectroscopy_program"),
    ("StorageSwapStarkPhaseProgram", "experiments.qsim.sideband_stark_shift_cal"),
])
def test_the_old_god_module_address_still_resolves(name, owner):
    """The acquisition notebooks address these through the god module.

    ``meas.qsim.floquet_dark_mode_readout.<Name>`` is what many submission
    cells say, so the address has to resolve to the defining module's class.
    Since step 10F the names are the new ones (DarkBaseExperiment and
    DarkBaseProgram were renamed, and ``dark_base`` is gone).
    """
    import importlib

    from experiments.qsim import floquet_dark_mode_readout as fdmr

    assert getattr(fdmr, name) is getattr(importlib.import_module(owner), name)


def test_averager_bases_play_the_qsim_man_reset():
    """``QsimProgram`` overrides MM_base's reset; ``DarkModeProgram`` inherits it.

    Until step 10B only the DarkBase (and MBR) programs had the override, from
    the ``ManipulateModePulses`` mixin; the ~25 QsimBase leaves played
    MM_base's. The two differ only by ``dump_reset_iter_num`` (default 1: the
    same pulses; the program golden checks it under active reset).
    """
    assert "man_reset" in vars(QsimProgram)
    assert QsimProgram.man_reset is not MM_base.man_reset
    assert DarkModeProgram.man_reset is QsimProgram.man_reset


def test_raverager_base_keeps_mm_base_man_reset():
    """The asymmetry that inheriting the mixin would have destroyed.

    ``QsimRProgram`` (``DarkBaseRProgram`` until 10F) borrows only two of the three manipulate-mode
    methods, so its ``active_reset`` path plays MM_base's ``man_reset``. If
    someone later swaps the explicit assignments for inheritance, the
    RAverager programs silently change which reset they emit -- with no
    failure anywhere else in the suite.
    """
    assert QsimRProgram.man_reset is MM_base.man_reset
    assert QsimRProgram.man_reset is not QsimProgram.man_reset


@pytest.mark.parametrize("method", ["prep_man_fock_state",
                                    "multi_parity_readout"])
def test_both_bases_share_the_other_two(method):
    assert method in vars(QsimProgram)
    assert (getattr(DarkModeProgram, method)
            is getattr(QsimProgram, method))
    assert (getattr(QsimRProgram, method)
            is getattr(QsimProgram, method))


def test_mbr_base_plays_the_dark_man_reset():
    """The MBR jobs left the dark-mode chain in step 8A1 but kept its reset.

    Through ``DarkBaseProgram`` they always played the qsim ``man_reset``
    (``ManipulateModePulses`` until step 10B, ``QsimProgram`` since). The ASM
    golden does not catch a change here: with its pinned configs, MM_base's ``man_reset`` compiles to the same program
    (checked by mutation, 2026-09-25). So this test is the only net.
    """
    from experiments.qsim.mbr_ramsey import MBRRamseyProgram

    assert MBRRamseyProgram.man_reset is QsimProgram.man_reset


def test_mbr_base_is_not_built_on_the_dark_mode_chain():
    from experiments.qsim.mbr_ramsey import MBRRamseyProgram
    from experiments.qsim.sideband_scramble import SidebandScrambleProgram

    assert DarkModeProgram not in MBRRamseyProgram.__mro__
    assert SidebandScrambleProgram not in MBRRamseyProgram.__mro__


def test_the_floquet_chain_is_single_inheritance():
    """Step 10C: the two mixins became classes in one chain.

    QsimProgram -> FloquetProgram -> DarkModeProgram, and each live leaf
    sits on the lowest class whose methods it plays. A second parent here
    would bring back the mixin tangle the chain replaced.
    """
    from experiments.qsim.dark_mode_encoding import DarkModeProgram
    from experiments.qsim.dark_mode_t1 import DarkT1Program
    from experiments.qsim.floquet_displacement_kerr import FloquetDisplacementKerrProgram
    from experiments.qsim.floquet_train import FloquetProgram
    from experiments.qsim.mbr_ramsey import MBRRamseyProgram
    from experiments.qsim.mbr_spectroscopy_program import DarkModeScrambleProgram
    from experiments.qsim.sideband_stark_shift_cal import (
        StorageSwapStarkPhaseProgram,
    )

    assert FloquetProgram.__bases__ == (QsimProgram,)
    assert DarkModeProgram.__bases__ == (FloquetProgram,)
    for leaf in (MBRRamseyProgram, FloquetDisplacementKerrProgram,
                 StorageSwapStarkPhaseProgram):
        assert leaf.__bases__ == (FloquetProgram,), leaf.__name__
    for leaf in (DarkT1Program, DarkModeScrambleProgram):
        assert leaf.__bases__ == (DarkModeProgram,), leaf.__name__
    assert DarkModeProgram not in MBRRamseyProgram.__mro__

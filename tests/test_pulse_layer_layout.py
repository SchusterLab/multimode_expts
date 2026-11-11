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
from experiments.qsim.t2_cavity_fluxexcursion import SidebandScrambleDarkProgram


def test_every_qsim_program_plays_the_one_mm_base_man_reset():
    """Since 2026-11-05 there is one ``man_reset``, in MM_base.

    The copies in ``QsimProgram`` and ``SidebandScrambleDarkProgram`` were
    deleted (docs/log/2026-11-05_man-reset-trim.md); ``dump_reset_iter_num``
    moved into MM_base. A new override would again let some programs play a
    different reset, with no failure anywhere else in the suite.
    """
    assert "man_reset" not in vars(QsimProgram)
    for program in (QsimProgram, QsimRProgram,
                    DarkModeProgram, SidebandScrambleDarkProgram):
        assert program.man_reset is MM_base.man_reset, program.__name__


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
    from experiments.qsim.dark_mode_scramble import DarkModeScrambleProgram
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

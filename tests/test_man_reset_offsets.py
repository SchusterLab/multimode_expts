"""MM_base.man_reset_offsets_mhz: one M-dump pulse per qubit state, f, e, g (since 2026-11-05)."""
import pytest
from slab import AttrDict

from experiments.MM_base import MM_base


def config(chi_ge, chi_ef):
    return AttrDict(dict(device=dict(manipulate=dict(chi_ge=[chi_ge], chi_ef=[chi_ef]))))


def test_dressed_reset_plays_the_f_e_g_dump_lines_once_each():
    # The 2026-11-05 pulse-probe values for M1 (CFG-HW-20261105-00040). D2 is above M1, so the
    # M1-D2 line moves by minus M1's shift; the order follows the qubit decay f -> e -> g.
    offsets = MM_base.man_reset_offsets_mhz(config(-0.3175, -0.2759), chi_dressed=True)
    assert offsets == pytest.approx([0.3175 + 0.2759, 0.3175, 0.])


def test_undressed_reset_plays_the_bare_line_only():
    assert MM_base.man_reset_offsets_mhz(config(-0.3175, -0.2759), chi_dressed=False) == [0.]

# -*- coding: utf-8 -*-
"""The phase ledger, checked without a program.

These four functions were methods on an 1,800-line program base, so the only
way to exercise them was to compile a whole Floquet experiment. They are pure
arithmetic, and a wrong phase is the failure mode that compiles and acquires
happily, so they get direct tests.

What is pinned here is the *convention*, in both its asymmetries:
the Floquet ledger skips the mode just pulsed, the matrix ledgers do not.
Those differ on purpose (see the module docstring) and a later "cleanup" that
unified them would be a physics change.
"""
import numpy as np
import pytest

from experiments.qsim.floquet_phase_frame import (
    advance_floquet_offsets,
    advance_matrix_offsets,
    detuning_phase_deg,
    mod360,
)


class FakeSwapDataset:
    """Pairwise Stark calibration, keyed the way the CSV is."""

    def __init__(self, phases):
        self.phases = phases
        self.asked = []

    def get_phase_from(self, stor_name, from_stor_name):
        self.asked.append((stor_name, from_stor_name))
        return self.phases[(stor_name, from_stor_name)]


@pytest.mark.parametrize("raw, wrapped", [
    (0.0, 0.0), (359.9, 359.9), (360.0, 0.0), (450.0, 90.0),
    (-10.0, 350.0), (-360.0, 0.0), (-370.0, 350.0),
])
def test_mod360_wraps_both_directions(raw, wrapped):
    assert mod360(raw) == pytest.approx(wrapped)


def test_floquet_ledger_skips_the_mode_it_just_pulsed():
    """The pulsed mode's own phase rides on the pulse, not on the ledger."""
    swap_stors = [1, 2, 3]
    swap_ds = FakeSwapDataset({
        ("M1-S1", "M1-S2"): 10.0,
        ("M1-S3", "M1-S2"): 25.0,
    })
    offsets = [100.0, 200.0, 300.0]

    advance_floquet_offsets(offsets, swap_stors, pulsed_stor=2,
                            swap_ds=swap_ds)

    assert offsets == [110.0, 200.0, 325.0]
    # S2's own row was never even looked up.
    assert ("M1-S2", "M1-S2") not in swap_ds.asked


def test_floquet_ledger_wraps_each_entry():
    swap_stors = [1, 2]
    swap_ds = FakeSwapDataset({("M1-S2", "M1-S1"): 90.0})
    offsets = [0.0, 300.0]

    advance_floquet_offsets(offsets, swap_stors, pulsed_stor=1,
                            swap_ds=swap_ds)

    assert offsets == [0.0, 30.0]


def test_matrix_ledger_includes_the_diagonal():
    """A ds_storage or decoder matrix may carry the active-access phase."""
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    offsets = [0.0, 0.0, 0.0]

    advance_matrix_offsets(offsets, matrix, pulsed_column=1)

    # The whole column, diagonal entry 5.0 included.
    assert offsets == [2.0, 5.0, 8.0]


def test_matrix_ledger_takes_the_column_not_the_row():
    """Direction matters: the decoder matrix is not symmetric."""
    matrix = np.array([[0.0, 30.0], [0.0, 0.0]])
    offsets = [0.0, 0.0]

    advance_matrix_offsets(offsets, matrix, pulsed_column=0)
    assert offsets == [0.0, 0.0], "column 0 is all zeros"

    advance_matrix_offsets(offsets, matrix, pulsed_column=1)
    assert offsets == [30.0, 0.0]


def test_matrix_ledger_may_be_longer_than_the_swap_list():
    """The decoder ledger has one extra axis: M1 photon lowering, first."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    offsets = [0.0, 0.0, 0.0]

    advance_matrix_offsets(offsets, matrix, pulsed_column=0)

    assert offsets == [1.0, 3.0, 5.0]


def test_detuning_phase_is_360_times_cycles():
    # 0.05 MHz for 2 us is 0.1 cycles.
    assert detuning_phase_deg(0.05, 2.0) == pytest.approx(36.0)
    assert detuning_phase_deg(-0.05, 2.0) == pytest.approx(-36.0)
    assert detuning_phase_deg(0.0, 1e6) == 0.0


def test_detuning_phase_is_not_pre_wrapped():
    """Callers wrap after summing; wrapping here would double-count nothing
    but would hide the size of a large accumulated phase in debug output."""
    assert detuning_phase_deg(1.0, 10.0) == pytest.approx(3600.0)

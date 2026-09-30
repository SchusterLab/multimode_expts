# -*- coding: utf-8 -*-
"""The swap-parameter reader six notebooks had their own copy of.

It is a reader, so a drift between those copies failed nothing -- it just
meant two notebooks disagreeing about which pulse they played. One definition
now, and these tests pin the parts a caller relies on: the tuple order it is
unpacked in, that the values come from the dataset rather than a default, and
that the prep/measure pulse strings are the right length and orientation.
"""
import pytest

from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from experiments.qsim.utils import floquet_pulse_parameters


@pytest.fixture(scope="module")
def station():
    return mock_station(**pinned_config_set("preload_current"))


def test_the_tuple_is_in_the_order_callers_unpack(station):
    freq, gain, length, pi_frac, ch, prepulse, postpulse = \
        floquet_pulse_parameters(station, 1, 3)

    assert freq == station.ds_floquet.get_freq("M1-S3")
    assert gain == station.ds_floquet.get_gain("M1-S3")
    assert length == station.ds_floquet.get_len("M1-S3")
    assert pi_frac == station.ds_floquet.get_pi_frac("M1-S3")
    assert ch in ("low", "high")


def test_it_reads_the_named_mode_not_a_fixed_one(station):
    third = floquet_pulse_parameters(station, 1, 3)
    fifth = floquet_pulse_parameters(station, 1, 5)
    assert third[0] != fifth[0], "both modes returned the same frequency"


def test_the_channel_label_uses_its_own_1000_MHz_cut(station):
    """Pinned because it differs from the pulse layer's 1800 MHz threshold.

    The two have always disagreed. This test exists so that anyone who
    "fixes" the label to match FLUX_HIGH_THRESHOLD_MHZ sees that it is a
    deliberate difference, and has to decide rather than assume.
    """
    for stor in range(1, 8):
        freq, _, _, _, ch, _, _ = floquet_pulse_parameters(station, 1, stor)
        assert ch == ("low" if freq < 1000 else "high"), (stor, freq, ch)


def test_the_prepulse_puts_one_photon_in_the_manipulate_mode(station):
    """Three steps: g0-e0, e0-f0, f0-g1. Seven pulse-table rows each."""
    *_, prepulse, _ = floquet_pulse_parameters(station, 1, 3)

    assert len(prepulse) == 7, "pulse table has freq/gain/len/phase/ch/type/sigma"
    assert all(len(row) == 3 for row in prepulse), prepulse


def test_the_postpulse_is_the_last_two_steps_reversed(station):
    """A ge measurement undoes f0g1 then ef, and nothing else."""
    *_, prepulse, postpulse = floquet_pulse_parameters(station, 1, 3)

    assert all(len(row) == 2 for row in postpulse), postpulse
    # Reversed: the postpulse's first step is the prepulse's last.
    assert postpulse[0][0] == prepulse[0][2]
    assert postpulse[0][1] == prepulse[0][1]

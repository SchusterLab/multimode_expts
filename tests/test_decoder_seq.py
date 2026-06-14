"""
Offline tests for the Fock-state decoder gate-sequence helpers
(``experiments/transduction/decoder.py``). Pure list manipulation -- no hardware.
"""

import pytest

import numpy as np

from experiments.transduction.decoder import (
    revert_pulse_seq, eta_to_swap_ratio, build_env_prep_seq,
    channel_swap_gate, scale_swap_length,
)


def test_revert_reverses_and_adds_180():
    seq = [['multiphoton', 'g0-e0', 'pi', 0],
           ['qubit', 'ef', 'pi', 90]]
    rev = revert_pulse_seq(seq)
    assert rev == [['qubit', 'ef', 'pi', 270],
                   ['multiphoton', 'g0-e0', 'pi', 180]]


def test_double_revert_is_identity_mod_360():
    seq = [['multiphoton', 'g0-e0', 'pi', 0],
           ['multiphoton', 'e0-f0', 'pi', 45],
           ['multiphoton', 'f0-g1', 'pi', 180],
           ['multiphoton', 'f1-g2', 'pi', 270]]
    back = revert_pulse_seq(revert_pulse_seq(seq))
    # order restored and phases back to original mod 360
    assert [g[:3] for g in back] == [g[:3] for g in seq]
    assert [g[3] % 360 for g in back] == [g[3] % 360 for g in seq]


def test_revert_does_not_mutate_input():
    seq = [['multiphoton', 'g0-e0', 'pi', 0]]
    _ = revert_pulse_seq(seq)
    assert seq == [['multiphoton', 'g0-e0', 'pi', 0]]


def test_phase_wraps_into_0_360():
    seq = [['qubit', 'ge', 'hpi', 270]]
    rev = revert_pulse_seq(seq)
    assert rev[0][3] == 90  # 270 + 180 = 450 -> 90


# --- channel helpers ---

def test_eta_to_swap_ratio_endpoints():
    assert eta_to_swap_ratio(1.0) == pytest.approx(0.0)      # no swap
    assert eta_to_swap_ratio(0.0) == pytest.approx(1.0)      # full SWAP
    assert eta_to_swap_ratio(0.5) == pytest.approx(0.5)      # theta = pi/2


def test_eta_to_swap_ratio_high_transmissivity_regime():
    # eta > 2/3 -> ratio < ~0.392
    assert eta_to_swap_ratio(2/3) == pytest.approx(0.3918, abs=1e-3)
    assert eta_to_swap_ratio(0.9) < eta_to_swap_ratio(0.7)   # monotonic


def test_env_prep_seq_targets_chosen_storage():
    seq = build_env_prep_seq(env_stor=3, man_no=1)
    assert seq[-1] == ['storage', 'M1-S3', 'pi', 0]
    # first three load |1> in the manipulate mode
    assert [g[1] for g in seq[:3]] == ['g0-e0', 'e0-f0', 'f0-g1']


def test_channel_swap_gate():
    assert channel_swap_gate(env_stor=3) == [['storage', 'M1-S3', 'pi', 0]]


def test_scale_swap_length_scales_only_length_row():
    # 7-row compiled-pulse form, one column
    compiled = [[5000.0], [12000], [1.0], [0], [3], ['flat_top'], [0.01]]
    scaled = scale_swap_length(compiled, 0.3918)
    assert scaled[2] == [pytest.approx(0.3918)]   # length scaled
    assert scaled[1] == [12000]                   # gain unchanged
    assert compiled[2] == [1.0]                   # input not mutated

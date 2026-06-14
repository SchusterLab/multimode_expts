'''
Offline validation of the sigma_z probe scalar / mask helpers in
``experiments/qubit_cavity/single_mode_wigner_tomography.py`` -- no hardware.

Checks:
  * _sigma_z_lane_shots pulls the correct innermost lane from a flat buffer,
  * _compute_sigma_z returns 1-2*P_e with an identity confusion matrix and the
    confusion-corrected value otherwise,
  * the with/without-active-reset matrix is selected by cfg.expt.active_reset,
  * _sigma_z_filter_map produces a (rounds*reps, expts) |g>-keep mask whose
    discard fraction matches the synthetic e-fraction.
'''

import numpy as np
import pytest
from slab import AttrDict

from experiments.qubit_cavity.single_mode_wigner_tomography import (
    _sigma_z_lane_shots, _compute_sigma_z, _sigma_z_filter_map,
)
from fitting.state_tomography import as_confusion_matrix, correct_readout_probs

THRESHOLD = 0.0          # I < 0 -> |g>, I > 0 -> |e>
G_VAL, E_VAL = -1.0, +1.0


def _cfg(active_reset=False, with_ar=None, without=None, rounds=1, reps=4, expts=3):
    readout = {'threshold': [THRESHOLD]}
    if with_ar is not None:
        readout['confusion_matrix_with_active_reset'] = with_ar
    if without is not None:
        readout['confusion_matrix_without_reset'] = without
    return AttrDict({
        'expt': AttrDict({'active_reset': active_reset, 'rounds': rounds,
                          'reps': reps, 'expts': expts}),
        'device': AttrDict({'readout': AttrDict(readout)}),
    })


# --- _sigma_z_lane_shots -----------------------------------------------------

def test_lane_extraction_picks_innermost_lane():
    read_num, n_groups, idx = 3, 5, 1
    buf = np.arange(n_groups * read_num).reshape(n_groups, read_num)
    lane = _sigma_z_lane_shots(buf, read_num, idx)
    assert np.array_equal(lane, buf[:, idx])


def test_lane_extraction_applies_herald_mask():
    # 4 lanes; herald at lane 1, sigma_z at lane 2. Half the shots fail the
    # herald (lane1 = e); those must be dropped from the sigma_z lane.
    read_num, idx_pre, idx_sz = 4, 1, 2
    n = 6
    per = np.zeros((n, read_num))
    per[:, idx_pre] = [G_VAL, E_VAL, G_VAL, E_VAL, G_VAL, E_VAL]   # herald: keep even rows
    per[:, idx_sz] = [10, 20, 11, 21, 12, 22]                     # tag sigma_z values
    lane = _sigma_z_lane_shots(per, read_num, idx_sz,
                               idx_pre_selection=idx_pre, threshold=THRESHOLD)
    assert np.array_equal(lane, [10, 11, 12])                     # only herald-passed rows
    # Without the mask, all shots are kept.
    lane_all = _sigma_z_lane_shots(per, read_num, idx_sz)
    assert np.array_equal(lane_all, [10, 20, 11, 21, 12, 22])


def test_lane_extraction_ands_two_herald_lanes():
    # 4 lanes; TWO heralds at lanes 0 and 1, sigma_z at lane 2. A shot is kept
    # only if it passes BOTH heralds (AND), matching bin_ss_data's combined mask.
    read_num, idx_sz = 4, 2
    per = np.zeros((6, read_num))
    per[:, 0] = [G_VAL, G_VAL, E_VAL, G_VAL, E_VAL, G_VAL]  # herald 1
    per[:, 1] = [G_VAL, E_VAL, G_VAL, G_VAL, E_VAL, G_VAL]  # herald 2
    per[:, idx_sz] = [10, 20, 30, 40, 50, 60]
    # rows passing BOTH heralds: 0, 3, 5
    lane = _sigma_z_lane_shots(per, read_num, idx_sz,
                               idx_pre_selection=[0, 1], threshold=THRESHOLD)
    assert np.array_equal(lane, [10, 40, 60])


# --- _compute_sigma_z --------------------------------------------------------

@pytest.mark.parametrize('p_e', [0.0, 0.25, 0.5, 1.0])
def test_identity_matrix_gives_raw(p_e):
    n = 1000
    n_e = int(round(p_e * n))
    lane = np.array([E_VAL] * n_e + [G_VAL] * (n - n_e))
    identity = [1.0, 0.0, 0.0, 1.0]   # [Pgg, Pge, Peg, Pee]
    cfg = _cfg(with_ar=identity, without=identity)
    sigma_z, sigma_z_raw = _compute_sigma_z(lane, THRESHOLD, cfg)
    assert sigma_z_raw == pytest.approx(1 - 2 * p_e)
    assert sigma_z == pytest.approx(1 - 2 * p_e, abs=1e-6)


def test_corrected_matches_correct_readout_probs():
    lane = np.array([E_VAL] * 300 + [G_VAL] * 700)   # raw P_e = 0.3
    conf = [0.95, 0.10, 0.05, 0.90]                  # realistic asymmetric readout
    cfg = _cfg(active_reset=False, without=conf)
    sigma_z, sigma_z_raw = _compute_sigma_z(lane, THRESHOLD, cfg)
    Pg_c, Pe_c = correct_readout_probs([0.7, 0.3], as_confusion_matrix(conf))
    assert sigma_z == pytest.approx(Pg_c - Pe_c)
    assert sigma_z_raw == pytest.approx(0.4)


def test_matrix_selected_by_active_reset():
    lane = np.array([E_VAL] * 300 + [G_VAL] * 700)
    with_ar = [0.99, 0.01, 0.01, 0.99]
    without = [0.90, 0.20, 0.10, 0.80]
    sz_ar, _ = _compute_sigma_z(lane, THRESHOLD, _cfg(active_reset=True, with_ar=with_ar, without=without))
    sz_no, _ = _compute_sigma_z(lane, THRESHOLD, _cfg(active_reset=False, with_ar=with_ar, without=without))
    expected_ar = np.subtract(*correct_readout_probs([0.7, 0.3], as_confusion_matrix(with_ar)))
    expected_no = np.subtract(*correct_readout_probs([0.7, 0.3], as_confusion_matrix(without)))
    assert sz_ar == pytest.approx(expected_ar)
    assert sz_no == pytest.approx(expected_no)
    assert sz_ar != pytest.approx(sz_no)


def test_missing_matrix_falls_back_to_raw_with_warning():
    lane = np.array([E_VAL] * 200 + [G_VAL] * 800)
    cfg = _cfg(active_reset=False)   # no matrices in cfg
    with pytest.warns(UserWarning):
        sigma_z, sigma_z_raw = _compute_sigma_z(lane, THRESHOLD, cfg)
    assert sigma_z == sigma_z_raw == pytest.approx(0.6)


def test_empty_lane_returns_none():
    assert _compute_sigma_z(np.array([]), THRESHOLD, _cfg()) == (None, None)


# --- _sigma_z_filter_map -----------------------------------------------------

def test_filter_map_shape_and_discard_fraction():
    rounds, reps, expts, read_num, idx_sz = 1, 6, 3, 4, 1
    # Build a raw (rounds, expts, reps, read_num) buffer; put e on the sigma_z
    # lane for the first two reps of every expt (so keep-fraction = 4/6).
    I_4d = np.full((rounds, expts, reps, read_num), G_VAL, dtype=float)
    I_4d[:, :, :2, idx_sz] = E_VAL
    cfg = _cfg(rounds=rounds, reps=reps, expts=expts)
    layout = {'read_num': read_num, 'idx_sigma_z': idx_sz}
    fmap = _sigma_z_filter_map(I_4d, cfg, layout)
    assert fmap.shape == (rounds * reps, expts)
    assert fmap.sum() == expts * (reps - 2)          # 4 kept per expt
    assert (1 - fmap.mean()) == pytest.approx(2 / 6)

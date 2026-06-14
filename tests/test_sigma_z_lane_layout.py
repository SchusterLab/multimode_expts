'''
Offline validation of the sigma_z probe lane bookkeeping in
``MM_base.lane_layout`` -- no hardware required.

The sigma_z probe inserts ONE post-prep transmon readout right after the
active_reset block when ``sigma_z_mode`` is 'reset' (Tier 1) or 'postselect'
(Tier 2). These tests pin down:
  * read_num accounting across the cross product of {active_reset,
    pre_selection_reset, parity_shot, post_select_pre_pulse} x sigma_z_mode,
  * idx_sigma_z lands on the first lane after the active_reset block,
  * idx_final == read_num - 1 and the pre_selection lane(s) are unchanged by the
    probe (active_reset may emit up to TWO pre-selection lanes: a bare g-herald
    and a parity herald),
  * sigma_z_mode resolution (explicit string, measure_sigma_z boolean, absent).
'''

import itertools

import pytest
from slab import AttrDict

from experiments.MM_base import MM_base


def _cfg(**expt):
    '''Minimal cfg accepted by lane_layout / get_active_reset_params.'''
    return AttrDict({'expt': AttrDict(expt)})


def _expected_n_active_reset(expt):
    if not expt.get('active_reset', False):
        return 0
    params = MM_base.get_active_reset_params(_cfg(**expt))
    return MM_base.active_reset_read_num(**params)


# --- sigma_z_mode resolution -------------------------------------------------

def test_mode_absent_is_off():
    layout = MM_base.lane_layout(_cfg())
    assert layout['sigma_z_mode'] == 'off'
    assert layout['n_sigma_z'] == 0
    assert layout['idx_sigma_z'] is None


def test_measure_sigma_z_bool_maps_to_reset():
    layout = MM_base.lane_layout(_cfg(measure_sigma_z=True))
    assert layout['sigma_z_mode'] == 'reset'
    assert layout['n_sigma_z'] == 1


def test_explicit_mode_overrides_bool():
    layout = MM_base.lane_layout(_cfg(measure_sigma_z=True,
                                      sigma_z_mode='postselect'))
    assert layout['sigma_z_mode'] == 'postselect'
    assert layout['n_sigma_z'] == 1


def test_explicit_off_overrides_bool():
    layout = MM_base.lane_layout(_cfg(measure_sigma_z=True, sigma_z_mode='off'))
    assert layout['sigma_z_mode'] == 'off'
    assert layout['n_sigma_z'] == 0


@pytest.mark.parametrize('mode,expected_n', [
    ('off', 0), ('reset', 1), ('postselect', 1), ('measure', 0),
])
def test_n_sigma_z_per_mode(mode, expected_n):
    layout = MM_base.lane_layout(_cfg(sigma_z_mode=mode))
    assert layout['n_sigma_z'] == expected_n
    assert layout['sigma_z_mode'] == mode


def test_measure_mode_uses_final_lane():
    # Tier 3: no dedicated lane; the final readout IS the sigma_z readout.
    layout = MM_base.lane_layout(_cfg(sigma_z_mode='measure'))
    assert layout['n_sigma_z'] == 0
    assert layout['idx_sigma_z'] == layout['idx_final']
    # with active reset, the final lane sits right after the reset block + 1 (final)
    layout_ar = MM_base.lane_layout(_cfg(active_reset=True, sigma_z_mode='measure'))
    assert layout_ar['idx_sigma_z'] == layout_ar['idx_final']
    assert layout_ar['n_sigma_z'] == 0


# --- read_num / lane-index accounting ----------------------------------------

_BOOL_FLAGS = ['active_reset', 'pre_selection_reset', 'pre_selection_parity',
               'parity_shot', 'post_select_pre_pulse']


@pytest.mark.parametrize('combo', list(itertools.product([False, True], repeat=5)))
@pytest.mark.parametrize('mode', ['off', 'reset', 'postselect'])
def test_read_num_and_indices(combo, mode):
    expt = dict(zip(_BOOL_FLAGS, combo))
    expt['sigma_z_mode'] = mode
    if expt['parity_shot']:
        expt['repeat_count'] = 3  # exercise multi-herald accounting
    layout = MM_base.lane_layout(_cfg(**expt))

    n_active_reset = _expected_n_active_reset(expt)
    n_sigma_z = 1 if mode in ('reset', 'postselect') else 0
    n_parity_shot = expt.get('repeat_count', 1) if expt['parity_shot'] else 0
    n_post_select = 1 if expt['post_select_pre_pulse'] else 0
    expected_read_num = 1 + n_active_reset + n_sigma_z + n_parity_shot + n_post_select

    assert layout['n_active_reset'] == n_active_reset
    assert layout['n_sigma_z'] == n_sigma_z
    assert layout['read_num'] == expected_read_num
    assert layout['idx_final'] == expected_read_num - 1

    # sigma_z lane is the first lane after the active_reset block.
    if n_sigma_z:
        assert layout['idx_sigma_z'] == n_active_reset
    else:
        assert layout['idx_sigma_z'] is None

    # The probe must NOT disturb the pre_selection lanes (which live inside the
    # active_reset block, ahead of the sigma_z lane). There are up to two:
    # the bare g-herald (pre_selection_reset) and the parity herald
    # (pre_selection_parity), contiguous and LAST in the active_reset block.
    n_pre = ((1 if expt['pre_selection_reset'] else 0)
             + (1 if expt['pre_selection_parity'] else 0))
    if expt['active_reset'] and n_active_reset > 0 and n_pre:
        assert layout['idx_pre_selection_list'] == list(
            range(n_active_reset - n_pre, n_active_reset))
        assert layout['idx_pre_selection'] == n_active_reset - 1
    else:
        assert layout['idx_pre_selection_list'] == []
        assert layout['idx_pre_selection'] is None


def test_probe_adds_exactly_one_lane():
    base = MM_base.lane_layout(_cfg(active_reset=True, sigma_z_mode='off'))
    for mode in ('reset', 'postselect'):
        probed = MM_base.lane_layout(_cfg(active_reset=True, sigma_z_mode=mode))
        assert probed['read_num'] == base['read_num'] + 1
        assert probed['idx_sigma_z'] == base['n_active_reset']


# --- two independent pre-selection rounds ------------------------------------

def test_two_preselection_rounds_add_two_contiguous_lanes():
    # Baseline: active reset, no pre-selection rounds.
    base = MM_base.lane_layout(_cfg(active_reset=True,
                                    pre_selection_reset=False,
                                    pre_selection_parity=False))
    # Both rounds on: bare g-herald + parity herald, each its own lane.
    both = MM_base.lane_layout(_cfg(active_reset=True,
                                    pre_selection_reset=True,
                                    pre_selection_parity=True))
    assert both['read_num'] == base['read_num'] + 2
    n_ar = both['n_active_reset']
    # The two pre-selection lanes are the LAST, contiguous lanes of the block.
    assert both['idx_pre_selection_list'] == [n_ar - 2, n_ar - 1]
    assert both['idx_pre_selection'] == n_ar - 1  # scalar = last lane
    assert both['idx_final'] == both['read_num'] - 1


def test_parity_round_alone_is_one_lane():
    layout = MM_base.lane_layout(_cfg(active_reset=True,
                                      pre_selection_reset=False,
                                      pre_selection_parity=True))
    n_ar = layout['n_active_reset']
    assert layout['idx_pre_selection_list'] == [n_ar - 1]
    assert layout['idx_pre_selection'] == n_ar - 1


def test_sigma_z_probe_sits_past_both_preselection_lanes():
    layout = MM_base.lane_layout(_cfg(active_reset=True,
                                      pre_selection_reset=True,
                                      pre_selection_parity=True,
                                      sigma_z_mode='reset'))
    # sigma_z lane is the first lane AFTER the whole active_reset block,
    # i.e. past both pre-selection lanes.
    assert layout['idx_sigma_z'] == layout['n_active_reset']
    assert layout['idx_sigma_z'] > layout['idx_pre_selection']

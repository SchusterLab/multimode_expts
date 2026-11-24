"""M1 compatibility and joint M1/M2 pulse storage, without hardware access."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from slab import AttrDict

from experiments.MM_base import MM_base, multiphoton_prep, multiphoton_slot, multiphoton_table


@pytest.fixture
def cfg():
    transitions = {}
    for name, frequency in [('gn-en', 3500), ('en-fn', 3400), ('fn-gn+1', 2000)]:
        transitions[name] = dict(
            frequency=[frequency + 10 * n for n in range(4)],
            gain=[1100] * 4,
            length=[0.4] * 4,
            type=['flat_top' if name == 'fn-gn+1' else 'gauss'] * 4,
            sigma=[0.005] * 4,
        )
    return AttrDict(dict(
        device=dict(
            multiphoton=dict(pi=transitions, hpi=deepcopy(transitions)),
            manipulate=dict(ramp_sigma=0.005),
            storage=dict(_ds_storage=None, _ds_floquet=None),
        ),
        hw=dict(soc=dict(dacs=dict(qubit=dict(ch=[2]), sideband=dict(ch=[0])))),
        expt={},
    ))


def save_field(cfg, mode, transition, field, m1_photons, m2_photons, value, pulse='pi'):
    values, key = multiphoton_slot(cfg, mode, pulse, transition, field, m1_photons, m2_photons)
    values[key] = value


def test_m1_keeps_existing_lists_and_four_item_preparation(cfg):
    assert multiphoton_table(cfg) is cfg.device.multiphoton
    assert multiphoton_table(cfg, resolve=False) is cfg.device.multiphoton
    base = MM_base(cfg, None)
    assert base.prep_man_fock_state(1, '1') == [
        ['multiphoton', 'g0-e0', 'pi', 0],
        ['multiphoton', 'e0-f0', 'pi', 0],
        ['multiphoton', 'f0-g1', 'pi', 0],
    ]
    save_field(cfg, 1, 'fn-gn+1', 'frequency', 5, 0, 2095)
    assert cfg.device.multiphoton.pi['fn-gn+1'].frequency == [2000, 2010, 2020, 2030, None, 2095]
    assert 'multiphoton_joint' not in cfg.device


@pytest.mark.parametrize('pulse', ['pi', 'hpi'])
def test_joint_keys_keep_modes_and_photon_pairs_separate(cfg, pulse):
    save_field(cfg, 1, 'fn-gn+1', 'frequency', 2, 1, 2091, pulse)
    save_field(cfg, 2, 'fn-gn+1', 'frequency', 2, 1, 1891, pulse)
    save_field(cfg, 2, 'fn-gn+1', 'frequency', 3, 1, 1892, pulse)
    assert cfg.device.multiphoton_joint.M1[pulse].frequency == {'2,1': 2091}
    assert cfg.device.multiphoton_joint.M2[pulse].frequency == {'2,1': 1891, '3,1': 1892}
    m1 = multiphoton_table(cfg, 1, other_mode_photon_number=1, resolve=False)
    m2 = multiphoton_table(cfg, 2, other_mode_photon_number=2, resolve=False)
    assert m1[pulse]['fn-gn+1'].frequency[2] == 2091
    assert m2[pulse]['fn-gn+1'].frequency[1] == 1891


@pytest.mark.parametrize('transition', ['gn-en', 'en-fn'])
def test_joint_qubit_settings_are_shared_by_both_modes(cfg, transition):
    save_field(cfg, 2, transition, 'frequency', 2, 1, 3577)
    from_m1, m1_key = multiphoton_slot(cfg, 1, 'pi', transition, 'frequency', 2, 1)
    from_m2, m2_key = multiphoton_slot(cfg, 2, 'pi', transition, 'frequency', 2, 1)
    assert from_m1 is from_m2
    assert m1_key == m2_key == '2,1'
    assert multiphoton_table(cfg, 1, other_mode_photon_number=1, resolve=False).pi[transition].frequency[2] == 3577
    assert multiphoton_table(cfg, 2, other_mode_photon_number=2, resolve=False).pi[transition].frequency[1] == 3577


def test_sparse_coordinates_extend_every_field_and_keep_unmeasured_values(cfg):
    save_field(cfg, 2, 'fn-gn+1', 'length', 7, 9, 0.65)
    table = multiphoton_table(cfg, 2, other_mode_photon_number=7, num_photon_levels=1, resolve=False)
    assert len(table.pi['gn-en'].gain) == 10
    assert table.pi['fn-gn+1'].length[9] == 0.65
    assert table.pi['fn-gn+1'].frequency[9] is None


def test_scan_estimates_do_not_fill_saved_config_or_strict_calibrations(cfg):
    save_field(cfg, 2, 'gn-en', 'frequency', 0, 1, 3507)
    save_field(cfg, 2, 'fn-gn+1', 'frequency', 0, 1, 1810)
    before = deepcopy(cfg)
    estimated = multiphoton_table(cfg, 2, other_mode_photon_number=2)
    strict = multiphoton_table(cfg, 2, other_mode_photon_number=2, allow_scan_estimates=False)
    assert estimated.pi['gn-en'].frequency[1] == 3527
    assert estimated.pi['fn-gn+1'].frequency[1] == 1810
    assert strict.pi['gn-en'].frequency[1] is None
    assert strict.pi['fn-gn+1'].frequency[1] is None
    assert cfg == before


def test_m2_vacuum_scan_reads_storage_dataset(cfg):
    dataset = Mock()
    dataset.has_row.return_value = True
    dataset.get_value.side_effect = lambda row, column: {
        'freq (MHz)': 1840.5, 'gain (DAC units)': 5100, 'pi (mus)': 0.8,
    }[column]
    cfg.device.storage._ds_storage = dataset
    pulse = multiphoton_table(cfg, 2).pi['fn-gn+1']
    assert (pulse.frequency[0], pulse.gain[0], pulse.length[0]) == (1840.5, 5100, 0.8)
    assert 'multiphoton_joint' not in cfg.device


def test_joint_preparation_fills_m1_before_m2_and_reads_each_saved_swap(cfg):
    for m2_photons, frequency in [(0, 1900), (1, 1910)]:
        for field, value in dict(frequency=frequency, length=0.6, gain=1200).items():
            save_field(cfg, 2, 'fn-gn+1', field, 2, m2_photons, value)
    for transition, frequency in [('gn-en', 3527), ('en-fn', 3427)]:
        save_field(cfg, 2, transition, 'frequency', 2, 1, frequency)
        save_field(cfg, 2, transition, 'gain', 2, 1, 1100)
    pulses = multiphoton_prep(cfg, 2, 2)
    assert [pulse[4:] for pulse in pulses] == [[1, 0]] * 6 + [[2, 2]] * 6
    assert [pulse[1] for pulse in pulses[2::3]] == ['f0-g1', 'f1-g2', 'f0-g1', 'f1-g2']
    creator = MM_base(cfg, None).get_prepulse_creator(pulses)
    assert creator.pulse[0, 2::3].tolist() == [2000, 2010, 1900, 1910]


def test_joint_preparation_requires_each_previous_m2_swap(cfg):
    save_field(cfg, 2, 'fn-gn+1', 'frequency', 2, 0, 1900)
    save_field(cfg, 2, 'fn-gn+1', 'length', 2, 0, 0.6)
    with pytest.raises(ValueError, match=r'requires the M2 swap.*\(2, 1\)'):
        multiphoton_prep(cfg, 2, 2)


@pytest.mark.parametrize('descriptor,defaults,frequency,gain,length', [
    (['multiphoton', 'f1-g2', 'pi', 0], {}, 2010, 1100, 0.4),
    (['multiphoton', 'f1-g2', 'pi', 0, 1],
     dict(multiphoton_mode=2, multiphoton_other_mode_photon_number=3), 2010, 1100, 0.4),
    (['multiphoton', 'f1-g2', 'pi', 0, 2],
     dict(multiphoton_mode=1, multiphoton_other_mode_photon_number=3), 1810, 1200, 0.6),
    (['multiphoton', 'f1-g2', 'pi', 0, 2, 2], {}, 1910, 1200, 0.6),
    (['multiphoton', 'f1-g2', 'pi', 0],
     dict(multiphoton_mode=2, multiphoton_other_mode_photon_number=2), 1910, 1200, 0.6),
])
def test_four_five_and_six_item_descriptors_use_the_correct_mode(
        cfg, descriptor, defaults, frequency, gain, length):
    cfg.expt.update(defaults)
    for m1_photons, stored_frequency in [(0, 1810), (2, 1910)]:
        for field, value in dict(frequency=stored_frequency, gain=1200, length=0.6).items():
            save_field(cfg, 2, 'fn-gn+1', field, m1_photons, 1, value)
    creator = MM_base(cfg, None).get_prepulse_creator([descriptor])
    assert creator.pulse[:, 0].tolist() == [frequency, gain, length, 0, 0, 'flat_top', 0.005]

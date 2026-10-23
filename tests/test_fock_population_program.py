"""Fock preparation leaves readout selection and lane counts to Qsim."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from slab import AttrDict

from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.floquet_train import FloquetProgram


@pytest.mark.parametrize('readout, lanes, occupations', [
    ('qubit', 1, [3, 1]),
    ('parity', 1, [1, 1]),
    ('multiparity', 2, [1, 1]),
    (None, 1, [3, 1]),
])
def test_preparation_preserves_readout_and_lane_count(
        monkeypatch, readout, lanes, occupations):
    cfg = AttrDict({'expt': {
        'occupations': occupations, 'swap_stors': [4],
    }})
    if readout is not None:
        cfg.expt.readout = readout
    assert FockPopulationProgram.readouts_per_shot(cfg) == lanes

    # Config setup does not play preparation pulses before hardware is ready.
    initialized = []
    monkeypatch.setattr(FloquetProgram, 'initialize', lambda self: initialized.append(self))
    program = FockPopulationProgram.__new__(FockPopulationProgram)
    program.cfg = cfg
    program.initialize()

    assert initialized == [program]
    assert cfg.expt.get('readout') == readout
    assert FockPopulationProgram.readouts_per_shot(cfg) == lanes


@pytest.fixture
def playback():
    """Use the real preparation and MM_base swap, recording hardware calls."""
    program = FockPopulationProgram.__new__(FockPopulationProgram)
    events = []
    program.get_prepulse_creator = lambda pulses: SimpleNamespace(pulse=deepcopy(pulses))
    program.custom_pulse = lambda cfg, pulses, prefix: events.append(('play', deepcopy(pulses)))
    program.us2cycles = lambda us: round(us * 1000)
    program.sync_all = lambda cycles=None: events.append(('wait', cycles)) if cycles else None
    program._play_scramble_with_phase_offsets = lambda phases, storages, disorder: events.append(
        ('evolution', phases.copy(), storages.copy(), disorder.copy()))
    return program, events


def test_core_uses_mm_base_swaps_after_each_preparation_and_loads_m1_last(playback):
    program, events = playback
    program.cfg = AttrDict({'expt': {
        'occupations': [2, 1, 0, 3], 'swap_stors': [4, 5, 7],
    }})

    program.core_pulses()

    assert events == [
        ('play', program.prep_man_fock_state(1, '1')),
        ('play', [['storage', 'M1-S4', 'pi', 0]]),
        ('wait', 200),
        ('play', program.prep_man_fock_state(1, '3')),
        ('play', [['storage', 'M1-S7', 'pi', 0]]),
        ('wait', 200),
        ('play', program.prep_man_fock_state(1, '2')),
        ('evolution', [0., 0., 0.], [4, 5, 7], [0., 0., 0.]),
    ]


def test_vacuum_goes_directly_to_evolution_without_preparation_or_swap(playback):
    program, events = playback
    program.cfg = AttrDict({'expt': {
        'occupations': [0, 0, 0], 'swap_stors': [4, 7],
    }})

    program.core_pulses()

    assert events == [('evolution', [0., 0.], [4, 7], [0., 0.])]

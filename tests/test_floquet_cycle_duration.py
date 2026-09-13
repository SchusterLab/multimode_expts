# -*- coding: utf-8 -*-
"""The live program and the offline resolver must agree on one cycle time.

Why this matters more than it looks
-----------------------------------
The Floquet cycle duration is not a display quantity. Every coupling rate is
``1 / (4 * pi_frac * cycle_us)``, so the cycle time divides straight into the
Hamiltonian the spectrum is compared against. Two implementations of it --
one in the pulse layer, one in the offline timing resolver -- drifting apart
would present as wrong physics, never as an error.

They are now one function with two adapters. What can still go wrong is the
*wiring*: an adapter passing lengths where ramps belong, or resolving the
low/high flux ramp per storage in the wrong order. Both adapters are
reconstructed here from the same mock station and compared.
"""
import pytest

from experiments.floquet_timing import floquet_cycle_us
from experiments.qsim.mbr_campaign import (
    mbr_defaults,
    mock_station,
    pinned_config_set,
    pinned_sets,
    run_stage,
)

SWAP_STORS = [1, 2, 3, 4]
OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]


@pytest.fixture(scope="module", params=sorted(pinned_sets()))
def program(request):
    """A compiled program per pinned config set (gauss and preload_flattop)."""
    station = mock_station(**pinned_config_set(request.param))
    acquired = run_stage(station, "propagator",
                         mbr_defaults(SWAP_STORS, reps=10),
                         SWAP_STORS, OCCUPATIONS, reps=10)
    return acquired[0].prog


def _resolver_style(prog):
    """Recompute the cycle the way the offline resolver assembles it.

    Deliberately does not reuse the program's own adapter: it goes through
    ``soccfg`` and the dataset, as ``resolve_floquet_timing`` does, so a
    wiring error in either adapter shows up as a disagreement.
    """
    soccfg = prog.soccfg
    ramp_sigma = prog.cfg.device.manipulate.ramp_sigma
    ramp_low = soccfg.us2cycles(ramp_sigma, gen_ch=prog.m1s_ch[0])
    return floquet_cycle_us(
        prog.cfg.expt.swap_stors,
        swap_ds=prog.swap_ds,
        waveform_modes=prog.m1s_waveform_mode,
        channels=prog.m1s_ch,
        lengths=prog.m1s_length,
        ramp_cycles=[soccfg.us2cycles(ramp_sigma, gen_ch=ch)
                     for ch in prog.m1s_ch],
        sync_cycles=int(prog.cfg.expt.get("scramble_sync_cycles", 10)),
        us2cycles=lambda us, ch: soccfg.us2cycles(us, gen_ch=ch),
        cycles2us=soccfg.cycles2us,
        clock_ratio=lambda ch: (float(soccfg["tprocs"][0]["f_time"])
                                / float(soccfg["gens"][ch]["f_fabric"])),
        gauss_sigma_override=prog.cfg.expt.get("floquet_gauss_sigma", None),
    )


def test_both_adapters_give_the_same_cycle(program):
    assert program.calculate_floquet_cycle_us() == pytest.approx(
        _resolver_style(program), rel=0, abs=0)


def test_the_cycle_is_physical(program):
    cycle_us = program.calculate_floquet_cycle_us()
    assert cycle_us > 0
    # Four swaps of order 0.1 us plus syncs; a result outside this bracket
    # means a unit slipped, which is the other way this quantity goes wrong.
    assert 0.05 < cycle_us < 50.0, cycle_us


def test_the_quantization_is_not_dropped(program):
    """A cycle is a sum of integer tProc advances, not of exact durations.

    Dropping the ``int()`` ran ~1.2% long on the August configs, which is
    well inside the range a spectrum still looks reasonable in. So check the
    result really is an integer number of tProc cycles.
    """
    cycle_us = program.calculate_floquet_cycle_us()
    tproc_cycles = program.soccfg.us2cycles(cycle_us)
    assert program.soccfg.cycles2us(tproc_cycles) == pytest.approx(cycle_us)


def test_fewer_swaps_make_a_shorter_cycle(program):
    """The argument is honoured, not ignored in favour of cfg.expt."""
    full = program.calculate_floquet_cycle_us()
    partial = program.calculate_floquet_cycle_us(swap_stors=SWAP_STORS[:2])
    assert 0 < partial < full

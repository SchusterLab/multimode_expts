"""encoder_pad_us: the decoder starts at the same time for every encoder (mock compile).

Off-diagonal MBR elements (encoder i, decoder j) pick up a phase like
omega_j * L_i when the decoder start depends on the encoder length L_i
(vault Lab/guan/HamTomo/2026/11/2026-11-22.md). ``encoder_pads_us`` plans a
wait before each encoder so that all encoders have the same length.
"""
import pytest

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.mbr_campaign import (
    build_campaign,
    encoder_pads_us,
    fixed_n_occupations,
    mock_station,
    pinned_config_set,
)
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS,
    FLOQUET_DEFAULTS,
    MEASUREMENT_CONFIG_DEFAULTS,
)

SWAP_STORS = [1, 2, 3, 4]
CYCLES = [0, 4]


@pytest.fixture(scope="module")
def setup():
    station = mock_station(**pinned_config_set())
    campaign = build_campaign(FLOQUET_DEFAULTS, ACTIVE_RESET_DEFAULTS, MEASUREMENT_CONFIG_DEFAULTS,
                              modes=SWAP_STORS, reps=1)
    runner = CharacterizationRunner(station=station, ExptClass=MBRTimeTraceExperiment,
                                    default_expt_cfg=campaign.defaults, show=False)
    return station, campaign, runner


def _decoder_starts(runner, campaign, pairs, pads=None):
    overrides = []
    for initial, final in pairs:
        override = MBRTimeTraceExperiment.job_config(initial, final, CYCLES, campaign.modes,
                                                     sync_cycles=campaign.sync_cycles, reps=1)
        if pads is not None:
            override["encoder_pad_us"] = pads[tuple(initial)]
        overrides.append(override)
    jobs = runner.execute(overrides=overrides, batch_size=len(overrides), log=False, show=False)
    return [job.prog.decoder_start_clocks for job in jobs], [job.prog.encoder_clocks for job in jobs]


@pytest.mark.parametrize("N", [1, 2])
def test_pads_equalize_the_decoder_start(setup, N):
    station, campaign, runner = setup
    occupations = [tuple(o) for o in fixed_n_occupations(N, len(campaign.mode_labels))]
    final = occupations[0]
    pairs = [(initial, final) for initial in occupations]

    unpadded, encoder_clocks = _decoder_starts(runner, campaign, pairs)
    assert len(set(encoder_clocks)) > 1, "encoders of this sector should differ in length"
    assert len(set(unpadded)) > 1

    pads = encoder_pads_us(station, campaign, occupations)
    assert min(pads.values()) == 0.0
    padded, _ = _decoder_starts(runner, campaign, pairs, pads)
    assert len(set(padded)) == 1, f"decoder start differs between encoders: {padded}"
    assert padded[0] == max(unpadded)


def test_zero_pad_leaves_the_program_unchanged(setup):
    _, campaign, runner = setup
    occupation = tuple(fixed_n_occupations(1, len(campaign.mode_labels))[1])
    pair = [(occupation, occupation)]
    plain_starts, plain_encoder = _decoder_starts(runner, campaign, pair)
    zero_starts, zero_encoder = _decoder_starts(runner, campaign, pair, {occupation: 0.0})
    assert plain_starts == zero_starts and plain_encoder == zero_encoder


def test_negative_pad_is_refused(setup):
    _, campaign, runner = setup
    occupation = tuple(fixed_n_occupations(1, len(campaign.mode_labels))[0])
    with pytest.raises(Exception, match="encoder_pad_us"):
        _decoder_starts(runner, campaign, [(occupation, occupation)], {occupation: -1.0})

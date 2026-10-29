"""Board resource checks in MockQickSoc and check_program_resources (spec item D).

The 10-28 case: JOB-20261028-00775 (probe sigma 1.2 us) and -00780 (sigma 1.05 us
after the Fock 1 prefix) failed at upload with "buffer length must be 32768
samples or less". The mock must fail on the same programs and pass the ones
that ran (sigma 0.8 us, with and without the prefix).
"""
import matplotlib
import numpy as np
import pytest
from slab import AttrDict

matplotlib.use("Agg")

from experiments.board_resources import (  # noqa: E402
    BoardResourceError,
    check_program_resources,
    program_resource_problems,
)
from experiments.characterization_runner import CharacterizationRunner  # noqa: E402
from experiments.mock_hardware import MockQickSoc  # noqa: E402
from experiments.qsim.mbr_campaign import mock_station  # noqa: E402
from experiments.single_qubit.pulse_probe_spectroscopy import (  # noqa: E402
    PulseProbeSpectroscopyExperiment,
)
from experiments.station import read_soccfg_snapshot  # noqa: E402

FOCK1 = [["multiphoton", "g0-e0", "pi", 0], ["multiphoton", "e0-f0", "pi", 0],
         ["multiphoton", "f0-g1", "pi", 0]]


@pytest.fixture(scope="module")
def soccfg():
    return read_soccfg_snapshot()


class FakeProgram:
    """The program attributes the checks read, with a few envelopes on ch 2."""

    def __init__(self, soccfg, sizes):
        self.envelopes = [{"next_addr": 0, "envs": {}} for _ in soccfg["gens"]]
        for k, size in enumerate(sizes):
            env = self.envelopes[2]
            env["envs"][f"pulse{k}"] = {"data": np.zeros((size, 2), np.int16),
                                        "addr": env["next_addr"]}
            env["next_addr"] += size
        self.gen_chs = {}
        self.binprog = [0] * 10


def test_program_check_names_the_channel_and_envelopes(soccfg):
    prog = FakeProgram(soccfg, [31000, 896, 896, 896])
    with pytest.raises(BoardResourceError) as err:
        check_program_resources(prog, soccfg)
    text = str(err.value)
    assert "gen ch 2" in text and "33688 of 32768" in text
    assert "3 x 896 samples" in text and "1 x 31000 samples" in text


def test_program_check_passes_at_the_limit(soccfg):
    assert program_resource_problems(FakeProgram(soccfg, [32768 - 896, 896]), soccfg) == []


def test_program_check_mux_tones_and_pmem(soccfg):
    prog = FakeProgram(soccfg, [])
    prog.gen_chs = {6: {"mux_tones": [{}] * 5}}
    prog.binprog = [0] * (soccfg["tprocs"][0]["pmem_size"] + 1)
    problems = program_resource_problems(prog, soccfg)
    assert len(problems) == 2
    assert "5 mux tones" in problems[0] and "pmem_size" in problems[1]


def test_mock_soc_checks_like_the_drivers(soccfg):
    soc = MockQickSoc(soccfg)
    soc.load_pulse_data(2, data=[[0, 0]] * 896, addr=32768 - 896)
    with pytest.raises(BoardResourceError, match="gen ch 2"):
        soc.load_pulse_data(2, data=[[0, 0]] * 897, addr=32768 - 896)
    with pytest.raises(BoardResourceError, match="mux tones"):
        soc.config_mux_gen(6, [{}] * 5)
    with pytest.raises(BoardResourceError, match="buf_maxlen"):
        soc.config_buf(0, length=1024)
    with pytest.raises(BoardResourceError, match="avg_maxlen"):
        soc.get_accumulated(0, length=16382)
    assert soc.get_accumulated(0, length=16380).shape == (16380, 2)
    with pytest.raises(BoardResourceError, match="pmem_size"):
        soc.load_bin_program([0] * (soccfg["tprocs"][0]["pmem_size"] + 1))


def test_mock_soc_without_soccfg_does_not_check():
    MockQickSoc().load_pulse_data(2, data=[[0, 0]] * 40000, addr=0)


@pytest.fixture(scope="module")
def probe_runner():
    station = mock_station()
    assert station.is_mock, "refusing to acquire against real instruments"
    defaults = AttrDict(dict(
        start=3560.1, step=0.02, expts=2, reps=1, rounds=1, length=0.02, sigma=0.8, gain=300,
        qubit=0, qubits=[0], active_reset=False, man_reset=False, storage_reset=False,
        prepulse=False, gate_based=True, pre_sweep_pulse=[], relax_delay=10))
    return CharacterizationRunner(station=station, ExptClass=PulseProbeSpectroscopyExperiment,
                                  default_expt_cfg=defaults, show=False)


@pytest.mark.parametrize("sigma, prefix", [(1.2, []), (1.05, FOCK1)])
def test_1028_probes_that_failed_on_the_board_fail_in_the_mock(probe_runner, sigma, prefix):
    with pytest.raises(BoardResourceError, match="gen ch 2"):
        probe_runner.execute(sigma=sigma, prepulse=bool(prefix), pre_sweep_pulse=prefix,
                             log=False, show=False)


@pytest.mark.parametrize("sigma, prefix", [(0.8, []), (0.8, FOCK1), (1.05, [])])
def test_1028_probes_that_fit_pass_in_the_mock(probe_runner, sigma, prefix):
    probe_runner.execute(sigma=sigma, prepulse=bool(prefix), pre_sweep_pulse=prefix,
                         log=False, show=False)

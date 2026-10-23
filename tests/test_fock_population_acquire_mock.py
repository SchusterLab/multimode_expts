"""Build, compile and acquire the population layers in mock mode (as tests/test_mbr_acquire_mock.py).

The other population tests feed synthetic shots to analyze(). This one runs
the real FockPopulationProgram through CharacterizationRunner on a mock
station, so the job class, its sweep and the assembled acquire() are checked
against the qick compile path.
"""
import matplotlib
import numpy as np
import pytest
from slab import AttrDict

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner  # noqa: E402
from experiments.qsim.agp import CentralBosonAGPDisorderScan  # noqa: E402
from experiments.qsim.fock_population import FockPopulationExperiment  # noqa: E402
from experiments.qsim.mbr_campaign import mock_station, pinned_config_set  # noqa: E402
from experiments.qsim.notebook_helpers.defaults import (  # noqa: E402
    ACTIVE_RESET_DEFAULTS,
    FLOQUET_DEFAULTS,
)

SWAP_STORS = [1, 2]
CONFUSION = .8 * np.eye(4) + .05


def population_defaults(swap_stors):
    """The population job defaults of measurement_notebooks/guan/mbr_s4s7_n2.py, cell 8."""
    defaults = AttrDict(dict(
        expts=1, reps=5, rounds=1, qubits=[0], f0g1_cavity=1,
        normalize=False, active_reset=True, man_reset=True,
        storage_reset=swap_stors, pre_relax_delay=100, relax_delay=200,
        prepulse=False, postpulse=True, init_fock=False, init_stor=0,
        readout="multiparity", parity_fast=False,
        cond_sec_phase=-90, phase_second_pulse=180,
        occupations=[1] + [0] * len(swap_stors), swap_stors=swap_stors,
        use_multiphoton_swap=False, storage_pulse_wait_us=0.2, ro_stor=0,
        update_phases=True, detunings=[0.0] * len(swap_stors),
        floquet_cycles=[0], swept_params=["floquet_cycle"],
    ))
    defaults.update(ACTIVE_RESET_DEFAULTS)
    defaults.update(FLOQUET_DEFAULTS)
    return defaults


@pytest.fixture(scope="module")
def runner():
    station = mock_station(**pinned_config_set("preload_current"))
    assert station.is_mock, "refusing to acquire against real instruments"
    return CharacterizationRunner(station=station, ExptClass=FockPopulationExperiment,
                                  default_expt_cfg=population_defaults(SWAP_STORS), show=False)


def test_scan_acquires_every_job_and_saves_small_layers(runner, tmp_path):
    scan = CentralBosonAGPDisorderScan.from_pattern(
        [0, 40], [1., -1.], photon_sector_number=1, direction=2, swap_stors=SWAP_STORS,
        cycles=[0, 2, 4], calibration=CONFUSION)
    # The worker analyzes before it saves; a local run does so only when asked.
    scan.acquire(runner, batch_size=2, use_queue=False, log=False, show=False,
                 go_kwargs=dict(analyze=True))
    for part in scan.children:
        (job,) = part.children  # N=1 with S2 occupied: one state
        assert isinstance(job, FockPopulationExperiment)
        assert np.asarray(job.data["idata"]).shape[0] == 3
        np.testing.assert_allclose(job.cfg.expt.detunings, part.detunings)
        assert "nmod4_mean_corrected" in job.data and "n2_err" in job.data
    scan.analyze()
    assert scan.data["C1_corrected"].shape == (2, 3)
    manifest = scan.save(directory=tmp_path)
    assert manifest.exists()
    view = CentralBosonAGPDisorderScan.from_h5file(manifest.with_suffix(".h5"))
    assert "idata" not in view.data
    view.display()

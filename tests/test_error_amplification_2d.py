"""ErrorAmplification2DExperiment: phase rule, and an end-to-end build + run on the mock station
(the QICK program build and ASM compile are real in mock mode; the data are zeros)."""
from copy import deepcopy

import numpy as np
import pytest
from slab import AttrDict

from experiments.single_qubit.error_amplification_2d import phase_sequence


def test_phase_sequence_presets():
    assert phase_sequence(4, 0) == [0.0, 0.0, 0.0, 0.0]                 # paper protocol a
    assert phase_sequence(4, 180) == [0.0, 180.0, 0.0, 180.0]           # = ErrorAmplification frequency mode
    assert phase_sequence(3, 75, 10) == [10.0, 85.0, 160.0]             # stepped (protocol b)
    assert phase_sequence(6, 75)[-1] == pytest.approx((5 * 75) % 360)
    assert phase_sequence(0, 75) == []


CONFIG_DICT = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}


@pytest.fixture(scope="module")
def mock_station():
    pytest.importorskip("qick")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from experiments import MultimodeStation
    from experiments.qsim.notebook_helpers.run_mode import RunSettings
    from job_server import JobClient
    try:
        station = MultimodeStation(user="pytest", experiment_name="260818_qsim_spectroscopy", project="EncSpec",
                                   mock=True, **RunSettings().station_configs(CONFIG_DICT))
    except Exception as exc:  # missing config versions, no soccfg snapshot, ...
        pytest.skip(f"cannot build a mock station here: {type(exc).__name__}: {exc}")
    assert station.is_mock
    return station, JobClient()


@pytest.mark.parametrize("y_param, y_values", [("length", [0.55, 0.57]), ("gain", [7900, 8100])])
def test_2d_error_amplification_builds_and_runs(mock_station, y_param, y_values):
    from experiments import CharacterizationRunner
    from experiments.single_qubit.error_amplification_2d import ErrorAmplification2DExperiment
    station, client = mock_station
    f0 = float(station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency'][0])
    cfg = AttrDict(dict(reps=20, rounds=1, qubit=0, qubits=[0], active_reset=False, man_reset=True,
                        storage_reset=True, relax_delay=200, man_no=1, pulse_name='pi',
                        pulse_type=['multiphoton', 'f0-g1', 'pi', 0], start=f0 - 0.1, step=0.05, expts=5,
                        y_param=y_param, y_values=y_values, n_pulses=[1, 2], phase_step_deg=75))
    runner = CharacterizationRunner(station=station, ExptClass=ErrorAmplification2DExperiment, default_expt_cfg=cfg,
                                    postprocessor=None, job_client=client, show=False)
    expt = runner.execute(go_kwargs=dict(analyze=False, progress=False, display=False))
    assert expt.data['avgi'].shape == (2, 2, 5)
    assert np.allclose(expt.data['y_pts'], y_values)
    assert list(expt.data['N_pts']) == [1, 2]
    d = expt.analyze()
    assert d['success'].shape == (2, 2, 5) and d['product'].shape == (2, 5)
    assert d['best_per_N'].shape == (2, 3)
    # the last program built is N = 2 at the last y: phases 0, 75
    assert expt.prog.phases == [0.0, 75.0]

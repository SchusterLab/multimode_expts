"""
Tests for verifying mock mode station creation.

These tests ensure the worker correctly creates a station in mock mode
and uses mock_data paths instead of production paths like D:/experiments.

Usage:
    pixi run pytest tests/test_mock_station.py -v
"""

import sys
from unittest.mock import MagicMock

# =============================================================================
# MOCK HARDWARE MODULES - Must happen BEFORE any other imports
# =============================================================================

def create_mock_module(name, **attrs):
    """Create a mock module with specified attributes."""
    mock = MagicMock()
    for k, v in attrs.items():
        setattr(mock, k, v)
    mock.__name__ = name
    return mock

# Mock qick and all its submodules (FPGA library)
class MockQickProgram:
    def __init__(self, soccfg=None, cfg=None):
        self.soccfg = soccfg
        self.cfg = cfg

class MockAveragerProgram(MockQickProgram):
    pass

class MockRAveragerProgram(MockQickProgram):
    pass

mock_qick = create_mock_module('qick')
mock_qick.QickProgram = MockQickProgram
mock_qick.AveragerProgram = MockAveragerProgram
mock_qick.RAveragerProgram = MockRAveragerProgram
mock_qick.QickConfig = MagicMock

mock_qick_helpers = create_mock_module('qick.helpers',
    gauss=lambda x: x,
    sin2=lambda x: x,
    tanh=lambda x: x,
    flat_top_gauss=lambda x: x,
)
mock_qick.helpers = mock_qick_helpers

sys.modules['qick'] = mock_qick
sys.modules['qick.helpers'] = mock_qick_helpers

# Mock telnetlib (removed in Python 3.12+)
sys.modules['telnetlib'] = MagicMock()
sys.modules['Pyro4'] = MagicMock()
sys.modules['visa'] = MagicMock()
sys.modules['pyvisa'] = MagicMock()

mock_lmfit = create_mock_module('lmfit')
mock_lmfit_models = create_mock_module('lmfit.models')
mock_lmfit.models = mock_lmfit_models
mock_lmfit.Model = MagicMock
sys.modules['lmfit'] = mock_lmfit
sys.modules['lmfit.models'] = mock_lmfit_models

# =============================================================================
# Now import the modules under test
# =============================================================================

from experiments.station import MultimodeStation
from experiments.mock_hardware import MockQickConfig, MockInstrumentManager, MockYokogawa
from job_server.worker import JobWorker


class TestMockStationCreation:
    """Test that MultimodeStation correctly initializes in mock mode."""

    def test_station_is_mock_when_mock_true(self):
        """Station.is_mock should be True when mock=True is passed."""
        station = MultimodeStation(mock=True)
        assert station.is_mock is True

    def test_station_uses_mock_data_path(self):
        """In mock mode, data_path should be under mock_data, not D:/experiments."""
        station = MultimodeStation(mock=True)

        # Should use mock_data directory, not production path
        assert "mock_data" in str(station.data_path)
        assert "D:" not in str(station.data_path)
        assert "experiments" not in str(station.output_root) or "mock" in str(station.output_root)

    def test_station_output_root_is_mock_data(self):
        """In mock mode, output_root should be repo_root/mock_data."""
        station = MultimodeStation(mock=True)

        # output_root should end with mock_data
        assert station.output_root.name == "mock_data"
        assert station.output_root.exists()

    def test_station_soc_is_mock(self):
        """Station.soc should be a MockQickConfig in mock mode."""
        station = MultimodeStation(mock=True)
        assert isinstance(station.soc, MockQickConfig)

    def test_station_im_is_mock(self):
        """Station.im should be a MockInstrumentManager in mock mode."""
        station = MultimodeStation(mock=True)
        assert isinstance(station.im, MockInstrumentManager)


class TestWorkerMockMode:
    """Test that JobWorker correctly creates station in mock mode."""

    def test_worker_station_is_mock_when_mock_mode_true(self):
        """Worker's station should be in mock mode when mock_mode=True."""
        worker = JobWorker(mock_mode=True)
        assert worker.station.is_mock is True
        assert worker.mock_mode is True

    def test_worker_station_uses_mock_data_path(self):
        """Worker's station should use mock_data path in mock mode."""
        worker = JobWorker(mock_mode=True)
        assert "mock_data" in str(worker.station.data_path)
        assert "D:" not in str(worker.station.data_path)

    def test_worker_station_preserves_mock_after_config_update(self):
        """Station should stay in mock mode after _update_station_from_job_config."""
        import json
        from copy import deepcopy

        worker = JobWorker(mock_mode=True)

        # Verify initial state
        assert worker.station.is_mock is True
        original_is_mock = worker.station.is_mock

        # Create a clean copy of hardware_cfg without the dataset objects
        # (this mimics what CharacterizationRunner does when serializing)
        hardware_cfg_clean = deepcopy(dict(worker.station.hardware_cfg))
        if "device" in hardware_cfg_clean and "storage" in hardware_cfg_clean["device"]:
            hardware_cfg_clean["device"]["storage"].pop("_ds_storage", None)
            hardware_cfg_clean["device"]["storage"].pop("_ds_floquet", None)

        # Create a mock job config (simulating what would come from notebook)
        station_config = {
            "experiment_name": "test_experiment",
            "hardware_cfg": hardware_cfg_clean,
            "multimode_cfg": dict(worker.station.multimode_cfg),
            "storage_man_data": worker.station.ds_storage.df.to_dict(),
            "floquet_data": worker.station.ds_floquet.df.to_dict(),
        }

        # Update station from job config
        worker._update_station_from_job_config(json.dumps(station_config))

        # Station should STILL be in mock mode
        assert worker.station.is_mock is True, \
            f"is_mock changed from {original_is_mock} to {worker.station.is_mock} after config update"
        assert "mock_data" in str(worker.station.data_path), \
            f"data_path is {worker.station.data_path}, expected mock_data"

    def test_debug_full_job_flow(self):
        """Debug test: trace exactly what happens during job execution."""
        import json
        from copy import deepcopy

        print("\n" + "="*60)
        print("DEBUG: Full job flow simulation")
        print("="*60)

        # Step 1: Create worker
        print("\n[1] Creating worker with mock_mode=True...")
        worker = JobWorker(mock_mode=True)
        print(f"    worker.mock_mode = {worker.mock_mode}")
        print(f"    worker.station.is_mock = {worker.station.is_mock}")
        print(f"    worker.station._is_mock = {worker.station._is_mock}")
        print(f"    worker.station.output_root = {worker.station.output_root}")
        print(f"    worker.station.data_path = {worker.station.data_path}")

        assert worker.mock_mode is True, "worker.mock_mode should be True"
        assert worker.station.is_mock is True, "station.is_mock should be True"

        # Step 2: Prepare job config (like CharacterizationRunner does)
        print("\n[2] Preparing job config...")
        hardware_cfg_clean = deepcopy(dict(worker.station.hardware_cfg))
        if "device" in hardware_cfg_clean and "storage" in hardware_cfg_clean["device"]:
            hardware_cfg_clean["device"]["storage"].pop("_ds_storage", None)
            hardware_cfg_clean["device"]["storage"].pop("_ds_floquet", None)

        station_config = {
            "experiment_name": "debug_test_experiment",
            "hardware_cfg": hardware_cfg_clean,
            "multimode_cfg": dict(worker.station.multimode_cfg),
            "storage_man_data": worker.station.ds_storage.df.to_dict(),
            "floquet_data": worker.station.ds_floquet.df.to_dict(),
        }
        station_config_json = json.dumps(station_config)
        print(f"    Config JSON length: {len(station_config_json)} bytes")

        # Step 3: Check state BEFORE update
        print("\n[3] State BEFORE _update_station_from_job_config:")
        print(f"    worker.station.is_mock = {worker.station.is_mock}")
        print(f"    worker.station._is_mock = {worker.station._is_mock}")

        # Step 4: Call _update_station_from_job_config
        print("\n[4] Calling _update_station_from_job_config...")
        worker._update_station_from_job_config(station_config_json)

        # Step 5: Check state AFTER update
        print("\n[5] State AFTER _update_station_from_job_config:")
        print(f"    worker.station.is_mock = {worker.station.is_mock}")
        print(f"    worker.station._is_mock = {worker.station._is_mock}")
        print(f"    worker.station.output_root = {worker.station.output_root}")
        print(f"    worker.station.data_path = {worker.station.data_path}")

        # Assertions
        assert worker.station.is_mock is True, \
            f"FAIL: is_mock became {worker.station.is_mock} after config update!"
        assert "mock_data" in str(worker.station.data_path), \
            f"FAIL: data_path is {worker.station.data_path}, should contain 'mock_data'"
        assert "D:" not in str(worker.station.data_path), \
            f"FAIL: data_path contains 'D:': {worker.station.data_path}"

        print("\n[6] All assertions passed!")
        print("="*60)

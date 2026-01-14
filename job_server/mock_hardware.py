"""
Mock hardware implementations for testing without real instruments.

This module provides mock versions of:
- MockQickConfig: Simulates QICK SoC configuration
- MockInstrumentManager: Simulates instrument access
- MockStation: Full MultimodeStation mock for job worker testing

Usage:
    from job_server.mock_hardware import MockStation

    station = MockStation(experiment_name="test_experiment")
    # Use station.soc, station.config_thisrun, etc. as normal
    # Hardware calls print to console instead of executing
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from copy import deepcopy
import yaml
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from slab.datamanagement import AttrDict


class MockQickConfig:
    """
    Mock QICK SoC configuration for testing.

    Simulates the QickConfig interface without real hardware.
    Provides conversion methods that return realistic values.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        """
        Initialize mock QICK config.

        Args:
            cfg: Optional config dict. If None, creates default mock config.
        """
        self._cfg = cfg or self._create_default_cfg()
        print("[MOCK] QickConfig initialized")

    def _create_default_cfg(self) -> dict:
        """Create a realistic mock QICK config structure."""
        return {
            "gens": [
                {
                    "maxv": 32767,
                    "maxv_scale": 1.0,
                    "samps_per_clk": 16,
                    "f_fabric": 430.08,
                    "type": "full",
                }
                for _ in range(7)
            ],
            "readouts": [
                {
                    "freq": 100.0,
                    "f_fabric": 430.08,
                }
            ],
            "tprocs": [
                {
                    "f_time": 384.0,
                }
            ],
        }

    def __getitem__(self, key):
        return self._cfg[key]

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def freq2reg(self, f, gen_ch=0, ro_ch=None):
        """Mock frequency to register conversion."""
        print(f"[MOCK] freq2reg: f={f} MHz, gen_ch={gen_ch}")
        return int(f * 1000)  # Simplified conversion

    def reg2freq(self, reg, gen_ch=0, ro_ch=None):
        """Mock register to frequency conversion."""
        return reg / 1000.0

    def us2cycles(self, us, gen_ch=0, ro_ch=None):
        """Mock microseconds to cycles conversion."""
        cycles = int(us * 430.08)  # Approximate conversion
        print(f"[MOCK] us2cycles: {us} us -> {cycles} cycles")
        return cycles

    def cycles2us(self, cycles, gen_ch=0, ro_ch=None):
        """Mock cycles to microseconds conversion."""
        return cycles / 430.08

    def deg2reg(self, deg, gen_ch=0):
        """Mock degrees to register conversion."""
        return int(deg * 65536 / 360)

    def reg2deg(self, reg, gen_ch=0):
        """Mock register to degrees conversion."""
        return reg * 360 / 65536

    def __repr__(self):
        return "<MockQickConfig: Simulated QICK hardware>"


class MockQickSoc:
    """Mock QICK SoC proxy for InstrumentManager."""

    def __init__(self):
        self._cfg = MockQickConfig()._cfg
        print("[MOCK] QickSoc proxy initialized")

    def get_cfg(self) -> dict:
        """Return mock configuration."""
        return self._cfg

    def acquire(self, *args, **kwargs):
        """
        Mock data acquisition.

        Returns simulated Rabi oscillation data:
        - 100 points on x-axis over 1 us (0 to 1.0 us)
        - Oscillation frequency: 5 MHz
        - Decay time constant: 10 us (T2)

        Returns:
            Tuple of (xpts, [[avgi]], [[avgq]]) matching real QICK format
        """
        n_expts = kwargs.get("expts", 100)
        print(f"[MOCK] acquire: generating {n_expts} data points")

        # Generate x points: 100 points over 1 us
        xpts = np.linspace(0, 1.0, n_expts)  # in microseconds

        # Simulation parameters
        osc_freq_mhz = 5.0  # 5 MHz oscillation
        decay_time_us = 10.0  # T2 = 10 us decay constant

        # Generate simulated Rabi oscillation with decay
        # I = A * cos(2*pi*f*t) * exp(-t/T2) + noise
        # Q = A * sin(2*pi*f*t) * exp(-t/T2) + noise
        omega = 2 * np.pi * osc_freq_mhz  # angular frequency (rad/us since t is in us)
        decay = np.exp(-xpts / decay_time_us)
        noise_amplitude = 5.0

        avgi = np.cos(omega * xpts) * decay * 100 + np.random.randn(n_expts) * noise_amplitude
        avgq = np.sin(omega * xpts) * decay * 100 + np.random.randn(n_expts) * noise_amplitude

        print(f"[MOCK] acquire: returning simulated I/Q data (f={osc_freq_mhz} MHz, T2={decay_time_us} us)")
        return xpts, [[avgi]], [[avgq]]


class MockInstrumentManager(dict):
    """
    Mock InstrumentManager that prints operations instead of executing.

    Inherits from dict to match the real InstrumentManager interface.
    """

    def __init__(self):
        super().__init__()
        self["Qick101"] = MockQickSoc()  # Default QICK alias
        print("[MOCK] InstrumentManager initialized")

    def keys(self):
        return ["Qick101"]

    def __repr__(self):
        return "<MockInstrumentManager: Simulated hardware access>"


class MockYokogawa:
    """Mock Yokogawa voltage source."""

    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        self._voltage = 0.0
        self._output_enabled = False
        print(f"[MOCK] Yokogawa {name} initialized at {address}")

    def set_voltage(self, v: float):
        print(f"[MOCK] {self.name}.set_voltage({v})")
        self._voltage = v

    def get_voltage(self) -> float:
        return self._voltage

    def output_on(self):
        print(f"[MOCK] {self.name}.output_on()")
        self._output_enabled = True

    def output_off(self):
        print(f"[MOCK] {self.name}.output_off()")
        self._output_enabled = False


class MockStation:
    """
    Mock version of MultimodeStation for testing without hardware.

    Mimics the MultimodeStation interface but:
    - Uses mock hardware objects (prints instead of executing)
    - Loads real YAML configs (for structure validation)
    - Creates real output directories (for data file testing)
    - Returns simulated data from acquire()

    This allows testing the full job execution pipeline without
    connecting to actual instruments.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        hardware_config: str = "hardware_config_202505.yml",
        load_ds_floquet: bool = False,
        qubit_i: int = 0,
    ):
        """
        Initialize the mock measurement station.

        Args:
            experiment_name: Format is yymmdd_name. None defaults to today's date.
            hardware_config: Filename for the yaml config (in config_dir).
            load_ds_floquet: Whether to load floquet dataset.
            qubit_i: Qubit index to use.
        """
        self.repo_root = Path(__file__).resolve().parent.parent
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_mock_experiment'
        )
        self.qubit_i = qubit_i
        self.load_ds_floquet = load_ds_floquet

        print(f"[MOCK] MockStation initializing: {self.experiment_name}")

        self._initialize_configs(hardware_config)
        self._initialize_output_paths()
        self._initialize_mock_hardware()

        self.print()

    def _initialize_configs(self, hardware_config: str):
        """Load real configuration files for structure."""
        self.config_dir = self.repo_root / "configs"
        self.hardware_config_file = self.config_dir / hardware_config

        # Load real YAML config
        if self.hardware_config_file.exists():
            with self.hardware_config_file.open("r") as cfg_file:
                self.yaml_cfg = AttrDict(yaml.safe_load(cfg_file))
            print(f"[MOCK] Loaded config from {self.hardware_config_file}")
        else:
            # Create minimal mock config if file doesn't exist
            print(f"[MOCK] Config file not found, using minimal mock config")
            self.yaml_cfg = AttrDict({
                "device": {
                    "qubit": {"f_ge": [5000], "T1": [100]},
                    "readout": {"frequency": [7000], "relax_delay": [1000]},
                    "storage": {
                        "storage_man_file": "man1_storage_swap_dataset.csv",
                        "floquet_man_stor_file": "floquet_storage_swap_dataset.csv",
                    },
                    "multiphoton_config": {"file": "multiphoton_config.yml"},
                },
                "aliases": {"soc": "Qick101"},
                "data_management": {"output_root": str(self.repo_root / "data")},
            })

        # Create working copy
        self.config_thisrun = AttrDict(deepcopy(self.yaml_cfg))

        # Load multiphoton config (or create mock)
        try:
            self.multiphoton_config_file = (
                self.config_dir / self.config_thisrun.device.multiphoton_config.file
            )
            if self.multiphoton_config_file.exists():
                with self.multiphoton_config_file.open("r") as f:
                    self.multimode_cfg = AttrDict(yaml.safe_load(f))
            else:
                self.multimode_cfg = AttrDict({})
        except Exception:
            self.multimode_cfg = AttrDict({})

        # Mock dataset (don't actually load CSV)
        self.ds_storage = None
        self.ds_floquet = None
        self.storage_man_file = self.yaml_cfg.device.storage.storage_man_file
        print(f"[MOCK] Configs initialized (storage dataset mocked)")

    def _initialize_output_paths(self):
        """Create real output directories for testing."""
        try:
            self.output_root = Path(self.yaml_cfg.data_management.output_root)
        except (KeyError, AttributeError):
            self.output_root = self.repo_root / "data"

        # Create directories (these are real for data file testing)
        self.experiment_path = self.output_root / self.experiment_name
        self.data_path = self.experiment_path / "data"
        self.plot_path = self.experiment_path / "plots"
        self.log_path = self.experiment_path / "logs"
        self.autocalib_path = (
            self.plot_path / f'autocalibration_{datetime.now().strftime("%Y-%m-%d")}'
        )

        for subpath in [
            self.experiment_path,
            self.data_path,
            self.plot_path,
            self.log_path,
            self.autocalib_path,
        ]:
            subpath.mkdir(parents=True, exist_ok=True)

        print(f"[MOCK] Output paths created at: {self.experiment_path}")

    def _initialize_mock_hardware(self):
        """Initialize mock hardware objects."""
        self.im = MockInstrumentManager()
        self.soc = MockQickConfig()
        self.yoko_coupler = MockYokogawa(
            name="yoko_coupler", address="mock://192.168.137.148"
        )
        self.yoko_jpa = MockYokogawa(
            name="yoko_jpa", address="mock://192.168.137.149"
        )
        print("[MOCK] Mock hardware initialized")

    def print(self):
        """Print station information."""
        print(f"[MOCK STATION] Data path: {self.data_path}")
        print(f"[MOCK STATION] Config file: {self.hardware_config_file}")
        print(f"[MOCK STATION] Instruments: {list(self.im.keys())}")
        print(f"[MOCK STATION] SOC: {self.soc}")

    def load_data(self, filename: Optional[str] = None, prefix: Optional[str] = None):
        """Mock load data - returns empty dict for testing."""
        print(f"[MOCK] load_data called with filename={filename}, prefix={prefix}")
        return {}, {}, self.data_path / (filename or "mock_data.h5")

    def save_config(self):
        """Mock save config - prints instead of saving."""
        print("[MOCK] save_config called (no-op in mock mode)")

    def handle_config_update(self, write_to_file=False):
        """Mock config update handler."""
        print(f"[MOCK] handle_config_update called (write_to_file={write_to_file})")


# Convenience function to get a station (mock or real based on flag)
def get_station(mock: bool = True, **kwargs):
    """
    Get a station instance (mock or real).

    Args:
        mock: If True, return MockStation. If False, import and return real MultimodeStation.
        **kwargs: Arguments passed to station constructor.

    Returns:
        MockStation or MultimodeStation instance
    """
    if mock:
        return MockStation(**kwargs)
    else:
        from experiments.station import MultimodeStation
        return MultimodeStation(**kwargs)

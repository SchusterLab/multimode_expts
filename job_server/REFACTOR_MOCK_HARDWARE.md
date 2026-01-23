# Refactor: Mock Hardware & Auto-Detection

## Goal

1. Move mock hardware to `experiments/` module where it belongs
2. Add auto-detection so code "just works" on any machine
3. On BF5 (hardware machine): real instruments + job queue by default
4. On dev machines: mock instruments + local execution by default
5. Allow per-call overrides in both cases

## Current State

```
experiments/
└── station.py                    # MultimodeStation - always tries real hardware + database

job_server/
├── mock_hardware.py              # MockStation (duplicates MultimodeStation logic)
└── worker.py                     # Imports MockStation when --mock flag used
```

**Problems:**
- `MockStation` duplicates `MultimodeStation` logic
- `MultimodeStation` always requires database (breaks on dev machines)
- No auto-detection - must manually configure mock/real mode
- `sys.path` hacks in mock_hardware.py

## Target State

```
experiments/
├── station.py                    # MultimodeStation with mock param + auto-detection
└── mock_hardware.py              # Just the mock hardware classes

job_server/
└── worker.py                     # Uses MultimodeStation(mock=args.mock)
```

## Implementation

### Step 1: Create `experiments/mock_hardware.py`

Create new file with only the mock hardware classes (no `MockStation`, no `sys.path` hacks):

```python
"""
Mock hardware implementations for testing without real instruments.

These are used automatically when MultimodeStation(mock=True) or when
running on a dev machine (auto-detected).
"""

import numpy as np
from typing import Optional, Dict, Any


class MockQickConfig:
    """Mock QICK SoC configuration."""

    def __init__(self, cfg: Optional[Dict] = None):
        self._cfg = cfg or self._create_default_cfg()
        print("[MOCK] QickConfig initialized")

    # ... copy methods from job_server/mock_hardware.py lines 39-109 ...


class MockQickSoc:
    """Mock QICK SoC for acquire() calls."""

    def __init__(self, cfg: Optional[Dict] = None):
        self._cfg = cfg or {}
        print("[MOCK] QickSoc initialized")

    def acquire(self, *args, **kwargs):
        """Return simulated Rabi oscillation data."""
        # ... copy from job_server/mock_hardware.py lines 123-156 ...


class MockInstrumentManager(dict):
    """Mock instrument manager."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        print("[MOCK] InstrumentManager initialized")


class MockYokogawa:
    """Mock Yokogawa voltage source."""

    def __init__(self, name: str = "mock_yoko", address: str = "mock://"):
        self.name = name
        self.address = address
        self._voltage = 0.0
        print(f"[MOCK] Yokogawa {name} initialized")

    def set_voltage(self, voltage: float):
        print(f"[MOCK] {self.name} set voltage: {voltage}V")
        self._voltage = voltage

    def get_voltage(self) -> float:
        return self._voltage
```

### Step 2: Modify `experiments/station.py`

#### 2a. Add auto-detection method

```python
def _detect_hardware_machine(self) -> bool:
    """Check if we're on the BF5 machine with real hardware."""
    import getpass
    import socket

    # Add your hardware machine identifiers here
    hardware_users = {"connie", "labuser"}
    hardware_hostnames = {"bf5", "bf5-control"}

    try:
        if getpass.getuser().lower() in hardware_users:
            return True
        if socket.gethostname().lower() in hardware_hostnames:
            return True
    except Exception:
        pass

    return False
```

#### 2b. Modify `__init__` signature and add flag

```python
def __init__(
    self,
    user: Optional[str] = '',
    experiment_name: Optional[str] = None,
    hardware_config: Optional[str] = None,
    multiphoton_config: Optional[str] = None,
    storage_man_file: Optional[str] = None,
    floquet_file: Optional[str] = None,
    qubit_i: int = 0,
    mock: Optional[bool] = None,  # ADD: None = auto-detect
):
    ...
    # Auto-detect environment
    self.on_hardware_machine = self._detect_hardware_machine()
    self.mock = mock if mock is not None else (not self.on_hardware_machine)

    # Log what mode we're in
    if self.mock:
        print(f"[STATION] Running in MOCK mode (on_hardware_machine={self.on_hardware_machine})")

    self._initialize_configs(...)
    self._initialize_output_paths()
    self._initialize_hardware()
    ...
```

#### 2c. Modify `_initialize_hardware` for mock support

```python
def _initialize_hardware(self):
    """Connect to hardware (real or mock based on self.mock flag)."""
    if self.mock:
        from experiments.mock_hardware import (
            MockQickConfig,
            MockInstrumentManager,
            MockYokogawa,
        )
        self.im = MockInstrumentManager()
        self.soc = MockQickConfig()
        self.yoko_coupler = MockYokogawa(name='yoko_coupler')
        self.yoko_jpa = MockYokogawa(name='yoko_jpa')
    else:
        self.im = InstrumentManager(ns_address="192.168.137.25")
        self.soc = QickConfig(self.im[self.hardware_cfg["aliases"]["soc"]].get_cfg())
        self.yoko_coupler = YokogawaGS200(name='yoko_coupler', address='192.168.137.148')
        self.yoko_jpa = YokogawaGS200(name='yoko_jpa', address='192.168.137.149')
```

#### 2d. Modify `_initialize_configs` to work without database in mock mode

```python
def _initialize_configs(self, hardware_config, multiphoton_config, storage_man_file, floquet_file):
    """Load configuration files."""
    self.config_dir = self.repo_root / "configs"

    # In mock mode on dev machine, skip database and load directly from files
    if self.mock and not self.on_hardware_machine:
        self._initialize_configs_from_files(
            hardware_config, multiphoton_config, storage_man_file, floquet_file
        )
        return

    # ... existing database-based config loading ...


def _initialize_configs_from_files(self, hardware_config, multiphoton_config, storage_man_file, floquet_file):
    """Load configs directly from files (for mock mode without database)."""
    # Hardware config - use provided filename or default
    hw_file = hardware_config or "hardware_config_202505.yml"
    self.hardware_config_file = self.config_dir / hw_file
    if not self.hardware_config_file.exists():
        raise FileNotFoundError(f"Config not found: {self.hardware_config_file}")
    with self.hardware_config_file.open("r") as f:
        self.hardware_cfg = AttrDict(yaml.safe_load(f))

    # Multiphoton config
    mp_file = multiphoton_config or self.hardware_cfg.device.multiphoton_config.file
    self.multiphoton_config_file = self.config_dir / mp_file
    with self.multiphoton_config_file.open("r") as f:
        self.multimode_cfg = AttrDict(yaml.safe_load(f))

    # Storage datasets
    storage_file = storage_man_file or self.hardware_cfg.device.storage.storage_man_file
    self.storage_man_file = storage_file
    self.ds_storage, _ = self.load_storage_man_swap_dataset(storage_file)

    floquet = floquet_file or self.hardware_cfg.device.storage.floquet_man_stor_file
    self.floquet_file = floquet
    self.ds_floquet, _ = self.load_floquet_swap_dataset(floquet)

    print("[STATION] Loaded configs from files (database skipped)")
```

### Step 3: Modify `experiments/characterization_runner.py`

Change `use_queue` default to read from station:

```python
def __init__(
    self,
    station: "MultimodeStation",
    ExptClass: type,
    default_expt_cfg: AttrDict,
    preprocessor: Optional[PreProcessor] = None,
    postprocessor: Optional[PostProcessor] = None,
    job_client: Optional["JobClient"] = None,
    program: Optional[Any] = None,
    use_queue: Optional[bool] = None,  # CHANGE: None = auto from station
):
    self.station = station
    ...
    # Default: use queue on hardware machine, local otherwise
    self.use_queue = use_queue if use_queue is not None else station.on_hardware_machine
```

### Step 4: Modify `experiments/sweep_runner.py`

Same change as characterization_runner:

```python
def __init__(
    self,
    station: "MultimodeStation",
    ...
    use_queue: Optional[bool] = None,  # CHANGE: None = auto from station
):
    ...
    self.use_queue = use_queue if use_queue is not None else station.on_hardware_machine
```

### Step 5: Modify `job_server/worker.py`

Simplify the station initialization:

```python
# BEFORE (lines 206-211):
if self.mock_mode:
    from multimode_expts.job_server.mock_hardware import MockStation
    self.station = MockStation(experiment_name=self.experiment_name)
else:
    from multimode_expts.experiments.station import MultimodeStation
    self.station = MultimodeStation(experiment_name=self.experiment_name)

# AFTER:
from multimode_expts.experiments.station import MultimodeStation
self.station = MultimodeStation(experiment_name=self.experiment_name, mock=self.mock_mode)
```

### Step 6: Delete `job_server/mock_hardware.py`

```bash
rm job_server/mock_hardware.py
```

## Final Usage

```python
# On BF5 (auto-detected)
station = MultimodeStation()
# → on_hardware_machine=True, mock=False
runner = CharacterizationRunner(station, ...)
# → use_queue=True (job queue)
runner.execute()

# On dev laptop (auto-detected)
station = MultimodeStation()
# → on_hardware_machine=False, mock=True
runner = CharacterizationRunner(station, ...)
# → use_queue=False (local execution)
runner.execute()

# Override when needed
station = MultimodeStation(mock=False)  # Force real hardware
runner = CharacterizationRunner(station, use_queue=False)  # Force local
```

## Files Changed Summary

| File | Action |
|------|--------|
| `experiments/mock_hardware.py` | CREATE |
| `experiments/station.py` | MODIFY - add mock param, auto-detection, file-based config loading |
| `experiments/characterization_runner.py` | MODIFY - default use_queue from station |
| `experiments/sweep_runner.py` | MODIFY - default use_queue from station |
| `job_server/worker.py` | MODIFY - use MultimodeStation(mock=...) |
| `job_server/mock_hardware.py` | DELETE |

## Testing

1. **On dev machine**: `station = MultimodeStation()` should auto-detect mock mode, skip database
2. **On BF5**: `station = MultimodeStation()` should use real hardware + database
3. **Worker mock mode**: `python -m multimode_expts.job_server.worker --mock` should work
4. **Worker real mode**: `python -m multimode_expts.job_server.worker` should work

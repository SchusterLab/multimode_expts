"""
MultimodeStation: Central hardware and configuration management.

This module provides the MultimodeStation class which manages:
- Hardware connections (QICK RFSoC, InstrumentManager, Yokogawa sources)
- Configuration files (hardware config, multiphoton config)
- Data paths and output directories
- Storage-manipulate swap dataset

Supports both real hardware and mock modes:
- Real hardware: Connects to actual instruments (production on Pippin)
- Mock mode: Real qick.QickConfig + stub MockQickSoc (no FPGA bytes go out).
  Used to validate qick programs without hardware overhead.
  See docs/mock_mode_architecture.md for the design.

Usage:
    from experiments.station import MultimodeStation

    # Default: connect to real hardware on the prod PC
    station = MultimodeStation(experiment_name="241215_calibration")

    # Force mock mode at construction (e.g., worker --mock)
    station = MultimodeStation(mock=True)

    # Mid-session swap (preserves all in-memory state — hardware_cfg fits,
    # multimode_cfg, datasets, etc.). Output paths auto-redirect to mock_data/.
    station.use_mock_instruments()
    # ... iterate on a buggy qick program against the stub ...
    station.use_real_instruments()

    # Access hardware: station.soccfg, station.im
    # Access config: station.hardware_cfg, station.multimode_cfg
    # Access paths: station.data_path, station.plot_path
    # Check mode: station.is_mock
"""

import copy
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import socket

import numpy as np
import yaml

from qick import QickConfig

from slab import AttrDict, get_current_filename
from slab.datamanagement import SlabFile
from slab.instruments import InstrumentManager
from slab.instruments.voltsource import YokogawaGS200

from experiments.dataset import FloquetStorageSwapDataset, StorageManSwapDataset
from experiments.mock_hardware import MockInstrumentManager, MockYokogawa

from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager, ConfigType

PIPPIN_HOSTNAME = 'pippin-meas'

def is_production_pc() -> bool:
    """Whether we're running on the Pippin measurement PC.

    Currently unused — kept for the eventual off-prod-PC support, which will
    need a separate code path for config loading and won't share the
    mock-instruments switch with the prod PC.
    """
    return socket.gethostname() == PIPPIN_HOSTNAME

# YAML representers for numpy types
def _np_float_representer(dumper, data):
    return dumper.represent_float(float(data))

def _np_int_representer(dumper, data):
    return dumper.represent_int(int(data))

yaml.add_representer(np.float64, _np_float_representer)
yaml.add_representer(np.float32, _np_float_representer)
yaml.add_representer(np.int64, _np_int_representer)
yaml.add_representer(np.int32, _np_int_representer)

# Prevent yaml from using anchors and aliases
yaml.Dumper.ignore_aliases = lambda *args: True


class MultimodeStation:
    """
    Central class representing a measurement setup.

    Controls:
        - InstrumentManager
        - QICK RFSoC
        - Hardware config yaml file
        - Manipulate-storage swap database file
        - Multiphoton config yaml file
        - Output paths for data/plots/logs
        - Yokogawa sources for JPA and coupler flux (optional)

    Supports both real hardware and mock modes:
        - Real hardware: Uses database for config versioning, connects to instruments
        - Mock mode: Loads configs directly from YAML files, uses simulated hardware

    Attributes:
        soc: QickConfig object for FPGA control (or MockQickConfig in mock mode)
        im: InstrumentManager for hardware access (or MockInstrumentManager in mock mode)
        hardware_cfg: AttrDict of current hardware configuration
        ds_storage: StorageManSwapDataset for this run
        data_path: Path to data directory
        plot_path: Path to plots directory
        log_path: Path to logs directory
        is_mock: Whether station connects to real or mock hardware
    """

    def __init__(
        self,
        user: Optional[str] = '',
        experiment_name: Optional[str] = None,
        hardware_config: Optional[str] = None,
        multiphoton_config: Optional[str] = None,
        storage_man_file: Optional[str] = None,
        floquet_file: Optional[str] = None,
        qubit_i: int = 0,
        mock: Optional[bool] = None,
        project: Optional[str] = None,
        log_measurements: bool = False,
    ):
        """
        Initialize the measurement station.

        Args:
            user: Username for tracking config changes
            experiment_name: Format is yymmdd_name. None defaults to today's date.
            hardware_config: Filename or version ID (e.g., CFG-HW-20260115-00001). If None, loads from main version in database.
            multiphoton_config: Filename or version ID (e.g., CFG-MP-20260115-00001). If None, loads from main version in database.
            storage_man_file: Filename or version ID (e.g., CFG-M1-20260115-00001). If None, loads from main version in database.
            qubit_i: Qubit index to use.
            mock: If True, install MockQickSoc + MockYokogawa stubs (no FPGA bytes go out).
                  If False or None (default), connect to real hardware. See use_mock_instruments()
                  for mid-session swap that preserves in-memory state.
        """
        self.repo_root = Path(__file__).resolve().parent.parent
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_experiment'
        )
        self.qubit_i = qubit_i
        self.user = user
        # Default project for lab-notebook entries. Settable post-init via
        # station.project = '...'. None falls back to module inference.
        self.project = project
        # Opt-in toggle for runner-driven auto-logging. Default False so the
        # vault stays silent for users who don't want lab-notebook entries.
        # The user enables per session via `station.log_measurements = True`
        # or via the `log_measurements=True` kwarg above. Per-call `log=True`
        # on a runner.run/.run_local/.execute always overrides.
        self.log_measurements = log_measurements

        # Determine mock mode. Default to real (mock must be explicit) —
        # off-prod-PC support will need its own code path, not the mock flag.
        self._is_mock = bool(mock) if mock is not None else False

        # Config loading always uses database/versioning (same for mock and real)
        self._initialize_configs(hardware_config, multiphoton_config, storage_man_file, floquet_file)

        # Output paths and hardware - routing handled internally based on mock mode
        self._initialize_output_paths()
        self._initialize_hardware()

        # Optional Obsidian-style lab-notebook vault (no-op if vault_root absent from config)
        self._initialize_vault_root()

        self.print()

    @property
    def is_mock(self) -> bool:
        """Whether instrument calls currently route to mocks vs real hardware.

        Reflects the current state: flipped by use_mock_instruments() /
        use_real_instruments() in addition to the constructor flag.
        """
        return self._is_mock

    def __getattr__(self, name):
        # Tripwire for the soc → soccfg rename. Plain AttributeError would be
        # less helpful — the new name is one char away, easy to mistype.
        # __getattr__ only fires when normal attribute lookup misses, so this
        # has no impact on hot paths.
        if name == "soc":
            raise AttributeError(
                "MultimodeStation.soc was renamed to .soccfg "
                "(it's a QickConfig, not a QickSoc — the old name was misleading). "
                "Replace all station.soc references with station.soccfg. "
                "See docs/mock_mode_architecture.md."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _initialize_configs(self, hardware_config, multiphoton_config, storage_man_file, floquet_file):
        """Load configuration files from paths, version IDs, or main versions."""
        self.config_dir = self.repo_root / "configs"

        db = get_database()
        config_manager = ConfigVersionManager(self.config_dir)

        with db.session() as session:
            # Load hardware config
            hw_config_path = self._resolve_config_path(
                hardware_config, ConfigType.HARDWARE_CONFIG, config_manager, session, required=True
            )
            with hw_config_path.open("r") as cfg_file:
                self.hardware_cfg = AttrDict(yaml.safe_load(cfg_file))
            self.hardware_config_file = hw_config_path


            # Load multiphoton config
            mp_config_path = self._resolve_config_path(
                multiphoton_config, ConfigType.MULTIPHOTON_CONFIG, config_manager, session, required=True
            )
            with mp_config_path.open("r") as f:
                self.multimode_cfg = AttrDict(yaml.safe_load(f))
            self.multiphoton_config_file = mp_config_path

            # Load storage-man swap dataset
            storage_man_path = self._resolve_config_path(
                storage_man_file, ConfigType.MAN1_STORAGE_SWAP, config_manager, session, required=True
            )
            self.storage_man_file = storage_man_path.name
            ds_storage, _ = self.load_storage_man_swap_dataset(
                storage_man_path.name, parent_path=storage_man_path.parent
            )
            self.ds_storage = ds_storage

            # Load floquet swap dataset
            self.ds_floquet = None
            floquet_path = self._resolve_config_path(
                floquet_file, ConfigType.FLOQUET_STORAGE_SWAP, config_manager, session, required=True
            )
            self.floquet_file = floquet_path.name
            ds_floquet, _ = self.load_floquet_swap_dataset(
                floquet_path.name, parent_path=floquet_path.parent
            )
            self.ds_floquet = ds_floquet

            # Make datasets available via hardware_cfg for all code paths
            # (CharacterizationRunner/worker will re-inject after deepcopy to ensure live reference)
            self.hardware_cfg.device.storage._ds_storage = self.ds_storage
            self.hardware_cfg.device.storage._ds_floquet = self.ds_floquet

    def _resolve_config_path(
        self,
        config_spec: Optional[str],
        config_type: ConfigType,
        config_manager: ConfigVersionManager,
        session,
        required: bool = False
    ) -> Optional[Path]:
        """
        Resolve a config specification to an actual file path.

        Args:
            config_spec: Can be:
                - None: Load from main version in database
                - Filename (e.g., "hardware_config_202505.yml"): Load from configs/ directory
                - Version ID (e.g., "CFG-HW-20260115-00001"): Load versioned snapshot
            config_type: Type of config being resolved
            config_manager: ConfigVersionManager instance
            session: Database session
            required: If True, raises error when config_spec is None and no main version exists

        Returns:
            Path to the config file

        Raises:
            ValueError: If required=True and no config can be found
        """
        # Case 1: Explicit version ID specified
        if config_spec and config_spec.startswith("CFG-"):
            version_path = config_manager.get_config_path(config_spec, session)
            if version_path is None:
                raise ValueError(f"Config version {config_spec} not found in database")
            print(f"[STATION] Using {config_type.value} version: {config_spec}")
            return version_path

        # Case 2: Filename specified
        elif config_spec is not None:
            config_path = self.config_dir / config_spec
            if not config_path.exists():
                raise FileNotFoundError(f"Config file {config_path} not found")
            print(f"[STATION] Using {config_type.value} file: {config_spec}")
            return config_path

        # Case 3: No specification - try to load main version
        else:
            main_version = config_manager.get_main_version(config_type, session)
            if main_version is None:
                if required:
                    raise ValueError(
                        f"No main {config_type.value} set in database and no file/version specified. "
                        f"Either pass {config_type.value} parameter or set a main version with "
                        f"config_manager.set_main_version(ConfigType.{config_type.name}, version_id, session)"
                    )
                else:
                    return None

            version_path = Path(main_version.snapshot_path)
            print(f"[STATION] Using main {config_type.value} version: {main_version.version_id}")
            return version_path


    def _initialize_output_paths(self):
        """Initialize output paths - routes to real or mock based on mode."""
        if self._is_mock:
            self._initialize_output_paths_mock()
        else:
            self._initialize_output_paths_real()

    def _initialize_output_paths_real(self):
        """Create output directories for real hardware mode."""
        self.output_root = Path(self.hardware_cfg.data_management.output_root)
        if not self.output_root.exists():
            raise FileNotFoundError(
                f"Output root {self.output_root} does not exist. "
                "Check your data_management config."
            )

        self.experiment_path = self.output_root / self.experiment_name
        self.data_path = self.experiment_path / "data"
        self.expt_objs_path = self.experiment_path / "expt_objs"
        self.plot_path = self.experiment_path / "plots"
        self.log_path = self.experiment_path / "logs"
        self.autocalib_path = (
            self.plot_path / f'autocalibration_{datetime.now().strftime("%Y-%m-%d")}'
        )

        for subpath in [
            self.experiment_path,
            self.data_path,
            self.expt_objs_path,
            self.plot_path,
            self.log_path,
            self.autocalib_path,
        ]:
            if not subpath.exists():
                os.makedirs(subpath)
                print("Directory created at:", subpath)

    def _initialize_hardware(self):
        """Initialize hardware - routes to real or mock based on mode."""
        if self._is_mock:
            self._initialize_hardware_mock()
        else:
            self._initialize_hardware_real()

    def _initialize_hardware_real(self):
        """Connect to real hardware."""
        self.im = InstrumentManager(ns_address="192.168.137.26")
        self.soccfg = QickConfig(self.im[self.hardware_cfg["aliases"]["soc"]].get_cfg())
        self.yoko_coupler = YokogawaGS200(name='yoko_coupler', address='192.168.137.148')
        self.yoko_jpa = YokogawaGS200(name='yoko_jpa', address='192.168.137.149')

    def _initialize_output_paths_mock(self):
        """Create output directories for mock mode.

        Hardcoded to C:/experiments/mock_data on the prod PC. Off-prod-PC mode
        will eventually need its own path resolution — flagged in
        docs/mock_mode_architecture_plan.md.
        """
        self.output_root = Path("C:/experiments/mock_data")

        # Create directories (real directories for data file testing)
        self.experiment_path = self.output_root / self.experiment_name
        self.data_path = self.experiment_path / "data"
        self.expt_objs_path = self.experiment_path / "expt_objs"
        self.plot_path = self.experiment_path / "plots"
        self.log_path = self.experiment_path / "logs"
        self.autocalib_path = (
            self.plot_path / f'autocalibration_{datetime.now().strftime("%Y-%m-%d")}'
        )

        for subpath in [
            self.experiment_path,
            self.data_path,
            self.expt_objs_path,
            self.plot_path,
            self.log_path,
            self.autocalib_path,
        ]:
            subpath.mkdir(parents=True, exist_ok=True)

        print(f"[MOCK STATION] Output paths created at: {self.experiment_path}")

    def _initialize_hardware_mock(self):
        """Initialize for mock-only mode.

        Fetches a real QickConfig from the live Pyro proxy (required so the
        qick library's program-init validators see a complete soccfg). Then
        installs MockQickSoc + MockYokogawa stubs for instrument access. Does
        not claim real yokos.

        After this, use_real_instruments() will raise — reconstruct the
        station with mock=False to switch to real mode.
        """
        try:
            real_im = InstrumentManager(ns_address="192.168.137.26")
            qick_alias = self.hardware_cfg["aliases"]["soc"]
            self.soccfg = QickConfig(real_im[qick_alias].get_cfg())
        except Exception as e:
            raise RuntimeError(
                f"Mock mode requires a reachable QICK Pyro proxy to fetch the "
                f"real soccfg (got: {e!r}). Off-prod-PC mock mode is not yet "
                f"supported — see docs/mock_mode_architecture_plan.md."
            ) from e

        self._install_mock_instruments()
        print("[MOCK STATION] Mock hardware initialized (real soccfg + MockQickSoc)")

    def _install_mock_instruments(self):
        """Build mock im + yokos and assign to self. Does not touch soc/configs."""
        qick_alias = self.hardware_cfg["aliases"]["soc"]
        self.im = MockInstrumentManager(qick_alias=qick_alias)
        self.yoko_coupler = MockYokogawa(name="yoko_coupler", address="mock://192.168.137.148")
        self.yoko_jpa = MockYokogawa(name="yoko_jpa", address="mock://192.168.137.149")

    # ---- Mid-session mock/real instrument swap ----

    _MOCK_SWAP_PATH_KEYS = (
        "output_root", "experiment_path", "data_path", "expt_objs_path",
        "plot_path", "log_path", "autocalib_path",
    )

    def use_mock_instruments(self):
        """Swap im, yokos, and output paths to mock versions.

        Preserves all in-memory state: hardware_cfg, multimode_cfg, ds_storage,
        ds_floquet, soc, experiment_name, user. Idempotent — no-op if already
        in mock mode.

        Cached real instruments and paths are restored by use_real_instruments().
        """
        if self._is_mock:
            return
        # cache real state
        self._real_im = self.im
        self._real_yoko_coupler = self.yoko_coupler
        self._real_yoko_jpa = self.yoko_jpa
        self._real_paths = {k: getattr(self, k) for k in self._MOCK_SWAP_PATH_KEYS}
        # install mocks + redirect paths
        self._install_mock_instruments()
        self._initialize_output_paths_mock()
        self._is_mock = True
        print("[STATION] switched to MOCK instruments")

    def use_real_instruments(self):
        """Restore real instruments and paths cached by use_mock_instruments().

        Raises if the station was constructed in mock mode (no real cache
        available — reconstruct with mock=False instead).
        """
        if not self._is_mock:
            return
        if not hasattr(self, "_real_im"):
            raise RuntimeError(
                "No real instruments cached — station was constructed with "
                "mock=True. Reconstruct with mock=False to use real hardware."
            )
        self.im = self._real_im
        self.yoko_coupler = self._real_yoko_coupler
        self.yoko_jpa = self._real_yoko_jpa
        for k, v in self._real_paths.items():
            setattr(self, k, v)
        self._is_mock = False
        print("[STATION] switched to REAL instruments")

    def print(self):
        """Print station information."""
        if self._is_mock:
            print(f"[MOCK STATION] Data path: {self.data_path}")
            print(f"[MOCK STATION] Config file: {self.hardware_config_file}")
            print(f"[MOCK STATION] Instruments: {list(self.im.keys())}")
            print(f"[MOCK STATION] soccfg: {self.soccfg}")
        else:
            print("Data, plots, logs will be stored in:", self.experiment_path)
            print("Hardware configs will be read from", self.hardware_config_file)
            print(self.im.keys())
            print(self.soccfg)

    def load_data(self, filename: Optional[str] = None, prefix: Optional[str] = None):
        """Load data from HDF5 file."""
        if prefix is not None:
            data_file = self.data_path / get_current_filename(
                self.data_path, prefix=prefix, suffix=".h5"
            )
        else:
            data_file = self.data_path / filename
        with SlabFile(data_file) as a:
            attrs = {key: json.loads(a.attrs[key]) for key in list(a.attrs)}
            data = {key: np.array(a[key]) for key in list(a)}
        return data, attrs, data_file

    def load_storage_man_swap_dataset(
        self, filename: str, parent_path: Optional[str | Path] = None
    ):
        """Load storage-manipulate swap dataset."""
        if parent_path is None:
            parent_path = self.config_dir
        ds_storage = StorageManSwapDataset(filename, parent_path)
        ds_storage_file_path = ds_storage.file_path
        return ds_storage, ds_storage_file_path

    def load_floquet_swap_dataset(
        self, filename: str, parent_path: Optional[str | Path] = None
    ):
        """Load floquet storage-manipulate swap dataset."""
        if parent_path is None:
            parent_path = self.config_dir
        ds_floquet = FloquetStorageSwapDataset(filename, parent_path)
        ds_floquet_file_path = ds_floquet.file_path
        return ds_floquet, ds_floquet_file_path

    def _initialize_vault_root(self):
        """Read optional `data_management.vault_root` from hardware config.

        Sets `self.vault_root: Optional[Path]`. When None, `log_measurement` no-ops.
        """
        vault_root: Optional[Path] = None
        try:
            data_mgmt = self.hardware_cfg.get("data_management") if hasattr(self.hardware_cfg, "get") else None
            if data_mgmt is not None:
                raw = data_mgmt.get("vault_root") if hasattr(data_mgmt, "get") else None
                if raw:
                    vault_root = Path(str(raw))
        except Exception as exc:
            print(f"[STATION] Could not read vault_root from config: {exc}")
        self.vault_root = vault_root
        if self.vault_root is None:
            print("[STATION] vault_root not set in config; lab-notebook logging disabled.")
        else:
            print(f"[STATION] Lab-notebook vault: {self.vault_root}")

    @staticmethod
    def _safe_filename(s: str) -> str:
        """Reduce a free-form string to a filesystem-safe slug."""
        return re.sub(r"[^\w.\-]+", "_", str(s).strip()) or "untitled"

    @staticmethod
    def _infer_project(experiment) -> str:
        """Infer project name from experiment.__module__.

        e.g. 'experiments.qsim.kerr' -> 'qsim'. Falls back to 'misc' if the
        module path doesn't fit the experiments/<project>/<file> layout.
        """
        module = getattr(experiment, "__module__", "") or ""
        parts = module.split(".")
        if len(parts) >= 3 and parts[0] == "experiments":
            return parts[1]
        return "misc"

    def _to_plain(self, obj):
        """Recursively convert AttrDict / numpy / Path to YAML-safe primitives."""
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Catch numpy scalars (np.float64, np.int64, np.bool_, etc.) BEFORE the
        # primitive checks below: in numpy<2, np.float64 IS-A Python float so it
        # would sneak past, but yaml.safe_dump dispatches by exact type and
        # chokes on it ("cannot represent an object").
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._to_plain(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_plain(item) for item in obj]
        return str(obj)

    def log_measurement(
        self,
        experiment=None,
        fig=None,
        title: Optional[str] = None,
        project: Optional[str] = None,
        parameters: Optional[dict] = None,
        data_path=None,
        notes: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """Append a measurement entry to the user's daily lab-notebook file in the vault.

        Two calling modes:
        - Mode A (`experiment` is a slab Experiment): title, project, parameters,
          data_path are auto-derived from the experiment object.
        - Mode B (`experiment=None`): pass `title`, `project`, `parameters`,
          and optionally `data_path` explicitly. For manual sweeps and other
          patterns where there is no single Experiment instance.

        No-op (returns None) if `vault_root` is not configured, or if the
        station is currently in mock mode (mock measurements would pollute the
        real lab notebook). Flip out of mock mode briefly if you need to test
        the logging path itself.

        Returns the path to the daily markdown file, or None if disabled.
        """
        if self._is_mock:
            print("[log_measurement] mock mode active; skipping vault write.")
            return None
        if self.vault_root is None:
            print(
                "[log_measurement] vault_root not configured; skipping. "
                "Add 'data_management.vault_root: <path>' to the hardware config."
            )
            return None

        # Resolve project: explicit kwarg > station.project > module-infer (Mode A) > 'misc'
        if project is None:
            project = getattr(self, "project", None)
        if project is None and experiment is not None:
            project = self._infer_project(experiment)

        # Resolve metadata (Mode A auto-fill, Mode B explicit)
        if experiment is not None:
            title = title or type(experiment).__name__
            if parameters is None:
                cfg_expt = getattr(getattr(experiment, "cfg", None), "expt", None)
                parameters = self._to_plain(cfg_expt) if cfg_expt is not None else None
            if data_path is None:
                data_path = getattr(experiment, "fname", None)
        else:
            if title is None:
                raise ValueError("title is required when experiment is None")
            if project is None:
                raise ValueError(
                    "project is required when experiment is None "
                    "(or set station.project once for the session)"
                )
            if parameters is not None:
                parameters = self._to_plain(parameters)

        project = project or "misc"
        user = self.user or "unknown"
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%S")
        time_str = now.strftime("%H:%M:%S")
        file_ts = now.strftime("%Y-%m-%d_%H-%M-%S")

        # No fig auto-capture from pyplot here: the latest fignum may belong
        # to a previous, unrelated cell, leading to wrong-fig embeds. Callers
        # should pass `fig=` explicitly when they have one. The runners do this
        # already after their own scoped, new-fignum-only capture.

        target_dir = self.vault_root / "Lab" / user / project / now.strftime("%Y") / now.strftime("%m")
        figures_dir = target_dir / "figures"
        target_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Save plot copies into the vault. `fig` may be a single Figure or a
        # list of Figures; multi-fig case happens for displays that emit a 2D
        # summary plus auxiliary panels (e.g. ChevronFitting.display_results).
        figs: list = []
        if fig is not None:
            figs = list(fig) if isinstance(fig, (list, tuple)) else [fig]
        plot_rels: List[str] = []
        for idx, f in enumerate(figs):
            suffix = f"_p{idx + 1}" if len(figs) > 1 else ""
            plot_filename = f"{file_ts}_{self._safe_filename(title)}{suffix}.png"
            plot_path = figures_dir / plot_filename
            try:
                f.savefig(plot_path, bbox_inches="tight")
                plot_rels.append(f"figures/{plot_filename}")
            except Exception as exc:
                print(f"[log_measurement] Could not save figure {idx + 1}: {exc}")

        # Build per-measurement section meta
        section_meta: dict = {"timestamp": timestamp_iso}
        if experiment is not None:
            section_meta["experiment_class"] = type(experiment).__name__
        if hasattr(self, "experiment_name") and self.experiment_name:
            section_meta["experiment_name"] = self.experiment_name
        if data_path is not None:
            if isinstance(data_path, (list, tuple)):
                section_meta["data_path"] = [str(p) for p in data_path]
            else:
                section_meta["data_path"] = str(data_path)
        if plot_rels:
            section_meta["plot_path"] = plot_rels[0] if len(plot_rels) == 1 else plot_rels
        if experiment is not None:
            qubit_i = None
            cfg_expt = getattr(getattr(experiment, "cfg", None), "expt", None)
            if cfg_expt is not None and hasattr(cfg_expt, "get"):
                qubit_i = cfg_expt.get("qubit_i")
            if qubit_i is None:
                qubit_i = getattr(experiment, "qubit_i", None)
            if qubit_i is not None:
                section_meta["qubit_i"] = self._to_plain(qubit_i)
        if parameters is not None:
            section_meta["parameters"] = parameters
        if tags:
            section_meta["extra_tags"] = list(tags)

        # Build the entry as a single string. If yaml.safe_dump throws here
        # (e.g. unrepresentable type), nothing has been written yet — the file
        # stays consistent.
        # default_flow_style=None: compact inline form for short collections
        # (`qubits: [0]` instead of `qubits:\n- 0`). Required because Obsidian's
        # callout parser breaks code-fence rendering when it sees `- ` markers
        # at the start of inner lines.
        try:
            section_yaml = yaml.safe_dump(
                section_meta, default_flow_style=None, sort_keys=False, allow_unicode=True
            )
        except Exception as exc:
            print(f"[log_measurement] Could not serialize parameters: {exc}")
            section_yaml = (
                f"timestamp: '{timestamp_iso}'\n"
                f"# parameters omitted: {type(exc).__name__}: {exc}\n"
            )

        # Obsidian callout, default-collapsed via the trailing `-`. Each yaml
        # line is prefixed with `> ` so the content stays inside the callout
        # and Obsidian still parses the inner code fence as a yaml block.
        # (HTML <details> doesn't reliably re-enter markdown for inner fences.)
        yaml_lines = section_yaml.rstrip("\n").split("\n")
        quoted_yaml = "\n".join(("> " + line) if line else ">" for line in yaml_lines)
        entry_parts = [
            f"## {time_str} \u2014 {title}\n\n",
            "> [!note]- parameters\n",
            "> ```yaml\n",
            quoted_yaml + "\n",
            "> ```\n\n",
        ]
        for rel in plot_rels:
            entry_parts.append(f"![[{rel}]]\n\n")
        entry_parts.append("### Notes\n\n")
        # Avoid `<...>` which Obsidian's HTML-block parser may treat as an
        # unclosed tag and let it consume subsequent content (breaking the
        # next entry's rendering).
        entry_parts.append((notes.rstrip() + "\n\n") if notes else "_add interpretation here_\n\n")
        entry_parts.append("---\n\n")
        entry_text = "".join(entry_parts)

        # Top-of-file: write frontmatter + H1 only on first creation. Same
        # buffer-then-write pattern, so a yaml failure can't leave a partial.
        daily_file = target_dir / f"{today}.md"
        if not daily_file.exists():
            top_tags = ["lab", "measurement", project, user]
            if tags:
                top_tags.extend(t for t in tags if t not in top_tags)
            top_meta = {
                "date": today,
                "user": user,
                "project": project,
                "tags": top_tags,
            }
            try:
                top_yaml = yaml.safe_dump(
                    top_meta, default_flow_style=False, sort_keys=False, allow_unicode=True
                )
            except Exception as exc:
                print(f"[log_measurement] Could not serialize top frontmatter: {exc}")
                top_yaml = f"date: '{today}'\nuser: {user}\nproject: {project}\n"
            top_text = (
                "---\n"
                + top_yaml
                + "---\n\n"
                + f"# {today} \u2014 {user} / {project}\n\n"
            )
            daily_file.write_text(top_text, encoding="utf-8")

        # Atomic single append of the fully-built entry
        with daily_file.open("a", encoding="utf-8") as f:
            f.write(entry_text)

        print(f"[log_measurement] Appended section to {daily_file}")
        return daily_file

    def save_plot(
        self, fig, filename: str = "plot.png", subdir: Optional[str | Path] = None
    ):
        """
        Save a matplotlib figure with timestamp and markdown logging.

        Args:
            fig: matplotlib.figure.Figure object
            filename: Base name for the file (timestamp will be prepended)
            subdir: Optional subdirectory within plot_path, or Path to override

        Returns:
            Path to saved file
        """
        save_path = self.plot_path
        if isinstance(subdir, str):
            save_path = save_path / subdir
            save_path.mkdir(parents=True, exist_ok=True)
        elif isinstance(subdir, Path):
            save_path = subdir

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        today_str = now.strftime("%Y-%m-%d")

        # Add timestamp to figure title
        if fig._suptitle is not None:
            fig._suptitle.set_text(
                f"{fig._suptitle.get_text()} | {timestamp} - {filename}"
            )
        else:
            fig.suptitle(f"{timestamp} - {filename}", fontsize=16)

        fig.tight_layout()

        # Save figure
        timestamped_filename = f"{timestamp}_{filename}"
        filepath = save_path / timestamped_filename
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

        # Markdown logging
        markdown_path = self.log_path / f"{today_str}.md"
        if not markdown_path.exists():
            with markdown_path.open("w") as f:
                f.write(f"# Plots for {today_str}\n\n")

        rel_path = os.path.relpath(filepath, markdown_path.parent)
        with markdown_path.open("a") as md_file:
            md_file.write(f"![{filename}]({rel_path})\n")
        print(f"Plot reference appended to {markdown_path}")

        return filepath

    def convert_attrdict_to_dict(self, attrdict):
        """Recursively convert AttrDict to standard dict."""
        if isinstance(attrdict, (AttrDict, dict)):
            return {
                key: self.convert_attrdict_to_dict(value)
                for key, value in attrdict.items()
            }
        elif isinstance(attrdict, np.float64):
            return float(attrdict)
        else:
            return attrdict

    def recursive_compare(self, d1, d2, path="", exclude_keys=None):
        """Recursively compare two dictionaries and print differences."""
        if exclude_keys is None:
            exclude_keys = {'_ds_storage', '_ds_floquet', 'storage_man_file', 'floquet_man_stor_file'}
        for key in d1.keys():
            if key in exclude_keys:
                continue
            current_path = f"{path}.{key}" if path else key
            if key not in d2:
                print(f"Key '{current_path}' is missing in config2.")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                self.recursive_compare(d1[key], d2[key], current_path, exclude_keys)
            elif d1[key] != d2[key]:
                print(f"Key '{current_path}' differs:")
                print(f"  Old value (config1): {d1[key]}")
                print(f"  New value (config2): {d2[key]}")
        for key in d2.keys():
            if key in exclude_keys:
                continue
            current_path = f"{path}.{key}" if path else key
            if key not in d1:
                print(f"Key '{current_path}' is missing in config1.")

    def _sanitize_config_fields(self):
        """Clean up config before saving (in-place, only for non-essential fields)."""
        if "expt" in self.hardware_cfg:
            self.hardware_cfg.pop("expt")

    def _get_sanitized_config_copy(self, config):
        """Return a deep copy of config with dataset objects and vestigial fields removed for YAML serialization."""
        sanitized = copy.deepcopy(config)
        if hasattr(sanitized, 'device') and hasattr(sanitized.device, 'storage'):
            # Remove runtime dataset objects (not serializable)
            if '_ds_storage' in sanitized.device.storage:
                sanitized.device.storage.pop('_ds_storage')
            if '_ds_floquet' in sanitized.device.storage:
                sanitized.device.storage.pop('_ds_floquet')
            # Remove vestigial file path fields (Station handles loading via versioning system)
            if 'storage_man_file' in sanitized.device.storage:
                sanitized.device.storage.pop('storage_man_file')
            if 'floquet_man_stor_file' in sanitized.device.storage:
                sanitized.device.storage.pop('floquet_man_stor_file')
        return sanitized

    # Default fields that are shared across all operating points
    DEFAULT_SHARED_FIELDS = [
        'device.readout',
        'hw.soc',
        'aliases',
        'data_management',
    ]

    @staticmethod
    def _get_nested(d, dot_path):
        """Get a value from a nested dict by dot-separated path. Returns (value, found)."""
        keys = dot_path.split('.')
        current = d
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None, False
        return current, True

    @staticmethod
    def _set_nested(d, dot_path, value):
        """Set a value in a nested dict by dot-separated path, creating intermediate dicts if needed."""
        keys = dot_path.split('.')
        current = d
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def import_shared_params(
        self,
        from_config: Optional[str] = None,
        from_version: Optional[str] = None,
        fields: Optional[List[str]] = None,
        dry_run: bool = False,
    ):
        """
        Import shared parameters from another config into the current station.

        Pulls specified fields from a source config and overwrites them in the
        current hardware_cfg. Useful for syncing shared calibration values
        (e.g., readout parameters) from another user's config without disturbing
        operating-point-dependent fields (e.g., manipulate frequencies).

        Args:
            from_config: Source config filename (e.g., "hardware_config.yml") or None.
            from_version: Source config version ID (e.g., "CFG-HW-20260414-00032") or None.
                If both from_config and from_version are None, pulls from the current main version.
            fields: List of dot-separated field paths to import.
                Defaults to DEFAULT_SHARED_FIELDS if None.
            dry_run: If True, only print what would change without applying.

        Returns:
            Dict of changed fields: {field_path: (old_value, new_value)}

        Example:
            # Import readout params from main config
            station.import_shared_params()

            # Import from a specific version
            station.import_shared_params(from_version="CFG-HW-20260414-00032")

            # Import specific fields
            station.import_shared_params(fields=['device.readout', 'device.active_reset'])

            # Preview changes without applying
            station.import_shared_params(dry_run=True)
        """
        if fields is None:
            fields = self.DEFAULT_SHARED_FIELDS

        # Resolve source config
        config_spec = from_version or from_config
        db = get_database()
        config_manager = ConfigVersionManager(self.config_dir)

        with db.session() as session:
            source_path = self._resolve_config_path(
                config_spec, ConfigType.HARDWARE_CONFIG, config_manager, session, required=True
            )

        with source_path.open("r") as f:
            source_cfg = yaml.safe_load(f)

        print(f"[IMPORT] Source: {source_path.name}")

        # Import each field
        changes = {}
        for field_path in fields:
            new_value, found = self._get_nested(source_cfg, field_path)
            if not found:
                print(f"[IMPORT]   {field_path}: not found in source, skipping")
                continue

            old_value, had_old = self._get_nested(self.hardware_cfg, field_path)

            if had_old and old_value == new_value:
                print(f"[IMPORT]   {field_path}: unchanged")
                continue

            changes[field_path] = (old_value if had_old else '<missing>', new_value)

            if dry_run:
                print(f"[IMPORT]   {field_path}: would update")
            else:
                self._set_nested(self.hardware_cfg, field_path, new_value)
                print(f"[IMPORT]   {field_path}: updated")

            # Print sub-field diffs for dict values
            if isinstance(new_value, dict) and isinstance(old_value, dict):
                self.recursive_compare(old_value, new_value, path=field_path)
            elif old_value != new_value:
                print(f"            old: {old_value}")
                print(f"            new: {new_value}")

        if not changes:
            print("[IMPORT] All fields already up to date.")
        elif dry_run:
            print(f"[IMPORT] Dry run: {len(changes)} field(s) would be updated. "
                  "Re-run without dry_run=True to apply.")
        else:
            print(f"[IMPORT] {len(changes)} field(s) updated.")

        return changes

    def preview_config_update(self):
        """Compare parent and current config to view updates."""
        print("Comparing configurations:")
        print("Parent config file:", self.hardware_config_file)
        with self.hardware_config_file.open("r") as cfg_file:
            old_cfg = AttrDict(yaml.safe_load(cfg_file))
        self.recursive_compare(old_cfg, self.hardware_cfg)
        self._sanitize_config_fields()

    def update_all_station_snapshots(self, update_main: bool = False) -> dict:
        """
        Create config snapshots from current station state.

        Args:
            update_main: If True, set the new snapshots as main versions

        Returns:
            Dict mapping config type name to version ID
        """
        versions = {}

        versions["hardware_config"] = self.snapshot_hardware_config(update_main=update_main)
        versions["multiphoton_config"] = self.snapshot_multiphoton_config(update_main=update_main)
        versions["man1_storage_swap"] = self.snapshot_man1_storage_swap(update_main=update_main)
        if self.ds_floquet is not None:
            versions["floquet_storage_swap"] = self.snapshot_floquet_storage_swap(update_main=update_main)

        print("Config snapshots for current station:")
        for config_type, version_id in versions.items():
            print(f"  {config_type}: {version_id}")

        if update_main:
            print("Configs saved and set as main!")

        return versions

    def snapshot_hardware_config(self, update_main: bool = False) -> str:
        """
        Create a snapshot of the current hardware config and optionally set as main.

        Args:
            update_main: If True, set the new snapshot as the main version

        Returns:
            The version ID of the created snapshot
        """
        db = get_database()
        config_manager = ConfigVersionManager(self.config_dir)
        self._sanitize_config_fields()
        sanitized_cfg = self._get_sanitized_config_copy(self.hardware_cfg)

        with db.session() as session:
            version_id, _ = config_manager._snapshot_dict_as_yaml(
                config_dict=sanitized_cfg,
                config_type=ConfigType.HARDWARE_CONFIG,
                original_filename=self.hardware_config_file.name,
                session=session,
            )

            if update_main:
                config_manager.set_main_version(
                    config_type=ConfigType.HARDWARE_CONFIG,
                    version_id=version_id,
                    session=session,
                    updated_by=self.user,
                )

        return version_id

    def snapshot_multiphoton_config(self, update_main: bool = False) -> str:
        """
        Create a snapshot of the current multiphoton config and optionally set as main.

        Args:
            update_main: If True, set the new snapshot as the main version

        Returns:
            The version ID of the created snapshot
        """
        db = get_database()
        config_manager = ConfigVersionManager(self.config_dir)

        with db.session() as session:
            version_id, _ = config_manager._snapshot_dict_as_yaml(
                config_dict=self.multimode_cfg,
                config_type=ConfigType.MULTIPHOTON_CONFIG,
                original_filename=self.multiphoton_config_file.name,
                session=session,
            )

            if update_main:
                config_manager.set_main_version(
                    config_type=ConfigType.MULTIPHOTON_CONFIG,
                    version_id=version_id,
                    session=session,
                    updated_by=self.user,
                )

        return version_id

    def snapshot_man1_storage_swap(self, update_main: bool = False) -> str:
        """
        Create a snapshot of the current man1 storage swap CSV (ds_storage)
        and optionally set as main.

        Args:
            update_main: If True, set the new snapshot as the main version

        Returns:
            The version ID of the created snapshot
        """
        db = get_database()
        config_manager = ConfigVersionManager(self.config_dir)

        with db.session() as session:
            version_id, _ = config_manager._snapshot_csv_from_dataframe(
                df=self.ds_storage.df,
                config_type=ConfigType.MAN1_STORAGE_SWAP,
                original_filename=self.storage_man_file,
                session=session,
            )

            if update_main:
                config_manager.set_main_version(
                    config_type=ConfigType.MAN1_STORAGE_SWAP,
                    version_id=version_id,
                    session=session,
                    updated_by=self.user,
                )

        return version_id

    def snapshot_floquet_storage_swap(self, update_main: bool = False) -> str:
        """
        Create a snapshot of the current floquet storage swap CSV and optionally set as main.

        Args:
            update_main: If True, set the new snapshot as the main version

        Returns:
            The version ID of the created snapshot

        Raises:
            ValueError: If no floquet dataset is loaded
        """
        if self.ds_floquet is None:
            raise ValueError("No floquet dataset loaded in station")

        db = get_database()
        config_manager = ConfigVersionManager(self.config_dir)

        with db.session() as session:
            version_id, _ = config_manager._snapshot_csv_from_dataframe(
                df=self.ds_floquet.df,
                config_type=ConfigType.FLOQUET_STORAGE_SWAP,
                original_filename=self.floquet_file,
                session=session,
            )

            if update_main:
                config_manager.set_main_version(
                    config_type=ConfigType.FLOQUET_STORAGE_SWAP,
                    version_id=version_id,
                    session=session,
                    updated_by=self.user,
                )

        return version_id

    def handle_multiphoton_config_update(self, updateConfig_bool=False):
        """Handle multiphoton config updates (not yet implemented)."""
        raise NotImplementedError("This is not properly coded yet")
        # print("Comparing configurations:")
        # self.recursive_compare(self.hardware_cfg, self.hardware_config)
        # autocalib_path = self.create_autocalib_path()
        # config_path = self.config_file
        # updated_config = self.update_yaml_config(self.hardware_cfg, self.hardware_cfg)
        # if updateConfig_bool:
        #     self.save_configurations(
        #         self.hardware_cfg, updated_config, autocalib_path, config_path
        #     )
        #     self.hardware_cfg = updated_config
        #     print(
        #         "Configuration updated and saved, excluding storage_man_file. \n!!!!Please set updateConfig to False after this run!!!!!!."
        #     )

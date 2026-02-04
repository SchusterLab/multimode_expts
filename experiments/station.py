"""
MultimodeStation: Central hardware and configuration management.

This module provides the MultimodeStation class which manages:
- Hardware connections (QICK RFSoC, InstrumentManager, Yokogawa sources)
- Configuration files (hardware config, multiphoton config)
- Data paths and output directories
- Storage-manipulate swap dataset

Supports both real hardware and mock modes:
- Real hardware: Connects to actual instruments (production on BF5)
- Mock mode: Uses simulated hardware for testing/development

Usage:
    from experiments.station import MultimodeStation

    # Auto-detect mode based on machine (mock on dev, real on BF5)
    station = MultimodeStation(experiment_name="241215_calibration")

    # Force mock mode for testing
    station = MultimodeStation(mock=True)

    # Force real hardware on dev machine
    station = MultimodeStation(mock=False)

    # Access hardware: station.soc, station.im
    # Access config: station.hardware_cfg, station.hardware_cfg
    # Access paths: station.data_path, station.plot_path
    # Check mode: station.is_mock
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import socket

import numpy as np
import yaml

from experiments.dataset import FloquetStorageSwapDataset, StorageManSwapDataset
from slab import AttrDict, get_current_filename
from slab.datamanagement import SlabFile

from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager, ConfigType

BF5_HOSTNAME = 'DESKTOP-GONKTN3'

def detect_mock_mode() -> bool:
    """
    Auto-detect mock mode based on machine identity.

    Returns:
        True for mock mode (dev machine), False for real hardware (BF5 production)

    Detection logic:
        - 'DESKTOP-GONKTN3' is the production host name on BF5
        - If host name matches, use real hardware
        - Otherwise, default to mock mode for safety
    """
    hostname = socket.gethostname()
    return hostname != BF5_HOSTNAME

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
            mock: If None, auto-detect based on machine identity. If True, force mock mode. If False, force real hardware.
        """
        self.repo_root = Path(__file__).resolve().parent.parent
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_experiment'
        )
        self.qubit_i = qubit_i
        self.user = user

        # Determine mock mode
        if mock is None:
            self._is_mock = detect_mock_mode()
        else:
            self._is_mock = mock

        # Config loading always uses database/versioning (same for mock and real)
        self._initialize_configs(hardware_config, multiphoton_config, storage_man_file, floquet_file)

        # Output paths and hardware - routing handled internally based on mock mode
        self._initialize_output_paths()
        self._initialize_hardware()

        self.print()

    @property
    def is_mock(self) -> bool:
        """Whether the station is running in mock mode."""
        return self._is_mock

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
        from qick import QickConfig
        from slab.instruments import InstrumentManager
        from slab.instruments.voltsource import YokogawaGS200

        self.im = InstrumentManager(ns_address="192.168.137.25")
        self.soc = QickConfig(self.im[self.hardware_cfg["aliases"]["soc"]].get_cfg())
        self.yoko_coupler = YokogawaGS200(name='yoko_coupler', address='192.168.137.148')
        self.yoko_jpa = YokogawaGS200(name='yoko_jpa', address='192.168.137.149')

    def _initialize_output_paths_mock(self):
        """Create output directories for mock mode."""
        self.output_root = self.repo_root / "mock_data"

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
        """Initialize mock hardware objects."""
        from experiments.mock_hardware import (
            MockInstrumentManager,
            MockQickConfig,
            MockYokogawa,
        )

        self.im = MockInstrumentManager()
        self.soc = MockQickConfig()
        self.yoko_coupler = MockYokogawa(
            name="yoko_coupler", address="mock://192.168.137.148"
        )
        self.yoko_jpa = MockYokogawa(
            name="yoko_jpa", address="mock://192.168.137.149"
        )
        print("[MOCK STATION] Mock hardware initialized")

    def print(self):
        """Print station information."""
        if self._is_mock:
            print(f"[MOCK STATION] Data path: {self.data_path}")
            print(f"[MOCK STATION] Config file: {self.hardware_config_file}")
            print(f"[MOCK STATION] Instruments: {list(self.im.keys())}")
            print(f"[MOCK STATION] SOC: {self.soc}")
        else:
            print("Data, plots, logs will be stored in:", self.experiment_path)
            print("Hardware configs will be read from", self.hardware_config_file)
            print(self.im.keys())
            print(self.soc)

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
        import copy
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

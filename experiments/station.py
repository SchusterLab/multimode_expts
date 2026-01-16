"""
MultimodeStation: Central hardware and configuration management.

This module provides the MultimodeStation class which manages:
- Hardware connections (QICK RFSoC, InstrumentManager, Yokogawa sources)
- Configuration files (hardware config, multiphoton config)
- Data paths and output directories
- Storage-manipulate swap dataset

Usage:
    from experiments.station import MultimodeStation

    station = MultimodeStation(experiment_name="241215_calibration")
    # Access hardware: station.soc, station.im
    # Access config: station.config_thisrun, station.yaml_cfg
    # Access paths: station.data_path, station.plot_path
"""

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from qick import QickConfig

from experiments.dataset import FloquetStorageSwapDataset, StorageManSwapDataset
from slab import AttrDict, get_current_filename
from slab.datamanagement import SlabFile
from slab.instruments import InstrumentManager
from slab.instruments.voltsource import YokogawaGS200

from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager, ConfigType

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

    Attributes:
        soc: QickConfig object for FPGA control
        im: InstrumentManager for hardware access
        config_thisrun: AttrDict of current run configuration
        yaml_cfg: AttrDict of original yaml configuration
        ds_storage: StorageManSwapDataset for this run
        data_path: Path to data directory
        plot_path: Path to plots directory
        log_path: Path to logs directory
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        hardware_config: Optional[str] = None,
        multiphoton_config: Optional[str] = None,
        storage_man_file: Optional[str] = None,
        floquet_file: Optional[str] = None,
        qubit_i: int = 0,
    ):
        """
        Initialize the measurement station.

        Args:
            experiment_name: Format is yymmdd_name. None defaults to today's date.
            hardware_config: Filename or version ID (e.g., CFG-HW-20260115-00001). If None, loads from main version in database.
            multiphoton_config: Filename or version ID (e.g., CFG-MP-20260115-00001). If None, loads from main version in database.
            storage_man_file: Filename or version ID (e.g., CFG-M1-20260115-00001). If None, loads from main version in database.
            qubit_i: Qubit index to use.
        """
        self.repo_root = Path(__file__).resolve().parent.parent
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_experiment'
        )
        self.qubit_i = qubit_i

        self._initialize_configs(hardware_config, multiphoton_config, storage_man_file, floquet_file)
        self._initialize_output_paths()
        self._initialize_hardware()

        self.print()

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
                self.yaml_cfg = AttrDict(yaml.safe_load(cfg_file))
            self.hardware_config_file = hw_config_path

            # Config for this instance (deepcopy of yaml_cfg)
            self.config_thisrun = AttrDict(deepcopy(self.yaml_cfg))

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

    def _resolve_config_path(
        self,
        config_spec: Optional[str],
        config_type: ConfigType,
        config_manager: ConfigVersionManager,
        session,
        required: bool = False
    ) -> Path:
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
        """Create output directories if needed."""
        self.output_root = Path(self.yaml_cfg.data_management.output_root)
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
        """Connect to hardware."""
        self.im = InstrumentManager(ns_address="192.168.137.25")
        self.soc = QickConfig(self.im[self.yaml_cfg["aliases"]["soc"]].get_cfg())
        self.yoko_coupler = YokogawaGS200(name='yoko_coupler', address='192.168.137.148')
        self.yoko_jpa = YokogawaGS200(name='yoko_jpa', address='192.168.137.149')

    def print(self):
        """Print station information."""
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

    def recursive_compare(self, d1, d2, path=""):
        """Recursively compare two dictionaries and print differences."""
        for key in d1.keys():
            current_path = f"{path}.{key}" if path else key
            if key not in d2:
                print(f"Key '{current_path}' is missing in config2.")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                self.recursive_compare(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                print(f"Key '{current_path}' differs:")
                print(f"  Old value (config1): {d1[key]}")
                print(f"  New value (config2): {d2[key]}")
        for key in d2.keys():
            current_path = f"{path}.{key}" if path else key
            if key not in d1:
                print(f"Key '{current_path}' is missing in config1.")

    def _sanitize_config_fields(self, config_thisrun) -> AttrDict:
        """Clean up config before saving."""
        updated_config = deepcopy(config_thisrun)
        updated_config.device.storage.storage_man_file = (
            self.yaml_cfg.device.storage.storage_man_file
        )
        updated_config.pop("expt", None)
        return updated_config

    def save_config(self):
        """Save current configuration to file."""
        yaml_dump_kwargs = dict(
            default_flow_style=False,
            indent=4,
            width=80,
            canonical=False,
            explicit_start=True,
            explicit_end=False,
            sort_keys=False,
            line_break=True,
        )

        # Save backup of old config
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        old_config_path = self.autocalib_path / f"old_config_{current_time}.yaml"
        old_config = self.convert_attrdict_to_dict(self.yaml_cfg)
        with old_config_path.open("w") as cfg_file:
            yaml.dump(old_config, cfg_file, **yaml_dump_kwargs)

        # Save updated config
        updated_config = self.convert_attrdict_to_dict(
            self._sanitize_config_fields(self.config_thisrun)
        )
        with self.hardware_config_file.open("w") as f:
            yaml.dump(updated_config, f, **yaml_dump_kwargs)

    def handle_config_update(self, write_to_file=False):
        """Compare and optionally save configuration updates."""
        print("Comparing configurations:")
        self.recursive_compare(self.yaml_cfg, self.config_thisrun)
        updated_config = self._sanitize_config_fields(self.config_thisrun)
        if write_to_file:
            self.save_config()
            self.yaml_cfg = updated_config
            print("Configuration updated and saved.")

    def update_all_station_snapshots(self, update_main=False, updated_by=None):
        """
        Create config snapshots from current station state.

        Args:
            update_main: If True, set the new snapshots as main versions
            updated_by: Username for tracking who set the main version
        """
        db = get_database()
        config_dir = self.config_dir
        config_manager = ConfigVersionManager(config_dir)

        with db.session() as session:
            # Create snapshots
            versions = config_manager.snapshot_station_configs(
                station=self,
                session=session,
            )

            print("Config snapshots for current station:")
            for config_type, version_id in versions.items():
                print(f"  {config_type}: {version_id}")

            # Set as main if requested (within the same session)
            if update_main:
                for config_type_str, version_id in versions.items():
                    config_type = ConfigType[config_type_str.upper()]
                    config_manager.set_main_version(
                        config_type=config_type,
                        version_id=version_id,
                        session=session,
                        updated_by=updated_by
                    )
                print("Configs saved and set as main!")

            # Commit happens automatically when exiting the context manager

        return versions

    def handle_multiphoton_config_update(self, updateConfig_bool=False):
        """Handle multiphoton config updates (not yet implemented)."""
        raise NotImplementedError("This is not properly coded yet")
        # print("Comparing configurations:")
        # self.recursive_compare(self.yaml_cfg, self.config_thisrun)
        # autocalib_path = self.create_autocalib_path()
        # config_path = self.config_file
        # updated_config = self.update_yaml_config(self.yaml_cfg, self.config_thisrun)
        # if updateConfig_bool:
        #     self.save_configurations(
        #         self.yaml_cfg, updated_config, autocalib_path, config_path
        #     )
        #     self.yaml_cfg = updated_config
        #     print(
        #         "Configuration updated and saved, excluding storage_man_file. \n!!!!Please set updateConfig to False after this run!!!!!!."
        #     )
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

from experiments.dataset import StorageManSwapDataset
from slab import AttrDict, get_current_filename
from slab.datamanagement import SlabFile
from slab.instruments import InstrumentManager
from slab.instruments.voltsource import YokogawaGS200


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
        ds_thisrun: StorageManSwapDataset for this run
        data_path: Path to data directory
        plot_path: Path to plots directory
        log_path: Path to logs directory
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        hardware_config: str = "hardware_config_202505.yml",
        storage_man_file: str = "man1_storage_swap_dataset.csv",
        qubit_i: int = 0,
    ):
        """
        Initialize the measurement station.

        Args:
            experiment_name: Format is yymmdd_name. None defaults to today's date.
            hardware_config: Filename for the yaml config (in config_dir).
            storage_man_file: Filename for storage-manipulate swap csv (in config_dir).
            qubit_i: Qubit index to use.
        """
        self.repo_root = Path(__file__).resolve().parent.parent
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_experiment'
        )
        self.qubit_i = qubit_i

        self._initialize_configs(hardware_config)
        self._initialize_output_paths()
        self._initialize_hardware()

        self.print()

    def _initialize_configs(self, hardware_config):
        """Load configuration files."""
        self.config_dir = self.repo_root / "configs"

        # Load hardware config
        self.hardware_config_file = self.config_dir / hardware_config
        with self.hardware_config_file.open("r") as cfg_file:
            self.yaml_cfg = AttrDict(yaml.safe_load(cfg_file))

        # Config for this instance (deepcopy of yaml_cfg)
        self.config_thisrun = AttrDict(deepcopy(self.yaml_cfg))

        # Load multiphoton config
        self.multiphoton_config_file = (
            self.config_dir / self.config_thisrun.device.multiphoton_config.file
        )
        with self.multiphoton_config_file.open("r") as f:
            self.multimode_cfg = AttrDict(yaml.safe_load(f))

        # Initialize the dataset
        self.storage_man_file = self.yaml_cfg.device.storage.storage_man_file
        ds, ds_thisrun, ds_thisrun_file_path = self.load_storage_man_swap_dataset(
            self.storage_man_file
        )
        self.ds_thisrun = ds_thisrun

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
        ds = StorageManSwapDataset(filename, parent_path)
        ds_thisrun = StorageManSwapDataset(ds.create_copy(), parent_path)
        ds_thisrun_file_path = ds_thisrun.file_path
        return ds, ds_thisrun, ds_thisrun_file_path

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

    def handle_multiphoton_config_update(self, updateConfig_bool=False):
        """Handle multiphoton config updates (not yet implemented)."""
        raise NotImplementedError("This is not properly coded yet")

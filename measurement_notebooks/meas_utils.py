import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from typing import Protocol

import numpy as np
import yaml
from qick import QickConfig

from experiments.dataset import StorageManSwapDataset
from slab import AttrDict, get_current_filename, Experiment
from slab.datamanagement import SlabFile
from slab.instruments import InstrumentManager
from slab.instruments.voltsource import YokogawaGS200

# add hooks to yaml so that np.float64 etc are written out as plain floats
def np_float_representer(dumper, data):
    return dumper.represent_float(float(data))

def np_int_representer(dumper, data): 
    return dumper.represent_float(int(data))

yaml.add_representer(np.float64, np_float_representer)
yaml.add_representer(np.float32, np_float_representer)
yaml.add_representer(np.int64, np_int_representer)
yaml.add_representer(np.int64, np_int_representer)

# this prevents yaml from using anchors and aliases (&id001 and *id001)
yaml.Dumper.ignore_aliases = lambda *args : True

# TODO: add a dummy station class to allow for testing its dependents without hardware
# TODO: add a state variable that records whether active reset (of qb, man, stor) should be enabled 
# and use this as the default that can still be overridden by user arg

class MultimodeStation:
    """
    This represents a measurement setup that controls at least:
        an InstrumentManager,
        a QICK RFSoC,
        a hardware config yaml file,
        a manipulate-storage swap database file,
        a multiphoton config yaml file.
        a path to save data/plot/logs to,
        (without any of these the station cannot initialize)
    and optionally:
        Yokogawa sources for JPA and coupler flux,
    In the future we should consolidate hardware-dependent configs
    so that the code can on run on different fridges.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        hardware_config: str = "hardware_config_202505.yml",
        storage_man_file: str = "man1_storage_swap_dataset.csv",
        qubit_i: int = 0,
    ):
        """
        Args:
            - experiment_name: format is yymmdd_name. None defaults to today's date
            - hardware_config: filename for the yaml config. To be found under station.config_dir
            - storage_man_file: filename for the storage manipulate swap csv. Under station.config_dir
        """
        self.repo_root = Path(__file__).resolve().parent.parent
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_experiment'
        )
        self.qubit_i = qubit_i

        self._initialize_configs(hardware_config)

        self._initialize_output_paths()

        self._initialize_hardware()

        # For config update logic
        # self.updateConfig_bool = False

        self.print()

    def _initialize_configs(self, hardware_config):
        self.config_dir = self.repo_root / "configs"

        # load hardware config
        self.hardware_config_file = self.config_dir / hardware_config
        with self.hardware_config_file.open("r") as cfg_file:
            self.yaml_cfg = AttrDict(yaml.safe_load(cfg_file))

        # Config for this instance (deepcopy of yaml_cfg)
        self.config_thisrun = AttrDict(deepcopy(self.yaml_cfg))

        # load the multiphoton config
        self.multiphoton_config_file = (
            self.config_dir / self.config_thisrun.device.multiphoton_config.file
        )
        with self.multiphoton_config_file.open("r") as f:
            self.multimode_cfg = AttrDict(yaml.safe_load(f))

        # Initailize the dataset
        self.storage_man_file = self.yaml_cfg.device.storage.storage_man_file
        ds, ds_thisrun, ds_thisrun_file_path = self.load_storage_man_swap_dataset(
            self.storage_man_file
        )
        self.ds_thisrun = ds_thisrun

    def _initialize_output_paths(self):
        self.output_root = Path(
            self.yaml_cfg.data_management.output_root
        )  # where data, plots, logs are saved
        if not self.output_root.exists():
            raise FileNotFoundError(
                f"""Output root {self.output_root} does not exist.
                This is not something that should be automatically created.
                Double check if your file system matches what hardware config wants
                and modify the data_management field accordingly."""
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
        self.im = InstrumentManager(ns_address="192.168.137.25")
        self.soc = QickConfig(self.im[self.yaml_cfg["aliases"]["soc"]].get_cfg())
        # TODO: add yokos to im
        self.yoko_coupler = YokogawaGS200(name='yoko_coupler', address='192.168.137.148')
        self.yoko_jpa = YokogawaGS200(name='yoko_jpa', address='192.168.137.149')

    def print(self):
        print("Data, plots, logs will be stored in: ", self.experiment_path)
        print("Hardware configs will be read from", self.hardware_config_file)
        print(self.im.keys())
        print(self.soc)

    def load_data(self, filename: Optional[str] = None, prefix: Optional[str] = None):
        if prefix is not None:
            data_file = self.data_path / get_current_filename(
                self.data_path, prefix=prefix, suffix=".h5"
            )
        else:
            data_file = self.data_path / filename
        with SlabFile(data_file) as a:
            attrs = dict()
            for key in list(a.attrs):
                attrs.update({key: json.loads(a.attrs[key])})
            keys = list(a)
            data = dict()
            for key in keys:
                data.update({key: np.array(a[key])})
        return data, attrs, data_file

    def load_storage_man_swap_dataset(
        self, filename: str, parent_path: Optional[str | Path] = None
    ):
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
        Save a matplotlib figure to the station's plot directory with markdown logging.

        Parameters:
        - fig: matplotlib.figure.Figure object to save
        - filename: Base name for the file (timestamp will be prepended)
        - subdir: Optional subdirectory:
            either a str within plot_path (e.g., "autocalibration")
            or a Path in which case it OVERWRITES the save path

        Returns:
        - filepath: Path object of the saved file
        """
        # Determine save path
        save_path = self.plot_path
        if isinstance(subdir, str):
            save_path = save_path / subdir
            save_path.mkdir(parents=True, exist_ok=True)
        elif isinstance(subdir, Path):
            save_path = subdir

        # Generate timestamp
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

        # Use relative path from markdown file to plot
        rel_path = os.path.relpath(filepath, markdown_path.parent)
        md_line = f"![{filename}]({rel_path})\n"
        with markdown_path.open("a") as md_file:
            md_file.write(md_line)
        print(f"Plot reference appended to {markdown_path}")

        return filepath

    def convert_attrdict_to_dict(self, attrdict):
        """
        Recursively converts an AttrDict or a nested dictionary into a standard Python dictionary.
        Converts np.float64 values to standard Python float.
        """
        if isinstance(attrdict, AttrDict):
            return {
                key: self.convert_attrdict_to_dict(value)
                for key, value in attrdict.items()
            }
        elif isinstance(attrdict, dict):
            return {
                key: self.convert_attrdict_to_dict(value)
                for key, value in attrdict.items()
            }
        elif isinstance(attrdict, np.float64):
            return float(attrdict)
        else:
            return attrdict

    def recursive_compare(self, d1, d2, path=""):
        """
        Recursively compares two dictionaries and prints differences.
        """
        for key in d1.keys():
            current_path = f"{path}.{key}" if path else key
            if key not in d2:
                print(f"Key '{current_path}' is missing in config2.")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                self.recursive_compare(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                print(f"Key '{current_path}' differs:")
                # if isinstance(d1[key], list) and len(d1[key]) == 1:
                #     print(f"  Old value (config1): {d1[key][0]}")
                #     print(f"  New value (config2): {d2[key][0]}")
                # else:
                print(f"  Old value (config1): {d1[key]}")
                print(f"  New value (config2): {d2[key]}")
        for key in d2.keys():
            current_path = f"{path}.{key}" if path else key
            if key not in d1:
                print(f"Key '{current_path}' is missing in config1.")

    def _sanitize_config_fields(self, config_thisrun) -> AttrDict:
        """
        Clean up a couple entries in config_thisrun in preparation for saving:
            - storage_man_file is restored to the value in self.yaml_cfg (why??)
            - remove the 'expt' field that got added to config_thisrun
        Returns a fresh deep copy with these updates
        """
        updated_config = deepcopy(config_thisrun)
        updated_config.device.storage.storage_man_file = (
            self.yaml_cfg.device.storage.storage_man_file
        )
        updated_config.pop("expt", None)  # this shouldn't be written to hardware config
        return updated_config

    def save_config(self):
        """
        Save the old and updated configurations to their respective files.
        """
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

        # first save a copy of the old config to a backup location before overwriting
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        old_config_path = self.autocalib_path / f"old_config_{current_time}.yaml"
        old_config = self.convert_attrdict_to_dict(self.yaml_cfg)
        with old_config_path.open("w") as cfg_file:
            yaml.dump(old_config, cfg_file, **yaml_dump_kwargs)

        # next save the updated config_thisrun to the hardware config yaml, overwriting it
        updated_config = self.convert_attrdict_to_dict(
            self._sanitize_config_fields(self.config_thisrun)
        )
        with self.hardware_config_file.open("w") as f:
            yaml.dump(updated_config, f, **yaml_dump_kwargs)

    def handle_config_update(self, write_to_file=False):
        """
        Main logic for comparing, updating, and saving configuration files.
        Only does config this run
        """
        print("Comparing configurations:")
        self.recursive_compare(self.yaml_cfg, self.config_thisrun)
        updated_config = self._sanitize_config_fields(self.config_thisrun)
        if write_to_file:
            self.save_config()
            self.yaml_cfg = updated_config
            print("Configuration updated and saved, excluding storage_man_file.")

    def handle_multiphoton_config_update(self, updateConfig_bool=False):
        """
        Main logic for comparing, updating, and saving configuration files.
        Only does config this run
        """
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


class PreProcessor(Protocol):
    def __call__(
        self, station: MultimodeStation, default_expt_cfg: AttrDict, **kwargs
    ) -> AttrDict:
        """Must take a default template + user defined kwargs and
        return a deepcopied expt config to be direclty passed to the Experiment."""
        ...


class PostProcessor(Protocol):
    def __call__(self, station: MultimodeStation, expt: Experiment) -> None:
        """Must extract results and mutate station.config_thisrun (returns None)."""
        ...


def default_preprocessor(station, default_expt_cfg, **kwargs):
    """
    If your preprocessor just needs to update default_expt_cfg with user kwargs,
        you don't have to write one at all. Just leave preprocessor=None in 
        CharacterizationRunner.__init__ and we'll use this automatically.
    In general your own override should insert logic between the first two lines here.
    """
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    return expt_cfg


def default_postprocessor(station, expt):
    """
    Extract results from the expt class and update station.config_thisrun here.
    This should not write the config file to disk (add that functionality to 
        the run method if needed).
    """
    return


class CharacterizationRunner:
    def __init__(
        self,
        station: MultimodeStation,
        ExptClass: type[Experiment],
        default_expt_cfg: AttrDict,
        preprocessor: Optional[PreProcessor] = None,
        postprocessor: Optional[PostProcessor] = None,
    ):
        """
        Manages the execution of one characterization experiment.
        Typically we need to create a default expt.cfg.expt template,
            modify it with user kwargs, run the measurement and 
            extract results to update station config.
        This class encapsulates that boilerplate.
        See meas_utils.default_preprocessor and default_postprocessor
            for how to write custom pre/post processors (they need to 
            strictly adhere to the Protocols including arg names etc).

        Args:
            station: MultimodeStation instance
            ExptClass: a child class of slab.Experiment
            default_expt_cfg: AttrDict template for expt.cfg.expt
            preprocessor: function that generates expt.cfg.expt from
                default_expt_cfg and user kwargs
            postprocessor: function that updates station.config_thisrun
        """
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.station = station
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor or default_postprocessor

    # def _set_nested_attr(self, obj, key_path, value):
    #     """
    #     Set a nested attribute using dot notation.
    #
    #     Args:
    #         obj: The object to modify (typically an AttrDict)
    #         key_path: Dot-separated path like 'readout.relax_delay'
    #         value: The value to set
    #
    #     Example:
    #         _set_nested_attr(cfg.device, 'readout.relax_delay', [50])
    #         # Sets cfg.device.readout.relax_delay = [50]
    #     """
    #     keys = key_path.split('.')
    #     for key in keys[:-1]:
    #         obj = getattr(obj, key)
    #     setattr(obj, keys[-1], value)

    def run(
        self, postprocess: bool = True, go_kwargs: dict = {}, **kwargs
    ) -> Experiment:
        """
        Standard measurement execution boilerplate.
        Use preprocessor and default config template to generate cfg.expt,
        create a new Experiment object, run its go() method, and optionally
        run postprocessor to update station config.

        Args:
            postprocess: enable whether to run self.postprocessor after expt.go()
            go_kwargs: dict that will be passed to expt.go()
                Defaults: analyze=True, display=True, progress=True, save=True
            kwargs: key-value pairs passed to preprocessor to update expt config

        Returns:
            Experiment object
        """
        # Create experiment instance
        expt = self.ExptClass(
            soccfg=self.station.soc,
            path=self.station.data_path,
            prefix=self.ExptClass.__name__,
            config_file=self.station.hardware_config_file,
        )

        # Deepcopy config for this measurement
        expt.cfg = AttrDict(deepcopy(self.station.config_thisrun))

        # Apply experiment-specific config
        expt.cfg.expt = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)
        expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

        # Execute with sensible defaults
        go_defaults = {"analyze": True, "display": True, "progress": True, "save": True}
        go_defaults.update(go_kwargs)
        expt.go(**go_defaults)

        # Update config
        if postprocess:
            self.postprocessor(self.station, expt)

        return expt


class SweepRunner:
    """
    Manages execution of sweep experiments (2D measurements).

    Unlike CharacterizationRunner which runs a single experiment,
    SweepRunner loops over a parameter, running the experiment at
    each point and saving data incrementally to a file.

    Key features:
    - Incremental file saving (data saved after each sweep point)
    - Optional live analysis callback for real-time visualization
    - Flexible analysis class initialization via factory function
    - Consistent pattern with CharacterizationRunner (preprocessor/postprocessor)

    Typical use case: Chevron (frequency vs time) sweeps

    Example:
        chevron_runner = SweepRunner(
            station=station,
            ExptClass=meas.LengthRabiGeneralF0g1Experiment,
            default_expt_cfg=config,
            sweep_param='freq',
            preprocessor=preproc_func,
            postprocessor=postproc_func,
            # Analysis can be class or factory function
            analysis_factory=lambda data, station: ChevronFitting(
                frequencies=data['freq_sweep'],
                time=data['xpts'],
                response_matrix=data['avgi'],
                config=station.config_thisrun,
                station=station,
            ),
            # Optional: live plotting callback
            live_analysis_fn=lambda data, station: update_live_plot(data),
        )
        analysis = chevron_runner.run(
            sweep_start=2000,
            sweep_stop=2010,
            sweep_step=0.5,
        )
    """

    def __init__(
        self,
        station: MultimodeStation,
        ExptClass: type[Experiment],
        default_expt_cfg: AttrDict,
        sweep_param: str = 'freq',
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
        analysis_factory: Optional[Callable] = None,
        live_analysis_fn: Optional[Callable] = None,
    ):
        """
        Args:
            station: MultimodeStation instance
            ExptClass: Experiment class to run at each sweep point
            default_expt_cfg: Default experiment config template
            sweep_param: Parameter to sweep (e.g., 'freq', 'gain')
            preprocessor: Optional function(station, default_cfg, **kwargs) -> expt_cfg
            postprocessor: Optional function(station, analysis) called after sweep
            analysis_factory: Optional function(sweep_data, station) -> analysis object.
                If None, returns raw sweep_data dict instead of analysis object.
            live_analysis_fn: Optional function(sweep_data, station) called after each
                sweep point for live visualization. Receives partial data during sweep.
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.sweep_param = sweep_param
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor
        self.analysis_factory = analysis_factory
        self.live_analysis_fn = live_analysis_fn

    def run(
        self,
        sweep_start: float,
        sweep_stop: float,
        sweep_step: float,
        postprocess: bool = True,
        go_kwargs: dict = None,
        incremental_save: bool = True,
        **kwargs
    ):
        """
        Run sweep experiment.

        Args:
            sweep_start: Starting value for sweep parameter
            sweep_stop: Ending value for sweep parameter
            sweep_step: Step size for sweep parameter
            postprocess: Whether to run postprocessor after sweep
            go_kwargs: Kwargs passed to expt.go()
            incremental_save: If True, save to file after each sweep point (safer).
                If False, save only at end (faster but lose data on crash).
            **kwargs: Passed to preprocessor

        Returns:
            Analysis object if analysis_factory provided, else sweep_data dict
        """
        from slab import get_next_filename

        if go_kwargs is None:
            go_kwargs = {}

        # Generate sweep values
        sweep_vals = np.arange(sweep_start, sweep_stop + sweep_step/2, sweep_step)

        # Preprocess config
        expt_cfg = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Create sweep file using Experiment object (like sequential_base_class does)
        # This gives us proper save_data() functionality
        sweep_expt = Experiment(
            path=self.station.data_path,
            prefix=f'{self.ExptClass.__name__}_sweep',
            config_file=self.station.hardware_config_file,
        )

        # Initialize sweep data structure
        sweep_key = f'{self.sweep_param}_sweep'
        sweep_expt.data = {sweep_key: []}

        print(f'Running sweep over {self.sweep_param}: {sweep_start} to {sweep_stop} (step {sweep_step})')
        print(f'  Total points: {len(sweep_vals)}')
        print(f'  Sweep file: {sweep_expt.fname}')

        # Loop over sweep parameter
        for idx, sweep_val in enumerate(sweep_vals):
            print(f'  [{idx+1}/{len(sweep_vals)}] {self.sweep_param} = {sweep_val:.4f}', end='')

            # Create experiment instance
            expt = self.ExptClass(
                soccfg=self.station.soc,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
            )

            # Setup config
            expt.cfg = AttrDict(deepcopy(self.station.config_thisrun))
            expt.cfg.expt = AttrDict(deepcopy(expt_cfg))

            # Set sweep parameter value
            expt.cfg.expt[self.sweep_param] = sweep_val

            # Handle relax_delay - may or may not exist in expt_cfg
            if hasattr(expt.cfg.expt, 'relax_delay'):
                expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

            # Run experiment (no analyze/display, no save individual runs)
            go_defaults = {"analyze": False, "display": False, "progress": False, "save": False}
            go_defaults.update(go_kwargs)
            expt.go(**go_defaults)

            # Store data - append sweep value
            sweep_expt.data[sweep_key].append(sweep_val)

            # Store experiment data (dynamically handle whatever keys exist)
            for data_key, data_val in expt.data.items():
                if data_key not in sweep_expt.data:
                    sweep_expt.data[data_key] = []
                sweep_expt.data[data_key].append(data_val)

            # Incremental save after each point (like sequential_base_class)
            if incremental_save:
                sweep_expt.save_data()
                print(' [saved]')
            else:
                print()

            # Optional live analysis/plotting callback
            if self.live_analysis_fn is not None:
                try:
                    self.live_analysis_fn(sweep_expt.data, self.station)
                except Exception as e:
                    print(f'    Live analysis warning: {e}')

        # Final save (in case incremental_save was False, or to ensure final state)
        sweep_expt.save_data()
        print(f'Sweep complete. Saved to {sweep_expt.fname}')

        # Convert lists to numpy arrays for analysis
        sweep_data = {}
        for key, val in sweep_expt.data.items():
            sweep_data[key] = np.array(val)

        # Store filename and config in sweep_data for reference
        sweep_data['_filename'] = sweep_expt.fname
        sweep_data['_config'] = expt.cfg if 'expt' in dir() else None

        # Create analysis object if factory provided
        if self.analysis_factory is not None:
            try:
                analysis = self.analysis_factory(sweep_data, self.station)
                analysis.analyze()

                # Run postprocessor if requested
                if postprocess and self.postprocessor is not None:
                    self.postprocessor(self.station, analysis)

                return analysis
            except Exception as e:
                print(f'Analysis failed: {e}')
                print('Returning raw sweep_data instead')
                return sweep_data
        else:
            # No analysis factory - just return the data
            # Postprocessor gets data dict instead of analysis object
            if postprocess and self.postprocessor is not None:
                self.postprocessor(self.station, sweep_data)
            return sweep_data


# Convenience factory functions for common analysis classes
def make_chevron_analysis_factory():
    """Returns a factory function for ChevronFitting analysis."""
    def factory(sweep_data, station):
        from fitting.fit_display_classes import ChevronFitting
        return ChevronFitting(
            frequencies=sweep_data['freq_sweep'],
            time=sweep_data['xpts'][0] if sweep_data['xpts'].ndim > 1 else sweep_data['xpts'],
            response_matrix=sweep_data['avgi'],
            config=station.config_thisrun,
            station=station,
        )
    return factory

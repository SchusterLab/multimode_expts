"""
SweepRunner: Clean runner for 2D sweep experiments.

This module provides a pattern for running parameter sweeps with:
- Incremental file saving (safe against crashes)
- Optional live plotting during sweep
- Automatic analysis at completion
- Minimal notebook boilerplate

Usage:
    from experiments.station import MultimodeStation
    from experiments.sweep_runner import SweepRunner
    from fitting.fit_display_classes import ChevronFitting
    import experiments as meas

    station = MultimodeStation(experiment_name="241215_calibration")

    # Clean notebook code - no callbacks or factories needed!
    runner = SweepRunner(
        station=station,
        ExptClass=meas.LengthRabiGeneralF0g1Experiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        analysis_class=ChevronFitting,  # Just pass the class
        live_plot=True,  # Simple flag
        preprocessor=my_preproc,  # Optional
        postprocessor=my_postproc,  # Optional
    )

    result = runner.run(
        sweep_start=1998,
        sweep_stop=2000,
        sweep_step=0.1,
    )
"""

from copy import deepcopy
from typing import Optional, Callable, Type, TYPE_CHECKING

import numpy as np
from slab import AttrDict
from slab.experiment import Experiment

if TYPE_CHECKING:
    from experiments.station import MultimodeStation


def default_preprocessor(station, default_expt_cfg, **kwargs):
    """Default preprocessor: merge kwargs into default config."""
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    return expt_cfg


class SweepRunner:
    """
    Manages execution of 2D sweep experiments.

    Unlike CharacterizationRunner which runs a single experiment,
    SweepRunner loops over a parameter, running the experiment at
    each point and saving data incrementally.

    Key features:
    - Incremental file saving (data saved after each sweep point)
    - Optional live analysis with automatic plotting
    - Automatic final analysis at completion
    - Clean notebook API (no callbacks needed)

    Supported analysis classes:
    - ChevronFitting: For frequency vs time sweeps
    - (Add more as needed)
    """

    # Registry of analysis classes and how to instantiate them from sweep_data
    ANALYSIS_CONFIGS = {
        'ChevronFitting': {
            'module': 'fitting.fit_display_classes',
            'args_map': lambda data, station: {
                'frequencies': data['freq_sweep'],
                'time': data['xpts'][0] if data['xpts'].ndim > 1 else data['xpts'],
                'response_matrix': data['avgi'],
                'config': station.config_thisrun,
                'station': station,
            }
        },
        # Add more analysis classes here as needed
    }

    def __init__(
        self,
        station: "MultimodeStation",
        ExptClass: type,
        default_expt_cfg: AttrDict,
        sweep_param: str = 'freq',
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
        analysis_class: Optional[Type] = None,
        live_plot: bool = False,
    ):
        """
        Initialize the sweep runner.

        Args:
            station: MultimodeStation instance
            ExptClass: Experiment class to run at each sweep point
            default_expt_cfg: Default experiment config template
            sweep_param: Parameter to sweep (e.g., 'freq', 'gain')
            preprocessor: Optional function(station, default_cfg, **kwargs) -> expt_cfg
            postprocessor: Optional function(station, analysis) called after sweep
            analysis_class: Analysis class (e.g., ChevronFitting). If None, returns raw data.
            live_plot: If True, show live analysis plot after each sweep point
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.sweep_param = sweep_param
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor
        self.analysis_class = analysis_class
        self.live_plot = live_plot

    def _convert_to_arrays(self, data_dict: dict) -> dict:
        """Convert all list values to numpy arrays."""
        return {key: np.array(val) for key, val in data_dict.items()}

    def _create_analysis(self, sweep_data: dict):
        """
        Create analysis object from sweep data.

        Args:
            sweep_data: Dict with numpy arrays of sweep results

        Returns:
            Analysis object (e.g., ChevronFitting instance)
        """
        if self.analysis_class is None:
            return None

        class_name = self.analysis_class.__name__

        # Check if we have a registered config for this class
        if class_name in self.ANALYSIS_CONFIGS:
            config = self.ANALYSIS_CONFIGS[class_name]
            args = config['args_map'](sweep_data, self.station)
            return self.analysis_class(**args)
        else:
            # Generic fallback: try to pass sweep_data and station
            try:
                return self.analysis_class(sweep_data, self.station)
            except TypeError:
                raise ValueError(
                    f"Analysis class {class_name} not registered in ANALYSIS_CONFIGS. "
                    f"Either register it or pass analysis_class=None and handle analysis manually."
                )

    def _do_live_plot(self, sweep_data_lists: dict):
        """
        Perform live analysis and display plot.

        Args:
            sweep_data_lists: Dict with lists (not yet converted to arrays)
        """
        try:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output

            # Convert to arrays for analysis
            sweep_data = self._convert_to_arrays(sweep_data_lists)

            # Create and run analysis
            analysis = self._create_analysis(sweep_data)
            if analysis is not None:
                analysis.analyze()

                # Update display
                clear_output(wait=True)
                plt.close('all')

                n_points = len(sweep_data[f'{self.sweep_param}_sweep'])
                analysis.display_results(
                    save_fig=False,
                    title=f'Live ({n_points} points)'
                )

        except Exception as e:
            print(f"    Live plot: {e}")

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
        Run the sweep experiment.

        Args:
            sweep_start: Starting value for sweep parameter
            sweep_stop: Ending value for sweep parameter
            sweep_step: Step size for sweep parameter
            postprocess: Whether to run postprocessor after sweep
            go_kwargs: Kwargs passed to expt.go()
            incremental_save: If True, save after each point (safer but slower)
            **kwargs: Passed to preprocessor

        Returns:
            Analysis object if analysis_class provided, else sweep_data dict
        """
        if go_kwargs is None:
            go_kwargs = {}

        # Generate sweep values (include endpoint)
        sweep_vals = np.arange(sweep_start, sweep_stop + sweep_step / 2, sweep_step)

        # Preprocess config
        expt_cfg = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Create sweep file
        sweep_expt = Experiment(
            path=self.station.data_path,
            prefix=f'{self.ExptClass.__name__}_sweep',
            config_file=self.station.hardware_config_file,
        )

        # Initialize data structure
        sweep_key = f'{self.sweep_param}_sweep'
        sweep_expt.data = {sweep_key: []}

        print(f'Sweep: {self.sweep_param} from {sweep_start} to {sweep_stop} (step {sweep_step})')
        print(f'  Points: {len(sweep_vals)}')
        print(f'  File: {sweep_expt.fname}')

        # Run sweep
        for idx, sweep_val in enumerate(sweep_vals):
            print(f'  [{idx+1}/{len(sweep_vals)}] {self.sweep_param}={sweep_val:.4f}', end='')

            # Create experiment
            expt = self.ExptClass(
                soccfg=self.station.soc,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
            )

            # Setup config
            expt.cfg = AttrDict(deepcopy(self.station.config_thisrun))
            expt.cfg.expt = AttrDict(deepcopy(expt_cfg))
            expt.cfg.expt[self.sweep_param] = sweep_val

            if hasattr(expt.cfg.expt, 'relax_delay'):
                expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

            # Run (no individual analysis/save)
            go_defaults = {"analyze": False, "display": False, "progress": False, "save": False}
            go_defaults.update(go_kwargs)
            expt.go(**go_defaults)

            # Store data
            sweep_expt.data[sweep_key].append(sweep_val)
            for data_key, data_val in expt.data.items():
                if data_key not in sweep_expt.data:
                    sweep_expt.data[data_key] = []
                sweep_expt.data[data_key].append(data_val)

            # Incremental save
            if incremental_save:
                sweep_expt.save_data()
                print(' [saved]')
            else:
                print()

            # Live plot
            if self.live_plot:
                self._do_live_plot(sweep_expt.data)

        # Final save
        sweep_expt.save_data()
        print(f'Complete. Saved to {sweep_expt.fname}')

        # Convert to arrays
        sweep_data = self._convert_to_arrays(sweep_expt.data)
        sweep_data['_filename'] = sweep_expt.fname
        sweep_data['_config'] = expt.cfg

        # Create analysis
        if self.analysis_class is not None:
            try:
                analysis = self._create_analysis(sweep_data)
                analysis.analyze()
                analysis.display_results(save_fig=False)

                if postprocess and self.postprocessor is not None:
                    self.postprocessor(self.station, analysis)

                return analysis

            except Exception as e:
                print(f'Analysis failed: {e}')
                print('Returning raw data')
                return sweep_data
        else:
            if postprocess and self.postprocessor is not None:
                self.postprocessor(self.station, sweep_data)
            return sweep_data


# Convenience function to register new analysis classes
def register_analysis_class(class_name: str, module: str, args_map: Callable):
    """
    Register a new analysis class for use with SweepRunner.

    Args:
        class_name: Name of the analysis class (e.g., 'SidebandFitting')
        module: Module path (e.g., 'fitting.fit_display_classes')
        args_map: Function(sweep_data, station) -> dict of constructor args

    Example:
        register_analysis_class(
            'SidebandFitting',
            'fitting.fit_display_classes',
            lambda data, station: {
                'frequencies': data['freq_sweep'],
                'time': data['xpts'][0],
                'response_matrix': data['avgi'],
                'config': station.config_thisrun,
            }
        )
    """
    SweepRunner.ANALYSIS_CONFIGS[class_name] = {
        'module': module,
        'args_map': args_map,
    }

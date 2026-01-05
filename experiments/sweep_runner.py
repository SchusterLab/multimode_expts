"""
SweepRunner: Clean runner for 2D sweep experiments.

This module provides a pattern for running parameter sweeps with:
- Incremental file saving (safe against crashes)
- Optional live plotting during sweep
- Automatic analysis at completion via Experiment.analyze()/display()
- Minimal notebook boilerplate

Usage:
    from experiments.station import MultimodeStation
    from experiments.sweep_runner import SweepRunner
    import experiments as meas

    station = MultimodeStation(experiment_name="241215_calibration")

    # Clean notebook code - analysis handled by Experiment class!
    runner = SweepRunner(
        station=station,
        ExptClass=meas.LengthRabiGeneralF0g1Experiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        live_plot=True,
        preprocessor=my_preproc,  # Optional
        postprocessor=my_postproc,  # Optional
    )

    result = runner.run(
        sweep_start=1998,
        sweep_stop=2000,
        sweep_npts=21,
    )

    # result is the "mother" experiment with 2D data
    # Access analysis results via result._chevron_analysis.results
"""

from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np
from slab import AttrDict

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
    - Optional live plotting via Experiment.display()
    - Automatic final analysis via Experiment.analyze()
    - Clean notebook API (no callbacks needed)

    The "mother" experiment pattern:
    - Creates an instance of ExptClass to hold the accumulated 2D data
    - Calls mother_expt.analyze() which detects 2D data and delegates
      to appropriate analysis (e.g., ChevronFitting for freq sweeps)
    - Calls mother_expt.display() which shows the 2D results
    """

    def __init__(
        self,
        station: "MultimodeStation",
        ExptClass: type,
        default_expt_cfg: AttrDict,
        sweep_param: str = 'freq',
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
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
            postprocessor: Optional function(station, mother_expt) called after sweep
            live_plot: If True, show live analysis plot after each sweep point
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.sweep_param = sweep_param
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor
        self.live_plot = live_plot

    def _convert_to_arrays(self, data_dict: dict) -> dict:
        """Convert all list values to numpy arrays."""
        return {key: np.array(val) for key, val in data_dict.items()}

    def _do_live_plot(self, mother_expt, n_points: int):
        """
        Perform live analysis and display plot using mother experiment.

        Args:
            mother_expt: The mother experiment instance with accumulated data
            n_points: Number of points collected so far (for title)
        """
        try:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output

            # Convert current data to arrays for analysis
            original_data = mother_expt.data
            mother_expt.data = self._convert_to_arrays(original_data)

            # Run analysis and display
            mother_expt.analyze(station=self.station)

            clear_output(wait=True)
            plt.close('all')

            mother_expt.display(title_str=f'Live ({n_points} points)')

            # Restore list-based data for continued accumulation
            mother_expt.data = original_data

        except Exception as e:
            print(f"    Live plot: {e}")

    def run(
        self,
        sweep_start: float,
        sweep_stop: float,
        sweep_npts: int,
        postprocess: bool = True,
        go_kwargs: Optional[dict] = None,
        incremental_save: bool = True,
        **kwargs
    ):
        """
        Run the sweep experiment.

        Args:
            sweep_start: Starting value for sweep parameter
            sweep_stop: Ending value for sweep parameter
            sweep_npts: Number of swept points for sweep parameter
            postprocess: Whether to run postprocessor after sweep
            go_kwargs: Dict passed to expt.go() (analyze, display, progress, save)
            incremental_save: If True, save after each point (safer but slower)
            **kwargs: Passed to preprocessor

        Returns:
            Mother experiment object with 2D data and analysis results.
            Access analysis via mother_expt._chevron_analysis (for freq sweeps)
            or mother_expt._length_rabi_analysis (for 1D).
        """
        go_kwargs = go_kwargs or {}

        # Generate sweep values (include endpoint)
        sweep_vals = np.linspace(sweep_start, sweep_stop, sweep_npts)

        # Preprocess config
        expt_cfg = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Create "mother" experiment to hold accumulated 2D data
        mother_expt = self.ExptClass(
            soccfg=self.station.soc,
            path=self.station.data_path,
            prefix=f'{self.ExptClass.__name__}_sweep',
            config_file=self.station.hardware_config_file,
        )
        mother_expt.cfg = AttrDict(deepcopy(self.station.config_thisrun))

        # Initialize data structure
        sweep_key = f'{self.sweep_param}_sweep'
        mother_expt.data = {sweep_key: []}

        print(f'Sweep: {self.sweep_param} from {sweep_start} to {sweep_stop} ({sweep_npts} pts)')
        print(f'  File: {mother_expt.fname}')

        # Run sweep
        for idx, sweep_val in enumerate(sweep_vals):
            print(f'  [{idx+1}/{len(sweep_vals)}] {self.sweep_param}={sweep_val:.4f}', end='')

            # Create individual experiment for this sweep point
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

            # Accumulate data into mother experiment
            mother_expt.data[sweep_key].append(sweep_val)
            for data_key, data_val in expt.data.items():
                if data_key not in mother_expt.data:
                    mother_expt.data[data_key] = []
                mother_expt.data[data_key].append(data_val)

            # Incremental save
            if incremental_save:
                mother_expt.save_data()
                print(' [saved]')
            else:
                print()

            # Live plot
            if self.live_plot:
                self._do_live_plot(mother_expt, idx + 1)

        # Final save
        mother_expt.save_data()
        print(f'Complete. Saved to {mother_expt.fname}')

        # Convert to arrays for final analysis
        mother_expt.data = self._convert_to_arrays(mother_expt.data)
        mother_expt.data['_filename'] = mother_expt.fname
        mother_expt.data['_config'] = expt.cfg

        # Run final analysis and display via Experiment methods
        try:
            mother_expt.analyze(station=self.station)
            mother_expt.display()

            if postprocess and self.postprocessor is not None:
                self.postprocessor(self.station, mother_expt)

        except Exception as e:
            print(f'Analysis failed: {e}')
            print('Returning mother experiment with raw data')

        return mother_expt

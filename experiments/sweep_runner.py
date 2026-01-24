"""
SweepRunner: Clean runner for 2D sweep experiments.

This module provides a pattern for running parameter sweeps with:
- Incremental file saving (safe against crashes)
- Optional live plotting during sweep
- Automatic analysis at completion via Experiment.analyze()/display()
- Minimal notebook boilerplate

By default, sweeps are submitted to the job queue server for execution.
This enables multi-user scheduling and hardware exclusivity. For direct local
execution (bypassing the queue), use run_local().

Usage (Queued Mode - Default):
    from multimode_expts.job_server.client import JobClient
    from experiments.station import MultimodeStation
    from experiments.sweep_runner import SweepRunner
    import experiments as meas

    client = JobClient()
    station = MultimodeStation(experiment_name="241215_calibration")

    runner = SweepRunner(
        station=station,
        ExptClass=meas.LengthRabiGeneralF0g1Experiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        live_plot=False,  # No live plot in queued mode
        preprocessor=my_preproc,
        postprocessor=my_postproc,
        job_client=client,
    )

    # Submits entire sweep as single job
    result = runner.run(
        sweep_start=1998,
        sweep_stop=2000,
        sweep_npts=21,
    )

Usage (Local Mode - Direct Execution):
    runner = SweepRunner(
        station=station,
        ExptClass=meas.LengthRabiGeneralF0g1Experiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        live_plot=True,
        preprocessor=my_preproc,
        postprocessor=my_postproc,
    )

    # Runs directly on hardware with live plotting
    result = runner.run_local(
        sweep_start=1998,
        sweep_stop=2000,
        sweep_npts=21,
    )

    # result is the "mother" experiment with 2D data
    # Access analysis results via result._chevron_analysis.results
"""

from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING, Any
import json

import numpy as np
from slab import AttrDict

if TYPE_CHECKING:
    from experiments.station import MultimodeStation
    from multimode_expts.job_server.client import JobClient, JobResult


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
        job_client: Optional["JobClient"] = None,
        use_queue: bool = True,
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
            job_client: JobClient instance for submitting to job queue (required for run())
            use_queue: If True, execute() uses run() (job queue). If False, uses run_local().
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.sweep_param = sweep_param
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor
        self.live_plot = live_plot
        self.job_client = job_client
        self.use_queue = use_queue
        self.last_job_ids = []  # Stores list of job IDs from most recent run()

    def _serialize_station_config(self) -> str:
        """
        Serialize the station's current config state to JSON.

        This captures the exact config that should be used for the experiment,
        including any updates made by previous postprocessors.

        Returns:
            JSON string containing hardware_cfg, multimode_cfg, and CSV data
        """
        # Convert hardware_cfg to a plain dict recursively, excluding non-serializable dataset objects
        def to_serializable_dict(obj, exclude_keys=None):
            """Recursively convert AttrDict/dict to plain dict, excluding specified keys."""
            if exclude_keys is None:
                exclude_keys = set()
            if isinstance(obj, dict):
                return {
                    k: to_serializable_dict(v, exclude_keys)
                    for k, v in obj.items()
                    if k not in exclude_keys
                }
            elif isinstance(obj, list):
                return [to_serializable_dict(item, exclude_keys) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        hardware_cfg_dict = to_serializable_dict(
            self.station.hardware_cfg,
            exclude_keys={'_ds_storage', '_ds_floquet'}
        )

        station_data = {
            "experiment_name": self.station.experiment_name,
            "hardware_cfg": hardware_cfg_dict,
            "hardware_config_file": str(self.station.hardware_config_file),
        }

        # Include multiphoton config if available
        if hasattr(self.station, 'multimode_cfg') and hasattr(self.station, 'multiphoton_config_file'):
            station_data["multimode_cfg"] = dict(self.station.multimode_cfg)
            station_data["multiphoton_config_file"] = str(self.station.multiphoton_config_file)

        # Include CSV dataframes as JSON-serializable data
        # Convert datetime columns (last_update) to strings for JSON serialization
        if hasattr(self.station, 'ds_storage'):
            df = self.station.ds_storage.df.copy()
            if 'last_update' in df.columns:
                df['last_update'] = df['last_update'].astype(str)
            station_data["storage_man_data"] = df.to_dict(orient='records')
            station_data["storage_man_file"] = self.station.storage_man_file

        if hasattr(self.station, 'ds_floquet') and self.station.ds_floquet is not None:
            df = self.station.ds_floquet.df.copy()
            if 'last_update' in df.columns:
                df['last_update'] = df['last_update'].astype(str)
            station_data["floquet_data"] = df.to_dict(orient='records')
            station_data["floquet_file"] = self.station.floquet_file

        return json.dumps(station_data)

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
        priority: int = 0,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        Submit sweep experiment to job queue, one job per sweep point.

        This is the default execution mode that enables multi-user scheduling.
        Each sweep point is submitted as a separate job, and results are
        accumulated into a mother experiment.

        Args:
            sweep_start: Starting value for sweep parameter
            sweep_stop: Ending value for sweep parameter
            sweep_npts: Number of swept points for sweep parameter
            postprocess: Whether to run postprocessor after sweep
            priority: Job priority (higher = runs sooner, default 0)
            poll_interval: Seconds between status checks while waiting
            timeout: Maximum seconds to wait per job (None = wait forever)
            incremental_save: If True, save mother expt after each point
            **kwargs: Passed to preprocessor

        Returns:
            Mother experiment object with 2D data and analysis results.
            Access analysis via mother_expt._chevron_analysis (for freq sweeps)
            or mother_expt._length_rabi_analysis (for 1D).

        Raises:
            ValueError: If job_client is not configured
            RuntimeError: If any job fails or is cancelled
        """
        if self.job_client is None:
            raise ValueError(
                "job_client is required for run(). Either pass job_client to "
                "SweepRunner() or use run_local() for direct execution."
            )

        # Get experiment module path from class
        experiment_module = self.ExptClass.__module__
        experiment_class = self.ExptClass.__name__

        # Generate sweep values
        sweep_vals = np.linspace(sweep_start, sweep_stop, sweep_npts)

        # Preprocess config
        base_expt_cfg = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Create "mother" experiment to hold accumulated 2D data
        mother_expt = self.ExptClass(
            soccfg=self.station.soc,
            path=self.station.data_path,
            prefix=f'{self.ExptClass.__name__}_sweep',
            config_file=self.station.hardware_config_file,
        )
        mother_expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))
        mother_expt.cfg.expt = base_expt_cfg

        # Initialize data structure
        sweep_key = f'{self.sweep_param}_sweep'
        mother_expt.data = {sweep_key: []}

        print(f'Sweep: {self.sweep_param} from {sweep_start} to {sweep_stop} ({sweep_npts} pts)')
        print(f'  File: {mother_expt.fname}')

        # Store job results for reference
        self.job_results = []

        # Run sweep - one job per point
        for idx, sweep_val in enumerate(sweep_vals):
            print(f'  [{idx+1}/{len(sweep_vals)}] {self.sweep_param}={sweep_val:.4f}', end=' ')

            # Create config for this sweep point
            expt_config = deepcopy(base_expt_cfg)
            expt_config[self.sweep_param] = sweep_val

            # Handle relax_delay if present (worker needs this in config)
            if hasattr(expt_config, 'relax_delay'):
                expt_config['_relax_delay'] = expt_config.relax_delay

            # Convert to dict for JSON serialization, handling numpy types
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                return obj

            expt_config_dict = convert_numpy(dict(expt_config))

            # Serialize station config to pass with job
            station_config_json = self._serialize_station_config()

            # Submit job
            job_id = self.job_client.submit_job(
                experiment_class=experiment_class,
                experiment_module=experiment_module,
                expt_config=expt_config_dict,
                station_config=station_config_json,
                user=self.station.user,
                priority=priority,
            )

            # Wait for completion
            result = self.job_client.wait_for_completion(
                job_id,
                poll_interval=poll_interval,
                timeout=timeout,
                verbose=False,
            )
            self.job_results.append(result)

            # Check for failure
            if not result.is_successful():
                print(f'FAILED: {result.error_message}')
                raise RuntimeError(
                    f"Job {job_id} {result.status}: {result.error_message or 'No details'}"
                )

            # Load expt and accumulate data
            expt = result.load_expt()

            mother_expt.data[sweep_key].append(sweep_val)
            for data_key, data_val in expt.data.items():
                if data_key not in mother_expt.data:
                    mother_expt.data[data_key] = []
                mother_expt.data[data_key].append(data_val)

        # Store all job IDs from this sweep for reference
        self.last_job_ids = [r.job_id for r in self.job_results]

        # Final save
        mother_expt.save_data()
        print(f'Complete. Saved to {mother_expt.fname}')

        # Convert to arrays for final analysis
        mother_expt.data = self._convert_to_arrays(mother_expt.data)
        mother_expt.data['_filename'] = mother_expt.fname
        mother_expt.data['_config'] = expt.cfg

        # Run final analysis and display
        try:
            # mother_expt.analyze(station=self.station)
            # mother_expt.display() # No display with job scheduler

            if postprocess and self.postprocessor is not None:
                self.postprocessor(self.station, mother_expt)

        except Exception as e:
            print(f'Analysis failed: {e}')
            print('Returning mother experiment with raw data')

        return mother_expt

    def run_local(
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
        Run the sweep experiment locally, bypassing the job queue.

        Use this for direct hardware access when the job queue is not needed
        (e.g., single-user mode, debugging, or when you have exclusive hardware access).

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
        mother_expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))

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
            expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))
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

    def execute(
        self,
        sweep_start: float,
        sweep_stop: float,
        sweep_npts: int,
        use_queue: Optional[bool] = None,
        **kwargs
    ):
        """
        Run sweep using configured or specified execution mode.

        This is a convenience method that dispatches to run() or run_local()
        based on the use_queue flag, allowing notebooks to toggle execution
        mode without changing individual experiment calls.

        Args:
            sweep_start: Starting value for sweep parameter
            sweep_stop: Ending value for sweep parameter
            sweep_npts: Number of swept points for sweep parameter
            use_queue: Override instance setting. If None, uses self.use_queue.
                       True = run() via job queue, False = run_local()
            **kwargs: Passed to run() or run_local()

        Returns:
            Mother experiment object with 2D data and analysis results.
        """
        mode = use_queue if use_queue is not None else self.use_queue

        if mode:
            return self.run(sweep_start, sweep_stop, sweep_npts, **kwargs)
        else:
            return self.run_local(sweep_start, sweep_stop, sweep_npts, **kwargs)

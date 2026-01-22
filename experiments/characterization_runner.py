"""
CharacterizationRunner: Simple runner for single-point experiments.

This module provides a clean pattern for running characterization experiments
with minimal boilerplate in notebooks. Just define:
- default_expt_cfg: Default experiment parameters
- preprocessor: Optional function to transform config (e.g., span/center -> start/step)
- postprocessor: Optional function to extract results and update station config

By default, experiments are submitted to the job queue server for execution.
This enables multi-user scheduling and hardware exclusivity. For direct local
execution (bypassing the queue), use run_local().

Usage (Queued Mode - Default):
    from multimode_expts.job_server.client import JobClient
    from experiments.station import MultimodeStation
    from experiments.characterization_runner import CharacterizationRunner
    import experiments as meas

    client = JobClient()
    station = MultimodeStation(experiment_name="241215_calibration")

    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.QubitSpectroscopyExperiment,
        default_expt_cfg=defaults,
        preprocessor=my_preproc,  # Optional
        postprocessor=my_postproc,  # Optional
        job_client=client,
    )

    # Submits to job queue (default behavior)
    result = runner.run(center=4500, span=100)

Usage (Local Mode - Direct Execution):
    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.ResonatorSpectroscopyExperiment,
        default_expt_cfg=defaults,
        preprocessor=my_preproc,
        postprocessor=my_postproc,
    )

    # Runs directly on hardware, bypassing job queue
    expt = runner.run_local(some_param=123)
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Callable, Protocol, TYPE_CHECKING, Any, Union
import json

from slab import AttrDict
from slab.experiment import Experiment

if TYPE_CHECKING:
    from experiments.station import MultimodeStation
    from multimode_expts.job_server.client import JobClient, JobResult


class PreProcessor(Protocol):
    """Protocol for preprocessor functions."""

    def __call__(
        self, station: "MultimodeStation", default_expt_cfg: AttrDict, **kwargs
    ) -> AttrDict:
        """
        Transform default config with user kwargs into final expt config.

        Args:
            station: MultimodeStation instance
            default_expt_cfg: Default experiment config template
            **kwargs: User-provided overrides

        Returns:
            Final AttrDict config for the experiment
        """
        ...


class PostProcessor(Protocol):
    """Protocol for postprocessor functions."""

    def __call__(self, station: "MultimodeStation", expt: Experiment) -> Any:
        """
        Extract results from experiment and update station config.

        Args:
            station: MultimodeStation instance
            expt: Completed experiment object with results

        Returns:
            Extracted result value (e.g., fitted frequency), or None
        """
        ...


def default_preprocessor(station, default_expt_cfg, **kwargs):
    """
    Default preprocessor: simply update default config with user kwargs.

    If your preprocessor just needs to merge kwargs into the default config,
    you don't need to write one - leave preprocessor=None and this is used.

    For custom logic (e.g., converting span/center to start/stop), write your
    own preprocessor following this pattern.
    """
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    return expt_cfg


def default_postprocessor(station, expt):
    """
    Default postprocessor: does nothing.

    Override this to extract fit results and update station.hardware_cfg.

    Returns:
        None
    """
    return None


class CharacterizationRunner:
    """
    Manages execution of single-point characterization experiments.

    Encapsulates the boilerplate of:
    - Creating experiment instance
    - Setting up configuration
    - Running the experiment (via job queue or locally)
    - Extracting results to update config

    By default, experiments are submitted to the job queue for multi-user
    scheduling. Use run_local() for direct execution without the queue.
    """

    def __init__(
        self,
        station: "MultimodeStation",
        ExptClass: type,
        default_expt_cfg: AttrDict,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
        ExptProgram: Optional[type] = None,
        job_client: Optional["JobClient"] = None,
        use_queue: bool = True,
    ):
        """
        Initialize the runner.

        Args:
            station: MultimodeStation instance for hardware access
            ExptClass: Experiment class to instantiate (e.g., meas.SomeExperiment)
            default_expt_cfg: AttrDict template for expt.cfg.expt
            preprocessor: Function to generate expt.cfg.expt from defaults + kwargs
            postprocessor: Function to extract results and update station.hardware_cfg
            ExptProgram: for QsimBaseExperiment, this is the program class to use
            job_client: JobClient instance for submitting to job queue (required for run())
            user: Username for job submission (default: "anonymous")
            use_queue: If True, execute() uses run() (job queue). If False, uses run_local().
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor or default_postprocessor
        self.program = ExptProgram
        self.job_client = job_client
        self.use_queue = use_queue
        self.last_job_result = None  # Stores JobResult from most recent run()

    def _serialize_station_config(self) -> str:
        """
        Serialize the station's current config state to JSON.

        This captures the exact config that should be used for the experiment,
        including any updates made by previous postprocessors.

        Returns:
            JSON string containing hardware_cfg, multimode_cfg, and CSV data
        """
        station_data = {
            "hardware_cfg": dict(self.station.hardware_cfg),
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

    def run(
        self,
        postprocess: bool = True,
        priority: int = 0,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Experiment:
        """
        Submit experiment to job queue and wait for completion.

        This is the default execution mode that enables multi-user scheduling.
        The experiment is submitted to the job server, executed by the worker,
        and the resulting expt object is loaded from disk.

        Args:
            postprocess: Whether to run postprocessor after experiment
            priority: Job priority (higher = runs sooner, default 0)
            poll_interval: Seconds between status checks while waiting
            timeout: Maximum seconds to wait (None = wait forever)
            **kwargs: Passed to preprocessor to modify config

        Returns:
            Completed Experiment object (loaded from worker's pickle file)

        Raises:
            ValueError: If job_client is not configured
            RuntimeError: If job fails or is cancelled
        """
        if self.job_client is None:
            raise ValueError(
                "job_client is required for run(). Either pass job_client to "
                "CharacterizationRunner() or use run_local() for direct execution."
            )

        # Get experiment module path from class
        experiment_module = self.ExptClass.__module__
        experiment_class = self.ExptClass.__name__

        # Run preprocessor to get final config
        expt_config = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Convert AttrDict to regular dict for JSON serialization
        if hasattr(expt_config, "to_dict"):
            expt_config_dict = dict(expt_config)
        else:
            expt_config_dict = dict(expt_config)

        # Serialize station config to pass with job
        station_config_json = self._serialize_station_config()

        # Submit job to queue
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
            verbose=True,
        )

        # Store result for later access (job_id, config versions, etc.)
        self.last_job_result = result

        # Check for failure
        if not result.is_successful():
            raise RuntimeError(
                f"Job {job_id} {result.status}: {result.error_message or 'No details'}"
            )

        # Load the expt object from pickle file
        expt = result.load_expt()

        # Run postprocessor
        if postprocess:
            self.postprocessor(self.station, expt)

        return expt

    def run_local(
        self, postprocess: bool = True, go_kwargs: Optional[dict] = None, **kwargs
    ) -> Experiment:
        """
        Run the experiment locally, bypassing the job queue.

        Use this for direct hardware access when the job queue is not needed
        (e.g., single-user mode, debugging, or when you have exclusive hardware access).

        Args:
            postprocess: Whether to run postprocessor after experiment
            go_kwargs: Dict passed to expt.go() (analyze, display, progress, save)
            **kwargs: Passed to preprocessor to modify config

        Returns:
            Completed Experiment object
        """
        go_kwargs = go_kwargs or {}

        # Create experiment instance
        if self.program is not None:
            expt = self.ExptClass(
                soccfg=self.station.soc,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
                program=self.program,
            )
        else:
            expt = self.ExptClass(
                soccfg=self.station.soc,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
            )

        # Setup config
        expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))
        expt.cfg.expt = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Handle relax_delay if present
        if hasattr(expt.cfg.expt, "relax_delay"):
            expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

        # Run with sensible defaults
        go_defaults = {"analyze": True, "display": True, "progress": True, "save": True}
        go_defaults.update(go_kwargs)
        expt.go(**go_defaults)

        # Run postprocessor
        if postprocess:
            self.postprocessor(self.station, expt)

        return expt

    def execute(self, use_queue: Optional[bool] = None, **kwargs) -> Experiment:
        """
        Run experiment using configured or specified execution mode.

        This is a convenience method that dispatches to run() or run_local()
        based on the use_queue flag, allowing notebooks to toggle execution
        mode without changing individual experiment calls.

        Args:
            use_queue: Override instance setting. If None, uses self.use_queue.
                       True = run() via job queue, False = run_local()
            **kwargs: Passed to run() or run_local()

        Returns:
            Completed Experiment object
        """
        mode = use_queue if use_queue is not None else self.use_queue

        if mode:
            return self.run(**kwargs)
        else:
            return self.run_local(**kwargs)

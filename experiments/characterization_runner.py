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
    from job_server.client import JobClient
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

Usage (Many Jobs):
    # One job per override dict. In queue mode at most batch_size jobs wait in
    # the queue at a time, so the worker always has work but a long campaign
    # is not queued all at once. Returns a list of Experiments, in order.
    expts = runner.execute(overrides=[dict(reps=100), dict(reps=200)], batch_size=10)
    runner.last_job_ids   # queue job IDs; empty for local runs
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Callable, Protocol, TYPE_CHECKING, Any, Union
import inspect
import json

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict
from slab.experiment import Experiment

if TYPE_CHECKING:
    from experiments.station import MultimodeStation
    from job_server.client import JobClient, JobResult


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


def json_plain(obj):
    """Convert numpy values in a config to plain types for the queue's JSON."""
    if isinstance(obj, dict):
        return {key: json_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_plain(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _is_real_job_client(client):
    """Does this client talk to the actual job server?

    Imported lazily so this module does not pull in job_server at import
    time, and returns False if job_server is unavailable -- in which case
    nothing could reach a real queue anyway.
    """
    try:
        from job_server import JobClient
    except Exception:
        return False
    return isinstance(client, JobClient)


# run() keywords that act after the job finishes; everything else except
# ``priority`` goes to the preprocessor.
_COLLECT_KEYS = ("postprocess", "poll_interval", "timeout", "log", "show", "display_kwargs")


def mock_run_defaults(kwargs: dict, local: bool) -> dict:
    """Defaults for a run on a mock station: acquire and save only.

    Mock data is all zeros, so a fit on it gives nothing useful, and a
    postprocessor would write that nothing into the station config. The
    caller's explicit arguments still win. Display is already skipped for
    mock stations in the render/log step.

    Args:
        kwargs: The keyword arguments for run() / run_local().
        local: True for run_local(), which also takes go_kwargs.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("postprocess", False)
    if local:
        kwargs["go_kwargs"] = {"analyze": False, **(kwargs.get("go_kwargs") or {})}
    return kwargs


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
        show: bool = True,
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
            use_queue: If True, execute() uses run() (job queue). If False, uses run_local().
            show: Default for whether to render the experiment's plot inline (per-call
                overridable via show=). Independent of logging (log=).
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor or default_postprocessor
        self.program = ExptProgram
        self.job_client = job_client
        self.last_job_result = None  # Stores JobResult from most recent run()
        self.last_job_ids = []  # Queue job IDs of the most recent run()/execute()
        self.use_queue = use_queue
        self.show = show

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

    def _render_log_show(self, experiment, show: Optional[bool], log: Optional[bool],
                         display_kwargs: Optional[dict] = None):
        """Render the experiment's plot once, then route the figure: show it
        inline (show) and/or attach it to the lab-notebook vault (log).

        show and log are INDEPENDENT controls:
          - show: render the plot inline. None -> the runner default (self.show).
          - log:  write to the vault. True forces; False skips; None defers to
                  `station.log_measurements` (a per-session opt-in, default False).
                  log_measurement is itself a no-op if vault_root is unset.
        The figure is rendered at most once, iff show or log is active.

        display_kwargs is filtered to experiment.display()'s signature (and
        station= is injected if accepted), so callers can pass the correct target
        (e.g. initial_state=, rotate=, state_label= for Wigner) instead of letting
        display() fall back to its generic default.
        """
        # Resolve logging intent.
        if log is False:
            do_log = False
        elif log is None:
            do_log = bool(getattr(self.station, "log_measurements", False))
        else:
            do_log = True
        if getattr(self.station, "is_mock", False):
            # Mock runs must not touch the real vault, and we skip rendering too
            # (tests run headless; nothing to display).
            if do_log:
                print("[char] mock mode active; skipping display + log_measurement.")
            return

        do_show = self.show if show is None else bool(show)
        if not (do_log or do_show):
            return

        display_kwargs = dict(display_kwargs or {})
        returned = None
        captured_fig = None
        try:
            # Filter display_kwargs to display()'s signature; inject station= if
            # it accepts it (e.g. HistogramExperiment needs the Histogram fitter).
            sig = inspect.signature(experiment.display)
            has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            call_kwargs = (display_kwargs if has_varkw
                           else {k: v for k, v in display_kwargs.items()
                                 if k in sig.parameters})
            if "station" in sig.parameters and "station" not in call_kwargs:
                call_kwargs["station"] = self.station

            # Snapshot existing figures so we only consider those newly created by
            # display(); patch plt.show to a no-op so the new fig stays tracked
            # (Jupyter's inline backend untracks figs when show() is called).
            fignums_before = set(plt.get_fignums())
            _orig_show = plt.show
            plt.show = lambda *a, **k: None
            try:
                returned = experiment.display(**call_kwargs)
            finally:
                plt.show = _orig_show
            new_fignums = sorted(set(plt.get_fignums()) - fignums_before)
            if new_fignums:
                captured_fig = [plt.figure(n) for n in new_fignums]
        except Exception as exc:
            print(f"[runner] experiment.display() failed: {exc}")

        # Prefer display()'s return only if it's a single Figure; otherwise fall
        # back to the list of new figs we captured (multi-panel case).
        if returned is not None and hasattr(returned, "savefig"):
            fig = returned
        else:
            fig = captured_fig

        figs_to_close = list(captured_fig) if captured_fig else []
        if returned is not None and hasattr(returned, "savefig") and returned not in figs_to_close:
            figs_to_close.append(returned)
        try:
            if do_log:
                try:
                    self.station.log_measurement(experiment, fig=fig)
                except Exception as exc:
                    print(f"[runner] log_measurement failed: {exc}")
            if do_show:
                plt.show()
        finally:
            # Close the figs we created (inline render already happened) so they
            # don't leak or double-render at cell end.
            for f in figs_to_close:
                try:
                    plt.close(f)
                except Exception:
                    pass

    def run(
        self,
        postprocess: bool = True,
        priority: int = 0,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        log: Optional[bool] = None,
        show: Optional[bool] = None,
        display_kwargs: Optional[dict] = None,
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
            log: Write to the lab-notebook vault. None -> station.log_measurements.
            show: Render the plot inline. None -> the runner default (self.show).
            display_kwargs: Dict forwarded (filtered) to expt.display(), e.g.
                dict(initial_state=..., rotate=..., state_label=...) for Wigner.
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
        job_id = self._submit(priority=priority, **kwargs)
        self.last_job_ids = [job_id]
        return self._collect(
            job_id,
            postprocess=postprocess,
            poll_interval=poll_interval,
            timeout=timeout,
            log=log,
            show=show,
            display_kwargs=display_kwargs,
        )

    def _submit(self, priority: int = 0, **kwargs) -> str:
        """Preprocess one config, submit it to the job queue, return the job ID."""
        program_module = None
        program_class = None
        if self.program is not None:
            program_module = self.program.__module__
            program_class = self.program.__name__

        # Run preprocessor to get final config
        expt_config = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        return self.job_client.submit_job(
            experiment_class=self.ExptClass.__name__,
            experiment_module=self.ExptClass.__module__,
            expt_config=json_plain(dict(expt_config)),
            # The station config as it is now, including any updates made by
            # earlier postprocessors.
            station_config=self._serialize_station_config(),
            user=self.station.user,
            priority=priority,
            program_class=program_class,
            program_module=program_module,
        )

    def _collect(
        self,
        job_id: str,
        postprocess: bool = True,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        log: Optional[bool] = None,
        show: Optional[bool] = None,
        display_kwargs: Optional[dict] = None,
        verbose: bool = True,
    ) -> Experiment:
        """Wait for one queued job, load its Experiment, postprocess and render."""
        result = self.job_client.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            timeout=timeout,
            verbose=verbose,
        )

        # Store result for later access (job_id, config versions, etc.)
        self.last_job_result = result

        if not result.is_successful():
            raise RuntimeError(
                f"Job {job_id} {result.status}: {result.error_message or 'No details'}"
            )

        # Load the expt object from pickle file
        expt = result.load_expt()

        if postprocess:
            self.postprocessor(self.station, expt)

        # Queue path: the worker ran go(display=False, save=True) — data only.
        # Render/show/log happen here in the notebook process, where the caller's
        # display_kwargs (e.g. a qutip initial_state) is available in memory.
        self._render_log_show(expt, show=show, log=log, display_kwargs=display_kwargs)

        return expt

    def run_local(
        self,
        postprocess: bool = True,
        go_kwargs: Optional[dict] = None,
        log: Optional[bool] = None,
        show: Optional[bool] = None,
        display_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Experiment:
        """
        Run the experiment locally, bypassing the job queue.

        Use this for direct hardware access when the job queue is not needed
        (e.g., single-user mode, debugging, or when you have exclusive hardware access).

        Args:
            postprocess: Whether to run postprocessor after experiment
            go_kwargs: Dict passed to expt.go() (analyze, progress, save). NOTE:
                'display' is ignored — the runner owns plotting now (use show=).
            log: Write to the lab-notebook vault. None -> station.log_measurements.
            show: Render the plot inline. None -> the runner default (self.show).
            display_kwargs: Dict forwarded (filtered) to expt.display() (e.g.
                initial_state=, rotate=, state_label= for Wigner).
            **kwargs: Passed to preprocessor to modify config

        Returns:
            Completed Experiment object
        """
        go_kwargs = dict(go_kwargs or {})
        go_kwargs.pop("display", None)  # runner owns plotting (use show=)

        # Create experiment instance
        if self.program is not None:
            expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
                program=self.program,
            )
        else:
            expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
            )

        # Use the station's instrument manager (mock or real) rather than the
        # real one slab's Experiment.__init__ builds by default. Otherwise
        # expt.acquire() calls self.im[aliases.soc] against a live Pyro proxy
        # and hangs in mock mode (never reaches the MockQickSoc).
        # expt.im = self.station.im
        # ^^ we are passing station to the Experiment now as a class property, shouldn't need this anymore?

        # Setup config
        expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))

        # Pass dataset objects via expt.cfg since Program classes cannot access station
        expt.cfg.device.storage._ds_storage = self.station.ds_storage
        expt.cfg.device.storage._ds_floquet = self.station.ds_floquet

        expt.cfg.expt = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Handle relax_delay if present
        if hasattr(expt.cfg.expt, "relax_delay"):
            expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

        # Run with sensible defaults (display off: the runner renders, not go()).
        go_defaults = {"analyze": True, "display": False, "progress": True, "save": True}
        go_defaults.update(go_kwargs)
        expt.go(**go_defaults)

        # Run postprocessor
        if postprocess:
            self.postprocessor(self.station, expt)

        # Runner owns render/show/log (single path for local and queue).
        self._render_log_show(expt, show=show, log=log, display_kwargs=display_kwargs)

        return expt

    def execute(
        self,
        overrides: Optional[list] = None,
        batch_size: int = 10,
        use_queue: Optional[bool] = None,
        allow_queue_in_mock: bool = False,
        **kwargs,
    ) -> Union[Experiment, list]:
        """
        Run one job, or one job per override dict, in the configured mode.

        Dispatches to run() or run_local() based on the use_queue flag, so
        notebooks can toggle execution mode without changing individual calls.

        Args:
            overrides: None runs one job and returns its Experiment. A list of
                override dicts runs one job per dict, in order, and returns a
                list of Experiments. Each dict is merged over **kwargs and goes
                to the preprocessor, like the kwargs of a single call; it is
                not a cfg.expt.
            batch_size: With overrides in queue mode, the most jobs waiting in the
                queue at a time. The next group is submitted when the current
                one is collected. Local runs go one at a time and ignore it.
            use_queue: Override instance setting. If None, uses self.use_queue,
                except on a mock station, which then runs locally.
                True = run() via job queue, False = run_local()
            allow_queue_in_mock: use_queue=True on a mock station raises,
                because the worker runs the main checkout against real
                hardware unless it was started with --mock. Set True if it was.
            **kwargs: Passed to run() or run_local(). On a mock station,
                postprocess and go_kwargs['analyze'] default to False
                (see mock_run_defaults).

        Returns:
            Completed Experiment, or a list of them if overrides is given.
            Queue job IDs are in self.last_job_ids.
        """
        mode = use_queue if use_queue is not None else self.use_queue
        is_mock = getattr(self.station, "is_mock", False)
        # A mock station runs locally unless this call asks for the queue.
        if is_mock and use_queue is None:
            mode = False

        if mode and self.job_client is None:
            raise ValueError(
                "job_client is required for queue mode. Pass job_client to "
                "CharacterizationRunner(), or use_queue=False for direct execution."
            )
        if (mode and is_mock and not allow_queue_in_mock
                and _is_real_job_client(self.job_client)):
            raise RuntimeError(
                "execute() would submit to the job queue, but this station has "
                "mock instruments. The queue worker would run the main checkout "
                "against real hardware unless it was started with --mock. Pass "
                "use_queue=False for local execution, or start a mock worker "
                "and pass allow_queue_in_mock=True."
            )

        if overrides is None:
            if is_mock:
                kwargs = mock_run_defaults(kwargs, local=not mode)
            return self.run(**kwargs) if mode else self.run_local(**kwargs)

        if (isinstance(batch_size, (bool, np.bool_))
                or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        overrides = list(overrides)
        if not overrides:
            raise ValueError("overrides cannot be empty")

        jobs = []
        for job_overrides in overrides:
            run_kwargs = {**kwargs, **job_overrides}
            if is_mock:
                run_kwargs = mock_run_defaults(run_kwargs, local=not mode)
            jobs.append(run_kwargs)

        if not mode:
            self.last_job_ids = []
            return [self.run_local(**run_kwargs) for run_kwargs in jobs]
        return self._run_queue_batches(jobs, batch_size)

    def _run_queue_batches(self, jobs: list, batch_size: int) -> list:
        """Submit at most batch_size jobs, collect them in order, repeat.

        If anything fails or is interrupted, the jobs still waiting in the
        queue are cancelled, so the queue is not left holding jobs nobody
        is waiting for.
        """
        expts = []
        self.last_job_ids = []
        self.last_job_result = None
        for start in range(0, len(jobs), batch_size):
            group = jobs[start:start + batch_size]
            print(f"batch {start // batch_size + 1}: {len(group)} jobs")
            pending = []
            try:
                for run_kwargs in group:
                    submit_kwargs = {k: v for k, v in run_kwargs.items()
                                     if k not in _COLLECT_KEYS}
                    job_id = self._submit(**submit_kwargs)
                    pending.append((job_id, run_kwargs))
                    self.last_job_ids.append(job_id)

                while pending:
                    job_id, run_kwargs = pending[0]
                    collect_kwargs = {k: run_kwargs[k] for k in _COLLECT_KEYS
                                      if k in run_kwargs}
                    expts.append(self._collect(job_id, verbose=False, **collect_kwargs))
                    pending.pop(0)
            except BaseException:  # includes KeyboardInterrupt
                for job_id, _ in pending:
                    try:
                        self.job_client.cancel_job(job_id)
                    except Exception:
                        pass
                raise
        return expts

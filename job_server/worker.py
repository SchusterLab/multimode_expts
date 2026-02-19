"""
Job worker daemon for executing queued experiments.

This worker:
- Polls the database for pending jobs
- Executes experiments one at a time (hardware exclusivity)
- Updates job status and saves results
- Handles graceful shutdown on SIGINT/SIGTERM

Run with:
    cd /Users/conniemiao/GDriveStanford/SchusterLab/local_multimode

    # Mock mode (for testing without hardware):
    python -m multimode_expts.job_server.worker --mock

    # Real hardware mode:
    python -m multimode_expts.job_server.worker
"""

import argparse
import atexit
import importlib
import json
import os
import pickle
import signal
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from experiments.station import MultimodeStation
from job_server.config_versioning import ConfigVersionManager
from job_server.database import get_database
from job_server.id_generator import IDGenerator
from job_server.models import Job, JobOutput, JobStatus
from job_server.output_capture import OutputCapture
from slab.datamanagement import AttrDict

# Patch tqdm_notebook to use regular tqdm (tqdm_notebook uses IPython widgets, not stdout)
# This must happen before experiment modules are imported
import tqdm as tqdm_module
import tqdm.notebook
tqdm_module.tqdm_notebook = tqdm_module.tqdm
tqdm_module.notebook.tqdm_notebook = tqdm_module.tqdm
tqdm_module.notebook.tqdm = tqdm_module.tqdm


# Default lock file location (in the job_server directory)
DEFAULT_LOCK_FILE = Path(__file__).parent / "worker.lock"


class WorkerLock:
    """
    PID-based lock to prevent multiple workers from running simultaneously.

    How it works:
    1. On acquire(): Check if lock file exists
       - If no lock file: create it with our PID, we have the lock
       - If lock file exists: read the PID and check if that process is alive
         - If process is dead: delete stale lock, create new one with our PID
         - If process is alive: raise error (another worker is running)
    2. On release(): Delete the lock file

    The lock is automatically released when the process exits (via atexit),
    but if the process crashes or is killed with SIGKILL, the lock file
    remains. The next worker will detect the stale lock and clean it up.
    """

    def __init__(self, lock_file: Path = DEFAULT_LOCK_FILE):
        self.lock_file = lock_file
        self._acquired = False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running."""
        if sys.platform == "win32":
            # Windows-specific check using psutil or ctypes
            try:
                import psutil
                return psutil.pid_exists(pid)
            except ImportError:
                # Fallback to ctypes if psutil not available
                import ctypes
                kernel32 = ctypes.windll.kernel32
                PROCESS_QUERY_INFORMATION = 0x0400
                handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
        else:
            # Unix/Linux: os.kill with signal 0 just checks existence
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def acquire(self) -> bool:
        """
        Try to acquire the lock.

        Returns:
            True if lock was acquired successfully

        Raises:
            RuntimeError: If another worker is already running
        """
        if self.lock_file.exists():
            # Lock file exists - check if the process is still running
            try:
                with open(self.lock_file, "r") as f:
                    old_pid = int(f.read().strip())

                if self._is_process_running(old_pid):
                    # Another worker is actually running
                    raise RuntimeError(
                        f"Another worker is already running (PID {old_pid}). "
                        f"If you believe this is an error, delete {self.lock_file}"
                    )
                else:
                    # Stale lock file - process is dead
                    print(f"[WORKER] Removing stale lock file (old PID {old_pid} is not running)")
                    self.lock_file.unlink()

            except (ValueError, IOError) as e:
                # Corrupted lock file - remove it
                print(f"[WORKER] Removing corrupted lock file: {e}")
                self.lock_file.unlink()

        # Create lock file with our PID
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

        self._acquired = True

        # Register cleanup on exit (handles graceful shutdown)
        atexit.register(self.release)

        return True

    def release(self):
        """Release the lock by deleting the lock file."""
        if self._acquired and self.lock_file.exists():
            try:
                # Only delete if the file contains our PID (safety check)
                with open(self.lock_file, "r") as f:
                    file_pid = int(f.read().strip())

                if file_pid == os.getpid():
                    self.lock_file.unlink()
                    print("[WORKER] Lock released")
            except (ValueError, IOError, FileNotFoundError):
                pass  # File already gone or corrupted
            self._acquired = False


class JobWorker:
    """
    Single-threaded worker that processes jobs from the queue.

    Only one worker should run at a time to ensure hardware exclusivity.
    The worker polls the database for pending jobs, executes them in
    priority order, and updates job status.
    """

    def __init__(
        self,
        mock_mode: bool = False,
        poll_interval: float = 2.0,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the job worker.

        Args:
            mock_mode: If True, use MockStation instead of real hardware
            poll_interval: Seconds between database polls when idle
            experiment_name: Name for the experiment session (default: auto-generated)
        """
        self.mock_mode = mock_mode
        self.poll_interval = poll_interval
        self.experiment_name = (
            experiment_name or f'{datetime.now().strftime("%y%m%d")}_job_worker'
        )

        self.running = True
        self.current_job: Optional[Job] = None

        # Initialize database
        self.db = get_database()

        # Initialize config version manager
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.config_manager = ConfigVersionManager(self.config_dir)

        # Initialize station with hardware connections
        # Config will be updated per-job from serialized notebook config
        # Uses unified MultimodeStation with mock parameter
        self.station = MultimodeStation(
            experiment_name=self.experiment_name,
            mock=self.mock_mode,
        )

        # Setup signal handlers for graceful shutdown
        self._interrupt_count = 0
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Clean up any jobs left in RUNNING state from previous crashes
        self._cleanup_incomplete_jobs()

        print(f"[WORKER] Initialized in {'MOCK' if mock_mode else 'REAL'} mode")

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C to kill current job immediately."""
        self._interrupt_count += 1

        if self._interrupt_count >= 2:
            # Second Ctrl+C: force exit immediately
            print("\n[WORKER] Second Ctrl+C received, forcing exit...")
            sys.exit(1)

        if self.current_job:
            # First Ctrl+C while job is running: cancel current job but keep worker running
            print(f"\n[WORKER] Ctrl+C received, cancelling current job {self.current_job.job_id}...")
            print("[WORKER] Worker will continue processing queue. Press Ctrl+C again to stop worker.")
            # Raise KeyboardInterrupt to stop the current experiment
            raise KeyboardInterrupt("Job cancelled by user")
        else:
            # First Ctrl+C while idle: just stop gracefully
            print("\n[WORKER] Ctrl+C received, shutting down...")
            self.running = False

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[WORKER] Received signal {signum}, shutting down after current job...")
        self.running = False

    def _update_station_from_job_config(self, station_config_json: str):
        """
        Update station's config attributes from serialized job config.

        This updates the worker's station with the exact config state from
        the notebook at job submission time, including any postprocessor updates.

        Args:
            station_config_json: JSON string containing station config data
        """

        station_data = json.loads(station_config_json)

        # Update experiment_name and reinitialize output paths if provided
        if "experiment_name" in station_data:
            self.station.experiment_name = station_data["experiment_name"]
            self.station._initialize_output_paths()  # Routes internally based on mock mode

        # Update station's hardware_cfg
        self.station.hardware_cfg = AttrDict(station_data["hardware_cfg"])

        # Update multiphoton config
        self.station.multimode_cfg = AttrDict(station_data["multimode_cfg"])

        # Update CSV dataframes
        self.station.ds_storage.df = pd.DataFrame(station_data["storage_man_data"])
        self.station.ds_floquet.df = pd.DataFrame(station_data["floquet_data"])

        print(f"[WORKER] Updated station config from job: {self.station.experiment_name}")

    def run(self):
        """
        Main worker loop.

        Continuously polls for jobs and executes them until shutdown.
        """
        print(f"[WORKER] Starting main loop (poll interval: {self.poll_interval}s)")
        print("[WORKER] Press Ctrl+C to cancel current job (worker continues), or Ctrl+C while idle to stop")

        while self.running:
            job = self._fetch_next_job()

            if job:
                self._execute_job(job)
            else:
                # No jobs available, wait before polling again
                time.sleep(self.poll_interval)

        print("[WORKER] Shutdown complete")

    def _fetch_next_job(self) -> Optional[Job]:
        """
        Fetch the next pending job from the queue.

        Jobs are selected in priority order (highest first), then by
        creation time (oldest first = FIFO for same priority).

        Returns:
            Job object if one is available, None otherwise
        """
        with self.db.session() as session:
            # Find highest priority pending job
            job = (
                session.query(Job)
                .filter_by(status=JobStatus.PENDING)
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .first()
            )

            if job:
                # Claim the job by setting status to RUNNING
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                session.flush()

                # Detach job from session for use outside
                session.expunge(job)
                print(f"[WORKER] Claimed job: {job.job_id} ({job.experiment_class})")
                return job

        return None

    def _execute_job(self, job: Job):
        """
        Execute a single job with output capture.

        Steps:
        1. Initialize station if needed
        2. Capture stdout/stderr for streaming to client
        3. Load experiment class dynamically
        4. Create and run experiment
        5. Snapshot config files
        6. Update job status with results

        Args:
            job: The Job object to execute
        """
        self.current_job = job
        print(f"[WORKER] Executing job: {job.job_id}")
        print(f"[WORKER]   Experiment: {job.experiment_class}")
        print(f"[WORKER]   User: {job.user}")

        # Create log directory for output capture
        log_dir = self.station.experiment_path / "logs"

        try:
            # Update station config from job's serialized config
            self._update_station_from_job_config(job.station_config)

            # Capture all output during experiment execution
            with OutputCapture(job.job_id, self.db, log_dir) as capture:
                # Store log path in job record
                self._update_job_log_path(job.job_id, str(capture.log_path))

                # Load experiment class dynamically
                ExptClass = self._load_experiment_class(job.experiment_module, job.experiment_class)

                # Parse experiment config
                expt_config = json.loads(job.experiment_config)

                # Run the experiment (all print output is captured)
                data_file_path, expt_pickle_path = self._run_experiment(ExptClass, expt_config, job)

            # Snapshot configs AFTER experiment runs, using the actual config that was used
            config_versions = self._snapshot_configs(job.job_id)

            # Update job as completed
            self._update_job_completed(
                job.job_id, str(data_file_path), str(expt_pickle_path), config_versions
            )

            print(f"[WORKER] Job completed: {job.job_id}")
            print(f"[WORKER]   Data saved to: {data_file_path}")
            print(f"[WORKER]   Expt object saved to: {expt_pickle_path}")

        except KeyboardInterrupt as e:
            # Job was cancelled by user via Ctrl+C
            error_msg = f"Job cancelled by user (Ctrl+C)"
            self._update_job_failed(job.job_id, error_msg)
            print(f"[WORKER] Job cancelled: {job.job_id}")
            # Reset interrupt count so user can Ctrl+C again for next job or to exit
            self._interrupt_count = 0

        except Exception as e:
            # Update job as failed
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._update_job_failed(job.job_id, error_msg)

            print(f"[WORKER] Job failed: {job.job_id}")
            print(f"[WORKER]   Error: {e}")

        finally:
            self.current_job = None

    def _snapshot_configs(self, job_id: str) -> dict:
        """
        Create versioned snapshots from station's in-memory configs.

        This snapshots the actual config that will be/was used during experiment
        execution (station.hardware_cfg), not the files on disk. This ensures
        that postprocessor changes are reflected in subsequent job configs.

        Args:
            job_id: The job ID to associate with these snapshots

        Returns:
            Dict mapping config type to version ID
        """
        with self.db.session() as session:
            versions = self.config_manager.snapshot_station_configs(
                station=self.station,
                session=session,
                job_id=job_id,
            )
            return versions

    def _load_experiment_class(self, module_path: str, class_name: str):
        """
        Dynamically load an experiment class with fresh code.

        Clears cached modules before importing to pick up any source code changes,
        similar to Jupyter's %autoreload magic.

        Args:
            module_path: Full module path (e.g., "multimode_expts.experiments.single_qubit.amplitude_rabi")
            class_name: Name of the class to load (e.g., "AmplitudeRabiExperiment")

        Returns:
            The experiment class
        """
        print(f"[WORKER] Loading {class_name} from {module_path}")

        # Clear cached modules to get fresh code (like autoreload)
        importlib.invalidate_caches()

        # Remove the experiment module and any modules that might have changed.
        # - multimode_expts.*: worker infrastructure (job_server, database, etc.)
        # - experiments.* submodules: experiment classes and base classes (MM_base, etc.)
        #
        # We preserve specific modules where objects persist in self.station:
        # - experiments.dataset: station.ds_storage is a StorageManSwapDataset instance
        # - experiments.station: the Station class itself
        # Clearing these would cause pickle identity errors (class object mismatch)
        preserved_modules = {'experiments', 'experiments.dataset', 'experiments.station'}
        modules_to_remove = [
            name for name in list(sys.modules.keys())
            if name == module_path
            or name.startswith('multimode_expts.')
            or (name.startswith('experiments.') and name not in preserved_modules)
        ]
        for name in modules_to_remove:
            del sys.modules[name]

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _run_experiment(self, ExptClass, expt_config: dict, job: Job) -> Tuple[Path, Path]:
        """
        Run an experiment using the loaded class.

        This follows the CharacterizationRunner pattern:
        1. Create experiment instance
        2. Set up configuration
        3. Call expt.go()
        4. Save expt object to pickle for client to load
        5. Return data file path and pickle path

        Args:
            ExptClass: The experiment class to instantiate
            expt_config: Experiment-specific configuration
            job: The job being executed

        Returns:
            Tuple of (data_file_path, expt_pickle_path)
        """
        # Generate data filename using job ID
        data_filename = IDGenerator.generate_data_filename(job.job_id, job.experiment_class)
        data_file_path = self.station.data_path / data_filename

        # Use expt_objs_path for pickle files if available, otherwise fall back to data_path
        expt_objs_path = getattr(self.station, 'expt_objs_path', None)
        if expt_objs_path is None:
            # Create expt_objs directory if station doesn't have it
            expt_objs_path = self.station.experiment_path / "expt_objs"
            expt_objs_path.mkdir(parents=True, exist_ok=True)
        expt_pickle_path = expt_objs_path / f"{job.job_id}_expt.pkl"

        print(f"[WORKER] Creating experiment instance")
        print(f"[WORKER]   Data file: {data_filename}")

        # Pass program info as tuple (module, class_name) if specified
        # The experiment will lazily load it when needed, avoiding pickle issues
        program = None
        if job.program_module and job.program_class:
            print(f"[WORKER]   Program: {job.program_class} from {job.program_module}")
            program = (job.program_module, job.program_class)

        # Create experiment instance
        # Note: In mock mode, this will use MockQickConfig
        if program is not None:
            expt = ExptClass(
                soccfg=self.station.soc,
                path=str(self.station.data_path),
                prefix=job.job_id,
                config_file=str(self.station.hardware_config_file),
                program=program,
            )
        else:
            expt = ExptClass(
                soccfg=self.station.soc,
                path=str(self.station.data_path),
                prefix=job.job_id,
                config_file=str(self.station.hardware_config_file),
            )

        # Setup configuration (following CharacterizationRunner pattern)
        expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))

        # Pass dataset objects from worker's station - required, never read from disk
        # (station datasets were already updated from serialized job config)
        expt.cfg.device.storage._ds_storage = self.station.ds_storage
        expt.cfg.device.storage._ds_floquet = self.station.ds_floquet

        expt.cfg.expt = AttrDict(expt_config)

        # Override filename to use job-based naming
        expt.fname = str(data_file_path)

        # Handle relax_delay if specified
        if hasattr(expt.cfg.expt, "relax_delay"):
            expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

        print(f"[WORKER] Running experiment...")

        if hasattr(expt.cfg.expt, "coupler_current"):
            coupler_current = expt.cfg.expt.coupler_current
            coupler_current_source = 'expt.cfg'
        else:
            coupler_current = self.station.hardware_cfg.hw.yoko_coupler.current
            coupler_current_source = 'hardware_cfg yaml'
        assert abs(coupler_current) < 5e-3, f"[WORKER] Coupler {coupler_current*1e3}mA sounds really high! Are you sure about the unit?"
        print(f"[WORKER] Setting coupler yoko current to {coupler_current*1e3}mA according to {coupler_current_source}...")
        self.station.yoko_coupler.ramp_current(coupler_current, sweeprate=1e-4)
        print("[WORKER] Done setting coupler current")

        # Run experiment
        # In mock mode, this will generate simulated data
        expt.go(
            analyze=True,
            display=False,  # Don't display in worker
            progress=True,
            save=True,
        )

        # Save expt object to pickle for client to load
        # This allows CharacterizationRunner postprocessors to work with the actual expt object
        print(f"[WORKER] Saving expt object to: {expt_pickle_path}")
        with open(expt_pickle_path, "wb") as f:
            pickle.dump(expt, f)

        return data_file_path, expt_pickle_path

    def _update_job_completed(
        self, job_id: str, data_file_path: str, expt_pickle_path: str, config_versions: dict
    ):
        """
        Update job status to completed.

        Args:
            job_id: The job ID
            data_file_path: Path to the saved data file
            expt_pickle_path: Path to the pickled expt object
            config_versions: Dict of config version IDs used
        """
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.data_file_path = data_file_path
                job.expt_pickle_path = expt_pickle_path

                # Link config versions
                job.hardware_config_version_id = config_versions.get("hardware_config")
                job.multiphoton_config_version_id = config_versions.get("multiphoton_config")
                job.floquet_storage_version_id = config_versions.get("floquet_storage_swap")
                job.man1_storage_version_id = config_versions.get("man1_storage_swap")

    def _update_job_failed(self, job_id: str, error_message: str):
        """
        Update job status to failed.

        Args:
            job_id: The job ID
            error_message: Error details
        """
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = error_message

    def _update_job_log_path(self, job_id: str, log_path: str):
        """
        Update job record with output log file path.

        Args:
            job_id: The job ID
            log_path: Path to the log file
        """
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.output_log_path = log_path

    def _cleanup_incomplete_jobs(self):
        """
        Mark any RUNNING jobs as FAILED on startup (crash recovery).

        If the worker crashes or is killed during job execution, jobs may be
        left in RUNNING state. This method cleans them up on restart.
        """
        with self.db.session() as session:
            running_jobs = session.query(Job).filter_by(status=JobStatus.RUNNING).all()
            for job in running_jobs:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = "Worker crashed or was restarted during execution"

                # Also mark output as complete
                output = session.query(JobOutput).filter_by(job_id=job.job_id).first()
                if output:
                    output.is_complete = True
                    output.output_text = (output.output_text or "") + "\n[WORKER CRASHED]"

            if running_jobs:
                print(f"[WORKER] Marked {len(running_jobs)} incomplete jobs as FAILED")


def main():
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(
        description="Job worker daemon for executing queued experiments"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (simulated hardware)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between database polls (default: 2.0)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for experiment session (default: auto-generated)",
    )

    args = parser.parse_args()

    # Acquire lock to prevent multiple workers
    lock = WorkerLock()
    try:
        lock.acquire()
        print(f"[WORKER] Lock acquired (PID {os.getpid()})")
    except RuntimeError as e:
        print(f"[WORKER] ERROR: {e}")
        sys.exit(1)

    worker = JobWorker(
        mock_mode=args.mock,
        poll_interval=args.poll_interval,
        experiment_name=args.experiment_name,
    )

    try:
        worker.run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()

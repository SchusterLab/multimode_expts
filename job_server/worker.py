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
import importlib
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimode_expts.job_server.database import get_database
from multimode_expts.job_server.models import Job, JobStatus
from multimode_expts.job_server.id_generator import IDGenerator
from multimode_expts.job_server.config_versioning import ConfigVersionManager
from slab.datamanagement import AttrDict
from copy import deepcopy


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

        # Initialize station (mock or real)
        self.station = None  # Lazy initialization

        # Initialize config version manager
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.config_manager = ConfigVersionManager(self.config_dir)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print(f"[WORKER] Initialized in {'MOCK' if mock_mode else 'REAL'} mode")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[WORKER] Received signal {signum}, shutting down after current job...")
        self.running = False

    def _initialize_station(self):
        """Initialize the station (lazy, on first job)."""
        if self.station is not None:
            return

        print(f"[WORKER] Initializing station: {self.experiment_name}")

        if self.mock_mode:
            from multimode_expts.job_server.mock_hardware import MockStation
            self.station = MockStation(experiment_name=self.experiment_name)
        else:
            from multimode_expts.experiments.station import MultimodeStation
            self.station = MultimodeStation(experiment_name=self.experiment_name)

    def run(self):
        """
        Main worker loop.

        Continuously polls for jobs and executes them until shutdown.
        """
        print(f"[WORKER] Starting main loop (poll interval: {self.poll_interval}s)")
        print("[WORKER] Press Ctrl+C to stop gracefully")

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
                job.started_at = datetime.now(timezone.utc)
                session.flush()

                # Detach job from session for use outside
                session.expunge(job)
                print(f"[WORKER] Claimed job: {job.job_id} ({job.experiment_class})")
                return job

        return None

    def _execute_job(self, job: Job):
        """
        Execute a single job.

        Steps:
        1. Initialize station if needed
        2. Snapshot config files
        3. Load experiment class dynamically
        4. Create and run experiment
        5. Update job status with results

        Args:
            job: The Job object to execute
        """
        self.current_job = job
        print(f"[WORKER] Executing job: {job.job_id}")
        print(f"[WORKER]   Experiment: {job.experiment_class}")
        print(f"[WORKER]   User: {job.user}")

        try:
            # Initialize station on first job
            self._initialize_station()

            # Snapshot configs
            config_versions = self._snapshot_configs(job.job_id)

            # Load experiment class dynamically
            ExptClass = self._load_experiment_class(job.experiment_module, job.experiment_class)

            # Parse experiment config
            expt_config = json.loads(job.experiment_config)

            # Run the experiment
            data_file_path = self._run_experiment(ExptClass, expt_config, job)

            # Update job as completed
            self._update_job_completed(job.job_id, str(data_file_path), config_versions)

            print(f"[WORKER] Job completed: {job.job_id}")
            print(f"[WORKER]   Data saved to: {data_file_path}")

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
        Create versioned snapshots of all config files.

        Args:
            job_id: The job ID to associate with these snapshots

        Returns:
            Dict mapping config type to version ID
        """
        with self.db.session() as session:
            versions = self.config_manager.snapshot_all_configs(
                hardware_config_path=self.station.hardware_config_file,
                multiphoton_config_path=getattr(self.station, 'multiphoton_config_file', None),
                floquet_csv_path=None,  # TODO: Add if needed
                man1_csv_path=self.config_dir / self.station.storage_man_file if hasattr(self.station, 'storage_man_file') else None,
                session=session,
                job_id=job_id,
            )
            return versions

    def _load_experiment_class(self, module_path: str, class_name: str):
        """
        Dynamically load an experiment class.

        Args:
            module_path: Full module path (e.g., "multimode_expts.experiments.single_qubit.amplitude_rabi")
            class_name: Name of the class to load (e.g., "AmplitudeRabiExperiment")

        Returns:
            The experiment class
        """
        print(f"[WORKER] Loading {class_name} from {module_path}")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _run_experiment(self, ExptClass, expt_config: dict, job: Job) -> Path:
        """
        Run an experiment using the loaded class.

        This follows the CharacterizationRunner pattern:
        1. Create experiment instance
        2. Set up configuration
        3. Call expt.go()
        4. Return data file path

        Args:
            ExptClass: The experiment class to instantiate
            expt_config: Experiment-specific configuration
            job: The job being executed

        Returns:
            Path to the saved data file
        """
        # Generate data filename using job ID
        data_filename = IDGenerator.generate_data_filename(job.job_id, job.experiment_class)
        data_file_path = self.station.data_path / data_filename

        print(f"[WORKER] Creating experiment instance")
        print(f"[WORKER]   Data file: {data_filename}")

        # Create experiment instance
        # Note: In mock mode, this will use MockQickConfig
        expt = ExptClass(
            soccfg=self.station.soc,
            path=str(self.station.data_path),
            prefix=job.job_id,
            config_file=str(self.station.hardware_config_file),
        )

        # Setup configuration (following CharacterizationRunner pattern)
        expt.cfg = AttrDict(deepcopy(self.station.config_thisrun))
        expt.cfg.expt = AttrDict(expt_config)

        # Override filename to use job-based naming
        expt.fname = str(data_file_path)

        # Handle relax_delay if specified
        if hasattr(expt.cfg.expt, "relax_delay"):
            expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

        print(f"[WORKER] Running experiment...")

        # Run experiment
        # In mock mode, this will generate simulated data
        expt.go(
            analyze=True,
            display=False,  # Don't display in worker
            progress=True,
            save=True,
        )

        return data_file_path

    def _update_job_completed(self, job_id: str, data_file_path: str, config_versions: dict):
        """
        Update job status to completed.

        Args:
            job_id: The job ID
            data_file_path: Path to the saved data file
            config_versions: Dict of config version IDs used
        """
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.data_file_path = data_file_path

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
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = error_message


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

    worker = JobWorker(
        mock_mode=args.mock,
        poll_interval=args.poll_interval,
        experiment_name=args.experiment_name,
    )

    worker.run()


if __name__ == "__main__":
    main()

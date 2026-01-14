"""
Client library for submitting and monitoring jobs.

This module provides a simple interface for users to:
- Submit experiment jobs to the queue
- Check job status
- Wait for job completion
- List queue state
- Cancel pending jobs

Usage in notebooks:
    from multimode_expts.job_server.client import JobClient

    client = JobClient()

    # Submit a job (all parameters are required)
    job_id = client.submit_job(
        experiment_class="AmplitudeRabiExperiment",
        experiment_module="multimode_expts.experiments.single_qubit.amplitude_rabi",
        expt_config={"start": 0, "step": 100, "expts": 50, "reps": 1000},
        user="connie"
    )

    # Wait for completion
    result = client.wait_for_completion(job_id)
    print(f"Data saved to: {result.data_file_path}")
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests


@dataclass
class JobResult:
    """Result of a job query."""

    job_id: str
    status: str
    user: Optional[str] = None
    experiment_class: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data_file_path: Optional[str] = None
    error_message: Optional[str] = None
    queue_position: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "JobResult":
        """Create JobResult from API response dict."""
        return cls(
            job_id=data.get("job_id"),
            status=data.get("status"),
            user=data.get("user"),
            experiment_class=data.get("experiment_class"),
            created_at=_parse_datetime(data.get("created_at")),
            started_at=_parse_datetime(data.get("started_at")),
            completed_at=_parse_datetime(data.get("completed_at")),
            data_file_path=data.get("data_file_path"),
            error_message=data.get("error_message"),
            queue_position=data.get("queue_position"),
        )

    def is_done(self) -> bool:
        """Check if job has finished (completed, failed, or cancelled)."""
        return self.status in ("completed", "failed", "cancelled")

    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "completed"


def _parse_datetime(value) -> Optional[datetime]:
    """Parse datetime from string or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Handle ISO format with or without timezone
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


class JobClient:
    """
    Client for interacting with the job queue server.

    Provides methods to submit jobs, check status, and wait for completion.
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize the job client.

        Args:
            server_url: URL of the job server (default: http://localhost:8000)
        """
        self.server_url = server_url.rstrip("/")

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the server."""
        url = f"{self.server_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response

    def health_check(self) -> dict:
        """
        Check if the server is healthy.

        Returns:
            Health status dict with database_connected, pending_jobs, running_jobs
        """
        response = self._request("GET", "/health")
        response.raise_for_status()
        return response.json()

    def submit_job(
        self,
        experiment_class: str,
        experiment_module: str,
        expt_config: Dict[str, Any],
        user: str,
        priority: int = 0,
    ) -> str:
        """
        Submit an experiment job to the queue.

        All parameters except priority are required and must be explicitly provided.

        Args:
            experiment_class: Name of experiment class (e.g., "AmplitudeRabiExperiment")
            experiment_module: Full module path (e.g., "multimode_expts.experiments.single_qubit.amplitude_rabi")
            expt_config: Experiment-specific configuration dict
            user: Username of submitter (required)
            priority: Job priority (higher = runs sooner, default 0)

        Returns:
            Unique job_id string

        Raises:
            ValueError: If any required parameter is missing or invalid

        Example:
            job_id = client.submit_job(
                experiment_class="AmplitudeRabiExperiment",
                experiment_module="multimode_expts.experiments.single_qubit.amplitude_rabi",
                expt_config={
                    "start": 0,
                    "step": 100,
                    "expts": 50,
                    "reps": 1000,
                    "rounds": 1,
                    "qubits": [0],
                },
                user="connie"
            )
        """
        # Validate required parameters
        if not experiment_class:
            raise ValueError("experiment_class is required")
        if not experiment_module:
            raise ValueError("experiment_module is required")
        if not expt_config:
            raise ValueError("expt_config is required")
        if not user:
            raise ValueError("user is required")
        if not isinstance(expt_config, dict):
            raise ValueError("expt_config must be a dict")

        response = self._request(
            "POST",
            "/jobs/submit",
            json={
                "experiment_class": experiment_class,
                "experiment_module": experiment_module,
                "expt_config": expt_config,
                "user": user,
                "priority": priority,
            },
        )
        response.raise_for_status()
        data = response.json()
        print(f"Job submitted: {data['job_id']} (queue position: {data.get('queue_position', '?')})")
        return data["job_id"]

    def get_status(self, job_id: str) -> JobResult:
        """
        Get the current status of a job.

        Args:
            job_id: The job ID to query

        Returns:
            JobResult with current status and details

        Raises:
            ValueError: If job_id is empty
        """
        if not job_id:
            raise ValueError("job_id is required")

        response = self._request("GET", f"/jobs/{job_id}")
        response.raise_for_status()
        return JobResult.from_dict(response.json())

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        verbose: bool = True,
    ) -> JobResult:
        """
        Wait for a job to complete.

        Polls the server until the job finishes (completed, failed, or cancelled).

        Args:
            job_id: The job ID to wait for (required)
            poll_interval: Seconds between status checks (default: 5.0)
            timeout: Maximum seconds to wait (None = wait forever)
            verbose: Print status updates while waiting

        Returns:
            Final JobResult

        Raises:
            ValueError: If job_id is empty
            TimeoutError: If timeout is exceeded
        """
        if not job_id:
            raise ValueError("job_id is required")

        start_time = time.time()
        last_status = None

        while True:
            result = self.get_status(job_id)

            # Print status changes
            if verbose and result.status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Job {job_id}: {result.status}")
                last_status = result.status

            # Check if done
            if result.is_done():
                if verbose:
                    if result.is_successful():
                        print(f"Job completed! Data: {result.data_file_path}")
                    else:
                        print(f"Job {result.status}: {result.error_message or 'No details'}")
                return result

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            time.sleep(poll_interval)

    def list_queue(self) -> dict:
        """
        List all pending and running jobs.

        Returns:
            Dict with 'pending_jobs', 'running_job', 'total_pending'
        """
        response = self._request("GET", "/jobs/queue")
        response.raise_for_status()
        return response.json()

    def print_queue(self):
        """Print the current queue status in a readable format."""
        queue = self.list_queue()

        print("\n=== Job Queue ===")

        if queue.get("running_job"):
            job = queue["running_job"]
            print(f"\nRunning: {job['job_id']}")
            print(f"  User: {job['user']}")
            print(f"  Experiment: {job['experiment_class']}")
            print(f"  Started: {job.get('started_at', 'Unknown')}")
        else:
            print("\nNo job currently running")

        print(f"\nPending: {queue['total_pending']} jobs")
        for i, job in enumerate(queue.get("pending_jobs", [])[:10], 1):
            print(f"  {i}. {job['job_id']} - {job['experiment_class']} (user: {job['user']}, priority: {job['priority']})")

        if queue["total_pending"] > 10:
            print(f"  ... and {queue['total_pending'] - 10} more")

        print()

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Only pending jobs can be cancelled. Running jobs cannot be interrupted.

        Args:
            job_id: The job ID to cancel (required)

        Returns:
            True if cancelled successfully

        Raises:
            ValueError: If job_id is empty
            requests.HTTPError: If job not found or cannot be cancelled
        """
        if not job_id:
            raise ValueError("job_id is required")

        response = self._request("DELETE", f"/jobs/{job_id}")
        response.raise_for_status()
        print(f"Job {job_id} cancelled")
        return True

    def get_history(
        self,
        limit: int = 50,
        user: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[dict]:
        """
        Get recent job history.

        Args:
            limit: Maximum number of jobs to return
            user: Filter by username (optional)
            status: Filter by status (optional)

        Returns:
            List of job dicts
        """
        params = {"limit": limit}
        if user:
            params["user"] = user
        if status:
            params["status"] = status

        response = self._request("GET", "/jobs/history", params=params)
        response.raise_for_status()
        return response.json()

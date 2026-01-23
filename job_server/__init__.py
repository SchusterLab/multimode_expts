"""
Job Server Package for Multi-User Experiment Scheduling

This package provides:
- JobClient: Client library for submitting and monitoring jobs
- Job server (FastAPI): Central job queue and ID management
- Job worker: Daemon that executes queued experiments

Usage:
    from job_server import JobClient

    client = JobClient()
    job_id = client.submit_job(
        experiment_class="AmplitudeRabiExperiment",
        experiment_module="experiments.single_qubit.amplitude_rabi",
        expt_config={"start": 0, "step": 100, "expts": 50}
    )
    result = client.wait_for_completion(job_id)
"""

from .client import JobClient

import sys

__all__ = ["JobClient"]

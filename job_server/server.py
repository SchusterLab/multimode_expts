"""
FastAPI server for the job queue system.

This server provides HTTP endpoints for:
- Submitting experiment jobs
- Checking job status
- Listing the job queue
- Cancelling pending jobs

Run with:
    cd /path/to/multimode_expts
    pixi run python -m uvicorn job_server.server:app --host 0.0.0.0 --port 8000

Or for development with auto-reload:
    pixi run python -m uvicorn job_server.server:app --reload --port 8000
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_database, Database
from .models import Job, JobStatus
from .id_generator import IDGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Multimode Experiment Job Server",
    description="Central job queue for multi-user experiment scheduling",
    version="1.0.0",
)

# Database instance (created on first request)
_db: Optional[Database] = None


def get_db() -> Session:
    """FastAPI dependency to get a database session."""
    global _db
    if _db is None:
        _db = get_database()
    session = _db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# Pydantic models for request/response validation
# ============================================================================


class JobSubmission(BaseModel):
    """Request body for submitting a new job."""

    experiment_class: str  # e.g., "AmplitudeRabiExperiment"
    experiment_module: str  # e.g., "multimode_expts.experiments.single_qubit.amplitude_rabi"
    expt_config: Dict[str, Any]  # The experiment-specific configuration
    station_config: str  # JSON-serialized station config (required)
    user: str  # Username of submitter
    priority: int = 0  # Higher priority = runs sooner (default 0)

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_class": "AmplitudeRabiExperiment",
                "experiment_module": "multimode_expts.experiments.single_qubit.amplitude_rabi",
                "expt_config": {
                    "start": 0,
                    "step": 100,
                    "expts": 50,
                    "reps": 1000,
                    "rounds": 1,
                    "qubits": [0],
                },
                "station_config": "{\"hardware_cfg\": {...}}",
                "user": "connie",
                "priority": 0,
            }
        }


class JobResponse(BaseModel):
    """Response after submitting a job."""

    job_id: str
    status: str
    created_at: datetime
    queue_position: Optional[int] = None

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    """Detailed job status response."""

    job_id: str
    user: str
    experiment_class: str
    status: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data_file_path: Optional[str] = None
    expt_pickle_path: Optional[str] = None
    error_message: Optional[str] = None
    hardware_config_version_id: Optional[str] = None
    multiphoton_config_version_id: Optional[str] = None
    man1_storage_version_id: Optional[str] = None
    floquet_storage_version_id: Optional[str] = None

    class Config:
        from_attributes = True


class QueueResponse(BaseModel):
    """Response for queue listing."""

    pending_jobs: List[JobStatusResponse]
    running_job: Optional[JobStatusResponse] = None
    total_pending: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database_connected: bool
    pending_jobs: int
    running_jobs: int


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multimode Experiment Job Server",
        "version": "1.0.0",
        "endpoints": {
            "submit": "POST /jobs/submit",
            "status": "GET /jobs/{job_id}",
            "queue": "GET /jobs/queue",
            "cancel": "DELETE /jobs/{job_id}",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(session: Session = Depends(get_db)):
    """
    Health check endpoint.

    Returns server status and job queue statistics.
    """
    try:
        pending_count = session.query(Job).filter_by(status=JobStatus.PENDING).count()
        running_count = session.query(Job).filter_by(status=JobStatus.RUNNING).count()

        return HealthResponse(
            status="healthy",
            database_connected=True,
            pending_jobs=pending_count,
            running_jobs=running_count,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            pending_jobs=0,
            running_jobs=0,
        )


@app.post("/jobs/submit", response_model=JobResponse, tags=["jobs"])
async def submit_job(submission: JobSubmission, session: Session = Depends(get_db)):
    """
    Submit a new experiment job to the queue.

    The job will be assigned a unique ID and queued for execution.
    Jobs are executed in priority order (higher priority first),
    then by submission time (FIFO for same priority).

    Returns the job_id which can be used to track the job's progress.
    """
    # Generate unique job ID
    job_id = IDGenerator.generate_job_id(session)

    # Create job record
    job = Job(
        job_id=job_id,
        user=submission.user,
        experiment_class=submission.experiment_class,
        experiment_module=submission.experiment_module,
        experiment_config=json.dumps(submission.expt_config),
        station_config=submission.station_config,
        status=JobStatus.PENDING,
        priority=submission.priority,
    )

    session.add(job)
    session.flush()

    # Calculate queue position
    queue_position = (
        session.query(Job)
        .filter(Job.status == JobStatus.PENDING)
        .filter(
            (Job.priority > submission.priority)
            | ((Job.priority == submission.priority) & (Job.created_at < job.created_at))
        )
        .count()
        + 1
    )

    print(f"[SERVER] Job submitted: {job_id} by {submission.user}")

    return JobResponse(
        job_id=job_id,
        status=job.status.value,
        created_at=job.created_at,
        queue_position=queue_position,
    )


@app.get("/jobs/queue", response_model=QueueResponse, tags=["jobs"])
async def list_queue(session: Session = Depends(get_db)):
    """
    List all pending and running jobs.

    Returns jobs sorted by priority (descending) then creation time (ascending).
    """
    # Get running job (should be at most one)
    running_job = session.query(Job).filter_by(status=JobStatus.RUNNING).first()

    # Get pending jobs
    pending_jobs = (
        session.query(Job)
        .filter_by(status=JobStatus.PENDING)
        .order_by(Job.priority.desc(), Job.created_at.asc())
        .all()
    )

    running_response = None
    if running_job:
        running_response = JobStatusResponse(
            job_id=running_job.job_id,
            user=running_job.user,
            experiment_class=running_job.experiment_class,
            status=running_job.status.value,
            priority=running_job.priority,
            created_at=running_job.created_at,
            started_at=running_job.started_at,
            completed_at=running_job.completed_at,
            data_file_path=running_job.data_file_path,
            expt_pickle_path=running_job.expt_pickle_path,
            error_message=running_job.error_message,
            hardware_config_version_id=running_job.hardware_config_version_id,
            multiphoton_config_version_id=running_job.multiphoton_config_version_id,
            man1_storage_version_id=running_job.man1_storage_version_id,
            floquet_storage_version_id=running_job.floquet_storage_version_id,
        )

    pending_responses = [
        JobStatusResponse(
            job_id=job.job_id,
            user=job.user,
            experiment_class=job.experiment_class,
            status=job.status.value,
            priority=job.priority,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            data_file_path=job.data_file_path,
            expt_pickle_path=job.expt_pickle_path,
            error_message=job.error_message,
            hardware_config_version_id=job.hardware_config_version_id,
            multiphoton_config_version_id=job.multiphoton_config_version_id,
            man1_storage_version_id=job.man1_storage_version_id,
            floquet_storage_version_id=job.floquet_storage_version_id,
        )
        for job in pending_jobs
    ]

    return QueueResponse(
        pending_jobs=pending_responses,
        running_job=running_response,
        total_pending=len(pending_jobs),
    )


@app.get("/jobs/history", tags=["jobs"])
async def get_job_history(
    limit: int = 50,
    user: Optional[str] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_db),
):
    """
    Get recent job history.

    Args:
        limit: Maximum number of jobs to return (default 50)
        user: Filter by username (optional)
        status: Filter by status (optional)

    Returns list of jobs sorted by creation time (newest first).
    """
    query = session.query(Job)

    if user:
        query = query.filter_by(user=user)

    if status:
        try:
            status_enum = JobStatus(status)
            query = query.filter_by(status=status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid values: {[s.value for s in JobStatus]}",
            )

    jobs = query.order_by(Job.created_at.desc()).limit(limit).all()

    return [
        {
            "job_id": job.job_id,
            "user": job.user,
            "experiment_class": job.experiment_class,
            "status": job.status.value,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "data_file_path": job.data_file_path,
        }
        for job in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, session: Session = Depends(get_db)):
    """
    Get the status of a specific job.

    Returns detailed information about the job including:
    - Current status (pending, running, completed, failed, cancelled)
    - Timestamps for creation, start, and completion
    - Path to data file (if completed)
    - Error message (if failed)
    """
    job = session.query(Job).filter_by(job_id=job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job.job_id,
        user=job.user,
        experiment_class=job.experiment_class,
        status=job.status.value,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        data_file_path=job.data_file_path,
        expt_pickle_path=job.expt_pickle_path,
        error_message=job.error_message,
        hardware_config_version_id=job.hardware_config_version_id,
        multiphoton_config_version_id=job.multiphoton_config_version_id,
        man1_storage_version_id=job.man1_storage_version_id,
        floquet_storage_version_id=job.floquet_storage_version_id,
    )


@app.delete("/jobs/{job_id}", tags=["jobs"])
async def cancel_job(job_id: str, session: Session = Depends(get_db)):
    """
    Cancel a pending job.

    Only pending jobs can be cancelled. Running jobs cannot be cancelled
    (they must complete or fail).
    """
    job = session.query(Job).filter_by(job_id=job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job {job_id}: status is {job.status.value} (only pending jobs can be cancelled)",
        )

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    session.flush()

    print(f"[SERVER] Job cancelled: {job_id}")

    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


# ============================================================================
# Main entry point (for running directly)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting Multimode Experiment Job Server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Database models for the job queue system.

Tables:
- Job: Tracks experiment jobs with status, config, and results
- ConfigVersion: Tracks versioned snapshots of config files
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class JobStatus(enum.Enum):
    """Status states for a job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfigType(enum.Enum):
    """Types of configuration files that can be versioned."""
    HARDWARE_CONFIG = "hardware_config"
    MULTIPHOTON_CONFIG = "multiphoton_config"
    FLOQUET_STORAGE_SWAP = "floquet_storage_swap"
    MAN1_STORAGE_SWAP = "man1_storage_swap"
    MM_DATASET_BASE = 'mm_dataset_base'


class Job(Base):
    """
    Represents an experiment job in the queue.

    Attributes:
        job_id: Unique identifier (format: JOB-YYYYMMDD-NNNN)
        user: Username of submitter (from SSH connection)
        experiment_class: Name of experiment class (e.g., "AmplitudeRabiExperiment")
        experiment_module: Python module path (e.g., "multimode_expts.experiments.single_qubit.amplitude_rabi")
        experiment_config: JSON-serialized experiment configuration
        hardware_config_version_id: Reference to config snapshot used
        multiphoton_config_version_id: Reference to multiphoton config snapshot
        floquet_storage_version_id: Reference to floquet CSV snapshot
        man1_storage_version_id: Reference to man1 storage CSV snapshot
        status: Current job status
        priority: Higher priority jobs run first (default 0)
        created_at: When job was submitted
        started_at: When job started executing
        completed_at: When job finished (success or failure)
        data_file_path: Path to resulting HDF5 data file
        error_message: Error details if job failed
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(20), unique=True, nullable=False, index=True)
    user = Column(String(100), nullable=False)
    experiment_class = Column(String(200), nullable=False)
    experiment_module = Column(String(300), nullable=False)
    experiment_config = Column(Text, nullable=False)  # JSON
    station_config = Column(Text, nullable=True)  # JSON-serialized station configs (hardware_cfg, etc.)

    # Optional program class for QsimBaseExperiment and similar
    program_class = Column(String(200), nullable=True)
    program_module = Column(String(300), nullable=True)

    # Config version references
    hardware_config_version_id = Column(String(50), ForeignKey("config_versions.version_id"), nullable=True)
    multiphoton_config_version_id = Column(String(50), ForeignKey("config_versions.version_id"), nullable=True)
    floquet_storage_version_id = Column(String(50), ForeignKey("config_versions.version_id"), nullable=True)
    man1_storage_version_id = Column(String(50), ForeignKey("config_versions.version_id"), nullable=True)

    status = Column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    priority = Column(Integer, default=0, index=True)

    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    data_file_path = Column(String(500), nullable=True)
    expt_pickle_path = Column(String(500), nullable=True)  # Path to pickled expt object
    output_log_path = Column(String(500), nullable=True)  # Path to job output log file
    error_message = Column(Text, nullable=True)

    # Relationships
    hardware_config_version = relationship(
        "ConfigVersion",
        foreign_keys=[hardware_config_version_id],
        backref="jobs_hardware"
    )
    multiphoton_config_version = relationship(
        "ConfigVersion",
        foreign_keys=[multiphoton_config_version_id],
        backref="jobs_multiphoton"
    )
    floquet_storage_version = relationship(
        "ConfigVersion",
        foreign_keys=[floquet_storage_version_id],
        backref="jobs_floquet"
    )
    man1_storage_version = relationship(
        "ConfigVersion",
        foreign_keys=[man1_storage_version_id],
        backref="jobs_man1"
    )

    def __repr__(self):
        return f"<Job({self.job_id}, {self.experiment_class}, {self.status.value})>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "user": self.user,
            "experiment_class": self.experiment_class,
            "experiment_module": self.experiment_module,
            "program_class": self.program_class,
            "program_module": self.program_module,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "data_file_path": self.data_file_path,
            "expt_pickle_path": self.expt_pickle_path,
            "error_message": self.error_message,
            "hardware_config_version_id": self.hardware_config_version_id,
            "multiphoton_config_version_id": self.multiphoton_config_version_id,
            "floquet_storage_version_id": self.floquet_storage_version_id,
            "man1_storage_version_id": self.man1_storage_version_id,
        }


class ConfigVersion(Base):
    """
    Tracks versioned snapshots of configuration files.

    Each time a job runs, the current config files are snapshotted
    so results can be reproduced later.

    Attributes:
        version_id: Unique identifier (format: CFG-{type}-YYYYMMDD-NNNN)
        config_type: Type of config (hardware, multiphoton, floquet_csv, man1_csv)
        original_filename: Original filename (e.g., "hardware_config_202505.yml")
        snapshot_path: Path to the versioned snapshot file
        checksum: SHA256 hash of file contents (for deduplication)
        created_at: When snapshot was created
        created_by_job_id: Job that triggered this snapshot (if any)
    """
    __tablename__ = "config_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(String(50), unique=True, nullable=False, index=True)
    config_type = Column(Enum(ConfigType), nullable=False, index=True)
    original_filename = Column(String(200), nullable=False)
    snapshot_path = Column(String(500), nullable=False)
    checksum = Column(String(64), nullable=True, index=True)  # SHA256
    created_at = Column(DateTime, default=datetime.now)
    created_by_job_id = Column(String(20), nullable=True)

    def __repr__(self):
        return f"<ConfigVersion({self.version_id}, {self.config_type.value})>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "version_id": self.version_id,
            "config_type": self.config_type.value,
            "original_filename": self.original_filename,
            "snapshot_path": self.snapshot_path,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by_job_id": self.created_by_job_id,
        }


class IDCounter(Base):
    """
    Persistent counter for ID generation.

    Stores the last used counter value for each ID type and date
    to ensure uniqueness across server restarts.
    """
    __tablename__ = "id_counters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prefix = Column(String(50), unique=True, nullable=False, index=True)  # e.g., "JOB-20260113"
    counter = Column(Integer, default=0, nullable=False)

    def __repr__(self):
        return f"<IDCounter({self.prefix}, {self.counter})>"


class MainConfig(Base):
    """
    Tracks the "main" (canonical, most up-to-date) version for each config type.

    This allows users to:
    - Pull the main config version to use with jobs
    - Push a new version to become the main one

    There is at most one record per ConfigType.
    """
    __tablename__ = "main_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_type = Column(Enum(ConfigType), unique=True, nullable=False, index=True)
    version_id = Column(String(50), ForeignKey("config_versions.version_id"), nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    updated_by = Column(String(100), nullable=True)  # Username who set this as main

    # Relationship to the actual version
    version = relationship("ConfigVersion", backref="main_config")

    def __repr__(self):
        return f"<MainConfig({self.config_type.value}, {self.version_id})>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "config_type": self.config_type.value,
            "version_id": self.version_id,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "updated_by": self.updated_by,
        }


class JobOutput(Base):
    """
    Stores output (stdout/stderr) from running jobs for streaming to clients.

    Each job has one output record that accumulates all output text.
    The output_text grows as the job runs, and is_complete is set when job finishes.

    Attributes:
        job_id: Reference to the job
        output_text: Accumulated stdout/stderr output
        line_count: Number of lines (for efficient offset queries)
        last_updated: When output was last updated
        is_complete: True when job finishes (no more output expected)
    """
    __tablename__ = "job_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(20), ForeignKey("jobs.job_id"), unique=True, nullable=False, index=True)
    output_text = Column(Text, default="")
    line_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_complete = Column(Boolean, default=False)

    # Relationship
    job = relationship("Job", backref="output")

    def __repr__(self):
        return f"<JobOutput({self.job_id}, lines={self.line_count}, complete={self.is_complete})>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "output_text": self.output_text,
            "line_count": self.line_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_complete": self.is_complete,
        }

"""
Centralized ID generation for jobs and config versions.

Generates unique IDs in the format:
- Jobs: JOB-YYYYMMDD-NNNNN (e.g., JOB-20260113-00042)
- Config versions: CFG-{TYPE}-YYYYMMDD-NNNNN (e.g., CFG-HW-20260113-00001)

IDs are:
- Thread-safe (uses database transactions for atomicity)
- Persistent (survives server restarts)
- Sortable by date and sequence number
"""

from datetime import datetime
from typing import Optional
import threading

from sqlalchemy.orm import Session

from .models import IDCounter, ConfigType


class IDGenerator:
    """
    Thread-safe ID generator using database-backed counters.

    The generator maintains daily counters that reset each day,
    ensuring IDs are both unique and sortable by date.
    """

    _lock = threading.Lock()

    @classmethod
    def generate_job_id(cls, session: Session) -> str:
        """
        Generate a unique job ID.

        Format: JOB-YYYYMMDD-NNNNN
        Example: JOB-20260113-00042

        Args:
            session: SQLAlchemy session for database access

        Returns:
            Unique job ID string
        """
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"JOB-{today}"
        counter = cls._get_next_counter(session, prefix)
        return f"{prefix}-{counter:05d}"

    @classmethod
    def generate_config_version_id(cls, config_type: ConfigType, session: Session) -> str:
        """
        Generate a unique config version ID.

        Format: CFG-{TYPE}-YYYYMMDD-NNNNN
        Example: CFG-HW-20260113-00001

        Args:
            config_type: Type of configuration being versioned
            session: SQLAlchemy session for database access

        Returns:
            Unique config version ID string
        """
        today = datetime.now().strftime("%Y%m%d")
        type_abbrev = cls._get_type_abbreviation(config_type)
        prefix = f"CFG-{type_abbrev}-{today}"
        counter = cls._get_next_counter(session, prefix)
        return f"{prefix}-{counter:05d}"

    @classmethod
    def generate_data_filename(cls, job_id: str, experiment_class: str) -> str:
        """
        Generate a data filename incorporating the job ID.

        Format: {job_id}_{ExperimentClass}.h5
        Example: JOB-20260113-00042_AmplitudeRabiExperiment.h5

        Args:
            job_id: The job's unique ID
            experiment_class: Name of the experiment class

        Returns:
            Filename for the HDF5 data file
        """
        return f"{job_id}_{experiment_class}.h5"

    @classmethod
    def _get_next_counter(cls, session: Session, prefix: str) -> int:
        """
        Get the next counter value for a prefix, creating if necessary.

        This method is thread-safe and uses database transactions
        to ensure uniqueness even with concurrent access.

        Args:
            session: SQLAlchemy session
            prefix: The ID prefix (e.g., "JOB-20260113")

        Returns:
            Next counter value (1-indexed)
        """
        with cls._lock:
            # Try to find existing counter
            counter_row = session.query(IDCounter).filter_by(prefix=prefix).first()

            if counter_row is None:
                # Create new counter starting at 1
                counter_row = IDCounter(prefix=prefix, counter=1)
                session.add(counter_row)
                session.flush()  # Ensure it's written
                return 1
            else:
                # Increment existing counter
                counter_row.counter += 1
                session.flush()
                return counter_row.counter

    @classmethod
    def _get_type_abbreviation(cls, config_type: ConfigType) -> str:
        """
        Get short abbreviation for config type.

        Args:
            config_type: The ConfigType enum value

        Returns:
            2-3 character abbreviation
        """
        abbreviations = {
            ConfigType.HARDWARE_CONFIG: "HW",
            ConfigType.MULTIPHOTON_CONFIG: "MP",
            ConfigType.FLOQUET_STORAGE_SWAP: "FL",
            ConfigType.MAN1_STORAGE_SWAP: "M1",
        }
        return abbreviations.get(config_type, "XX")

    @classmethod
    def get_current_counter(cls, session: Session, prefix: str) -> Optional[int]:
        """
        Get the current counter value without incrementing.

        Useful for debugging or displaying queue position.

        Args:
            session: SQLAlchemy session
            prefix: The ID prefix to query

        Returns:
            Current counter value, or None if prefix doesn't exist
        """
        counter_row = session.query(IDCounter).filter_by(prefix=prefix).first()
        return counter_row.counter if counter_row else None

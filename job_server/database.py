"""
Database connection and session management for the job queue.

Uses SQLite for simplicity - the database file is stored in the
multimode_expts/data directory by default.

Purpose:
- Creates and manages the SQLite database that stores job queue state
- Provides session management with automatic commit/rollback
- Shared by both the FastAPI server (to receive jobs) and worker (to execute them)
"""

from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from .models import Base


def _enable_sqlite_concurrency(dbapi_connection, connection_record):
    """Set WAL + busy_timeout on every new SQLite connection.

    WAL lets readers and one writer proceed without blocking each other;
    busy_timeout makes SQLite itself retry on lock for up to 10s before
    raising OperationalError. Together these eliminate the "database is
    locked" errors that hit when the server, worker output flush, and
    closed-loop service all touch jobs.db concurrently.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

# Default database path (relative to this file's location)
DEFAULT_DB_PATH = Path(__file__).parent / "jobs.db"


class Database:
    """
    Database connection manager.

    Usage:
        db = Database()
        db.create_tables()

        with db.session() as session:
            session.add(job)
            session.commit()
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Creates parent directories
                     if they don't exist. Defaults to data/jobs.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},  # Allow multi-thread access
            echo=False,  # Set to True for SQL debugging
        )
        event.listen(self.engine, "connect", _enable_sqlite_concurrency)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def create_tables(self):
        """Create all tables defined in models.py if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all tables. USE WITH CAUTION - destroys all data."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Automatically commits on success, rolls back on exception.

        Usage:
            with db.session() as session:
                job = session.query(Job).filter_by(job_id=job_id).first()
                job.status = JobStatus.RUNNING
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """
        Get a new session (caller responsible for closing).

        For FastAPI dependency injection:
            def get_db():
                db = Database()
                session = db.get_session()
                try:
                    yield session
                finally:
                    session.close()
        """
        return self.SessionLocal()


# Global database instance (initialized lazily)
_db_instance: Optional[Database] = None


def get_database(db_path: Optional[Path] = None) -> Database:
    """
    Get the global database instance, creating it if necessary.

    Args:
        db_path: Path to database file (only used on first call)

    Returns:
        Database instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
        _db_instance.create_tables()
    return _db_instance


def reset_database(db_path: Optional[Path] = None):
    """
    Reset the global database instance.

    Useful for testing or when switching databases.
    """
    global _db_instance
    _db_instance = None
    if db_path:
        get_database(db_path)

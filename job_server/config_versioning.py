"""
Configuration versioning for reproducible experiment runs.

This module manages immutable snapshots of configuration files,
allowing any job's results to be reproduced by loading the exact
config state that was used when the job ran.

Versioned file types:
- hardware_config: YAML hardware configuration
- multiphoton_config: YAML multiphoton pulse configuration
- floquet_storage_swap: CSV floquet calibration data
- man1_storage_swap: CSV storage-manipulate swap calibration data

Usage:
    from job_server.database import get_database
    from job_server.config_versioning import ConfigVersionManager

    db = get_database()
    manager = ConfigVersionManager(config_dir=Path("configs"))

    with db.session() as session:
        # Snapshot a single config
        version_id, snapshot_path = manager.snapshot_hardware_config(
            source_path=Path("configs/hardware_config_202505.yml"),
            session=session,
            job_id="JOB-20260113-00001"
        )

        # Or snapshot all configs at once
        versions = manager.snapshot_all_configs(
            hardware_config_path=Path("configs/hardware_config_202505.yml"),
            multiphoton_config_path=Path("configs/multiphoton_config.yml"),
            floquet_csv_path=None,
            man1_csv_path=Path("configs/man1_storage_swap_dataset.csv"),
            session=session,
            job_id="JOB-20260113-00001"
        )
"""

import hashlib
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict

from sqlalchemy.orm import Session

from .models import ConfigVersion, ConfigType, MainConfig
from .id_generator import IDGenerator


class ConfigVersionManager:
    """
    Manages versioned snapshots of configuration files.

    Creates immutable copies of config files tied to job submissions,
    enabling full reproducibility of experiment results.

    Version directories:
        configs/versions/hardware_config/
        configs/versions/multiphoton_config/
        configs/versions/floquet_storage_swap/
        configs/versions/man1_storage_swap/
    """

    # Mapping from ConfigType to directory name
    TYPE_TO_DIR = {
        ConfigType.HARDWARE_CONFIG: "hardware_config",
        ConfigType.MULTIPHOTON_CONFIG: "multiphoton_config",
        ConfigType.FLOQUET_STORAGE_SWAP: "floquet_storage_swap",
        ConfigType.MAN1_STORAGE_SWAP: "man1_storage_swap",
    }

    def __init__(self, config_dir: Path):
        """
        Initialize the config version manager.

        Args:
            config_dir: Path to the configs directory (contains versions/ subdir)
        """
        self.config_dir = Path(config_dir)
        self.version_base_dir = self.config_dir / "versions"

        # Ensure version directories exist
        for dir_name in self.TYPE_TO_DIR.values():
            (self.version_base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _get_version_dir(self, config_type: ConfigType) -> Path:
        """Get the version directory for a config type."""
        return self.version_base_dir / self.TYPE_TO_DIR[config_type]

    def snapshot_config(
        self,
        source_path: Path,
        config_type: ConfigType,
        session: Session,
        job_id: Optional[str] = None,
    ) -> Tuple[str, Path]:
        """
        Create a versioned snapshot of a configuration file.

        If an identical file already exists (same checksum), returns
        the existing version instead of creating a duplicate.

        Args:
            source_path: Path to the source config file
            config_type: Type of configuration being snapshotted
            session: SQLAlchemy session for database access
            job_id: Optional job ID that triggered this snapshot

        Returns:
            Tuple of (version_id, snapshot_path)
        """
        # Compute checksum of source file
        checksum = self._compute_checksum(source_path)

        # Check if identical version already exists
        existing = self._find_by_checksum(checksum, config_type, session)
        if existing:
            print(f"[CONFIG] Reusing existing version {existing.version_id} (same checksum)")
            return existing.version_id, Path(existing.snapshot_path)

        # Generate new version ID
        version_id = IDGenerator.generate_config_version_id(config_type, session)

        # Determine snapshot filename and path
        version_dir = self._get_version_dir(config_type)
        original_name = source_path.name
        snapshot_name = f"{version_id}_{original_name}"
        snapshot_path = version_dir / snapshot_name

        # Copy file to version directory
        shutil.copy2(source_path, snapshot_path)
        print(f"[CONFIG] Created snapshot: {snapshot_path}")

        # Record in database
        version_record = ConfigVersion(
            version_id=version_id,
            config_type=config_type,
            original_filename=original_name,
            snapshot_path=str(snapshot_path),
            checksum=checksum,
            created_by_job_id=job_id,
        )
        session.add(version_record)
        session.flush()

        return version_id, snapshot_path

    def snapshot_hardware_config(
        self, source_path: Path, session: Session, job_id: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Convenience method to snapshot hardware config."""
        return self.snapshot_config(
            source_path, ConfigType.HARDWARE_CONFIG, session, job_id
        )

    def snapshot_multiphoton_config(
        self, source_path: Path, session: Session, job_id: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Convenience method to snapshot multiphoton config."""
        return self.snapshot_config(
            source_path, ConfigType.MULTIPHOTON_CONFIG, session, job_id
        )

    def snapshot_floquet_csv(
        self, source_path: Path, session: Session, job_id: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Convenience method to snapshot floquet storage swap CSV."""
        return self.snapshot_config(
            source_path, ConfigType.FLOQUET_STORAGE_SWAP, session, job_id
        )

    def snapshot_man1_csv(
        self, source_path: Path, session: Session, job_id: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Convenience method to snapshot man1 storage swap CSV."""
        return self.snapshot_config(
            source_path, ConfigType.MAN1_STORAGE_SWAP, session, job_id
        )

    def snapshot_all_configs(
        self,
        hardware_config_path: Path,
        multiphoton_config_path: Optional[Path],
        floquet_csv_path: Optional[Path],
        man1_csv_path: Optional[Path],
        session: Session,
        job_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Snapshot all configuration files for a job.

        Args:
            hardware_config_path: Path to hardware config YAML
            multiphoton_config_path: Path to multiphoton config YAML (optional)
            floquet_csv_path: Path to floquet CSV (optional)
            man1_csv_path: Path to man1 CSV (optional)
            session: SQLAlchemy session
            job_id: Job ID that triggered these snapshots

        Returns:
            Dict mapping config type to version ID
        """
        versions = {}

        # Hardware config is required
        hw_version, _ = self.snapshot_hardware_config(hardware_config_path, session, job_id)
        versions["hardware_config"] = hw_version

        # Optional configs
        if multiphoton_config_path and multiphoton_config_path.exists():
            mp_version, _ = self.snapshot_multiphoton_config(multiphoton_config_path, session, job_id)
            versions["multiphoton_config"] = mp_version

        if floquet_csv_path and floquet_csv_path.exists():
            fl_version, _ = self.snapshot_floquet_csv(floquet_csv_path, session, job_id)
            versions["floquet_storage_swap"] = fl_version

        if man1_csv_path and man1_csv_path.exists():
            m1_version, _ = self.snapshot_man1_csv(man1_csv_path, session, job_id)
            versions["man1_storage_swap"] = m1_version

        return versions

    def get_config_path(self, version_id: str, session: Session) -> Optional[Path]:
        """
        Get the path to a versioned config file.

        Args:
            version_id: The config version ID
            session: SQLAlchemy session

        Returns:
            Path to the snapshot file, or None if not found
        """
        version = session.query(ConfigVersion).filter_by(version_id=version_id).first()
        if version:
            return Path(version.snapshot_path)
        return None

    def get_config_for_job(
        self, job_id: str, config_type: ConfigType, session: Session
    ) -> Optional[Path]:
        """
        Get the config snapshot path used for a specific job.

        Args:
            job_id: The job ID
            config_type: Type of config to retrieve
            session: SQLAlchemy session

        Returns:
            Path to the snapshot file, or None if not found
        """
        version = (
            session.query(ConfigVersion)
            .filter_by(created_by_job_id=job_id, config_type=config_type)
            .first()
        )
        if version:
            return Path(version.snapshot_path)
        return None

    def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex string of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _find_by_checksum(
        self, checksum: str, config_type: ConfigType, session: Session
    ) -> Optional[ConfigVersion]:
        """
        Find an existing config version with matching checksum.

        Args:
            checksum: SHA256 checksum to match
            config_type: Type of config
            session: SQLAlchemy session

        Returns:
            ConfigVersion if found, None otherwise
        """
        return (
            session.query(ConfigVersion)
            .filter_by(checksum=checksum, config_type=config_type)
            .first()
        )

    def list_versions(
        self, config_type: Optional[ConfigType] = None, session: Session = None
    ) -> list:
        """
        List all config versions, optionally filtered by type.

        Args:
            config_type: Optional filter by config type
            session: SQLAlchemy session

        Returns:
            List of ConfigVersion records
        """
        query = session.query(ConfigVersion)
        if config_type:
            query = query.filter_by(config_type=config_type)
        return query.order_by(ConfigVersion.created_at.desc()).all()

    # =========================================================================
    # Main Config (Pull/Push) Functions
    # =========================================================================

    def get_main_version(
        self, config_type: ConfigType, session: Session
    ) -> Optional[ConfigVersion]:
        """
        Pull the main (canonical, most up-to-date) config version for a type.

        Args:
            config_type: Type of config to retrieve
            session: SQLAlchemy session

        Returns:
            ConfigVersion record if a main version is set, None otherwise
        """
        main_config = (
            session.query(MainConfig)
            .filter_by(config_type=config_type)
            .first()
        )
        if main_config:
            return main_config.version
        return None

    def get_main_config_path(
        self, config_type: ConfigType, session: Session
    ) -> Optional[Path]:
        """
        Pull the path to the main config file for a type.

        Args:
            config_type: Type of config to retrieve
            session: SQLAlchemy session

        Returns:
            Path to the main config snapshot, or None if not set
        """
        version = self.get_main_version(config_type, session)
        if version:
            return Path(version.snapshot_path)
        return None

    def set_main_version(
        self,
        config_type: ConfigType,
        version_id: str,
        session: Session,
        updated_by: Optional[str] = None,
    ) -> MainConfig:
        """
        Set an existing version as the main config for a type.

        Args:
            config_type: Type of config
            version_id: ID of the version to set as main
            session: SQLAlchemy session
            updated_by: Optional username who made this change

        Returns:
            The MainConfig record

        Raises:
            ValueError: If the version_id doesn't exist
        """
        # Verify the version exists
        version = session.query(ConfigVersion).filter_by(version_id=version_id).first()
        if not version:
            raise ValueError(f"Config version {version_id} not found")

        if version.config_type != config_type:
            raise ValueError(
                f"Version {version_id} is type {version.config_type.value}, "
                f"expected {config_type.value}"
            )

        # Find or create main config record
        main_config = (
            session.query(MainConfig)
            .filter_by(config_type=config_type)
            .first()
        )

        if main_config:
            main_config.version_id = version_id
            main_config.updated_by = updated_by
            print(f"[CONFIG] Updated main {config_type.value} to {version_id}")
        else:
            main_config = MainConfig(
                config_type=config_type,
                version_id=version_id,
                updated_by=updated_by,
            )
            session.add(main_config)
            print(f"[CONFIG] Set main {config_type.value} to {version_id}")

        session.flush()
        return main_config

    def push_to_main(
        self,
        source_path: Path,
        config_type: ConfigType,
        session: Session,
        updated_by: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Tuple[str, Path]:
        """
        Push a new config file: create a snapshot and set it as main.

        This is a convenience method that combines snapshot_config and
        set_main_version in one operation.

        Args:
            source_path: Path to the source config file
            config_type: Type of configuration
            session: SQLAlchemy session
            updated_by: Optional username who is pushing this config
            job_id: Optional job ID that triggered this push

        Returns:
            Tuple of (version_id, snapshot_path)
        """
        # Create snapshot
        version_id, snapshot_path = self.snapshot_config(
            source_path, config_type, session, job_id
        )

        # Set as main
        self.set_main_version(config_type, version_id, session, updated_by)

        return version_id, snapshot_path

    def push_hardware_config_to_main(
        self, source_path: Path, session: Session, updated_by: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Push hardware config and set as main."""
        return self.push_to_main(
            source_path, ConfigType.HARDWARE_CONFIG, session, updated_by
        )

    def push_multiphoton_config_to_main(
        self, source_path: Path, session: Session, updated_by: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Push multiphoton config and set as main."""
        return self.push_to_main(
            source_path, ConfigType.MULTIPHOTON_CONFIG, session, updated_by
        )

    def push_floquet_csv_to_main(
        self, source_path: Path, session: Session, updated_by: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Push floquet CSV and set as main."""
        return self.push_to_main(
            source_path, ConfigType.FLOQUET_STORAGE_SWAP, session, updated_by
        )

    def push_man1_csv_to_main(
        self, source_path: Path, session: Session, updated_by: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Push man1 CSV and set as main."""
        return self.push_to_main(
            source_path, ConfigType.MAN1_STORAGE_SWAP, session, updated_by
        )

    def get_all_main_configs(self, session: Session) -> Dict[str, Optional[str]]:
        """
        Get the main version IDs for all config types.

        Args:
            session: SQLAlchemy session

        Returns:
            Dict mapping config type name to version ID (or None if not set)
        """
        result = {ct.value: None for ct in ConfigType}

        main_configs = session.query(MainConfig).all()
        for main_config in main_configs:
            result[main_config.config_type.value] = main_config.version_id

        return result

    def get_all_main_config_paths(self, session: Session) -> Dict[str, Optional[Path]]:
        """
        Get the paths to all main config files.

        Args:
            session: SQLAlchemy session

        Returns:
            Dict mapping config type name to path (or None if not set)
        """
        result = {ct.value: None for ct in ConfigType}

        for config_type in ConfigType:
            path = self.get_main_config_path(config_type, session)
            result[config_type.value] = path

        return result

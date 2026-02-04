"""
Initialize a new jobs.db database with main config versions.

This script is used when setting up the job server on a new machine. It:
1. Creates the SQLite database (jobs.db) if it doesn't exist
2. Snapshots the default config files from configs/
3. Sets these snapshots as the "main" versions in the database

The worker requires main config versions to be set in order to bootstrap
its station. Without this initialization, `pixi run worker` will fail with
an error about missing main config.

Note: jobs.db and configs/versions/ are gitignored, so each machine needs
to run this setup independently.

Usage:
    cd /path/to/multimode_expts
    pixi run python -m job_server.setup_new_db

    # Or with custom config directory:
    pixi run python -m job_server.setup_new_db --config-dir /path/to/configs
"""

import argparse
from pathlib import Path

from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager


def setup_main_configs(config_dir: Path, updated_by: str = "setup_new_db"):
    """
    Initialize database with main config versions from default YAML/CSV files.

    Args:
        config_dir: Path to the configs directory containing default config files
        updated_by: Username to record as the creator of these versions
    """
    db = get_database()
    manager = ConfigVersionManager(config_dir)

    # Default config filenames
    hardware_config = config_dir / "hardware_config.yml"
    multiphoton_config = config_dir / "multiphoton_config.yml"
    man1_storage = config_dir / "man1_storage_swap_dataset.csv"
    floquet_storage = config_dir / "floquet_storage_swap_dataset.csv"

    with db.session() as session:
        # Hardware config
        if hardware_config.exists():
            version_id, path = manager.push_hardware_config_to_main(
                hardware_config, session, updated_by
            )
            print(f"Hardware config:    {version_id}")
        else:
            print(f"WARNING: {hardware_config} not found, skipping")

        # Multiphoton config
        if multiphoton_config.exists():
            version_id, path = manager.push_multiphoton_config_to_main(
                multiphoton_config, session, updated_by
            )
            print(f"Multiphoton config: {version_id}")
        else:
            print(f"WARNING: {multiphoton_config} not found, skipping")

        # Man1 storage swap CSV
        if man1_storage.exists():
            version_id, path = manager.push_man1_csv_to_main(
                man1_storage, session, updated_by
            )
            print(f"Man1 storage:       {version_id}")
        else:
            print(f"WARNING: {man1_storage} not found, skipping")

        # Floquet storage swap CSV
        if floquet_storage.exists():
            version_id, path = manager.push_floquet_csv_to_main(
                floquet_storage, session, updated_by
            )
            print(f"Floquet storage:    {version_id}")
        else:
            print(f"WARNING: {floquet_storage} not found, skipping")

    print("\nDatabase initialized! You can now run: pixi run worker")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize jobs.db with main config versions"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent.parent / "configs",
        help="Path to configs directory (default: ../configs relative to this script)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="setup_new_db",
        help="Username to record as creator (default: setup_new_db)",
    )

    args = parser.parse_args()

    print(f"Initializing database with configs from: {args.config_dir}")
    print("-" * 60)

    setup_main_configs(args.config_dir, args.user)


if __name__ == "__main__":
    main()

"""A linked jobs.db sees the writes still held in the real file's WAL.

Worktrees reach the measurement PC's jobs.db through a file link. SQLite puts
the WAL beside the path it opens, so a connection through an unresolved link
reads the main file without the main WAL: rows not yet checkpointed are
missing, and the b-tree can look "malformed". `Database` resolves the path.

Run:  pixi run python -m pytest tests/test_database_symlink.py -v
"""
import os

import pytest
from sqlalchemy import text

from job_server.database import Database


def _make_link(link, target):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"cannot create a file link here: {error}")


def test_linked_database_sees_uncheckpointed_writes(tmp_path):
    real_dir = tmp_path / "main"
    link_dir = tmp_path / "worktree"
    real_dir.mkdir()
    link_dir.mkdir()
    real = real_dir / "jobs.db"

    writer = Database(real)
    with writer.engine.connect() as conn:
        conn.execute(text("PRAGMA wal_autocheckpoint=0"))   # keep rows in the WAL
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
        conn.execute(text("INSERT INTO t VALUES (1), (2), (3)"))
        conn.commit()
        assert (real_dir / "jobs.db-wal").stat().st_size > 0

        link = link_dir / "jobs.db"
        _make_link(link, real)
        reader = Database(link)
        with reader.engine.connect() as other:
            count = other.execute(text("SELECT count(*) FROM t")).scalar()
        reader.engine.dispose()

    writer.engine.dispose()
    assert count == 3
    assert not (link_dir / "jobs.db-wal").exists(), "a second WAL beside the link"

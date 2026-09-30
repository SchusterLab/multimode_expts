"""One-time, read-only export of job provenance from jobs.db to a JSON sidecar.

Spec section 3.2: raw HDF5 files written before the stamping work carry only
their ``config`` attribute -- no job ID, class name or configuration version
IDs. Those references exist, but only in the live job database on the
acquisition workstation, which offline analysis must not depend on.

Rather than rewriting already-acquired raw files, export the mapping once and
carry it alongside. Analysis then reads HDF5 plus this sidecar plus the
versions archive, and never touches the database.

Read-only by construction: the live worker holds the same database open in WAL
mode, so the connection uses ``mode=ro`` and any write raises.

Usage::

    pixi run python tools/export_job_provenance.py JOB-20260815-00009 ... -o out.json
    pixi run python tools/export_job_provenance.py --range 20260723 48 85 -o out.json

TODO(spec 3.1): once the persistence layer stamps these into new HDF5 files,
this export is only needed for pre-stamp data.
"""

import argparse
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("C:/python/multimode_expts/job_server/jobs.db")

FIELDS = (
    "job_id",
    "experiment_class",
    "experiment_module",
    "program_class",
    "program_module",
    "hardware_config_version_id",
    "multiphoton_config_version_id",
    "floquet_storage_version_id",
    "man1_storage_version_id",
    "status",
    "created_at",
    "data_file_path",
)


def read_only(db_path: Path) -> sqlite3.Connection:
    uri = "file:///" + str(db_path).replace("\\", "/").lstrip("/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def export(job_ids, db_path=DEFAULT_DB):
    """-> {job_id: {field: value}} for the jobs that exist, in input order."""
    conn = read_only(Path(db_path))
    out = {}
    # Chunked so a large ID list cannot exceed SQLite's variable limit.
    for start in range(0, len(job_ids), 500):
        chunk = job_ids[start:start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM jobs WHERE job_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            record = dict(zip(FIELDS, row))
            out[record["job_id"]] = record
    return {j: out[j] for j in job_ids if j in out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_ids", nargs="*", help="explicit JOB-YYYYMMDD-NNNNN ids")
    parser.add_argument("--range", nargs=3, action="append", metavar=("DATE", "FIRST", "LAST"),
                        help="expand an inclusive job-number range; repeatable")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    ids = list(args.job_ids)
    for date, first, last in args.range or []:
        ids += [f"JOB-{date}-{n:05d}"
                for n in range(int(first), int(last) + 1, args.step)]

    records = export(ids, args.db)
    missing = [j for j in ids if j not in records]

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=1, sort_keys=True))

    print(f"exported {len(records)} of {len(ids)} requested -> {out} "
          f"({len(existing)} total)")
    if missing:
        print(f"  absent from the database: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""))


if __name__ == "__main__":
    main()

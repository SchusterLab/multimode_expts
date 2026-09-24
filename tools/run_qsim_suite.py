"""Run the qsim migration notebooks top to bottom, as the refactor's test suite.

    pixi run python tools/run_qsim_suite.py --hardware
    pixi run python tools/run_qsim_suite.py --hardware --only mbr
    pixi run python tools/run_qsim_suite.py --suite analysis
    pixi run python tools/run_qsim_suite.py --list

Each notebook is executed in a fresh kernel and saved as an executed
``.ipynb`` with its plots, under ``--out``. Judge a run by opening those;
this script only reports which notebooks raised, and where.

The measurement notebooks are a suite for hardware runs only. What mock mode
can check is checked by pytest instead (``tests/test_qsim_notebook_mock_acquisition.py``,
``tests/test_mbr_acquire_mock.py``), and ``tests/test_qsim_notebooks_static.py``
checks every notebook's names and imports. See docs/qsim/mock_suite_plan.md.

This script passes its settings to the kernels through the environment; see
``experiments/qsim/notebook_helpers/run_mode.py``. Every runner executes
directly (``use_queue`` off), because the queue worker runs only the main
checkout. ``--profile`` defaults to ``smoke``.

Cell tags (jupytext: ``# %% tags=["suite-skip"]``):

- ``suite-skip``: never run by the suite. Alternatives to another cell ("run
  this instead"), and cells that need a person's judgement first.
- ``raises-exception`` (nbclient's own tag) marks a known failure: the cell
  may raise and the run continues. The summary counts these as xfail, and a
  tagged cell that ran clean as xpass, so a fixed bug shows up.

The measurement suite drives the real device from this checkout, so it runs
only with ``--hardware``. Before it starts, it takes the main checkout's worker lock: it refuses if a worker is
running, and while it holds the lock no worker can start. Tell the other
users first.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MEASUREMENT_DIR = REPO_ROOT / "measurement_notebooks" / "202609_qsim_migration"
ANALYSIS_DIR = REPO_ROOT / "analysis_notebooks" / "202609_qsim_migration"

# Each notebook opens its own station from its own config versions, so no
# state passes between them. Calibration still goes first: a failure there
# is the cheapest to see.
SUITES = {
    "measurement": [
        MEASUREMENT_DIR / f"{name}.py"
        for name in (
            "multiphoton_calibration",
            "floquet_calibration",
            "mbr",
            "mbr_tomography",
            "mbr_sff",
            "mbr_disorder",
            "floquet_displacement_kerr",
        )
    ],
    "analysis": [
        ANALYSIS_DIR / f"{name}.py"
        for name in ("mbr", "mbr_disorder", "mbr_sampling", "mbr_spectral_validation")
    ],
}

MAIN_WORKER_LOCK_VAR = "MULTIMODE_MAIN_WORKER_LOCK"
DEFAULT_MAIN_WORKER_LOCK = "C:/python/multimode_expts/job_server/worker.lock"


SKIPPED_TAGS = {"suite-skip"}


def load_notebook(path, skip):
    """-> (notebook, number of cells dropped) with tagged cells removed."""
    import jupytext

    nb = jupytext.read(path)
    kept = [c for c in nb.cells if not skip & set(c.metadata.get("tags", []))]
    dropped = len(nb.cells) - len(kept)
    nb.cells = kept
    return nb, dropped


def first_line(cell):
    for line in cell.source.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:70]
    return cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""


def expected_failures(nb):
    """-> (xfail, xpass): indices of raises-exception cells that raised / did not."""
    xfail, xpass = [], []
    for index, cell in enumerate(nb.cells):
        if "raises-exception" not in cell.metadata.get("tags", []):
            continue
        raised = any(o.get("output_type") == "error" for o in cell.get("outputs", []))
        (xfail if raised else xpass).append(index)
    return xfail, xpass


def run_notebook(path, out_dir, skip, cell_timeout):
    """Execute one notebook; -> (ok, seconds, error summary or None, xfail, xpass)."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb, dropped = load_notebook(path, skip)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"\n=== {path.relative_to(REPO_ROOT)}: {len(code_cells)} code cells"
          f" ({dropped} tagged cells skipped)")

    started = {}

    def on_start(cell, cell_index, **_):
        started[cell_index] = time.monotonic()
        print(f"  [{cell_index:3d}] {first_line(cell)}", flush=True)

    def on_done(cell, cell_index, **_):
        took = time.monotonic() - started.get(cell_index, time.monotonic())
        if took > 5:
            print(f"        ({took:.0f} s)", flush=True)

    client = NotebookClient(
        nb,
        timeout=cell_timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_execute=on_start,
        on_cell_executed=on_done,
    )
    t0 = time.monotonic()
    error = None
    try:
        client.execute()
    except CellExecutionError as exc:
        error = str(exc).strip().splitlines()[-1]
    except Exception as exc:  # timeouts, dead kernels
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.parent.parent.name}__{path.stem}.ipynb"
    nbformat.write(nb, out)
    print(f"  -> {out}")
    return (error is None, seconds, error, *expected_failures(nb))


def hold_worker_lock():
    """Take the main checkout's worker lock, or exit if a worker holds it."""
    from job_server.worker import WorkerLock

    lock_path = Path(os.environ.get(MAIN_WORKER_LOCK_VAR, DEFAULT_MAIN_WORKER_LOCK))
    if not lock_path.parent.is_dir():
        sys.exit(f"main checkout's job_server/ not found at {lock_path.parent}; "
                 f"set {MAIN_WORKER_LOCK_VAR}")
    lock = WorkerLock(lock_file=lock_path)
    try:
        lock.acquire()
    except RuntimeError as exc:
        sys.exit(f"{exc}\nStop the worker before a hardware run.")
    print(f"holding {lock_path}: no worker can start until this run ends")
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default="measurement")
    parser.add_argument("--hardware", action="store_true",
                        help="required for the measurement suite: it drives the real device")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--configs", default=None,
                        help="'main', a JSON file of version IDs, or unset for each "
                             "notebook's own config_dict")
    parser.add_argument("--only", default=None,
                        help="comma-separated notebook names, e.g. mbr,mbr_sff")
    parser.add_argument("--keep-going", action="store_true",
                        help="run the remaining notebooks after one fails")
    parser.add_argument("--cell-timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: .suite_runs/<timestamp>_<suite>_<mode>/")
    parser.add_argument("--list", action="store_true", help="list notebooks and exit")
    args = parser.parse_args(argv)

    notebooks = SUITES[args.suite]
    if args.only:
        wanted = [name.strip() for name in args.only.split(",")]
        by_name = {path.stem: path for path in notebooks}
        unknown = sorted(set(wanted) - set(by_name))
        if unknown:
            sys.exit(f"unknown notebooks {unknown}; this suite has {sorted(by_name)}")
        notebooks = [by_name[name] for name in wanted]
    if args.list:
        for path in notebooks:
            print(path.relative_to(REPO_ROOT))
        return 0

    mode = "hardware" if args.suite == "measurement" else "analysis"
    if mode == "hardware" and not args.hardware:
        sys.exit("the measurement suite drives the real device and takes the "
                 "main worker lock; pass --hardware to run it")
    os.environ["MULTIMODE_RUN_PROFILE"] = args.profile
    if args.suite == "measurement":
        os.environ["MULTIMODE_RUN_USE_QUEUE"] = "0"
    if args.configs:
        os.environ["MULTIMODE_RUN_CONFIGS"] = args.configs
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    out_dir = args.out or (
        REPO_ROOT / ".suite_runs" / f"{datetime.now():%y%m%d_%H%M%S}_{args.suite}_{mode}"
    )
    print(f"suite={args.suite} mode={mode} profile={args.profile} "
          f"configs={args.configs or 'notebook'}")
    print(f"executed notebooks -> {out_dir}")

    lock = hold_worker_lock() if mode == "hardware" else None
    results = []
    try:
        for path in notebooks:
            ok, seconds, error, xfail, xpass = run_notebook(
                path, out_dir, SKIPPED_TAGS, args.cell_timeout
            )
            results.append((path.stem, ok, seconds, error, xfail, xpass))
            if not ok and not args.keep_going:
                break
    finally:
        if lock is not None:
            lock.release()

    print("\nsummary")
    for name, ok, seconds, error, xfail, xpass in results:
        status = "ok  " if ok else "FAIL"
        known = "".join([
            f" xfail cells {xfail}" if xfail else "",
            f" XPASS cells {xpass}" if xpass else "",
        ])
        print(f"  {status} {name:28s} {seconds:7.0f} s {known} {error or ''}")
    not_run = [p.stem for p in notebooks][len(results):]
    if not_run:
        print(f"  not run: {', '.join(not_run)}")
    return 0 if all(r[1] for r in results) and not not_run else 1


if __name__ == "__main__":
    sys.exit(main())

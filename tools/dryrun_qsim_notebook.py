"""Run a qsim measurement notebook's code cells on a mock station.

    pixi run python tools/dryrun_qsim_notebook.py measurement_notebooks/202609_qsim_migration/mbr.py
    pixi run python tools/dryrun_qsim_notebook.py NOTEBOOK.py --config-set august_n3 --keep-mpm

A development check, not a test and not the suite. ``tools/run_qsim_suite.py``
runs measurement notebooks only on real hardware; this runs the same cells
against ``mock_station`` (pinned config set), with the queue off, so a ported
notebook's plumbing -- imports, runner construction, ``acquire``/``analyze``/
``save`` calls -- is exercised off hardware. Written for MBR redesign step 6.

What mock data cannot do, and what this script does about it:

- The mock returns the same signal for every Ramsey phase, so a Stark-cal
  phase fit fails and ``MBRCalibrationSetExperiment.analyze`` refuses. The fit
  is replaced by a zero fit (phase 0 deg / cycle).
- The same makes every return A(0) = 0, which Matrix Pencil refuses; cells
  that call ``spectrum_method="mpm"`` are skipped unless ``--keep-mpm``.
- A mock calibration over 65 cycle pairs takes about 20 min, so
  ``np.arange(0, 65, dtype=int)`` in the source is cut to 3 pairs.
- The tomography math needs a well-conditioned M_0; mock data gives zeros,
  so ``mbr_tomography.py`` stops at ``analyze_tomography``. Expected.

Cells tagged ``suite-skip`` are skipped, as in the suite. Mock job files go
to the mock station's data path (``C:/experiments/mock_data``). The mock
station never logs to the vault.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("notebook", help="jupytext py:percent notebook")
    parser.add_argument("--config-set", default="preload_current",
                        help="pinned config set for the mock station")
    parser.add_argument("--keep-mpm", action="store_true",
                        help="also run cells that use Matrix Pencil")
    args = parser.parse_args(argv)

    os.environ["MULTIMODE_RUN_PROFILE"] = "smoke"
    os.environ["MULTIMODE_RUN_USE_QUEUE"] = "0"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import experiments
    import job_server
    import experiments.qsim.notebook_helpers.run_mode as run_mode
    from experiments.qsim import mbr_stark_cal
    from experiments.qsim.mbr_campaign import mock_station, pinned_config_set

    station = mock_station(**pinned_config_set(args.config_set))
    experiments.MultimodeStation = lambda *a, **k: station
    job_server.JobClient = lambda *a, **k: None
    run_mode.RunSettings.station_configs = lambda self, config_dict: {}

    def zero_fit(complex_return, cycles):
        n = len(cycles)
        return dict(return_phase=np.zeros(n), phase_fit=np.zeros(n),
                    relative_return=np.ones(n), valid_mask=np.ones(n, bool),
                    phase_per_cycle=0.0, phase_error=0.0)

    mbr_stark_cal.mbr_phase.fit_closed_cycle_phase = zero_fit

    source = Path(args.notebook).read_text(encoding="utf-8")
    source = source.replace("np.arange(0, 65, dtype=int)", "np.arange(0, 3, dtype=int)")
    cells = re.split(r"^# %%", source, flags=re.M)[1:]
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(cells):
        head = cell.split("\n", 1)[0]
        if "[markdown]" in head or "suite-skip" in head:
            continue
        if not args.keep_mpm and 'spectrum_method="mpm"' in cell:
            print(f"--- cell {index} skipped (Matrix Pencil on mock data)", flush=True)
            continue
        start = time.time()
        exec(compile(cell, f"{args.notebook}:cell{index}", "exec"), namespace)
        plt.close("all")
        print(f"--- cell {index} ok ({time.time() - start:.1f}s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

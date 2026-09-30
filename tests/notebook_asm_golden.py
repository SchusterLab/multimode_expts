# -*- coding: utf-8 -*-
"""Pulse-level golden for the calibration notebooks, taken from the notebooks.

``asm_golden`` pins the MBR job classes, which a campaign builds from one
default config. The calibration notebooks are different: each measurement is
the notebook's own defaults and pre/post-processor hooks around a runner, and
those hooks stay in the notebook (``docs/testing.md``). So the programs are
taken from the notebook itself. ``tools/dryrun_qsim_notebook.py`` runs its
code cells on the mock station, and every program that reaches qick's
``config_all`` (every ``acquire``) is recorded.

A runner call compiles one program per sweep point -- 441 for a Floquet gain
chevron -- so each outermost ``CharacterizationRunner.execute`` or
``SweepRunner.execute`` keeps its first and its last program: both ends of
the sweep. Programs compiled outside a runner call are grouped by cell.

One file per notebook and config set, the programs in notebook order, so a
new cell reads as an insertion in the diff and not as renamed files. The
header also lists the cells that fail on mock data, so a change there shows.

This is a characterization pin: it records what the notebook plays now, not
that it is right. Mock data is empty, so a value that a notebook takes from a
fit (an accepted gain, say) is what the fit gives on empty data. Cells after
a failing cell run with whatever the failure left behind.

Regenerate with ``pixi run python -m tests.notebook_asm_golden``, and read
the diff before committing it -- that diff is the review.
"""
import argparse
import gzip
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).parent / "data" / "notebook_asm_golden"
NOTEBOOK_DIR = REPO_ROOT / "measurement_notebooks" / "202609_qsim_migration"

NOTEBOOKS = ("multiphoton_calibration", "floquet_calibration", "floquet_displacement_kerr")
CONFIG_SET = "preload_current"


def capture(name, config_set=CONFIG_SET):
    """Run one notebook on the mock station; -> the pinned text.

    Patches module state for the rest of the process (see
    ``run_notebook``), so :func:`render_in_subprocess` is the way to call it
    from a test.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from qick.qick_asm import AbsQickProgram

    from experiments.characterization_runner import CharacterizationRunner
    from experiments.sweep_runner import SweepRunner
    from tests.asm_golden import render
    from tools.dryrun_qsim_notebook import run_notebook

    state = dict(cell=None, depth=0)
    groups = []          # dicts: cell, label, first, last, count

    def open_group(label):
        groups.append(dict(cell=state["cell"], label=label,
                           first=None, last=None, count=0))

    def record(prog):
        if state["depth"] == 0:
            last = groups[-1] if groups else None
            if last is None or last["label"] != "outside a runner" or last["cell"] != state["cell"]:
                open_group("outside a runner")
        group = groups[-1]
        if group["first"] is None:
            group["first"] = f"### first: {type(prog).__name__}\n{render(prog)}"
        group["last"] = prog
        group["count"] += 1

    config_all = AbsQickProgram.config_all

    def recording_config_all(self, *args, **kwargs):
        record(self)
        return config_all(self, *args, **kwargs)

    def wrap(runner_class):
        execute = runner_class.execute

        def recording_execute(self, *args, **kwargs):
            if state["depth"] == 0:
                open_group(f"{runner_class.__name__}.execute "
                           f"({getattr(self.ExptClass, '__name__', self.ExptClass)})")
            state["depth"] += 1
            try:
                return execute(self, *args, **kwargs)
            finally:
                state["depth"] -= 1

        runner_class.execute = recording_execute

    AbsQickProgram.config_all = recording_config_all
    wrap(CharacterizationRunner)
    wrap(SweepRunner)

    def on_cell(index):
        state["cell"] = index

    failed = run_notebook(NOTEBOOK_DIR / f"{name}.py", config_set,
                          keep_going=True, on_cell=on_cell)

    parts = [f"# notebook {name}, config set {config_set}",
             f"# cells that fail on mock data: {failed}", ""]
    number = 0
    for group in groups:
        if not group["count"]:
            continue            # a runner call that compiled nothing
        number += 1
        head = (f"## program group {number}: cell {group['cell']}, {group['label']}, "
                f"{group['count']} programs")
        last = render(group["last"])
        parts.append(f"{head}\n{group['first']}")
        if group["count"] > 1:
            parts.append(f"### last: {type(group['last']).__name__}\n{last}")
    return "\n".join(parts)


def render_in_subprocess(name, config_set=CONFIG_SET):
    """-> :func:`capture`'s text, from a fresh interpreter."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "tests.notebook_asm_golden", "--print", name,
         "--config-set", config_set],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"capture of {name} failed:\n{result.stderr[-4000:]}")
    marker = "=====BEGIN GOLDEN=====\n"
    return result.stdout.split(marker, 1)[1]


def path_for(name, config_set=CONFIG_SET):
    return GOLDEN_DIR / f"{name}__{config_set}.txt.gz"


def read(name, config_set=CONFIG_SET):
    with gzip.open(path_for(name, config_set), "rt", encoding="utf-8") as handle:
        return handle.read()


def write(name, text, config_set=CONFIG_SET):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    # mtime=0: a regenerated-but-identical file must not show up as a diff.
    with gzip.GzipFile(path_for(name, config_set), "wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--print", dest="print_name", default=None,
                        help="capture one notebook and print the text (used by the test)")
    parser.add_argument("--config-set", default=CONFIG_SET)
    args = parser.parse_args(argv)

    if args.print_name:
        text = capture(args.print_name, args.config_set)
        sys.stdout.reconfigure(encoding="utf-8")
        print("=====BEGIN GOLDEN=====\n" + text, end="")
        return
    for name in NOTEBOOKS:
        text = render_in_subprocess(name, args.config_set)
        write(name, text, args.config_set)
        size = path_for(name, args.config_set).stat().st_size
        print(f"wrote {path_for(name, args.config_set).name} ({size / 1024:.1f} KiB gz)")


if __name__ == "__main__":
    main()

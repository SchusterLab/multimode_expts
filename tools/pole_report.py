"""Run the pole-finding report headless (docs/qsim/pole_finding.md 8.1).

    pixi run pole-report [--fitters A,B,E] [--size small|full] [--experiment 260818_qsim_spectroscopy]

Executes ``analysis_notebooks/pole_finding/report.py`` in a fresh kernel and writes, under
``<data root>/<experiment>/derived_data/pole_finding/<run>/``, the executed notebook as HTML
and the results HDF5 (``results.h5``). The run settings reach the kernel as the JSON
environment variable ``POLE_REPORT_RUN``. ``--registry`` (the real data sets, spec 8.2) is
phase 2 and not read yet.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from experiments.job_paths import data_root

REPORT = Path(__file__).parents[1] / "analysis_notebooks" / "pole_finding" / "report.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fitters", default="A,B,E")
    parser.add_argument("--size", choices=["small", "full"], default="small")
    parser.add_argument("--experiment", default="260818_qsim_spectroscopy")
    parser.add_argument("--registry", default=None, help="phase 2; not read yet")
    args = parser.parse_args()

    name = f"{datetime.now():%y%m%d_%H%M%S}_{args.size}_{args.fitters.replace(',', '')}"
    output_dir = data_root() / args.experiment / "derived_data" / "pole_finding" / name
    output_dir.mkdir(parents=True)
    os.environ["POLE_REPORT_RUN"] = json.dumps(dict(
        fitters=args.fitters.split(","), size=args.size, output_dir=str(output_dir), registry=args.registry))

    notebook = jupytext.read(REPORT)
    try:
        NotebookClient(notebook, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": str(REPORT.parent)}}).execute()
    finally:  # a failed run still leaves its notebook and HTML, up to the failing cell
        nbformat.write(notebook, output_dir / "report.ipynb")
        html, _ = HTMLExporter().from_notebook_node(notebook)
        (output_dir / "report.html").write_text(html, encoding="utf-8")
        print(output_dir)


if __name__ == "__main__":
    main()

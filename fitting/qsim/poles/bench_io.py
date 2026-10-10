"""The report's method text and the saved results (spec 8.1, 8.3).

The method text is built from the fitters' own docstrings and settings source,
so it cannot drift from what ran.
"""
import inspect
import json
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

from fitting.qsim.poles.bench_summaries import scores_table


def method_markdown(fitters):
    """-> markdown: per fitter, its module docstring, its ``fit`` docstring and its settings."""
    parts = []
    for name, (fit, settings) in fitters.items():
        module = sys.modules[fit.__module__]
        parts.append(f"### Fitter {name}: `{module.__name__}`\n\n{inspect.getdoc(module)}\n\n"
                     f"`fit`: {inspect.getdoc(fit)}\n\n"
                     f"```python\n{inspect.getsource(type(settings))}```\n\n"
                     f"Settings used: `{settings.model_dump_json()}`\n")
    return "\n".join(parts)


def code_version():
    """-> the commit of this checkout, with ``+dirty`` if it has uncommitted changes."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return commit + ("+dirty" if dirty else "")


def save_results(path, results, **provenance):
    """Write each BenchResult's scores table to one HDF5 group, with the settings,
    hardware, match tolerance, code version and ``provenance`` as attributes."""
    names = [result.name for result in results]
    if len(set(names)) != len(names):
        raise ValueError(f"results need distinct names: {names}")
    with h5py.File(path, "w") as file:
        file.attrs["code_version"] = code_version()
        file.attrs["provenance"] = json.dumps(provenance, default=str)
        for result in results:
            group = file.create_group(result.name)
            group.attrs["fitter_settings"] = json.dumps(result.fitter_settings, default=str)
            group.attrs["hardware"] = result.hardware.model_dump_json()
            group.attrs["match_tolerance_bins"] = result.match_tolerance_bins
            for column, values in scores_table(result).items():
                values = values.to_numpy()
                group[column] = values.astype("S") if values.dtype == object else values
    return path


def load_scores(path, name):
    """-> the scores table of one saved benchmark, as a pandas DataFrame."""
    with h5py.File(path, "r") as file:
        group = file[name]
        return pd.DataFrame({column: np.char.decode(group[column][()]) if group[column].dtype.kind == "S"
                             else group[column][()] for column in group})

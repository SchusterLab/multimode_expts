"""The data set registry (spec 8.2): which real data sets the benchmarks read.

A checked-in YAML list, ``analysis_notebooks/pole_finding/registry.yaml``, one entry per
data set. Benchmarks 3 and 4 read only the registry, never a path of their own.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

REGISTRY = Path(__file__).parents[3] / "analysis_notebooks" / "pole_finding" / "registry.yaml"


class DataSet(BaseModel):
    """One data set: an assembled manifest, and what the benchmarks must know about it."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    #: Relative to ``data_root()``: an MBRSpectrumExperiment or
    #: MBRDisorderEnsembleExperiment manifest.
    manifest: str
    #: complete: all D occupations measured (integer pole weights); partial:
    #: selected rows only (only 0 <= w <= m holds). Spec section 3.
    basis: Literal["complete", "partial"]
    photon_number: int
    #: For a disorder ensemble: the strength (kHz) and master seed; else empty.
    disorder: dict = {}
    #: Where the data set is described: the log, notebook cells, dataset list.
    source: str


def load_registry(path=REGISTRY):
    """-> the registry's DataSets, in file order."""
    return [DataSet(**entry) for entry in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


def manifest_path(data_set, root):
    """-> the absolute manifest path under the data tree ``root``
    (``experiments.job_paths.data_root()`` on a machine that has one)."""
    return Path(root) / data_set.manifest

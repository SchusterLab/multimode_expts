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
    #: The frame its analysis notebook uses (``MBRSpectrumExperiment.analyze``).
    analysis: "Analysis" = None


class Analysis(BaseModel):
    """How a data set's spectra are analyzed and which model they are compared with."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    phase_frame: Literal["as_acquired", "manual_kerr"] = "as_acquired"
    #: A number (MHz), or "recorded": each realization's saved self-Kerr.
    manual_kerr_MHz: float | Literal["recorded"] | None = None
    #: [occupation, branch] pairs; unlisted occupations use branch 0.
    cycle_branches: list[tuple[tuple[int, ...], int]] = []
    #: Occupations left out of the analysis (the 7-1 analysis leaves out (0, 3, 0, 0, 0)).
    excluded_occupations: list[tuple[int, ...]] = []
    #: The model's Kerr: the spectrum's analysis Kerr, or each realization's recorded one.
    model_kerr: Literal["analysis", "recorded"] = "analysis"

    def branches(self, occupations):
        """-> {occupation: branch} for these occupations."""
        table = dict(self.cycle_branches)
        return {tuple(o): table.get(tuple(o), 0) for o in occupations}


DataSet.model_rebuild()


def load_registry(path=REGISTRY):
    """-> the registry's DataSets, in file order."""
    return [DataSet(**entry) for entry in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


def data_set(label, path=REGISTRY):
    """-> the registry entry with this label."""
    return next(entry for entry in load_registry(path) if entry.label == label)


def manifest_path(data_set, root):
    """-> the absolute manifest path under the data tree ``root``
    (``experiments.job_paths.data_root()`` on a machine that has one)."""
    return Path(root) / data_set.manifest

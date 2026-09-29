"""The fixed synthetic set of the gap score (plan ``docs/qsim/pole_finding_explore.md``, T0).

Saved once, so that every exploration task scores on the same data; the fits of slow fitters
(C 30-120 s, F minutes) are cached next to it, one HDF5 file per fitter. The set:

- two model points: the August disorder point (g 8.615 kHz, K / g -1.22, delta / g 5.80) and
  the September complete-basis regime (g 29.2 kHz, K -3.76 kHz, disorder 50 kHz: K / g -0.129,
  delta / g 1.71);
- both sampled as in August (dt 1.4509 us, 300 samples, a 435 us window, about 2 T2, the right
  length by the design calculator): the September time grid is not known offline yet (its jobs
  carry no timing; T4). With the August g the September point is 3.4 times denser in kHz than
  the data, and 5-9 of 34 gaps are resolvable at all; with its own g its span (about 390 kHz)
  still fits the 690 kHz sampling band;
- T2 100 and 200 us, row offsets of 0.5 and 1 kHz, 10 rows (chosen as the campaigns chose them)
  and 35 rows (complete basis), 5 draws each (disorder direction ``100 + draw``, noise seed
  ``draw``); the noise per sample 0.075 of A_b(0) in every row (the August data's).

Each case carries its Cramér-Rao gap bounds for free complex and for real amplitudes
(``design.gap_errors``, offsets free with the true prior). Pure numerics and HDF5.
"""
import json
from dataclasses import dataclass
from itertools import product
from typing import Annotated, Literal

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fitting.qsim.poles.bench_io import code_version
from fitting.qsim.poles.design import Design, gap_errors
from fitting.qsim.poles.pole_fit import PoleFit
from fitting.qsim.poles.synthetic import Hardware, ModelPoint, Nonideal, synthetic_returns
from fitting.qsim.mbr_disorder import disorder_direction

#: (g in MHz, K / g, delta / g) of the model points.
POINTS = {"august": (8.615e-3, -1.22, 5.80), "september": (29.2e-3, -0.129, 1.71)}
GRID = dict(dt_us=1.4509, samples=300)
NOISE_PER_SAMPLE = 0.075
DIRECTION_SEED = 100
BOUNDS = ("complex", "real")


class Condition(BaseModel):
    """One case of the set."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    point: Literal["august", "september"]
    decay_per_us: Annotated[float, Field(gt=0.)]
    offset_sigma_MHz: Annotated[float, Field(gt=0.)]
    #: None: the complete basis (35 rows).
    partial_rows: int | None
    draw: Annotated[int, Field(ge=0)]

    @property
    def key(self):
        rows = self.partial_rows or "all"
        return (f"{self.point}_T2-{1 / self.decay_per_us:.0f}us_offset-{1e6 * self.offset_sigma_MHz:.0f}Hz"
                f"_rows-{rows}_draw-{self.draw}")

    @property
    def hardware(self):
        return Hardware(coupling_MHz=POINTS[self.point][0], **GRID, partial_rows=self.partial_rows)

    @property
    def nonideal(self):
        return Nonideal(snr=1 / NOISE_PER_SAMPLE, decay_per_us=self.decay_per_us,
                        offset_sigma_MHz=self.offset_sigma_MHz, seed=self.draw)


def conditions(draws=5):
    """-> the Conditions of the set, in a fixed order."""
    return [Condition(point=point, decay_per_us=decay, offset_sigma_MHz=sigma, partial_rows=rows, draw=draw)
            for point, decay, sigma, rows, draw in product(POINTS, (0.01, 0.005), (0.5e-3, 1e-3), (10, None), range(draws))]


@dataclass(frozen=True)
class StoredCase:
    """A case as saved: the rows, the truth, the injected offsets and the gap bounds."""
    condition: Condition
    time_us: np.ndarray
    A: np.ndarray
    levels_MHz: np.ndarray
    multiplicities: np.ndarray
    row_weights: np.ndarray
    occupations: np.ndarray
    offsets_MHz: np.ndarray
    gap_bounds_MHz: dict

    @property
    def bin_MHz(self):
        return 1 / (len(self.time_us) * (self.time_us[1] - self.time_us[0]))

    @property
    def sampling_MHz(self):
        return 1 / (self.time_us[1] - self.time_us[0])


def make_case(condition):
    """-> the StoredCase of a Condition (bounds included; a few seconds for 35 rows)."""
    _, kerr, disorder = POINTS[condition.point]
    hardware, nonideal = condition.hardware, condition.nonideal
    point = ModelPoint(kerr, disorder, disorder_direction(DIRECTION_SEED + condition.draw, hardware.mode_count - 1),
                       condition.draw)
    case = synthetic_returns(point, hardware, nonideal)
    truth = case.truth
    offsets = np.random.default_rng(nonideal.seed).normal(0, nonideal.offset_sigma_MHz, size=len(case.A))
    occupations = np.asarray(truth.occupations, dtype=float)
    noise = np.full(len(case.A), NOISE_PER_SAMPLE)
    bounds = {name: gap_errors(case.time_us, truth.levels_MHz, truth.row_weights, occupations, noise, hardware.bin_MHz,
                               Design(amplitudes=name, offset_prior_MHz=condition.offset_sigma_MHz,
                                      decay_per_us=condition.decay_per_us))
              for name in BOUNDS}
    return StoredCase(condition, case.time_us, case.A, truth.levels_MHz, truth.multiplicities, truth.row_weights,
                      occupations, offsets, bounds)


def save_set(path, cases, **provenance):
    """Write the StoredCases, one group per case key, with the code version."""
    with h5py.File(path, "w") as file:
        file.attrs["code_version"] = code_version()
        file.attrs["provenance"] = json.dumps(provenance, default=str)
        file.attrs["noise_per_sample"] = NOISE_PER_SAMPLE
        for case in cases:
            group = file.create_group(case.condition.key)
            group.attrs["condition"] = case.condition.model_dump_json()
            for name in ("time_us", "A", "levels_MHz", "multiplicities", "row_weights", "occupations", "offsets_MHz"):
                group[name] = getattr(case, name)
            for name, bound in case.gap_bounds_MHz.items():
                group[f"gap_bounds_MHz/{name}"] = bound
    return path


def load_set(path):
    """-> the StoredCases of a saved set, in the saved order of ``conditions``."""
    with h5py.File(path, "r") as file:
        cases = []
        for key in file:
            group = file[key]
            cases.append(StoredCase(
                Condition.model_validate_json(group.attrs["condition"]),
                *(group[name][()] for name in ("time_us", "A", "levels_MHz", "multiplicities", "row_weights",
                                                "occupations", "offsets_MHz")),
                {name: group[f"gap_bounds_MHz/{name}"][()] for name in group["gap_bounds_MHz"]}))
    order = {condition.key: i for i, condition in enumerate(conditions(max(c.condition.draw for c in cases) + 1))}
    return sorted(cases, key=lambda case: order.get(case.condition.key, len(order)))


_FIT_FIELDS = ("frequencies_MHz", "decays_per_us", "amplitudes", "frequency_errors_MHz", "row_offsets_MHz")


def save_fit(path, key, fit, seconds, settings):
    """Add (or replace) one case's PoleFit in a fitter's cache file."""
    with h5py.File(path, "a") as file:
        if key in file:
            del file[key]
        group = file.create_group(key)
        group.attrs.update(rank=fit.rank, seconds=seconds, settings=settings.model_dump_json(), code_version=code_version())
        for name in _FIT_FIELDS:
            if getattr(fit, name) is not None:
                group[name] = getattr(fit, name)


def load_fits(path):
    """-> {case key: (PoleFit, seconds)} of a fitter's cache file; empty if there is none."""
    try:
        file = h5py.File(path, "r")
    except FileNotFoundError:
        return {}
    with file:
        return {key: (PoleFit(rank=int(group.attrs["rank"]),
                              **{name: group[name][()] if name in group else None for name in _FIT_FIELDS}),
                      float(group.attrs["seconds"]))
                for key, group in file.items()}

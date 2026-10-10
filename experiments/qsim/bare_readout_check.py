# -*- coding: utf-8 -*-
"""The bare-readout check before an MBR campaign: pool and plot the scramble jobs.

One photon is loaded into M1, scrambled by the Floquet train, and read out of
each mode in turn. The sweep over Floquet cycles is split into many jobs (the
tProc instruction memory), so the check pools them by readout mode.

Used by `measurement_notebooks/202609_qsim_migration/floquet_calibration.py`.
Replaces `notebook_helpers/floquet_bare_readout.py` (MBR redesign step 9D). The
time axis now comes from each job's compiled Floquet cycle
(``derived_params()['floquet_cycle_us']``, the definition in
`experiments/floquet_timing.py`); the old notebook estimate summed exact pulse
lengths and ran about 1.2% long.
"""
import textwrap
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def pool_by_readout(expts):
    """-> {ro_stor: dict(floquet_cycle, avgi, avgq)}, each sorted by cycle, over all jobs."""
    pooled = defaultdict(lambda: {"floquet_cycle": [], "avgi": [], "avgq": []})
    for expt in expts:
        entry = pooled[int(expt.cfg.expt.ro_stor)]
        entry["floquet_cycle"].extend(np.asarray(expt.data["xpts"]).reshape(-1))
        entry["avgi"].extend(np.asarray(expt.data["avgi"]).reshape(-1))
        entry["avgq"].extend(np.asarray(expt.data["avgq"]).reshape(-1))
    out = {}
    for ro_stor, entry in pooled.items():
        order = np.argsort(entry["floquet_cycle"])
        out[ro_stor] = {key: np.asarray(values)[order] for key, values in entry.items()}
    return out


def floquet_cycle_us(expts):
    """-> the compiled Floquet cycle of the jobs, in us; they must agree."""
    cycles = {round(float(expt.derived_params()["floquet_cycle_us"]), 12) for expt in expts}
    if len(cycles) != 1:
        raise ValueError(f"the jobs have different Floquet cycles: {sorted(cycles)} us")
    return cycles.pop()


def display_bare_scramble(expts, plot_time=True, plot_q=False, scatter=False):
    """Avg I (and Q) per readout mode vs Floquet cycles or time.

    ``expts`` is every scramble job of the check (any nesting of lists).
    -> (fig, pooled data, cycle time in us).
    """
    flat = list(dict.fromkeys(_flatten(expts)))
    pooled = pool_by_readout(flat)
    cycle_us = floquet_cycle_us(flat)

    fig, axes = plt.subplots(2 if plot_q else 1, 1, figsize=(9, 10 if plot_q else 5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ro_stor, entry in pooled.items():
        x = entry["floquet_cycle"] * cycle_us if plot_time else entry["floquet_cycle"]
        label = "M1" if ro_stor == 0 else f"S{ro_stor}"
        for axis, key in zip(axes, ("avgi", "avgq")):
            if scatter:
                axis.scatter(x, entry[key], alpha=0.6, label=label)
            else:
                axis.plot(x, entry[key], label=label)
    for axis, key in zip(axes, ("Avg I", "Avg Q")):
        axis.set_ylabel(f"{key} (ADC unit)")
        axis.set_xlabel("Time (us)" if plot_time else "Floquet Cycles")
        axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle(textwrap.fill(str([expt.fname for expt in flat]), width=80), fontsize=7)
    fig.tight_layout()
    plt.show()
    return fig, pooled, cycle_us


def _flatten(items):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item

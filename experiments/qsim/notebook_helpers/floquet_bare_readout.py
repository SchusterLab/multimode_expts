"""Bare-readout check that the MBR campaigns need before they run.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
150-158 by the stage-2 notebook decomposition. Primary caller:
`measurement_notebooks/202609_qsim_migration/floquet_calibration.py`.

The surface map assigns this range here rather than to the dark-mode notebook:
cells 150-158 are the "Bare" subsection, which is a prerequisite check, while
the displace/multiparity dark-mode work that follows it (cells 159-262) is
dormant.

`sideband_scramble_preproc` and the four collection helpers came straight from
cells 151 and 152. Note that cell 152 redefined `floquet_cycle_list_gen`
identically to cell 151, so it lives once, in `floquet_calibration.py`. The
dormant dark-mode notebook redefines `unique_expts`, `flatten_exp_lists` and
`stor_label_for_expt` with *different* bodies; those stayed there and were not
merged with these.

Two things that were notebook globals and are now arguments, because the move
is what broke them: `floquet_cycle_to_us` read `station` directly, and the
plotting cell reached for `fname_list` through `globals()`, which the stage-2
instructions rule out.

The sweep loop itself is **not** here. An earlier pass wrapped cell 154 in
`run_bare_scramble_sweep`, whose closed keyword list cut the notebook off from
`runner.execute` -- no `use_queue`, no `priority`, no expt_cfg override. It is
back inline in the notebook, next to the defaults dict and the runner it uses.
What remains here is preprocessing, pooling and plotting: code with real logic
in it that the notebook should not carry.
"""

import textwrap
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


def sideband_scramble_preproc(station, default_expt_cfg, **kwargs):
    assert 'swept_params' in kwargs
    assert len(kwargs['swept_params']) > 0

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    assert 'init_stor' in kwargs
    if not expt_cfg.init_fock:
        assert 'init_alpha' or 'init_man_fock_state' in kwargs

    # print(expt_cfg)
    return expt_cfg


def unique_expts(nested_expts):
    return list(dict.fromkeys(expt for sub in nested_expts for expt in sub))


def collect_multiparity(expts):
    by_stor = {}

    for expt in expts:
        ro_stor = expt.cfg.expt.ro_stor
        mp = expt.analyze_multiparity()

        if ro_stor not in by_stor:
            by_stor[ro_stor] = []

        by_stor[ro_stor].append(mp)

    out = {}
    for ro_stor, mp_list in by_stor.items():
        keys = mp_list[0].keys()
        out[ro_stor] = {
            key: np.concatenate([mp[key] for mp in mp_list])
            for key in keys
        }

    return out


def flatten_exp_lists(items, container_types=(list, tuple, set)):
    for x in items:
        if isinstance(x, container_types):
            yield from flatten_exp_lists(x, container_types)
        else:
            yield x


def _flat_unique_expts(expts_nested):
    out = []
    for item in expts_nested:
        if isinstance(item, (list, tuple)):
            for expt in item:
                if expt not in out:
                    out.append(expt)
        else:
            if item not in out:
                out.append(item)
    return out


def stor_label_for_expt(ro_stor, expt):
    ro_stor = int(ro_stor)
    ecfg = expt.cfg.expt

    if not ecfg.get("swap_man_dark", False):
        return "M1" if ro_stor == 0 else f"S{ro_stor}"

    dark_order = list(ecfg.get("dark_swap_order", []))

    if ro_stor == 0:
        return "Dark Mode"
    if len(dark_order) > 1 and ro_stor == int(dark_order[1]):
        return "Bright Mode"
    if len(dark_order) > 0 and ro_stor == int(dark_order[0]):
        return "Central Mode"

    return f"S{ro_stor}"


def floquet_cycle_to_us(expt, station):
    """Notebook estimate of one Floquet cycle's duration, from cell 156.

    TODO(stage2): this disagrees with the canonical definition on purpose --
    it is preserved as found, not endorsed.
    `experiments/floquet_timing.floquet_cycle_us` is the one definition, and
    its docstring records that summing exact pulse durations the way this
    function does ran about 1.2% long on the August 2026 configs. QICK v1's
    `sync_all` emits `synci(int(...))`, so a real cycle is a sum of *integer*
    tProc advances. This function also uses `ramp_sigma * 4` where the
    canonical arithmetic uses six ramps around the plateau.

    Only the x axis of the bare-readout diagnostic plot depends on it, so the
    error does not propagate into a Hamiltonian here. Reconciling the two is a
    later, theme-specific task, which the stage-2 instructions put out of
    scope; fixing it during extraction would have been a silent physics
    change.

    `station` was a notebook global that this body closed over. It is an
    argument now because the move is what broke that.
    """
    total = 0.0
    sync_cycles = int(expt.cfg.expt.get("scramble_sync_cycles", 10))

    for stor in expt.cfg.expt.swap_stors:
        name = f"M1-S{stor}"
        if station.ds_floquet.get_waveform(name) == "gauss":
            total += float(
                station.ds_floquet.get_gauss_sigma(name)
                * station.ds_floquet.get_gauss_n_sigma(name)
            )
        else:
            total += float(station.ds_floquet.get_len(name))
            total += float(station.ds_floquet.get_ramp_sigma(name)) * 4

        total += station.soccfg.cycles2us(sync_cycles)

    return total


def plot_bare_scramble(scramble_expts, station, fname_list=None,
                       plot_time=True, plot_q=False, scatter=False):
    """Pool every job by readout mode and plot the bare traces (cell 156).

    `fname_list` was picked up through `globals()` in the source; it is an
    optional argument now. Pass the list built by `flatten_exp_lists` if you
    want the filenames in the figure title.

    Returns (fig, ax, combined_data, cycle_us). `cycle_us` comes from
    `floquet_cycle_to_us` -- see the TODO on that function before trusting the
    time axis quantitatively.
    """
    expts_flat = _flat_unique_expts(scramble_expts)
    ref_expt = expts_flat[0]
    cycle_us = floquet_cycle_to_us(ref_expt, station)

    combined_data = defaultdict(lambda: {"xpts": [], "avgi": [], "avgq": []})

    for expt in expts_flat:
        ro_stor = int(expt.cfg.expt.ro_stor)
        combined_data[ro_stor]["xpts"].extend(np.asarray(expt.data["xpts"]).reshape(-1))
        combined_data[ro_stor]["avgi"].extend(np.asarray(expt.data["avgi"]).reshape(-1))
        combined_data[ro_stor]["avgq"].extend(np.asarray(expt.data["avgq"]).reshape(-1))

    if plot_q:
        fig, ax = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(9, 5))
        ax = [ax0]

    for ro_stor, data in combined_data.items():
        order = np.argsort(data["xpts"])

        x = np.asarray(data["xpts"])[order]
        if plot_time:
            x = x * cycle_us

        avgi = np.asarray(data["avgi"])[order]
        avgq = np.asarray(data["avgq"])[order]
        label = stor_label_for_expt(ro_stor, ref_expt)

        if scatter:
            ax[0].scatter(x, avgi, alpha=0.6, label=label)
            if plot_q:
                ax[1].scatter(x, avgq, alpha=0.6, label=label)
        else:
            ax[0].plot(x, avgi, label=label)
            if plot_q:
                ax[1].plot(x, avgq, label=label)

    xlabel = "Time (us)" if plot_time else "Floquet Cycles"

    ax[0].set_ylabel("Avg I (ADC unit)")
    ax[0].set_xlabel(xlabel)
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    if plot_q:
        ax[1].set_ylabel("Avg Q (ADC unit)")
        ax[1].set_xlabel(xlabel)
        ax[1].legend()
        ax[1].grid(alpha=0.25)

    if fname_list is not None:
        fig.suptitle(textwrap.fill(str(fname_list), width=80))

    fig.tight_layout()
    plt.show()
    return fig, ax, combined_data, cycle_us

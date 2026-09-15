# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [markdown]
# # Dark mode and multiphoton Fock post-processing (dormant)
#
# Relocated from `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 104-153 by the stage-2 notebook decomposition. **Dormant**: kept findable and
# diffable, not maintained. Covers dark-mode post-processing and its
# disorder-dependent decay, the multiphoton Fock confusion-matrix work, the
# multiparity/mod-4 correction and plots, the large-support dark-mode
# bare/readout/load-and-readout sections, the disorder-ensemble decay with its
# threshold power law, and a trailing unperused note on IQ rotation.
#
# Relocation only, per the stage-2 instructions. The measurement counterpart is
# `measurement_notebooks/202609_qsim_migration/dormant/dark_mode.py`; the library
# side is `experiments/qsim/dark_*.py` and `floquet_dark_mode_readout.py`.
#
# Sibling dormant notebooks: `flux_excursion.py`, `wigner.py`,
# `pulse_scratch.py`.

# %%
# %load_ext autoreload
# %autoreload 2


import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, glob
import qutip as qt
import textwrap
import experiments as meas

from copy import deepcopy
from collections import defaultdict
from tqdm.notebook import tqdm
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display_classes import GeneralFitting
from fitting.wigner import WignerAnalysis
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner


REPO_ROOT = r"C:\python\multimode_expts" 
BASE_DIR  = r"C:\experiments"
RUN_PREFIX = "JOB-20260414"  

# The four aggregate MBR stages. `EncodingHamiltonianSpectroscopyExperiment`
# is still the loading layer and the shared numerics, and is still the class
# every job here was acquired under -- so it stays, and these four sit beside
# it. See analysis_notebooks/guan/MBR_analysis.py for the worked example.
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_propagator import MBRPropagatorExperiment

# %% [markdown]
# Helpers this notebook uses, sliced out of the original
# `data_postprocess.ipynb` cell 4 -- a single 1563-line cell that defined all 71
# helpers for every section at once. Only this notebook's share is carried here.
# Shared helpers are copied rather than imported, so a dormant notebook never
# depends on an active theme's module.

# %%
# =====================================================================
# Helper Functions & Definitions -- run this ONE cell before any section
# below. Every function/class the notebook uses is defined here, so no
# section depends on another section's cells having been run first.
# (Requires the import cell above to have run: np, plt, meas, AttrDict, ...)
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import os, textwrap
from pathlib import Path
from collections import namedtuple, defaultdict


def hdf5_path_generator(project_name,
                        job_date,
                        job_start_num,
                        job_finish_num, 
                        experiement_name,
                        basedir = BASE_DIR,
                        verbose = False):
    object_directory = os.path.join(basedir, project_name, "data")
    path = {}
    hdf5_path_to_return = []
    for job_id in job_id_generator(job_date, job_start_num, job_finish_num):
        filepath = os.path.join(object_directory, f"{job_id}_{experiement_name}.h5")
        if os.path.exists(filepath):
            hdf5_path_to_return.append(filepath)
            if verbose: print(f"[O] Found: {filepath}")
        elif verbose: print(f"[X] Missing: {filepath}")
    path[experiement_name] = hdf5_path_to_return
    return path


def path_to_experiment(hdf5_path, ExpClass):
    
    exp_name = list(hdf5_path.keys())[0]
    path_list = hdf5_path[exp_name]

    exp_list = []

    for fname in tqdm(path_list):
        obj = ExpClass.from_h5file(fname)

        obj.path = os.path.dirname(fname)
        obj.config_file = fname
        obj.prefix = os.path.splitext(os.path.basename(fname))[0]

        exp_list.append(obj)

    return exp_list


def _ensure_multiparity_read_num(expt):
    """Backfill expt.cfg.read_num when it was not saved to the .h5 (e.g. Qsim
    runs). Defined here so this cell is self-contained -- do not rely on the copy
    in the confusion-matrix cell being run first."""
    if 'read_num' in expt.cfg:
        return

    reps = int(expt.cfg.expt.get('reps', 1))
    rounds = int(expt.cfg.expt.get('rounds', 1))
    shots_per_point = np.asarray(expt.data['idata'][0]).size
    shots_per_round = max(reps * rounds, 1)
    if shots_per_point % shots_per_round == 0:
        expt.cfg.read_num = shots_per_point // shots_per_round
        return

    read_num = 1
    if expt.cfg.expt.get('parity_check', False):
        read_num += 1
    if expt.cfg.expt.get('active_reset', False):
        try:
            from experiments.MM_base import MMAveragerProgram
            params = MMAveragerProgram.get_active_reset_params(expt.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)
        except Exception:
            pass
    if expt.cfg.expt.get('multiparity_readout', False):
        read_num += 1
    expt.cfg.read_num = read_num


def plot_thick_line_with_dots(ax, x, y, label=None, color=None):
    line, = ax.plot(x, y, linestyle='-', linewidth=4.0, alpha=0.3, label=label, color=color)
    col = line.get_color()
    ax.plot(x, y, marker='o', linestyle='None', markersize=8, alpha=1.0, color=col)
    return col


def _sine(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C


def fit_sine(t, y):
    """Fit y(t) to a sine. Returns dict with f [MHz], omega [Mrad/s], A, phi, C, popt; None on failure."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    if len(t) < 4 or np.ptp(y) == 0 or np.ptp(t) == 0:
        return None
    C0 = float(np.mean(y))
    A0 = float(np.ptp(y) / 2) or 1.0
    # FFT-based frequency guess (assumes ~uniform sampling in t)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None
    spec = np.abs(np.fft.rfft(y - C0))
    freqs = np.fft.rfftfreq(len(t), d=dt)
    f0 = freqs[1:][np.argmax(spec[1:])] if len(freqs) > 2 else 1.0 / np.ptp(t)
    p0 = [A0, max(float(f0), 1e-9), 0.0, C0]
    try:
        popt, _ = curve_fit(_sine, t, y, p0=p0, maxfev=20000)
    except Exception:
        return None
    A, f, phi, C = popt
    f = abs(float(f))
    return {'f': f, 'omega': 2 * np.pi * f, 'A': abs(float(A)),
            'phi': float(phi) % (2 * np.pi), 'C': float(C), 'popt': popt}


def exp_decay(x, y0, amp, tau):
    return y0 + amp * np.exp(-(x - np.min(x)) / tau)


def fit_t1(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    y0_guess = np.mean(y[-max(5, len(y)//10):])
    amp_guess = y[0] - y0_guess
    tau_guess = 0.5 * (np.max(x) - np.min(x))

    model = Model(exp_decay)
    params = model.make_params(y0=y0_guess, amp=amp_guess, tau=tau_guess)
    params['tau'].min = 0

    result = model.fit(y, params, x=x)

    return result


def set_confusion_matrix(C):
    """Set the global confusion matrix used by correct_distribution().

    Stores the row-normalized matrix in `confusion_matrix` and its inverse
    transpose in `confusion_inv`  (p_meas = C.T @ p_true -> p_true = inv(C.T) @ p_meas).
    """
    global confusion_matrix, confusion_inv
    C = np.asarray(C, dtype=float)
    confusion_matrix = C / C.sum(axis=1, keepdims=True)
    confusion_inv = np.linalg.inv(confusion_matrix.T)
    return confusion_matrix


def get_dark_params(expts):
    """Extract (and validate identical across `expts`) the dark-readout config
    from cfg.expt, returning a DarkParams. Also updates the module-level
    DARK_PARAMS so downstream label/plot helpers pick up this dataset's params."""
    global DARK_PARAMS
    e0 = expts[0].cfg.expt
    params = DarkParams(e0.swap_stors, e0.swap_man_dark, e0.dark_swap_order)
    for exp in expts:
        c = exp.cfg.expt
        if (c.swap_stors, c.swap_man_dark, c.dark_swap_order) != tuple(params):
            raise Exception("Exps are not identical")
    DARK_PARAMS = params
    return params


def stor_label(ro_stor, params=None):
    """Human-readable label for a readout storage index. `params` is a DarkParams;
    if omitted, falls back to the module-level DARK_PARAMS set by the most recent
    get_dark_params() call."""
    if params is None:
        params = DARK_PARAMS
    if params is None or not params.swap_man_dark:
        return "M1" if int(ro_stor) == 0 else f"S{int(ro_stor)}"
    if int(ro_stor) == 0:
        return "Dark Mode"
    elif int(ro_stor) == params.dark_swap_order[1]:
        return "Bright Mode"
    else:
        return "Central Mode"


def load_dark_experiments(project, job_date_list, start_num_list, finish_num_list,
                          ExpClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment):
    """Reconstruct (and dedup) an experiment dataset from *.h5 files.

    Defaults to DarkBaseExperiment; pass e.g. meas.qsim.qsim_base.QsimBaseExperiment
    to load Qsim runs instead. The .h5 filename tag is taken from ExpClass.__name__,
    so it must match how the files were saved: JOB-<date>-<num>_<ExpClass>.h5."""
    path = hdf5_path_generator(project, job_date_list, start_num_list,
                               finish_num_list, ExpClass.__name__)
    expts = path_to_experiment(path, ExpClass)
    return list(dict.fromkeys(expts))


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


def measured_mod4_distribution(expts, e_is_high_I=True):
    """Per-time-point measured n-mod-4 distribution from single-shot data, grouped
    by readout storage. Same single-shot classification as the confusion-matrix
    cell: the last two readouts (stride read_num) give b0, b1; n_mod4 = b0 + 2*b1."""
    by_stor = {}
    for expt in expts:
        _ensure_multiparity_read_num(expt)  # backfill cfg.read_num if not saved (e.g. Qsim runs)
        ro_stor = expt.cfg.expt.ro_stor
        rn = expt.cfg.read_num
        qTest = expt.cfg.expt.qubits[0]
        threshold = expt.cfg.device.readout.threshold[qTest]
        xpts = np.asarray(expt.data["xpts"]).reshape(-1)

        p_points = []
        for j in range(len(xpts)):
            idata = np.asarray(expt.data["idata"][j])
            i_first  = idata[rn-2::rn]
            i_second = idata[rn-1::rn]
            if e_is_high_I:
                b0 = (i_first  > threshold).astype(int)
                b1 = (i_second > threshold).astype(int)
            else:
                b0 = (i_first  < threshold).astype(int)
                b1 = (i_second < threshold).astype(int)
            n_mod4 = b0 + 2 * b1
            p_points.append([np.mean(n_mod4 == k) for k in range(4)])

        entry = by_stor.setdefault(ro_stor, {"xpts": [], "p": []})
        entry["xpts"].append(xpts)
        entry["p"].append(np.asarray(p_points))

    out = {}
    for ro_stor, entry in by_stor.items():
        out[ro_stor] = {
            "xpts": np.concatenate(entry["xpts"]),
            "p_meas": np.concatenate(entry["p"], axis=0),  # (n_points, 4)
        }
    return out


def correct_distribution(p_meas, clip_renormalize=True):
    """Apply the confusion-matrix inverse to each time point's distribution."""
    p_corr = p_meas @ confusion_inv.T
    if clip_renormalize:
        p_corr = np.clip(p_corr, 0.0, None)
        p_corr = p_corr / p_corr.sum(axis=1, keepdims=True)
    return p_corr


def observables_from_distribution(p):
    """Multiparity observables from a (n_points, 4) modulo-4 distribution."""
    return {
        "p_mod0": p[:, 0], "p_mod1": p[:, 1], "p_mod2": p[:, 2], "p_mod3": p[:, 3],
        "nmod4_mean": p @ _mod_vals,
        "mean_parity_first":  p @ _parity_first_signs,
        "mean_parity_second": p @ _parity_second_signs,
    }


def process_multiparity(expts, clip_renormalize=True):
    """Raw + confusion-corrected per-readout observables, rebuilt from single shots."""
    measured = measured_mod4_distribution(expts)
    raw, corrected = {}, {}
    for ro_stor, d in measured.items():
        si = np.argsort(d["xpts"])
        x = d["xpts"][si]
        p_meas = d["p_meas"][si]
        p_corr = correct_distribution(p_meas, clip_renormalize=clip_renormalize)
        r = observables_from_distribution(p_meas); r["xpts"] = x
        c = observables_from_distribution(p_corr); c["xpts"] = x
        raw[ro_stor] = r
        corrected[ro_stor] = c
    return raw, corrected


def plot_multiparity_and_mod4(raw, corrected, params=None, title="", figsize=(14, 10), title_size=12):
    """2x2 Grid Layout: Parities on top (small), n mod 4 on bottom (large)."""
    fig = plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    ax_p1 = fig.add_subplot(gs[0, 0])      
    ax_p2 = fig.add_subplot(gs[0, 1])     
    ax_mod4 = fig.add_subplot(gs[1, :])  
    
    axes = [ax_p1, ax_p2, ax_mod4]
    
    for ro_stor in corrected:
        x = corrected[ro_stor]["xpts"]
        lbl = stor_label(ro_stor, params)
        
        # 1) n mod 4 (Bottom - ax_mod4)
        col = plot_thick_line_with_dots(ax_mod4, x, corrected[ro_stor]["nmod4_mean"], label=lbl)
        ax_mod4.plot(x, raw[ro_stor]["nmod4_mean"], "--", alpha=0.4, color=col)
        
        # 2) First Parity (Top Left - ax_p1)
        plot_thick_line_with_dots(ax_p1, x, corrected[ro_stor]["mean_parity_first"], label=lbl, color=col)
        ax_p1.plot(x, raw[ro_stor]["mean_parity_first"], "--", alpha=0.4, color=col)
        
        # 3) Second Parity (Top Right - ax_p2)
        plot_thick_line_with_dots(ax_p2, x, corrected[ro_stor]["mean_parity_second"], label=lbl, color=col)
        ax_p2.plot(x, raw[ro_stor]["mean_parity_second"], "--", alpha=0.4, color=col)
        
    ax_p1.set_title("First parity expectation")
    ax_p2.set_title("Second parity expectation")
    ax_mod4.set_title("Mean n mod 4 (dashed=raw, solid=corrected)")
    
    ax_p1.set_ylim(-1.05, 1.05)
    ax_p2.set_ylim(-1.05, 1.05)
    
    ax_p1.set_ylabel("Parity Expectation")
    ax_mod4.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
    
    for a in axes:
        a.set_xlabel("Floquet cycle")
        a.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        a.grid(alpha=0.25)
        a.legend(fontsize=10)
        
    if title:
        fig.suptitle(textwrap.fill(title, 60), fontsize=title_size)
        
    plt.tight_layout()
    plt.show()


def plot_nmod4_corrected(corrected, params=None, title="", figsize=(9, 7), title_size=12, custom_labels = None):
    """Clean single-panel confusion-corrected <n mod 4>."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    idx = 0
    for ro_stor in corrected:
        x = corrected[ro_stor]["xpts"]
        if custom_labels is None:
            lbl = stor_label(ro_stor, params)
        else:
            lbl =custom_labels[idx]
            idx += 1
        plot_thick_line_with_dots(ax, x, corrected[ro_stor]["nmod4_mean"], label=lbl)
        
    ax.set_xlabel("Floquet cycle")
    ax.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
    ax.legend()
    
    if title:
        ax.set_title(textwrap.fill(title, 60), fontsize=title_size)
        
    plt.tight_layout()
    plt.show()


def plot_multiparity(raw, corrected, params=None, title="", figsize=(20, 5), title_size=12):
    """3-panel raw (dashed) vs confusion-corrected (solid): <n mod 4>, parities."""
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    
    for ro_stor in corrected:
        x = corrected[ro_stor]["xpts"]
        lbl = stor_label(ro_stor, params)
        
        # Panel 0: nmod4_mean
        col = plot_thick_line_with_dots(ax[0], x, corrected[ro_stor]["nmod4_mean"], label=lbl)
        ax[0].plot(x, raw[ro_stor]["nmod4_mean"], "--", alpha=0.4, color=col)
        
        # Panel 1: mean_parity_first
        plot_thick_line_with_dots(ax[1], x, corrected[ro_stor]["mean_parity_first"], label=lbl, color=col)
        ax[1].plot(x, raw[ro_stor]["mean_parity_first"], "--", alpha=0.4, color=col)
        
        # Panel 2: mean_parity_second
        plot_thick_line_with_dots(ax[2], x, corrected[ro_stor]["mean_parity_second"], label=lbl, color=col)
        ax[2].plot(x, raw[ro_stor]["mean_parity_second"], "--", alpha=0.4, color=col)
        
    ax[0].set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
    ax[0].set_title("(dashed=raw, solid=corrected)")
    ax[1].set_title("First parity expectation")
    ax[2].set_title("Second parity expectation")
    ax[1].set_ylim(-1.05, 1.05)
    ax[2].set_ylim(-1.05, 1.05)
    
    for a in ax:
        a.set_xlabel("Floquet cycle")
        a.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        a.grid(alpha=0.25)
        a.legend(fontsize=10) # 폰트 크기 살짝 조정
        
    if title:
        fig.suptitle(textwrap.fill(title, 60), fontsize=title_size)
        
    plt.tight_layout()
    plt.show()


def build_confusion_matrix_from_calibration(cal_expts, e_is_high_I=True, point_idx=0):
    """C[i, j] = P(measured n=j | prepared Fock |i>) from the calibration runs.
    Each calibration experiment prepares one Fock state (cfg.expt.init_man_fock_state)."""
    counts = np.zeros((4, 4), dtype=float)
    for expt in cal_expts:
        true_n = int(expt.cfg.expt.init_man_fock_state)
        if true_n not in (0, 1, 2, 3):
            continue
        rn = expt.cfg.read_num
        qTest = expt.cfg.expt.qubits[0]
        threshold = expt.cfg.device.readout.threshold[qTest]
        idata = np.asarray(expt.data["idata"][point_idx])
        i_first  = idata[rn-2::rn]
        i_second = idata[rn-1::rn]
        if e_is_high_I:
            b0 = (i_first  > threshold).astype(int)
            b1 = (i_second > threshold).astype(int)
        else:
            b0 = (i_first  < threshold).astype(int)
            b1 = (i_second < threshold).astype(int)
        n_mod4 = b0 + 2 * b1
        for pred_n in range(4):
            counts[true_n, pred_n] += np.sum(n_mod4 == pred_n)
    return counts


def reconstruct_disorder_records(expts):
    """Group disorder-sweep experiments into per-(epsilon, realization) records
    using the disorder_* metadata saved in cfg.expt. Each record's `expts` are the
    floquet-cycle chunks belonging to that realization."""
    groups = defaultdict(list)
    for e in expts:
        cfg = e.cfg.expt
        if "disorder_epsilon" in cfg:
            eps = round(float(cfg["disorder_epsilon"]), 6)
        else:
            eps = round(float(np.max(np.abs(cfg["detunings"]))), 6)
        real = int(cfg.get("disorder_realization", 0))
        groups[(eps, real)].append(e)
    return [{"epsilon": eps, "realization": real, "expts": grp}
            for (eps, real), grp in groups.items()]


def disorder_ensemble_summary(records, obs_key="nmod4_mean", clip_renormalize=True, atol=1e-6):
    """For each epsilon and readout storage, stack the per-realization raw and
    confusion-corrected obs traces and compute mean / SEM across realizations.

    The averaging is pointwise over realizations (axis 0). Before stacking, every
    realization's (sorted) floquet grid is checked against the first realization's;
    a warning is printed and `aligned=False` recorded if they differ, so the mean
    is never silently computed over mismatched time points.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: {"x": [], "raw": [], "cor": []}))
    for rec in records:
        raw, cor = process_multiparity(rec["expts"], clip_renormalize=clip_renormalize)
        eps = rec["epsilon"]
        for ro_stor in cor:
            b = grouped[eps][ro_stor]
            b["x"].append(np.asarray(cor[ro_stor]["xpts"], dtype=float))
            b["raw"].append(np.asarray(raw[ro_stor][obs_key], dtype=float))
            b["cor"].append(np.asarray(cor[ro_stor][obs_key], dtype=float))

    summary = {}
    for eps, ro_dict in grouped.items():
        summary[eps] = {}
        for ro_stor, b in ro_dict.items():
            n = min(len(t) for t in b["cor"])
            xref = b["x"][0][:n]
            aligned = all(len(xx) >= n and np.allclose(xx[:n], xref, atol=atol)
                          for xx in b["x"])
            if not aligned:
                print(f"[warning] eps={eps} ro={ro_stor}: realization floquet grids "
                      f"differ -- averaging the first {n} aligned points only.")
            raw = np.vstack([t[:n] for t in b["raw"]])
            cor = np.vstack([t[:n] for t in b["cor"]])
            sl = np.sqrt(cor.shape[0])
            summary[eps][ro_stor] = {
                "x": xref, "n": cor.shape[0], "aligned": aligned,
                "raw_traces": raw, "raw_mean": raw.mean(0), "raw_sem": raw.std(0) / sl,
                "cor_traces": cor, "cor_mean": cor.mean(0), "cor_sem": cor.std(0) / sl,
            }
    return summary


def plot_disorder_ensemble(summary, ro_stor=0, corrected=True,
                           strengths=None, show_realizations=True, show_sem=True,
                           params=None, title="", figsize=(11, 6), dpi=130,
                           legend_fontsize=None, legend_ncol=2,
                           legend_loc=None, legend_frameon=None,
                           show_realization_count=False):
    """Ensemble mean of <n mod 4> vs floquet cycle, one curve per disorder strength.

    strengths : optional list of epsilon values to include (default: all)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for eps in sorted(summary):
        if strengths is not None and eps not in strengths:
            continue
        if ro_stor not in summary[eps]:
            continue
        b = summary[eps][ro_stor]
        x = b["x"]
        mean    = b["cor_mean"]   if corrected else b["raw_mean"]
        sem     = b["cor_sem"]    if corrected else b["raw_sem"]
        traces  = b["cor_traces"] if corrected else b["raw_traces"]

        label = fr"$\epsilon={eps:g}$"
        if show_realization_count:
            label += fr", R={b['n']}"
        line, = ax.plot(x, mean, lw=2, zorder=3, label=label)
        col = line.get_color()
        if show_realizations:
            for tr in traces:
                ax.plot(x, tr, lw=0.8, alpha=0.18, color=col, zorder=1)
        if show_sem and b["n"] > 1:
            ax.fill_between(x, mean - sem, mean + sem, alpha=0.13, color=col,
                            linewidth=0, zorder=2)

    ax.set_xlabel("Floquet cycle")
    ax.set_ylabel("$\\langle n_{\\rm dark}\\ {\\rm mod}\\ 4\\rangle$")
    ax.grid(alpha=0.25)
    legend_kwargs = {"ncol": legend_ncol}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    if legend_loc is not None:
        legend_kwargs["loc"] = legend_loc
    if legend_frameon is not None:
        legend_kwargs["frameon"] = legend_frameon
    ax.legend(**legend_kwargs)
    kind = "confusion-corrected" if corrected else "raw"
    full_title = (title + f" ({kind})") if title else f"disorder ensemble mean, readout={stor_label(ro_stor, params)} ({kind})"
    ax.set_title(textwrap.fill(full_title, 60), fontsize=8)
    fig.tight_layout()
    plt.show()


def _threshold_crossing_x(x, y, threshold, direction="auto"):
    """First x where y reaches threshold, using linear interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    threshold = float(threshold)

    if direction == "auto":
        direction = "down" if y[-1] < y[0] else "up"
    if direction not in ("down", "up"):
        raise ValueError("direction must be 'auto', 'down', or 'up'")

    if direction == "down":
        if y[0] <= threshold:
            return float(x[0])
        hit = np.where((y[:-1] >= threshold) & (y[1:] <= threshold))[0]
    else:
        if y[0] >= threshold:
            return float(x[0])
        hit = np.where((y[:-1] <= threshold) & (y[1:] >= threshold))[0]
    if len(hit) == 0:
        return np.nan

    i = int(hit[0])
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return float(x0)
    frac = (threshold - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def disorder_threshold_cycles(summary, threshold, ro_stor=0, corrected=True,
                              strengths=None, statistic="mean", direction="auto"):
    """Threshold-crossing Floquet cycle for each disorder strength.

    statistic='mean' crosses the ensemble mean curve. statistic='median_trace'
    takes the median of per-realization crossing cycles.
    """
    rows = []
    eps_list = sorted(summary) if strengths is None else strengths
    for eps in eps_list:
        if eps not in summary or ro_stor not in summary[eps]:
            continue
        b = summary[eps][ro_stor]
        x = b["x"]
        mean = b["cor_mean"] if corrected else b["raw_mean"]
        traces = b["cor_traces"] if corrected else b["raw_traces"]
        trace_cycles = np.array([
            _threshold_crossing_x(x, tr, threshold, direction=direction)
            for tr in traces
        ], dtype=float)

        if statistic == "mean":
            cycle = _threshold_crossing_x(x, mean, threshold, direction=direction)
        elif statistic == "median_trace":
            cycle = np.nanmedian(trace_cycles)
        else:
            raise ValueError("statistic must be 'mean' or 'median_trace'")

        finite_trace = np.isfinite(trace_cycles)
        if np.sum(finite_trace) > 1:
            cycle_sem = np.nanstd(trace_cycles[finite_trace], ddof=1) / np.sqrt(np.sum(finite_trace))
        else:
            cycle_sem = np.nan

        rows.append({
            "epsilon": float(eps),
            "cycle": float(cycle),
            "cycle_sem": float(cycle_sem),
            "n": int(b["n"]),
            "trace_cycles": trace_cycles,
        })

    return {
        "epsilon": np.array([r["epsilon"] for r in rows], dtype=float),
        "cycle": np.array([r["cycle"] for r in rows], dtype=float),
        "cycle_sem": np.array([r["cycle_sem"] for r in rows], dtype=float),
        "n": np.array([r["n"] for r in rows], dtype=int),
        "trace_cycles": [r["trace_cycles"] for r in rows],
        "threshold": float(threshold),
        "statistic": statistic,
    }


def fit_power_law(x, y):
    """Fit y = prefactor * x**alpha in log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        raise ValueError("Need at least two finite positive points for a power-law fit")

    logx = np.log(x[mask])
    logy = np.log(y[mask])
    xbar = np.mean(logx)
    ybar = np.mean(logy)
    denom = np.sum((logx - xbar) ** 2)
    if denom == 0:
        raise ValueError("Power-law fit needs at least two distinct x values")
    alpha = np.sum((logx - xbar) * (logy - ybar)) / denom
    log_prefactor = ybar - alpha * xbar
    pred = log_prefactor + alpha * logx
    ss_res = np.sum((logy - pred) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "alpha": float(alpha),
        "prefactor": float(np.exp(log_prefactor)),
        "r2": float(r2),
        "mask": mask,
    }


def plot_threshold_power_law(summary, threshold, ro_stor=0, corrected=True,
                             strengths=None, statistic="mean", direction="auto",
                             params=None, title=""):
    """Plot threshold-cycle scaling and extract cycle = A * epsilon**alpha."""
    data = disorder_threshold_cycles(
        summary,
        threshold=threshold,
        ro_stor=ro_stor,
        corrected=corrected,
        strengths=strengths,
        statistic=statistic,
        direction=direction,
    )
    eps = data["epsilon"]
    cycle = data["cycle"]
    cycle_sem = data["cycle_sem"]
    fit = fit_power_law(eps, cycle)
    valid = fit["mask"]
    eps_fit = eps[valid]
    eps_grid = np.geomspace(np.min(eps_fit), np.max(eps_fit), 200)
    cycle_grid = fit["prefactor"] * eps_grid ** fit["alpha"]
    yerr = np.where(np.isfinite(cycle_sem), cycle_sem, 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=130)
    ax[0].errorbar(eps, cycle, yerr=yerr, fmt="o", capsize=3, label="threshold cycle")
    ax[0].plot(eps_grid, cycle_grid, "--",
               label=f"fit: alpha={fit['alpha']:.3g}, R2={fit['r2']:.3f}")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Disorder strength")
    ax[0].set_ylabel("Floquet cycle at threshold")
    ax[0].grid(alpha=0.25, which="both")
    ax[0].legend()

    eps_pow = eps ** fit["alpha"]
    eps_pow_grid = eps_grid ** fit["alpha"]
    order = np.argsort(eps_pow_grid)
    ax[1].errorbar(eps_pow, cycle, yerr=yerr, fmt="o", capsize=3, label="data")
    ax[1].plot(eps_pow_grid[order], fit["prefactor"] * eps_pow_grid[order], "--",
               label="power-law guide")
    ax[1].set_xlabel(f"Disorder strength ** alpha (alpha={fit['alpha']:.3g})")
    ax[1].set_ylabel("Floquet cycle at threshold")
    ax[1].grid(alpha=0.25)
    ax[1].legend()

    kind = "confusion-corrected" if corrected else "raw"
    if not title:
        title = f"{stor_label(ro_stor, params)} {kind}, threshold={threshold:g}, {statistic}"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return data, fit, (fig, ax)


def rotate_iq(I, Q, angle_deg):
    
    theta = np.deg2rad(angle_deg)

    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Irot = I*np.cos(theta) - Q*np.sin(theta)
    Qrot = I*np.sin(theta) + Q*np.cos(theta)

    return Irot, Qrot


def state_label(state_idx):
    display_idx = state_idx
    state_prefix = "S"

    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    return f"State: {state_prefix}{display_idx}"


def unique_expts(nested_expts):
    return list(dict.fromkeys(expt for sub in nested_expts for expt in sub))

# %% [markdown]
# # Dark Mode Post Processing

# %%
job_date_list = [20260707]
start_num_list = [324]
finish_num_list = [326]
# disregarded_indicies = [10, 11, 12, 13, 14, 60, 59, 41, 42, 190, 135]
# job_date_list = [20260430]
# start_num_list = [13]JOB-20260513-01206_QsimBaseExperiment.h5
# finish_num_list = [15]
# disregarded_indicies = []\data\JOB-20260513-01206_QsimBaseExperiment.h5
path = hdf5_path_generator("260526_qsim_darkmode",
                           job_date_list, 
                           start_num_list,
                           finish_num_list, 
                           "QsimBaseExperiment")
exp_list = path_to_experiment(path, meas.qsim.qsim_base.QsimBaseExperiment)


config_dict = {'hardware_config': 'CFG-HW-20260703-00046',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20260707-00009',
 'floquet_storage_swap': 'CFG-FL-20260707-00029'}


station = MultimodeStation(
    user = 'Jonginn',
    experiment_name = "260701_Calibration",
    project = "AutoCalibrate",
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    floquet_file=config_dict['floquet_storage_swap'],

    log_measurements=False,
)

# %%
exp_list_params = get_dark_params(exp_list)
swap_stors, swap_man_dark, dark_swap_order = exp_list_params

# %%
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 24          
mpl.rcParams['axes.labelsize'] = 28      
mpl.rcParams['xtick.labelsize'] = 24     
mpl.rcParams['ytick.labelsize'] = 24     
mpl.rcParams['legend.fontsize'] = 20     

mpl.rcParams['axes.linewidth'] = 3.0    
mpl.rcParams['xtick.major.width'] = 3.0  
mpl.rcParams['ytick.major.width'] = 3.0  
mpl.rcParams['xtick.major.size'] = 10    
mpl.rcParams['ytick.major.size'] = 10    
mpl.rcParams['xtick.direction'] = 'in'   
mpl.rcParams['ytick.direction'] = 'in'

mpl.rcParams['legend.frameon'] = True   
mpl.rcParams['legend.fontsize'] = 24     
mpl.rcParams['legend.frameon'] = True         
mpl.rcParams['legend.framealpha'] = 0.8       
mpl.rcParams['legend.facecolor'] = 'white'    
mpl.rcParams['legend.edgecolor'] = 'black'   
mpl.rcParams['legend.fancybox'] = False       
mpl.rcParams['legend.borderaxespad'] = 1.0    
mpl.rcParams['legend.handlelength'] = 2.0

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit

# fig, ax = plt.subplots(1, 2, figsize=(18, 12))
fig, ax = plt.subplots(figsize=(12, 7))
scatter = False
combined_data = defaultdict(lambda: {'xpts': [], 'avgi': [], 'avgq': []})

floquet_cycle2us = 0

plot_time = False
do_fitting = False   # set False to skip the sine fit / frequency extraction

for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)


# --- sine fit: y(t) = A*sin(2*pi*f*t + phi) + C ; f in MHz (t in us), omega=2*pi*f ---




# Build the deduped, flattened experiment list this cell iterates over.
# Uses its own name so it never collides with the unique_expts() helper
# function defined elsewhere in this section.


for expt in exp_list:
    state_idx = expt.cfg.expt.ro_stor

    combined_data[state_idx]['xpts'].extend(expt.data['xpts'])
    combined_data[state_idx]['avgi'].extend(expt.data['avgi'])
    combined_data[state_idx]['avgq'].extend(expt.data['avgq'])

sine_fits = {}  # {label: {'I': {...}, 'Q': {...}}}

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices]  # * floquet_cycle2us # comment out if you don't want to convert cycle into time.
    if plot_time:
        x_sorted = np.array(data['xpts'])[sort_indices] * floquet_cycle2us
    i_sorted = np.array(data['avgi'])[sort_indices]
    q_sorted = np.array(data['avgq'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1
    if swap_man_dark == False:
        label_str = f"State: {state_prefix}{display_idx}"
    else:
        # print("True")
        if state_idx == 0:
            label_str = "Dark Mode"
        elif state_idx == dark_swap_order[1]:
            label_str = "Bright Mode"
        else:
            label_str = "Central Mode"
    line_i, = ax.plot(x_sorted, i_sorted, linestyle='-', linewidth=4.0, alpha=0.3, label=label_str)
    # line_q, = ax[1].plot(x_sorted, q_sorted, linestyle='-', linewidth=4.0, alpha=0.3, label=label_str)
    
    ax.plot(x_sorted, i_sorted, marker='o', linestyle='None', markersize=8, alpha=1.0, color=line_i.get_color())
    # ax[1].plot(x_sorted, q_sorted, marker='o', linestyle='None', markersize=8, alpha=1.0, color=line_q.get_color())

    # Only fit vs real time -- frequency in MHz / omega in Mrad/s are meaningful then.
    if plot_time and do_fitting:
        this = {}
        t_dense = np.linspace(np.min(x_sorted), np.max(x_sorted), 500)
        for chan, y_sorted, a in (("I", i_sorted, ax[0]), ("Q", q_sorted, ax[1])):
            fit = fit_sine(x_sorted, y_sorted)
            this[chan] = fit
            if fit is not None:
                a.plot(t_dense, _sine(t_dense, *fit['popt']), "k--", alpha=0.6,
                       label=f"{label_str} fit: f={fit['f']:.4f} MHz, "
                             f"$\\omega$={fit['omega']:.4f} Mrad/s")
        sine_fits[label_str] = this

if plot_time:
    ax.set_xlabel("time (us)")
    # ax[1].set_xlabel("time (us)")
else:
    ax.set_xlabel("Floquet Cycles")
    # ax[1].set_xlabel("Floquet Cycles")

ax.set_ylabel("Avg I (ADC unit)")
# ax[1].set_ylabel("Avg Q (ADC unit)")
ax.legend(loc = "right")
# ax.set_ylim([-60, -10])
# ax[1].legend()
plt.tight_layout()
plt.show()

# --- report extracted frequency / angular frequency (time-domain fits only) ---
if plot_time and do_fitting:
    print("Sine-fit results (t in us  ->  f in MHz, omega in Mrad/s):")
    for label, chans in sine_fits.items():
        for chan in ("I", "Q"):
            fit = chans.get(chan)
            if fit is None:
                print(f"  {label} [{chan}]: fit failed")
            else:
                print(f"  {label} [{chan}]: f = {fit['f']:.6f} MHz, "
                      f"omega = {fit['omega']:.6f} Mrad/s, "
                      f"A = {fit['A']:.3g}, phi = {fit['phi']:.3f} rad")

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

combined_data = defaultdict(lambda: {'xpts': [], 'avgi': [], 'avgq': []})

unique_expts = []
for expt in exp_list:
    if expt not in unique_expts:
        unique_expts.append(expt)

for expt in unique_expts:
    detune_idx = np.max(np.abs(expt.cfg.expt.detunings))

    combined_data[detune_idx]['xpts'].append(np.asarray(expt.data['xpts']))
    combined_data[detune_idx]['avgi'].append(np.asarray(expt.data['avgi']))
    combined_data[detune_idx]['avgq'].append(np.asarray(expt.data['avgq']))

averaged_data = {}

for detune_idx, data in combined_data.items():
    x_stack = np.stack(data['xpts'], axis=0)
    i_stack = np.stack(data['avgi'], axis=0)
    q_stack = np.stack(data['avgq'], axis=0)

    averaged_data[detune_idx] = {
        'xpts': np.mean(x_stack, axis=0),
        'avgi': np.mean(i_stack, axis=0),
        'avgq': np.mean(q_stack, axis=0),
    }

# %%
fig, ax = plt.subplots(1, 2, figsize=(18, 5))

for detune_idx, data in averaged_data.items():
    sort_indices = np.argsort(data['xpts'])

    x_sorted = data['xpts'][sort_indices]
    i_sorted = data['avgi'][sort_indices]
    q_sorted = data['avgq'][sort_indices]

    label_str = f"Detune: {detune_idx}"

    ax[0].plot(x_sorted, i_sorted, label=label_str)
    ax[1].plot(x_sorted, q_sorted, label=label_str)

ax[0].set_xlabel("Floquet Cycle")
ax[1].set_xlabel("Floquet Cycle")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg Q (ADC unit)")
# ax[0].legend()
ax[1].legend()

plt.show()

# %%
import matplotlib.colors as mcolors
import matplotlib.cm as cm

items = sorted(averaged_data.items(), key=lambda kv: kv[0])
detunes = np.array([k for k, _ in items], dtype=float)

positive_detunes = detunes[detunes > 0]
norm = mcolors.LogNorm(vmin=positive_detunes.min(), vmax=positive_detunes.max())
cmap = cm.turbo

fig, ax = plt.subplots(1, 2, figsize=(18, 5), constrained_layout=True)

for detune_idx, data in items:
    sort_indices = np.argsort(data['xpts'])

    x_sorted = data['xpts'][sort_indices]
    i_sorted = data['avgi'][sort_indices]
    q_sorted = data['avgq'][sort_indices]

    if detune_idx == 0:
        color = "black"
        linestyle = "--"
        linewidth = 2.5
        alpha = 1.0
        label = "Detune: 0"
    else:
        color = cmap(norm(detune_idx))
        linestyle = "-"
        linewidth = 2.0
        alpha = 0.9
        label = None

    ax[0].plot(x_sorted, i_sorted, color=color, lw=linewidth, ls=linestyle, alpha=alpha, label=label)
    ax[1].plot(x_sorted, q_sorted, color=color, lw=linewidth, ls=linestyle, alpha=alpha, label=label)

ax[0].set_xlabel("Floquet Cycle")
ax[1].set_xlabel("Floquet Cycle")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg Q (ADC unit)")

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Detune")

for a in ax:
    a.grid(alpha=0.25)
    a.legend()

plt.show()

# %%
cycle_to_time_us = 0.687521

fig, ax = plt.subplots(1,2, figsize=(18, 5))
data = combined_data[0]
sort_indices = np.argsort(data['xpts'])
x_sorted = np.array(data['xpts'])[sort_indices] * cycle_to_time_us
i_sorted = np.array(data['avgi'])[sort_indices]

data_1 = combined_data[2]
data_2 = combined_data[3]
i_sorted1 = np.array(data_1['avgi'])[sort_indices]
i_sorted2 = np.array(data_2['avgi'])[sort_indices]
i_sorted_sum = i_sorted1 + i_sorted2
ax[0].plot(x_sorted, i_sorted, label="Dark 1 number")
ax[1].plot(x_sorted, i_sorted_sum, label="Dark 0 number + Man 1 number")

ax[0].set_xlabel("Time (us)")
ax[1].set_xlabel("Time (us)")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg I (ADC unit)")
ax[0].legend()
ax[1].legend()
# ax[0].set_xlim([0, 100])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

fig, ax = plt.subplots(1, 2, figsize=(18, 5))


for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices] * cycle_to_time_us
    i_sorted = np.array(data['avgi'])[sort_indices]
    q_sorted = np.array(data['avgq'])[sort_indices]

    display_idx = state_idx
    if display_idx == 0:
        label_str = f"State: Dark 1"
    elif display_idx == 2:
        label_str = f"State: Man"
    elif display_idx == 3:
        label_str = f"State: Dark 2"
        
    
    ax[0].plot(x_sorted, i_sorted, label=label_str)
data = combined_data[0]
sort_indices = np.argsort(data['xpts'])
i_sorted = np.array(data['avgi'])[sort_indices]
ax[1].plot(x_sorted, i_sorted, label="Dark 1 number")
ax[1].plot(x_sorted, i_sorted_sum+80, label="Dark 0 + Man 1 number (added 80 for comparison)")

ax[0].set_xlabel("Floquet Cycle")
ax[1].set_xlabel("Floquet Cycle")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg I (ADC unit)")
ax[0].legend()
ax[1].legend()
# ax[0].set_xlim([0, 100])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from lmfit import Model



fig, ax = plt.subplots(1, 2, figsize=(18, 5))

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices] * cycle_to_time_us
    i_sorted = np.array(data['avgi'])[sort_indices]
    q_sorted = np.array(data['avgq'])[sort_indices]

    display_idx = state_idx
    if display_idx == 0:
        label_str = f"State: Dark 2"
    elif display_idx == 2:
        label_str = f"State: Man"
    elif display_idx == 3:
        label_str = f"State: Dark 1"
        
    ax[0].plot(x_sorted, i_sorted, label=label_str)

data = combined_data[0]
sort_indices = np.argsort(data['xpts'])
x_sorted = np.array(data['xpts'])[sort_indices] * cycle_to_time_us
i_sorted = np.array(data['avgi'])[sort_indices]

y_sum = np.array(i_sorted_sum)[sort_indices] + 80

dark1_fit = fit_t1(x_sorted, i_sorted)
sum_fit = fit_t1(x_sorted, y_sum)

x_fit = np.linspace(np.min(x_sorted), np.max(x_sorted), 1000)

dark1_t1 = dark1_fit.params['tau'].value
sum_t1 = sum_fit.params['tau'].value

ax[1].plot(x_sorted, i_sorted, label="Dark 2 number")
ax[1].plot(x_fit, dark1_fit.eval(x=x_fit), '--')

ax[1].plot(x_sorted, y_sum, label="Dark 1 + Man 1 number (added 80 for comparison)")
ax[1].plot(x_fit, sum_fit.eval(x=x_fit), '--')

ax[0].set_xlabel("Time (us)")
ax[1].set_xlabel("Time (us)")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg I (ADC unit)")
ax[0].legend()
ax[1].legend()
fig.suptitle(f"State initialized to S2 = D2-D1")
# ax[0].set_xlim([0, 100])
plt.show()

# %% [markdown]
# ## Disorder-dependent decay

# %%
fig, ax = plt.subplots(1, 2, figsize = (18, 5))

for scramble_expt in exp_list:
    state_idx = scramble_expt.cfg.expt.ro_stor
    detuning = np.max(scramble_expt.cfg.expt.detunings)
    print(detuning)
    state_prefix = "S"
    if state_idx == 0:
        state_prefix = "M"
        state_idx = 1
    ax[0].plot(scramble_expt.data['xpts'], scramble_expt.data['avgi'], label = F"State: {state_prefix}{state_idx}")
    ax[1].plot(scramble_expt.data['xpts'], scramble_expt.data['avgq'], label = F"State: {state_prefix}{state_idx}")
ax[0].set_xlabel("Floquet Cycle")
ax[1].set_xlabel("Floquet Cycle")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg Q (ADC unit)")
# ax[0].legend()
# ax[1].legend()

# %% [markdown]
# # Multiphoton Fock Post Processing
#
# Post-processing for the `Multi photon fock` subsection of
# `qsim_experiments.ipynb`. Every section below reconstructs the relevant
# `DarkBaseExperiment` multiparity data from `*.h5`, builds the per-readout
# modulo-4 observables from the single-shot data, and applies the readout
# confusion-matrix correction.
#
# The confusion matrix and all helper functions are defined **once** in the
# "Shared definitions" cells right below; the calibration cell can optionally
# recompute the matrix from data. Every experiment section after that is just a
# load cell plus a process-and-plot cell.

# %% [markdown]
# ## Shared definitions (confusion matrix + multiparity helpers)

# %%
# ---- Multiparity helper functions (shared by every section below) -----------
import matplotlib.pyplot as plt
import textwrap

# n_mod4 = b0 + 2*b1, with b0 = n % 2 (first parity), b1 = n // 2 (second parity).
# parity expectation = +1 for bit 0, -1 for bit 1  (i.e. 1 - 2*bit).


from collections import namedtuple

# Single source of truth for a dataset's dark-readout config. These values live
# in each experiment's cfg.expt, so we read them from the experiments themselves
# instead of re-declaring them by hand -- post-processing then always matches the
# data it is plotting.

# Module-level "current" dark params, set as a side effect of get_dark_params() so
# label/plot helpers can fall back to the most recently loaded dataset.

# ---- Multiparity plot labels + line statistics override --------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
from collections import namedtuple

MULTIPARITY_SHOW_LINE_STATS = True     # std/mean 표시 on/off
MULTIPARITY_STATS_FOR = "corrected"    # "corrected", "raw", "both"
MULTIPARITY_USE_DARK_LABELS = True     # large DM에서 끄고 싶으면 False

try:
    DarkParams
except NameError:
    DarkParams = namedtuple("DarkParams", ["swap_stors", "swap_man_dark", "dark_swap_order"])
    DARK_PARAMS = None

def _cfg_get(obj, key, default=None):
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _as_tuple(x):
    if x is None:
        return tuple()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)

def _flatten_expts(items):
    if items is None:
        return []
    out = []
    for x in items:
        if isinstance(x, (list, tuple, set)):
            out.extend(_flatten_expts(x))
        else:
            out.append(x)
    return out

def get_dark_params(expts):
    global DARK_PARAMS

    flat = _flatten_expts(expts)
    if len(flat) == 0:
        DARK_PARAMS = None
        return None

    def one(exp):
        c = exp.cfg.expt
        return DarkParams(
            _as_tuple(_cfg_get(c, "swap_stors", [])),
            bool(_cfg_get(c, "swap_man_dark", False)),
            _as_tuple(_cfg_get(c, "dark_swap_order", [])),
        )

    params = one(flat[0])
    for exp in flat:
        if one(exp) != params:
            raise Exception("Exps are not identical in swap_stors/swap_man_dark/dark_swap_order")

    DARK_PARAMS = params
    return params

def stor_label(ro_stor, params=None):
    ro_stor = int(ro_stor)

    if params is None:
        params = DARK_PARAMS

    if (not MULTIPARITY_USE_DARK_LABELS) or params is None or not params.swap_man_dark:
        return "M1" if ro_stor == 0 else f"S{ro_stor}"

    dark_order = list(params.dark_swap_order)

    if ro_stor == 0:
        return "Dark Mode"
    if len(dark_order) > 1 and ro_stor == int(dark_order[1]):
        return "Bright Mode"
    if len(dark_order) > 0 and ro_stor == int(dark_order[0]):
        return "Central Mode"

    return f"S{ro_stor}"

def _std_over_mean_text(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        return r"$\sigma/|\mu|$=nan"

    mu = np.mean(y)
    sig = np.std(y)

    if np.isclose(mu, 0.0):
        return r"$\sigma/|\mu|$=inf"

    return rf"$\sigma/|\mu|$={sig / abs(mu):.3g}"

def _line_label(label, y, kind):
    if not MULTIPARITY_SHOW_LINE_STATS:
        return str(label)
    if MULTIPARITY_STATS_FOR not in ("both", kind):
        return str(label)
    return f"{label} ({_std_over_mean_text(y)})"

def _raw_label(label, y):
    if MULTIPARITY_SHOW_LINE_STATS and MULTIPARITY_STATS_FOR in ("raw", "both"):
        return _line_label(f"{label} raw", y, "raw")
    return None

def plot_thick_line_with_dots(ax, x, y, label=None, color=None, stats_kind="corrected"):
    if label is not None:
        label = _line_label(label, y, stats_kind)

    line, = ax.plot(x, y, linestyle="-", linewidth=4.0, alpha=0.3,
                    label=label, color=color)
    col = line.get_color()
    ax.plot(x, y, marker="o", linestyle="None", markersize=8,
            alpha=1.0, color=col)
    return col

def plot_nmod4_corrected(corrected, params=None, title="", figsize=(9, 7),
                         title_size=12, custom_labels=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    idx = 0

    for ro_stor in corrected:
        x = corrected[ro_stor]["xpts"]
        y = corrected[ro_stor]["nmod4_mean"]

        if custom_labels is None:
            lbl = stor_label(ro_stor, params)
        else:
            lbl = custom_labels[idx]
            idx += 1

        plot_thick_line_with_dots(ax, x, y, label=lbl)

    ax.set_xlabel("Floquet cycle")
    ax.set_ylabel(r"$\langle n\ mod\ 4\rangle$")
    ax.legend()

    if title:
        ax.set_title(textwrap.fill(str(title), 60), fontsize=title_size)

    plt.tight_layout()
    plt.show()

def plot_multiparity(raw, corrected, params=None, title="", figsize=(20, 5), title_size=12):
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    keys = [
        ("nmod4_mean", r"$\langle n\ mod\ 4\rangle$", "(dashed=raw, solid=corrected)"),
        ("mean_parity_first", "parity value", "First parity expectation"),
        ("mean_parity_second", "parity value", "Second parity expectation"),
    ]

    for ro_stor in corrected:
        x = corrected[ro_stor]["xpts"]
        lbl = stor_label(ro_stor, params)

        for i, (key, _, _) in enumerate(keys):
            y_corr = corrected[ro_stor][key]
            y_raw = raw[ro_stor][key]

            col = plot_thick_line_with_dots(ax[i], x, y_corr, label=lbl)
            ax[i].plot(x, y_raw, "--", alpha=0.4, color=col,
                       label=_raw_label(lbl, y_raw))

    for i, (_, ylabel, title_i) in enumerate(keys):
        ax[i].set_title(title_i)
        ax[i].set_xlabel("Floquet cycle")
        ax[i].set_ylabel(ylabel)
        ax[i].axhline(0, color="k", linewidth=0.8, alpha=0.4)
        ax[i].grid(alpha=0.25)
        ax[i].legend(fontsize=10)

    ax[1].set_ylim(-1.05, 1.05)
    ax[2].set_ylim(-1.05, 1.05)

    if title:
        fig.suptitle(textwrap.fill(str(title), 60), fontsize=title_size)

    plt.tight_layout()
    plt.show()






import matplotlib.pyplot as plt
import textwrap

import matplotlib.gridspec as gridspec




import matplotlib.pyplot as plt
import textwrap

# %% [markdown]
# ## Multi photon fock - confusion matrix from calibration (optional)
#
# Mirrors the `Multi photon fock` calibration: load the Fock |0>,|1>,|2>,|3>
# preparation runs (`init_man_fock_state` = 0..3, `ro_stor = 0`, single floquet
# point) and build the confusion matrix from their single-shot data. Running this
# cell **overrides** the manual matrix above; skip it to keep the manual one.

# %%
# Set this to the job range of your Fock |0>..|3> calibration runs.
cal_expts = load_dark_experiments("260526_qsim_darkmode", [20260625], [280], [283])
# cal_expts = load_dark_experiments("260526_qsim_darkmode", [20260702], [356], [359])

print("loaded", len(cal_expts), "calibration experiments")

# %%


# Support both: compute from calibration data if it covers all four Fock states,
# otherwise keep the manually-entered matrix.
_counts = build_confusion_matrix_from_calibration(cal_expts)
if np.all(_counts.sum(axis=1) > 0):
    set_confusion_matrix(_counts)
    print("Confusion matrix computed from calibration data:")
else:
    set_confusion_matrix(confusion_matrix_manual)
    print("Calibration incomplete; using the manually-entered confusion matrix:")
print(np.round(confusion_matrix, 4))

# %%
probs = _counts / _counts.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(probs, vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(["0", "1", "2", "3"]); ax.set_yticklabels(["0", "1", "2", "3"])
ax.set_xlabel(r" $\langle n \;\mathrm{mod}\; 4\rangle$")
ax.set_ylabel("Prepared Fock state")
# wrapped_title = textwrap.fill(str(fname_list), width=60)
# ax.set_title(f"Modulo-4 readout confusion matrix {wrapped_title}")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{probs[i, j]:.3f}\n({int(_counts[i, j])})",
                ha="center", va="center")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$P(\mathrm{measured}\mid\mathrm{prepared})$")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Floquet with Initial Costate and Multiparity Readout

# %%
# Set this to the job range of your Floquet-with-initial-costate run.
# fic_expts = load_dark_experiments("260526_qsim_darkmode", [20260625], [271], [273])
# fic_expts = load_dark_experiments("260526_qsim_darkmode", [20260625], [274], [276])
fic_expts = load_dark_experiments("260526_qsim_darkmode", [20260707], [429], [433])
len(fic_expts)

fic_params = get_dark_params(fic_expts)
swap_stors, swap_man_dark, dark_swap_order = fic_params

# %%
MULTIPARITY_STATS_FOR = "both"
MULTIPARITY_SHOW_LINE_STATS = True

fic_raw, fic_corr = process_multiparity(fic_expts)
fic_title = str([exp.config_file for exp in fic_expts])

plot_multiparity(fic_raw, fic_corr, fic_params)
# plot_nmod4_corrected(fic_corr, fic_params, title=" (confusion-corrected)")

# %% [markdown]
# ### plot fluctuations

# %%
n_photons = [1, 2, 3]
fluctuations = {
    '40' : 
        {
            'n_mod_4' : 
                {
                    'raw': [0.0624, 0.0769, 0.115],
                    'corrected': [0.108, 0.123, 0.196]
                },
            'first_parity' : 
                {
                    'raw': [1.19, ],
                    'corrected': []
                },
            'second_parity' : 
                {
                    'raw': [],
                    'corrected': []
                },
        },
    '80' : 
        {
            'n_mod_4' : 
                {
                    'raw': [0.0369, 0.0305, 0.0423],
                    'corrected': [0.0635, 0.0457, 0.0629]
                },
            'first_parity' : 
                {
                    'raw': [1.19, ],
                    'corrected': []
                },
            'second_parity' : 
                {
                    'raw': [],
                    'corrected': []
                },
        },
    }

# %%
fig, ax = plt.subplots(figsize = (10, 10))
color_list = ['blue', 'green']
alpha_list = [1, 0.3]
for idx, keys in enumerate(['40', '80']):
    for data_idx, data_key in enumerate(['raw', 'corrected']):
        _color = color_list[idx] 
        ax.scatter(n_photons, 
                fluctuations[keys]['n_mod_4'][data_key],
                alpha = alpha_list[data_idx],
                color = _color,
                label = '$n_{frac}$'+f' = {keys}, {data_key}')
        
ax.set_xticks([1, 2, 3])
ax.set_ylabel("$\\sigma/\\mu$")
ax.set_xlabel("$n_{photon}$")
ax.legend()

# %% [markdown]
# ## Large Support DM

# %% [markdown]
# ### Bare

# %%
# "Bare" Large-Support-DM run (no dark-mode swap; meas_stors = [0, 2, 5, 6, 7]).
bare_expts = load_dark_experiments("260526_qsim_darkmode", [20260611], [1], [500])
len(bare_expts)

# %%
bare_raw, bare_corr = process_multiparity(bare_expts)
bare_title = str([exp.config_file for exp in bare_expts])
bare_params = get_dark_params(bare_expts)

plot_multiparity(bare_raw, bare_corr, bare_params,)
plot_nmod4_corrected(bare_corr, bare_params, " (confusion-corrected)")

# %% [markdown]
# ### DM readout

# %%
# "DM readout" Large-Support-DM run (init_stor = 2, swap_man_dark = True).
dmro_expts = load_dark_experiments("260526_qsim_darkmode", [2026625], [0], [300])
len(dmro_expts)

# %%
dmro_raw, dmro_corr = process_multiparity(dmro_expts)
dmro_title = str([exp.config_file for exp in dmro_expts])
dmro_params = get_dark_params(dmro_expts)

plot_multiparity(dmro_raw, dmro_corr, dmro_params, title=dmro_title)
plot_nmod4_corrected(dmro_corr, dmro_params, title=dmro_title + " (confusion-corrected)")

# %% [markdown]
# ### DM load and readout

# %%
# "DM load and readout" multiparity run.
dmlr_expts = load_dark_experiments("260526_qsim_darkmode", [20260625], [202], [206], meas.qsim.qsim_base.QsimBaseExperiment)
len(dmlr_expts)

# %%
dmlr_raw, dmlr_corr = process_multiparity(dmlr_expts)
dmlr_title = str([exp.config_file for exp in dmlr_expts])
dmlr_params = get_dark_params(dmlr_expts)

plot_multiparity(dmlr_raw, dmlr_corr, dmlr_params, title=dmlr_title)
plot_nmod4_corrected(dmlr_corr, dmlr_params, figsize = (16, 6))

# %%
# "DM load and readout" multiparity run.
dmlr_expt_list = []
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260611], [14], [14]))
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260611], [15], [15]))
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260611], [16], [16]))
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260611], [17], [17]))
len(dmlr_expt_list)

# %%
# "DM load and readout" multiparity run.
dmlr_expt_list = []
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260702], [378], [379]))
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260702], [380], [381]))
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260702], [382], [383]))
dmlr_expt_list.append(load_dark_experiments("260526_qsim_darkmode", [20260702], [384], [385]))
# dmlr_expts2 = load_dark_experiments("260526_qsim_darkmode", [20260611], [15], [15])
# dmlr_expts3 = load_dark_experiments("260526_qsim_darkmode", [20260611], [16], [16])
# dmlr_expts4 = load_dark_experiments("260526_qsim_darkmode", [20260611], [17], [17])
len(dmlr_expt_list)

# %%
dmlr_corr_list = []
dmlr_params_list = []
for dmlr_expts in dmlr_expt_list:
    dmlr_raw, dmlr_corr = process_multiparity(dmlr_expts)
    dmlr_title = str([exp.config_file for exp in dmlr_expts])
    dmlr_params = get_dark_params(dmlr_expts)
    
    dmlr_corr_list.append(dmlr_corr)
    dmlr_params_list.append(dmlr_params)

    
    plot_multiparity(dmlr_raw, dmlr_corr, dmlr_params, title=dmlr_title)

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
idx = 0
lbl_list = [3, 2, 1, 0]
for idx, dmlr_corr in enumerate(dmlr_corr_list):
    x = dmlr_corr[0]["xpts"]
    plot_thick_line_with_dots(ax, x, dmlr_corr[0]["nmod4_mean"], label=lbl_list[idx])
    
ax.set_xlabel("Floquet cycle")
ax.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
ax.legend(loc = "upper right")

    
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Large DM detuning dependent decay (disorder ensemble)
#
# Reconstructs the disorder-realization ensemble purely from the saved `*.h5`
# files. Each file carries its `disorder_epsilon` / `disorder_realization` in
# `cfg.expt`, so the per-realization records (and their floquet-cycle chunks) are
# regrouped automatically -- no execution code needed. Each realization is
# confusion-corrected at the single-shot level (via `process_multiparity`), then
# averaged over realizations to give the ensemble mean +/- SEM per disorder
# strength.

# %%
from collections import defaultdict
import textwrap

# %%
# Set this to the job range of your disorder sweep (DarkBaseExperiment files).
disorder_expts = load_dark_experiments("260526_qsim_darkmode", [20260704], [127], [487])
len(disorder_expts)

# %%
disorder_records = reconstruct_disorder_records(disorder_expts)
disorder_summary = disorder_ensemble_summary(disorder_records, obs_key="nmod4_mean")
disorder_params = get_dark_params(disorder_expts)

# Quick look at how many realizations were found per disorder strength.
for eps in sorted(disorder_summary):
    for ro_stor, b in disorder_summary[eps].items():
        print(f"epsilon={eps:<6g} readout={stor_label(ro_stor, disorder_params):<3s} R={b['n']}")

# %%
disorder_title = "Central + four storage modes: disorder ensemble mean"
ensemble_figsize = (22, 12)  # keep this ratio if you only want the whole image smaller
ensemble_dpi = 80           # lower dpi shrinks the rendered notebook image without changing relative style
ensemble_legend_fontsize = None  # None -> use mpl.rcParams['legend.fontsize']; try 12, 14, 16, ...
ensemble_legend_ncol = 2
ensemble_legend_loc = "upper center"  # e.g. "best", "upper right", "upper center"
ensemble_legend_frameon = None  # None -> use mpl.rcParams['legend.frameon']
show_realization_count = False  # False -> legend only shows epsilon

# Confusion-corrected ensemble (set corrected=False to see the raw ensemble).
plot_disorder_ensemble(
    disorder_summary,
    ro_stor=0,
    corrected=True,
    params=disorder_params,
    title=disorder_title,
    figsize=ensemble_figsize,
    dpi=ensemble_dpi,
    legend_fontsize=ensemble_legend_fontsize,
    legend_ncol=ensemble_legend_ncol,
    legend_loc=ensemble_legend_loc,
    legend_frameon=ensemble_legend_frameon,
    show_realization_count=show_realization_count,
)

# %%
# Threshold-crossing cycle vs disorder strength, plus power-law fit.
# This cell is self-contained: it only needs disorder_summary and disorder_params.
threshold = 2.5 * 0.7
ro_stor = 0
corrected = True
direction = "down"   # use "up" if the observable rises through threshold
strengths_to_plot = None  # None -> all strengths; e.g. [0.002, 0.005, 0.01]
plot_panels = "first"     # "first" or "both"
figsize = (7.2, 5.0)      # change this directly if labels/title collide
dpi = 130
figure_title = None       # e.g. fr"threshold={threshold:g}, readout={stor_label(ro_stor, disorder_params)}"

def crossing_x(x, y, threshold, direction="down"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if direction == "down":
        hit = np.where((y[:-1] >= threshold) & (y[1:] <= threshold))[0]
    else:
        hit = np.where((y[:-1] <= threshold) & (y[1:] >= threshold))[0]
    if len(hit) == 0:
        return np.nan
    i = int(hit[0])
    if y[i + 1] == y[i]:
        return float(x[i])
    return float(x[i] + (threshold - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))

eps_list = []
cycle_list = []
cycle_sem_list = []
if strengths_to_plot is None:
    selected_strengths = sorted(disorder_summary)
else:
    selected_strengths = []
    for target in strengths_to_plot:
        nearest = min(disorder_summary, key=lambda eps0: abs(float(eps0) - float(target)))
        selected_strengths.append(nearest)
    selected_strengths = list(dict.fromkeys(selected_strengths))

print("using strengths:", [f"{float(eps):g}" for eps in selected_strengths])

for eps in selected_strengths:
    if ro_stor not in disorder_summary[eps]:
        continue
    b = disorder_summary[eps][ro_stor]
    x = b["x"]
    mean = b["cor_mean"] if corrected else b["raw_mean"]
    traces = b["cor_traces"] if corrected else b["raw_traces"]
    trace_cycles = np.array([crossing_x(x, tr, threshold, direction) for tr in traces])
    finite_trace = np.isfinite(trace_cycles)
    eps_list.append(float(eps))
    cycle_list.append(crossing_x(x, mean, threshold, direction))
    cycle_sem_list.append(np.nanstd(trace_cycles[finite_trace], ddof=1) / np.sqrt(np.sum(finite_trace))
                          if np.sum(finite_trace) > 1 else np.nan)

eps = np.asarray(eps_list, dtype=float)
cycles = np.asarray(cycle_list, dtype=float)
cycle_sem = np.asarray(cycle_sem_list, dtype=float)
fit_mask = np.isfinite(eps) & np.isfinite(cycles) & (eps > 0) & (cycles > 0)
can_fit = np.sum(fit_mask) >= 2 and len(np.unique(eps[fit_mask])) >= 2
if can_fit:
    log_eps = np.log(eps[fit_mask])
    log_cycles = np.log(cycles[fit_mask])
    alpha = np.sum((log_eps - log_eps.mean()) * (log_cycles - log_cycles.mean())) / np.sum((log_eps - log_eps.mean())**2)
    log_A = log_cycles.mean() - alpha * log_eps.mean()
    A = np.exp(log_A)
    fit_cycles = A * eps[fit_mask]**alpha
    ss_tot = np.sum((log_cycles - log_cycles.mean())**2)
    r2 = 1 - np.sum((log_cycles - np.log(fit_cycles))**2) / ss_tot if ss_tot > 0 else np.nan
    eps_grid = np.geomspace(eps[fit_mask].min(), eps[fit_mask].max(), 300)
else:
    alpha, A, r2, eps_grid = np.nan, np.nan, np.nan, None

print("cycle = A * epsilon**alpha")
if can_fit:
    print(f"A = {A:.6g}, alpha = {alpha:.6g}, R^2 = {r2:.4f}")
else:
    print("Not enough finite positive strengths for a power-law fit; plotting data only.")

ncols = 1 if plot_panels == "first" else 2
fig, ax = plt.subplots(1, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
axes = np.atleast_1d(ax)
yerr = np.where(np.isfinite(cycle_sem), cycle_sem, 0)

axes[0].errorbar(eps, cycles, yerr=yerr, fmt="o", capsize=3, label="data")
if can_fit:
    axes[0].plot(eps_grid, A * eps_grid**alpha, "--", label=fr"$\alpha={alpha:.3g}$")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Disorder strength")
axes[0].set_ylabel("Floquet cycle at threshold")
axes[0].grid(alpha=0.25, which="both")
axes[0].legend()

if plot_panels == "both":
    if not can_fit:
        raise ValueError("plot_panels='both' requires at least two finite positive strengths for alpha")
    eps_alpha = eps**alpha
    eps_alpha_grid = eps_grid**alpha
    order = np.argsort(eps_alpha_grid)
    axes[1].errorbar(eps_alpha, cycles, yerr=yerr, fmt="o", capsize=3, label="data")
    axes[1].plot(eps_alpha_grid[order], A * eps_alpha_grid[order], "--", label="guide")
    axes[1].set_xlabel(fr"Disorder strength$^{{\alpha}}$, $\alpha={alpha:.3g}$")
    axes[1].set_ylabel("Floquet cycle at threshold")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

if figure_title:
    fig.suptitle(figure_title)
plt.show()

# %% [markdown]
# ### Check the disorder averaging
#
# Confirms, for one disorder strength, that (a) every realization shares the same
# floquet grid, (b) the plotted mean equals the pointwise mean of exactly the faint
# traces, and (c) shows the per-realization spread. The isolated single-strength
# plot lets you eyeball that the mean sits among its own realizations. A mean that
# lies above the fastest-decaying traces is expected: the ensemble mean is
# right-skew-pulled by the slower realizations, so it decays slower than the
# typical (median) trace.

# %%
eps_check = 0.02          # pick a disorder strength to inspect (e.g. the green curve)
ro = 0

b = disorder_summary[eps_check][ro]
traces, mean = b["cor_traces"], b["cor_mean"]

print(f"eps={eps_check}: R={b['n']} realizations, grids aligned = {b['aligned']}")
print("mean is exactly traces.mean(0):", np.allclose(mean, traces.mean(0)))
print()
print(f"at the last floquet-cycle point ({b['n']} realizations):")
print("  per-realization :", np.round(np.sort(traces[:, -1]), 3))
print(f"  ensemble mean   : {mean[-1]:.3f}")
print(f"  median trace    : {np.median(traces[:, -1]):.3f}   "
      f"(mean > median => right-skewed, mean decays slower than typical trace)")

# Isolated view of this one strength: mean + its own realizations only.
plot_disorder_ensemble(
    disorder_summary,
    ro_stor=ro,
    corrected=True,
    params=disorder_params,
    strengths=[eps_check],
    title=f"isolated check, eps={eps_check}",
    figsize=ensemble_figsize,
    dpi=ensemble_dpi,
    legend_fontsize=ensemble_legend_fontsize,
    legend_ncol=ensemble_legend_ncol,
    legend_loc=ensemble_legend_loc,
    legend_frameon=ensemble_legend_frameon,
    show_realization_count=show_realization_count,
)

# %% [markdown]
# ## GPT's suggestion on post processing iq rotation. Has not perused.

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------------------------------------------------
# 0. Load saved HistogramExperiment
# ------------------------------------------------------------
fname = r"D:\experiments\260511_qsim_darkmode\data\JOB-20260515-00230_HistogramExperiment.h5"

ss = meas.HistogramExperiment.from_h5file(fname)

print(ss.data.keys())
print(ss.cfg.expt)

if "angle" not in ss.data:
    raise KeyError(
        "There's no ss.data['angle']"
    )

angle_deg = float(np.ravel(ss.data["angle"])[0])
print("Using saved histogram IQ rotation angle [deg] =", angle_deg)


# ------------------------------------------------------------
# 1. IQ rotation only
# ------------------------------------------------------------




# ------------------------------------------------------------
# 2. If exp_list was acquired before this histogram calibration,
#    apply the histogram rotation.
# ------------------------------------------------------------
APPLY_HISTOGRAM_ROTATION = True

angle_for_data_deg = angle_deg if APPLY_HISTOGRAM_ROTATION else 0.0


# ------------------------------------------------------------
# 3. Combine exp_list and rotate avg I/Q only
# ------------------------------------------------------------
combined_data = defaultdict(lambda: {
    "xpts": [],
    "avgi": [],
    "avgq": [],
    "avgi_rot": [],
    "avgq_rot": [],
})

unique_expts = []
for expt in exp_list:
    if expt not in unique_expts:
        unique_expts.append(expt)

for expt in unique_expts:
    state_idx = expt.cfg.expt.ro_stor

    xpts = np.asarray(expt.data["xpts"]).ravel()
    avgi = np.asarray(expt.data["avgi"]).ravel()
    avgq = np.asarray(expt.data["avgq"]).ravel()

    avgi_rot, avgq_rot = rotate_iq(avgi, avgq, angle_for_data_deg)

    combined_data[state_idx]["xpts"].extend(xpts)
    combined_data[state_idx]["avgi"].extend(avgi)
    combined_data[state_idx]["avgq"].extend(avgq)
    combined_data[state_idx]["avgi_rot"].extend(avgi_rot)
    combined_data[state_idx]["avgq_rot"].extend(avgq_rot)


# ------------------------------------------------------------
# 4. Plot: original vs rotated avg I/Q
# ------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data["xpts"])

    x_sorted = np.asarray(data["xpts"])[sort_indices]

    i_sorted = np.asarray(data["avgi"])[sort_indices]
    q_sorted = np.asarray(data["avgq"])[sort_indices]

    i_rot_sorted = np.asarray(data["avgi_rot"])[sort_indices]
    q_rot_sorted = np.asarray(data["avgq_rot"])[sort_indices]

    label_str = state_label(state_idx)

    ax[0, 0].plot(x_sorted, i_sorted, label=label_str)
    ax[0, 1].plot(x_sorted, q_sorted, label=label_str)

    ax[1, 0].plot(x_sorted, i_rot_sorted, label=label_str)
    ax[1, 1].plot(x_sorted, q_rot_sorted, label=label_str)


ax[0, 0].set_title("Original Avg I")
ax[0, 1].set_title("Original Avg Q")
ax[1, 0].set_title("Rotated Avg I")
ax[1, 1].set_title("Rotated Avg Q")

ax[1, 0].set_xlabel("Floquet Cycle")
ax[1, 1].set_xlabel("Floquet Cycle")

ax[0, 0].set_ylabel("Avg I (ADC unit)")
ax[0, 1].set_ylabel("Avg Q (ADC unit)")
ax[1, 0].set_ylabel("Rotated Avg I (ADC unit)")
ax[1, 1].set_ylabel("Rotated Avg Q (ADC unit)")

for a in ax.ravel():
    a.legend()

plt.tight_layout()
plt.show()

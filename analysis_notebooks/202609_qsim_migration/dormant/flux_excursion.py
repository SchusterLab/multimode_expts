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
# # Cavity Ramsey / flux excursion post-processing (dormant)
#
# Relocated from `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 5-93 by the stage-2 notebook decomposition. **Dormant**: kept findable and
# diffable, not maintained. Covers the sweep plots, the HDF5 and
# manual/pickle-based cavity Ramsey post-processing, the excursion-transition
# gain-vs-current and fine sweeps, self-Kerr vs average I/Q, spectroscopy, and
# the manual Ramsey refitting.
#
# Relocation only -- definitions were not hoisted into modules and procedural
# blocks were not wrapped, per the stage-2 instructions. Note that this range
# reads from experiment-object pickles (`pickle_return`,
# `experiment_list_return`), which the repo treats as ephemeral debugging
# artifacts rather than durable data; that is part of why it is dormant.
#
# The library side is `experiments/qsim/cavity_ramsey_flux_excursion.py` and
# `experiments/qsim/t2_cavity_fluxexcursion.py`. Sibling dormant notebooks:
# `wigner.py`, `dark_mode.py`, `pulse_scratch.py`.

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
# Moved to deprecated/ in MBR redesign step 7e (docs/qsim/mbr_step7_plan.md).
from experiments.qsim.deprecated.encoding_spectroscopy import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.deprecated.legacy_mbr import MBROrthogonalityExperiment
from experiments.qsim.deprecated.legacy_mbr import MBRPropagatorExperiment

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


def pickle_return(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def experiment_list_return(experiment_name,
                           job_date,
                           job_start_num,
                           job_finish_num,
                           basedir = BASE_DIR,
                           verbose = False):
    object_directory = os.path.join(basedir, experiment_name, "expt_objs")
    exp_list_to_return = []
    job_path = f"JOB-{job_date}"
    for job_num in range(job_start_num, job_finish_num + 1):
        filepath = os.path.join(object_directory, f"{job_path}-{job_num:05d}_expt.pkl")
        try:
            exp_list_to_return.append(pickle_return(filepath))
        except Exception as e:
            if verbose == True:
                print(f"Error loading {filepath}: {e}")
            continue
    return exp_list_to_return


def retrieve_from_multiple_jobs(experiment_nane,
                                job_date_list,
                                job_start_num_list,
                                job_finish_num_list,
                                basedir = BASE_DIR):
    all_exp_list = []
    for job_date, job_start_num, job_finish_num in zip(job_date_list,job_start_num_list,job_finish_num_list):
        exp_list = experiment_list_return(experiment_nane,
                                          job_date,
                                          job_start_num,
                                          job_finish_num,
                                          basedir)
        all_exp_list.extend(exp_list)
    return all_exp_list


def check_program_class(obj,
                        class_name):
    if obj.prog.__class__.__name__ == class_name:
        return True
    return False


def job_id_generator(job_date, job_start_num, job_finish_num, step=1):
    starts = [job_start_num] if isinstance(job_start_num, (int, np.integer)) else list(job_start_num)
    finishes = [job_finish_num] if isinstance(job_finish_num, (int, np.integer)) else list(job_finish_num)
    dates = [job_date] * len(starts) if isinstance(job_date, (str, int, np.integer)) else list(job_date)
    steps = [step] * len(starts) if isinstance(step, (int, np.integer)) else list(step)
    if not (len(dates) == len(starts) == len(finishes) == len(steps)): raise ValueError('job range arguments must have matching lengths')
    return [f'JOB-{date}-{job:05d}' for date, start, finish, stride in zip(dates, starts, finishes, steps) for job in range(int(start), int(finish) + 1, int(stride))]


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


def get_inlier_indices(x, y, threshold=2.0):
    valid = np.isfinite(x) & np.isfinite(y)
    current_valid = valid.copy()
    indices = np.arange(len(x))
    
    for _ in range(10):  
        x_fit = x[current_valid]
        y_fit = y[current_valid]
        
        if len(x_fit) < 2:
            break 
            
        p = np.polyfit(x_fit, y_fit, 1)
        y_pred = np.polyval(p, x)
        
        residuals = np.abs(y - y_pred)
        
        mad = np.median(residuals[current_valid])
        sigma = 1.4826 * mad
        
        if sigma < 1e-10:  
            break
            
        new_valid = valid & (residuals < threshold * sigma)
        
        if np.array_equal(new_valid, current_valid):
            break 
            
        current_valid = new_valid
        
    return np.where(current_valid)[0]


def plot_freq_line_cut(target_freq):
    plot_gains = []
    plot_kerrs = []
    
    for expt in excursion_list:
        is_good = not getattr(expt, 'is_bad_data', False)
        is_target = (expt.cfg.expt.get('kerr_freq', np.nan) == target_freq)
        
        
        if is_target and is_good:
            plot_gains.append(expt.cfg.expt['kerr_gain'])
            plot_kerrs.append(np.abs(expt.data['Kerr'])) 
            
    if not plot_gains:
        print(f"No valid data for kerr freq = {target_freq}")
        return
        
    plt.figure(figsize=(8, 5))
    plt.scatter(plot_gains, plot_kerrs, s=50, label=f'Freq = {target_freq}')
    
    plt.xlabel('Gain [a.u.]', fontsize=12)
    plt.ylabel('|Kerr|', fontsize=12)
    plt.title(f'Kerr vs Gain Line Cut (Freq: {target_freq})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def estimate_periodicity(y, sampling_rate=1.0):
        # Compute FFT
        fft = np.fft.fft(y - np.mean(y))  # remove DC offset
        freqs = np.fft.fftfreq(len(y), d=1/sampling_rate)

        # Only take the positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = np.abs(fft[pos_mask])

        # Find the dominant frequency
        dominant_freq = freqs[np.argmax(power)]

        # Convert frequency to period
        estimated_period = 1 / dominant_freq if dominant_freq != 0 else np.inf
        return estimated_period


def fit_model(x, alpha2, f, scale, offset, phase):
    return scale * np.exp(-2.0*alpha2*(1.0 + np.cos(2.0*np.pi*f*x - phase))) + offset


def fit_func(x, kc, delta):
    return  kc * x + delta


def estimate_phase(y, debug = False):
    if debug == True:
        print("y[0]", y[0], "min", np.min(y), "max", np.max(y))
    return np.abs(y[0] - np.min(y)) / np.abs(np.max(y) - np.min(y)) * np.pi/2


def normalize(z, exp_obj):
    Ig = exp_obj.cfg.device.readout.Ig[0]
    Ie = exp_obj.cfg.device.readout.Ie[0]
    return (z - Ig) / (Ie - Ig)


def _cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _normalize_from_obj_or_cfg(obj, avgi_2d):
    if hasattr(obj, "normalize") and callable(getattr(obj, "normalize")):
        try:
            return obj.normalize(avgi_2d)
        except Exception:
            pass

    Ig = _cfg_get(obj.cfg, "device.readout.Ig", None)
    Ie = _cfg_get(obj.cfg, "device.readout.Ie", None)
    if Ig is None or Ie is None:
        return np.array(avgi_2d, float)

    Ig = float(Ig[0] if isinstance(Ig, (list, tuple, np.ndarray)) else Ig)
    Ie = float(Ie[0] if isinstance(Ie, (list, tuple, np.ndarray)) else Ie)
    denom = (Ie - Ig)
    if denom == 0:
        return np.array(avgi_2d, float)
    return (np.array(avgi_2d, float) - Ig) / denom


def kerr_ramsey_model_phi(x, alpha2, f, phi0, scale, offset):
    return scale * np.exp(-2.0 * alpha2 * (1.0 + np.cos(2.0*np.pi*f*x + phi0))) + offset


def linear_model(x, kc, delta):
    return kc * x + delta


def post_refit_one_cavity_ramsey_v2(
    obj,
    *,
    smooth_sigma=1.5,
    rsq_threshold=0.2,
    f_window_mhz=5.0,       
    # alpha2_mode="bounded",     
    alpha2_bound_ratio=(0.3, 3.0),
    use_smoothed_for_fit=True,
    verbose=False,
    method_setting = None
):
    # x/y
    x = np.array(obj.data.get("xpts", _cfg_get(obj.cfg, "expt.kerr_lengths")), float)
    y = np.array(obj.data.get("gain_list", _cfg_get(obj.cfg, "expt.displace_gains")), float)

    avgi_2d = np.array(obj.data["g_avgi"], float)
    n_lines, n_x = avgi_2d.shape

    z = _normalize_from_obj_or_cfg(obj, avgi_2d)

    virtual_freq = float(_cfg_get(obj.cfg, "expt.ramsey_freq", 0.0) or 0.0)

    g2a = _cfg_get(obj.cfg, "device.manipulate.gain_to_alpha", None)
    man_mode_no = obj.cfg.expt.get('man_mode_no', 1)
    if isinstance(g2a, (list, tuple, np.ndarray)):
        g2a = g2a[man_mode_no-1] if len(g2a) else None

    model = Model(kerr_ramsey_model_phi)

    fit_lids, fit_results, z_fits, rsqs = [], [], [], []
    alpha2_fits, f_fits, phi0_fits = [], [], []

    for lid in range(n_lines):
        line = np.array(z[lid, :], float)
        line_sm = gaussian_filter1d(line, sigma=smooth_sigma)

        yfit = line_sm if use_smoothed_for_fit else line

        off0 = float(np.nanmin(yfit))
        sc0 = float(np.nanmax(yfit) - off0)
        sc0 = max(sc0, 1e-3)

        if g2a is not None:
            alpha_guess = float(y[lid]) * float(g2a)
            alpha2_0 = max(alpha_guess*alpha_guess, 0.0)
        else:
            alpha2_0 = 0.2  # fallback

        params = model.make_params(
            alpha2=alpha2_0,
            f=virtual_freq,
            phi0=0.0,
            scale=sc0,
            offset=off0,
        )

        # if alpha2_mode == "fixed":
        params["alpha2"].set(value=alpha2_0, vary=False)
        # elif alpha2_mode == "bounded":
        #     lo, hi = alpha2_bound_ratio
        #     params["alpha2"].set(value=alpha2_0,
        #                          min=max(lo*alpha2_0, 0.0),
        #                          max=max(hi*alpha2_0, 1e-12),
        #                          vary=True)
        # elif alpha2_mode == "free":
        #     params["alpha2"].set(value=alpha2_0, min=0.0, vary=True)
        # else:
        #     raise ValueError("alpha2_mode must be fixed|bounded|free")
        if f_window_mhz == None:
            params["f"].set(value=virtual_freq)
        else:
            params["f"].set(value=virtual_freq #)
                            , min=virtual_freq - f_window_mhz
                            , max=virtual_freq + f_window_mhz)

        params["phi0"].set(value=0.0, min=-np.pi, max=np.pi)

        params["scale"].set(value=sc0, min=0.0, max=2.0)
        params["offset"].set(value=off0)#, min=-0.5, max=1.5)

        try:
            # res = model.fit(yfit, params, x=x[lid], nan_policy="omit",
            if method_setting == "basinhopping" :
                res_global = model.fit(yfit, params, x=x[lid], nan_policy="omit", 
                                    method="basinhopping")
                res = model.fit(yfit, res_global.params, x=x[lid], nan_policy="omit", 
                                method="leastsq")
            else:
                res = model.fit(yfit, params, x=x[lid], nan_policy="omit", 
                                    method="leastsq")
            fit_lids.append(lid)
            fit_results.append(res)
            z_fits.append(res.best_fit)

            alpha2_fits.append(float(res.params["alpha2"].value))
            f_fits.append(float(res.params["f"].value))
            phi0_fits.append(float(res.params["phi0"].value))
            rsq = float(getattr(res, "rsquared", np.nan))
            rsqs.append(rsq)

            if verbose:
                print(f"lid={lid:3d} rsq={rsq:.3f} f={f_fits[-1]:.4f} phi0={phi0_fits[-1]:+.3f} alpha2={alpha2_fits[-1]:.3g}")

        except Exception as e:
            if verbose:
                print(f"lid={lid:3d} failed: {e!r}")
            continue

    fit_lids = np.array(fit_lids, int)
    rsqs = np.array(rsqs, float)
    alpha2_fits = np.array(alpha2_fits, float)
    f_fits = np.array(f_fits, float)
    phi0_fits = np.array(phi0_fits, float)
    z_fits = np.array(z_fits, float) if len(z_fits) else np.zeros((0, n_x), float)

    fit_good = np.isfinite(rsqs) & (rsqs > rsq_threshold)

    alpha2_good = alpha2_fits[fit_good]
    f_det = (f_fits[fit_good] - virtual_freq)

    # linear fit
    kc = np.nan
    delta = np.nan
    linear_fit_result = None
    try:
        if len(alpha2_good) >= 2:
            lm = Model(linear_model, independent_vars=["x"])
            lp = lm.make_params(kc=1.0, delta=0.0)
            linear_fit_result = lm.fit(f_det, lp, x=alpha2_good, nan_policy="omit")
            kc = float(linear_fit_result.params["kc"].value)
            delta = float(linear_fit_result.params["delta"].value)
    except Exception:
        pass

    obj.fit_results = AttrDict(dict(
        alpha2=alpha2_good,
        f=f_det,
        kc=kc,
        delta=delta,
        linear_fit_result=linear_fit_result,

        results=fit_results,
        z_fits=z_fits,
        rsquared=rsqs,
        fit_lids=fit_lids,
        fit_good=fit_good,

        virtual_freq=virtual_freq,
        f_window_mhz=f_window_mhz,
        # alpha2_mode=alpha2_mode,
    ))

    return obj.fit_results

# %% [markdown]
# # Sweep Plot

# %%
job_date_list = [20260730]
start_num_list = [105]
finish_num_list = [105]



spect_list = retrieve_from_multiple_jobs("260729_Calibration_0.5mA", 
                                        job_date_list,
                                        start_num_list, 
                                        finish_num_list)

# %%
spect_list[0].cfg.expt

# %%
fig, ax = plt.subplots(figsize = (12, 4))
ax.plot(spect_list[0].data['xpts'], spect_list[0].data['avgi'])
# ax.set_xlim([2100, 2300])

# %% [markdown]
# # Cavity Ramsey Post Processing using HDF5

# %%
from experiments.qsim.cavity_ramsey_flux_excursion import CavityFluxExcursionRamseyExperiment

job_date_list = [20260424, 20260416, 20260415,20260421, 20260422, 20260423]
start_num_list = [0, 0, 0, 0, 0, 0]
finish_num_list = [500, 500, 500, 500, 500, 500]
disregarded_indicies = [10, 11, 12, 13, 14, 60, 59, 41, 42, 190, 135]
# job_date_list = [20260430]
# start_num_list = [13]
# finish_num_list = [15]
# disregarded_indicies = []
path = hdf5_path_generator("260403_kerr_excusion_redone",
                           job_date_list, 
                           start_num_list,
                           finish_num_list, "KerrCavityRamseyExcursionExperiment")
exp_list = path_to_experiment(path, CavityFluxExcursionRamseyExperiment)

# %%
extracted_kerrs = []
extracted_kerr_err = []
extracted_freqs = []
extracted_gains = []
extracted_deltas = []
for idx, exp in enumerate(tqdm(exp_list)):
    exp.analyze()
    if idx in disregarded_indicies:
        continue
    try:
        extracted_freqs.append(exp.cfg.expt['kerr_freq'])
        extracted_gains.append(exp.cfg.expt['kerr_gain'])
        extracted_kerrs.append(exp.fit_results['kc'])
        extracted_kerr_err.append(exp.fit_results['linear_fit_result'].params['kc'].stderr)
        extracted_deltas.append(exp.fit_results['delta'])
    except:
        disregarded_indicies.append(idx)

# %%
target_freq = 20
target_gain = 12000
gain_x = []
kerr_y = []
whatisidx = 0
for kerr, gain, freq in zip(tqdm(extracted_kerrs), extracted_gains, extracted_freqs):
    if freq == target_freq:
        gain_x.append(gain)
        kerr_y.append(np.abs(kerr))
        if gain == target_gain:
            offset = 0
            for i in disregarded_indicies:
                if not i > whatisidx:
                    offset +=1
            print(whatisidx + offset)
    plt.scatter(gain_x, kerr_y)
    whatisidx = whatisidx + 1

# %%
extracted_gains.index(10000)

# %%
print(f"gain: {gain_x} ")
print(f"Kerr: {kerr_y} ")

# %%
idx = 29
disregard = False
print(exp_list[idx].fit_results['kc'])
try:
    exp_list[idx].display()
except:
    pass
if disregard:
    exp_list[idx].fit_results['kc'] = np.nan
    extracted_kerrs = []
    extracted_freqs = []
    extracted_gains = []
    extracted_kerr_err = []
    for idx, exp in enumerate(tqdm(exp_list)):
        if idx in disregarded_indicies:
            continue
        extracted_freqs.append(exp.cfg.expt['kerr_freq'])
        extracted_gains.append(exp.cfg.expt['kerr_gain'])
        extracted_kerr_err.append(exp.fit_results['linear_fit_result'].params['kc'].stderr)
        extracted_kerrs.append(exp.fit_results['kc'])
    disregarded_indicies.append(idx)

# %%
from collections import defaultdict

target_freq_list = [50, 60, 70, 80, 90, 100]
conversion_factor_list = [3.0334e-05, 2.6487e-05, 2.4430e-05, 2.4690e-05, 2.3238e-05, 2.4358e-05]
data_by_freq = defaultdict(lambda: {'gain_x': [], 'kerr_y': [], 'cc': []})

for idx, (kerr, gain, freq) in enumerate(zip(tqdm(extracted_kerrs), extracted_gains, extracted_freqs)):
    if freq in target_freq_list:
        idx = target_freq_list.index(freq)
        data_by_freq[freq]['gain_x'].append(gain)
        data_by_freq[freq]['cc'].append(gain * conversion_factor_list[idx])
        data_by_freq[freq]['kerr_y'].append(np.abs(kerr))
    

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

for freq, data in data_by_freq.items():
    ax[0].scatter(data['cc'], data['kerr_y'], label=f'Freq: {freq}')
    ax[1].scatter(data['gain_x'], data['kerr_y'], label=f'Freq: {freq}')

ax[0].set_xlabel('Converted Current (mA)')
ax[1].set_xlabel('Gain')
ax[0].set_ylabel('|Kerr|')
ax[0].legend()
ax[1].legend()
plt.show()

# %%
data_by_freq

# %% [markdown]
# ### whatt

# %%

target_freq_list = [50, 60, 70, 80, 90, 100]
conversion_factor_list = [2.8e-05, 2.4627e-05, 2.4430e-05, 2.4690e-05, 2.6054e-05, 2.6893e-05]

job_date_list = [20260427]
start_num_list = [390]
finish_num_list = [408]


transition_list = retrieve_from_multiple_jobs("260403_kerr_excusion_redone", 
                                              job_date_list,
                                              start_num_list, 
                                              finish_num_list)

y = transition_list[0].cfg.expt.kerr_gains
x = [trans.cfg.expt.kerr_freq for trans in transition_list]
z = [trans.data['avgi'][:, 0] for trans in transition_list]

# %%
fig, ax = plt.subplots(1, 2, figsize = (14, 6))
y = transition_list[0].cfg.expt.kerr_gains
x = [trans.cfg.expt.kerr_freq for trans in transition_list]
z = [trans.data['avgi'][:, 0] for trans in transition_list]
offset = 4
for i in range(6):
    current_conv = np.array(y) * conversion_factor_list[i]
    ax[0].plot(y, z[i+offset])
    ax[0].set_xlim([10000, 20000])
    ax[1].plot(current_conv, z[i+offset])
    ax[1].set_xlim([0.2, 0.6])

# %%
import numpy as np
import matplotlib.pyplot as plt

y = transition_list[0].cfg.expt.kerr_gains
x = [trans.cfg.expt.kerr_freq for trans in transition_list]
z = [trans.data['avgi'][:, 0] for trans in transition_list]
offset = 4

target_freqs = x[offset : offset+6]

Z_data = np.column_stack([z[i + offset] for i in range(6)])

X_grid = np.zeros_like(Z_data)
Y_current_grid = np.zeros_like(Z_data)
Y_gain_grid = np.zeros_like(Z_data)

for i in range(6):
    X_grid[:, i] = target_freqs[i]  
    Y_current_grid[:, i] = np.array(y) * conversion_factor_list[i]
    Y_gain_grid[:, i] = y           


y_min, y_max = np.min(Y_current_grid), np.max(Y_current_grid)
num_points = len(y) 
flat_current_axis = np.linspace(y_min, y_max, num_points)

# 2. 보간된 Z 데이터를 담을 빈 2D 행렬 생성
Z_interp = np.zeros((num_points, 6))

for i in range(6):
    curr_col = Y_current_grid[:, i]
    z_col = Z_data[:, i]
    
    sort_idx = np.argsort(curr_col)
    
    Z_interp[:, i] = np.interp(flat_current_axis, curr_col[sort_idx], z_col[sort_idx])


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sc1 = ax[0].pcolormesh(target_freqs, flat_current_axis, Z_interp, cmap='viridis', shading='nearest')

sc2 = ax[1].pcolormesh(X_grid, Y_gain_grid, Z_data, cmap='viridis', shading='nearest')

fig.colorbar(sc1, ax=ax[0], label='Signal (avgi)')
fig.colorbar(sc2, ax=ax[1], label='Signal (avgi)')

ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('Converted Current (mA)')
ax[0].set_title('Frequency vs Current (Flat Interpolation)')
ax[0].set_xticks(target_freqs)

ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Gain')
ax[1].set_title('Frequency vs Gain (pcolormesh)')
ax[1].set_xticks(target_freqs)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

all_freqs = []
all_gains = []
all_ccs = []
all_kerrs = []

for kerr, gain, freq in zip(tqdm(extracted_kerrs), extracted_gains, extracted_freqs):
    if freq in target_freq_list:
        idx = target_freq_list.index(freq)
        
        all_freqs.append(freq)          # X축 데이터
        all_gains.append(gain)          # Y축 데이터 1
        all_ccs.append(gain * conversion_factor_list[idx])  # Y축 데이터 2
        all_kerrs.append(np.abs(kerr))  # Color 기준 데이터 (Z축)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# X축: Frequency, Y축: Current/Gain, Color(c): |Kerr|
# s(점 크기)를 조금 키우면 2D 컬러 맵처럼 보기 더 좋습니다. (필요시 조절)
sc1 = ax[0].scatter(all_freqs, all_ccs, c=all_kerrs, cmap='viridis', s=1000)
sc2 = ax[1].scatter(all_freqs, all_gains, c=all_kerrs, cmap='viridis', s=1000)

# 컬러바(Colorbar)는 이제 주파수가 아니라 |Kerr|의 크기를 나타냅니다.
fig.colorbar(sc1, ax=ax[0], label='|Kerr| Magnitude')
fig.colorbar(sc2, ax=ax[1], label='|Kerr| Magnitude')

# 첫 번째 그래프 세팅
ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('Converted Current (mA)')
ax[0].set_title('Frequency vs Current (Color = |Kerr|)')
ax[0].set_xticks(target_freq_list)  # X축 눈금을 타겟 주파수로 딱 떨어지게 설정

# 두 번째 그래프 세팅
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Gain')
ax[1].set_title('Frequency vs Gain (Color = |Kerr|)')
ax[1].set_xticks(target_freq_list)

plt.tight_layout()
plt.show()

# %%
from matplotlib.colors import Normalize

KERR_RANGE_MIN = 0.0
KERR_RANGE_MAX = 0.03
COLORMAP_NAME = 'viridis' 

print("Generating 2D Kerr Map...")


err_count_fitting = 0
err_count_object_type = 0

unique_freqs = np.unique(extracted_freqs)
unique_gains = np.unique(extracted_gains)

freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
gain_to_idx = {g: i for i, g in enumerate(unique_gains)}

kerr_matrix = np.full((len(unique_gains), len(unique_freqs)), np.nan)

for freq, gain, kerr in zip(extracted_freqs, extracted_gains, extracted_kerrs):
    row = gain_to_idx[gain]
    col = freq_to_idx[freq]
    kerr_matrix[row, col] = np.abs(kerr)
    
print("\n[Method 1] Plotting Exact Measured Points (Pcolormesh)")

fig1, ax1 = plt.subplots(figsize=(9, 6))

cmap = plt.get_cmap(COLORMAP_NAME).copy()
cmap.set_bad(color='white')

pc = ax1.pcolormesh(unique_freqs, unique_gains, kerr_matrix, 
                    shading='auto', cmap=cmap, 
                    vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)

cbar = fig1.colorbar(pc, ax=ax1)
cbar.set_label('|$\\Delta$Kerr| (MHz) ')

ax1.set_xlabel('Excursion Frequency (MHz)')
ax1.set_ylabel('Excursion Gain (DAC Unit)')
# ax1.set_title(f'2D Map of Surviving |Kerr| Excursion (Exact Mesh)\n(Color Range: {KERR_RANGE_MIN} ~ {KERR_RANGE_MAX})')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 20000)
plt.tight_layout()
plt.show()

# %%
delta_matrix

# %%
from matplotlib.colors import Normalize

KERR_RANGE_MIN = -4
KERR_RANGE_MAX = 4
COLORMAP_NAME = 'viridis' 

print("Generating 2D Kerr Map...")


err_count_fitting = 0
err_count_object_type = 0

unique_freqs = np.unique(extracted_freqs)
unique_gains = np.unique(extracted_gains)

freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
gain_to_idx = {g: i for i, g in enumerate(unique_gains)}

delta_matrix = np.full((len(unique_gains), len(unique_freqs)), np.nan)

for freq, gain, delta in zip(extracted_freqs, extracted_gains, extracted_deltas):
    row = gain_to_idx[gain]
    col = freq_to_idx[freq]
    delta_matrix[row, col] = delta
    
print("\n[Method 1] Plotting Exact Measured Points (Pcolormesh)")

fig1, ax1 = plt.subplots(figsize=(9, 6))

cmap = plt.get_cmap(COLORMAP_NAME).copy()
cmap.set_bad(color='white')
delta_matrix = delta_matrix- delta_matrix[0,1]
pc = ax1.pcolormesh(unique_freqs, unique_gains, delta_matrix , 
                    shading='auto', cmap=cmap,
                    vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)

cbar = fig1.colorbar(pc, ax=ax1)
cbar.set_label('|$\\Delta \\omega _m$| (MHz) ')

ax1.set_xlabel('Excursion Frequency (MHz)')
ax1.set_ylabel('Excursion Gain (DAC Unit)')
# ax1.set_title(f'2D Map of Surviving |Kerr| Excursion (Exact Mesh)\n(Color Range: {KERR_RANGE_MIN} ~ {KERR_RANGE_MAX})')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 20000)
plt.tight_layout()
plt.show()

# %%
unique_freqs

# %%
delta_matrix[0,1]

# %%
print("job_date_list: ", job_date_list)
print("start_num_list: ", start_num_list)
print("finish_num_list: ", finish_num_list)
print("disregarded_indicies: ", disregarded_indicies)

# %% [markdown]
# # Cavity Ramsey Post Processing

# %%
# job_date_list = [20260424]
# start_num_list = [0]
# finish_num_list = [500]
job_date_list = [20260413]
start_num_list = [18]
finish_num_list = [31]


excursion_list = retrieve_from_multiple_jobs("260403_kerr_excusion_redone", 
                                        job_date_list,
                                        start_num_list, 
                                        finish_num_list)

# %%

kerr_list = []
kerr_err_list = []
excursion_freq_list = []
excursion_gain_list = []


for excursion_exp in excursion_list:
    try:
        excursion_exp.analyze_peak(#fit_indices_g=[0, 1, 2, 3, 4, 5],
                                   track_peaks=True,
                                   release_constraint = False,)
        
        alpha2_data = excursion_exp.data['alpha_list'] ** 2
        g_omega = excursion_exp.data['g_omega']

        good_indices_g = get_inlier_indices(alpha2_data, g_omega, threshold=2.0)
        excursion_exp.analyze_peak(
            track_peaks=True, 
            release_constraint=False, 
            fit_indices_g=good_indices_g,
        )
        kerr_list.append(excursion_exp.data['Kerr'])
        kerr_err_list.append(excursion_exp.data['Kerr_err'])
        excursion_freq_list.append(excursion_exp.cfg.expt['kerr_freq'])
        excursion_gain_list.append(excursion_exp.cfg.expt['kerr_gain'])
        if np.isnan(excursion_exp.data['Kerr']):
            excursion_exp.is_bad_data = True
        else:
            excursion_exp.is_bad_data = False  
    except:
        freq = excursion_exp.cfg.expt.get('kerr_freq', 'Unknown')
        gain = excursion_exp.cfg.expt.get('kerr_gain', 'Unknown')
        
        obj_type = type(excursion_exp).__name__
        
        excursion_exp.is_bad_data = True

# %% [markdown]
# ## Manual Postprocessing after Data Fetch

# %%
all_freqs = sorted(list(set(
    e.cfg.expt['kerr_freq'] 
    for e in excursion_list 
    if not getattr(e, 'is_bad_data', False)
)))


freq_to_gains = defaultdict(list)

for e in excursion_list:
    if not getattr(e, 'is_bad_data', False):
        freq = e.cfg.expt['kerr_freq']
        gain = e.cfg.expt['kerr_gain']
        freq_to_gains[freq].append(gain)

for freq in freq_to_gains:
    freq_to_gains[freq] = sorted(freq_to_gains[freq])
freq_to_gains = dict(freq_to_gains)  


print("============================================================================")
print("Excursion Freq Lists:", all_freqs)
print("----------------------------------------------------------------------------")
print("Gain Dict for Each Freq:")
for freq, gains in freq_to_gains.items():
    print(f"  - {freq} MHz (Total number of  {len(gains)}) : {gains}")
print("============================================================================")

# %%
excursion_list[0].cfg.hw.yoko_coupler.current * 1e3

# %%
plot_freq_line_cut(200)

# %%
import numpy as np

target_freq = 200
target_gain = 0
idx_to_plot = 0

discard = False
refit = False

manual_fit_indices = [3, 4, 5]
# =================================================

inspect_list = [
    e for e in excursion_list 
    if np.isclose(e.cfg.expt.get('kerr_gain', np.nan), target_gain, atol=1e-3) 
    and np.isclose(e.cfg.expt.get('kerr_freq', np.nan), target_freq, atol=1e-3) 
    and not getattr(e, 'is_bad_data', False)
]

print(f"Freq: {target_freq}, Gain: {target_gain} datas: {len(inspect_list)}")

if len(inspect_list) > idx_to_plot:
    target_expt = inspect_list[idx_to_plot]
    
    if refit:
        try:
            target_expt.analyze_peak(
                track_peaks=True, 
                release_constraint=False,
                fit_indices_g=manual_fit_indices 
            )
        except Exception as err:
            print(f"Refit Failed: {err}")
            
    target_expt.display_peak()
    
    current_status = getattr(target_expt, 'is_bad_data', False)
    print(f"is_bad_data?: {current_status}")
    
    if discard:
        target_expt.is_bad_data = True
        print(f"Freq {target_freq}, Gain {target_gain} data is discarded by setting is_bad_data = True")
        
else:
    print("There is no valid data for the specified frequency and gain combination.")
    
plot_freq_line_cut(target_freq)

# %%
import numpy as np

target_freq = 200
target_gain = 7000

revived_count = 0

for e in excursion_list:
    freq = e.cfg.expt.get('kerr_freq', np.nan)
    gain = e.cfg.expt.get('kerr_gain', np.nan)
    
    if freq == target_freq and gain == target_gain:
        
        if getattr(e, 'is_bad_data', False):
            
            try:
                kerr_val = e.data['Kerr']
                
                if not np.isnan(kerr_val):
                    e.is_bad_data = False  
                    revived_count += 1
            except (AttributeError, KeyError):
                pass

print(f"{revived_count} of Freq {target_freq}, Gain {target_gain} has resurrected")

plot_freq_line_cut(target_freq)

# %% [markdown]
# ## Plot

# %%
from matplotlib.colors import Normalize

KERR_RANGE_MIN = 0.0
KERR_RANGE_MAX = 0.03
COLORMAP_NAME = 'viridis' 

print("Generating 2D Kerr Map...")

extracted_freqs = []
extracted_gains = []
extracted_kerrs = []

err_count_fitting = 0
err_count_object_type = 0

for expt in excursion_list:
    if not getattr(expt, 'is_bad_data', False):
        try:
            freq = expt.cfg.expt.get('kerr_freq')
            gain = expt.cfg.expt.get('kerr_gain')
            
            kerr_raw = expt.data.get('Kerr', np.nan)
            
            if isinstance(kerr_raw, (list, np.ndarray)):
                kerr_val = np.abs(np.nanmean(kerr_raw)) 
            else:
                kerr_val = np.abs(kerr_raw)

            if freq is None or gain is None or np.isnan(kerr_val):
                continue
                
            extracted_freqs.append(freq)
            extracted_gains.append(gain)
            extracted_kerrs.append(kerr_val)
            
        except (AttributeError, KeyError, TypeError):
            err_count_fitting += 1
            pass
    else:
        pass

print(f"Total survived datapoints used for plotting: {len(extracted_kerrs)}")
if err_count_fitting > 0:
    print(f"(Skipped {err_count_fitting} points due to extraction errors/missing keys)")

if len(extracted_kerrs) == 0:
    print(" Plotting Failed: No valid, non-bad data found in excursion_list.")
else:
    unique_freqs = np.unique(extracted_freqs)
    unique_gains = np.unique(extracted_gains)
    
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    gain_to_idx = {g: i for i, g in enumerate(unique_gains)}
    
    kerr_matrix = np.full((len(unique_gains), len(unique_freqs)), np.nan)
    
    for freq, gain, kerr in zip(extracted_freqs, extracted_gains, extracted_kerrs):
        row = gain_to_idx[gain]
        col = freq_to_idx[freq]
        kerr_matrix[row, col] = kerr
        
    print("\n[Method 1] Plotting Exact Measured Points (Pcolormesh)")
    
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    cmap = plt.get_cmap(COLORMAP_NAME).copy()
    cmap.set_bad(color='white')
    
    pc = ax1.pcolormesh(unique_freqs, unique_gains, kerr_matrix, 
                       shading='auto', cmap=cmap, 
                       vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)
    
    cbar = fig1.colorbar(pc, ax=ax1)
    cbar.set_label('|Kerr| [kHz] (Assumed unit)')
    
    ax1.set_xlabel('Excursion Frequency [kHz]')
    ax1.set_ylabel('Excursion Gain [a.u.]')
    # ax1.set_title(f'2D Map of Surviving |Kerr| Excursion (Exact Mesh)\n(Color Range: {KERR_RANGE_MIN} ~ {KERR_RANGE_MAX})')
    ax1.set_xlim(0, 100)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Save Manually Post Processed Data

# %%
import pickle
import datetime
from pathlib import Path  


cleaned_excursion_list = [
    e for e in excursion_list 
    if not getattr(e, 'is_bad_data', False)
]

now = datetime.datetime.now().strftime("%y%m%d_%H%M")
save_dir = Path("postprocessed_pickles")
save_path = save_dir / f"excursion_list_CLEANED_{now}.pkl"

save_dir.mkdir(parents=True, exist_ok=True)

with open(save_path, 'wb') as f:
    pickle.dump(cleaned_excursion_list, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Pickled: {save_path.absolute()}")

# %% [markdown]
# ## Data Reconstruction From Manually Post Processed Pickles

# %%
pkl_dirs = [
    "postprocessed_pickles\\excursion_list_CLEANED_260424_1030.pkl",
    "postprocessed_pickles\\excursion_list_CLEANED_260423_1610.pkl",
    ]

# %%
pp_data = []
for pkl in pkl_dirs:
    with open(pkl, "rb") as f:
        pp_data.extend(pickle.load(f))

# %%
from matplotlib.colors import Normalize

KERR_RANGE_MIN = 0.0
KERR_RANGE_MAX = 0.03
COLORMAP_NAME = 'viridis' 

print("Generating 2D Kerr Map...")

extracted_freqs = []
extracted_gains = []
extracted_kerrs = []

err_count_fitting = 0
err_count_object_type = 0

for expt in pp_data:
    if not getattr(expt, 'is_bad_data', False):
        try:
            freq = expt.cfg.expt.get('kerr_freq')
            gain = expt.cfg.expt.get('kerr_gain')
            
            kerr_raw = expt.data.get('Kerr', np.nan)
            
            if isinstance(kerr_raw, (list, np.ndarray)):
                kerr_val = np.abs(np.nanmean(kerr_raw)) 
            else:
                kerr_val = np.abs(kerr_raw)

            if freq is None or gain is None or np.isnan(kerr_val):
                continue
                
            extracted_freqs.append(freq)
            extracted_gains.append(gain)
            extracted_kerrs.append(kerr_val)
            
        except (AttributeError, KeyError, TypeError):
            err_count_fitting += 1
            pass
    else:
        pass

print(f"Total survived datapoints used for plotting: {len(extracted_kerrs)}")
if err_count_fitting > 0:
    print(f"(Skipped {err_count_fitting} points due to extraction errors/missing keys)")

if len(extracted_kerrs) == 0:
    print(" Plotting Failed: No valid, non-bad data found in excursion_list.")
else:
    unique_freqs = np.unique(extracted_freqs)
    unique_gains = np.unique(extracted_gains)
    
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    gain_to_idx = {g: i for i, g in enumerate(unique_gains)}
    
    kerr_matrix = np.full((len(unique_gains), len(unique_freqs)), np.nan)
    
    for freq, gain, kerr in zip(extracted_freqs, extracted_gains, extracted_kerrs):
        row = gain_to_idx[gain]
        col = freq_to_idx[freq]
        kerr_matrix[row, col] = kerr
        
    print("\n[Method 1] Plotting Exact Measured Points (Pcolormesh)")
    
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    
    cmap = plt.get_cmap(COLORMAP_NAME).copy()
    cmap.set_bad(color='white')
    
    pc = ax1.pcolormesh(unique_freqs, unique_gains, kerr_matrix, 
                       shading='auto', cmap=cmap, 
                       vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)
    
    cbar = fig1.colorbar(pc, ax=ax1)
    cbar.set_label('|$\\Delta$Kerr| (MHz) ')
    
    ax1.set_xlabel('Excursion Frequency (MHz)')
    ax1.set_ylabel('Excursion Gain (DAC Unit)')
    # ax1.set_title(f'2D Map of Surviving |Kerr| Excursion (Exact Mesh)\n(Color Range: {KERR_RANGE_MIN} ~ {KERR_RANGE_MAX})')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 20000)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # Cavity Ramsey Manual Post Fitting

# %%
from experiments.qsim.cavity_ramsey_flux_excursion import CavityFluxExcursionRamseyExperiment


job_date_list = [20260430]
start_num_list = [13]
finish_num_list = [15]
disregarded_indicies = []
path = hdf5_path_generator("260403_kerr_excusion_redone",
                           job_date_list, 
                           start_num_list,
                           finish_num_list, "KerrCavityRamseyExcursionExperiment")
exp_list = path_to_experiment(path, CavityFluxExcursionRamseyExperiment)

# %%
idx_to_analyze = 0

otmf = exp_list[idx_to_analyze]


# @staticmethod
# def fit_model(x, alpha2, f, scale, offset):
#     """Fitting model: exp(2*alpha2*(-cos(2*pi*f*x)-1))"""
#     return scale * np.exp(2 * alpha2 * (-np.cos(2 * np.pi   * f * x) - 1)) + offset
    # return scale * np.exp(-2.0*alpha2*(1.0 + np.exp(-x/tphi)*np.cos(2.0*np.pi*f*x))) + offset




x, y, z = otmf.data['xpts'], otmf.data['ypts'], normalize(otmf.data['avgi'], otmf)

# Lists to collect fit results
alpha2_fits = []
f_fits = []
fit_results = []
z_fits = []
z_smooths = []

period_estimate = estimate_periodicity(z[0], sampling_rate=1/(x[1] - x[0]))
f_initial = 1.0 / period_estimate if period_estimate != 0 and np.isfinite(period_estimate) else 0.1

# %%
plt.pcolor(*np.meshgrid(x, y), z)

# %%

signal_smooth = gaussian_filter1d(line, sigma=1.5)
z_smooths.append(signal_smooth)

alpha_guess = y[lid]*self.cfg.device.manipulate.gain_to_alpha[0]
phase_guess = self.estimate_phase(line)
scale_guess = np.max(line) - np.min(line)

# Create lmfit Model
model = Model(self.fit_model)

# Set initial parameters
# params = model.make_params(
#     # alpha2 = dict(value=alpha_guess**2, min=alpha_guess**2/2, max=alpha_guess**2*2),
#     alpha2 = dict(value=alpha_guess**2, vary=False),
#     f=f_initial,
#     scale=dict(value=1, min=0.1, max=1.2),
#     offset=dict(value=0, min=-0.1, max=0.5))
params = model.make_params(
    alpha2=dict(value=alpha_guess**2, vary=False),
    f=f_initial,
    # tphi=dict(value=100*(x[-1]-x[0]), min=10*(x[-1]-x[0]), max=1e6),  # units are the same as x
    scale =dict(value=scale_guess, min=0.75*scale_guess, max=1.25 * scale_guess),
    offset =dict(value=0, min=-0.1, max=0.5),
    phase = dict(value=phase_guess, min=0, max=np.pi)
    )
try:
    # Perform fit
    result = model.fit(line, params, x=x)

    # Collect best-fit parameters
    alpha2_fits.append(result.params['alpha2'].value)
    f_fits.append(result.params['f'].value)
    fit_results.append(result)
    z_fits.append(result.best_fit)
except:
    if debug == True:
        print(f"fit_failed for index: {lid}")
    None
# print(f_fits)
f_fits = np.array(f_fits)
alpha2_fits = np.array(alpha2_fits)
fit_rsq_threshold = kwargs.get('fit_rsq_threshold', kwargs.get("rsq_threshold", 0.2))
fit_good = [res.rsquared > fit_rsq_threshold for res in fit_results]
# TODO(stage2): source cell P57 lost the indentation of everything from
# `f_fits = np.array(f_fits)` onward. It was originally a function
# body. Left byte-faithful, so this cell does not parse.
if debug == True:
print(fit_good)
filtered_alpha2 = alpha2_fits[fit_good]
filtered_f = f_fits[fit_good]

# here we deduct the virtual ramsey from fitted f
# self.cfg.expt got erased during initialization... so extracting it another way
virtual_freq = self.cfg.expt.ramsey_freq
kerr_gain = self.cfg.expt.kerr_gain

alpha2_array = np.array(filtered_alpha2)
f_array = np.array(filtered_f) - virtual_freq
z_smooths = np.array(z_smooths)
try:
# Create linear model: w = kc * alpha2 + delta
linear_model = Model(self.fit_func, independent_vars=['x'])
linear_params = linear_model.make_params(kc=1.0, delta=0.0)
linear_result = linear_model.fit(f_array, linear_params, x=alpha2_array)

# Store results
self.fit_results = {
    'alpha2': alpha2_array,
    'f': f_array,
    'results': fit_results,
    'z_fits': np.array(z_fits),
    'kc': linear_result.params['kc'].value,
    'delta': linear_result.params['delta'].value,
    'linear_fit_result': linear_result,
    'z_smooths': z_smooths,
    'fit_good': fit_good,
    'kerr_gain': kerr_gain,
}
except:
self.fit_results = {
    'alpha2': alpha2_array,
    'f': f_array,
    'kc': np.nan,
    'delta': np.nan,
    'linear_fit_result': None,
}

# %% [markdown]
# # Excursion Transition Post Processing

# %% [markdown]
# ## gain vs current using tansition

# %%
job_date_list = [20260430, 20260501]
start_num_list = [218, 41]
finish_num_list = [238,149]



transition_list = retrieve_from_multiple_jobs("260403_kerr_excusion_redone", 
                                              job_date_list,
                                              start_num_list, 
                                              finish_num_list)

# %%
transition_prog_list = [exp for exp in transition_list if check_program_class(exp, "ExcursionTransitionDebuggingProgram")]
total_result_list = [transition_prog_list[i*11:i*11+10] for i in range(5)]
dc_set_points = np.array([res.cfg.expt.coupler_current for res in total_result_list[0]])

# %%
len(transition_prog_list)

# %%
transition_prog_list[0].cfg.expt.coupler_current

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

trans_result_list = total_result_list[0]

y_data = dc_set_points * 1e3
x_data = trans_result_list[0].cfg.expt.kerr_gains
z_data_i = np.transpose([result.data['avgi'] for result in trans_result_list])
z_data_q = np.transpose([result.data['avgq'] for result in trans_result_list])

fig, ax = plt.subplots(1, 2, figsize = (14, 6)) 
c = ax[0].pcolor(*np.meshgrid(x_data, y_data), np.transpose(z_data_i))
ax[0].set_xlabel("kerr excursion gain (DAC unit)")
ax[0].set_ylabel("DC set point (mA)")
# ax[0].set_ylim(0, 20000)
fig.colorbar(c, ax = ax[0], label = "Avgi")
c = ax[1].pcolor(*np.meshgrid(x_data, y_data), np.transpose(z_data_q))
ax[1].set_xlabel("Kerr excursion freq (MHz)")
ax[1].set_ylabel("kerr excursion gain (DAC unit)")
# ax[1].set_ylim(0, 20000)
fig.colorbar(c, ax = ax[1], label = "Avgq")
fig.suptitle("$\\langle 1 |U(t, 0)| 1 \\rangle$, t = 10 $\\mu s$")
fig.tight_layout()
Z_grid = np.transpose(z_data_i) 

boundary_x = []
boundary_y = []

for i, y_val in enumerate(y_data):
    row_data = Z_grid[i, :]
    
    smoothed_data = gaussian_filter1d(row_data, sigma = 20) 
    
    diff_data = np.diff(smoothed_data)
    
    boundary_idx = np.argmax(diff_data) 
    
    boundary_x.append(x_data[boundary_idx])
    boundary_y.append(y_val)

boundary_x = np.array(boundary_x)
boundary_y = np.array(boundary_y)

valid_idx = (boundary_y > -0.1) & (boundary_x > 10000)

fit_x = boundary_x[valid_idx]
fit_y = boundary_y[valid_idx]

slope, intercept = np.polyfit(fit_x, fit_y, 1)

print(f"Estimated Slope (dCurrent/dGain): {slope:.4e}")

plt.figure(figsize=(8, 6))

c = plt.pcolor(*np.meshgrid(x_data, y_data), Z_grid)
plt.colorbar(c, label="Avgi")

plt.plot(boundary_x, boundary_y, 'r.', label='Extracted Boundary')

plt.plot(fit_x, slope * fit_x + intercept, 'w--', linewidth=2, label=f'Fit Slope: {slope:.2e}')

plt.xlabel("kerr excursion gain (DAC unit)")
plt.ylabel("DC set point (mA)")
plt.title(f"Boundary Extraction & Linear Fit (Smoothed) at kerr freq {trans_result_list[0].cfg.expt.kerr_freq} MHz")
plt.legend(loc='lower left') 
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Freq fine sweep

# %%
len(transition_prog_list)

# %%
x = transition_prog_list[0].data['ypts']
y = []
z = []
for i in range(6):
    idx_to_plot = 10*i-1
    y.append(transition_prog_list[idx_to_plot].cfg.expt.kerr_freq)
    z.append(np.array(transition_prog_list[idx_to_plot].data['avgi'])[:, 0])
    # plt.plot(transition_prog_list[idx_to_plot].data['ypts'], transition_prog_list[idx_to_plot].data['avgi'])
    # plt.title(f"freq = {transition_prog_list[idx_to_plot].cfg.expt.kerr_freq}, time = {transition_prog_list[idx_to_plot].data['xpts'][0]}")
fig, ax = plt.subplots(figsize = (6, 6))
ax.pcolor(*np.meshgrid(y, x), np.transpose(z))
ax.set_xlabel("Excursion Freq (MHz)")
ax.set_ylabel("Excursion gain (DAC unit)")

# %% [markdown]
# ## Gain fine sweep

# %%
# job_date_list = [20260424]
# start_num_list = [338]
# finish_num_list = [358]
job_date_list = [20260427]
start_num_list = [390]
finish_num_list = [408]


transition_list = retrieve_from_multiple_jobs("260403_kerr_excusion_redone", 
                                              job_date_list,
                                              start_num_list, 
                                              finish_num_list)

# %%
len(transition_list)

# %%
fig, ax = plt.subplots()
y = transition_list[0].cfg.expt.kerr_gains
x = [trans.cfg.expt.kerr_freq for trans in transition_list]
z = [trans.data['avgi'][:, 0] for trans in transition_list]
ax.pcolor(*(x, y), np.transpose(z))

# %% [markdown]
# ## Self Kerr vs avg I and Q value

# %%
unique_freqs = np.unique(extracted_freqs)
unique_gains = np.unique(extracted_gains)

freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
gain_to_idx = {g: i for i, g in enumerate(unique_gains)}

kerr_matrix = np.full((len(unique_gains), len(unique_freqs)), np.nan)

for freq, gain, kerr in zip(extracted_freqs, extracted_gains, extracted_kerrs):
    row = gain_to_idx[gain]
    col = freq_to_idx[freq]
    kerr_matrix[row, col] = kerr
    
print("\n[Method 1] Plotting Exact Measured Points (Pcolormesh)")

fig1, ax1 = plt.subplots(figsize=(9, 6))

cmap = plt.get_cmap(COLORMAP_NAME).copy()
cmap.set_bad(color='white')

pc = ax1.pcolormesh(unique_freqs, unique_gains, kerr_matrix, 
                    shading='auto', cmap=cmap, 
                    vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)

cbar = fig1.colorbar(pc, ax=ax1)
cbar.set_label('|$\\Delta$Kerr| (MHz) ')

ax1.set_xlabel('Excursion Frequency (MHz)')
ax1.set_ylabel('Excursion Gain (DAC Unit)')
# ax1.set_title(f'2D Map of Surviving |Kerr| Excursion (Exact Mesh)\n(Color Range: {KERR_RANGE_MIN} ~ {KERR_RANGE_MAX})')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 20000)
plt.tight_layout()
plt.show()

# %%
unique_freqs

# %%
y

# %%
extracted_freqs

# %%

kerr_dict = {}
for f, g, k in zip(extracted_freqs, extracted_gains, extracted_kerrs):
    key = (np.round(f, 6), np.round(g, 6))
    kerr_dict[key] = k

avgi_dict = {}
avgi_gains = transition_list[0].cfg.expt.kerr_gains 

for trans in transition_list:
    f = trans.cfg.expt.kerr_freq
    avgi_array = trans.data['avgi'][:, 0]
    
    for g, avgi_val in zip(avgi_gains, avgi_array):
        key = (np.round(f, 6), np.round(g, 6))
        avgi_dict[key] = avgi_val

common_keys = set(kerr_dict.keys()).intersection(set(avgi_dict.keys()))


matched_kerrs = np.array([kerr_dict[k] for k in common_keys])
matched_avgis = np.array([avgi_dict[k] for k in common_keys])

common_freqs = np.unique([k[0] for k in common_keys])
common_gains = np.unique([k[1] for k in common_keys])

freq_to_idx = {f: i for i, f in enumerate(common_freqs)}
gain_to_idx = {g: i for i, g in enumerate(common_gains)}

kerr_matrix_shared = np.full((len(common_gains), len(common_freqs)), np.nan)
avgi_matrix_shared = np.full((len(common_gains), len(common_freqs)), np.nan)

for (f, g) in common_keys:
    row = gain_to_idx[g]
    col = freq_to_idx[f]
    kerr_matrix_shared[row, col] = kerr_dict[(f, g)]
    avgi_matrix_shared[row, col] = avgi_dict[(f, g)]


fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131)
ax1.scatter(matched_avgis, matched_kerrs, alpha=0.6, edgecolors='none')
ax1.set_xlabel('AvgI')
ax1.set_ylabel('|$\\Delta$Kerr| (MHz)')
ax1.set_title('Correlation: AvgI vs |Kerr|')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = fig.add_subplot(132)
cmap_kerr = plt.get_cmap(COLORMAP_NAME).copy() 
cmap_kerr.set_bad(color='white')
pc1 = ax2.pcolormesh(common_freqs, common_gains, kerr_matrix_shared, 
                     shading='auto', cmap=cmap_kerr, 
                     vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)
fig.colorbar(pc1, ax=ax2, label='|Kerr| (MHz)')
ax2.set_xlabel('Excursion Frequency (MHz)')
ax2.set_ylabel('Excursion Gain (DAC Unit)')
ax2.set_title('Filtered |Kerr| Map (Intersection)')

ax3 = fig.add_subplot(133)
cmap_avgi = plt.get_cmap('viridis').copy() 
cmap_avgi.set_bad(color='white')
pc2 = ax3.pcolormesh(common_freqs, common_gains, avgi_matrix_shared, 
                     shading='auto', cmap=cmap_avgi)
fig.colorbar(pc2, ax=ax3, label='AvgI')
ax3.set_xlabel('Excursion Frequency (MHz)')
ax3.set_ylabel('Excursion Gain (DAC Unit)')
ax3.set_title('Filtered AvgI Map (Intersection)')

plt.tight_layout()
plt.show()

# %%

kerr_dict = {}
for f, g, k in zip(extracted_freqs, extracted_gains, extracted_kerrs):
    key = (np.round(f, 6), np.round(g, 6))
    kerr_dict[key] = k

avgi_dict = {}
avgi_gains = transition_list[0].cfg.expt.kerr_gains 

for trans in transition_list:
    f = trans.cfg.expt.kerr_freq
    avgi_array = trans.data['avgq'][:, 0]
    
    for g, avgi_val in zip(avgi_gains, avgi_array):
        key = (np.round(f, 6), np.round(g, 6))
        avgi_dict[key] = avgi_val

common_keys = set(kerr_dict.keys()).intersection(set(avgi_dict.keys()))


matched_kerrs = np.array([kerr_dict[k] for k in common_keys])
matched_avgis = np.array([avgi_dict[k] for k in common_keys])

common_freqs = np.unique([k[0] for k in common_keys])
common_gains = np.unique([k[1] for k in common_keys])

freq_to_idx = {f: i for i, f in enumerate(common_freqs)}
gain_to_idx = {g: i for i, g in enumerate(common_gains)}

kerr_matrix_shared = np.full((len(common_gains), len(common_freqs)), np.nan)
avgi_matrix_shared = np.full((len(common_gains), len(common_freqs)), np.nan)

for (f, g) in common_keys:
    row = gain_to_idx[g]
    col = freq_to_idx[f]
    kerr_matrix_shared[row, col] = kerr_dict[(f, g)]
    avgi_matrix_shared[row, col] = avgi_dict[(f, g)]


fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131)
ax1.scatter(matched_avgis, matched_kerrs, alpha=0.6, edgecolors='none')
ax1.set_xlabel('AvgI')
ax1.set_ylabel('|$\\Delta$Kerr| (MHz)')
ax1.set_title('Correlation: AvgI vs |Kerr|')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = fig.add_subplot(132)
cmap_kerr = plt.get_cmap(COLORMAP_NAME).copy() 
cmap_kerr.set_bad(color='white')
pc1 = ax2.pcolormesh(common_freqs, common_gains, kerr_matrix_shared, 
                     shading='auto', cmap=cmap_kerr, 
                     vmin=KERR_RANGE_MIN, vmax=KERR_RANGE_MAX)
fig.colorbar(pc1, ax=ax2, label='|$\\Delta$Kerr| (MHz)')
ax2.set_xlabel('Excursion Frequency (MHz)')
ax2.set_ylabel('Excursion Gain (DAC Unit)')
ax2.set_title('Filtered |Kerr| Map (Intersection)')

ax3 = fig.add_subplot(133)
cmap_avgi = plt.get_cmap('viridis').copy() 
cmap_avgi.set_bad(color='white')
pc2 = ax3.pcolormesh(common_freqs, common_gains, avgi_matrix_shared, 
                     shading='auto', cmap=cmap_avgi)
fig.colorbar(pc2, ax=ax3, label='AvgQ')
ax3.set_xlabel('Excursion Frequency (MHz)')
ax3.set_ylabel('Excursion Gain (DAC Unit)')
ax3.set_title('Filtered AvgQ Map (Intersection)')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Spectroscopy

# %%
job_date_list = [20260428]
start_num_list = [218]
finish_num_list = [253]



spect_list = retrieve_from_multiple_jobs("260403_kerr_excusion_redone", 
                                        job_date_list,
                                        start_num_list, 
                                        finish_num_list)

# %%
fig, ax = plt.subplots()
x = spect_list[0].data['xpts']
y = [res.cfg.expt['flux_drive_gain'] for res in spect_list]
z = [res.data['avgi'] for res in spect_list]
ax.pcolor(*np.meshgrid(x, y), z)

# %% [markdown]
# # Cavity Ramsey Refitting

# %%
import numpy as np
from scipy.ndimage import gaussian_filter1d
from lmfit import Model

try:
    from slab import AttrDict
except Exception:
    class AttrDict(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError as e: raise AttributeError(k) from e
        def __setattr__(self, k, v): self[k] = v

# %%
import pickle

base_dir = "D:\\experiments\\250223_storage_swap_adding\\expt_objs\\"
pkl_files = [
    # "JOB-20260224-00259_expt.pkl", #S1
    # "JOB-20260224-00261_expt.pkl"  #S7
    # "JOB-20260224-00247_expt.pkl", #S1
    # "JOB-20260224-00251_expt.pkl",  #S7
    # "JOB-20260225-00049_expt.pkl", #S7
    "JOB-20260225-00054_expt.pkl", #S7
    "JOB-20260225-00058_expt.pkl", #S1
    "JOB-20260225-00061_expt.pkl", #S7
    "JOB-20260226-00133_expt.pkl", #S7 0.5mA
    "JOB-20260226-00134_expt.pkl", #S1 0.5mA
    "JOB-20260226-00135_expt.pkl", #S1 0.5mA
    "JOB-20260226-00136_expt.pkl", #S1 0.5mA
    "JOB-20260226-00137_expt.pkl", #S1 0.5mA
    
]
cavity_kerr_list = []
for i, pkl_file in enumerate(pkl_files):
    with open(base_dir+pkl_file, "rb") as f:
        cavity_kerr_list.append( pickle.load(f))

# %%
test = cavity_kerr_list[7]
tests=post_refit_one_cavity_ramsey_v2(test,
                                      f_window_mhz = 0.5,
                                      method_setting="basinhopping")
# test.data.keys()
data_attr = AttrDict(test.data)
fig, ax = plt.subplots(1, 3, figsize = (21,7))
X, y = data_attr.xpts, data_attr.gain_list
y_fine = np.linspace(y[0], y[-1], 1001)
Y = np.ones(np.shape(X))
# freq = np.array([tests.results[i].params["f"].value for i in range(len(y))])
freq = tests.f
freq_err = [tests.results[i].params["f"].stderr for i in range(len(y))]
kc, delta = tests.linear_fit_result.params["kc"].value, tests.linear_fit_result.params["delta"].value
alpha_2 = tests.alpha2
alpha2_fine = np.linspace(alpha_2[0], alpha_2[-1], 1001)
for i in range(len(y)):
    Y[i, :] *= y[i]
ax[0].pcolor(X, Y,
             data_attr.g_avgi)
ax[0].set_xlabel("Wait time (us)")
ax[0].set_ylabel("DAC gain")
ax[0].set_title("measured data")
ax[1].pcolor(X, Y,
             tests.z_fits)
ax[1].set_xlabel("Wait time (us)")
ax[1].set_ylabel("DAC gain")
ax[1].set_title("fitted data")
ax[2].errorbar(y[tests.fit_good], freq, 
               yerr = freq_err,
               fmt='o')
ax[2].set_xlabel("DAC gain")
ax[2].set_ylabel("Frequency w.r.f. $f_{\\mathrm{ramsey}}$")
# ax[2].errorbar(y[tests.fit_good], freq_err)
ax[2].plot(y_fine, kc * (y_fine *test.cfg.device.manipulate.gain_to_alpha[0])**2+delta)
fig.suptitle(f"kc = {kc} MHz for S_{test.cfg.expt.storage_ramsey[1]}")

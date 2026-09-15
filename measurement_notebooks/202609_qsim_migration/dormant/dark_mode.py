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
# # Dark mode (dormant)
#
# Relocated verbatim from `measurement_notebooks/jonginn/qsim_experiments.ipynb`
# cells 159-262 by the stage-2 notebook decomposition. This is **dormant**: it is
# here to stay findable and diffable, not to be maintained. Nothing in the active
# migration path imports it.
#
# Relocation only -- definitions were not hoisted and procedural blocks were not
# wrapped, per the stage-2 instructions. That matters here more than elsewhere:
# this range defines `stor_label` six times in three mutually incompatible
# versions (cells 171 / 178,192,197,202 / 206), and also redefines
# `stor_label_for_expt`, `_flat_unique_expts`, `flatten_exp_lists` and
# `unique_expts` with bodies that differ from the active floquet-calibration
# copies. Source cell order is preserved, so each section still sees the same
# definition it saw in the original notebook. Do not deduplicate these names
# without first deciding, per section, which version that section needs.
#
# **This notebook does not stand alone.** It read these names from the live
# kernel, having had the calibration sections run first in the same notebook;
# relocation does not fix that, and the stage-2 instructions do not ask it to:
#
#   from what is now floquet_calibration.py
#     `dm_sideband_scramble_defaults`, `sideband_scramble_preproc`,
#     `floquet_cycle_list_gen`, `_flat_unique_expts`
#   from what is now multiphoton_calibration.py
#     `singleshot_defaults`, `singleshot_postproc`
#
# The first three are importable from
# `experiments/qsim/notebook_helpers/floquet_bare_readout.py` and
# `floquet_calibration.py` if this section is ever revived; the two single-shot
# names are in `notebook_helpers/multiphoton_calibration.py`. They are
# deliberately not imported here, so that reviving this notebook is a decision
# someone makes rather than something that silently half-works.
#
# Sibling dormant notebooks: `debug.py`; on the analysis side,
# `analysis_notebooks/202609_qsim_migration/dormant/`.

# %% [markdown]
# # Prepare

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner

from job_server import JobClient
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager

# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
print(f"Server status: {health['status']}")
print(f"Pending jobs: {health['pending_jobs']}")

# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
client.print_queue()

# Who is running these experiments??
user = 'jonginn'

print(f"Welcome {user}!")

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

# %%
# Initialize station to retrieve soc and configs
config_dict = {'hardware_config': 'CFG-HW-20260904-00019',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20260904-00014',
 'floquet_storage_swap': 'CFG-FL-20260904-00042'}


station = MultimodeStation(
    user = user,
    experiment_name = "260818_qsim_spectroscopy",
    project = "EncSpec",
    log_measurements=True,
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    floquet_file=config_dict['floquet_storage_swap'],
)

# %% [markdown]
# # Global Exp Config Setting

# %%
station.ds_storage.df

# %%
measurement_config_default_dict = {
    'avoid_yoko': False,
    'use_multiphoton_swap': False,
    
    
}

active_reset_default_dict = {
    "reset_dump_mode": 2,
    "dump_reset_iter_num": 1,
}

floquet_default_dict = {
    #For phase accumulation:
    "include_10cycles_buffer": True,
    "include_10cycles_buffer_in_pi_half" : True,
    
    # flat_top: legacy 3-segment pulse; preload_flattop: one preloaded arb envelope
    "floquet_waveform": "preload_flattop",
    "floquet_hardware_loop" : True,
    "scramble_sync_cycles" : 1,
    "palindrome_scramble": False,    
}

# %% [markdown]
# ## Displace & Multiparity

# %% [markdown]
# ### Under various displacement, & single shot comparison

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)


meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [2, 3]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:
displacement_list = np.linspace(0.1, np.sqrt(20), 101)

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for displacement in tqdm(displacement_list):
        scramble = dmscramble_runner.execute(
            reps=1000,
            init_fock=False,
            init_alpha = displacement,
            # init_man_fock_state = '0',
            init_stor=0,
            ro_stor = meas_stor,
            relax_delay=8000,
            active_reset=False,
            pre_relax_delay = 2500, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=[0],
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = False,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,
            
            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
for i, disp in enumerate(displacement_list):
    expt_to_ana = scramble_expts[0][i]

    i_first, q_first    = expt_to_ana.data['idata'][0][0::2], expt_to_ana.data['qdata'][0][0::2]
    i_second, q_second  = expt_to_ana.data['idata'][0][1::2], expt_to_ana.data['qdata'][0][1::2]
    plt.scatter(i_first, q_first, label = f"First blob of $\\alpha$ = {disp}")
    plt.scatter(i_second, q_second, label = f"Second blob of $\\alpha$ = {disp}")
    
plt.xlabel("I (adc unit)")
plt.ylabel("Q (adc unit)")
# plt.legend()

# %%
avg_i_first_list = []
avg_i_second_list = []

for i, disp in enumerate(displacement_list):
    expt_to_ana = scramble_expts[0][i]

    i_first, q_first    = expt_to_ana.data['idata'][0][0::2], expt_to_ana.data['qdata'][0][0::2]
    i_second, q_second  = expt_to_ana.data['idata'][0][1::2], expt_to_ana.data['qdata'][0][1::2]
    avg_i_first_list.append(np.average(i_first))
    avg_i_second_list.append(np.average(i_second))
    
    
    
plt.scatter(displacement_list**2, avg_i_first_list, label = f"First average I")
plt.scatter(displacement_list**2, avg_i_second_list, label = f"Second average I")
    
plt.xlabel("average_photon_number")
plt.ylabel("average I")
# plt.legend()

# %% [markdown]
# ## Multi photon fock

# %%
# ---- Confusion-matrix correction for multiparity (shared by this section) ----
# Applies a readout confusion-matrix inverse to the modulo-4 distributions from
# collect_multiparity()/analyze_multiparity() (which already return p_mod0..p_mod3
# per point). Same machinery as data_postprocess.ipynb, but operating on the
# already-computed `multiparity` dict instead of re-reading single shots.
#
# The active matrix is set by the calibration cell below (set_confusion_matrix(counts));
# `confusion_matrix_manual` is only a fallback used until that cell has been run.
import numpy as np

# Fallback confusion matrix (used only if the calibration cell has not been run).
# C[i, j] = P(measured n = j | prepared Fock |i>); rows = prepared, cols = measured.
confusion_matrix_manual = np.array([
    [0.8065,  0.07175, 0.06725, 0.0545 ],
    [0.07225, 0.7735,  0.04925, 0.105  ],
    [0.10225, 0.07575, 0.74325, 0.07875],
    [0.05075, 0.13825, 0.07075, 0.74025],
])

_mod_vals = np.array([0, 1, 2, 3])
_parity_first_signs  = 1 - 2 * (_mod_vals % 2)    # [+1, -1, +1, -1]
_parity_second_signs = 1 - 2 * (_mod_vals // 2)   # [+1, +1, -1, -1]


def set_confusion_matrix(C):
    """Store row-normalized confusion matrix and its inverse-transpose globally.
    p_meas = C.T @ p_true  ->  p_true = inv(C.T) @ p_meas."""
    global confusion_matrix, confusion_inv
    C = np.asarray(C, dtype=float)
    confusion_matrix = C / C.sum(axis=1, keepdims=True)
    confusion_inv = np.linalg.inv(confusion_matrix.T)
    return confusion_matrix


def build_confusion_matrix_from_calibration(cal_expts, e_is_high_I=True, point_idx=0):
    """counts[i, j] = # single shots measured at n=j when Fock |i> was prepared.

    `cal_expts` is the Fock |0>..|3> calibration set (single floquet point). Accepts
    either a nested [[e], [e], ...] list (as produced by the runner) or a flat list.
    The prepared state is read from each experiment's cfg.expt.init_man_fock_state,
    so the result does not depend on calibration ordering. Returns (counts, fnames).
    """
    counts = np.zeros((4, 4), dtype=float)
    fnames = []
    for item in cal_expts:
        expts = item if isinstance(item, (list, tuple, set)) else [item]
        for expt in expts:
            true_n = int(expt.cfg.expt.init_man_fock_state)
            if true_n not in (0, 1, 2, 3):
                continue
            fnames.append(expt.fname)
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
            n_mod4 = b0 + 2 * b1          # cond_sec_phase=-90: (g,g)->0,(e,g)->1,(g,e)->2,(e,e)->3
            for pred_n in range(4):
                counts[true_n, pred_n] += np.sum(n_mod4 == pred_n)
    return counts, fnames


def correct_distribution(p_meas, clip_renormalize=True):
    """Apply the confusion-matrix inverse to each row (a mod-4 distribution)."""
    p_corr = np.asarray(p_meas) @ confusion_inv.T
    if clip_renormalize:
        p_corr = np.clip(p_corr, 0.0, None)
        p_corr = p_corr / p_corr.sum(axis=1, keepdims=True)
    return p_corr


def correct_multiparity(multiparity, clip_renormalize=True):
    """Confusion-correct a collect_multiparity() dict.

    Rebuilds nmod4_mean / mean_parity_first / mean_parity_second from the
    corrected per-point p_mod0..3 distribution. Returns a dict with the SAME
    schema as `multiparity`, so it drops straight into the existing plot cells.
    """
    corrected = {}
    for ro_stor, data in multiparity.items():
        p_meas = np.stack([data['p_mod0'], data['p_mod1'],
                           data['p_mod2'], data['p_mod3']], axis=1)
        p_corr = correct_distribution(p_meas, clip_renormalize=clip_renormalize)
        corrected[ro_stor] = {
            'xpts': data['xpts'],
            'p_mod0': p_corr[:, 0], 'p_mod1': p_corr[:, 1],
            'p_mod2': p_corr[:, 2], 'p_mod3': p_corr[:, 3],
            'nmod4_mean':         p_corr @ _mod_vals,
            'mean_parity_first':  p_corr @ _parity_first_signs,
            'mean_parity_second': p_corr @ _parity_second_signs,
        }
    return corrected


def _corr_stor_label(ro_stor):
    return "M1" if ro_stor == 0 else f"S{ro_stor}"


def plot_parity_raw_vs_corrected(multiparity, corrected,
                                 floquet_cycle2us=1.0, title=""):
    """2-panel first/second parity expectation; dashed=raw, solid=corrected."""
    import matplotlib.pyplot as plt
    import textwrap
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for ro_stor in corrected:
        c = corrected[ro_stor]
        r = multiparity[ro_stor]
        si = np.argsort(c['xpts'])
        x = c['xpts'][si] * floquet_cycle2us
        lbl = _corr_stor_label(ro_stor)
        line, = ax[0].plot(x, c['mean_parity_first'][si], "-", label=lbl)
        col = line.get_color()
        ax[0].plot(x, r['mean_parity_first'][si], "--", alpha=0.4, color=col)
        ax[1].plot(x, c['mean_parity_second'][si], "-", color=col, label=lbl)
        ax[1].plot(x, r['mean_parity_second'][si], "--", alpha=0.4, color=col)
    ax[0].set_title("First parity expectation (dashed=raw, solid=corrected)")
    ax[1].set_title("Second parity-bit expectation (dashed=raw, solid=corrected)")
    for a in ax:
        a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
        a.set_xlabel("time (us)")
        a.set_ylabel("parity value")
        a.set_ylim(-1.05, 1.05)
        a.legend()
    if title:
        fig.suptitle(textwrap.fill(str(title), 80), fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_nmod4_raw_vs_corrected(multiparity, corrected,
                                floquet_cycle2us=1.0, title=""):
    """Single-panel <n mod 4>; dashed=raw, solid=corrected."""
    import matplotlib.pyplot as plt
    import textwrap
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    for ro_stor in corrected:
        c = corrected[ro_stor]
        r = multiparity[ro_stor]
        si = np.argsort(c['xpts'])
        x = c['xpts'][si] * floquet_cycle2us
        line, = ax.plot(x, c['nmod4_mean'][si], "-", label=_corr_stor_label(ro_stor))
        ax.plot(x, r['nmod4_mean'][si], "--", alpha=0.4, color=line.get_color())
    ax.set_xlabel("time (us)")
    ax.set_ylabel(r"$\langle n\ mod\ 4\rangle$")
    ax.legend()
    if title:
        ax.set_title(textwrap.fill(str(title), 60), fontsize=8)
    plt.tight_layout()
    plt.show()


# Default to the fallback matrix so the helpers are usable before calibration;
# the calibration cell below overrides it via set_confusion_matrix(counts).
set_confusion_matrix(confusion_matrix_manual)

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)


meas_stors = [0]
# meas_stors = [0]
swap_stors = [6, 7]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [6, 7]
scramble_expts = []
init_states = ['0', '1', '2', '3']

rel_phase_list = [0]
# for phase in rel_phase_list:

for init_state in tqdm(init_states):
    scramble_sub_expts = []
    for _ in [0]:
        scramble = dmscramble_runner.execute(
            reps=6000,
            init_fock=False,
            # init_alpha = displacement,
            init_man_fock_state = init_state,
            init_stor=0,
            ro_stor = 0,
            relax_delay=8000,
            active_reset=True,
            pre_relax_delay = 2500, 
            man_reset=False, 
            storage_reset = False, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=[0],
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = True,
            swap_man_dark = False,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            fg_area_comp = "length",
            
            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()


# Keep a dedicated handle to the Fock |0>..|3> calibration so the confusion-
# matrix estimate below survives even after `scramble_expts` is overwritten by
# the later measurement subsections.
fock_cal_expts = scramble_expts

# %%

expt_to_ana = scramble_expts[0][0]
rn = expt_to_ana.cfg.read_num
fig, ax = plt.subplots(1, 2, figsize = (12, 5))
i_first, q_first    = expt_to_ana.data['idata'][0][rn-2::rn], expt_to_ana.data['qdata'][0][rn-2::rn]
i_second, q_second  = expt_to_ana.data['idata'][0][rn-1::rn], expt_to_ana.data['qdata'][0][rn-1::rn]
ax[0].scatter(i_first, q_first, label = f"First blob", alpha = 0.1)
ax[1].scatter(i_second, q_second, label = f"Second blob", alpha = 0.1)
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("I (adc unit)")
ax[0].set_ylabel("Q (adc unit)")
ax[1].set_xlabel("I (adc unit)")
ax[1].set_ylabel("Q (adc unit)")
fig.tight_layout()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Estimate the modulo-4 readout confusion matrix from the Fock |0>..|3>
# calibration (`fock_cal_expts`, captured by the calibration runner above) and
# make it the matrix used by every correction cell below.
counts, fname_list = build_confusion_matrix_from_calibration(fock_cal_expts)
probs = counts / counts.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(probs, vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(["0", "1", "2", "3"]); ax.set_yticklabels(["0", "1", "2", "3"])
ax.set_xlabel(r"Measured $n \;\mathrm{mod}\; 4$")
ax.set_ylabel("Prepared Fock state")
wrapped_title = textwrap.fill(str(fname_list), width=60)
ax.set_title(f"Modulo-4 readout confusion matrix {wrapped_title}")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{probs[i, j]:.3f}\n({int(counts[i, j])})",
                ha="center", va="center")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$P(\mathrm{measured}\mid\mathrm{prepared})$")
fig.tight_layout()
plt.show()

print("Counts")
print(counts.astype(int))
print("\nP(measured | prepared)")
print(np.round(probs, 5))
print("\nAssignment fidelities")
for n in range(4):
    print(f"|{n}>: {probs[n, n]:.4f}")
print(f"\nAverage assignment fidelity: {np.mean(np.diag(probs)):.4f}")

# Make this MEASURED matrix the source of truth for all correction cells below.
if np.all(counts.sum(axis=1) > 0):
    set_confusion_matrix(counts)
    print("\n-> Downstream cells now corrected with this measured confusion matrix.")
else:
    set_confusion_matrix(confusion_matrix_manual)
    print("\n-> Calibration incomplete; fell back to the manual confusion matrix.")

# %% [markdown]
# ### Floquet with single parity readout (debugging)

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 200, 200, 2)

meas_stors = [0, 4, 5]
# meas_stors = [0]
swap_stors = [4, 5]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [4, 5]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=500,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '1',
            init_stor= swap_stors[0],
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = True,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = False,
            cond_sec_phase= -90,

            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            
            palindrome_scramble = floquet_default_dict["palindrome_scramble"]
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import os
import textwrap

fname_list = [
    os.path.basename(str(expt.fname))
    for expt in _flat_unique_expts(scramble_expts)
    if getattr(expt, "fname", None) is not None
]

def _flat_unique_expts(items):
    out, seen = [], set()
    def rec(x):
        if isinstance(x, (list, tuple, set)):
            for y in x:
                rec(y)
        else:
            if id(x) not in seen:
                seen.add(id(x))
                out.append(x)
    rec(items)
    return out

def stor_label(ro_stor, params=None):
    """Human-readable label for a readout storage index. `params` is a DarkParams;
    if omitted, falls back to the module-level DARK_PARAMS set by the most recent
    get_dark_params() call."""
    if params is None or not params.swap_man_dark:
        return "M1" if int(ro_stor) == 0 else f"S{int(ro_stor)}"
    if int(ro_stor) == 0:
        return "Dark Mode"
    elif int(ro_stor) == params.dark_swap_order[1]:
        return "Bright Mode"
    else:
        return "Central Mode"

def _get_readout_confusion_2x2(expt):
    rd = expt.cfg.device.readout
    key = (
        "confusion_matrix_with_active_reset"
        if expt.cfg.expt.get("active_reset", False)
        else "confusion_matrix_without_reset"
    )
    C = rd.get(key, None)

    if C is None:
        print(f"WARNING: {key} not found. Using identity correction.")
        return np.eye(2)

    C = np.asarray(C, dtype=float)

    # Lab native format: [Pgg, Pge, Peg, Pee]
    # rows = prepared g/e, cols = measured g/e
    if C.shape == (4,):
        Pgg, Pge, Peg, Pee = C
        C = np.array([[Pgg, Pge],
                      [Peg, Pee]], dtype=float)

    elif C.shape != (2, 2):
        raise ValueError(f"{key} must be flat [Pgg, Pge, Peg, Pee] or 2x2, got {C.shape}")

    return C / C.sum(axis=1, keepdims=True)

def _single_parity_point(expt, point_idx=0, e_is_high_I=None):
    rn = expt.cfg.read_num
    q = expt.cfg.expt.qubits[0]
    threshold = expt.cfg.device.readout.threshold[q]

    if e_is_high_I is None:
        e_is_high_I = expt.cfg.device.readout.Ie[q] > expt.cfg.device.readout.Ig[q]

    idata = np.asarray(expt.data["idata"][point_idx])
    i_final = idata[rn - 1::rn]

    bit1 = i_final > threshold if e_is_high_I else i_final < threshold
    p1 = np.mean(bit1)
    p0 = 1.0 - p1

    return {
        "p_mod0": p0,
        "p_mod1": p1,
        "nmod2_mean": p1,
        "mean_parity": 1.0 - 2.0 * p1,
    }

def _floquet_cycle_us_for_expt(expt):
    ecfg = expt.cfg.expt
    sync_us = float(expt.prog.cycles2us(int(ecfg.get("scramble_sync_cycles", 10)))) if hasattr(expt, "prog") else station.soccfg.cycles2us(10)

    total = 0.0
    for s in ecfg.swap_stors:
        name = f"M1-S{s}"
        if station.ds_floquet.get_waveform(name) in ("gauss", "gaussian", "arb"):
            total += float(station.ds_floquet.get_gauss_sigma(name)) * float(station.ds_floquet.get_gauss_n_sigma(name))
        else:
            total += float(station.ds_floquet.get_len(name))
            total += 4.0 * float(station.ds_floquet.get_ramp_sigma(name))
        total += sync_us
    return total

single_parity_raw = defaultdict(lambda: {
    "xpts": [], "p_mod0": [], "p_mod1": [], "nmod2_mean": [], "mean_parity": []
})

for expt in _flat_unique_expts(scramble_expts):
    ro_stor = expt.cfg.expt.ro_stor
    xpts = np.asarray(expt.data["xpts"]).reshape(-1)

    for j, x in enumerate(xpts):
        r = _single_parity_point(expt, point_idx=j)
        single_parity_raw[ro_stor]["xpts"].append(x)
        for k in ("p_mod0", "p_mod1", "nmod2_mean", "mean_parity"):
            single_parity_raw[ro_stor][k].append(r[k])

for ro_stor in single_parity_raw:
    for k in single_parity_raw[ro_stor]:
        single_parity_raw[ro_stor][k] = np.asarray(single_parity_raw[ro_stor][k])

C2 = _get_readout_confusion_2x2(_flat_unique_expts(scramble_expts)[0])
C2_inv = np.linalg.inv(C2.T)

single_parity_corr = {}
for ro_stor, d in single_parity_raw.items():
    p_meas = np.stack([d["p_mod0"], d["p_mod1"]], axis=1)
    p_corr = p_meas @ C2_inv.T
    p_corr = np.clip(p_corr, 0.0, None)
    p_corr = p_corr / p_corr.sum(axis=1, keepdims=True)

    single_parity_corr[ro_stor] = {
        "xpts": d["xpts"],
        "p_mod0": p_corr[:, 0],
        "p_mod1": p_corr[:, 1],
        "nmod2_mean": p_corr[:, 1],
        "mean_parity": p_corr @ np.array([1.0, -1.0]),
    }

floquet_cycle2us = _floquet_cycle_us_for_expt(_flat_unique_expts(scramble_expts)[0])

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
for ro_stor in single_parity_corr:
    raw = single_parity_raw[ro_stor]
    cor = single_parity_corr[ro_stor]
    si = np.argsort(raw["xpts"])
    x = raw["xpts"][si] * floquet_cycle2us

    line, = ax.plot(x, cor["nmod2_mean"][si], "-", label=stor_label(ro_stor))
    ax.plot(x, raw["nmod2_mean"][si], "--", alpha=0.45, color=line.get_color())

ax.set_xlabel("time (us)")
ax.set_ylabel(r"$\langle n\ {\rm mod}\ 2\rangle = P({\rm odd})$")
ax.set_title("Single parity readout: raw (dashed) vs confusion-corrected (solid)")
# ax.set_ylim(-0.05, 1.05)
ax.legend()

wrapped_title = textwrap.fill(str(fname_list), width=60)
plt.suptitle(f"{wrapped_title}")
plt.tight_layout()
plt.show()

# %%
SHOW_RAW = True
SHOW_CORRECTED = False
PLOT_TIME = False

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

ref_expt = _flat_unique_expts(scramble_expts)[0]

fig, ax = plt.subplots(1, 1, figsize=(9, 5))

for ro_stor in single_parity_raw:
    raw = single_parity_raw[ro_stor]
    cor = single_parity_corr[ro_stor]
    si = np.argsort(raw["xpts"])
    x = raw["xpts"][si] 
    if PLOT_TIME:
        x *= floquet_cycle2us
    label = stor_label_for_expt(ro_stor, ref_expt)

    if SHOW_CORRECTED:
        suffix = " corrected" if SHOW_RAW else ""
        line, = ax.plot(x, cor["nmod2_mean"][si], "-", label=label + suffix)
        color = line.get_color()
    else:
        color = None

    if SHOW_RAW:
        suffix = " raw" if SHOW_CORRECTED else ""
        ls = "--" if SHOW_CORRECTED else "-"
        alpha = 0.45 if SHOW_CORRECTED else 1
        kwargs = dict(linestyle=ls, alpha=alpha, label=label + suffix)
        if color is not None:
            kwargs["color"] = color
        ax.plot(x, raw["nmod2_mean"][si], **kwargs)
if PLOT_TIME:
    ax.set_xlabel("time (us)")
else:
    ax.set_xlabel("Floquet cycles")
    
ax.set_ylabel(r"$\langle n\ {\rm mod}\ 2\rangle = P({\rm odd})$")
ax.set_title("Single parity readout")

# ax.set_ylim(-0.05, 1.05)
ax.legend(loc = "right")

wrapped_title = textwrap.fill(str(fname_list), width=60)
plt.suptitle(f"{wrapped_title}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Floquet with Initial Costate and Multiparity Readout

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 400, 400, 4)

meas_stors = [0, 4, 5]
# meas_stors = [0]
swap_stors = [4, 5]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [4, 5]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=500,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '3',
            init_stor= swap_stors[0],
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = True,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            
            palindrome_scramble = True
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
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

# %%
floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

multiparity = collect_multiparity(unique_expts(scramble_expts))

# %%
for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    parity_first = data['mean_parity_first'][sort_idx]
    parity_second = data['mean_parity_second'][sort_idx]
    nmod4_mean = data['nmod4_mean'][sort_idx]

# %%
import matplotlib.pyplot as plt

def stor_label(ro_stor):
    if ro_stor == 0:
        return "M1"
    return f"S{ro_stor}"


fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    p_first = data['mean_parity_first'][sort_idx]
    p_second = data['mean_parity_second'][sort_idx]

    label = stor_label(ro_stor)

    ax[0].plot(x, p_first, plot_style, label=label)
    ax[1].plot(x, p_second, plot_style, label=label)

ax[0].set_title("First parity expectation")
ax[1].set_title("Second parity-bit expectation")

for a in ax:
    a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    a.set_xlabel("time (us)")
    a.set_ylabel("parity value")
    a.set_ylim(-1.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %%
import textwrap

def flatten_exp_lists(items, container_types=(list, tuple, set)):
    for x in items:
        if isinstance(x, container_types):
            yield from flatten_exp_lists(x, container_types)
        else:
            yield x
            
            
fname_list = []
for exp in flatten_exp_lists(scramble_expts):
    fname_list.append(exp.fname)

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    nmod4_mean = data['nmod4_mean'][sort_idx]

    ax.plot(x, nmod4_mean, plot_style, label=stor_label(ro_stor))

ax.set_xlabel("time (us)")
ax.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
ax.legend()
wrapped_title = textwrap.fill(str(fname_list), width=60)
plt.title(f"{wrapped_title}")  # TODO(stage2): source cell Q180 had a trailing stray 's'
plt.tight_layout()
plt.show()

# %%
# --- Confusion-corrected multiparity (dashed = raw, solid = corrected) ---
multiparity_corr = correct_multiparity(multiparity)
plot_parity_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)
plot_nmod4_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def classify_two_parity_readouts(expt, point_idx=0, threshold=None, e_is_high_I=True):
    rn = expt.cfg.read_num
    qTest = expt.cfg.expt.qubits[0]

    if threshold is None:
        threshold = expt.cfg.device.readout.threshold[qTest]

    idata = np.asarray(expt.data['idata'][point_idx])
    qdata = np.asarray(expt.data['qdata'][point_idx])

    i_first  = idata[rn-2::rn]
    q_first  = qdata[rn-2::rn]
    i_second = idata[rn-1::rn]
    q_second = qdata[rn-1::rn]

    if e_is_high_I:
        first_e = i_first > threshold
        second_e = i_second > threshold
    else:
        first_e = i_first < threshold
        second_e = i_second < threshold

    b0 = first_e.astype(int)
    b1 = second_e.astype(int)

    # cond_sec_phase = -90 convention:
    # (g,g)->0, (e,g)->1, (g,e)->2, (e,e)->3
    n_mod4 = b0 + 2*b1

    out = {
        'i_first': i_first,
        'q_first': q_first,
        'i_second': i_second,
        'q_second': q_second,

        'first_e': first_e,
        'second_e': second_e,

        # parity expectation values:
        # +1 means bit=0, -1 means bit=1.
        # first parity = (-1)^n
        # second parity = +1 for n=0,1 mod 4 and -1 for n=2,3 mod 4.
        'parity_first': 1 - 2*b0,
        'parity_second': 1 - 2*b1,

        'n_mod4': n_mod4,

        'p_first_e': np.mean(first_e),
        'p_second_e': np.mean(second_e),

        'p_gg': np.mean((~first_e) & (~second_e)),
        'p_eg': np.mean(( first_e) & (~second_e)),
        'p_ge': np.mean((~first_e) & ( second_e)),
        'p_ee': np.mean(( first_e) & ( second_e)),
    }

    out['p_mod0'] = np.mean(n_mod4 == 0)
    out['p_mod1'] = np.mean(n_mod4 == 1)
    out['p_mod2'] = np.mean(n_mod4 == 2)
    out['p_mod3'] = np.mean(n_mod4 == 3)

    out['mean_parity_first'] = np.mean(out['parity_first'])
    out['mean_parity_second'] = np.mean(out['parity_second'])
    out['mean_n_mod4'] = np.mean(n_mod4)

    return out

# %%
combined_bits = defaultdict(lambda: {
    'xpts': [],

    'mean_parity_first': [],
    'mean_parity_second': [],

    'p_first_e': [],
    'p_second_e': [],

    'p_mod0': [],
    'p_mod1': [],
    'p_mod2': [],
    'p_mod3': [],

    'p_gg': [],
    'p_eg': [],
    'p_ge': [],
    'p_ee': [],

    'mean_n_mod4': [],
})

floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

unique_expts = []
for sub_list in scramble_expts:
    for expt in sub_list:
        if expt not in unique_expts:
            unique_expts.append(expt)

for expt in unique_expts:
    state_idx = expt.cfg.expt.ro_stor
    xpts = np.asarray(expt.data['xpts']).reshape(-1)

    for j, x in enumerate(xpts):
        r = classify_two_parity_readouts(expt, point_idx=j)

        combined_bits[state_idx]['xpts'].append(x)
        combined_bits[state_idx]['mean_parity_first'].append(r['mean_parity_first'])
        combined_bits[state_idx]['mean_parity_second'].append(r['mean_parity_second'])

        combined_bits[state_idx]['p_first_e'].append(r['p_first_e'])
        combined_bits[state_idx]['p_second_e'].append(r['p_second_e'])

        combined_bits[state_idx]['p_mod0'].append(r['p_mod0'])
        combined_bits[state_idx]['p_mod1'].append(r['p_mod1'])
        combined_bits[state_idx]['p_mod2'].append(r['p_mod2'])
        combined_bits[state_idx]['p_mod3'].append(r['p_mod3'])

        combined_bits[state_idx]['p_gg'].append(r['p_gg'])
        combined_bits[state_idx]['p_eg'].append(r['p_eg'])
        combined_bits[state_idx]['p_ge'].append(r['p_ge'])
        combined_bits[state_idx]['p_ee'].append(r['p_ee'])

        combined_bits[state_idx]['mean_n_mod4'].append(r['mean_n_mod4'])

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
only_line = True
for state_idx, data in combined_bits.items():
    sort_indices = np.argsort(data['xpts'])
    x = np.array(data['xpts'])[sort_indices] * floquet_cycle2us

    p1 = np.array(data['mean_parity_first'])[sort_indices]
    p2 = np.array(data['mean_parity_second'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    label = f"{state_prefix}{display_idx}"
    if not only_line:
        ax[0].plot(x, p1, "o-", label=label)
        ax[1].plot(x, p2, "o-", label=label)
    else:
        ax[0].plot(x, p1,  label=label)
        ax[1].plot(x, p2,  label=label)

ax[0].set_title("First parity expectation")
ax[1].set_title("Second parity-bit expectation")

for a in ax:
    a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    a.set_xlabel("time (us)")
    a.set_ylabel("parity value")
    a.set_ylim(-1.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
max_list = []
only_line = True

for state_idx, data in combined_bits.items():
    sort_indices = np.argsort(data['xpts'])
    x = np.array(data['xpts'])[sort_indices] * floquet_cycle2us

    p0 = np.array(data['p_mod0'])[sort_indices]
    p1 = np.array(data['p_mod1'])[sort_indices]
    p2 = np.array(data['p_mod2'])[sort_indices]
    p3 = np.array(data['p_mod3'])[sort_indices]

    nmod4_mean = p1 + 2*p2 + 3*p3

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    label = f"{state_prefix}{display_idx}"
    if not only_line:
        ax.plot(x, nmod4_mean, "o-", label=label)
    else:
        ax.plot(x, nmod4_mean, label=label)
    max_list.append(nmod4_mean)
ax.set_xlabel("time (us)")
ax.set_ylabel("$\\langle n\\ mod 4\\rangle$")
# ax.set_ylim(np.max(max_list)*0.8, np.max(max_list)*1)
# ax.set_ylim(-0.05, np.max(max_list)*1.2)
ax.legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

scatter = False

combined_data = defaultdict(lambda: {
    'xpts': [],
    'i_first': [],
    'q_first': [],
    'i_second': [],
    'q_second': [],
    'p_first_e': [],
    'p_second_e': [],
    'p_gg': [],
    'p_eg': [],
    'p_ge': [],
    'p_ee': [],
})

floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

unique_expts = []
for sub_list in scramble_expts:
    for expt in sub_list:
        if expt not in unique_expts:
            unique_expts.append(expt)

for expt in unique_expts:
    state_idx = expt.cfg.expt.ro_stor
    rn = expt.cfg.read_num
    qTest = expt.cfg.expt.qubits[0]
    th = expt.cfg.device.readout.threshold[qTest]

    xpts = np.array(expt.data['xpts']).reshape(-1)

    for j, x in enumerate(xpts):
        idata = np.asarray(expt.data['idata'][j])
        qdata = np.asarray(expt.data['qdata'][j])

        i_first  = idata[rn-2::rn]
        q_first  = qdata[rn-2::rn]
        i_second = idata[rn-1::rn]
        q_second = qdata[rn-1::rn]

        first_e = i_first > th
        second_e = i_second > th

        combined_data[state_idx]['xpts'].append(x)
        combined_data[state_idx]['i_first'].append(np.mean(i_first))
        combined_data[state_idx]['q_first'].append(np.mean(q_first))
        combined_data[state_idx]['i_second'].append(np.mean(i_second))
        combined_data[state_idx]['q_second'].append(np.mean(q_second))

        combined_data[state_idx]['p_first_e'].append(np.mean(first_e))
        combined_data[state_idx]['p_second_e'].append(np.mean(second_e))

        # cond_sec_phase = -90 convention:
        # |0> -> (g,g), |1> -> (e,g), |2> -> (g,e), |3> -> (e,e)
        combined_data[state_idx]['p_gg'].append(np.mean((~first_e) & (~second_e)))
        combined_data[state_idx]['p_eg'].append(np.mean(( first_e) & (~second_e)))
        combined_data[state_idx]['p_ge'].append(np.mean((~first_e) & ( second_e)))
        combined_data[state_idx]['p_ee'].append(np.mean(( first_e) & ( second_e)))

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices] * floquet_cycle2us

    i_first = np.array(data['i_first'])[sort_indices]
    q_first = np.array(data['q_first'])[sort_indices]
    i_second = np.array(data['i_second'])[sort_indices]
    q_second = np.array(data['q_second'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    label_str = f"State: {state_prefix}{display_idx}"

    if not scatter:
        ax[0, 0].plot(x_sorted, i_first, "o-", label=label_str)
        ax[0, 1].plot(x_sorted, q_first, "o-", label=label_str)
        ax[1, 0].plot(x_sorted, i_second, "o-", label=label_str)
        ax[1, 1].plot(x_sorted, q_second, "o-", label=label_str)
    else:
        ax[0, 0].scatter(x_sorted, i_first, alpha=0.5, label=label_str)
        ax[0, 1].scatter(x_sorted, q_first, alpha=0.5, label=label_str)
        ax[1, 0].scatter(x_sorted, i_second, alpha=0.5, label=label_str)
        ax[1, 1].scatter(x_sorted, q_second, alpha=0.5, label=label_str)

ax[0, 0].set_title("First parity readout: I")
ax[0, 1].set_title("First parity readout: Q")
ax[1, 0].set_title("Second parity readout: I")
ax[1, 1].set_title("Second parity readout: Q")

for a in ax.flat:
    a.set_xlabel("time (us)")
    a.legend()

ax[0, 0].set_ylabel("Avg I (ADC unit)")
ax[1, 0].set_ylabel("Avg I (ADC unit)")
ax[0, 1].set_ylabel("Avg Q (ADC unit)")
ax[1, 1].set_ylabel("Avg Q (ADC unit)")

plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices] * floquet_cycle2us

    p_first_e = np.array(data['p_first_e'])[sort_indices]
    p_second_e = np.array(data['p_second_e'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    label_str = f"State: {state_prefix}{display_idx}"

    ax[0].plot(x_sorted, p_first_e, "o-", label=label_str)
    ax[1].plot(x_sorted, p_second_e, "o-", label=label_str)

ax[0].set_title("First parity: P(e)")
ax[1].set_title("Second parity: P(e)")

for a in ax:
    a.set_xlabel("time (us)")
    a.set_ylabel("probability")
    a.set_ylim(-0.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Large Support DM

# %% [markdown]
# #### Bare

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 200, 200, 2)

meas_stors = [0, 6, 7]
# meas_stors = [0]
swap_stors = [6, 7]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [6, 7]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=300,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '3',
            init_stor= 6,
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = True,
            swap_man_large_dark = False,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            
            large_dark_direct_sequence = True
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

def unique_expts(nested_expts):
    return list(dict.fromkeys(expt for sub in nested_expts for expt in sub))

multiparity = collect_multiparity(unique_expts(scramble_expts))

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    parity_first = data['mean_parity_first'][sort_idx]
    parity_second = data['mean_parity_second'][sort_idx]
    nmod4_mean = data['nmod4_mean'][sort_idx]

import matplotlib.pyplot as plt

def stor_label(ro_stor):
    if ro_stor == 0:
        return "M1"
    return f"S{ro_stor}"


fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    p_first = data['mean_parity_first'][sort_idx]
    p_second = data['mean_parity_second'][sort_idx]

    label = stor_label(ro_stor)

    ax[0].plot(x, p_first, plot_style, label=label)
    ax[1].plot(x, p_second, plot_style, label=label)

ax[0].set_title("First parity expectation")
ax[1].set_title("Second parity-bit expectation")

for a in ax:
    a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    a.set_xlabel("time (us)")
    a.set_ylabel("parity value")
    a.set_ylim(-1.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %%
fname_list = []
for exp in flatten_exp_lists(scramble_expts):
    fname_list.append(exp.fname)
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    nmod4_mean = data['nmod4_mean'][sort_idx]

    ax.plot(x, nmod4_mean, plot_style, label=stor_label(ro_stor))

ax.set_xlabel("time (us)")
ax.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
ax.legend()
wrapped_title = textwrap.fill(str(fname_list), width=60)
plt.title(f"{wrapped_title}")
plt.tight_layout()
plt.show()

# %%
# --- Confusion-corrected multiparity (dashed = raw, solid = corrected) ---
multiparity_corr = correct_multiparity(multiparity)
plot_parity_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)
plot_nmod4_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)

# %% [markdown]
# #### DM readout

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 300, 300, 2)

meas_stors = [0, 4, 5, 6, 7]
# meas_stors = [0]
swap_stors = phase_modes.copy()
detunings = [0, 0, 0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [4, 5, 6, 7]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=300,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '3',
            init_stor= 4,
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = True,
            swap_man_large_dark = True,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            
            
            large_dark_direct_sequence = True, 
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

multiparity = collect_multiparity(unique_expts(scramble_expts))

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    parity_first = data['mean_parity_first'][sort_idx]
    parity_second = data['mean_parity_second'][sort_idx]
    nmod4_mean = data['nmod4_mean'][sort_idx]

import matplotlib.pyplot as plt

def stor_label(ro_stor):
    if ro_stor == 0:
        return "M1"
    return f"S{ro_stor}"


fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    p_first = data['mean_parity_first'][sort_idx]
    p_second = data['mean_parity_second'][sort_idx]

    label = stor_label(ro_stor)

    ax[0].plot(x, p_first, plot_style, label=label)
    ax[1].plot(x, p_second, plot_style, label=label)

ax[0].set_title("First parity expectation")
ax[1].set_title("Second parity-bit expectation")

for a in ax:
    a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    a.set_xlabel("time (us)")
    a.set_ylabel("parity value")
    a.set_ylim(-1.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %%
fname_list = []
for exp in flatten_exp_lists(scramble_expts):
    fname_list.append(exp.fname)
fig, ax = plt.subplots(1, 1, figsize=(9, 10))

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    nmod4_mean = data['nmod4_mean'][sort_idx]

    ax.plot(x, nmod4_mean, plot_style, label=stor_label(ro_stor))

ax.set_xlabel("time (us)")
ax.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
ax.legend()
wrapped_title = textwrap.fill(str(fname_list), width=60)
plt.title(f"{wrapped_title}")
plt.tight_layout()
plt.show()

# %%
# --- Confusion-corrected multiparity (dashed = raw, solid = corrected) ---
multiparity_corr = correct_multiparity(multiparity)
plot_parity_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)
plot_nmod4_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)

# %% [markdown]
# #### DM load and readout

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 300, 200, 2)

meas_stors = [0, 4, 5, 6, 7]
# meas_stors = [0]
swap_stors = [4, 5, 6, 7]
detunings = [0, 0, 0, 0] # None/False/unspecified all default to all zeros
# detunings = [150e-3, 150e-3, 150e-3, 150e-3] # None/False/unspecified all default to all zeros
dark_swaps = [4, 5, 6, 7]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=300,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '3',
            init_stor= 0,
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            load_man_dark = True,
            swap_man_dark = True,
            swap_man_large_dark = True,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            
            large_dark_direct_sequence = True, 
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

multiparity = collect_multiparity(unique_expts(scramble_expts))

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    parity_first = data['mean_parity_first'][sort_idx]
    parity_second = data['mean_parity_second'][sort_idx]
    nmod4_mean = data['nmod4_mean'][sort_idx]

import matplotlib.pyplot as plt

def stor_label(ro_stor):
    if ro_stor == 0:
        return "M1"
    return f"S{ro_stor}"


fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    p_first = data['mean_parity_first'][sort_idx]
    p_second = data['mean_parity_second'][sort_idx]

    label = stor_label(ro_stor)

    ax[0].plot(x, p_first, plot_style, label=label)
    ax[1].plot(x, p_second, plot_style, label=label)

ax[0].set_title("First parity expectation")
ax[1].set_title("Second parity-bit expectation")

for a in ax:
    a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    a.set_xlabel("time (us)")
    a.set_ylabel("parity value")
    a.set_ylim(-1.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %%
fname_list = []
for exp in flatten_exp_lists(scramble_expts):
    fname_list.append(exp.fname)
fig, ax = plt.subplots(1, 1, figsize=(9, 10))

only_line = True
plot_style = "-" if only_line else "o-"

for ro_stor, data in multiparity.items():
    sort_idx = np.argsort(data['xpts'])
    x = data['xpts'][sort_idx] * floquet_cycle2us

    nmod4_mean = data['nmod4_mean'][sort_idx]

    ax.plot(x, nmod4_mean, plot_style, label=stor_label(ro_stor))

ax.set_xlabel("time (us)")
ax.set_ylabel("$\\langle n\\ mod\\ 4\\rangle$")
ax.legend()
wrapped_title = textwrap.fill(str(fname_list), width=60)
plt.title(f"{wrapped_title}")
plt.tight_layout()
plt.show()

# %%
# --- Confusion-corrected multiparity (dashed = raw, solid = corrected) ---
multiparity_corr = correct_multiparity(multiparity)
plot_parity_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)
plot_nmod4_raw_vs_corrected(
    multiparity, multiparity_corr,
    floquet_cycle2us=floquet_cycle2us, title=fname_list,
)

# %% [markdown]
# #### Large DM detuning dependent decay analysis

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.notebook import tqdm
import pandas as pd


def as_floquet_sweep_list(floquet_chunk):
    """
    floquet_cycle_list_gen(...) output can be scalar, list, or ndarray.
    DarkBaseExperiment expects cfg.expt.floquet_cycles for swept_params=['floquet_cycle'].
    Therefore always pass a 1D Python list.
    """
    arr = np.asarray(floquet_chunk)

    if arr.ndim == 0:
        return [int(arr.item())]

    return [int(x) for x in arr.reshape(-1)]


def flatten_floquet_chunks(floquet_cycles_list):
    vals = []
    for chunk in floquet_cycles_list:
        vals.extend(as_floquet_sweep_list(chunk))
    return vals


def stor_label(ro_stor):
    if ro_stor == 0:
        return "M1"
    return f"S{ro_stor}"

# %%
def sample_storage_disorder(
    epsilon,
    swap_stors,
    rng,
    distribution="uniform",
    zero_mean_each_realization=False,
):
    """
    detunings[k] is applied to M1-S{swap_stors[k]}.

    epsilon has the same unit as cfg.expt.detunings, i.e. the unit expected by
        pulse_args["freq"] += self.freq2reg(detunings[k], ...)
    """
    n = len(swap_stors)
    epsilon = float(epsilon)

    if abs(epsilon) < 1e-15:
        draws = np.zeros(n, dtype=float)

    elif distribution == "uniform":
        # Notebook-style random disorder: [-epsilon, +epsilon]
        draws = rng.uniform(-1.0, 1.0, size=n)

    elif distribution == "normal":
        draws = rng.normal(0.0, 1.0, size=n)

    elif distribution == "binary":
        draws = rng.choice([-1.0, 1.0], size=n)

    elif distribution == "alt":
        # deterministic debug disorder
        draws = np.array([1.0 if k % 2 == 0 else -1.0 for k in range(n)])

    else:
        raise ValueError(f"Unknown disorder distribution: {distribution}")

    if zero_mean_each_realization and n > 0:
        draws = draws - np.mean(draws)

    detunings = epsilon * draws
    return detunings.astype(float), draws.astype(float)


def build_disorder_tasks(
    disorder_strengths,
    num_realizations,
    seed,
    zero_epsilon_realizations=1,
):
    """
    Build reproducible disorder task list.

    For epsilon=0, repeated realizations are physically identical except shot noise,
    so by default only one epsilon=0 realization is run.
    """
    disorder_strengths = [float(x) for x in disorder_strengths]

    n_total = 0
    for eps in disorder_strengths:
        n_total += zero_epsilon_realizations if abs(eps) < 1e-15 else num_realizations

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(n_total)

    tasks = []
    task_index = 0

    for epsilon in disorder_strengths:
        n_rep = zero_epsilon_realizations if abs(epsilon) < 1e-15 else num_realizations

        for realization in range(n_rep):
            child_seed = int(
                child_seeds[task_index].generate_state(1, dtype=np.uint32)[0]
            )

            tasks.append({
                "epsilon": float(epsilon),
                "realization": int(realization),
                "seed": child_seed,
            })

            task_index += 1

    return tasks

# %%
dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 300, 300, 2)

meas_stors = [0]
swap_stors = [4, 5, 6, 7]
dark_swaps = [4, 5, 6, 7]

disorder_strengths = np.array([0.0, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1]) * 30.9 * 1e-3
num_disorder_realizations = 8
disorder_rng_seed = 20260719
disorder_distribution = "uniform"
zero_mean_each_realization = False

floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

print("floquet_cycle2us =", floquet_cycle2us)

# %%
base_execute_kwargs = dict(
    reps=100,

    init_fock=False,
    init_man_fock_state="3",
    init_stor=0,

    relax_delay=1000,
    active_reset=True,
    pre_relax_delay=100,
    man_reset=True,
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    dump_reset_iter_num=active_reset_default_dict["dump_reset_iter_num"],

    update_phases=True,

    custom_prepulse=False,
    custom_postpulse=False,
    debug=False,

    load_man_dark=True,
    swap_man_dark=True,
    swap_man_large_dark=True,

    second_rel_phase=180,
    map_to_qubit_ge=True,

    multiparity_readout=True,
    cond_sec_phase=-90,

    prepulse=True,
    postpulse=True,
    
    
    large_dark_direct_sequence = True, 
)

# %%
def run_one_disorder_record(task):
    epsilon = float(task["epsilon"])
    realization = int(task["realization"])
    seed = int(task["seed"])

    rng = np.random.default_rng(seed)

    detunings_this, draws_this = sample_storage_disorder(
        epsilon=epsilon,
        swap_stors=swap_stors,
        rng=rng,
        distribution=disorder_distribution,
        zero_mean_each_realization=zero_mean_each_realization,
    )

    record = {
        "epsilon": epsilon,
        "realization": realization,
        "seed": seed,
        "swap_stors": list(swap_stors),
        "dark_swaps": list(dark_swaps),
        "draws": draws_this.copy(),
        "detunings": detunings_this.copy(),
        "expts": [],
        "fnames": [],
    }

    desc_prefix = (
        f"eps={epsilon:g}, r={realization}, "
        f"det={np.round(detunings_this, 6).tolist()}"
    )
    print(desc_prefix)

    for meas_stor in meas_stors:
        expts_for_this_ro = []

        for floquet_chunk in tqdm(
            floquet_cycles_list,
            desc=f"{desc_prefix}, ro={stor_label(meas_stor)}",
            leave=False,
        ):
            floquet_sweep_values = as_floquet_sweep_list(floquet_chunk)

            kwargs = dict(base_execute_kwargs)
            kwargs.update(
                ro_stor=meas_stor,

                storage_reset=list(swap_stors),
                swap_stors=list(swap_stors),
                dark_swap_order=list(dark_swaps),

                detunings=detunings_this.tolist(),

                # Important:
                # floquet_sweep_values is a list, not an int.
                floquet_cycles=floquet_sweep_values,
                swept_params=["floquet_cycle"],

                # Not used by the pulse program, but saved into exp.cfg.expt.
                disorder_epsilon=epsilon,
                disorder_realization=realization,
                disorder_seed=seed,
                disorder_distribution=disorder_distribution,
                disorder_draws=draws_this.tolist(),
                disorder_detunings=detunings_this.tolist(),
            )

            exp = dmscramble_runner.execute(**kwargs)

            expts_for_this_ro.append(exp)

            if hasattr(exp, "fname"):
                record["fnames"].append(exp.fname)

        record["expts"].append(expts_for_this_ro)

    return record

# %%
disorder_tasks = build_disorder_tasks(
    disorder_strengths=disorder_strengths,
    num_realizations=num_disorder_realizations,
    seed=disorder_rng_seed,
    zero_epsilon_realizations=1,
)

disorder_records = []

for task in tqdm(disorder_tasks, desc="large-dark disorder sweep"):
    rec = run_one_disorder_record(task)
    disorder_records.append(rec)

len(disorder_records)

# %%
def make_disorder_fname_table(disorder_records):
    rows = []

    for rec in disorder_records:
        rows.append({
            "epsilon": rec["epsilon"],
            "realization": rec["realization"],
            "seed": rec["seed"],
            "detunings": np.round(rec["detunings"], 6).tolist(),
            "num_files": len(rec["fnames"]),
            "fnames": rec["fnames"],
        })

    return pd.DataFrame(rows)
pd.set_option('display.max_colwidth', None)

fname_df = make_disorder_fname_table(disorder_records)
fname_df

# %%
def _get_obs_array(data, obs_key):
    if obs_key in data:
        return np.asarray(data[obs_key])

    if obs_key == "nmod4_mean" and "mean_n_mod4" in data:
        return np.asarray(data["mean_n_mod4"])

    raise KeyError(
        f"Observable key {obs_key!r} not found. Available keys: {list(data.keys())}"
    )


def build_disorder_ensemble_summary(
    disorder_records,
    floquet_cycle2us,
    obs_keys=(
        "mean_parity_first",
        "mean_parity_second",
        "nmod4_mean",
        "p_mod0",
        "p_mod1",
        "p_mod2",
        "p_mod3",
    ),
    ddof=0,
):
    grouped = defaultdict(lambda: defaultdict(lambda: {
        "x": None,
        "records": [],
        "traces": defaultdict(list),
        "fnames": [],
    }))

    for rec in disorder_records:
        epsilon = float(rec["epsilon"])
        realization = int(rec["realization"])

        multiparity = collect_multiparity(unique_expts(rec["expts"]))

        for ro_stor, data in multiparity.items():
            ro_stor = int(ro_stor)

            xpts = np.asarray(data["xpts"], dtype=float)
            sort_idx = np.argsort(xpts)
            x = xpts[sort_idx] * float(floquet_cycle2us)

            bucket = grouped[epsilon][ro_stor]

            if bucket["x"] is None:
                bucket["x"] = x
            elif not np.allclose(bucket["x"], x, rtol=0, atol=1e-9):
                raise ValueError(
                    f"x grid mismatch: epsilon={epsilon}, "
                    f"ro_stor={ro_stor}, realization={realization}"
                )

            for obs_key in obs_keys:
                y = _get_obs_array(data, obs_key)[sort_idx]
                bucket["traces"][obs_key].append(np.asarray(y, dtype=float))

            bucket["records"].append({
                "realization": realization,
                "seed": int(rec["seed"]),
                "detunings": np.asarray(rec["detunings"], dtype=float),
                "draws": np.asarray(rec["draws"], dtype=float),
                "fnames": list(rec["fnames"]),
            })

            bucket["fnames"].extend(rec["fnames"])

    summary = {}

    for epsilon, ro_dict in grouped.items():
        summary[epsilon] = {}

        for ro_stor, bucket in ro_dict.items():
            summary[epsilon][ro_stor] = {
                "x": bucket["x"],
                "records": bucket["records"],
                "fnames": bucket["fnames"],
            }

            for obs_key, trace_list in bucket["traces"].items():
                traces = np.vstack(trace_list)

                std = traces.std(
                    axis=0,
                    ddof=ddof if traces.shape[0] > ddof else 0,
                )
                sem = std / np.sqrt(traces.shape[0])

                summary[epsilon][ro_stor][obs_key] = {
                    "traces": traces,
                    "mean": traces.mean(axis=0),
                    "std": std,
                    "sem": sem,
                    "n_realizations": traces.shape[0],
                }

    return summary


disorder_summary = build_disorder_ensemble_summary(
    disorder_records,
    floquet_cycle2us=floquet_cycle2us,
)

# %%
def plot_ensemble_with_realizations(
    disorder_summary,
    *,
    obs_key="nmod4_mean",
    ro_stor=0,
    strengths=None,
    ylabel=None,
    title=None,
    show_realizations=True,
    show_sem=True,
    raw_alpha=0.18,
    raw_lw=0.8,
    mean_lw=2.0,
    ax=None,
):
    if strengths is None:
        strengths = sorted(disorder_summary.keys())
    else:
        strengths = [float(x) for x in strengths]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    else:
        fig = ax.figure

    for epsilon in strengths:
        if epsilon not in disorder_summary:
            continue
        if ro_stor not in disorder_summary[epsilon]:
            continue
        if obs_key not in disorder_summary[epsilon][ro_stor]:
            continue

        entry = disorder_summary[epsilon][ro_stor]
        info = entry[obs_key]

        x = entry["x"]
        traces = info["traces"]
        y_mean = info["mean"]
        y_sem = info["sem"]
        n_realizations = info["n_realizations"]

        mean_line, = ax.plot(
            x,
            y_mean,
            lw=mean_lw,
            label=fr"$\epsilon={epsilon:g}$, R={n_realizations}",
            zorder=3,
        )
        color = mean_line.get_color()

        if show_realizations:
            for y in traces:
                ax.plot(
                    x,
                    y,
                    lw=raw_lw,
                    alpha=raw_alpha,
                    color=color,
                    zorder=1,
                )

        if show_sem and n_realizations > 1:
            ax.fill_between(
                x,
                y_mean - y_sem,
                y_mean + y_sem,
                alpha=0.13,
                color=color,
                linewidth=0,
                zorder=2,
            )

    ax.set_xlabel("time (us)")

    if ylabel is None:
        ylabel = obs_key
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"{obs_key}: disorder ensemble mean, readout={stor_label(ro_stor)}"
    ax.set_title(title)

    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()

    return fig, ax

# %%
fig, ax = plot_ensemble_with_realizations(
    disorder_summary,
    obs_key="nmod4_mean",
    ro_stor=0,
    strengths=disorder_strengths,
    ylabel=r"$\langle n_{\rm dark}\ {\rm mod}\ 4\rangle$",
    title="Central + four storage modes: disorder ensemble mean",
    show_realizations=True,
    show_sem=True,
)

plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=130, sharex=True)

plot_ensemble_with_realizations(
    disorder_summary,
    obs_key="mean_parity_first",
    ro_stor=0,
    strengths=disorder_strengths,
    ylabel=r"$\langle \Pi_1\rangle$",
    title="First parity-bit expectation",
    show_realizations=True,
    show_sem=True,
    ax=axs[0],
)

plot_ensemble_with_realizations(
    disorder_summary,
    obs_key="mean_parity_second",
    ro_stor=0,
    strengths=disorder_strengths,
    ylabel=r"$\langle \Pi_2\rangle$",
    title="Second parity-bit expectation",
    show_realizations=True,
    show_sem=True,
    ax=axs[1],
)

for ax in axs:
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.set_ylim(-1.05, 1.05)

plt.tight_layout()
plt.show()

# %%
def print_fnames_for_epsilon(disorder_summary, epsilon, ro_stor=0):
    epsilon = float(epsilon)

    fnames = disorder_summary[epsilon][ro_stor]["fnames"]

    print(f"epsilon = {epsilon:g}, ro_stor = {stor_label(ro_stor)}")
    print(f"num files = {len(fnames)}")
    print()

    for k, fname in enumerate(fnames):
        print(f"{k:03d}: {fname}")


print_fnames_for_epsilon(disorder_summary, epsilon=0.05, ro_stor=0)

# %% [markdown]
# ##### Confusion-corrected disorder ensemble
#
# Same ensemble as above, with the readout modulo-4 confusion-matrix correction applied per realization / per time point (uses the matrix set by the `Multi photon fock` confusion-matrix cell), then re-averaged. Reuses `plot_ensemble_with_realizations`.

# %%
# ---- Confusion-corrected disorder ensemble ------------------------------------
# Re-derive the ensemble summary with the readout confusion-matrix correction
# applied per realization / per time point, reusing the per-realization p_mod0..3
# traces already stored in `disorder_summary` (no experiment re-reads). Requires
# the confusion matrix to have been set (Multi photon fock -> confusion-matrix
# cell, which calls set_confusion_matrix()).
assert 'confusion_inv' in globals(), (
    "Confusion matrix not set -- run the 'Multi photon fock' confusion-matrix "
    "cell (set_confusion_matrix) before correcting the disorder ensemble."
)


def correct_disorder_summary(disorder_summary, clip_renormalize=True, ddof=0):
    """Confusion-correct a disorder ensemble summary.

    Inverts the confusion matrix on the per-realization modulo-4 distribution at
    every time point and recomputes nmod4_mean / mean_parity_first /
    mean_parity_second (and the p_mod* traces). Returns a summary with the SAME
    structure as the input, so plot_ensemble_with_realizations() works unchanged.
    """
    mod_vals = np.array([0, 1, 2, 3])
    parity_first_signs  = 1 - 2 * (mod_vals % 2)     # [+1, -1, +1, -1]
    parity_second_signs = 1 - 2 * (mod_vals // 2)    # [+1, +1, -1, -1]

    def _stat(traces):
        traces = np.asarray(traces, dtype=float)
        n = traces.shape[0]
        std = traces.std(axis=0, ddof=ddof if n > ddof else 0)
        return {"traces": traces, "mean": traces.mean(axis=0),
                "std": std, "sem": std / np.sqrt(n), "n_realizations": n}

    corrected = {}
    for epsilon, ro_dict in disorder_summary.items():
        corrected[epsilon] = {}
        for ro_stor, entry in ro_dict.items():
            # measured modulo-4 distribution: (n_real, n_points, 4)
            p_meas = np.stack(
                [entry["p_mod0"]["traces"], entry["p_mod1"]["traces"],
                 entry["p_mod2"]["traces"], entry["p_mod3"]["traces"]],
                axis=-1,
            )
            n_real, n_pts, _ = p_meas.shape
            p_corr = correct_distribution(
                p_meas.reshape(-1, 4), clip_renormalize=clip_renormalize
            ).reshape(n_real, n_pts, 4)

            corrected[epsilon][ro_stor] = {
                "x": entry["x"],
                "records": entry["records"],
                "fnames": entry["fnames"],
                "nmod4_mean":         _stat(p_corr @ mod_vals),
                "mean_parity_first":  _stat(p_corr @ parity_first_signs),
                "mean_parity_second": _stat(p_corr @ parity_second_signs),
                "p_mod0": _stat(p_corr[:, :, 0]),
                "p_mod1": _stat(p_corr[:, :, 1]),
                "p_mod2": _stat(p_corr[:, :, 2]),
                "p_mod3": _stat(p_corr[:, :, 3]),
            }
    return corrected


disorder_summary_corrected = correct_disorder_summary(disorder_summary)

# %%
fig, ax = plot_ensemble_with_realizations(
    disorder_summary_corrected,
    obs_key="nmod4_mean",
    ro_stor=0,
    strengths=disorder_strengths,
    ylabel=r"$\langle n_{\rm dark}\ {\rm mod}\ 4\rangle$ (confusion-corrected)",
    title="Central + four storage modes: disorder ensemble mean (confusion-corrected)",
    show_realizations=True,
    show_sem=True,
)

plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=130, sharex=True)

plot_ensemble_with_realizations(
    disorder_summary_corrected,
    obs_key="mean_parity_first",
    ro_stor=0,
    strengths=disorder_strengths,
    ylabel=r"$\langle \Pi_1\rangle$",
    title="First parity-bit expectation (confusion-corrected)",
    show_realizations=True,
    show_sem=True,
    ax=axs[0],
)

plot_ensemble_with_realizations(
    disorder_summary_corrected,
    obs_key="mean_parity_second",
    ro_stor=0,
    strengths=disorder_strengths,
    ylabel=r"$\langle \Pi_2\rangle$",
    title="Second parity-bit expectation (confusion-corrected)",
    show_realizations=True,
    show_sem=True,
    ax=axs[1],
)

for ax in axs:
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.set_ylim(-1.05, 1.05)

plt.tight_layout()
plt.show()

# %%
# Raw vs confusion-corrected ensemble-mean <n mod 4> (dashed = raw, solid = corrected).
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
ro_stor = 0
for epsilon in sorted(disorder_summary.keys()):
    if ro_stor not in disorder_summary.get(epsilon, {}):
        continue
    if ro_stor not in disorder_summary_corrected.get(epsilon, {}):
        continue
    raw = disorder_summary[epsilon][ro_stor]
    cor = disorder_summary_corrected[epsilon][ro_stor]
    line, = ax.plot(cor["x"], cor["nmod4_mean"]["mean"], "-",
                    lw=2.0, label=fr"$\epsilon={epsilon:g}$")
    ax.plot(raw["x"], raw["nmod4_mean"]["mean"], "--",
            lw=1.5, alpha=0.4, color=line.get_color())

ax.set_xlabel("time (us)")
ax.set_ylabel(r"$\langle n_{\rm dark}\ {\rm mod}\ 4\rangle$")
ax.set_title(f"Disorder ensemble mean: raw (dashed) vs confusion-corrected (solid), "
             f"readout={stor_label(ro_stor)}")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Debyggubg

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandScrambleDarkProgramDebug,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

num_list = np.arange(1, 21, 1)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [6, 7]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [6, 7]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble = dmscramble_runner.execute(
        reps=1000,
        init_fock=True,
        # init_alpha = np.sqrt(3),
        init_man_fock_state = '1',
        init_stor= 0,
        ro_stor = meas_stor,
        relax_delay=8000,
        active_reset=False,
        pre_relax_delay = 300, 
        man_reset=True, 
        storage_reset = swap_stors, 
        reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
        dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        swap_stors=swap_stors,
        update_phases=True,
        detunings=detunings,
        floquet_cycles=[0],
        number_of_load_unloads = num_list,
        swept_params = ['number_of_load_unload'],
        custom_prepulse = False,
        custom_postpulse = False,
        debug = True,
        swap_man_dark = True,
        swap_man_large_dark = False,
        dark_swap_order = dark_swaps,
        second_rel_phase = 180,
        map_to_qubit_ge = True, 
        
        multiparity_readout = False,
        cond_sec_phase= -90,

        prepulse = True, #for debugging. Should always be true
        postpulse = True, #for debugging. Should always be true
        )
    scramble_expts.append(scramble)
# scramble.display()

# %% [markdown]
# ## Detuning - dependent decay - First Attempt ()

# %% [markdown]
# ### beam splitting interaction calibration

# %%
from tqdm.notebook import tqdm


dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 120, 120, 10)

meas_stors = [0, 3]
# meas_stors = [0]
swap_stors = [3]
detunings = [0] # None/False/unspecified all default to all zeros
dark_swaps = [4, 5, 6, 7]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=1000,
            init_fock=True,
            init_stor=[2,0],
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 200,
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = True,
            swap_man_dark = False,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            map_to_qubit_ge = True,
            parity_readout = True,
            parity_fast = False,
            phase_second_pulse = 0,
            prepulse = True, #for debugging. Should always be true
            postpulse = True #for debugging. Should always be true
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
def floquet_cycle_to_time(swap_stors):
    length_per_cycle = 0
    for stor in swap_stors:
        length_per_cycle += station.ds_floquet.get_len(f"M1-S{stor}")
        length_per_cycle += station.soccfg.cycles2us(10)
    return length_per_cycle

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

fig, ax = plt.subplots(1, 2, figsize=(18, 5))
scatter = False
combined_data = defaultdict(lambda: {'xpts': [], 'avgi': [], 'avgq': []})

unique_expts = []
for sub_list in scramble_expts:
    for expt in sub_list:
        if expt not in unique_expts:
            unique_expts.append(expt)

for expt in unique_expts:
    state_idx = expt.cfg.expt.ro_stor
    
    combined_data[state_idx]['xpts'].extend(expt.data['xpts'])
    combined_data[state_idx]['avgi'].extend(expt.data['avgi'])
    combined_data[state_idx]['avgq'].extend(expt.data['avgq']) 

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices] * floquet_cycle_to_time(swap_stors)
    i_sorted = np.array(data['avgi'])[sort_indices]
    q_sorted = np.array(data['avgq'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1
        
    label_str = f"State: {state_prefix}{display_idx}"
    if not scatter:
        ax[0].plot(x_sorted, i_sorted, label=label_str)
        ax[1].plot(x_sorted, q_sorted, label=label_str)
    else:
        ax[0].scatter(x_sorted, i_sorted, alpha=0.5, label=label_str)
        ax[1].scatter(x_sorted, q_sorted, alpha=0.5, label=label_str)

ax[0].set_xlabel("Time (us)")
ax[1].set_xlabel("Time (us)")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg Q (ADC unit)")
ax[0].legend()
ax[1].legend()

plt.show()

# %%
from scipy.optimize import curve_fit

def cos_model(t, offset, amp, freq, phase):
    return offset + amp * np.cos(2 * np.pi * freq * t + phase)

def guess_freq_fft(t, y):
    t = np.asarray(t)
    y = np.asarray(y) - np.mean(y)

    tu = np.linspace(t.min(), t.max(), len(t))
    yu = np.interp(tu, t, y)

    dt = tu[1] - tu[0]
    freqs = np.fft.rfftfreq(len(tu), d=dt)
    spec = np.abs(np.fft.rfft(yu))
    spec[0] = 0

    return freqs[np.argmax(spec)]

# %%
t_fit_data = x_sorted - x_sorted.min()
i_fit_data = i_sorted

freq_guess = guess_freq_fft(t_fit_data, i_fit_data)
offset_guess = np.mean(i_fit_data)
amp_guess = 0.5 * (np.max(i_fit_data) - np.min(i_fit_data))
phase_guess = 0.0

p0 = [offset_guess, amp_guess, freq_guess, phase_guess]

popt, pcov = curve_fit(
    cos_model,
    t_fit_data,
    i_fit_data,
    p0=p0,
    maxfev=20000
)

offset, amp, freq, phase = popt
freq_err = np.sqrt(np.diag(pcov))[2]

print(f"{label_str}: freq = {freq:.6f} MHz = {freq * 1e3:.3f} kHz")
print(f"{label_str}: freq err ~ {freq_err:.6f} MHz")

# %%
t_dense = np.linspace(t_fit_data.min(), t_fit_data.max(), 1000)
i_dense = cos_model(t_dense, *popt)

plt.plot(x_sorted, i_sorted, ".", alpha=0.5, label=label_str)
plt.plot(t_dense + x_sorted.min(), i_dense, "-", label=f"{label_str} fit, f={freq*1e3:.2f} kHz")

# %% [markdown]
# ### decay analysis

# %%
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import qmc

sampler = qmc.Sobol(d=4, scramble=True)
m = 4
sobol_floats = sampler.random_base2(m=m)  
random_binarys = np.where(sobol_floats < 0.5, -1, 1) # have no idea what does this do, but confirmed that for m = 4 and d = 4, the result is all possible combinations.


dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)


g = 11.09e-3 #should be measured above
detuning_list = g * np.logspace(-2, 1, 5)
np.insert(detuning_list, 0, 0)
# detuning_list = [0]

floquet_cycles_list = floquet_cycle_list_gen(0, 300, 100, 5)
meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3, 5, 6]
# detunings = [0, 0, 0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [3, 5]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for detun in tqdm(detuning_list):
    scramble_sub_expts = []
    for random_binary in tqdm(random_binarys):
        scramble_subsub_expts = []
        detunings = (random_binary * detun).astype(float).tolist()
        for floquet_cycles in floquet_cycles_list:
            scramble = dmscramble_runner.execute(
                reps=100,
                # init_fock=True,
                init_alpha = 2,
                init_stor=3,
                ro_stor = meas_stors[0],
                relax_delay=200,
                active_reset=True,
                pre_relax_delay = 3000,
                man_reset=True, 
                storage_reset = swap_stors, 
                reset_dump_mode = 1,
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
                swap_stors=swap_stors,
                update_phases=True,
                detunings=detunings,
                floquet_cycles=floquet_cycles,
                swept_params=['floquet_cycle'],
                custom_prepulse = False,
                custom_postpulse = False,
                debug = False,
                swap_man_dark = True,
                dark_swap_order = dark_swaps,
                second_rel_phase = 180,
                map_to_qubit_ge = True, 
                parity_readout = True,
                parity_fast = False,
                phase_second_pulse = 180,
                prepulse = True, #for debugging. Should always be true
                postpulse = True #for debugging. Should always be trues
                )
            scramble_subsub_expts.append(scramble)
        scramble_sub_expts.append(scramble_subsub_expts)
    scramble_expts.append(scramble_sub_expts)

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

fig, ax = plt.subplots(1, 2, figsize=(18, 5))
scatter = False
combined_data = defaultdict(lambda: {'xpts': [], 'avgi': [], 'avgq': []})

unique_expts = []
for subsub_list in scramble_expts:
    for expt in sub_list:
        if expt not in unique_expts:
            unique_expts.append(expt)

for expt in unique_expts:
    state_idx = expt.cfg.expt.ro_stor
    
    combined_data[state_idx]['xpts'].extend(expt.data['xpts'])
    combined_data[state_idx]['avgi'].extend(expt.data['avgi'])
    combined_data[state_idx]['avgq'].extend(expt.data['avgq']) 

for state_idx, data in combined_data.items():
    sort_indices = np.argsort(data['xpts'])
    x_sorted = np.array(data['xpts'])[sort_indices] * floquet_cycle_to_time(swap_stors)
    i_sorted = np.array(data['avgi'])[sort_indices]
    q_sorted = np.array(data['avgq'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1
        
    label_str = f"State: {state_prefix}{display_idx}"
    if not scatter:
        ax[0].plot(x_sorted, i_sorted, label=label_str)
        ax[1].plot(x_sorted, q_sorted, label=label_str)
    else:
        ax[0].scatter(x_sorted, i_sorted, alpha=0.5, label=label_str)
        ax[1].scatter(x_sorted, q_sorted, alpha=0.5, label=label_str)

ax[0].set_xlabel("Time (us)")
ax[1].set_xlabel("Time (us)")
ax[0].set_ylabel("Avg I (ADC unit)")
ax[1].set_ylabel("Avg Q (ADC unit)")
ax[0].legend()
ax[1].legend()

plt.show()

# %% [markdown]
# ## Single shot distribution inv

# %%
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = singleshot_postproc,
    job_client=client,
)

ss = ss_runner.execute(
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    priority=1,
)
ss.display(station)

# %%
from tqdm.notebook import tqdm


dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles = [0]

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros

scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble = dmscramble_runner.execute(
        reps=5000,
        init_fock=True,
        init_stor=0,
        ro_stor = meas_stor,
        relax_delay=200,
        active_reset=True,
        pre_relax_delay = 1000,
        man_reset=True, 
        storage_reset = [2, 3], 
        reset_dump_mode = 1,
        dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        swap_stors=swap_stors,
        update_phases=True,
        detunings=detunings,
        floquet_cycles=floquet_cycles,  
        swept_params=['floquet_cycle'],
        custom_prepulse = False,
        custom_postpulse = False,
        debug = False,
        swap_man_dark = True,
        dark_swap_order = [2, 3],
        second_rel_phase = 180,
        map_to_qubit_ge = True,
        prepulse = True, #for debugging. Should always be true
        postpulse = True #for debugging. Should always be true
        )
    scramble_expts.append(scramble)
    # scramble.display()

# %%
import numpy as np
import matplotlib.pyplot as plt
from experiments.MM_base import MMAveragerProgram

idx_to_plot = 0
subidx = 0

I_threshold = 0.0 

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax, ax1 = axes

read_num = 1
if scramble_expts[idx_to_plot].cfg.expt.get('parity_check', False):
    read_num += 1
if scramble_expts[idx_to_plot].cfg.expt.get('active_reset', False):
    params = MMAveragerProgram.get_active_reset_params(scramble_expts[idx_to_plot].cfg)
    read_num += MMAveragerProgram.active_reset_read_num(**params)
_start_idx = read_num-1

idata_g = ss.data['Ig']
qdata_g = ss.data['Qg']
idata_scramble = scramble_expts[idx_to_plot].data['idata'][subidx][_start_idx::read_num]
qdata_scramble = scramble_expts[idx_to_plot].data['qdata'][subidx][_start_idx::read_num]

frac_g = np.mean(idata_g > I_threshold) * 100 
frac_scramble = np.mean(idata_scramble > I_threshold) * 100

ax.scatter(idata_g, qdata_g, alpha=0.1, label='single shot of g')
ax.scatter(idata_scramble, qdata_scramble, alpha=0.2, label='single shot after initialization')

ax1.hist(idata_scramble, bins=100)

ax.axvline(x=I_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({I_threshold})')
ax1.axvline(x=I_threshold, color='red', linestyle='--', alpha=0.7)

ax.set_title(
    f"Average I: {np.round(np.average(idata_scramble), 2)} | Desired Avg I: {np.round(np.average(idata_g), 2)}\n"
    f"g > threshold: {frac_g:.1f}% | init > threshold: {frac_scramble:.1f}%"
)
ax1.set_title(f"Histogram (init > threshold: {frac_scramble:.1f}%)")

ax.legend()
plt.show()

# %% [markdown]
# ## Prerelax/Relax delay vs average I

# %%
dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles = [0]

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros


rel_phase_list = [0]
total_exps = []
pre_relax_delay_list = np.arange(0, 5001, 500)
for pre_relax_delay in pre_relax_delay_list:
    scramble_expts = []
    for meas_stor in meas_stors:
        scramble = dmscramble_runner.execute(
            reps=5000,
            init_fock=True,
            init_stor=1,
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = pre_relax_delay,
            man_reset=True, 
            storage_reset = [2, 3], 
            reset_dump_mode = 1,
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,  
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = True,
            dark_swap_order = [2, 3],
            second_rel_phase = 180,
            map_to_qubit_ge = True,
            prepulse = True, #for debugging. Should always be true
            postpulse = True #for debugging. Should always be true
            )
        scramble_expts.append(scramble)
    total_exps.append(scramble_expts)

# %%
import numpy as np
import matplotlib.pyplot as plt
from experiments.MM_base import MMAveragerProgram

idx_to_plot = 0
subidx = 0
I_threshold = 0.0 

idata_g = ss.data['Ig']
qdata_g = ss.data['Qg']

avg = np.round(np.average(idata_g), 2)
avg_list_prerelax = []
frac_list_prerelax = []
frac_g = np.mean(idata_g > I_threshold) * 100 

for scramble_expts in total_exps:
    read_num = 1
    if scramble_expts[idx_to_plot].cfg.expt.get('parity_check', False):
        read_num += 1
    if scramble_expts[idx_to_plot].cfg.expt.get('active_reset', False):
        params = MMAveragerProgram.get_active_reset_params(scramble_expts[idx_to_plot].cfg)
        read_num += MMAveragerProgram.active_reset_read_num(**params)
    _start_idx = read_num-1

    idata_scramble = scramble_expts[idx_to_plot].data['idata'][subidx][_start_idx::read_num]
    qdata_scramble = scramble_expts[idx_to_plot].data['qdata'][subidx][_start_idx::read_num]

    frac_scramble = np.mean(idata_scramble > I_threshold) * 100
    each_avg = np.round(np.average(idata_scramble), 2)
    avg_list_prerelax.append(each_avg)
    frac_list_prerelax.append(frac_scramble)
fig, ax = plt.subplots()
ax.scatter(pre_relax_delay_list, avg_list_prerelax, label = "active reset ON")
# ax.scatter(relax_delay_list, avg_list, label = "active reset OFF")
# ax.axhline(y=avg, color='red', linestyle='--', alpha=0.7, label=f'Mean avg I of g from single shot')
ax.legend()
plt.show()

# %%
dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles = [0]

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros


rel_phase_list = [0]
total_exps = []
relax_delay_list = np.arange(2001, 8001, 1000)
for relax_delay in relax_delay_list:
    scramble_expts = []
    for meas_stor in meas_stors:
        scramble = dmscramble_runner.execute(
            reps=5000,
            init_fock=True,
            init_stor=0,
            ro_stor = meas_stor,
            relax_delay=relax_delay,
            active_reset=False,
            pre_relax_delay = 0,
            man_reset=True, 
            storage_reset = [2, 3], 
            reset_dump_mode = 1,
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,  
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = False,
            swap_man_dark = True,
            dark_swap_order = [2, 3],
            second_rel_phase = 180,
            map_to_qubit_ge = True,
            prepulse = True, #for debugging. Should always be true
            postpulse = True #for debugging. Should always be true
            )
        scramble_expts.append(scramble)
    total_exps.append(scramble_expts)

# %%
import numpy as np
import matplotlib.pyplot as plt
from experiments.MM_base import MMAveragerProgram

idx_to_plot = 0
subidx = 0
I_threshold = 0.0 

idata_g = ss.data['Ig']
qdata_g = ss.data['Qg']

avg = np.round(np.average(idata_g), 2)
avg_list = []
frac_list = []
frac_g = np.mean(idata_g > I_threshold) * 100 

for scramble_expts in total_exps:
    read_num = 1
    if scramble_expts[idx_to_plot].cfg.expt.get('parity_check', False):
        read_num += 1
    if scramble_expts[idx_to_plot].cfg.expt.get('active_reset', False):
        params = MMAveragerProgram.get_active_reset_params(scramble_expts[idx_to_plot].cfg)
        read_num += MMAveragerProgram.active_reset_read_num(**params)
    _start_idx = read_num-1

    idata_scramble = scramble_expts[idx_to_plot].data['idata'][subidx][_start_idx::read_num]
    qdata_scramble = scramble_expts[idx_to_plot].data['qdata'][subidx][_start_idx::read_num]

    frac_scramble = np.mean(idata_scramble > I_threshold) * 100
    each_avg = np.round(np.average(idata_scramble), 2)
    avg_list.append(each_avg)
    frac_list.append(frac_scramble)
fig, ax = plt.subplots()
ax.scatter(relax_delay_list, avg_list)
# ax.axhline(y=avg, color='red', linestyle='--', alpha=0.7, label=f'Mean g from single shot')
ax.legend()
ax.set_xlabel("relax delay (us)")
ax.set_ylabel("Avg I [ADC unit]")
plt.show()

# %% [markdown]
# ## Dark Mode T1

# %%
dm_T1_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end
    init_fock=True,
    normalize=False,
    post_select_pre_pulse=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    prepulse=True,
    postpulse=True,
)) # Shouldn't be modifying this on the fly!
dm_T1_defaults.update(active_reset_default_dict)
# You can use kwargs in the run function to override these values

def dm_T1_preproc(station, default_expt_cfg, **kwargs):
    assert 'swept_params' in kwargs
    assert len(kwargs['swept_params']) > 0

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    assert 'init_stor' in kwargs
    if not expt_cfg.init_fock:
        assert 'init_alpha' in kwargs
        
    # print(expt_cfg)
    return expt_cfg

dmt1runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkT1Experiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.DarkT1Program,
    default_expt_cfg=dm_T1_defaults,
    preprocessor=dm_T1_preproc,
    postprocessor=None,
    job_client=client,
)

# %%
np.arange(0, 10, 1)

# %%

# meas_stors = [0]
swap_stors = [1, 6]
detunings = [0, 0] # None/False/unspecified all default to all zeros

scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:
wait_lengths = np.arange(0, 3000, 100)

dmt1expt = dmt1runner.execute(
    reps=500,
    init_fock=True,
    init_stor=0,
    ro_stor = 0,
    relax_delay=200,
    active_reset=True,
    pre_relax_delay = 200,
    man_reset=True, 
    storage_reset = swap_stors, 
    reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
    swap_stors=swap_stors,
    update_phases=True,
    wait_lengths=wait_lengths,  
    swept_params=['wait_length'],
    custom_prepulse = False,
    custom_postpulse = False,
    debug = False,
    swap_man_dark = True,
    dark_swap_order = swap_stors,
    second_rel_phase = 0,
    map_to_qubit_ge = True,
    prepulse = True, #for debugging. Should always be true
    postpulse = True, #for debugging. Should always be true
    track_dark_wait_phase = True,
    dark_wait_phase_rate_MHz = -0.09581519209787519 - 0.005269872674496976, #-,

    # If the oscillation gets worse or shifts the wrong way, flip the sign.
    # dark_wait_phase_rate_MHz = -0.0175
    dark_wait_phase_offset_deg = 0.0
    )
dmt1expt.display()

# %%
from lmfit import Model

def exp_decay(x, f, phi, A, T1, C):
    return A * np.exp(-x / T1) * np.sin(2*np.pi*f*x+phi) + C

fit = Model(exp_decay)

params = fit.make_params(f=0.015, phi=0, A=70, T1=100, C=-70)

result = fit.fit(dmt1expt.data['avgi'], params, x=dmt1expt.data['xpts'])

plt.plot(dmt1expt.data['xpts'], dmt1expt.data['avgi'], 'bo', label='Experimental Data', markersize=4)
plt.plot(dmt1expt.data['xpts'], result.best_fit, 'r-', label='Best Fit')       # 최적화된 결과 그래프
plt.legend()
plt.show()

result.values['f']

# %% [markdown]
# ## Rel Ramsey

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 150, 150, 2)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [2, 3]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=100,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '3',
            init_stor= 2,
            ro_stor = meas_stor,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=False,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = True,
            swap_man_dark = True,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            dark_virtual_ramsey = True,
            dark_virtual_ramsey_phase_per_cycle_deg = 20,
            
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            fg_area_comp = "length",
            
            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def classify_two_parity_readouts(expt, point_idx=0, threshold=None, e_is_high_I=True):
    rn = expt.cfg.read_num
    qTest = expt.cfg.expt.qubits[0]

    if threshold is None:
        threshold = expt.cfg.device.readout.threshold[qTest]

    idata = np.asarray(expt.data['idata'][point_idx])
    qdata = np.asarray(expt.data['qdata'][point_idx])

    i_first  = idata[rn-2::rn]
    q_first  = qdata[rn-2::rn]
    i_second = idata[rn-1::rn]
    q_second = qdata[rn-1::rn]

    if e_is_high_I:
        first_e = i_first > threshold
        second_e = i_second > threshold
    else:
        first_e = i_first < threshold
        second_e = i_second < threshold

    b0 = first_e.astype(int)
    b1 = second_e.astype(int)

    # cond_sec_phase = -90 convention:
    # (g,g)->0, (e,g)->1, (g,e)->2, (e,e)->3
    n_mod4 = b0 + 2*b1

    out = {
        'i_first': i_first,
        'q_first': q_first,
        'i_second': i_second,
        'q_second': q_second,

        'first_e': first_e,
        'second_e': second_e,

        # parity expectation values:
        # +1 means bit=0, -1 means bit=1.
        # first parity = (-1)^n
        # second parity = +1 for n=0,1 mod 4 and -1 for n=2,3 mod 4.
        'parity_first': 1 - 2*b0,
        'parity_second': 1 - 2*b1,

        'n_mod4': n_mod4,

        'p_first_e': np.mean(first_e),
        'p_second_e': np.mean(second_e),

        'p_gg': np.mean((~first_e) & (~second_e)),
        'p_eg': np.mean(( first_e) & (~second_e)),
        'p_ge': np.mean((~first_e) & ( second_e)),
        'p_ee': np.mean(( first_e) & ( second_e)),
    }

    out['p_mod0'] = np.mean(n_mod4 == 0)
    out['p_mod1'] = np.mean(n_mod4 == 1)
    out['p_mod2'] = np.mean(n_mod4 == 2)
    out['p_mod3'] = np.mean(n_mod4 == 3)

    out['mean_parity_first'] = np.mean(out['parity_first'])
    out['mean_parity_second'] = np.mean(out['parity_second'])
    out['mean_n_mod4'] = np.mean(n_mod4)

    return out

# %%
combined_bits = defaultdict(lambda: {
    'xpts': [],

    'mean_parity_first': [],
    'mean_parity_second': [],

    'p_first_e': [],
    'p_second_e': [],

    'p_mod0': [],
    'p_mod1': [],
    'p_mod2': [],
    'p_mod3': [],

    'p_gg': [],
    'p_eg': [],
    'p_ge': [],
    'p_ee': [],

    'mean_n_mod4': [],
})

floquet_cycle2us = 0
for i in swap_stors:
    if station.ds_floquet.get_waveform(f"M1-S{i}") == 'gauss':
        floquet_cycle2us += float(station.ds_floquet.get_gauss_sigma(f"M1-S{i}")
                                  * station.ds_floquet.get_gauss_n_sigma(f"M1-S{i}"))
        floquet_cycle2us += station.soccfg.cycles2us(10)
    else:
        floquet_cycle2us += float(station.ds_floquet.get_len(f"M1-S{i}"))
        floquet_cycle2us += float(station.ds_floquet.get_ramp_sigma(f"M1-S{i}")) * 4
        floquet_cycle2us += station.soccfg.cycles2us(10)

unique_expts = []
for sub_list in scramble_expts:
    for expt in sub_list:
        if expt not in unique_expts:
            unique_expts.append(expt)

for expt in unique_expts:
    state_idx = expt.cfg.expt.ro_stor
    xpts = np.asarray(expt.data['xpts']).reshape(-1)

    for j, x in enumerate(xpts):
        r = classify_two_parity_readouts(expt, point_idx=j)

        combined_bits[state_idx]['xpts'].append(x)
        combined_bits[state_idx]['mean_parity_first'].append(r['mean_parity_first'])
        combined_bits[state_idx]['mean_parity_second'].append(r['mean_parity_second'])

        combined_bits[state_idx]['p_first_e'].append(r['p_first_e'])
        combined_bits[state_idx]['p_second_e'].append(r['p_second_e'])

        combined_bits[state_idx]['p_mod0'].append(r['p_mod0'])
        combined_bits[state_idx]['p_mod1'].append(r['p_mod1'])
        combined_bits[state_idx]['p_mod2'].append(r['p_mod2'])
        combined_bits[state_idx]['p_mod3'].append(r['p_mod3'])

        combined_bits[state_idx]['p_gg'].append(r['p_gg'])
        combined_bits[state_idx]['p_eg'].append(r['p_eg'])
        combined_bits[state_idx]['p_ge'].append(r['p_ge'])
        combined_bits[state_idx]['p_ee'].append(r['p_ee'])

        combined_bits[state_idx]['mean_n_mod4'].append(r['mean_n_mod4'])

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
only_line = True
for state_idx, data in combined_bits.items():
    sort_indices = np.argsort(data['xpts'])
    x = np.array(data['xpts'])[sort_indices] * floquet_cycle2us

    p1 = np.array(data['mean_parity_first'])[sort_indices]
    p2 = np.array(data['mean_parity_second'])[sort_indices]

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    label = f"{state_prefix}{display_idx}"
    if not only_line:
        ax[0].plot(x, p1, "o-", label=label)
        ax[1].plot(x, p2, "o-", label=label)
    else:
        ax[0].plot(x, p1,  label=label)
        ax[1].plot(x, p2,  label=label)

ax[0].set_title("First parity expectation")
ax[1].set_title("Second parity-bit expectation")

for a in ax:
    a.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    a.set_xlabel("time (us)")
    a.set_ylabel("parity value")
    a.set_ylim(-1.05, 1.05)
    a.legend()

plt.tight_layout()
plt.show()

# %%

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
max_list = []
only_line = True

for state_idx, data in combined_bits.items():
    sort_indices = np.argsort(data['xpts'])
    x = np.array(data['xpts'])[sort_indices] * floquet_cycle2us

    p0 = np.array(data['p_mod0'])[sort_indices]
    p1 = np.array(data['p_mod1'])[sort_indices]
    p2 = np.array(data['p_mod2'])[sort_indices]
    p3 = np.array(data['p_mod3'])[sort_indices]

    nmod4_mean = p1 + 2*p2 + 3*p3

    display_idx = state_idx
    state_prefix = "S"
    if display_idx == 0:
        state_prefix = "M"
        display_idx = 1

    label = f"{state_prefix}{display_idx}"
    if not only_line:
        ax.plot(x, nmod4_mean, "o-", label=label)
    else:
        ax.plot(x, nmod4_mean, label=label)
    max_list.append(nmod4_mean)
ax.set_xlabel("time (us)")
ax.set_ylabel("$\\langle n\\ mod 4\\rangle$")
# ax.set_ylim(np.max(max_list)*0.8, np.max(max_list)*1)
# ax.set_ylim(-0.05, np.max(max_list)*1.2)
ax.legend()
plt.tight_layout()
plt.show()
plt.suptitle()

# %%
station.ds_floquet.df

# %% [markdown]
# ## Rel Phase

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 150, 150, 2)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [2, 3]
scramble_expts = []

rel_phase_list = np.linspace(170, 190, 20)
for phase in tqdm(rel_phase_list):
    scramble_sub_expts = []
    for floquet_cycles in tqdm(floquet_cycles_list):
        scramble = dmscramble_runner.execute(
            reps=100,
            init_fock=False,
            # init_alpha = np.sqrt(3),
            init_man_fock_state = '3',
            init_stor= 2,
            ro_stor = 0,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay = 100, 
            man_reset=True, 
            storage_reset = swap_stors, 
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=False,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse = False,
            custom_postpulse = False,
            debug = True,
            swap_man_dark = True,
            dark_swap_order = dark_swaps,
            second_rel_phase = 180,
            dark_virtual_ramsey = True,
            dark_virtual_ramsey_phase_per_cycle_deg = 20,
            
            map_to_qubit_ge = True, 
            
            multiparity_readout = True,
            cond_sec_phase= -90,

            fg_area_comp = "length",
            
            prepulse = True, #for debugging. Should always be true
            postpulse = True, #for debugging. Should always be true
            )
        scramble_sub_expts.append(scramble)
    scramble_expts.append(scramble_sub_expts)
    # scramble.display()

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 150, 150, 2)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [2, 3]

rel_phase_list = np.linspace(160, 190, 10)
test_expts = []
for floquet_cycles in tqdm(floquet_cycles_list):
    scramble = dmscramble_runner.execute(
        reps=100,
        init_fock=False,
        # init_alpha = np.sqrt(3),
        init_man_fock_state = '3',
        init_stor= 2,
        ro_stor = 0,
        relax_delay=200,
        active_reset=True,
        pre_relax_delay = 100, 
        man_reset=True, 
        storage_reset = swap_stors, 
        reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
        dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        swap_stors=swap_stors,
        update_phases=False,
        detunings=detunings,
        floquet_cycles=floquet_cycles,
        swept_params=['floquet_cycle', 'second_rel_phase'],
        custom_prepulse = False,
        custom_postpulse = False,
        debug = True,
        swap_man_dark = True,
        dark_swap_order = dark_swaps,
        second_rel_phases = rel_phase_list,
        dark_virtual_ramsey = False,
        dark_virtual_ramsey_phase_per_cycle_deg = 20,
        
        map_to_qubit_ge = True, 
        
        multiparity_readout = True,
        cond_sec_phase= -90,

        fg_area_comp = "length",
        
        prepulse = True, #for debugging. Should always be true
        postpulse = True, #for debugging. Should always be true
        )
    test_expts.append(scramble)

# %%
test_expts[0].data['avgi']

# %%
from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(0, 150, 150, 2)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 3]
detunings = [0, 0] # None/False/unspecified all default to all zeros
dark_swaps = [2, 3]

rel_phase_list = np.linspace(0, 2, 10)
test_expts = []
for floquet_cycles in tqdm(floquet_cycles_list):
    scramble = dmscramble_runner.execute(
        reps=100,
        init_fock=False,
        # init_alpha = np.sqrt(3),
        init_man_fock_state = '1',
        init_stor= 2,
        ro_stor = 0,
        relax_delay=200,
        active_reset=True,
        pre_relax_delay = 100, 
        man_reset=True, 
        storage_reset = swap_stors, 
        reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
        dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        swap_stors=swap_stors,
        update_phases=False,
        detunings=detunings,
        floquet_cycles=floquet_cycles,
        swept_params=['floquet_cycle', 'dark_virtual_ramsey_phase_per_cycle_deg'],
        custom_prepulse = False,
        custom_postpulse = False,
        debug = True,
        swap_man_dark = True,
        dark_swap_order = dark_swaps,
        second_rel_phase = 180,
        dark_virtual_ramsey = True,
        dark_virtual_ramsey_phase_per_cycle_degs = rel_phase_list,
        
        map_to_qubit_ge = True, 
        
        multiparity_readout = True,
        cond_sec_phase= -90,
        
        prepulse = True, #for debugging. Should always be true
        postpulse = True, #for debugging. Should always be true
        )
    test_expts.append(scramble)

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
# # Wigner analysis (dormant)
#
# Relocated from `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 94-103 by the stage-2 notebook decomposition. **Dormant**: kept findable and
# diffable, not maintained. Covers the Wigner HDF5 analysis and the ideal-state
# fidelity/purity comparison, including the mod-4 confusion construction and the
# single-parity confusion correction those metrics depend on.
#
# Relocation only, per the stage-2 instructions. The library side is
# `fitting/wigner.py` and `experiments/qsim/qsim_base_wigner.py`; the measurement
# side lives in `measurement_notebooks/jonginn/qsim_wigner*.ipynb` and
# `measurement_notebooks/guan/qsim_wigner.py`, which this pass leaves alone.
#
# Sibling dormant notebooks: `flux_excursion.py`, `dark_mode.py`,
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


def _first_scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    out = arr.reshape(-1)[0]
    return out.item() if hasattr(out, 'item') else out


def wigner_stor_label(ro_stor):
    return 'M1' if int(ro_stor) == 0 else f'S{int(ro_stor)}'


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


def _nmod4_from_two_parity_readouts(expt, point_idx=0, threshold=None, e_is_high_I=True):
    _ensure_multiparity_read_num(expt)
    rn = int(expt.cfg.read_num)
    q_test = int(expt.cfg.expt.qubits[0])
    if threshold is None:
        threshold = expt.cfg.device.readout.threshold[q_test]

    idata = np.asarray(expt.data['idata'][point_idx])
    i_first = idata[rn - 2::rn]
    i_second = idata[rn - 1::rn]

    if e_is_high_I:
        b0 = (i_first > threshold).astype(int)
        b1 = (i_second > threshold).astype(int)
    else:
        b0 = (i_first < threshold).astype(int)
        b1 = (i_second < threshold).astype(int)
    return b0 + 2 * b1


def build_mod4_confusion_from_h5_files(h5_files, prepared_states=(0, 1, 2, 3), point_idx=0):
    """Return C[row prepared n, col measured n mod 4] and integer counts."""
    counts = np.zeros((4, 4), dtype=int)
    for fallback_state, h5_file in zip(prepared_states, h5_files):
        expt = DarkBaseExperiment.from_h5file(h5_file)
        prepared = int(expt.cfg.expt.get('init_man_fock_state', fallback_state))
        if prepared not in (0, 1, 2, 3):
            prepared = int(fallback_state)

        xpts = np.asarray(expt.data.get('xpts', [0])).reshape(-1)
        point_indices = range(len(xpts)) if point_idx is None else [point_idx]
        for idx in point_indices:
            n_mod4 = _nmod4_from_two_parity_readouts(expt, point_idx=idx)
            counts[prepared] += np.bincount(n_mod4.astype(int), minlength=4)

    confusion = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )
    return confusion, counts


def single_parity_confusion_from_mod4_counts(
    mod4_counts,
    prepared_groups=((0,), (1,)),
    measured_groups=((0, 2), (1, 3)),
):
    """Convert modulo-4 counts to [P00, P01, P10, P11] for WignerAnalysis.

    The default uses prepared Fock |0>, |1> and collapses measured modulo-4
    outcomes into even/odd parity. For all modulo-4 rows, use
    prepared_groups=((0, 2), (1, 3)). For a literal top-left 2x2 block, use
    measured_groups=((0,), (1,)).
    """
    mod4_counts = np.asarray(mod4_counts, dtype=float)
    parity_counts = np.array([
        [mod4_counts[list(prepared), :][:, list(measured)].sum()
         for measured in measured_groups]
        for prepared in prepared_groups
    ])
    parity_probs = np.divide(
        parity_counts,
        parity_counts.sum(axis=1, keepdims=True),
        out=np.full(parity_counts.shape, np.nan, dtype=float),
        where=parity_counts.sum(axis=1, keepdims=True) > 0,
    )
    return parity_probs.reshape(-1).tolist(), parity_probs, parity_counts.astype(int)


def plot_mod4_confusion(matrix, counts=None, title='Modulo-4 readout confusion matrix'):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel(r'Measured $n$ mod 4')
    ax.set_ylabel('Prepared Fock state')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            label = f'{matrix[i, j]:.3f}'
            if counts is not None:
                label += f'\n({int(counts[i, j])})'
            color = 'white' if matrix[i, j] < 0.45 else 'black'
            ax.text(j, i, label, ha='center', va='center', color=color)
    fig.colorbar(im, ax=ax, label=r'$P(\mathrm{measured}\mid\mathrm{prepared})$')
    fig.tight_layout()
    return fig, ax


def apply_single_parity_confusion(wigner_expt, confusion_matrix, apply_to='auto'):
    if confusion_matrix is None:
        return

    cm = [float(x) for x in confusion_matrix]
    ro = wigner_expt.cfg.device.readout
    use_active = wigner_expt.cfg.expt.get('active_reset', False)

    if apply_to == 'both' or (apply_to == 'auto' and use_active) or apply_to == 'active':
        ro.confusion_matrix_with_active_reset = cm
        ro.confusion_matrix_with_reset = cm
    if apply_to == 'both' or (apply_to == 'auto' and not use_active) or apply_to == 'without_reset':
        ro.confusion_matrix_without_reset = cm


def load_wigner_h5(h5_file, confusion_matrix=None, confusion_apply_to='auto'):
    wigner_expt = QsimWignerBaseExperiment.from_h5file(h5_file)
    wigner_expt.path = os.path.dirname(h5_file)
    wigner_expt.config_file = h5_file
    wigner_expt.prefix = os.path.splitext(os.path.basename(h5_file))[0]
    apply_single_parity_confusion(wigner_expt, confusion_matrix, apply_to=confusion_apply_to)
    return wigner_expt


def analyze_wigner_h5_files(
    h5_files,
    cutoff=10,
    initial_state=None,
    num_shots_sample=None,
    parity_post_select=None,
    confusion_matrix=None,
    confusion_apply_to='auto',
    rotate=True,
    display=True,
    clear_each=False,
    debug=True,
    analysis_label=None,
):
    if initial_state is None:
        initial_state = qt.basis(cutoff, 0).unit()

    station_arg = station if 'station' in globals() else None
    data_by_stor = defaultdict(lambda: {
        'file': [], 'floquet': [], 'purity': [], 'rho': [], 'W_fits': [], 'x_vecs': []
    })
    records = []
    outer_param = []
    purity_list = []
    rho_list = []
    W_fits = []
    x_vecs = []

    for h5_file in h5_files:
        if not Path(h5_file).exists():
            print('Missing Wigner HDF5 file:', h5_file)
            continue

        test_wigner = load_wigner_h5(
            h5_file,
            confusion_matrix=confusion_matrix,
            confusion_apply_to=confusion_apply_to,
        )
        if parity_post_select is not None:
            test_wigner.cfg.expt.parity_post_select = parity_post_select

        if num_shots_sample is None:
            test_wigner.analyze_wigner(mode_state_num=cutoff, debug=debug)
        else:
            test_wigner.analyze_wigner_temp(
                mode_state_num=cutoff,
                num_shots_sample=num_shots_sample,
                debug=debug,
            )

        if display:
            test_wigner.display(
                rotate=rotate,
                initial_state=initial_state,
                mode_state_num=cutoff,
                station=station_arg,
                save_fig=False,
            )
            fig = plt.gcf()
            if analysis_label:
                current_title = fig._suptitle.get_text() if fig._suptitle is not None else ''
                fig.suptitle(f'{analysis_label}\n{current_title}', fontsize=16)
            if clear_each:
                plt.close(fig)

        wigner_outputs = test_wigner.data['wigner_outputs']
        rho = np.asarray(wigner_outputs['rho'][0][0])
        W_fit = wigner_outputs['W_fit'][0][0]
        x_vec = wigner_outputs['alpha_wigner'][0][0]
        purity = np.real(np.trace(rho @ rho))
        ro_stor = int(test_wigner.cfg.expt.get('ro_stor', 0))
        floquet = test_wigner.cfg.expt.get('floquet_cycle', _first_scalar(test_wigner.outer_params))

        outer_param.append(test_wigner.outer_params)
        purity_list.append(purity)
        rho_list.append(rho)
        W_fits.append(W_fit)
        x_vecs.append(x_vec)

        entry = data_by_stor[ro_stor]
        entry['file'].append(h5_file)
        entry['floquet'].append(floquet)
        entry['purity'].append(purity)
        entry['rho'].append(rho)
        entry['W_fits'].append(W_fit)
        entry['x_vecs'].append(x_vec)

        records.append({
            'file': h5_file,
            'expt': test_wigner,
            'ro_stor': ro_stor,
            'floquet': floquet,
            'label': analysis_label,
            'purity': purity,
            'rho': rho,
            'W_fit': W_fit,
            'x_vec': x_vec,
        })

    return {
        'records': records,
        'data': dict(data_by_stor),
        'outer_param': outer_param,
        'purity_list': purity_list,
        'rho_list': rho_list,
        'W_fits': W_fits,
        'x_vecs': x_vecs,
    }


def _as_float_scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    return float(np.real(arr.reshape(-1)[0]))


def _target_density_matrix(target_state):
    target_state = target_state.unit()
    return target_state if target_state.isoper else qt.ket2dm(target_state)


def wigner_metric_summary(records, target_state=None, label='corrected'):
    """Return per-record rho purity and ideal-state fidelity/overlap.

    Purity is Tr(rho^2). It does not use the ideal state. The ideal-state
    comparison is reported separately as fidelity_to_ideal, matching the
    fidelity shown in the Wigner display figure.
    """
    target_dm = _target_density_matrix(target_state) if target_state is not None else None
    target_arr = target_dm.full() if target_dm is not None else None

    rows = []
    for rec in records:
        wo = rec['expt'].data['wigner_outputs']
        rho = np.asarray(wo.get('rho_rotated', wo['rho'])[0][0])
        rho_unrotated = np.asarray(wo['rho'][0][0])
        purity = float(np.real(np.trace(rho @ rho)))
        purity_unrotated = float(np.real(np.trace(rho_unrotated @ rho_unrotated)))
        fidelity = _as_float_scalar(wo['fidelity'][0][0]) if 'fidelity' in wo else np.nan
        target_overlap = float(np.real(np.trace(rho @ target_arr))) if target_arr is not None else np.nan

        rows.append({
            'label': label,
            'file': rec['file'],
            'ro_stor': rec['ro_stor'],
            'mode': wigner_stor_label(rec['ro_stor']),
            'floquet': rec['floquet'],
            'purity': purity,
            'purity_unrotated': purity_unrotated,
            'fidelity_to_ideal': fidelity,
            'target_overlap': target_overlap,
        })
    return rows


def build_wigner_metrics(raw_records, corrected_records, target_state):
    rows = []
    rows.extend(wigner_metric_summary(raw_records, target_state=target_state, label='raw'))
    rows.extend(wigner_metric_summary(corrected_records, target_state=target_state, label='corrected'))
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows


def _metric_column(metrics, key):
    if hasattr(metrics, 'columns'):
        return np.asarray(metrics[key])
    return np.asarray([row[key] for row in metrics])


def _metric_rows(metrics):
    if hasattr(metrics, 'to_dict'):
        return metrics.to_dict('records')
    return metrics


def compare_wigner_correction_h5_files(
    h5_files,
    corrected_confusion_matrix,
    cutoff=10,
    initial_state=None,
    num_shots_sample=None,
    parity_post_select=False,
    rotate=True,
    display=True,
    debug=True,
):
    """Run Wigner tomography twice: identity readout matrix, then corrected matrix.

    The "No readout correction" pass is implemented by injecting the identity
    assignment matrix into the same WignerAnalysis path used by the corrected
    pass. That leaves the parity mapping unchanged and removes only the readout
    confusion inversion.
    """
    raw = analyze_wigner_h5_files(
        h5_files,
        cutoff=cutoff,
        initial_state=initial_state,
        num_shots_sample=num_shots_sample,
        parity_post_select=parity_post_select,
        confusion_matrix=IDENTITY_PARITY_CONFUSION,
        confusion_apply_to='both',
        rotate=rotate,
        display=display,
        clear_each=False,
        debug=debug,
        analysis_label='No readout correction',
    )

    corrected = analyze_wigner_h5_files(
        h5_files,
        cutoff=cutoff,
        initial_state=initial_state,
        num_shots_sample=num_shots_sample,
        parity_post_select=parity_post_select,
        confusion_matrix=corrected_confusion_matrix,
        confusion_apply_to='both',
        rotate=rotate,
        display=display,
        clear_each=False,
        debug=debug,
        analysis_label='Corrected with multiparity confusion matrix',
    )

    return {'raw': raw, 'corrected': corrected}


# The one module-level constant from the tail of cell 4 that the
# Wigner helpers read.
IDENTITY_PARITY_CONFUSION = [1.0, 0.0, 0.0, 1.0]

# %% [markdown]
# # Wigner Analysis

# %%
# Run this setup cell once before the Wigner analysis cells below.
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import experiments as meas
from experiments.qsim.qsim_base_wigner import QsimWignerBaseExperiment
from experiments.qsim.floquet_dark_mode_readout import DarkBaseExperiment

# %%
# Attached HDF5 paths.
# Multiparity calibration order is prepared Fock n mod 4 = 0, 1, 2, 3.
multiparity_h5_files = [
    r"C:\experiments\260526_qsim_darkmode\data\JOB-20260625-00280_DarkBaseExperiment.h5",
    r"C:\experiments\260526_qsim_darkmode\data\JOB-20260625-00281_DarkBaseExperiment.h5",
    r"C:\experiments\260526_qsim_darkmode\data\JOB-20260625-00282_DarkBaseExperiment.h5",
    r"C:\experiments\260526_qsim_darkmode\data\JOB-20260625-00283_DarkBaseExperiment.h5",
]

wigner_h5_files = [
    r"C:\experiments\260625DarkModeErrBudgetting\data\JOB-20260625-00305_QsimWignerBaseExperiment.h5",
]

cutoff = 10
target_fock = 3
ideal_state = qt.basis(cutoff, target_fock).unit()
num_shots_sample = None  # set to e.g. 500 to redraw using only the first 500 shots
use_multiparity_confusion = True
# 'top_left_subblock' uses prepared/measured n mod 4 = 0,1 only.
# 'collapse_even_odd' uses prepared 0/1 and maps measured {0,2}->{even}, {1,3}->{odd}.
single_parity_confusion_mode = 'top_left_subblock'

mod4_confusion = None
mod4_counts = None
single_parity_confusion = None
single_parity_probs = None
single_parity_counts = None

missing_cal = [p for p in multiparity_h5_files if not Path(p).exists()]
if use_multiparity_confusion and not missing_cal:
    mod4_confusion, mod4_counts = build_mod4_confusion_from_h5_files(multiparity_h5_files)
    if single_parity_confusion_mode == 'top_left_subblock':
        prepared_groups = ((0,), (1,))
        measured_groups = ((0,), (1,))
    elif single_parity_confusion_mode == 'collapse_even_odd':
        prepared_groups = ((0,), (1,))
        measured_groups = ((0, 2), (1, 3))
    else:
        raise ValueError(f'Unknown single_parity_confusion_mode: {single_parity_confusion_mode}')

    single_parity_confusion, single_parity_probs, single_parity_counts = single_parity_confusion_from_mod4_counts(
        mod4_counts,
        prepared_groups=prepared_groups,
        measured_groups=measured_groups,
    )
    print('Single-parity confusion [P00, P01, P10, P11]:', np.round(single_parity_confusion, 4))
    print('Single-parity counts:\n', single_parity_counts)
    plot_mod4_confusion(mod4_confusion, mod4_counts, title=f'Modulo-4 readout confusion matrix {multiparity_h5_files}')
elif use_multiparity_confusion:
    print('Skipping multiparity confusion load; missing calibration files:')
    for path in missing_cal:
        print(' ', path)
else:
    print('Using Wigner file config readout confusion matrix.')

# %%
if wigner_h5_files and Path(wigner_h5_files[0]).exists():
    test_wigner = load_wigner_h5(wigner_h5_files[0], confusion_matrix=single_parity_confusion)
    print('ro_stor:', test_wigner.cfg.expt.get('ro_stor', None))
    print('floquet_cycle:', test_wigner.cfg.expt.get('floquet_cycle', None))
    print('reps:', test_wigner.cfg.expt.get('reps', None))
    print('displacement_path:', test_wigner.cfg.expt.get('displacement_path', None))
else:
    print('Check wigner_h5_files; first file is missing:', wigner_h5_files[0] if wigner_h5_files else None)

# %%
wigner_comparison = compare_wigner_correction_h5_files(
    wigner_h5_files,
    corrected_confusion_matrix=single_parity_confusion,
    cutoff=cutoff,
    initial_state=ideal_state,
    num_shots_sample=num_shots_sample,
    parity_post_select=False,
    rotate=True,
    display=True,
)

wigner_results_raw = wigner_comparison['raw']
wigner_results = wigner_comparison['corrected']

wigner_records_raw = wigner_results_raw['records']
wigner_records = wigner_results['records']
data_raw = wigner_results_raw['data']
data = wigner_results['data']
outer_param = wigner_results['outer_param']
purity_list = wigner_results['purity_list']
rho_list = wigner_results['rho_list']
W_fits = wigner_results['W_fits']
x_vecs = wigner_results['x_vecs']
test_wigner = wigner_records[-1]['expt'] if wigner_records else None

print('Loaded Wigner HDF5 files:', len(wigner_records))
print('Raw fidelities:', [rec['expt'].data['wigner_outputs']['fidelity'][0][0] for rec in wigner_records_raw])
print('Corrected fidelities:', [rec['expt'].data['wigner_outputs']['fidelity'][0][0] for rec in wigner_records])

# %% [markdown]
# ## Ideal-State Fidelity and Purity

# %%
wigner_metrics = build_wigner_metrics(
    raw_records=wigner_records_raw,
    corrected_records=wigner_records,
    target_state=ideal_state,
)

wigner_metrics

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
rows = _metric_rows(wigner_metrics)
labels = sorted(set(row['label'] for row in rows))
markers = {'raw': 'o', 'corrected': 's'}

for label in labels:
    label_rows = [row for row in rows if row['label'] == label]
    for ro_stor in sorted(set(row['ro_stor'] for row in label_rows)):
        group = [row for row in label_rows if row['ro_stor'] == ro_stor]
        group = sorted(group, key=lambda row: float(row['floquet']))
        x = [row['floquet'] for row in group]
        axes[0].plot(x, [row['purity'] for row in group], marker=markers.get(label, 'o'), label=f'{wigner_stor_label(ro_stor)} {label}')
        axes[1].plot(x, [row['fidelity_to_ideal'] for row in group], marker=markers.get(label, 'o'), label=f'{wigner_stor_label(ro_stor)} {label}')

axes[0].set_ylabel(r'Purity $Tr(\rho^2)$')
axes[1].set_ylabel('Fidelity to ideal state')
for ax in axes:
    ax.set_xlabel('Floquet cycle')
    ax.set_ylim(0, 1.05)
    ax.legend()
fig.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(6, 4))
for label, results_data, marker in [
    ('raw', data_raw, 'o'),
    ('corrected', data, 's'),
]:
    for ro_stor, entry in sorted(results_data.items()):
        if not entry['floquet']:
            continue
        order = np.argsort(np.asarray(entry['floquet'], dtype=float))
        floquet = np.asarray(entry['floquet'], dtype=float)[order]
        purity = np.asarray(entry['purity'], dtype=float)[order]
        ax.plot(floquet, purity, marker=marker, label=f'{wigner_stor_label(ro_stor)} {label}')

ax.set_xlabel('Floquet cycle')
ax.set_ylabel('Purity')
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()

# %%
plot_records = [('raw', rec) for rec in wigner_records_raw] + [('corrected', rec) for rec in wigner_records]
_n = len(plot_records)
if _n == 0:
    print('No Wigner records to plot.')
else:
    num_cols = min(4, _n)
    num_rows = int(np.ceil(_n / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5), squeeze=False)
    vmin = -2 / np.pi
    vmax = 2 / np.pi
    last = None

    for i, (label, rec) in enumerate(plot_records):
        r = i // num_cols
        c = i % num_cols
        ax_i = axes[r, c]
        ax_i.set_aspect('equal')
        last = ax_i.pcolormesh(rec['x_vec'], rec['x_vec'], rec['W_fit'], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        if c == 0:
            ax_i.set_ylabel(r'Im($\alpha$)', fontsize=10)
        if r == num_rows - 1:
            ax_i.set_xlabel(r'Re($\alpha$)', fontsize=10)
        ax_i.set_title(f"{label}: {wigner_stor_label(rec['ro_stor'])}, floquet={rec['floquet']}", fontsize=9)
        ax_i.tick_params(axis='both', which='major', labelsize=8)

    for i in range(_n, num_rows * num_cols):
        axes[i // num_cols, i % num_cols].set_visible(False)

    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.tight_layout()
    plt.show()

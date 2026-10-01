"""
Process tomography of the transduction channel (Rung 1 and Rung 2).

A *set* is the four outputs for the inputs {'0', '1', '+', '+i'} at one eta,
each measured by Wigner tomography. This module

- submits a set (all inputs queued together, so they see the same device state),
- loads a set from the HDF5 files of the jobs (no jobs.db needed),
- reconstructs the outputs, applies ONE phase-correction rule for data and theory,
  and computes Fe, Ic and the linearity check, with a joint bootstrap,
- saves the result of a set as a new HDF5 file with its provenance.

Rung 1: no post-selection, decoder K0 + K1.  Rung 2: even-parity post-selected
outputs (``measure_parity=True``), decoder K0 only; Ic is then conditional.
"""

import datetime
import glob
import json
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import qutip as qt
from scipy.optimize import minimize_scalar

from experiments.assembled_data import code_version, new_stem
from experiments.transduction.channel_model import (
    apply_logical_z, coherent_information, entanglement_fidelity, even_survival,
    ideal_channel_output, logical_ket, process_operators,
)
from experiments.transduction.estimators import estimate_ic
from experiments.transduction.sequences import PROC_INPUTS, channel_prep
from fitting.wigner import WignerAnalysis, parity_from_counts

DIM = 4                       # Fock {0, 1, 2, 3}: the channel output space
DATA_ROOTS = (r'C:\experiments',)
_BRANCH = {1: 'full', 2: 'even'}


# ---------------------------------------------------------------------------
# Sets: measure / load
# ---------------------------------------------------------------------------

@dataclass
class ChannelSet:
    """One process-tomography set. ``jobs`` maps input label -> job ID."""
    eta: float
    env_stor: int
    rung: int = 1
    jobs: dict = field(default_factory=dict)
    expts: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def load(self, roots=DATA_ROOTS):
        """Load the missing experiments from the HDF5 files of ``jobs``."""
        for L, job in self.jobs.items():
            if L not in self.expts:
                self.expts[L] = load_wigner(job, roots)
        return self

    @property
    def complete(self):
        return all(L in self.expts or L in self.jobs for L in PROC_INPUTS)


def find_wigner_h5(job_id, roots=DATA_ROOTS):
    """Path of the Wigner HDF5 file of ``job_id`` under any experiment folder."""
    hits = []
    for root in roots:
        hits += glob.glob(os.path.join(root, '*', 'data', f'{job_id}_WignerTomography1ModeExperiment.h5'))
    if len(hits) != 1:
        raise FileNotFoundError(f'{job_id}: expected one Wigner file, found {hits}')
    return hits[0]


def load_wigner(job_or_path, roots=DATA_ROOTS):
    """WignerTomography1ModeExperiment from a job ID or an HDF5 path, analyzed."""
    from experiments.qubit_cavity.single_mode_wigner_tomography import WignerTomography1ModeExperiment
    path = job_or_path if os.path.exists(str(job_or_path)) else find_wigner_h5(job_or_path, roots)
    expt = WignerTomography1ModeExperiment.from_h5file(path)
    expt.man_mode_idx = expt.cfg.expt.get('man_mode_no', 1) - 1   # analyze() needs it
    expt.analyze()
    return expt


def measure(runner, preps, reps, **run_kwargs):
    """Queue one Wigner job per compiled preparation in ``preps`` (label -> pulse
    list) in one batch. Returns {label: (job_id or None, expt)}."""
    labels = list(preps)
    expts = runner.execute(overrides=[dict(pre_sweep_pulse=preps[L]) for L in labels],
                           batch_size=len(labels), reps=reps, prepulse=True, **run_kwargs)
    ids = list(runner.last_job_ids) or [None] * len(labels)
    return {L: (j, e) for L, j, e in zip(labels, ids, expts)}


def measure_set(runner, mm, eta, env_stor, reps, rung=1, enc_phase_corr_deg=0.0,
                inputs=PROC_INPUTS, man_no=1, **run_kwargs):
    """Measure a set: the channel at ``eta`` for each input, queued together."""
    preps = {L: channel_prep(mm, eta, L, env_stor, enc_phase_corr_deg, man_no) for L in inputs}
    out = measure(runner, preps, reps, measure_parity=(rung == 2), **run_kwargs)
    return ChannelSet(eta=eta, env_stor=env_stor, rung=rung,
                      jobs={L: j for L, (j, _) in out.items() if j is not None},
                      expts={L: e for L, (_, e) in out.items()},
                      meta=dict(reps=reps, enc_phase_corr_deg=enc_phase_corr_deg,
                                measured=datetime.datetime.now().isoformat(timespec='seconds')))


# ---------------------------------------------------------------------------
# Reconstruction and metrics
# ---------------------------------------------------------------------------

def wigner_analysis(expt, dim=DIM):
    return WignerAnalysis(expt.data, config=expt.cfg, mode_state_num=dim, alphas=expt.data['alpha'])


def rho_linear(wa, parity, dim=DIM):
    """Linear-inversion estimate (not projected); independent of the target."""
    res = wa.wigner_analysis_results(parity, initial_state=qt.fock(dim, 0), rotate=False)
    return qt.Qobj(np.asarray(res['rho_linear']))


def ideal_outputs(eta, dim=DIM, rung=1):
    post = 'even' if rung == 2 else None
    return {L: qt.Qobj(ideal_channel_output(logical_ket(L, dim), eta, dim, postselect=post))
            for L in PROC_INPUTS}


def phase_correct(rhos, eta, dim=DIM, rung=1):
    """Remove the hardware logical-Z phase: the measured angle of rho_+[2, 0]
    minus the angle of the ideal output (pi for eta < 2/3, which the decoder
    sign already undoes). One angle for all inputs. Ic does not change."""
    ideal = ideal_outputs(eta, dim, rung)['+'].full()[2, 0]
    phi = float(np.angle(rhos['+'].full()[2, 0] / ideal))
    return {L: apply_logical_z(r, -phi) for L, r in rhos.items()}, phi


def linearity(rhos):
    """diag(rho_+) - (diag rho_0 + diag rho_1)/2, same for '+i'. Zero for any
    linear channel; a large value means the inputs saw different device states."""
    avg = 0.5 * (np.diag(rhos['0'].full()) + np.diag(rhos['1'].full()))
    return {L: np.real(np.diag(rhos[L].full()) - avg) for L in ('+', '+i')}


def metrics(rhos, eta, dim=DIM, rung=1):
    """Fe (decoded), Fe_phys (CP-projected Choi), Ic (raw channel) of one set
    of output states {label: Qobj}. Used for data and for theory."""
    rr, phi = phase_correct(rhos, eta, dim, rung)
    N = process_operators(rr['0'], rr['1'], rr['+'], rr['+i'])
    branch = _BRANCH[rung]
    return dict(
        Fe=entanglement_fidelity(*N, eta, dim, decode=True, branch=branch, physical=False),
        Fe_phys=entanglement_fidelity(*N, eta, dim, decode=True, branch=branch, physical=True),
        Ic=coherent_information(*N, eta, dim, decode=False),
        phi_ch=float(np.degrees(phi)),
    )


def theory_metrics(eta, dim=DIM, rung=1):
    """The ideal channel through the same pipeline (plus survival for Rung 2)."""
    out = metrics(ideal_outputs(eta, dim, rung), eta, dim, rung)
    if rung == 2:
        out['survival'] = float(np.mean([even_survival(logical_ket(L, dim), eta, dim)
                                         for L in PROC_INPUTS]))
    return out


def analyze_set(cset, dim=DIM, n_boot=200, seed=0, ic_estimator='ml', n_boot_ml=100, n_thru=0):
    """Point estimate, joint bootstrap (all four inputs resampled together),
    linearity, populations and the theory at the set eta.

    Fe: linear inversion (unbiased). Ic (Rung 1, ``ic_estimator='ml'``): joint ML
    fit of the channel (``estimators.estimate_ic``), with ``n_boot_ml`` parametric
    bootstrap refits; the linear value stays as 'Ic_linear'. ``n_thru`` > 0 adds
    'theory_thru': the ideal channel through the same estimator on this set's
    grid and shots (compare the data with this, not only with the theory)."""
    cset.load()
    missing = [L for L in PROC_INPUTS if L not in cset.expts]
    if missing:
        raise ValueError(f'set at eta {cset.eta} has no output for {missing}')
    wa = {L: wigner_analysis(cset.expts[L], dim) for L in PROC_INPUTS}
    counts = {L: cset.expts[L].data.get('parity_counts') for L in PROC_INPUTS}

    def rhos_from(rng):
        if rng is None:
            return {L: rho_linear(wa[L], cset.expts[L].data['parity'], dim) for L in PROC_INPUTS}
        return {L: rho_linear(wa[L], parity_from_counts(counts[L], rng=rng), dim) for L in PROC_INPUTS}

    rhos = rhos_from(None)
    res = metrics(rhos, cset.eta, dim, cset.rung)
    res['linearity'] = {L: v.tolist() for L, v in linearity(rhos).items()}
    res['pops'] = {L: np.real(np.diag(r.full())).tolist() for L, r in rhos.items()}
    if cset.rung == 2:
        res['survival'] = float(np.mean([1.0 - (cset.expts[L].data.get('sigma_z_discard_frac', 0.0) or 0.0)
                                         for L in PROC_INPUTS]))
    res['theory'] = theory_metrics(cset.eta, dim, cset.rung)
    if n_boot and all(c is not None for c in counts.values()):
        rng = np.random.default_rng(seed)
        draws = [metrics(rhos_from(rng), cset.eta, dim, cset.rung) for _ in range(int(n_boot))]
        for k in ('Fe', 'Fe_phys', 'Ic'):
            xs = np.array([d[k] for d in draws])
            res[k + '_std'] = float(np.std(xs, ddof=1))
            res[k + '_ci'] = [float(np.percentile(xs, 16)), float(np.percentile(xs, 84))]
    res['rhos'] = rhos
    res['estimator'] = dict(Fe='linear', Ic='linear')
    # Ic: joint ML fit of the channel (Rung 1 only: it assumes a trace-preserving map)
    if ic_estimator == 'ml' and cset.rung == 1:
        for k in ('Ic', 'Ic_std', 'Ic_ci'):
            if k in res:
                res[k.replace('Ic', 'Ic_linear')] = res.pop(k)
        ml = estimate_ic(cset.expts, rhos, dim, n_boot=n_boot_ml, n_thru=n_thru, eta=cset.eta, seed=seed)
        res['estimator']['Ic'] = ml.pop('estimator')
        res['choi'] = ml.pop('choi')
        res.update(ml)
    return res


def plot_sets(results, rung=1, eta_th=np.linspace(0.02, 0.98, 97)):
    """Fe and Ic against eta for a list of ``analyze_set`` / ``load_result``
    outputs (each needs 'eta'), with the theory through the same pipeline.
    Rung 2 adds the even survival and the Rung-1 theory for comparison."""
    import matplotlib.pyplot as plt
    results = sorted(results, key=lambda r: r['eta'])
    eta = [r['eta'] for r in results]
    th = [theory_metrics(e, rung=rung) for e in eta_th]
    keys = [('Fe', 'entanglement fidelity F_e'), ('Ic', 'I_c (nats)')]
    if rung == 2:
        keys.append(('survival', 'even survival'))
    fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4))
    for ax, (k, label) in zip(axs, keys):
        err = [r.get(k + '_std', 0.0) for r in results]
        ax.errorbar(eta, [r[k] for r in results], yerr=err, fmt='o', capsize=3, label='measured')
        ax.plot(eta_th, [t[k] for t in th], 'k-', label='theory')
        thru = [(r['eta'], r['theory_thru']) for r in results if k == 'Ic' and 'theory_thru' in r]
        if thru:
            e_t, t_t = zip(*thru)
            ax.errorbar(e_t, [t['mean'] for t in t_t],
                        yerr=[[t['mean'] - t['lo'] for t in t_t], [t['hi'] - t['mean'] for t in t_t]],
                        fmt='s', color='gray', mfc='none', capsize=3, label='ideal channel through the estimator')
        if rung == 2 and k != 'survival':
            ax.plot(eta_th, [theory_metrics(e)[k] for e in eta_th], 'r--', label='theory, Rung 1')
        ax.axvline(2 / 3, color='gray', ls=':')
        ax.set_xlabel('eta')
        ax.set_ylabel(label + (' [conditional]' if rung == 2 and k == 'Ic' else ''))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Population-only fast check (radial lines, no full tomography)
# ---------------------------------------------------------------------------

FAST_RADII = np.linspace(0, 1.65, 12)
_NT_FAST = 30


def fast_alpha_list(n_angles, radii=FAST_RADII):
    """Origin + ``n_angles`` radial lines (alpha_list form [[re, im], ...])."""
    ang = np.arange(n_angles) * (2 * np.pi / n_angles)
    pts = [0j] + [r * np.exp(1j * t) for r in radii[1:] for t in ang]
    return [[float(a.real), float(a.imag)] for a in pts]


def _radial_wigner(n, r):
    par = (-1.0) ** np.arange(_NT_FAST)
    return np.array([np.sum(par * np.abs((qt.displace(_NT_FAST, -x) * qt.basis(_NT_FAST, n)).full().ravel()) ** 2)
                     for x in np.atleast_1d(r)])


def fast_pops(expt, dim=DIM):
    """Populations from the angle-averaged parity: W(r) = sum_n P_n W_n(r), sum P_n = 1."""
    a = np.asarray(expt.data['alpha'])
    p = np.asarray(expt.data['parity'])
    r = np.round(np.abs(a), 6)
    rs = np.unique(r)
    w = np.array([p[r == x].mean() for x in rs])
    A = np.column_stack([_radial_wigner(n, rs) for n in range(dim)])
    K = np.block([[A.T @ A, np.ones((dim, 1))], [np.ones((1, dim)), np.zeros((1, 1))]])
    return np.linalg.solve(K, np.concatenate([A.T @ w, [1.0]]))[:dim]


def fit_eta(pops, logical, dim=DIM):
    """eta for which the ideal output populations of ``logical`` best match ``pops``.
    Returns (eta, rms)."""
    def cost(x):
        ideal = np.real(np.diag(ideal_channel_output(logical_ket(logical, dim), x, dim).full()))
        return np.sum((ideal - pops) ** 2)
    r = minimize_scalar(cost, bounds=(0.02, 0.98), method='bounded')
    return float(r.x), float(np.sqrt(r.fun / dim))


def fast_report(pops):
    """Linearity and per-input eta fit of fast-check populations {label: pops}."""
    avg = 0.5 * (pops['0'] + pops['1'])
    out = {'linearity': {L: (pops[L] - avg).tolist() for L in ('+', '+i') if L in pops}}
    out['eta_fit'] = {L: fit_eta(p, L, len(p)) for L, p in pops.items()}
    return out


# ---------------------------------------------------------------------------
# Results file (one per set, with provenance)
# ---------------------------------------------------------------------------

_SCALARS = ('Fe', 'Fe_phys', 'Ic', 'phi_ch', 'survival', 'Fe_std', 'Fe_phys_std', 'Ic_std',
            'Ic_linear', 'Ic_linear_std', 'Ic_ml_covariant', 'Ic_ml_cptp', 'Ic_bias_est', 'Ic_bias_corrected',
            'chi2', 'n_points', 'cov_check_p')
_JSON = ('theory', 'linearity', 'estimator', 'theory_thru')


def save_result(directory, cset, res, extra=None):
    """Write one analyzed set as a new HDF5 file in ``directory`` (normally
    ``<experiment root>/assembled_data``). Never overwrites; returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, new_stem('TransductionChannelSet', directory) + '.h5')
    raw = {}
    for L, job in cset.jobs.items():
        try:
            raw[L] = find_wigner_h5(job)
        except FileNotFoundError:
            raw[L] = ''
    with h5py.File(path, 'w') as f:
        f.attrs.update(dict(eta=cset.eta, env_stor=cset.env_stor, rung=cset.rung,
                            jobs=json.dumps(cset.jobs), raw_files=json.dumps(raw),
                            meta=json.dumps(cset.meta),
                            dim=res['rhos']['0'].shape[0],
                            created=datetime.datetime.now().isoformat(timespec='seconds'),
                            code_version=code_version(),
                            extra=json.dumps(extra or {})))
        g = f.create_group('metrics')
        for k in _SCALARS:
            if k in res:
                g.attrs[k] = res[k]
        for k in ('Fe_ci', 'Fe_phys_ci', 'Ic_ci', 'Ic_linear_ci'):
            if k in res:
                g.attrs[k] = res[k]
        for k in _JSON:
            if k in res:
                g.attrs[k] = json.dumps(res[k])
        r = f.create_group('rho')
        for L, rho in res['rhos'].items():
            r.create_dataset(L, data=rho.full())
        if 'choi' in res:
            f.create_dataset('choi', data=res['choi'])
    return path


def load_result(path):
    """Read a results file back: dict with the metrics, attrs and rho per input."""
    with h5py.File(path, 'r') as f:
        out = dict(f.attrs)
        for k in ('jobs', 'raw_files', 'meta', 'extra'):
            out[k] = json.loads(out[k])
        m = dict(f['metrics'].attrs)
        for k in _JSON:
            if k in m:
                m[k] = json.loads(m[k])
        out.update(m)
        out['rhos'] = {L: qt.Qobj(f['rho'][L][()]) for L in f['rho']}
        if 'choi' in f:
            out['choi'] = f['choi'][()]
    return out

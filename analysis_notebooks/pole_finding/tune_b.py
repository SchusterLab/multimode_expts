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
# # Tuning fitter B: rank rule and pencil length
#
# B's two free parameters (docs/qsim/pole_finding.md 4.1, 4.6), chosen on synthetic data
# under the conditions of the August disorder set, then checked on the real August sets.
# August: g = 8.615 kHz, K = -10.5 kHz (K/g = -1.22), delta = 50 kHz (delta/g = 5.80),
# 300 samples at 1.4509 us, 10 rows chosen by max-min support, noise about 0.075 per
# sample (SNR about 13), decay about 0.01 per us.

# %%
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles import joint_pencil
from fitting.qsim.poles.bench_summaries import scores_table, small_gap_table
from fitting.qsim.poles.benchmarks import run_nonideal_bench
from fitting.qsim.poles.matching import alias_distance, level_tolerances, match_poles
from fitting.qsim.poles.rank import RankRule
from fitting.qsim.poles.statistics import small_gap_ratio_fraction
from fitting.qsim.poles.synthetic import Hardware, Nonideal, distinct_levels, sample_phase_diagram

pd.set_option("display.width", 220)
VARIANTS = {f"{rule} L={fraction:.2f}N": (joint_pencil.fit, joint_pencil.JointPencilSettings(
    rank_rule=RankRule(rule=rule), pencil_fraction=fraction))
    for rule in ("mdl", "threshold") for fraction in (0.5, 2 / 3, 0.8)}

# %% [markdown]
# ## Synthetic, August conditions
#
# 20 disorder draws at the August point, 2 noise seeds, SNR 13 to 300 (the real SNR is not
# known well: B's residual on the real data gives about 13, if it is all white noise).

# %%
august = Hardware(coupling_MHz=8.615e-3, dt_us=1.4509, samples=300, partial_rows=10)
points = sample_phase_diagram([-1.22], [5.80], draws=20, hardware=august, seed=100)
synthetic = run_nonideal_bench(points, [Nonideal(snr=snr, decay_per_us=0.01) for snr in (13, 30, 100, 300)],
                               seeds=2, hardware=august, fitters=VARIANTS)
table = scores_table(synthetic)
gaps = small_gap_table(synthetic)
summary = (table.groupby(["fitter", "snr"])[["levels", "rank", "matched", "resolved", "false_poles"]].mean()
           .join(gaps.set_index(["fitter", "snr"])[["found", "true"]].rename(columns=dict(found="I_found", true="I_true"))))
summary.round(2)

# %% [markdown]
# ## Real August sets
#
# Without an exact model: the fit residual, the match to the model within
# max(level tolerance, 0.3 kHz) against random poles of the same count, P(r < 0.25) against
# the model's, and on august_N3 the weights per multiplet (as in first_look.py).

# %%
MODEL_ERROR_MHz = 0.3e-3
BRANCHES = {(2, 1, 0, 0, 0): 1, (2, 0, 1, 0, 0): 1, (1, 1, 0, 1, 0): 1, (1, 1, 0, 0, 1): 1,
            (1, 0, 1, 1, 0): 1, (1, 0, 1, 0, 1): 1}
root = data_root() / "260526_qsim_darkmode/assembled_data"
spectra = {"august_N3": MBRSpectrumExperiment.from_manifest(root / "260924_163516_MBRSpectrumExperiment.yaml").analyze(
    phase_frame="manual_kerr", manual_kerr_MHz=-10.5e-3, cycle_branches=BRANCHES, spectrum_method="fft")}
ensemble = MBRDisorderEnsembleExperiment.from_manifest(root / "260924_195547_MBRDisorderEnsembleExperiment.yaml")
for record, part in zip(ensemble.realizations, ensemble.children):
    spectra[f"august_disorder/{record['realization']}"] = part.analyze(
        phase_frame="manual_kerr", manual_kerr_MHz=1e-3 * float(record["self_kerr_kHz"]),
        cycle_branches={o: BRANCHES.get(o, 0) for o in part.occupations}, spectrum_method="fft")


def model_levels(data):
    """-> (distinct levels, multiplicities, bin, dt) of a spectrum's model."""
    time_us = np.asarray(data.spectrum.time_us)
    dt_us = time_us[1] - time_us[0]
    bin_MHz = 1 / (len(time_us) * dt_us)
    model = fixed_n_hamiltonian(3, 5, data.detunings, data.hardware.couplings_MHz, data.spectrum.physical_kerr_MHz)
    rows = [model.fock_index[tuple(o)] for o in data.reconstruction.occupations]
    levels, multiplicities, _ = distinct_levels(model.energies_MHz, model.basis_eigenstate_weights[rows], 1e-3 * bin_MHz)
    return levels, multiplicities, bin_MHz, dt_us


rng = np.random.default_rng(0)
rows, found = [], {}
for label, data in spectra.items():
    A, time_us = np.asarray(data.reconstruction.A), np.asarray(data.spectrum.time_us)
    levels, multiplicities, bin_MHz, dt_us = model_levels(data)
    tolerance = np.maximum(level_tolerances(levels, bin_MHz), MODEL_ERROR_MHz)
    for name, (fit, settings) in VARIANTS.items():
        f = fit(A, time_us, settings)
        found[label, name] = f.frequencies_MHz
        match = match_poles(f.frequencies_MHz, levels, tolerance, 1 / dt_us)
        random = np.mean([len(match_poles(rng.uniform(levels.min(), levels.max(), len(f.frequencies_MHz)),
                                          levels, tolerance, 1 / dt_us).level_index) for _ in range(50)])
        row = dict(set=label.split("/")[0], fitter=name, poles=len(f.frequencies_MHz),
                   matched=len(match.level_index), random=random)
        if label == "august_N3":
            distance = alias_distance(f.frequencies_MHz, levels, 1 / dt_us)
            nearest = np.argmin(np.abs(distance), axis=1)
            near = np.abs(distance[np.arange(len(nearest)), nearest]) <= 0.5 * bin_MHz
            sums = np.bincount(nearest[near], weights=f.weights[near], minlength=len(levels))
            row.update(multiplet_error=np.mean(np.abs(sums - multiplicities)), far_poles=int(np.sum(~near)))
        rows.append(row)
real = pd.DataFrame(rows).groupby(["set", "fitter"]).mean(numeric_only=True)
labels = [label for label in spectra if label.startswith("august_disorder")]
model_I = small_gap_ratio_fraction([model_levels(spectra[l])[0] for l in labels])
real["I_found"] = pd.Series({("august_disorder", name): small_gap_ratio_fraction([found[l, name] for l in labels])
                             for name in VARIANTS})
print(f"model I(0.25) on august_disorder: {model_I:.2f}")
real.round(2)

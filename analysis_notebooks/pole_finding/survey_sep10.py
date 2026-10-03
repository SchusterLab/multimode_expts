"""Unattended survey of the pole-finding methods on the Sep 10 complete-basis ensemble.

Runs every method we have on ``sep10_full_K3p6_g29p2`` (9 realizations x 35 rows) and on
``august25_N3``, caches every fit on disk (one pickle per spectrum and fitter; the run resumes),
and writes the summary tables as CSV. Order: the cheap steps on all spectra first (form factor,
A, B, T1, T1T3, the design bounds, the T3 merge-threshold scan, the diagnosis), then C on all
spectra, then F from T1 on realization 0. Started detached (tmux window in ``nb``) on
2026-10-02; read the results from the CSVs and the log in the output folder.

    pixi run python analysis_notebooks/pole_finding/survey_sep10.py [stage,...]

Stages: ff, fast, design, merge, diagnose, C, F (default: all, in this order).
"""
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.poles import hamiltonian_fit as hf
from fitting.qsim.poles import joint_pencil, joint_refined, per_row_reconciled, pursuit, sparse_fit
from fitting.qsim.poles.design import Design, design_summary
from fitting.qsim.poles.diagnosis import cause_counts, diagnose_levels, residual_excess, row_noise
from fitting.qsim.poles.pole_fit import normalize_to_initial_return
from fitting.qsim.poles.real_benchmarks import multiplet_weights
from fitting.qsim.poles.registry import data_set, manifest_path
from fitting.qsim.poles.statistics import small_gap_ratio_fraction

FOLDER = data_root() / "260818_qsim_spectroscopy" / "derived_data" / "pole_finding" / "sep10_survey"
FOLDER.mkdir(parents=True, exist_ok=True)
LOG = FOLDER / "survey.log"
STAGES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ff", "fast", "design", "merge", "diagnose", "C", "F"]
SETS = ("sep10_full_K3p6_g29p2", "august25_N3")


def log(*parts):
    line = time.strftime("%H:%M:%S ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def file_label(spectrum):
    return spectrum.label.replace("/", "_")


def cached(spectrum, name, make):
    """-> (result, seconds) from the cache, or made now and cached."""
    path = FOLDER / f"fit_{file_label(spectrum)}_{name}.pkl"
    if path.exists():
        with open(path, "rb") as file:
            return pickle.load(file)
    t0 = time.time()
    result = make()
    seconds = time.time() - t0
    with open(path, "wb") as file:
        pickle.dump((result, seconds), file)
    return result, seconds


def summary_row(spectrum, name, fit, seconds):
    sums, far = multiplet_weights(fit, spectrum)
    excess = residual_excess(fit, spectrum)
    offsets = fit.row_offsets_MHz
    return dict(spectrum=spectrum.label, fitter=name, seconds=round(seconds, 1), poles=len(fit.frequencies_MHz),
                levels_hit=int(np.sum(sums > 0.3)), levels=len(spectrum.levels_MHz), far_poles=far,
                multiplet_error=float(np.mean(np.abs(sums - spectrum.multiplicities))),
                excess_mean=float(excess.mean()), excess_max=float(excess.max()),
                P_found=small_gap_ratio_fraction([fit.frequencies_MHz]),
                P_model=small_gap_ratio_fraction([spectrum.levels_MHz]),
                offset_rms_kHz=float(1e3 * np.std(offsets)) if offsets is not None else np.nan,
                offset_max_kHz=float(1e3 * np.abs(offsets).max()) if offsets is not None else np.nan)


def recorded_start(spectrum):
    p = spectrum.model_parameters
    return hf.ModelParameters(detunings_MHz=p["detunings_MHz"], couplings_MHz=p["couplings_MHz"], kerr_MHz=p["kerr_MHz"])


def t1(spectrum):
    return cached(spectrum, "T1", lambda: hf.fit_hamiltonian(spectrum.A, spectrum.time_us, spectrum.occupations,
                                                             recorded_start(spectrum), hf.HamiltonianFitSettings()))


def append(name, rows):
    """Add rows to ``<name>.csv``, replacing rows with the same key columns."""
    if not rows:
        return
    path = FOLDER / f"{name}.csv"
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = [k for k in ("spectrum", "fitter", "design", "merge_chi2", "cause") if k in frame.columns]
        if keys:
            new_keys = set(map(tuple, frame[keys].astype(str).to_numpy()))
            old = old[[tuple(r) not in new_keys for r in old[keys].astype(str).to_numpy()]]
        frame = pd.concat([old, frame])
    frame.to_csv(path, index=False)


def t1_model_check(spectrum, result, seconds):
    start = recorded_start(spectrum)
    shift = 1e3 * (result.parameters.vector - start.vector)
    return dict(spectrum=spectrum.label, seconds=round(seconds, 1), reduced_chi2=result.reduced_chi2,
                T2_us=1 / result.decay_per_us,
                **{f"d{k}_kHz": 1e3 * v for k, v in enumerate(result.parameters.detunings_MHz, 1)},
                **{f"g{k}_kHz": 1e3 * v for k, v in enumerate(result.parameters.couplings_MHz, 1)},
                K_kHz=1e3 * result.parameters.kerr_MHz, K_recorded_kHz=1e3 * start.kerr_MHz,
                shift_max_kHz=np.abs(shift).max(), shift_detuning_rms_kHz=np.sqrt(np.mean(shift[:4] ** 2)),
                shift_coupling_rms_kHz=np.sqrt(np.mean(shift[4:8] ** 2)), shift_K_kHz=shift[8],
                error_median_kHz=np.median(1e3 * result.errors.vector),
                offsets_rms_kHz=1e3 * np.std(result.pole_fit.row_offsets_MHz),
                offsets_max_kHz=1e3 * np.abs(result.pole_fit.row_offsets_MHz).max(),
                starts_at_best=int(np.sum(result.start_chi2 < result.chi2 + 1)))


def fit_of(result):
    return result.pole_fit if hasattr(result, "pole_fit") else result


def diagnose(suffix=""):
    """The spec 5.5 diagnosis of every spectrum with every cached fit (merge-scan fits left out)."""
    for s in spectra:
        try:
            fits = {}
            for path in sorted(FOLDER.glob(f"fit_{file_label(s)}_*.pkl")):
                name = path.stem[len(f"fit_{file_label(s)}_"):]
                if "merge" in name or name == "A_campaign":
                    continue
                with open(path, "rb") as file:
                    fits[name] = fit_of(pickle.load(file)[0])
            table, oracle = diagnose_levels(s, fits)
            table.insert(0, "spectrum", s.label)
            table.to_csv(FOLDER / f"diagnosis_{file_label(s)}{suffix}.csv", index=False)
            counts = cause_counts(table, list(fits)).reset_index().rename(columns={"index": "cause"})
            counts.insert(0, "spectrum", s.label)
            append("diagnosis_counts" + suffix, counts.to_dict("records"))
            log(s.label, "diagnosis: resolvable", int(table.resolvable.sum()), "held", int(table.held.sum()), "of", len(table),
                "| found:", " ".join(f"{n} {int(table[f'found_{n}'].sum())}" for n in fits))
        except Exception:
            log(s.label, "diagnosis FAILED\n" + traceback.format_exc())


# ---------------------------------------------------------------- load
log("survey start; stages", STAGES)
spectra = [s for label in SETS for s in load_spectra(data_set(label), data_root())]
log(len(spectra), "spectra:", [s.label for s in spectra])

# ---------------------------------------------------------------- form factor (ensemble analysis, MPM default)
if "ff" in STAGES:
    try:
        entry = data_set(SETS[0])
        ensemble = MBRDisorderEnsembleExperiment.from_manifest(manifest_path(entry, data_root()))
        data = ensemble.analyze(phase_frame="as_acquired", on_error="skip")
        t = data.form_factor.time_us
        model = np.array([np.abs(np.exp(-2j * np.pi * np.outer(t, s.levels_MHz)) @ s.multiplicities) ** 2
                          for s in spectra if s.data_set == SETS[0]])
        out = dict(time_us=t, sff=data.form_factor.sff, sff_normalized=data.form_factor.sff_normalized,
                   trace=data.form_factor.trace, sff_model=model.mean(axis=0), dimension=data.form_factor.dimension,
                   theory_mean_r=data.theory.mean, measured_mean_r=data.measured.mean if data.measured else None,
                   realizations=[dict(realization=r.realization, poles_MHz=np.asarray(r.poles_MHz),
                                      theory_levels_MHz=np.asarray(r.theory_levels_MHz))
                                 for r in data.realizations if "error" not in r],
                   ratio_failures=data.ratio_failures)
        with open(FOLDER / "form_factor.pkl", "wb") as file:
            pickle.dump(out, file)
        log("form factor done; MPM-default mean r measured", out["measured_mean_r"], "theory", out["theory_mean_r"],
            "failures", out["ratio_failures"])
    except Exception:
        log("form factor FAILED\n" + traceback.format_exc())

# ---------------------------------------------------------------- fast fitters
if "fast" in STAGES:
    for s in spectra:
        rows = []
        try:
            fit, sec = cached(s, "A", lambda: per_row_reconciled.fit(s.A, s.time_us, per_row_reconciled.ANALYSIS_SETTINGS))
            rows.append(summary_row(s, "A", fit, sec))
            fit, sec = cached(s, "A_campaign", lambda: per_row_reconciled.fit(s.A, s.time_us, per_row_reconciled.CAMPAIGN_SETTINGS))
            rows.append(summary_row(s, "A_campaign", fit, sec))
            fit, sec = cached(s, "B", lambda: joint_pencil.fit(s.A, s.time_us, joint_pencil.JointPencilSettings()))
            rows.append(summary_row(s, "B", fit, sec))
            result, sec = t1(s)
            rows.append(summary_row(s, "T1", result.pole_fit, sec))
            append("t1_model_check", [t1_model_check(s, result, sec)])
            fit, sec = cached(s, "T1T3", lambda: sparse_fit.fit(s.A, s.time_us, sparse_fit.SparseFitSettings(), start=result.pole_fit))
            rows.append(summary_row(s, "T1T3", fit, sec))
            free, sec = cached(s, "T1free", lambda: hf.fit_hamiltonian(
                s.A, s.time_us, s.occupations, recorded_start(s), hf.HamiltonianFitSettings(amplitudes="free", starts=4)))
            rows.append(summary_row(s, "T1free", free.pole_fit, sec))
        except Exception:
            log(s.label, "fast FAILED\n" + traceback.format_exc())
        append("fits_summary", rows)
        log(s.label, "fast done:", " ".join(f"{r['fitter']} {r['seconds']}s {r['levels_hit']}/{r['levels']} far {r['far_poles']}"
                                             for r in rows))

# ---------------------------------------------------------------- design bounds (T2 question; column sums only so far)
if "design" in STAGES:
    for s in spectra:
        try:
            a = normalize_to_initial_return(s.A)
            noise = row_noise(a, s.dt_us, s.levels_MHz, 4.)
            decay = 0.005
            try:
                decay = max(float(np.median(t1(s)[0].pole_fit.decays_per_us)), 1e-4)
            except Exception:
                pass
            rows = []
            occupations = np.asarray(s.occupations, dtype=float)
            for amplitudes in ("complex", "real", "real_sums"):
                for offsets in ("free", "per_photon"):
                    design = Design(amplitudes=amplitudes, offsets=offsets, decay_per_us=decay)
                    t0 = time.time()
                    summary = design_summary(s.time_us, s.levels_MHz, s.row_weights, occupations, noise, s.bin_MHz, design)
                    rows.append(dict(spectrum=s.label, design=f"{amplitudes}/{offsets}", decay_per_us=decay,
                                     noise_median=float(np.median(noise)), seconds=round(time.time() - t0, 1), **summary))
            # 10 rows chosen at random: what a partial basis of this set would resolve
            rng = np.random.default_rng(0)
            index = np.sort(rng.choice(len(s.occupations), 10, replace=False))
            kept = s.row_weights[index].sum(axis=0) >= 0.1
            for amplitudes in ("complex", "real"):
                design = Design(amplitudes=amplitudes, decay_per_us=decay)
                summary = design_summary(s.time_us, s.levels_MHz[kept], s.row_weights[index][:, kept], occupations[index],
                                         noise[index], s.bin_MHz, design)
                rows.append(dict(spectrum=s.label, design=f"{amplitudes}/free/10rows", decay_per_us=decay,
                                 noise_median=float(np.median(noise)), seconds=np.nan, **summary))
            append("design_bounds", rows)
            log(s.label, "design:", " ".join(f"{r['design']} {r['resolvable']}" for r in rows))
        except Exception:
            log(s.label, "design FAILED\n" + traceback.format_exc())

# ---------------------------------------------------------------- T3 merge threshold at 35 rows
if "merge" in STAGES:
    for s in spectra[:4]:
        rows = []
        for merge_chi2 in (50., 100., 200., 400.):
            try:
                settings = sparse_fit.SparseFitSettings(merge_chi2=merge_chi2)
                fit, sec = cached(s, f"T1T3_merge{int(merge_chi2)}",
                                  lambda: sparse_fit.fit(s.A, s.time_us, settings, start=t1(s)[0].pole_fit))
                rows.append(dict(merge_chi2=merge_chi2, **summary_row(s, f"T1T3_merge{int(merge_chi2)}", fit, sec)))
            except Exception:
                log(s.label, "merge", merge_chi2, "FAILED\n" + traceback.format_exc())
        append("merge_scan", rows)
        log(s.label, "merge scan:", " ".join(f"{r['merge_chi2']:.0f}: {r['poles']} poles far {r['far_poles']}" for r in rows))

if "diagnose" in STAGES:
    diagnose()

# ---------------------------------------------------------------- C on all spectra (slow)
if "C" in STAGES:
    for s in spectra:
        try:
            fit, sec = cached(s, "C", lambda: joint_refined.fit(s.A, s.time_us, joint_refined.JointRefinedSettings()))
            row = summary_row(s, "C", fit, sec)
            append("fits_summary", [row])
            log(s.label, f"C done: {sec:.0f}s {row['levels_hit']}/{row['levels']} far {row['far_poles']}"
                         f" offsets rms {row['offset_rms_kHz']:.2f} kHz")
        except Exception:
            log(s.label, "C FAILED\n" + traceback.format_exc())
    diagnose("_withC")

# ---------------------------------------------------------------- F from T1 on realization 0 (very slow)
if "F" in STAGES:
    s = spectra[0]
    try:
        fit, sec = cached(s, "T1F", lambda: pursuit.fit(s.A, s.time_us, pursuit.PursuitSettings(), start=t1(s)[0].pole_fit))
        row = summary_row(s, "T1F", fit, sec)
        append("fits_summary", [row])
        log(s.label, f"T1F done: {sec:.0f}s {row['levels_hit']}/{row['levels']} far {row['far_poles']}")
    except Exception:
        log(s.label, "T1F FAILED\n" + traceback.format_exc())
    diagnose("_final")

log("survey end")

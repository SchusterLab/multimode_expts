# -*- coding: utf-8 -*-
"""Convert the data sets of ``configs/datasets/mbr_datasets.yaml`` to the new job layout.

A one-off driver around :mod:`tools.migrate_mbr_jobs` (docs/qsim/mbr_redesign.md,
section 6). It reads the catalog, and for each named data set that is not yet
converted it runs the conversion of its ``kind``. Raw files are never changed:
converted job files go to ``<experiment root>/converted_data/``, manifests and
assembled HDF5s to ``<experiment root>/assembled_data/``. At the end it prints,
for each data set, the ``converted`` / ``manifest`` lines to paste into the
catalog (the catalog is hand-commented YAML, so the script does not rewrite it).

    pixi run python tools/convert_mbr_catalog.py --check            # resolve files and timing only
    pixi run python tools/convert_mbr_catalog.py                    # every unconverted data set
    pixi run python tools/convert_mbr_catalog.py sep10_full_K3p6_g29p2 august25_N3

Per-data-set behaviour that differs from the plain kinds lives in the blocks
below: the disorder record (see ``disorder_record``) and the old section 7-2
pair jobs (``convert_disorder_pairs``).

Disorder record
---------------
The old jobs recorded their disorder realization under several key spellings:
``disorder_*`` (Aug 16-19), ``diagonal_disorder_*`` (Aug 27-30), ``d72_*``
(Sep 01-08, the section 7-2 pair jobs), ``d73_*`` (Sep 10) and flat ``realization`` / ``seed`` / ``disorder_strength_kHz`` /
``target_onsite_MHz`` (Sep 11-14). The Sep 11-14 files also number their
realizations differently from the catalog (r1-r8 are recorded as 0-7, and the
r0 remeasurement as a seedless "manual" realization). The record written to the
manifest therefore takes its ``realization`` index from the catalog, reads the
physics (strength, onsite detunings, seed, self-Kerr) from whichever keys are
present, requires the onsite detunings to agree across the realization's jobs,
and keeps what the files said under ``file_realization`` / ``file_seeds`` /
``source_keys`` for provenance. The new classes never see the old keys.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import migrate_mbr_jobs as mig  # noqa: E402
from experiments.job_paths import data_root, resolve_job_paths  # noqa: E402
from legacy_saved_jobs import job_records  # noqa: E402
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment  # noqa: E402

CATALOG = REPO_ROOT / "configs" / "datasets" / "mbr_datasets.yaml"

# Old-key spellings of one disorder realization, in the order they appeared.
# Each maps record field -> old cfg.expt key; "realization" marks the spelling.
RECORD_KEYS = {
    "disorder_": dict(realization="disorder_realization", seed="disorder_seed",
                      strength_kHz="disorder_strength_kHz", direction="disorder_direction",
                      onsite_MHz="disorder_target_onsite_MHz",
                      selected_occupations="selected_occupations",
                      recorded_theory_energies_MHz="theory_energies_MHz"),
    "diagonal_disorder_": dict(realization="diagonal_disorder_realization",
                               seed="diagonal_disorder_seed",
                               strength_kHz="diagonal_disorder_strength_kHz",
                               direction="diagonal_disorder_direction",
                               onsite_MHz="diagonal_disorder_target_onsite_MHz",
                               selected_occupations="diagonal_disorder_selected_occupations",
                               self_kerr_kHz="diagonal_disorder_self_kerr_kHz"),
    "d73_": dict(realization="d73_realization", seed="d73_seed",
                 strength_kHz="d73_disorder_strength_kHz", onsite_MHz="d73_target_onsite_MHz",
                 self_kerr_kHz="d73_self_kerr_kHz", basis_dimension="d73_basis_dimension"),
    "d72_": dict(realization="d72_realization", seed="d72_seed",
                 strength_kHz="d72_disorder_strength_kHz", onsite_MHz="d72_target_onsite_MHz",
                 self_kerr_kHz="d72_self_kerr_kHz", selected_pairs="d72_selected_pairs"),
    "": dict(realization="realization", seed="seed", strength_kHz="disorder_strength_kHz",
             onsite_MHz="target_onsite_MHz", realization_source="realization_source"),
}


def load_catalog():
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["datasets"]


def _file_record(ecfg):
    """-> (spelling, record) read from one job's cfg.expt, or raise."""
    for prefix, keys in RECORD_KEYS.items():
        if keys["realization"] in ecfg:
            record = {name: ecfg[key] for name, key in keys.items() if key in ecfg}
            if prefix == "disorder_" and "target_manual_kerr_MHz" in ecfg:
                record["self_kerr_kHz"] = 1e3 * float(ecfg.target_manual_kerr_MHz)
            return prefix, record
    raise ValueError("no disorder realization keys in this job")


def disorder_record(spectrum, index):
    """-> the realization record of one converted Spectrum, indexed by the catalog."""
    found = [_file_record(child.cfg.expt) for child in spectrum.children]
    onsite = np.asarray([r["onsite_MHz"] for _, r in found], dtype=float)
    if not np.allclose(onsite, onsite[0], atol=1e-9):
        raise ValueError(f"r={index}: the jobs disagree on the onsite detunings")
    strengths = {round(float(r["strength_kHz"]), 6) for _, r in found}
    if len(strengths) != 1:
        raise ValueError(f"r={index}: the jobs record strengths {sorted(strengths)}")
    first = found[0][1]
    record = dict(realization=int(index),
                  strength_kHz=float(first["strength_kHz"]),
                  onsite_MHz=onsite[0].tolist())
    for key in ("direction", "selected_occupations", "selected_pairs",
                "recorded_theory_energies_MHz", "self_kerr_kHz", "basis_dimension"):
        if key in first:
            record[key] = first[key]
    if "direction" not in record:
        record["direction"] = (1e3 * onsite[0] / record["strength_kHz"]).tolist()
    seeds = sorted({r.get("seed") for _, r in found if r.get("seed") is not None})
    record["seed"] = int(seeds[0]) if len(seeds) == 1 else None
    record["file_seeds"] = [int(s) for s in seeds]
    record["file_realization"] = sorted({int(r["realization"]) for _, r in found})
    record["source_keys"] = sorted({prefix + "*" for prefix, _ in found})
    sources = sorted({r["realization_source"] for _, r in found if "realization_source" in r})
    if sources:
        record["realization_source"] = sources
    return mig.json.loads(mig.json.dumps(record, cls=mig.NpEncoder))


# --------------------------------------------------------------------------
# one block per kind
# --------------------------------------------------------------------------

def convert_disorder(name, entry, load_shots):
    notes = f"{name} ({entry['label']})"
    calibration = mig.migrate_stark_cal(entry["calibration_job_ids"], load_shots=load_shots,
                                        notes=notes)
    parts, records = [], []
    for index, job_ids in sorted(entry["realizations"].items(), key=lambda kv: int(kv[0])):
        spectrum = mig.migrate_spectrum(job_ids, load_shots=load_shots,
                                        notes=f"{notes} r={index}", calibration=calibration)
        parts.append(spectrum)
        records.append(disorder_record(spectrum, index))
        print(f"  r={index}: {len(spectrum.children)} traces, record {records[-1]['source_keys']} "
              f"file_realization={records[-1]['file_realization']}")
    ensemble = MBRDisorderEnsembleExperiment.from_parts(
        parts, realizations=records, calibration=calibration, notes=notes)
    ensemble.analyze(on_error="skip")
    ensemble.save(directory=Path(parts[0].manifest_path).parent)
    return ensemble, calibration


def convert_disorder_pairs(name, entry, load_shots):
    """Old section 7-2 pair jobs: diagonal pairs -> ensemble, the rest -> trace sets.

    Each realization's pairs become TimeTraces (``mig.migrate_pairs``). The
    diagonal ones form that realization's Spectrum, and the Spectra one
    ``MBRDisorderEnsembleExperiment`` (the result). The off-diagonal ones of
    each realization form one ``MBRTimeTraceSetExperiment`` with the same
    realization record; their manifests are printed as ``offdiag_manifests``.
    """
    from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
    from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
    from experiments.qsim.mbr_time_trace_set import MBRTimeTraceSetExperiment

    notes = f"{name} ({entry['label']})"
    if entry.get("calibration_manifest"):
        calibration = MBRCalibrationSetExperiment.from_manifest(
            data_root() / entry["calibration_manifest"])
    else:
        calibration = mig.migrate_stark_cal(entry["calibration_job_ids"],
                                            load_shots=load_shots, notes=notes)
    parts, records, offdiag = [], [], {}
    for index, job_ids in sorted(entry["realizations"].items(), key=lambda kv: int(kv[0])):
        traces = mig.migrate_pairs(job_ids, load_shots=load_shots, notes=notes)
        diagonal = [(trace, source) for (i, f), (trace, source) in traces.items() if i == f]
        others = [(trace, source) for (i, f), (trace, source) in traces.items() if i != f]
        spectrum = MBRSpectrumExperiment.from_children(
            [t for t, _ in diagonal], job_ids=[s for _, s in diagonal],
            notes=f"{notes} r={index} diagonal pairs", calibration=calibration)
        spectrum.analyze()
        spectrum.save()
        record = disorder_record(spectrum, index)
        parts.append(spectrum)
        records.append(record)
        if others:
            traces_set = MBRTimeTraceSetExperiment.from_children(
                [t for t, _ in others], job_ids=[s for _, s in others],
                notes=f"{notes} r={index} off-diagonal pairs", calibration=calibration,
                realization_record=record)
            traces_set.analyze()
            offdiag[int(index)] = traces_set.save(directory=Path(spectrum.manifest_path).parent)
        print(f"  r={index}: {len(diagonal)} diagonal, {len(others)} off-diagonal traces, "
              f"record {record['source_keys']} file_realization={record['file_realization']}")
    ensemble = MBRDisorderEnsembleExperiment.from_parts(
        parts, realizations=records, calibration=calibration, notes=notes)
    ensemble.analyze(on_error="skip")
    ensemble.save(directory=Path(parts[0].manifest_path).parent)
    ensemble.offdiag_manifests = offdiag
    return ensemble, calibration


def convert_spectrum(name, entry, load_shots):
    notes = f"{name} ({entry['label']})"
    spectrum = mig.migrate_spectrum(entry["job_ids"], load_shots=load_shots, notes=notes,
                                    calibration_job_ids=entry.get("calibration_job_ids"))
    return spectrum, spectrum.calibration


def convert_plain(kind):
    def convert(name, entry, load_shots):
        notes = f"{name} ({entry['label']})"
        return mig.MIGRATIONS[kind](entry["job_ids"], load_shots=load_shots, notes=notes), None
    return convert


CONVERTERS = {"disorder": convert_disorder, "spectrum": convert_spectrum,
              "disorder_pairs": convert_disorder_pairs,
              "stark_cal": convert_plain("stark_cal"),
              "orthogonality": convert_plain("orthogonality"),
              "propagator": convert_plain("propagator")}


def all_job_ids(entry):
    ids = list(entry.get("calibration_job_ids", [])) + list(entry.get("job_ids", []))
    for job_ids in entry.get("realizations", {}).values():
        ids += list(job_ids)
    return ids


def check(names, catalog):
    """Resolve every file and its Floquet config version; report what is missing."""
    records = job_records(required=False)
    ok = True
    for name in names:
        ids = all_job_ids(catalog[name])
        try:
            paths = resolve_job_paths(ids)
        except Exception as error:  # noqa: BLE001 -- reported per data set
            print(f"{name}: {error}")
            ok = False
            continue
        roots = {mig.assembled_data.experiment_root(p).name for p in paths.values()}
        no_timing = [j for j in ids if not records.get(j, {}).get("floquet_storage_version_id")]
        print(f"{name}: {len(ids)} jobs, roots {sorted(roots)}, "
              f"{len(no_timing)} without a Floquet version in the sidecar")
        ok = ok and not no_timing
    return ok


def relative_manifest(path):
    path = Path(path)
    return "/".join(path.parts[-3:])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="catalog names (default: all unconverted)")
    parser.add_argument("--check", action="store_true", help="resolve files and timing only")
    parser.add_argument("--no-shots", action="store_true",
                        help="do not copy the per-shot idata/qdata arrays")
    args = parser.parse_args(argv)

    catalog = load_catalog()
    names = args.names or [n for n, e in catalog.items()
                           if not e.get("converted") and e["kind"] in CONVERTERS]
    unknown = [n for n in names if n not in catalog]
    if unknown:
        parser.error(f"not in the catalog: {unknown}")
    skipped = [n for n in names if catalog[n]["kind"] not in CONVERTERS]
    if skipped:
        print(f"not convertible (kind): {skipped}")
        names = [n for n in names if n not in skipped]

    if args.check:
        sys.exit(0 if check(names, catalog) else 1)

    results, failures = {}, {}
    for name in names:
        entry = catalog[name]
        print(f"\n=== {name}: kind {entry['kind']}, {len(all_job_ids(entry))} old jobs")
        try:
            result, calibration = CONVERTERS[entry["kind"]](name, entry, not args.no_shots)
        except Exception:  # noqa: BLE001 -- one failure must not stop the round
            failures[name] = traceback.format_exc()
            print(failures[name])
            continue
        results[name] = (result, calibration)
        print(f"  -> {result.manifest_path}")

    print("\n# paste into the catalog:")
    for name, (result, calibration) in results.items():
        print(f"  {name}:")
        print("    converted: true")
        print(f"    manifest: {relative_manifest(result.manifest_path)!r}")
        if calibration is not None and calibration.manifest_path:
            print(f"    calibration_manifest: {relative_manifest(calibration.manifest_path)!r}")
        for index, path in getattr(result, "offdiag_manifests", {}).items():
            print(f"    offdiag_manifests[{index}]: {relative_manifest(path)!r}")
    if failures:
        print(f"\nFAILED: {sorted(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

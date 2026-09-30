"""Read occupation return traces from the curated JOB manifest.

Only requested HDF5 headers and averaged measurements are read. The NumPy
reconstruction, fixed-N Hamiltonian, and hardware timing below follow
EncodingHamiltonianSpectroscopyExperiment in
experiments/qsim/floquet_dark_mode_readout.py. They omit acquisition classes,
FFT/display work, and instrument dependencies, so remote archives remain read-only.
Calibration JOB IDs are retained as references; calibration data are not loaded
because the decay analysis uses phase-invariant return power.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


_REALIZATION_KEYS = (
    "disorder_realization", "diagonal_disorder_realization", "d72_realization",
    "realization",
)
_STRENGTH_KEYS = (
    "disorder_strength_kHz", "diagonal_disorder_strength_kHz",
    "d72_disorder_strength_kHz",
)
_AUDIT_COLUMNS = (
    "dataset_id", "dataset_title", "realization", "job_id", "occupation",
    "status", "reason", "path", "trace_id",
)


def _index_requested_files(manifest, base_dir):
    """List each project directory once; keep paths for requested JOB IDs only."""
    wanted = defaultdict(set)
    for dataset in manifest:
        for ids in dataset["spectroscopy_job_ids_by_r"].values():
            wanted[dataset["folder"]].update(ids)
    index, errors = {}, {}
    for folder, ids in wanted.items():
        directory = Path(base_dir) / folder / "data"
        paths = defaultdict(list)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    name = entry.name
                    job_id = name[:18]
                    if (job_id in ids and name.startswith(job_id + "_")
                            and name.lower().endswith(".h5") and entry.is_file()):
                        paths[job_id].append(Path(entry.path))
        except OSError as error:
            errors[folder] = f"{type(error).__name__}: {error}"
        index[folder] = {job: sorted(files) for job, files in paths.items()}
    return index, errors


def preflight_decay_data(manifest, base_dir, repo_root, *,
                         config_versions_dir=None, clock_snapshot_path=None):
    """Check paths without importing QICK or opening experiment HDF5 files.

    Missing historical CSVs do not necessarily block a dataset: an HDF5 file
    may contain its own complete ``spectroscopy_hardware`` metadata.
    """
    manifest = list(manifest)
    versions = Path(config_versions_dir) if config_versions_dir is not None else Path(repo_root) / "configs" / "versions"
    snapshot = Path(clock_snapshot_path) if clock_snapshot_path is not None else Path(repo_root) / "configs" / "soccfg_snapshot.json"
    index, errors = _index_requested_files(manifest, base_dir)
    rows = []
    for dataset in manifest:
        folder = dataset["folder"]
        directory = Path(base_dir) / folder / "data"
        ids = list(dict.fromkeys(
            job for jobs in dataset["spectroscopy_job_ids_by_r"].values()
            for job in jobs
        ))
        found = sum(job in index[folder] for job in ids)
        config = (versions / "floquet_storage_swap"
                  / f"{dataset['floquet_config_version']}.csv")
        rows.append(dict(
            dataset_id=dataset["dataset_id"], dataset_title=dataset["dataset_title"],
            data_directory=str(directory), requested_jobs=len(ids),
            found_jobs=found, missing_jobs=len(ids) - found,
            ambiguous_jobs=sum(len(index[folder].get(job, [])) > 1 for job in ids),
            config_version=dataset["floquet_config_version"],
            config_path=str(config), config_exists=config.is_file(),
            clock_snapshot_exists=snapshot.is_file(),
            status=("directory_unavailable" if folder in errors else
                    "no_requested_files" if found == 0 else
                    "partial_files" if found < len(ids) else "files_available"),
            reason=errors.get(folder, ""),
        ))
    return pd.DataFrame(rows)


def _read_header(path):
    with h5py.File(path, "r") as handle:
        cfg = json.loads(handle.attrs["config"])
        ecfg = cfg["expt"]
        if "spectroscopy_occupations" not in ecfg:
            raise ValueError("HDF5 config does not describe occupation spectroscopy")
        if "offdiag_cycles" not in ecfg and "floquet_cycles" not in ecfg:
            raise ValueError("HDF5 config has no spectroscopy cycle sweep")
        if "n_cycle_pairs" in ecfg and len(ecfg.get("floquet_cycles", [0])) < 2:
            raise ValueError("HDF5 file is a phase-calibration job")
        missing = set(("avgi", "xpts", "ypts")).difference(handle.keys())
        if missing:
            raise ValueError(f"HDF5 averages/axes missing: {sorted(missing)}")
        saved_hardware = None
        if "spectroscopy_hardware" in handle.attrs:
            saved_hardware = json.loads(handle.attrs["spectroscopy_hardware"])
        # The selected jobs' averaged arrays are small. Retain them during this
        # same read-only open to avoid a second network HDF5 open per job.
        arrays = {name: np.asarray(handle[name]) for name in ("avgi", "xpts", "ypts")}
    return dict(path=path, cfg=cfg, saved_hardware=saved_hardware, arrays=arrays)


def _occupations(header):
    ecfg = header["cfg"]["expt"]
    initial = tuple(ecfg["spectroscopy_occupations"])
    final = tuple(ecfg.get("offdiag_decoder_occupation",
                          ecfg.get("spectroscopy_final_occupations", initial)))
    for occupation in (initial, final):
        if (len(occupation) != len(ecfg["swap_stors"]) + 1
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                       for n in occupation)
                or sum(occupation) < 1):
            raise ValueError(f"Invalid saved occupation: {occupation}")
    if sum(initial) != sum(final):
        raise ValueError("Initial/final occupations have different photon numbers")
    return initial, final


def _saved_kerr_MHz(header):
    cfg = header["cfg"]
    values = np.asarray(cfg["device"]["manipulate"]["kerr"], dtype=float).reshape(-1)
    man = int(cfg["expt"].get("man_mode_no", 1))
    if not values.size or man < 1 or (len(values) > 1 and man > len(values)):
        raise ValueError("Saved manipulate Kerr does not cover the selected mode")
    value = float(values[0 if len(values) == 1 else man - 1])
    if not np.isfinite(value):
        raise ValueError("Saved manipulate Kerr is not finite")
    return -abs(value)


def _detunings_MHz(header):
    ecfg = header["cfg"]["expt"]
    value = ecfg.get("detunings")
    if value is None or value is False or np.asarray(value).size == 0:
        value = np.zeros(len(ecfg["swap_stors"]))
    result = np.asarray(value, dtype=float)
    if result.shape != (len(ecfg["swap_stors"]),) or not np.isfinite(result).all():
        raise ValueError("Saved detunings do not match the selected storage modes")
    return result


def _physical_key(header):
    """Split differing physical conditions explicitly, never choose a majority."""
    ecfg = header["cfg"]["expt"]
    fields = (
        "swap_stors", "man_mode_no", "floquet_waveform", "floquet_gauss_sigma",
        "scramble_sync_cycles", "use_multiphoton_swap", "floquet_cycle_us",
        "decoder_cycle_us", "spectroscopy_phase_correction_mode",
        "final_analyzer_phase_per_cycle_deg", "final_analyzer_phase_application_sign",
        "offdiag_decoder_phase_correction_deg",
    )
    settings = {key: ecfg.get(key) for key in fields}
    saved_hardware = header["saved_hardware"]
    # Provenance strings can contain the individual filename. They must not
    # split the phi=0 and phi=90 jobs when the physical timing is identical.
    hardware_key = None if saved_hardware is None else {
        key: saved_hardware.get(key)
        for key in ("floquet_cycle_us", "couplings_MHz", "decoder_cycle_us")
    }
    settings.update(
        kerr_MHz=_saved_kerr_MHz(header),
        detunings_MHz=_detunings_MHz(header).tolist(),
        saved_hardware=hardware_key,
        acquisition_kind="offdiag" if "offdiag_cycles" in ecfg else "standard",
    )
    return json.dumps(settings, sort_keys=True, allow_nan=False)


def _hardware_from_clock(cfg, table, snapshot):
    """Match spectroscopy hardware_parameters() and QICK scalar clock rounding.

    QickConfig.us2cycles uses int(np.round(us * fclk)); cycles2us divides
    by fclk. References: openquantumhardware/qick, qick_lib/qick/qick_asm.py
    (us2cycles/cycles2us), and qick_lib/qick/helpers.py (to_int).
    Clock frequencies come only from the supplied RFSoC snapshot.
    """
    ecfg = cfg["expt"]
    sync = ecfg.get("scramble_sync_cycles", 10)
    if isinstance(sync, bool) or not isinstance(sync, int) or sync < 0:
        raise ValueError("Saved sync_cycles must be a nonnegative integer")
    tproc = float(snapshot["tprocs"][0]["f_time"])
    if not np.isfinite(tproc) or tproc <= 0:
        raise ValueError("RFSoC snapshot tProc clock is invalid")
    ramp_sigma = float(np.asarray(cfg["device"]["manipulate"]["ramp_sigma"]).reshape(-1)[0])
    cycle_tproc_cycles = 0
    pi_fracs = []
    for mode in ecfg["swap_stors"]:
        row = table.loc[f"M1-S{int(mode)}"]
        flux = "flux_low" if float(row["freq (MHz)"]) < 1800 else "flux_high"
        channel = int(cfg["hw"]["soc"]["dacs"][flux]["ch"][0])
        fabric = float(snapshot["gens"][channel]["f_fabric"])
        if not np.isfinite(fabric) or fabric <= 0:
            raise ValueError("RFSoC snapshot generator clock is invalid")

        def ticks(value):
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Historical pulse lengths must be finite and nonnegative")
            return int(np.round(value * fabric))

        waveform = ecfg.get("floquet_waveform")
        if waveform is None:
            waveform = row["waveform"]
        if waveform in ("gauss", "gaussian", "arb"):
            sigma = ecfg.get("floquet_gauss_sigma")
            if sigma is None:
                sigma = row["gauss_sigma (mus)"]
            pulse_cycles = ticks(sigma) * int(row["gauss_n_sigma"])
        elif waveform == "preload_flattop":
            pulse_cycles = ticks(row["len (mus)"]) + 6 * ticks(row["ramp_sigma (mus)"])
        else:
            pulse_cycles = ticks(row["len (mus)"]) + 6 * ticks(ramp_sigma)
        if pulse_cycles <= 0:
            raise ValueError("Historical pulse duration must be positive")
        # Match integer synci, including the ratio-first floating calculation.
        cycle_tproc_cycles += int(pulse_cycles * (tproc / fabric) + sync)
        pi_fracs.append(float(row["pi_frac"]))
    cycle_us = cycle_tproc_cycles / tproc
    pi_fracs = np.asarray(pi_fracs, dtype=float)
    if not np.isfinite(pi_fracs).all() or np.any(pi_fracs <= 0) or cycle_us <= 0:
        raise ValueError("Historical cycle duration and pi fractions must be positive")
    return cycle_us, 1. / (4. * pi_fracs * cycle_us)


def _hardware(header, version, versions, snapshot_path, station_cache):
    saved = header["saved_hardware"]
    ecfg = header["cfg"]["expt"]
    modes = list(map(int, ecfg["swap_stors"]))
    if int(ecfg.get("man_mode_no", 1)) != 1:
        raise ValueError("The existing Floquet Hamiltonian API supports M1 only")
    if saved is not None:
        cycle_us = float(saved["floquet_cycle_us"])
        couplings = np.asarray(saved["couplings_MHz"], dtype=float)
        source = "HDF5 spectroscopy_hardware"
    else:
        csv_path = (versions / "floquet_storage_swap"
                    / f"{version}.csv")
        if not csv_path.is_file():
            raise FileNotFoundError(f"No saved HDF5 T/g and historical Floquet CSV missing: {csv_path}")
        if version not in station_cache:
            table = pd.read_csv(csv_path).set_index("stor_name", verify_integrity=True)
            # These are precisely FloquetStorageSwapDataset's legacy-column
            # defaults; waveform-specific numeric parameters still come from CSV/config.
            for key, value in {"waveform": "flat_top", "gauss_sigma (mus)": -1., "gauss_n_sigma": 4}.items():
                if key not in table:
                    table[key] = value
            station_cache[version] = table
        table = station_cache[version]
        recorded_cycle = ecfg.get("floquet_cycle_us")
        if recorded_cycle is not None:
            cycle_us = float(recorded_cycle)
            pi_fracs = np.asarray([
                table.loc[f"M1-S{mode}", "pi_frac"] for mode in modes
            ], dtype=float)
            if not np.isfinite(pi_fracs).all() or np.any(pi_fracs <= 0):
                raise ValueError("Historical Floquet pi fractions must be finite and positive")
            couplings = 1. / (4. * pi_fracs * cycle_us)
            source = f"HDF5 expt.floquet_cycle_us + Floquet CSV {version}"
        else:
            if "clock_snapshot" not in station_cache:
                station_cache["clock_snapshot"] = json.loads(snapshot_path.read_text(encoding="utf-8"))
            cycle_us, couplings = _hardware_from_clock(
                header["cfg"], table, station_cache["clock_snapshot"],
            )
            source = f"Floquet CSV {version} + HDF5 config + RFSoC clock snapshot"
    if (not np.isfinite(cycle_us) or cycle_us <= 0
            or couplings.shape != (len(modes),)
            or not np.isfinite(couplings).all() or np.any(couplings <= 0)):
        raise ValueError("Floquet cycle time/couplings must be finite, positive, and match the modes")
    return cycle_us, couplings, source


def _return_quadrature(cfg, arrays):
    """Same averaged-readout normalization as the experiment's _quadrature()."""
    xpts = np.asarray(arrays["xpts"])
    ypts = np.asarray(arrays["ypts"])
    if xpts.shape != (2,) or not np.allclose(xpts, [0., 180.]):
        raise ValueError("Saved preparation phases must be [0, 180]")
    signal = np.asarray(arrays["avgi"], dtype=float).reshape(len(ypts), len(xpts))
    qubit = int(cfg["expt"]["qubits"][0])
    readout = cfg["device"]["readout"]
    Ig, Ie = float(readout["Ig"][qubit]), float(readout["Ie"][qubit])
    if not np.isfinite([Ig, Ie]).all() or np.isclose(Ig, Ie):
        raise ValueError("Saved readout Ig/Ie are invalid or equal")
    Pe = (signal - Ig) / (Ie - Ig)
    return Pe[:, 0] - Pe[:, 1]


def _reconstruct_return(headers, arrays):
    """Match reconstruct_spectroscopy() for one occupation and its chunks.

    The interleaved acquisition path follows reconstruct_pair_spectroscopy().
    Its phase rotation is retained even though it leaves return power unchanged.
    """
    offdiag = "offdiag_cycles" in headers[0]["cfg"]["expt"]
    if offdiag:
        cycle_chunks, return_chunks = [], []
        for header, values in zip(headers, arrays):
            ecfg = header["cfg"]["expt"]
            cycles = np.asarray(ecfg["offdiag_cycles"], dtype=float)
            q = _return_quadrature(header["cfg"], values).reshape(-1, 2)
            if len(q) != len(cycles):
                raise ValueError("Interleaved quadratures do not match the saved cycle list")
            phase = float(ecfg["offdiag_decoder_phase_correction_deg"])
            cycle_chunks.append(cycles)
            return_chunks.append((q[:, 0] - 1j * q[:, 1]) * np.exp(-1j * np.deg2rad(phase * cycles)))
        cycles = np.concatenate(cycle_chunks)
        order = np.argsort(cycles)
        return cycles[order], np.concatenate(return_chunks)[order]
    chunks = {0.: [], 90.: []}
    for header, values in zip(headers, arrays):
        ecfg = header["cfg"]["expt"]
        cycles = np.asarray(values["ypts"], dtype=float)
        if not np.array_equal(cycles, np.asarray(ecfg["floquet_cycles"])):
            raise ValueError("Saved HDF5 cycle axis does not match the job config")
        phase = float(ecfg["spectroscopy_analyzer_phase"])
        chunks[phase].append((cycles, _return_quadrature(header["cfg"], values)))
    quadratures = []
    expected_cycles = None
    for phase in (0., 90.):
        cycles = np.concatenate([item[0] for item in chunks[phase]])
        q = np.concatenate([item[1] for item in chunks[phase]])
        order = np.argsort(cycles)
        cycles = cycles[order]
        if expected_cycles is None:
            expected_cycles = cycles
        elif not np.array_equal(cycles, expected_cycles):
            raise ValueError("The analyzer quadratures have different saved cycle axes")
        quadratures.append(q[order])
    return expected_cycles, quadratures[0] - 1j * quadratures[1]


def _coherent_return(occupation, time_us, detunings, couplings, signed_K_MHz,
                     eigensystem_cache=None):
    """Exact fixed-N diagonal return from the existing analyze_spectrum model.

    H/h has storage onsite energies -detunings, central self-Kerr
    K*n_M1*(n_M1-1)/2, and star couplings g*sqrt(n_M1*(n_storage+1)).
    This is the undamped coherent return, before any FFT or magnitude scaling.
    """
    occupation = tuple(occupation)
    N = sum(occupation)
    mode_count = len(occupation)
    key = (N, mode_count, tuple(detunings), tuple(couplings), float(signed_K_MHz))
    if eigensystem_cache is None:
        eigensystem_cache = {}
    if key not in eigensystem_cache:
        basis = [state for state in product(range(N + 1), repeat=mode_count)
                 if sum(state) == N]
        basis_index = {state: index for index, state in enumerate(basis)}
        H = np.zeros((len(basis), len(basis)))
        onsite = np.concatenate(([0.], -np.asarray(detunings)))
        for column, state in enumerate(basis):
            n_man = state[0]
            H[column, column] = np.dot(onsite, state) + .5 * signed_K_MHz * n_man * (n_man - 1)
            if n_man == 0:
                continue
            for mode, coupling in enumerate(couplings, start=1):
                final = list(state)
                final[0] -= 1
                final[mode] += 1
                row = basis_index[tuple(final)]
                element = coupling * np.sqrt(n_man * (state[mode] + 1))
                H[row, column] += element
                H[column, row] += element
        energies, states = np.linalg.eigh(H)
        eigensystem_cache[key] = energies, states, basis_index
    energies, states, basis_index = eigensystem_cache[key]
    weights = np.abs(states[basis_index[occupation]]) ** 2
    return weights @ np.exp(-2j * np.pi * np.outer(energies, time_us))


def _validate_pair(headers):
    offdiag = "offdiag_cycles" in headers[0]["cfg"]["expt"]
    coverage = {0.: [], 90.: []}
    for header in headers:
        ecfg = header["cfg"]["expt"]
        cycles = np.asarray(ecfg["offdiag_cycles"] if offdiag else ecfg["floquet_cycles"], dtype=float)
        if cycles.ndim != 1 or not np.isfinite(cycles).all() or np.any(cycles < 0):
            raise ValueError("Saved spectroscopy cycle points are invalid")
        phases = (0., 90.) if offdiag else (float(ecfg["spectroscopy_analyzer_phase"]),)
        for phase in phases:
            if phase not in coverage:
                raise ValueError(f"Unsupported analyzer phase: {phase}")
            coverage[phase].extend(cycles.tolist())
    first, second = (np.sort(coverage[phase]) for phase in (0., 90.))
    if len(first) < 3 or not np.array_equal(first, second):
        raise ValueError("Occupation needs matching phi=0/90 coverage with at least three cycle points")
    if len(np.unique(first)) != len(first):
        raise ValueError("Duplicate/overlapping cycle points; repeated measurements were not silently averaged")
    # Direct time-domain fitting supports gaps/unequal spacing. Only matching
    # quadratures and unique physical cycle points are required; no FFT is used.


def _stored_realization(headers):
    fields = {}
    for key in _REALIZATION_KEYS:
        values = sorted({int(header["cfg"]["expt"][key]) for header in headers
                         if header["cfg"]["expt"].get(key) is not None})
        if values:
            fields[key] = values
    all_values = sorted({value for values in fields.values() for value in values})
    value = all_values[0] if len(all_values) == 1 else tuple(all_values) if all_values else None
    return value, fields


def _disorder_strength(headers, detunings):
    for key in _STRENGTH_KEYS:
        values = [header["cfg"]["expt"].get(key) for header in headers]
        if all(value is not None for value in values):
            values = np.asarray(values, dtype=float)
            if np.isfinite(values).all() and np.allclose(values, values[0], rtol=0, atol=1e-6):
                return float(values[0]), f"saved {key}"
    return float(1e3 * np.sqrt(np.mean(detunings ** 2))), "RMS of saved pulse detunings"


def load_decay_traces(manifest, base_dir, repo_root, *, kerr_source="document",
                      config_versions_dir=None, clock_snapshot_path=None,
                      progress=True):
    """Return ``(trace_dicts, audit_dataframe)`` from documented spectroscopy jobs.

    Every complete occupation pair keeps its own time grid. Physical settings
    that differ are split into separate conditions, rather than filtered by a
    majority vote. Missing files, invalid pairs, missing hardware, and analysis
    failures are explicit audit rows, and other datasets continue loading.

    ``kerr_source='document'`` uses the curated document's K magnitude for the
    coherent Hamiltonian; some entries were refined from spectroscopy. ``saved``
    uses the HDF5 manipulate config. Both are retained with provenance. The
    Hamiltonian uses ``-abs(K)``. No Kerr or g is inferred from the observed decay.
    Calibration IDs are references only and are never treated as loaded or used.
    """
    if kerr_source not in {"document", "saved"}:
        raise ValueError("kerr_source must be 'document' or 'saved'")
    manifest = list(manifest)
    versions = Path(config_versions_dir) if config_versions_dir is not None else Path(repo_root) / "configs" / "versions"
    snapshot = Path(clock_snapshot_path) if clock_snapshot_path is not None else Path(repo_root) / "configs" / "soccfg_snapshot.json"

    def report(message):
        if callable(progress):
            progress(message)
        elif progress:
            print(message, flush=True)

    report("Indexing requested spectroscopy files (read-only)")
    index, directory_errors = _index_requested_files(manifest, base_dir)
    traces, audit = [], []
    header_cache, station_cache = {}, {}
    eigensystem_cache = {}

    for dataset in manifest:
        folder = dataset["folder"]
        dataset_trace_start = len(traces)
        report(f"Dataset {dataset['dataset_id']}: {dataset['dataset_title']}")
        for realization, requested_ids in dataset["spectroscopy_job_ids_by_r"].items():
            report(f"  r={realization}: reading {len(requested_ids)} requested headers")
            grouped = defaultdict(list)
            for job_index, job_id in enumerate(requested_ids, start=1):
                if job_index % 25 == 0:
                    report(f"  r={realization}: header {job_index}/{len(requested_ids)}")
                entry = dict(
                    dataset_id=dataset["dataset_id"], dataset_title=dataset["dataset_title"],
                    realization=realization, job_id=job_id, occupation=None,
                    status="pending", reason="", path=None, trace_id=None,
                )
                audit.append(entry)
                paths = index[folder].get(job_id, [])
                if not paths:
                    entry.update(status="missing_file", reason=directory_errors.get(
                        folder, f"No matching HDF5 in {Path(base_dir) / folder / 'data'}"))
                    continue
                candidates, errors = [], []
                for path in paths:
                    cache_key = str(path)
                    if cache_key not in header_cache:
                        try:
                            header_cache[cache_key] = _read_header(path)
                        except (OSError, KeyError, ValueError, TypeError) as error:
                            header_cache[cache_key] = error
                    header = header_cache[cache_key]
                    if isinstance(header, Exception):
                        errors.append(f"{path.name}: {type(header).__name__}: {header}")
                    else:
                        candidates.append(header)
                if len(candidates) != 1:
                    entry.update(
                        status="ambiguous_files" if len(candidates) > 1 else "rejected_header",
                        reason=("Multiple spectroscopy HDF5 files for one JOB ID" if len(candidates) > 1
                                else "; ".join(errors)),
                        path="; ".join(map(str, paths)),
                    )
                    continue
                header = candidates[0]
                entry["path"] = str(header["path"])
                try:
                    initial, final = _occupations(header)
                    if initial != final:
                        entry.update(status="excluded_nonreturn", occupation=initial,
                                     reason="This investigation uses diagonal occupation return amplitudes only")
                        continue
                    key = _physical_key(header)
                except (KeyError, ValueError, TypeError) as error:
                    entry.update(status="invalid_metadata", reason=f"{type(error).__name__}: {error}")
                    continue
                entry["occupation"] = initial
                grouped[(initial, final, key)].append((job_id, header, entry))

            for (initial, final, physical_key), items in grouped.items():
                headers = [item[1] for item in items]
                condition_id = sha256(physical_key.encode()).hexdigest()[:12]
                trace_id = (f"{dataset['dataset_id']}/r={realization}/condition={condition_id}"
                            f"/occupation={initial}/final={final}")
                stage = "invalid_pair"
                try:
                    _validate_pair(headers)
                    stage = "hardware_unavailable"
                    cycle_us, couplings, source = _hardware(
                        headers[0], dataset["floquet_config_version"], versions,
                        snapshot, station_cache,
                    )
                    stage = "array_read_error"
                    arrays = [header["arrays"] for header in headers]
                    stage = "analysis_error"
                    cycles, A = _reconstruct_return(headers, arrays)
                    saved_kerr = _saved_kerr_MHz(headers[0])
                    document_K_kHz = float(dataset["nominal_K_kHz"])
                    selected_K = abs(document_K_kHz) if kerr_source == "document" else abs(1e3 * saved_kerr)
                    if not np.isfinite(selected_K):
                        raise ValueError("Selected Kerr is not finite")
                    detunings = _detunings_MHz(headers[0])
                    time_us = cycles * cycle_us
                    coherent_A = _coherent_return(
                        initial, time_us, detunings, couplings, -selected_K / 1e3,
                        eigensystem_cache,
                    )
                    if (A.shape != time_us.shape or coherent_A.shape != A.shape
                            or not np.isfinite(A).all() or not np.isfinite(coherent_A).all()):
                        raise ValueError("Measured/theory trace shape mismatch or nonfinite amplitudes")
                    stored_realization, stored_fields = _stored_realization(headers)
                    strength, strength_source = _disorder_strength(headers, detunings)
                    modes = list(map(int, headers[0]["cfg"]["expt"]["swap_stors"]))
                    traces.append(dict(
                        trace_id=trace_id, condition_id=condition_id,
                        dataset_id=dataset["dataset_id"], dataset_title=dataset["dataset_title"],
                        group_title=dataset["group_title"], folder=folder,
                        realization=realization, stored_realization=stored_realization,
                        stored_realization_fields=stored_fields,
                        occupation=initial, final_occupation=final,
                        mode_labels=["M1"] + [f"S{mode}" for mode in modes],
                        n_central=initial[0], total_photons=sum(initial),
                        time_us=time_us, cycles=cycles,
                        A=A, coherent_A=coherent_A, floquet_cycle_us=cycle_us,
                        document_K_kHz=document_K_kHz, saved_K_kHz=abs(1e3 * saved_kerr),
                        saved_signed_K_kHz=1e3 * saved_kerr,
                        K_kHz=selected_K, theory_signed_K_kHz=-selected_K,
                        kerr_source=kerr_source,
                        g_kHz=(1e3 * couplings).tolist(),
                        nominal_g_kHz=float(dataset["nominal_g_kHz"]),
                        g_mean_kHz=float(1e3 * np.mean(couplings)),
                        detunings_kHz=(1e3 * detunings).tolist(),
                        disorder_strength_kHz=strength, disorder_strength_source=strength_source,
                        documented_disorder=dataset.get("documented_disorder"),
                        job_ids=[item[0] for item in items],
                        job_paths=[str(header["path"]) for header in headers],
                        calibration_job_ids=list(dataset["calibration_job_ids"]),
                        calibration_used=False,
                        config_version=dataset["floquet_config_version"], hardware_source=source,
                        source_path=dataset.get("source_path"), source_line=dataset.get("source_line"),
                        phase_frame="as_acquired", fit_observable="return_power",
                    ))
                    for _, _, entry in items:
                        entry.update(status="used", reason="", trace_id=trace_id)
                except (OSError, KeyError, ValueError, TypeError, RuntimeError, IndexError, np.linalg.LinAlgError) as error:
                    reason = f"{type(error).__name__}: {error}"
                    for _, _, entry in items:
                        entry.update(status=stage, reason=reason, trace_id=trace_id)
            report(f"  r={realization}: occupation groups processed; {len(traces) - dataset_trace_start} dataset traces loaded")
        report(f"Dataset {dataset['dataset_id']}: loaded {len(traces) - dataset_trace_start} occupation traces")

    return traces, pd.DataFrame(audit, columns=_AUDIT_COLUMNS)

"""tools/convert_fock_population_catalog.py on synthetic old population jobs."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from experiments import job_paths
from experiments.qsim.agp import CentralBosonAGPDisorderScan, direction_sector
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationSetExperiment
from tests.population_jobs import shots_from_labels

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import convert_fock_population_catalog as tool  # noqa: E402

CYCLES = [0, 5, 10]
TRUE_CONFUSION = .9 * np.eye(4) + .025


def old_job(root, job_id, occupation, ro_stor, labels, detunings, cycles, threshold=0.):
    """Write one job as the old QsimExperiment + FockPopulationProgram saved it."""
    labels = np.asarray(labels)
    cfg = dict(read_num=4, device=dict(readout=dict(threshold=[threshold])),
               expt=dict(qubits=[0], readout="multiparity", cond_sec_phase=-90,
                         phase_second_pulse=180, active_reset=True, occupations=occupation,
                         ro_stor=ro_stor, swap_stors=[4, 7], detunings=detunings,
                         floquet_cycles=cycles, swept_params=["floquet_cycle"],
                         reps=labels.shape[1]))
    path = root / "exp" / "data" / f"{job_id}_QsimExperiment.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    idata = shots_from_labels(labels)
    with h5py.File(path, "w") as handle:
        handle.attrs["config"] = json.dumps(cfg)
        handle.attrs["derived_params"] = json.dumps(dict(floquet_cycle_us=.3))
        handle.attrs["config_versions"] = json.dumps(dict(hardware_config="CFG-HW-1"))
        handle["xpts"] = np.asarray(cycles)
        handle["idata"] = idata
        handle["qdata"] = np.zeros_like(idata)
        for key in ("avgi", "avgq", "amps", "phases"):
            handle[key] = np.zeros(len(cycles))
    return path


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv(job_paths.DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(job_paths.BACKEND_ENV, "index")
    job_paths.clear_cache()
    rng = np.random.default_rng(7)
    number = iter(range(1, 100))

    def job_id():
        return f"JOB-20261017-{next(number):05d}"

    def labels(cycles, shots=60):
        return [rng.choice(4, size=shots, p=rng.dirichlet(np.ones(4))) for _ in cycles]

    calibration = []
    for prepared in range(4):
        calibration.append(job_id())
        old_job(tmp_path, calibration[-1], [prepared, 0, 0], 0,
                [rng.choice(4, size=400, p=TRUE_CONFUSION[prepared])], [0., 0.], [0])
    walk = []
    for mode in (0, 4, 7):
        walk.append(job_id())
        old_job(tmp_path, walk[-1], [1, 0, 0], mode, labels(CYCLES), [0., 0.], CYCLES)
    parts = {}
    for strength in (30, 0):
        parts[strength] = dict(calibration="cal", job_ids=[])
        for state in direction_sector(2, [4, 7], 7):
            parts[strength]["job_ids"].append(job_id())
            old_job(tmp_path, parts[strength]["job_ids"][-1], state, 7, labels(CYCLES),
                    [.03 * strength / 30, -.03 * strength / 30], CYCLES)
    return dict(
        calibrations=dict(cal=dict(folder="exp", job_ids=calibration)),
        datasets=dict(
            walk=dict(label="walk", kind="walk", folder="exp", calibration="cal", job_ids=walk),
            scan=dict(label="scan", kind="agp_scan", folder="exp", direction=7, parts=parts)))


def digests(root):
    return {path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "exp" / "data").glob("*.h5"))}


def test_converts_jobs_and_writes_three_layers_without_changing_raw_files(catalog, tmp_path):
    before = digests(tmp_path)
    manifests = tool.Converter(catalog).run(["walk", "scan"])
    assert digests(tmp_path) == before

    converted = tmp_path / "exp" / "converted_data"
    assert len(list(converted.glob("*_FockPopulationExperiment.h5"))) == 3 + 6
    calibration_file, = converted.glob("*_MultiParityConfusionMatrixExperiment.h5")
    with h5py.File(calibration_file, "r") as handle:
        matrix = handle["multiparity_confusion_matrix"][()]
        assert json.loads(handle.attrs["converted_from"])["job_ids"] == catalog["calibrations"]["cal"]["job_ids"]
    np.testing.assert_allclose(matrix, TRUE_CONFUSION, atol=.05)

    job_id = catalog["datasets"]["walk"]["job_ids"][1]
    job = FockPopulationExperiment.from_h5file(converted / f"converted_{job_id}_FockPopulationExperiment.h5")
    with h5py.File(job.fname, "r") as handle:
        assert handle.attrs["job_class"] == "FockPopulationExperiment"
        assert json.loads(handle.attrs["config_versions"]) == dict(hardware_config="CFG-HW-1")
        assert json.loads(handle.attrs["converted_from"])["job_ids"] == [job_id]
    with h5py.File(tmp_path / "exp" / "data" / f"{job_id}_QsimExperiment.h5", "r") as handle:
        np.testing.assert_array_equal(job.data["idata"], handle["idata"][()])
    np.testing.assert_allclose(job.data["confusion_matrix"], matrix)
    raw = np.stack([job.data[f"p_mod{n}"] for n in range(4)], axis=-1)
    np.testing.assert_allclose(job.data["nmod4_mean_corrected"],
                               np.linalg.solve(matrix.T, raw.T).T @ np.arange(4))

    walk = FockPopulationSetExperiment.from_manifest(manifests["walk"])
    walk.analyze()
    np.testing.assert_array_equal(walk.data["ro_stor"], [0, 4, 7])
    np.testing.assert_allclose(walk.data["time_us"], np.asarray(CYCLES) * .3)

    scan = CentralBosonAGPDisorderScan.from_manifest(manifests["scan"])
    assert [part.disorder_khz for part in scan.children] == [0, 30]
    with h5py.File(manifests["scan"].with_suffix(".h5"), "r") as handle:
        assert "idata" not in handle and handle["C1_corrected"].shape == (2, 3)


def test_existing_converted_files_stop_the_run_unless_reused(catalog):
    tool.Converter(catalog).run(["walk"])
    with pytest.raises(FileExistsError, match="--reuse"):
        tool.Converter(catalog).run(["walk"])
    manifest = tool.Converter(catalog, reuse=True).run(["walk"])["walk"]
    assert yaml.safe_load(manifest.read_text(encoding="utf-8"))["class"] == "FockPopulationSetExperiment"


def test_check_finds_a_job_read_with_another_threshold(catalog, tmp_path):
    job_id = catalog["datasets"]["walk"]["job_ids"][0]
    path = tmp_path / "exp" / "data" / f"{job_id}_QsimExperiment.h5"
    with h5py.File(path, "r+") as handle:
        cfg = json.loads(handle.attrs["config"])
        cfg["device"]["readout"]["threshold"] = [.5]
        handle.attrs["config"] = json.dumps(cfg)
    with pytest.raises(ValueError, match="readout settings differ"):
        tool.Converter(catalog, check=True).run(["walk"])


def test_a_wrong_disorder_label_is_refused(catalog):
    parts = catalog["datasets"]["scan"]["parts"]
    parts[45] = parts.pop(30)
    with pytest.raises(ValueError, match="detuning RMS"):
        tool.Converter(catalog).run(["scan"])

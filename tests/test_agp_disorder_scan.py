"""CentralBosonAGPDisorderScan: one CentralBosonAGPExperiment per W, saved as C_k(W, t) maps."""
import h5py
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from experiments.qsim.agp import (  # noqa: E402
    CentralBosonAGPDisorderScan,
    CentralBosonAGPExperiment,
    direction_sector,
)
from experiments.qsim.fock_population import FockPopulationExperiment  # noqa: E402
from tests.population_jobs import CONFUSION, FakeRunner  # noqa: E402

CYCLES = [0, 2, 4, 6]
PATTERN = [1., -3.]
SECTOR = dict(photon_sector_number=2, direction=7, swap_stors=[4, 7], cycles=CYCLES)


def planned(strengths=(60, 0, 30), **changes):
    return CentralBosonAGPDisorderScan.from_pattern(
        list(strengths), PATTERN, **dict(SECTOR, **changes), calibration=CONFUSION)


def test_pattern_sets_each_parts_detunings_and_parts_are_sorted_by_w():
    scan = planned()
    assert [part.disorder_khz for part in scan.children] == [0, 30, 60]
    unit = np.asarray(PATTERN) / np.sqrt(np.mean(np.square(PATTERN)))
    for part in scan.children:
        np.testing.assert_allclose(part.detunings, 1e-3 * part.disorder_khz * unit)
    assert len(scan.job_overrides()) == 3 * len(direction_sector(2, [4, 7], 7))
    with pytest.raises(ValueError, match="nonzero pattern"):
        CentralBosonAGPDisorderScan.from_pattern([30], [0., 0.], **SECTOR)


def test_acquire_splits_the_jobs_into_parts_and_saves_three_layers(tmp_path):
    scan = planned()
    runner = FakeRunner(tmp_path / "data")
    scan.acquire(runner, batch_size=4, use_queue=True)
    assert runner.calls == [dict(count=9, batch_size=4, use_queue=True)]
    for part in scan.children:
        assert len(part.children) == 3 and len(part.job_ids) == 3
        assert all(isinstance(child, FockPopulationExperiment) for child in part.children)
        assert {tuple(child.cfg.expt.detunings) for child in part.children} == {tuple(part.detunings)}

    data = scan.analyze()
    assert data["C1"].shape == (3, 4) and data["C1_corrected"].shape == (3, 4)
    np.testing.assert_array_equal(data["disorder_khz"], [0, 30, 60])
    assert data["detunings"].shape == (3, 2)

    manifest_path = scan.save()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["class"] == "CentralBosonAGPDisorderScan"
    assert all("_CentralBosonAGPExperiment" in path for path in manifest["raw_files"])
    assert manifest["parameters"]["disorder_khz"] == [0, 30, 60]
    with h5py.File(manifest_path.with_suffix(".h5"), "r") as handle:
        assert set(handle) == {"disorder_khz", "cycles", "detunings", "time_us", "C1", "C2",
                               "C1_err", "C2_err", "C1_corrected", "C2_corrected"}

    reloaded = CentralBosonAGPDisorderScan.from_manifest(manifest_path)
    reloaded.analyze()
    for key, value in data.items():
        np.testing.assert_allclose(reloaded.data[key], value)
    view = CentralBosonAGPDisorderScan.from_h5file(manifest_path.with_suffix(".h5"))
    for moment in (1, 2):
        for raw in (False, True):
            fig, ax = view.display(moment=moment, raw=raw)
            assert ("corrected" in ax.get_title()) != raw


def test_saved_parts_combine_without_saving_them_again(tmp_path):
    scan = planned((0, 30))
    scan.acquire(FakeRunner(tmp_path / "data"))
    for part in scan.children:
        part.analyze()
    part_paths = [part.save() for part in scan.children]
    parts = [CentralBosonAGPExperiment.from_manifest(path) for path in reversed(part_paths)]
    combined = CentralBosonAGPDisorderScan.from_children(parts, notes="two W")
    combined.save()
    assert [part.manifest_path for part in combined.children] == part_paths
    assert len(list((tmp_path / "assembled_data").glob("*_CentralBosonAGPExperiment*.yaml"))) == 2


@pytest.mark.parametrize("change", [dict(cycles=[0, 2]), dict(direction=4)])
def test_parts_with_different_settings_are_not_combined(change):
    first = planned((0,)).children[0]
    other = planned((30,), **change).children[0]
    with pytest.raises(ValueError, match="different"):
        CentralBosonAGPDisorderScan([first, other])


def test_acquire_needs_a_population_runner(tmp_path):
    runner = FakeRunner(tmp_path)
    runner.ExptClass = CentralBosonAGPExperiment
    with pytest.raises(TypeError, match="FockPopulationExperiment"):
        planned().acquire(runner)


def test_display_needs_one_part_per_w(tmp_path):
    scan = planned((30, 30))
    scan.acquire(FakeRunner(tmp_path / "data"))
    scan.analyze()
    with pytest.raises(ValueError, match="one part per W"):
        scan.display()

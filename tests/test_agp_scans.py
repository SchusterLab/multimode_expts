"""AGP data sets: background-subtracted estimator, list/load/compare disorder scans."""
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from experiments.qsim import agp  # noqa: E402
from experiments.qsim.agp import (  # noqa: E402
    CentralBosonAGPDisorderScan,
    display_scans,
    list_scans,
    load_scan,
    local_path,
)
from fitting.qsim.agp import agp_kernel_weights, subtract_background  # noqa: E402
from tests.population_jobs import CONFUSION, FakeRunner  # noqa: E402

SECTOR = dict(photon_sector_number=2, direction=7, swap_stors=[4, 7], cycles=[0, 2, 4, 6, 8, 10])


def acquired(tmp_path, strengths=(0, 30, 60)):
    scan = CentralBosonAGPDisorderScan.from_pattern(list(strengths), [1., -3.], **SECTOR,
                                                    calibration=CONFUSION)
    scan.acquire(FakeRunner(tmp_path / "data"))
    return scan


def test_subtract_background_recovers_a_known_connected_part():
    """Synthetic truth: oscillations well above 1/T on a slow exponential background."""
    t = np.arange(0, 245, 0.31)
    oscillating = (0.05 * np.cos(2 * np.pi * 0.061 * t) + 0.03 * np.cos(2 * np.pi * 0.113 * t)) \
        * np.exp(-t / 300)
    background = 0.4 * np.exp(-t / 450)
    connected, fit = subtract_background(t, oscillating + background, start_us=30.)
    assert fit["lifetime_us"] == pytest.approx(450, rel=0.03)
    assert fit["amplitude"] == pytest.approx(0.4, rel=0.03)
    weights = agp_kernel_weights(t, 0.010)
    # The background is most of the raw integral; after the subtraction it is gone to 3 %.
    assert abs(weights @ background) > 3 * abs(weights @ oscillating)
    assert weights @ connected == pytest.approx(weights @ oscillating,
                                                abs=0.03 * abs(weights @ background))


def test_scan_collects_the_background_subtracted_agp(tmp_path):
    scan = acquired(tmp_path)
    data = scan.analyze(mu=[5., 10.], background_start_us=0.)
    for key in ("agp_connected", "agp_connected_err"):
        assert data[key].shape == (3, 2)
        np.testing.assert_allclose(data[key], [part.data[key] for part in scan.children])
    np.testing.assert_allclose(data["agp_connected_err"], data["agp_err"])
    assert data["background_rate_per_us"].shape == (3,)
    assert "agp_connected" not in scan.analyze(mu=[5., 10.])  # only when asked for


def test_list_scans_reads_manifests_and_flags_a_subset(tmp_path):
    scan = acquired(tmp_path)
    scan.analyze(mu=[10.])
    full = scan.save("full", notes="full")
    subset = CentralBosonAGPDisorderScan.from_children(scan.children[1:2])
    subset.analyze(mu=[10.])
    part = subset.save("one_w", notes="one W")
    table = list_scans(tmp_path).set_index("notes")
    stem = agp.scan_stem
    assert table.loc["one W", "part_of"] == stem(full)
    assert table.loc["full", "part_of"] == ""
    assert table.loc["full", "W_khz"] == [0, 30, 60]
    assert table.loc["one W", "jobs"] == len(scan.children[1].job_ids)
    assert stem(part) in table["scan"].tolist()


def test_load_scan_reads_a_saved_manifest(tmp_path):
    scan = acquired(tmp_path)
    scan.analyze(mu=[10.])
    loaded = load_scan(scan.save("loaded"))
    assert [p.disorder_khz for p in loaded.children] == [0, 30, 60]
    assert loaded.data == {}  # loaded, not analyzed


def test_local_path_reroots_a_measurement_pc_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIMODE_DATA_ROOT", str(tmp_path))
    target = tmp_path / "some_expt" / "assembled_data" / "scan.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("x")
    assert local_path(r"C:\experiments\some_expt\assembled_data\scan.yaml") == target
    assert local_path(target) == target
    with pytest.raises(FileNotFoundError):
        local_path(r"C:\experiments\some_expt\missing.yaml")


def test_display_scans_needs_each_cutoff_and_the_estimator(tmp_path):
    scan = acquired(tmp_path)
    scan.analyze(mu=[5., 10.])
    with pytest.raises(ValueError, match="no agp_connected"):
        display_scans({"a": scan}, mu=[10.])
    with pytest.raises(ValueError, match="not analyzed at mu = 7"):
        display_scans({"a": scan}, mu=[7.], estimator="agp")
    scan.analyze(mu=[5., 10.], background_start_us=0.)
    fig, axes = display_scans({"a": scan, "b": scan}, mu=[5., 10.])
    assert len(axes) == 2 and len(axes[0].get_legend().get_texts()) == 2


def test_every_registered_scan_names_a_manifest_and_a_label():
    with open(agp.SCAN_REGISTRY, encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)["scans"]
    assert entries
    for key, entry in entries.items():
        assert agp.SCAN_CLASS in entry["manifest"], key
        assert entry["label"], key
        assert entry.get("superseded", next(iter(entries))) in entries, key


def _nyquist_khz(scan):
    return 1000 / (2 * np.median(np.diff(scan.data["time_us"])))


def test_spectrum_range_follows_the_closed_system_frequencies(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30))
    scan.analyze()
    model = scan.model_spectra_khz()
    assert len(model) == 2 and all(len(f) == len(w) > 0 for f, w in model)
    top = max(f[w > 1e-3 * w.max()].max() for f, w in model)
    fig, axes = scan.display_spectrum()
    assert axes[-1].get_xlim()[1] == pytest.approx(
        min(_nyquist_khz(scan), CentralBosonAGPDisorderScan.SPECTRUM_MARGIN * top))
    fig, axes = scan.display_spectrum(max_khz=50.)
    assert axes[-1].get_xlim()[1] == 50.
    matplotlib.pyplot.close("all")


def test_spectrum_marks_closed_system_weight_above_nyquist(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30000))   # 30 MHz: far above Nyquist
    scan.analyze()
    nyquist = _nyquist_khz(scan)
    folded = [w[f > nyquist].sum() / w.sum() for f, w in scan.model_spectra_khz()]
    assert folded[0] < 1e-3 < folded[1]
    fig, axes = scan.display_spectrum()
    marked = ["folds back" in " ".join(t.get_text() for t in ax.texts) for ax in axes]
    assert marked == [False, True]
    assert axes[-1].get_xlim()[1] == pytest.approx(nyquist)
    matplotlib.pyplot.close("all")


def test_save_needs_a_name_and_never_overwrites(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30))
    scan.analyze(mu=[10.])
    with pytest.raises(TypeError):
        scan.save()
    with pytest.raises(ValueError, match="letters"):
        scan.save("bad name")
    manifest = scan.save("s7_test")
    assert manifest.name == "CentralBosonAGPDisorderScan_s7_test.yaml"
    assert agp.scan_stem(manifest) == "s7_test"
    names = sorted(path.name for path in manifest.parent.glob("CentralBosonAGPExperiment_*.yaml"))
    assert names == ["CentralBosonAGPExperiment_s7_test_W0kHz.yaml",
                     "CentralBosonAGPExperiment_s7_test_W30kHz.yaml"]
    again = CentralBosonAGPDisorderScan.from_children(scan.children)
    again.analyze(mu=[10.])
    with pytest.raises(FileExistsError):
        again.save("s7_test")


def test_a_taken_scan_name_writes_no_parts(tmp_path):
    first = acquired(tmp_path, strengths=(0,))
    first.analyze(mu=[10.])
    first.save("taken")
    second = acquired(tmp_path / "other", strengths=(30,))
    second.analyze(mu=[10.])
    with pytest.raises(FileExistsError):
        second.save("taken", directory=tmp_path / "assembled_data")
    assert not list((tmp_path / "assembled_data").glob("*_taken_W30kHz*"))


def test_part_names_and_old_stems():
    assert agp.part_name("s7", 120.) == "s7_W120kHz"
    assert agp.part_name("s7", 0.5) == "s7_W0p5kHz"
    assert agp.scan_stem("x/261030_123324_CentralBosonAGPDisorderScan.yaml") == "261030_123324"
    assert agp.scan_stem("x/CentralBosonAGPDisorderScan_s7_high_w.yaml") == "s7_high_w"

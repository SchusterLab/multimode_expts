"""AGP data sets: background-subtracted estimator, list/load/compare disorder scans."""
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from experiments.qsim import agp  # noqa: E402
from experiments.qsim.agp import (  # noqa: E402
    CentralBosonAGPExperiment,
    CentralBosonAGPDisorderScan,
    display_scans,
    list_scans,
    load_scan,
)
from fitting.qsim.agp import agp_kernel_weights, subtract_background  # noqa: E402
from tests.population_jobs import CONFUSION, FakeRunner  # noqa: E402

SECTOR = dict(photon_sector_number=2, direction=7, swap_stors=[4, 7], cycles=[0, 2, 4, 6, 8, 10])


def acquired(tmp_path, strengths=(0, 30, 60), name="acquired"):
    scan = CentralBosonAGPDisorderScan.from_pattern(list(strengths), [1., -3.], **SECTOR,
                                                    calibration=CONFUSION)
    scan.acquire(FakeRunner(tmp_path / "data"), name)
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
    full = scan.save("full", notes="full")  # the same parts as "acquired"
    subset = CentralBosonAGPDisorderScan.from_children(scan.children[1:2])
    subset.analyze(mu=[10.])
    part = subset.save("one_w", notes="one W")
    table = list_scans(tmp_path).set_index("notes")
    stem = agp.scan_stem
    assert table.loc["one W", "part_of"].split(", ") == ["acquired", stem(full)]
    assert table.loc["full", "part_of"] == ""
    assert table.loc["full", "W_khz"] == [0, 30, 60]
    assert table.loc["one W", "jobs"] == len(scan.children[1].job_ids)
    assert stem(part) in table["scan"].tolist()


def test_load_scan_with_jobs_reads_the_manifest(tmp_path):
    scan = acquired(tmp_path)
    scan.analyze(mu=[10.])
    loaded = load_scan(scan.save("loaded"), jobs=True)
    assert [p.disorder_khz for p in loaded.children] == [0, 30, 60]
    assert all(part.children for part in loaded.children)
    assert loaded.data == {}  # loaded, not analyzed


SETTINGS = [dict(mu=[5., 10.]), dict(mu=[2., 10., 20.], background_start_us=0., t1_us=500.,
                                     t1_err_us=50.), dict(mu=[10.], kerr_mhz=0.)]


@pytest.mark.parametrize("settings", SETTINGS)
def test_a_scan_from_its_h5_alone_analyzes_like_the_jobs(tmp_path, settings):
    """The saved inputs give every curve the job files give, at settings other than the saved ones."""
    scan = acquired(tmp_path)
    scan.analyze(mu=[10.])
    manifest = scan.save("frozen")
    frozen = load_scan(manifest)
    assert all(not part.children for part in frozen.children)  # no job file was opened
    reference = load_scan(manifest, jobs=True)
    expected = reference.analyze(**settings)
    data = frozen.analyze(**settings)
    assert set(data) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(data[key], dtype=float),
                                   np.asarray(expected[key], dtype=float), err_msg=key)
    np.testing.assert_allclose(frozen.theory_agp([0., 45.], mu=[10.]),
                               reference.theory_agp([0., 45.], mu=[10.]))
    assert frozen.summary().splitlines()[1:] == reference.summary().splitlines()[1:]
    for part, other in zip(frozen.children, reference.children):
        assert part.job_ids == other.job_ids


def test_a_frozen_scan_displays_every_stage(tmp_path):
    scan = acquired(tmp_path)
    frozen = load_scan(scan.save("shown"))
    frozen.analyze(mu=[5., 10.], background_start_us=0.)
    assert len(frozen.display()) == 6


def test_a_frozen_scan_cannot_be_saved_or_rewritten(tmp_path):
    scan = acquired(tmp_path)
    frozen = load_scan(scan.save("once"))
    with pytest.raises(ValueError, match="without jobs"):
        frozen.save("twice")
    with pytest.raises(ValueError, match="needs the jobs"):
        frozen.rewrite_h5()


def test_rewrite_h5_adds_the_inputs_to_an_old_file(tmp_path):
    """A scan HDF5 from before the inputs were saved: arrays only, until rewrite_h5."""
    import h5py

    scan = acquired(tmp_path)
    manifest = scan.save("old")
    h5_path = manifest.with_suffix(".h5")
    with h5py.File(h5_path, "a") as handle:  # make it look like a file from before 2026-11-10
        for key in [key for key in handle if key.startswith("inputs_")]:
            del handle[key]
        del handle.attrs["parameters"]
    old = load_scan(manifest)
    assert all(part.children for part in old.children)  # fell back to the jobs
    assert not CentralBosonAGPDisorderScan.from_h5file(h5_path).children
    old.analyze(mu=[10.])
    before = h5_path.read_bytes()
    retired = tmp_path / "retired"
    assert old.rewrite_h5(retired_dir=retired) == h5_path
    assert not any(part.children for part in load_scan(manifest).children)
    assert (retired / h5_path.name).read_bytes() == before  # moved, not deleted or changed
    assert not list(h5_path.parent.glob("*.new"))
    with pytest.raises(FileExistsError):  # a retired file is never overwritten
        old.rewrite_h5(retired_dir=retired)
    assert (retired / h5_path.name).read_bytes() == before


def test_rewrite_h5_retires_beside_assembled_data_by_default(tmp_path):
    scan = acquired(tmp_path)
    manifest = scan.save("dated")
    loaded = load_scan(manifest, jobs=True)
    loaded.analyze(mu=[10.])
    loaded.rewrite_h5()
    (folder,) = manifest.parent.parent.glob("assembled_data_old_*")
    assert (folder / manifest.with_suffix(".h5").name).is_file()


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
    axes = scan.display_spectrum().axes
    assert axes[1].get_xlim()[1] == pytest.approx(
        min(_nyquist_khz(scan), CentralBosonAGPDisorderScan.SPECTRUM_MARGIN * top))
    axes = scan.display_spectrum(max_khz=50.).axes
    assert axes[1].get_xlim()[1] == 50.
    matplotlib.pyplot.close("all")


def test_spectrum_marks_closed_system_weight_above_nyquist(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30000))   # 30 MHz: far above Nyquist
    scan.analyze()
    nyquist = _nyquist_khz(scan)
    folded = [w[f > nyquist].sum() / w.sum() for f, w in scan.model_spectra_khz()]
    assert folded[0] < 1e-3 < folded[1]
    axes = scan.display_spectrum().axes[:2]
    marked = ["folds back" in " ".join(t.get_text() for t in ax.texts) for ax in axes]
    assert marked == [False, True]
    assert axes[-1].get_xlim()[1] == pytest.approx(nyquist)
    matplotlib.pyplot.close("all")


def test_save_needs_a_name_and_never_overwrites(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30), name="s7_test")
    with pytest.raises(TypeError):
        scan.save()
    with pytest.raises(ValueError, match="letters"):
        scan.save("bad name")
    manifest = scan.manifest_path
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
    acquired(tmp_path, strengths=(0,), name="taken")
    second = CentralBosonAGPDisorderScan.from_pattern([30], [1., -3.], **SECTOR, calibration=CONFUSION)
    with pytest.raises(FileExistsError):
        second.acquire(FakeRunner(tmp_path / "data"), "taken")
    assert not list((tmp_path / "assembled_data").glob("*_taken_W30kHz*"))


class StopsAtSecondW(FakeRunner):
    def execute(self, overrides, batch_size=10, **kwargs):
        if len(self.calls) == 1:
            raise KeyboardInterrupt
        return super().execute(overrides, batch_size=batch_size, **kwargs)


def test_a_stopped_run_keeps_its_finished_w(tmp_path):
    scan = CentralBosonAGPDisorderScan.from_pattern([0, 30, 60], [1., -3.], **SECTOR,
                                                    calibration=CONFUSION)
    with pytest.raises(KeyboardInterrupt):
        scan.acquire(StopsAtSecondW(tmp_path / "data"), "stopped", analyze_kwargs=dict(mu=[10.]))
    folder = tmp_path / "assembled_data"
    assert [p.name for p in folder.glob("*.yaml")] == ["CentralBosonAGPExperiment_stopped_W0kHz.yaml"]
    partial = CentralBosonAGPDisorderScan.from_saved_parts("stopped", folder)
    assert [part.disorder_khz for part in partial.children] == [0]
    partial.analyze(mu=[10.])
    assert partial.save("stopped").name == "CentralBosonAGPDisorderScan_stopped.yaml"


def test_part_names_and_old_stems():
    assert agp.part_name("s7", 120.) == "s7_W120kHz"
    assert agp.part_name("s7", 0.5) == "s7_W0p5kHz"
    assert agp.scan_stem("x/261030_123324_CentralBosonAGPDisorderScan.yaml") == "261030_123324"
    assert agp.scan_stem("x/CentralBosonAGPDisorderScan_s7_high_w.yaml") == "s7_high_w"


def test_each_w_has_its_curves_on_shared_axes(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30), name="curves")
    data = scan.analyze(background_start_us=0.)
    part = scan.children[1]
    np.testing.assert_allclose(data["agp_mu"], CentralBosonAGPExperiment.DEFAULT_MU_KHZ)
    assert {2., 5., 10., 20., 50.} <= set(np.round(data["agp_mu"], 9))
    assert data["agp"].shape == (2, len(data["agp_mu"]))
    assert data["spectrum"].shape == (2, len(data["frequency_khz"]))
    np.testing.assert_allclose(data["spectrum"][1], part.data["spectrum"])
    # The stored S is what the plot computes from C_1 with the same settings.
    axes = scan.display_spectrum().axes
    computed = scan.display_spectrum(window="hann", raw=False, background_start_us=0.).axes
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), computed[1].lines[0].get_ydata())
    assert list(scan.shown_cutoffs()) == [5., 10., 20.]
    assert scan.display_agp_mu().axes[0].get_title() == "S7: agp_connected"
    fitted = [line.get_label() for line in scan.display_traces().axes[1].lines]
    assert fitted == ["C_1 (corrected)", "closed system", r"fit $A e^{-\Gamma t}$"]
    matplotlib.pyplot.close("all")


def test_parts_with_different_couplings_share_the_default_axis(tmp_path):
    scan = acquired(tmp_path, strengths=(0, 30), name="couplings")
    scan.analyze()
    first, second = scan.children
    for child in second.children:  # other recorded couplings for the second W only
        params = child.recorded_derived_params()
        params["couplings_MHz"] = [2 * c for c in params["couplings_MHz"]]
        child.recorded_derived_params = (lambda p=params: p)
    data = scan.analyze()
    assert data["mu_gld_khz"][1] == pytest.approx(2 * data["mu_gld_khz"][0])
    assert data["agp"].shape == (2, len(CentralBosonAGPExperiment.DEFAULT_MU_KHZ))


def test_agp_table_has_one_row_per_scan_w_and_cutoff(tmp_path):
    scan = acquired(tmp_path)
    scan.analyze(mu=[5., 10.], background_start_us=0.)
    table = agp.agp_table({"a": scan, "b": scan}, mu=[10.])
    assert len(table) == 2 * 3 and set(table["probe"]) == {"S7"}
    row = table[(table.scan == "a") & (table.W == 30)].iloc[0]
    assert row.value == pytest.approx(scan.data["agp_connected"][1, 1])
    assert row.theory == pytest.approx(scan.data["agp_theory"][1, 1])
    assert row.free == pytest.approx(scan.theory_agp([30.], mu=[10.], kerr_mhz=0.)[0, 0])
    assert row.total >= row.noise
    with pytest.raises(ValueError, match="not analyzed at mu = 7"):
        agp.agp_table({"a": scan}, mu=[7.])

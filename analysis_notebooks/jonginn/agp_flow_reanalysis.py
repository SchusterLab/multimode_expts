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
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

# %% [markdown]
# # AGP flow reanalysis
#
# Read completed **per-W assembled results**, each containing the full initial
# Fock-state measurement. These results have no independent JOB ID. Select them
# using their member JOB IDs (default), or their creation dates and saved names.
# Individual Fock JOB data are not reassembled. One raw metadata record per W
# supplies the recorded couplings and Kerr; raw IQ and shot arrays are not loaded.
#
# Run in order: settings -> load -> integrate -> AGP -> spectrum -> flow.
# After editing MU_KHZ, rerun from "Integrate at the selected cutoffs" downward.
# Measured values use the saved calibrated correlations and the finite measured
# time window. No decay correction, background subtraction or tail completion
# is applied. Independent closed-system theory is only a comparison.

# %%
from fnmatch import fnmatch
from pathlib import Path
import json

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from slab import AttrDict

from experiments.assembled_data import read_manifest
from experiments.qsim.agp_sequential_ensemble import (
    CentralBosonAGPSequentialEnsembleDisorderScan,
    CentralBosonAGPSequentialEnsembleExperiment,
)
from fitting.qsim.agp import (
    agp_kernel_weights, cosine_transform_weights, spectrum_frequencies_khz,
)
from fitting.qsim.agp_metric import _eigensystem, _integrate_correlations
from theory.qsim.agp_ensemble import ensemble_metric_prediction

# %% [markdown]
# ## Settings
#
# BASE_DIR is the experiment folder containing assembled_data/ and data/.
# JOB ID endpoints include the date and are inclusive. All member JOBs of a W
# must fall inside the selected ranges; a partially selected W is omitted.
# With SELECT_BY_JOB_ID=False, DATE_RANGE (inclusive dates) and EXPERIMENT_NAME
# are used instead. The name is the saved measurement name; '*' is a wildcard.
# If several saved copies reference identical JOBs, use the newest created copy.

# %%
SAVE_ASSEMBLED_DATA = False
BASE_DIR = Path(r"C:\experiments\agp_sequential_disorder_ensemble")

SELECT_BY_JOB_ID = True
JOB_ID_RANGES = [("JOB-20261116-00273", "JOB-20261118-00118")]
DATE_RANGE = ("2026-11-16", "2026-11-18")
EXPERIMENT_NAME = "agp_sequential_disorder_ensemble*"

MU_KHZ = [5.915, 10., 20., 30., 50, 70]  # Cyclic frequency: tau_us = 1000 / (2*pi*mu_kHz).
ASSEMBLED_NAME = "agp_flow_reanalysis_20261118"  # Choose a new name for each save.
SPECTRUM_MAX_KHZ = 150.
SPECTRUM_WINDOW = "hann"

# %% [markdown]
# ## Load completed per-W results
#
# Each W keeps its own calibration and original JOB provenance. Different Kerr,
# couplings, time grids or disorder ensembles form separate scan series.
# Ordinary from_manifest() would reopen every raw child; this loader reads the
# saved correlators instead. Do not call analyze() on these lightweight views.

# %%
def selected_assembled_results():
    """Select completed per-W YAML/HDF5 pairs, not whole-scan manifests."""
    prefix = CentralBosonAGPSequentialEnsembleExperiment.__name__ + "_"
    selected_by_jobs = {}
    for path in sorted((BASE_DIR / "assembled_data").glob(prefix + "*.yaml")):
        if not path.with_suffix(".h5").is_file():
            continue
        manifest = read_manifest(path)
        job_ids = manifest["job_ids"]
        if SELECT_BY_JOB_ID:
            include = bool(job_ids) and all(
                any(first <= job_id <= last for first, last in JOB_ID_RANGES)
                for job_id in job_ids)
        else:
            created_date = manifest["created"][:10]
            saved_name = path.stem[len(prefix):]
            include = (DATE_RANGE[0] <= created_date <= DATE_RANGE[1]
                       and fnmatch(saved_name, EXPERIMENT_NAME))
        if include:
            identity = tuple(sorted(job_ids))
            previous = selected_by_jobs.get(identity)
            if previous is None or manifest["created"] > previous[1]["created"]:
                selected_by_jobs[identity] = (path, manifest)
            if previous is not None:
                print(f"Same member JOBs: using {selected_by_jobs[identity][0].name}")
    return list(selected_by_jobs.values())


def validate_assembled_selection(selected_results):
    """Stop before plotting if the directory or selection matched no results."""
    if not selected_results:
        raise FileNotFoundError("No completed per-W results matched BASE_DIR and the selection settings.")


def load_assembled_scans(selected_results):
    """Hydrate saved parts and group compatible parts without loading shots."""
    grouped_parts = {}
    for path, manifest in selected_results:
        part = CentralBosonAGPSequentialEnsembleExperiment(
            **manifest["parameters"], notes=manifest.get("notes", ""))
        with h5py.File(path.with_suffix(".h5"), "r") as handle:
            part.data = AttrDict({name: handle[name][()] for name in handle
                                  if name not in ("readout_labels", "herald_keep")})
            part.analysis = json.loads(handle.attrs.get("analysis", "{}"))
        # Read only metadata from one constituent JOB, never its data arrays.
        with h5py.File(manifest["raw_files"][0], "r") as handle:
            derived = json.loads(handle.attrs["derived_params"])
            config = json.loads(handle.attrs["config"])
        part.data["couplings_mhz"] = np.asarray(derived["couplings_MHz"])[
            np.asarray(part.swap_stors) - 1]
        part.data["recorded_kerr_mhz"] = np.asarray(
            config["device"]["manipulate"]["kerr"]).reshape(-1)[0]
        part.manifest_path, part.job_ids = path, manifest["job_ids"]
        shared_settings = [getattr(part, name)
                           for name in CentralBosonAGPSequentialEnsembleDisorderScan.SHARED]
        signature = (float(part.data["recorded_kerr_mhz"]),
                     tuple(part.data["couplings_mhz"]), float(derived["floquet_cycle_us"]),
                     json.dumps(shared_settings))
        grouped_parts.setdefault(signature, []).append(part)
        print(f"K = {1000 * signature[0]:.3f}, W = {part.disorder_khz:g} kHz | "
              f"{min(part.job_ids)} .. {max(part.job_ids)}")
    scans = []
    for _, compatible_parts in sorted(grouped_parts.items()):
        # Genuine repeated W measurements have different JOBs: keep separate series.
        separate_series = []
        for part in sorted(compatible_parts, key=lambda point: (point.disorder_khz, min(point.job_ids))):
            for series_parts in separate_series:
                if part.disorder_khz not in [point.disorder_khz for point in series_parts]:
                    series_parts.append(part)
                    break
            else:
                separate_series.append([part])
        for series_parts in separate_series:
            scans.append(CentralBosonAGPSequentialEnsembleDisorderScan.from_children(
                series_parts, job_ids=[job_id for part in series_parts for job_id in part.job_ids]))
    return scans


selected_results = selected_assembled_results()
validate_assembled_selection(selected_results)
scans = load_assembled_scans(selected_results)
parts = [part for scan in scans for part in scan.children]
print(f"Selected {len(parts)} completed W points in {len(scans)} scan series.")

# %% [markdown]
# ## Integrate at the selected cutoffs
#
# C_metric(t) is the already calibrated, realization-weighted K/W correlator.
# The existing kernel weights integrate a piecewise-linear trace over its
# measured window. Mu is supplied in kHz and converted to MHz at the kernel API.
# Physical metric entries have units kHz^-2; plots use the existing g^2 scaling.
#
# This cell does not repeat bootstrap. New-mu plots show central estimates;
# errors from an older cutoff are not reused. The independent theory uses the
# recorded Kerr, couplings and original ordered disorder vectors, including W=0.

# %%
def integrate_saved_correlations(part, cutoffs_khz):
    """Recompute kernel integrals and a separate spectral-theory comparison."""
    data = part.data
    # Discard old-cutoff theory arrays before adding the new independent comparison.
    for name in list(data):
        if "theory" in name:
            data.pop(name)
    weights = np.array([agp_kernel_weights(data["time_us"], cutoff / 1000)
                        for cutoff in cutoffs_khz])
    metric = _integrate_correlations(weights, data["C_metric"])
    local_metric = _integrate_correlations(weights, data["C_metric_local"])
    coupling_squared = (1000 * float(data["coupling_mhz"]))**2
    local_agp = np.diagonal(local_metric, axis1=-2, axis2=-1).T * coupling_squared
    eigenvalues, eigenvectors = _eigensystem(metric)
    data.update(agp_mu=np.asarray(cutoffs_khz), agp=local_agp,
                agp_err=np.full(local_agp.shape, np.nan),
                agp_metric=metric, agp_metric_err=np.full(metric.shape, np.nan),
                agp_metric_scaled=metric * coupling_squared,
                agp_metric_scaled_err=np.full(metric.shape, np.nan),
                agp_metric_local=local_metric,
                agp_metric_local_err=np.full(local_metric.shape, np.nan),
                agp_metric_eigenvalues=eigenvalues, agp_metric_eigenvectors=eigenvectors)
    theory_directions = part.directions.copy()
    if "agp_metric_representative_storage" in data:
        theory_directions += [storage for storage in part.swap_stors
                              if storage not in theory_directions]
    # Endpoints suffice for spectral sums and analytic finite-window theory.
    prediction = ensemble_metric_prediction(
        part.photon_sector_number, data["disorder_detunings_mhz"],
        data["couplings_mhz"], float(data["recorded_kerr_mhz"]), theory_directions,
        part.swap_stors, data["disorder_patterns"], [0., float(data["time_us"][-1])],
        cutoffs_khz)
    data.update(agp_metric_theory=prediction["agp_metric_theory"],
                agp_metric_theory_window=prediction["agp_metric_theory_window"],
                agp_theory=prediction["agp_theory"][:len(part.directions)],
                agp_theory_window=prediction["agp_theory_window"][:len(part.directions)],
                theory_kerr_mhz=data["recorded_kerr_mhz"])
    part.analysis = dict(mu_khz=list(cutoffs_khz), bootstrap_samples=0,
                         estimator="Saved calibrated C_metric times finite-window kernel",
                         uncertainty="Central estimates; new-cutoff errors not recomputed",
                         theory="Independent spectral sum; no correction",
                         kerr_mhz=float(data["recorded_kerr_mhz"]))


for part in parts:
    integrate_saved_correlations(part, MU_KHZ)
    print(f"W = {part.disorder_khz:g} kHz: kernel and independent theory complete")
for scan in scans:
    scan._stack_parts()
    scan.data["couplings_mhz"] = np.array([part.data["couplings_mhz"] for part in scan.children])
    scan.data["recorded_kerr_mhz"] = np.array([
        part.data["recorded_kerr_mhz"] for part in scan.children])

# %%
# Optional: save a new combined YAML/HDF5 pair, referring to original per-W files.
# Original per-W files are never overwritten. Existing output names are protected.
if SAVE_ASSEMBLED_DATA:
    for series_index, scan in enumerate(scans, start=1):
        print(scan.save(f"{ASSEMBLED_NAME}_series{series_index}",
                        directory=BASE_DIR / "assembled_data"))

# %% [markdown]
# ## 1. AGP versus disorder
#
# Select mu below. Measured points are direct kernel integrals. Gray crosses
# show independent same-window theory; black dashes show infinite-time theory.
# No new-mu error bars are inferred from the saved errors.

# %%
def display_agp(mu_khz):
    """Show the measured KK, KW and WW components for every scan series."""
    cutoff_index = MU_KHZ.index(mu_khz)
    for scan in scans:
        data = scan.data
        scale = (1000 * data["coupling_mhz"])**2
        kerr_khz = 1000 * float(data["recorded_kerr_mhz"][0])
        with plt.style.context(["default", {"font.size": 14}]):
            figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
            for axis, (first, second, label) in zip(axes, [(0, 0, "KK"), (0, 1, "KW"), (1, 1, "WW")]):
                axis.plot(data["disorder_khz"], scale * data["agp_metric_theory_window"][
                    :, cutoff_index, first, second], "x:", color="gray", label="closed system, same window")
                axis.plot(data["disorder_khz"], scale * data["agp_metric_theory"][
                    :, cutoff_index, first, second], "k--", label="closed system, infinite time")
                axis.plot(data["disorder_khz"], data["agp_metric_scaled"][
                    :, cutoff_index, first, second], "o-", label="measured")
                axis.set(xlabel="Disorder RMS W (kHz)", ylabel=rf"$g^2G_{{{label}}}$")
            axes[0].legend(fontsize=8)
            figure.suptitle(rf"$K={kerr_khz:.3f}$ kHz, $\mu={mu_khz:g}$ kHz")
            plt.show()


display(widgets.interactive(display_agp, mu_khz=widgets.SelectionSlider(
    options=MU_KHZ, value=MU_KHZ[-1], description="mu (kHz)")))

# %% [markdown]
# ## 2. Spectral function
#
# These are one-sided cosine transforms of the same measured C_metric(t),
# showing KK, KW and WW. The chosen window is explicit; DC and signed cross
# weights are retained. The dotted filter has unit peak and changes with mu.
# Selecting a point or mu does not reload any file or repeat the theory calculation.

# %%
def display_spectrum(point_index, mu_khz):
    """Transform the measured K/W correlator of one completed W point."""
    part = parts[point_index]
    data = part.data
    frequencies = spectrum_frequencies_khz(data["time_us"])
    frequencies = frequencies[frequencies <= SPECTRUM_MAX_KHZ]
    transform = cosine_transform_weights(
        data["time_us"], frequencies / 1000, SPECTRUM_WINDOW) / 1000
    filter_shape = 4 * mu_khz**2 * frequencies**2 / (frequencies**2 + mu_khz**2)**2
    with plt.style.context(["default", {"font.size": 14}]):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for axis, (first, second, label) in zip(axes, [(0, 0, "KK"), (0, 1, "KW"), (1, 1, "WW")]):
            spectrum = transform @ data["C_metric"][:, first, second]
            axis.plot(frequencies, spectrum, label="measured")
            filter_axis = axis.twinx()
            filter_axis.plot(frequencies, filter_shape, "k:", linewidth=1, label="normalized filter")
            filter_axis.set(ylim=(0, 1.05), yticks=[])
            axis.set(xlabel="f (kHz)", ylabel=rf"$S_{{{label}}}$ (1/kHz)")
        axes[0].legend(fontsize=8)
        figure.suptitle(rf"$K={1000 * float(data['recorded_kerr_mhz']):.3f}$ kHz, "
                       rf"$W={part.disorder_khz:g}$ kHz, $\mu={mu_khz:g}$ kHz; {SPECTRUM_WINDOW} window")
        plt.show()


point_options = [(f"K={1000 * float(part.data['recorded_kerr_mhz']):.3f}, "
                  f"W={part.disorder_khz:g} kHz", index) for index, part in enumerate(parts)]
display(widgets.interactive(
    display_spectrum, point_index=widgets.Dropdown(options=point_options, description="Point"),
    mu_khz=widgets.SelectionSlider(options=MU_KHZ, value=MU_KHZ[-1], description="mu (kHz)")))

# %% [markdown]
# ## 3. AGP flow directions
#
# Segments follow the eigenvector of the smaller eigenvalue of the measured
# 2x2 K/W metric. Their length is only for display; each axis is unoriented.
# Color is the measured anisotropy lambda_max/lambda_min, on a linear 1--30 scale.
# Nonpositive minimum eigenvalues are retained and shown gray. Black dashed
# segments show independent theory at the same measured parameter points.
# Sparse points determine local directions, not continuous measured streamlines.

# %%
def display_flow(mu_khz):
    """Compare local minimum-metric directions without interpolating measured paths."""
    cutoff_index = MU_KHZ.index(mu_khz)
    color_scale, color_map = plt.Normalize(1., 30., clip=True), plt.get_cmap("plasma_r")
    with plt.style.context(["default", {"font.size": 14}]):
        figure, (flow_axis, angle_axis) = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
        for series_index, scan in enumerate(scans, start=1):
            data = scan.data
            eigenvalues, eigenvectors = _eigensystem(data["agp_metric"][:, cutoff_index])
            _, theory_vectors = _eigensystem(data["agp_metric_theory"][:, cutoff_index])
            directions, theory_directions = eigenvectors[:, :, 0], theory_vectors[:, :, 0]
            angles = (np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) + 90) % 180 - 90
            theory_angles = (np.degrees(np.arctan2(
                theory_directions[:, 1], theory_directions[:, 0])) + 90) % 180 - 90
            anisotropy = np.full(len(eigenvalues), np.nan)
            np.divide(eigenvalues[:, 1], eigenvalues[:, 0], out=anisotropy, where=eigenvalues[:, 0] > 0)
            for point_index, strength in enumerate(data["disorder_khz"]):
                center = np.array([1000 * data["recorded_kerr_mhz"][point_index], strength])
                measured_segment = center + np.array([[-3.], [3.]]) * directions[point_index]
                theory_segment = center + np.array([[-4.5], [4.5]]) * theory_directions[point_index]
                color = color_map(color_scale(anisotropy[point_index])) if np.isfinite(anisotropy[point_index]) else "0.6"
                flow_axis.plot(theory_segment[:, 0], theory_segment[:, 1], "k--", linewidth=1.5)
                flow_axis.plot(measured_segment[:, 0], measured_segment[:, 1], color=color, linewidth=3)
                angle_axis.scatter(strength, angles[point_index], color=color, edgecolor="0.25", zorder=3)
            theory_angles[1:][np.abs(np.diff(theory_angles)) > 90] = np.nan
            kerr_khz = 1000 * float(data["recorded_kerr_mhz"][0])
            angle_axis.plot(data["disorder_khz"], theory_angles, "k--",
                            label=f"theory: K={kerr_khz:.3f} kHz, series {series_index}")
        flow_axis.set(xlabel="Kerr K (kHz)", ylabel="Disorder RMS W (kHz)")
        flow_axis.set_aspect("equal", adjustable="datalim")
        angle_axis.set(xlabel="Disorder RMS W (kHz)", ylabel="Minimum-direction angle (deg)", ylim=(-90, 90))
        angle_axis.legend(fontsize=8)
        colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=color_scale, cmap=color_map),
                                   ax=[flow_axis, angle_axis], ticks=[1, 5, 10, 20, 30])
        colorbar.ax.set_yticklabels(["1", "5", "10", "20", ">=30"])
        colorbar.set_label(r"Anisotropy $\lambda_{\max}/\lambda_{\min}$")
        figure.suptitle(rf"$\mu={mu_khz:g}$ kHz; central estimates")
        plt.show()


display(widgets.interactive(display_flow, mu_khz=widgets.SelectionSlider(
    options=MU_KHZ, value=MU_KHZ[-1], description="mu (kHz)")))

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}

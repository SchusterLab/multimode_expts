"""Tables and diagnostic views for decay_investigation.ipynb."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from decay_fitting import fit_complex_decay


def fit_trace_collection(traces, *, fit_min_us=0.0, fit_max_us=None, progress=True):
    """Fit each occupation independently; keep unsuccessful results in the table."""
    rows, details = [], {}
    for index, trace in enumerate(traces):
        if progress and index % 100 == 0:
            print(f"Fitting occupation {index + 1}/{len(traces)} (both envelopes)", flush=True)
        t = np.asarray(trace["time_us"])
        mask = t >= fit_min_us
        if fit_max_us is not None:
            mask &= t <= fit_max_us
        metadata = {key: value for key, value in trace.items()
                    if key not in ("time_us", "A", "coherent_A", "cycles")}
        for mode, number in zip(trace["mode_labels"], trace["occupation"]):
            metadata[f"n_{mode}"] = int(number)
        metadata["pure_central"] = not any(trace["occupation"][1:])
        for envelope, beta in (("exponential", 1.0), ("gaussian", 2.0)):
            row = dict(metadata, envelope=envelope, fit_min_us=fit_min_us,
                       fit_max_us=fit_max_us)
            try:
                result = fit_complex_decay(
                    t[mask], np.asarray(trace["A"])[mask],
                    np.asarray(trace["coherent_A"])[mask], beta=beta,
                )
                details[(trace["trace_id"], envelope)] = dict(
                    result=result, mask=mask, trace=trace)
                row.update({key: value for key, value in result.items()
                            if np.isscalar(value) or value is None})
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row.update(tau_us=np.nan, tau_estimate_us=np.nan, aicc=np.nan,
                           tau_ci_low_us=np.nan, tau_ci_high_us=np.nan,
                           accepted=False, status="fit_error", message=str(error))
                details[(trace["trace_id"], envelope)] = dict(
                    result=None, mask=mask, trace=trace, error=str(error))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["trace_id", "dataset_id", "envelope", "tau_us",
                                     "accepted", "status", "n_central", "K_kHz"]), details
    return pd.DataFrame(rows), details


def select_results(results, *, envelope="exponential", accepted_only=True,
                   central_photons=None, pure_central=False, dataset_ids=None):
    selected = results.loc[results.envelope.eq(envelope)].copy()
    if accepted_only:
        selected = selected.loc[selected.accepted.fillna(False)]
    if central_photons is not None:
        selected = selected.loc[selected.n_central.isin(central_photons)]
    if pure_central and not selected.empty:
        selected = selected.loc[selected.pure_central]
    if dataset_ids is not None:
        selected = selected.loc[selected.dataset_id.isin(dataset_ids)]
    return selected


def _empty(frame):
    if frame.empty:
        print("No matching fitted traces. Check the loading audit and fit status table.")
        return True
    return False


def plot_trace_fits(results, details, *, envelope="exponential",
                    central_photons=(2, 3), pure_central=True, dataset_ids=None):
    """One figure containing every selected trace, including rejected fits."""
    selected = select_results(results, envelope=envelope, accepted_only=False,
                              central_photons=central_photons,
                              pure_central=pure_central, dataset_ids=dataset_ids)
    available = [row for _, row in selected.iterrows()
                 if (row.trace_id, envelope) in details]
    if not available:
        print("No selected trace fits to display.")
        return None
    fig, axes = plt.subplots(len(available), 3, figsize=(17, 3.0 * len(available)),
                             squeeze=False, constrained_layout=True)
    for axes_row, row in zip(axes, available):
        detail = details[(row.trace_id, envelope)]
        trace, result, mask = detail["trace"], detail["result"], detail["mask"]
        t, A = np.asarray(trace["time_us"]), np.asarray(trace["A"])
        axes_row[0].plot(t, A.real, label="Re A")
        axes_row[0].plot(t, A.imag, label="Im A")
        axes_row[0].set_ylabel("Acquired complex A")
        axes_row[1].plot(t, abs(A), color="black", label="Measured |A|")
        if result is not None:
            axes_row[1].plot(t[mask], result["model_amplitude"], color="tab:red",
                             label=f"{envelope} fit")
            coherent = abs(np.asarray(trace["coherent_A"]))
            axes_row[1].plot(t, np.sqrt(result["scale"]) * coherent, alpha=0.5,
                             color="tab:blue", label="Coherent model (no decay)")
            axes_row[2].plot(t[mask], result["residual_power"], lw=0.8)
        else:
            axes_row[2].text(0.05, 0.5, detail["error"], wrap=True,
                             transform=axes_row[2].transAxes)
        axes_row[1].set_ylabel("Amplitude")
        axes_row[2].axhline(0, color="0.5", lw=0.7)
        axes_row[2].set_ylabel("Power residual")
        tau = row.get("tau_estimate_us", np.nan)
        realization_label = "unindexed run" if pd.isna(row.realization) else f"r={int(row.realization)}"
        axes_row[0].set_title(
            f"{row.dataset_id}, {realization_label}, n={tuple(row.occupation)}\n"
            f"|K|={row.K_kHz:g} kHz, g={row.g_mean_kHz:.3g} kHz; "
            + ",".join(row.mode_labels))
        axes_row[1].set_title(f"tau estimate={tau:.4g} us; {row.status}")
        axes_row[2].set_title("Measured |A|^2 - fitted power")
        for axis in axes_row:
            axis.set_xlabel("Evolution time (us)")
        axes_row[0].legend(fontsize=8)
        axes_row[1].legend(fontsize=8)
    fig.suptitle("Initial central occupations: measured traces and decay fits"
                 + (" (all other modes empty)" if pure_central else ""))
    return fig


_COHORT_COLUMNS = ["coupling_key", "total_photons", "disorder_group_kHz",
                   "mode_set", "disorder_metric"]


def _cohort_frame(frame):
    frame = frame.copy()
    # Group only floating-point roundoff, while retaining unrounded values in results.
    frame["coupling_key"] = frame.g_kHz.map(lambda values: tuple(np.round(values, 9)))
    frame["disorder_group_kHz"] = frame.disorder_strength_kHz.round(6)
    frame["mode_set"] = frame.mode_labels.map(tuple)
    frame["disorder_metric"] = frame.disorder_strength_source.map(
        lambda source: "saved strength" if str(source).startswith("saved") else "detuning RMS")
    return frame


def _cohorts(frame):
    # Occupation coordinates refer to particular physical modes.
    return list(_cohort_frame(frame).groupby(_COHORT_COLUMNS, dropna=False, sort=True))


def _cohort_title(key):
    couplings, N, disorder, modes, metric = key
    d = "unknown" if pd.isna(disorder) else f"{disorder:g} kHz"
    return f"{_coupling_label(couplings)}, N={N:g}; {metric}={d}\n{', '.join(modes)}"


def _coupling_label(couplings):
    if len(set(couplings)) == 1:
        return f"g={couplings[0]:.5g} kHz"
    return "g=[" + ", ".join(f"{g:.5g}" for g in couplings) + "] kHz"


def _actual_g_label(frame):
    values = np.concatenate([np.asarray(g) for g in frame.g_kHz])
    return f"actual g range {values.min():.4g}..{values.max():.4g} kHz"


def _mean_sd(values, x_column):
    """Unweighted trace means and sample SD; a singleton has undefined SD."""
    return values.groupby(x_column, sort=True, dropna=False).agg(
        mean_tau_us=("tau_us", "mean"),
        std_tau_us=("tau_us", "std"),
        n_traces=("tau_us", "count"),
    ).reset_index()


def _plot_mean_sd(axis, values, x_column, *, label=None):
    summary = _mean_sd(values, x_column)
    if summary.empty:
        return summary
    x, y = summary[x_column].to_numpy(), summary.mean_tau_us.to_numpy()
    points, = axis.plot(x, y, "o", ms=7, label=label)
    repeated = summary.n_traces.to_numpy() > 1
    if repeated.any():
        axis.errorbar(x[repeated], y[repeated],
                      yerr=summary.std_tau_us.to_numpy()[repeated],
                      fmt="none", color=points.get_color(), capsize=5, elinewidth=1.6)
    x_span = np.ptp(axis.get_xlim())
    for index, (point_x, point_y, n) in enumerate(zip(x, y, summary.n_traces)):
        nearby_next = index + 1 < len(x) and x[index + 1] - point_x < 0.06 * x_span
        offset, alignment = ((-8, 7), "right") if nearby_next else ((8, 7), "left")
        axis.annotate(f"n={n}", (point_x, point_y), xytext=offset, ha=alignment,
                      textcoords="offset points", fontsize=8,
                      color=points.get_color())
    axis.margins(x=0.15, y=0.15)
    return summary


def plot_kerr_overview(results, *, envelope="exponential", pure_central=False):
    """One mean and sample SD per Kerr/occupation-count/mode/disorder condition."""
    all_selected = select_results(results, envelope=envelope, accepted_only=False,
                                  central_photons=(2, 3), pure_central=pure_central)
    if _empty(all_selected):
        return None
    groups = list(_cohort_frame(all_selected).groupby(
        ["coupling_key", "total_photons"], sort=True, dropna=False))
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 4.0 * len(groups)),
                             squeeze=False, constrained_layout=True)
    for row_axes, ((couplings, N), cohort) in zip(axes, groups):
        for axis, central in zip(row_axes, (2, 3)):
            selected = cohort.loc[cohort.n_central.eq(central)]
            if selected.empty:
                axis.set_axis_off()
                axis.text(0.5, 0.5, f"N={N:g}, n_central={central}: no acquired occupations",
                          ha="center", va="center", transform=axis.transAxes)
                continue
            accepted = selected.loc[selected.accepted]
            for key, values in accepted.groupby(
                    ["mode_set", "disorder_metric", "disorder_group_kHz"], sort=True, dropna=False):
                modes, metric, strength = key
                metric_label = "strength" if metric == "saved strength" else "RMS"
                label = f"{','.join(modes[1:])}; {metric_label}={strength:g} kHz"
                _plot_mean_sd(axis, values, "K_kHz", label=label)
            axis.set(xlabel="Self-Kerr magnitude |K| (kHz)",
                     ylabel="Mean amplitude decay time (us)",
                     title=f"{_coupling_label(couplings)}, N={N:g}, n_central={central}\n"
                           f"accepted {len(accepted)}/{len(selected)} traces")
            axis.grid(alpha=0.2)
            if not accepted.empty:
                axis.legend(fontsize=8, title="Storage modes / disorder", title_fontsize=9)
    fig.suptitle(f"{envelope}: mean decay time +/- 1 sample SD\n"
                 "n = accepted traces; n=1 has no error bar")
    return fig


def plot_kerr_dependence(results, *, envelope="exponential", pure_central=True):
    frame = select_results(results, envelope=envelope, central_photons=(2, 3),
                           pure_central=pure_central)
    if _empty(frame):
        return None
    groups = _cohorts(frame)
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 3.5 * len(groups)),
                             squeeze=False, constrained_layout=True)
    for row_axes, (key, cohort) in zip(axes, groups):
        for axis, n in zip(row_axes, (2, 3)):
            values = cohort.loc[cohort.n_central.eq(n)]
            if values.empty:
                axis.set_axis_off()
                axis.text(0.5, 0.5, f"n_central={n}: no accepted fits in this condition",
                          ha="center", va="center", transform=axis.transAxes)
                continue
            for occupation, subset in values.groupby("occupation", sort=True):
                _plot_mean_sd(axis, subset, "K_kHz", label=str(occupation))
            axis.set(xlabel="Self-Kerr magnitude |K| (kHz)",
                     ylabel="Mean amplitude decay time (us)",
                     title=f"n_central={n}; {_cohort_title(key)}\n{_actual_g_label(cohort)}")
            axis.grid(alpha=0.2)
            if not values.empty:
                axis.legend(title="Initial occupation", fontsize=8)
    fig.suptitle(f"{envelope}: mean +/- 1 sample SD for each full occupation\n"
                 "n = accepted traces; n=1 has no error bar")
    return fig


def plot_photon_dependence(results, *, mode="M1", envelope="exponential"):
    frame = select_results(results, envelope=envelope)
    if _empty(frame):
        return None
    column = f"n_{mode}"
    if column not in frame:
        print(f"Mode {mode} was not acquired in these traces.")
        return None
    frame = frame.loc[frame[column].notna()]
    groups = list(_cohort_frame(frame).groupby(["K_kHz", *_COHORT_COLUMNS],
                                              dropna=False, sort=True))
    if not groups:
        return None
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 3.2 * len(groups)),
                             squeeze=False, constrained_layout=True)
    for axis, (key, values) in zip(axes[:, 0], groups):
        K, *cohort_key = key
        _plot_mean_sd(axis, values, column)
        axis.set(xlabel=f"Initial photon number in {mode}",
                 ylabel="Mean amplitude decay time (us)",
                 title=f"|K|={K:g} kHz; {_cohort_title(cohort_key)}\n{_actual_g_label(values)}")
        axis.set_xticks(sorted(values[column].unique()))
        axis.grid(alpha=0.2)
    fig.suptitle(f"{envelope}: mean +/- 1 sample SD at each photon number\n"
                 "n = accepted traces; n=1 has no error bar")
    return fig


def plot_decay_map(results, *, mode="M1", envelope="exponential"):
    frame = select_results(results, envelope=envelope)
    if _empty(frame):
        return None
    column = f"n_{mode}"
    if column not in frame:
        print(f"Mode {mode} was not acquired in these traces.")
        return None
    frame = frame.loc[frame[column].notna()]
    groups = _cohorts(frame)
    if not groups:
        return None
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 3.8 * len(groups)),
                             squeeze=False, constrained_layout=True)
    pivots = [values.pivot_table(index=column, columns="K_kHz", values="tau_us", aggfunc="mean")
              for _, values in groups]
    means = np.concatenate([pivot.to_numpy().ravel() for pivot in pivots])
    vmin, vmax = np.nanmin(means), np.nanmax(means)
    for axis, (key, values), pivot in zip(axes[:, 0], groups, pivots):
        counts = values.pivot_table(index=column, columns="K_kHz", values="tau_us", aggfunc="count")
        im = axis.imshow(np.ma.masked_invalid(pivot.to_numpy()), origin="lower", aspect="auto",
                         cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                if np.isfinite(value):
                    axis.text(j, i, f"{value:.3g}\n(n={int(counts.iloc[i,j])})",
                              ha="center", va="center", color="white", fontsize=9)
        axis.set(xlabel="Self-Kerr magnitude |K| (kHz)", ylabel=f"Initial photons in {mode}",
                 title=f"{_cohort_title(key)}\n{_actual_g_label(values)}")
        fig.colorbar(im, ax=axis, label="Mean accepted amplitude tau (us)")
    fig.suptitle("Mean decay time at each measured combination; n = accepted traces")
    return fig

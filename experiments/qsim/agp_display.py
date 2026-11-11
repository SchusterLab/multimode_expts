"""Displays of the central-boson AGP layers, in the order of the processing.

One method per stage, the same on one W (``CentralBosonAGPExperiment``) and on
the disorder scan, with single or sequential readout:

1. ``display_populations``: <i|n_r(t)|i> of every prepared state |i> (the jobs).
2. ``display_correlations``: C_1(t), C_2(t) with their errors and the closed system.
3. ``display_spectrum``: the spectral function S(f) of C_1.
4. ``display_agp_mu``: the AGP against the cutoff mu.
5. ``display_agp`` (scan only): the AGP against the disorder W.

``display()`` draws every stage that has data, in this order, and returns the
list of figures; each stage method returns its Figure. The methods only read
``data``: run ``analyze()`` first. A stage without its data raises
``MissingStageData``, which ``display()`` reports and skips.

The panels are a grid of W (rows; one row for one W) by direction (columns;
one column for an experiment without a direction axis). ``_grid`` gives every
array these two leading axes, so the code is the same for all four classes.
"""
import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.fock_population import mode_name
from fitting.qsim.agp import spectral_function, spectrum_frequencies_khz, subtract_background
from theory.qsim.agp import split_dc

AGP_LABEL = r"$g^2\|A_\mu\|^2$"


class MissingStageData(ValueError):
    """A display stage has no data: analyze with the settings it needs first."""


def _grid_figure(rows, columns, panel, **kwargs):
    """-> (fig, axes (rows, columns)) with panels of size ``panel`` (inches)."""
    width, height = panel
    return plt.subplots(rows, columns, figsize=(max(7., width * columns), height * rows + .8),
                        squeeze=False, constrained_layout=True, **kwargs)


class AGPDisplayMixin:
    """The display stages of the AGP layers (module docstring)."""

    # True on the disorder scan: its arrays have a leading W axis.
    HAS_W_AXIS = False
    # The cutoffs (kHz) a plot shows by default, when more than 4 were analyzed.
    SHOWN_CUTOFFS_KHZ = (5., 10., 20.)
    # The default upper limit of S(f): this times the highest closed-system frequency.
    SPECTRUM_MARGIN = 1.3
    # Set by the classes that use this mixin.
    data: dict
    children: list

    def display(self, raw=False, theory=True, mu=None):
        """Every stage that has data, from the populations to the AGP. -> [Figure].

        ``raw``: raw instead of readout-corrected values. ``theory``: the
        closed-system curves. ``mu``: the cutoffs shown (kHz; default
        ``shown_cutoffs``). For the options of one stage, call its method.
        """
        if not self.data:
            raise ValueError(f"{type(self).__name__} has no data: run analyze() first")
        figures = []
        for name, draw in self._stages(raw, theory, mu):
            try:
                figures.append(draw())
            except MissingStageData as reason:
                print(f"[{type(self).__name__}.display] {name} skipped: {reason}")
        return figures

    def _stages(self, raw, theory, mu):
        """-> [(name, draw)] of ``display()``, in the order of the processing."""
        return [("populations", lambda: self.display_populations(raw=raw)),
                ("correlations", lambda: self.display_correlations(raw=raw, theory=theory)),
                ("spectrum", lambda: self.display_spectrum(mu=mu, raw=raw, theory=theory)),
                ("agp_mu", lambda: self.display_agp_mu(theory=theory))]

    # -- 1. populations -----------------------------------------------------

    def display_populations(self, raw=False):
        """<i|n_r(t)|i> of every prepared state |i> as a map, per W and direction. -> Figure.

        Read from each W's own data, so a scan needs its parts loaded and analyzed.
        """
        parts = self.children if self.HAS_W_AXIS else [self]
        if not parts or any("n_mean" not in part.data for part in parts):
            raise MissingStageData("needs the analyzed parts (from_manifest, then analyze)")
        maps = [_population_maps(part, raw) for part in parts]
        columns = max(len(entry[0]) for entry in maps)
        states = len(parts[0].data["occupations"])
        fig, axes = _grid_figure(len(parts), columns,
                                 (10 if columns == 1 else 5, min(6., max(2.5, .2 * states + 1))))
        for row, (part, (panels, kind)) in enumerate(zip(parts, maps)):
            labels = [str(list(occupation)) for occupation in part.data["occupations"]]
            for column, (values, read_mode) in enumerate(panels):
                ax = axes[row, column]
                mesh = ax.pcolormesh(np.asarray(part.data["cycles"]), np.arange(len(labels)),
                                     values, shading="auto")
                if column == 0:
                    ax.set_yticks(range(len(labels)), labels, fontsize=6)
                else:
                    ax.set_yticks([])
                fig.colorbar(mesh, ax=ax, label=f"<i|n_{mode_name(read_mode)}|i> ({kind})")
                ax.set_title(self._label(row, self._directions()[column]), fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Floquet cycles")
        fig.suptitle(f"{type(self).__name__}: populations", fontsize=10)
        return fig

    # -- 2. correlations ----------------------------------------------------

    def display_correlations(self, raw=False, theory=True, style="traces", moment=1):
        """C_k(t) of every W and direction. -> Figure.

        ``style="traces"``: one panel per W and direction with C_1 and C_2 and
        their shot errors (corrected when there are corrected values;
        ``raw=True`` for raw), dashed the closed-system C_1. ``style="map"``
        (scan): one map of C_``moment`` against cycles and W per direction.
        """
        self._require("C1")
        if style == "map":
            return self._correlation_map(moment, raw)
        if style != "traces":
            raise ValueError(f"style must be 'traces' or 'map', not {style!r}")
        cycles = np.asarray(self.data["cycles"])
        directions, strengths = self._directions(), self._strengths()
        fig, axes = _grid_figure(len(strengths), len(directions),
                                 (9 if len(directions) == 1 else 4.5, 2.2), sharex=True)
        for k in (1, 2):
            if f"C{k}" not in self.data:
                continue
            key, kind, with_error = self._shown(f"C{k}", raw)
            values = self._grid(key)
            errors = self._grid(f"C{k}_err") if with_error and f"C{k}_err" in self.data else None
            for (row, column), ax in np.ndenumerate(axes):
                if np.all(np.isnan(values[row, column])):  # no C_2 for the Kerr direction
                    continue
                line, = ax.plot(cycles, values[row, column], lw=1, label=f"C_{k} ({kind})")
                if errors is not None:
                    ax.fill_between(cycles, values[row, column] - errors[row, column],
                                    values[row, column] + errors[row, column],
                                    color=line.get_color(), alpha=.3, lw=0)
        if theory and "C1_theory" in self.data:
            curves = self._grid("C1_theory")
            for (row, column), ax in np.ndenumerate(axes):
                ax.plot(cycles, curves[row, column], "k--", lw=1, label="C_1, closed system")
        for (row, column), ax in np.ndenumerate(axes):
            ax.set_title(self._label(row, directions[column]), fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Floquet cycles")
        for ax in axes[0]:
            ax.legend(fontsize=7, ncol=3)
        fig.suptitle(f"{type(self).__name__}: correlations", fontsize=10)
        return fig

    def _correlation_map(self, moment, raw):
        if moment not in (1, 2):
            raise ValueError("moment must be 1 or 2")
        strengths = self._strengths()
        if not self.HAS_W_AXIS:
            raise ValueError("a correlation map needs a disorder scan")
        if np.any(np.diff(strengths) == 0):
            raise ValueError("a disorder map needs one part per W")
        key, kind, _ = self._shown(f"C{moment}", raw)
        if key not in self.data:
            raise MissingStageData(f"no {key}")
        values, directions = self._grid(key), self._directions()
        fig, axes = _grid_figure(1, len(directions), (10 if len(directions) == 1 else 5, 4.5))
        for column, (ax, direction) in enumerate(zip(axes[0], directions)):
            mesh = ax.pcolormesh(self.data["cycles"], strengths, values[:, column], shading="auto")
            fig.colorbar(mesh, ax=ax, label=f"C_{moment} ({kind})")
            title = f"C_{moment} ({kind})"
            ax.set(xlabel="Floquet cycles", ylabel="Disorder W (kHz)",
                   title=title if direction is None else f"{mode_name(direction)}: {title}")
        fig.suptitle(f"{type(self).__name__}: correlation map", fontsize=10)
        return fig

    # -- 3. spectrum ----------------------------------------------------------

    def display_spectrum(self, mu=None, raw=False, theory=True, window=None,
                         background_start_us=None, max_khz=None, yscale="linear"):
        """The one-sided spectral function S(f) of C_1, per W and direction. -> Figure.

        By default the analyzed curves (``spectrum``, ``spectrum_err``,
        ``spectrum_with_background`` in grey, ``spectrum_theory`` dashed): S is
        the cosine transform of C_1 over its window, per kHz, with the shot
        error, background-subtracted when analyzed so. A ``window``,
        ``background_start_us`` other than the analyzed ones, or ``raw``,
        computes S from C_1 here instead (``fitting.qsim.agp.spectral_function``).
        With the jobs loaded: the closed-system weights in bins of 1/T (steps).
        The AGP filters f^2 / (f^2 + mu^2)^2, scaled to a peak of 1, are on the
        right axis (``mu`` in kHz; default ``shown_cutoffs``). ``max_khz``
        defaults to ``SPECTRUM_MARGIN`` times the highest closed-system
        frequency, at most the Nyquist frequency. A panel whose closed-system
        weight is partly above the Nyquist frequency (it folds back into the
        measured S) is marked with that fraction.
        """
        self._require("C1")
        if "time_us" not in self.data:
            raise MissingStageData("needs the times of the cycles (time_us): recorded timing")
        analysis = getattr(self, "analysis", None) or {}
        analyzed_window = analysis.get("window", "hann")
        analyzed_background = analysis.get("background_start_us")
        stored = ("spectrum" in self.data and not raw
                  and window in (None, analyzed_window)
                  and background_start_us in (None, analyzed_background))
        window = window or analyzed_window
        if stored:
            background_start_us = analyzed_background
        subtract = background_start_us is not None
        key, kind, _ = self._shown("C1", raw)
        times = np.asarray(self.data["time_us"])
        resolution_khz = 1000 / times[-1]
        nyquist_khz = 1000 / (2 * np.median(np.diff(times)))
        frequency_khz = (np.asarray(self.data["frequency_khz"]) if stored
                         else spectrum_frequencies_khz(times))
        directions, strengths = self._directions(), self._strengths()
        model = {(row, column): self._theory_weights(row, column)
                 for row in range(len(strengths)) for column in range(len(directions))}
        lines = {index: _without_dc(pair) for index, pair in model.items() if pair is not None}
        if max_khz is None:
            top = max((f[w > 1e-3 * w.max()].max() for f, w in lines.values() if len(f)),
                      default=None)
            max_khz = nyquist_khz if top is None else min(nyquist_khz, self.SPECTRUM_MARGIN * top)
        cutoffs = self.shown_cutoffs(mu)
        fig, axes = _grid_figure(len(strengths), len(directions),
                                 (9 if len(directions) == 1 else 4.5, 2.), sharex=True)
        label = f"{key.replace('_', ' ')}" + (", background subtracted" if subtract else "")
        for (row, column), ax in np.ndenumerate(axes):
            theory_curve = None
            if stored:
                spectrum = self._grid("spectrum")[row, column]
                spread = self._grid("spectrum_err")[row, column]
                unsubtracted = (self._grid("spectrum_with_background")[row, column]
                                if subtract and "spectrum_with_background" in self.data else None)
                if theory and "spectrum_theory" in self.data:
                    theory_curve = self._grid("spectrum_theory")[row, column]
            else:
                values = self._grid(key)[row, column]
                error = self._grid("C1_err")[row, column]
                unsubtracted = spectral_function(times, values, error, frequency_khz, window)[0]
                if subtract:
                    values = subtract_background(times, values, error, background_start_us)[0]
                spectrum, spread = spectral_function(times, values, error, frequency_khz, window)
            ax.plot(frequency_khz, spectrum, lw=1, label=label)
            ax.fill_between(frequency_khz, spectrum - spread, spectrum + spread, alpha=.3, lw=0)
            pair = model[row, column]
            if theory and pair is not None:
                differences, weights = pair
                dc, mask = split_dc(differences, weights)
                kept = ~mask if subtract else np.ones(mask.shape, dtype=bool)
                edges = np.arange(0, max_khz + resolution_khz, resolution_khz)
                binned, _ = np.histogram(1000 * np.abs(differences[kept]), edges,
                                         weights=weights[kept])
                ax.stairs(binned / resolution_khz, edges, color="0.5", lw=.8,
                          label="closed system, weights in bins of 1/T")
                if theory_curve is None and "C1_theory" in self.data:
                    theory_curve = spectral_function(
                        times, self._grid("C1_theory")[row, column] - (dc if subtract else 0.),
                        np.zeros(len(times)), frequency_khz, window)[0]
            elif theory_curve is None and theory and "C1_theory" in self.data and not subtract:
                theory_curve = spectral_function(times, self._grid("C1_theory")[row, column],
                                                 np.zeros(len(times)), frequency_khz, window)[0]
            if theory_curve is not None:
                ax.plot(frequency_khz, theory_curve, "k--", lw=1, label="closed system, same window")
            ax.set_yscale(yscale)
            if unsubtracted is not None and subtract:  # grey; the y range stays on the subtracted S
                limits = ax.get_ylim()
                ax.plot(frequency_khz, unsubtracted, color="0.7", lw=1,
                        label=f"{key.replace('_', ' ')}, with background")
                ax.set_ylim(limits)
            ax.set_title(self._label(row, directions[column]), fontsize=8)
            if column == 0:
                ax.set_ylabel("S (1/kHz)", fontsize=8)
            if (row, column) in lines:
                f, w = lines[row, column]
                folded = w[f > nyquist_khz].sum() / w.sum() if w.sum() > 0 else 0.
                if folded >= 1e-3:
                    ax.text(.99, .95, f"{100 * folded:.1f} % of the closed-system weight is above "
                            f"Nyquist ({nyquist_khz:.0f} kHz) and folds back",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="C3")
            filters = None
            if len(cutoffs):
                filters = ax.twinx()
                for cutoff in cutoffs:
                    shape = 4 * cutoff**2 * frequency_khz**2 / (frequency_khz**2 + cutoff**2)**2
                    filters.plot(frequency_khz, shape, ":", lw=1,
                                 label=rf"filter, $\mu$ = {cutoff:g} kHz")
                low, high = ax.get_ylim()  # the zero of the filters on the zero of S
                filters.set(ylim=(1.05 * min(low, 0) / max(high, 1e-12), 1.05), yticks=[])
            if (row, column) == (0, 0):
                handles, labels = ax.get_legend_handles_labels()
                if filters is not None:
                    more = filters.get_legend_handles_labels()
                    handles, labels = handles + more[0], labels + more[1]
                ax.legend(handles, labels, fontsize=7, ncol=2)
        for ax in axes[-1]:
            ax.set(xlabel="f (kHz)", xlim=(0, max_khz))
        fig.suptitle(f"{type(self).__name__}: spectral function of C_1 ({kind}), {window} window, "
                     f"1/T = {resolution_khz:.2g} kHz", fontsize=10)
        return fig

    def model_spectra_khz(self, direction=None):
        """-> per W, (f_kHz, |weights|) of the closed-system C_1 without its DC terms.

        Of ``direction`` (default the first). At the Kerr of the last analysis.
        ``[]`` when a W has no jobs, recorded couplings or analyzed theory.
        """
        directions = self._directions()
        column = 0 if direction is None else directions.index(direction)
        pairs = [self._theory_weights(row, column) for row in range(len(self._strengths()))]
        return [] if any(pair is None for pair in pairs) else [_without_dc(p) for p in pairs]

    # -- 4. AGP against mu ----------------------------------------------------

    def display_agp_mu(self, estimator=None, theory=True):
        """AGP(mu) on a log mu axis, one panel per direction, one curve per W. -> Figure.

        ``estimator``: "agp", "agp_t1" or "agp_connected"; default
        "agp_connected" when analyzed with a background, else "agp". The band
        is its error. Dashed: the infinite-time closed system; dots: g L/D.
        """
        estimator = estimator or ("agp_connected" if "agp_connected" in self.data else "agp")
        if estimator not in self.data:
            raise MissingStageData(f"no {estimator}: analyze with recorded couplings and timing"
                                   + (", and background_start_us" if estimator == "agp_connected"
                                      else ""))
        cutoffs = np.asarray(self.data["agp_mu"])
        values, errors = self._grid(estimator), self._grid(f"{estimator}_err")
        directions, strengths = self._directions(), self._strengths()
        colors = plt.get_cmap("viridis")(np.linspace(0, .9, len(strengths)))
        fig, axes = _grid_figure(1, len(directions), (7 if len(directions) == 1 else 4.5, 4.5))
        for column, ax in enumerate(axes[0]):
            for row, strength in enumerate(strengths):
                value, error = values[row, column], errors[row, column]
                ax.plot(cutoffs, value, lw=1.2, color=colors[row], label=f"W = {strength:g} kHz")
                ax.fill_between(cutoffs, value - error, value + error, color=colors[row],
                                alpha=.25, lw=0)
                if theory and "agp_theory" in self.data:
                    ax.plot(cutoffs, self._grid("agp_theory")[row, column], "--", lw=.8,
                            color=colors[row])
                if "mu_gld_khz" in self.data:
                    gld = float(self._grid("mu_gld_khz")[row, column])
                    ax.plot(gld, np.interp(gld, cutoffs, value), "o", ms=4, color=colors[row])
            direction = directions[column]
            ax.set(xscale="log", xlabel=r"$\mu$ (kHz)",
                   title=estimator if direction is None else f"{mode_name(direction)}: {estimator}")
        axes[0, 0].set_ylabel(AGP_LABEL)
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{type(self).__name__}: AGP against mu; dashed: closed system, "
                     "infinite time; dots: g L/D", fontsize=10)
        return fig

    # -- helpers ----------------------------------------------------------------

    def shown_cutoffs(self, mu=None):
        """-> the cutoffs (kHz) a plot shows: ``mu``; else the analyzed ones, when 4 or fewer;
        else the analyzed ones nearest to ``SHOWN_CUTOFFS_KHZ``."""
        if mu is not None:
            return np.atleast_1d(mu).astype(float)
        analyzed = np.asarray(self.data.get("agp_mu", []), dtype=float)
        if len(analyzed) <= 4:
            return analyzed
        return np.unique([analyzed[np.argmin(abs(analyzed - value))]
                          for value in self.SHOWN_CUTOFFS_KHZ])

    def _grid(self, key):
        """-> ``data[key]`` with leading (W, direction) axes, whatever the class."""
        value = np.asarray(self.data[key])
        if "directions" not in self.data:
            value = np.expand_dims(value, 1 if self.HAS_W_AXIS else 0)
        return value if self.HAS_W_AXIS else value[None]

    def _strengths(self):
        return np.atleast_1d(np.asarray(self.data["disorder_khz"], dtype=float))

    def _directions(self):
        """-> the directions of the columns; [None] when no data or attribute names one."""
        if "directions" in self.data:
            return [int(direction) for direction in self.data["directions"]]
        if "direction" in self.data:
            return [int(np.ravel(self.data["direction"])[0])]
        source = self.children[0] if self.HAS_W_AXIS and self.children else self
        return [getattr(source, "direction", None)]

    def _label(self, row, direction):
        label = f"W = {self._strengths()[row]:g} kHz"
        return label if direction is None else f"{label}, {mode_name(direction)}"

    def _require(self, *keys):
        missing = [key for key in keys if key not in self.data]
        if missing:
            raise MissingStageData(f"no {', '.join(missing)}: run analyze() first")

    def _shown(self, name, raw):
        """-> (key, kind, with_error) of the correlation ``name`` ("C1", "C2") a plot shows.

        The errors belong to the corrected values when there are some.
        """
        corrected = f"{name}_corrected" in self.data
        if corrected and not raw:
            return f"{name}_corrected", "corrected", True
        return name, "raw", not corrected

    def _theory_weights(self, row, column):
        """-> (differences_MHz, weights) of the closed-system C_1 of one panel, or None.

        Needs the jobs (recorded couplings) and the Kerr of an analyzed theory.
        """
        parts = self.children if self.HAS_W_AXIS else [self]
        part = parts[row] if row < len(parts) else None
        if part is None or not part.children or "theory_kerr_mhz" not in part.data:
            return None
        kerr = float(np.ravel(part.data["theory_kerr_mhz"])[column])
        direction = self._directions()[column]
        if not np.isfinite(kerr) or direction is None:
            return None
        try:
            return part.theory_spectrum(kerr_mhz=kerr, direction=direction)
        except ValueError:
            return None


def _population_maps(part, raw):
    """-> ([(map (states, cycles), read mode)] per direction, kind) of one W."""
    corrected = "n_mean_corrected" in part.data and not raw
    values = np.asarray(part.data["n_mean_corrected" if corrected else "n_mean"])
    kind = "corrected" if corrected else "raw"
    if values.ndim == 3:  # sequential readout: (state, cycle, readout step)
        modes = np.asarray(part.data["ro_stors"])
        return [(values[:, :, step], modes[step]) for step in part.data["readout_steps"]], kind
    return [(values, np.ravel(part.data["ro_stor"])[0])], kind


def _without_dc(pair):
    """-> (f_kHz, |weights|) of closed-system weights without their DC terms."""
    differences, weights = pair
    _, mask = split_dc(differences, weights)
    return 1000 * np.abs(np.asarray(differences)[~mask]), np.abs(np.asarray(weights)[~mask])

"""Displays of the central-boson AGP layers, in the order of the processing.

One method per stage, the same on one W (``CentralBosonAGPExperiment``) and on
the disorder scan, with single or sequential readout:

1. ``display_populations``: <i|n_r(t)|i> of every prepared state |i> (the jobs).
2. ``display_correlations``: C_1 and C_2 as measured, no theory: maps against
   cycles and W on a scan, lines for one W. Needs only ``analyze_correlations``.
3. ``display_traces``: C_1, the input of the AGP, against the closed system and
   the fitted decay A e^{-rate t}.
4. ``display_spectrum``: the spectral function S(f) of C_1.
5. ``display_agp_mu``: the AGP against the cutoff mu.
6. ``display_agp`` (scan only): the AGP against the disorder W.

``display()`` draws every stage that has data, in this order, and returns the
list of figures; each stage method returns its Figure. A stage without its
data raises ``MissingStageData``, which ``display()`` reports and skips. The
stage methods only read ``data``. ``display()`` on an experiment that was not
analyzed runs ``analyze_correlations()`` (no settings) and shows the stages its
data allow: no theory and no AGP.

The panels are a grid of W (rows; one row for one W) by direction (columns;
one column for an experiment without a direction axis). ``_grid`` gives every
array these two leading axes, so the code is the same for all four classes.
"""
import functools

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.fock_population import mode_name
from fitting.qsim.agp import spectral_function, spectrum_frequencies_khz, subtract_background
from theory.qsim.agp import split_dc

AGP_LABEL = r"$g^2\|A_\mu\|^2$"
# Small panels: small text.
SMALL_TEXT = {"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
              "ytick.labelsize": 6, "legend.fontsize": 6, "figure.titlesize": 8}


class MissingStageData(ValueError):
    """A display stage has no data: analyze with the settings it needs first."""


def _small_text(method):
    """Draw the figure of ``method`` with ``SMALL_TEXT``."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with plt.rc_context(SMALL_TEXT):
            return method(*args, **kwargs)
    return wrapper


def _grid_figure(rows, columns, panel, **kwargs):
    """-> (fig, axes (rows, columns)) with panels of size ``panel`` (inches)."""
    width, height = panel
    return plt.subplots(rows, columns, figsize=(max(4.5, width * columns), height * rows + .5),
                        squeeze=False, constrained_layout=True, **kwargs)


def _width(columns, single=5., each=3.2):
    """-> the panel width: wider when there is only one column."""
    return single if columns == 1 else each


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

    def analyze_correlations(self):
        """The settings-free first step of ``analyze`` (each class has its own)."""
        raise NotImplementedError

    def display(self, raw=False, theory=True, mu=None):
        """Every stage that has data, from the populations to the AGP. -> [Figure].

        ``raw``: raw instead of readout-corrected values. ``theory``: the
        closed-system curves. ``mu``: the cutoffs shown (kHz; default
        ``shown_cutoffs``). Not analyzed: ``analyze_correlations()`` first,
        then only the stages that need no settings. For the options of one
        stage, call its method.
        """
        if not self.data:
            if not self.children:
                raise ValueError(f"{type(self).__name__} has no data and no jobs")
            print(f"[{type(self).__name__}.display] not analyzed: analyze_correlations() only; "
                  "analyze() for the spectrum and the AGP")
            self.analyze_correlations()
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
                ("correlations", lambda: self.display_correlations(raw=raw)),
                ("traces", lambda: self.display_traces(raw=raw, theory=theory)),
                ("spectrum", lambda: self.display_spectrum(mu=mu, raw=raw, theory=theory)),
                ("agp_mu", lambda: self.display_agp_mu(theory=theory))]

    # -- 1. populations -----------------------------------------------------

    @_small_text
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
                                 (_width(columns), min(4., max(1.8, .12 * states + .8))))
        for row, (part, (panels, kind)) in enumerate(zip(parts, maps)):
            labels = [str(list(occupation)) for occupation in part.data["occupations"]]
            for column, (values, read_mode) in enumerate(panels):
                ax = axes[row, column]
                mesh = ax.pcolormesh(np.asarray(part.data["cycles"]), np.arange(len(labels)),
                                     values, shading="auto")
                if column == 0:
                    ax.set_yticks(range(len(labels)), labels, fontsize=5)
                else:
                    ax.set_yticks([])
                fig.colorbar(mesh, ax=ax, label=f"<i|n_{mode_name(read_mode)}|i> ({kind})")
                ax.set_title(self._label(row, self._directions()[column]))
        for ax in axes[-1]:
            ax.set_xlabel("Floquet cycles")
        fig.suptitle(f"{type(self).__name__}: populations")
        return fig

    # -- 2. correlations, as measured -----------------------------------------

    @_small_text
    def display_correlations(self, raw=False):
        """C_1 and C_2 as measured (corrected; ``raw=True`` for raw), no theory. -> Figure.

        Rows C_1, C_2 (separate scales), one column per direction; the Kerr
        direction has no C_2. A scan: maps against cycles and W (one part per
        W). One W: lines with the shot error.
        """
        self._require("C1")
        directions, strengths = self._directions(), self._strengths()
        moments = [1] + ([2] if "C2" in self.data else [])
        if self.HAS_W_AXIS and np.any(np.diff(strengths) == 0):
            raise MissingStageData("a disorder map needs one part per W")
        cycles = np.asarray(self.data["cycles"])
        fig, axes = _grid_figure(len(moments), len(directions),
                                 (_width(len(directions)), 2.2 if self.HAS_W_AXIS else 1.5),
                                 sharex=True)
        for row, moment in enumerate(moments):
            key, kind, with_error = self._shown(f"C{moment}", raw)
            values = self._grid(key)
            for column, direction in enumerate(directions):
                ax = axes[row, column]
                if np.all(np.isnan(values[:, column])):  # no C_2 for the Kerr direction
                    ax.set_axis_off()
                    ax.text(.5, .5, f"no C_{moment}", ha="center", va="center",
                            transform=ax.transAxes)
                    continue
                name = f"C_{moment} ({kind})"
                ax.set_title(name if direction is None else f"{mode_name(direction)}: {name}")
                if self.HAS_W_AXIS:
                    mesh = ax.pcolormesh(cycles, strengths, values[:, column], shading="auto")
                    fig.colorbar(mesh, ax=ax)
                    ax.set_ylabel("W (kHz)")
                else:
                    ax.plot(cycles, values[0, column], lw=1)
                    if with_error and f"C{moment}_err" in self.data:
                        error = self._grid(f"C{moment}_err")[0, column]
                        ax.fill_between(cycles, values[0, column] - error,
                                        values[0, column] + error, alpha=.3, lw=0)
        for ax in axes[-1]:
            ax.set_xlabel("Floquet cycles")
        fig.suptitle(f"{type(self).__name__}: correlations"
                     + ("" if self.HAS_W_AXIS else f", W = {strengths[0]:g} kHz"))
        return fig

    # -- 3. C_1 against the models ------------------------------------------------

    @_small_text
    def display_traces(self, raw=False, theory=True):
        """C_1, the input of the AGP, against the models, per W and direction. -> Figure.

        C_1 with its shot error (corrected; ``raw=True`` for raw). Dashed: the
        closed system (``theory``). Red: the fitted decay A e^{-rate t}, from
        the fit start on, when analyzed with ``background_start_us``.
        """
        self._require("C1")
        cycles = np.asarray(self.data["cycles"])
        directions, strengths = self._directions(), self._strengths()
        key, kind, with_error = self._shown("C1", raw)
        values = self._grid(key)
        errors = self._grid("C1_err") if with_error and "C1_err" in self.data else None
        curves = self._grid("C1_theory") if theory and "C1_theory" in self.data else None
        fit = None
        if "background_amplitude" in self.data and "time_us" in self.data:
            times = np.asarray(self.data["time_us"])
            start = (getattr(self, "analysis", None) or {}).get("background_start_us")
            if start is None:  # an arrays-only view: the start the parts saved, else 0
                saved = self.data.get("background_start_us")
                start = 0. if saved is None else float(np.nanmax(saved))
            fitted = times >= start
            fit = (fitted, times[fitted], self._grid("background_amplitude"),
                   self._grid("background_rate_per_us"))
        fig, axes = _grid_figure(len(strengths), len(directions),
                                 (_width(len(directions)), 1.3), sharex=True)
        for (row, column), ax in np.ndenumerate(axes):
            line, = ax.plot(cycles, values[row, column], lw=1, label=f"C_1 ({kind})")
            if errors is not None:
                ax.fill_between(cycles, values[row, column] - errors[row, column],
                                values[row, column] + errors[row, column],
                                color=line.get_color(), alpha=.3, lw=0)
            if curves is not None:
                ax.plot(cycles, curves[row, column], "k--", lw=.8, label="closed system")
            if fit is not None:
                fitted, times, amplitude, rate = fit
                ax.plot(cycles[fitted], amplitude[row, column] * np.exp(-rate[row, column] * times),
                        color="C3", lw=.8, label=r"fit $A e^{-\Gamma t}$")
            ax.set_title(self._label(row, directions[column]))
        for ax in axes[-1]:
            ax.set_xlabel("Floquet cycles")
        for ax in axes[0]:
            ax.legend(ncol=3)
        fig.suptitle(f"{type(self).__name__}: C_1 against the models")
        return fig

    # -- 4. spectrum ----------------------------------------------------------

    @_small_text
    def display_spectrum(self, mu=None, raw=False, theory=True, window=None,
                         background_start_us=None, max_khz=None, yscale="symlog"):
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
        measured S) is marked with that fraction. ``yscale``: "symlog" (linear
        within the median shot error of S, so the noise and the DC peak both
        show) or a matplotlib scale.
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
        panels = {}
        for row in range(len(strengths)):
            for column in range(len(directions)):
                if stored:
                    panels[row, column] = (
                        self._grid("spectrum")[row, column], self._grid("spectrum_err")[row, column],
                        self._grid("spectrum_with_background")[row, column]
                        if subtract and "spectrum_with_background" in self.data else None)
                else:
                    values = self._grid(key)[row, column]
                    error = self._grid("C1_err")[row, column]
                    unsubtracted = spectral_function(times, values, error, frequency_khz, window)[0]
                    if subtract:
                        values = subtract_background(times, values, error, background_start_us)[0]
                    panels[row, column] = (*spectral_function(times, values, error, frequency_khz,
                                                              window), unsubtracted)
        spreads = np.concatenate([np.ravel(spread) for _, spread, _ in panels.values()])
        linthresh = float(np.nanmedian(spreads)) if np.any(spreads > 0) else 1e-6
        fig, axes = _grid_figure(len(strengths), len(directions),
                                 (_width(len(directions)), 1.3), sharex=True)
        label = f"{key.replace('_', ' ')}" + (", background subtracted" if subtract else "")
        for (row, column), ax in np.ndenumerate(axes):
            spectrum, spread, unsubtracted = panels[row, column]
            theory_curve = None
            if stored and theory and "spectrum_theory" in self.data:
                theory_curve = self._grid("spectrum_theory")[row, column]
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
                ax.plot(frequency_khz, theory_curve, "k--", lw=.8, label="closed system, same window")
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=linthresh)
            else:
                ax.set_yscale(yscale)
            if unsubtracted is not None and subtract:  # grey; the y range stays on the subtracted S
                limits = ax.get_ylim()
                ax.plot(frequency_khz, unsubtracted, color="0.7", lw=1,
                        label=f"{key.replace('_', ' ')}, with background")
                ax.set_ylim(limits)
            ax.set_title(self._label(row, directions[column]))
            if column == 0:
                ax.set_ylabel("S (1/kHz)")
            if (row, column) in lines:
                f, w = lines[row, column]
                folded = w[f > nyquist_khz].sum() / w.sum() if w.sum() > 0 else 0.
                if folded >= 1e-3:
                    ax.text(.99, .95, f"{100 * folded:.1f} % of the closed-system weight is above "
                            f"Nyquist ({nyquist_khz:.0f} kHz) and folds back",
                            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="C3")
            filters = None
            if len(cutoffs):
                filters = ax.twinx()
                for cutoff in cutoffs:
                    shape = 4 * cutoff**2 * frequency_khz**2 / (frequency_khz**2 + cutoff**2)**2
                    filters.plot(frequency_khz, shape, ":", lw=1,
                                 label=rf"filter, $\mu$ = {cutoff:g} kHz")
                filters.set(ylim=(0, 1.05), yticks=[])
            if (row, column) == (0, 0):
                handles, labels = ax.get_legend_handles_labels()
                if filters is not None:
                    more = filters.get_legend_handles_labels()
                    handles, labels = handles + more[0], labels + more[1]
                ax.legend(handles, labels, ncol=2)
        for ax in axes[-1]:
            ax.set(xlabel="f (kHz)", xlim=(0, max_khz))
        fig.suptitle(f"{type(self).__name__}: spectral function of C_1\n({kind}, {window} window, "
                     f"1/T = {resolution_khz:.2g} kHz)")
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

    # -- 5. AGP against mu ----------------------------------------------------

    @_small_text
    def display_agp_mu(self, estimator=None, theory=True, yscale="log"):
        """AGP(mu) on log axes, one panel per direction, one curve per W. -> Figure.

        ``estimator``: "agp", "agp_t1" or "agp_connected"; default
        "agp_connected" when analyzed with a background, else "agp". The band
        is its error. Dashed: the infinite-time closed system; dots: g L/D.
        ``yscale``: "log" (values at or below 0 are not shown) or "linear".
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
        fig, axes = _grid_figure(1, len(directions), (_width(len(directions), 4.), 3.))
        fig.set_figwidth(fig.get_figwidth() + 1.2)  # the legend, outside on the right
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
                    ax.plot(gld, np.interp(gld, cutoffs, value), "o", ms=3, color=colors[row])
            direction = directions[column]
            ax.set(xscale="log", yscale=yscale, xlabel=r"$\mu$ (kHz)",
                   title=estimator if direction is None else f"{mode_name(direction)}: {estimator}")
        axes[0, 0].set_ylabel(AGP_LABEL)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right upper")
        fig.suptitle(f"{type(self).__name__}: AGP against mu\n"
                     "(dashed: closed system, infinite time; dots: g L/D)")
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

"""M1 Fock ladder with a qubit herald, or a herald and a qubit reset, after its steps.

Question (guan, 2026-11-13): after a Fock 3 prep in M1 the qubit is in f at readout time in about
5 % of the shots at 0.45 mA, and such a shot reads n = 3 in every mode (vault
``Lab/guan/AGP/2026/11/2026-11-13``; ``fitting/qsim/f_postselect.py``). Does a qubit readout after
the ladder steps, with post-selection, or with a qubit reset, remove these shots, and what does it
cost (M1 decays during each readout)?

``cfg.expt.prep_herald``:
    "none": the prep of ``FockPopulationProgram``, unchanged.
    "end": one bare qubit readout after each ladder (the qubit should be in g).
    "each": one bare qubit readout after each ladder step (g_i-e_i, e_i-f_i, f_i-g_(i+1)).
    "each_reset": after each step, a qubit-only active reset (readout, ge pi if e; ef pi, readout,
        ge pi if e) and then a g herald: ``MM_base.active_reset`` with ``RESET_PARAMS``.
A bare readout is ``MM_base.post_selection_measure`` (readout, then a 2 us wait).

``FockPrepHeraldExperiment``: one prep and one herald mode per job (all points of a job must have
the same readout lanes), the read mode ``ro_stor`` swept over ``cfg.expt.ro_stors``, other keys as
for ``FockPopulationExperiment`` (normally ``floquet_cycle=0``). Raw shots are kept; ``analyze()``
gives p(n mod 4) with and without post-selection on the heralds. ``FockPrepHeraldProgram`` also
runs under ``FockPopulationExperiment`` (``CharacterizationRunner(..., ExptProgram=
FockPrepHeraldProgram)``): a cycle sweep with heralds; its analysis then uses all shots, and
``labels_from_h5`` gives the herald cut.

Data of a ``FockPrepHeraldExperiment`` file (one row per read mode; ``xpts`` = ``ro_stors``):
    idata, qdata (points, reps * read_num): the raw shots. The readouts of one shot are
        interleaved: readout j of shot s is at index s * read_num + j (``read_num`` is in the
        ``config`` attribute). Readout order: the active-reset readouts at the start of the shot
        (2 with ef_reset), then the prep readouts (``prep_lanes(cfg.expt)``), then the two parity
        readouts. A readout reads g when I < ``device.readout.threshold``.
    p_all, p_herald, p_after_reset (points, 4): raw p(n mod 4) of the multiparity readout (no
        confusion correction): all shots; the shots where every readout right after a ladder step
        reads g; the shots where every final reset herald reads g (each_reset only). NaN where
        it does not apply.
    kept_herald, kept_after_reset (points,): the fraction of shots that the two cuts keep.
    step_g (points, steps): the fraction that reads g right after each herald step.
    avgi, avgq, amps, phases: the generic ``QsimExperiment`` averages of the last readout; not
        useful here.
Per-shot labels from a file: ``labels_from_h5(path, centers)``; a plot: ``display()``.

Herald cut in the population jobs (2026-11-14): ``FockPopulationExperiment`` run with
``FockPrepHeraldProgram`` and ``prep_herald`` "end" or "each" analyzes only the shots whose
heralds all read g (``multiparity.herald_keep``). Its readout correction must have the same cut:
``HeraldedConfusionMatrixExperiment`` is the multiparity confusion matrix as one job, Fock 0..3
in M1 through the population ladder (``HeraldedCalibrationProgram``) with one herald readout per
shot (a bare one for Fock 0, so that all points have the same readouts). Its file holds
``multiparity_confusion_matrix`` from the kept shots and ``multiparity_confusion_matrix_shot_counts``.
"""
import json
import os
import traceback

import h5py
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.qsim_base import QsimExperiment, _herald_readouts, saved_readout_mode
from fitting import plot_style
from fitting.qsim import f_postselect

HERALD_MODES = ("none", "end", "each", "each_reset")
RESET_PARAMS = dict(man_reset=False, storage_reset=False, coupler_reset=False, ef_reset=True,
                    pre_selection_reset=True, use_qubit_man_reset=False, pre_selection_parity=False)
RESET_LANE_KINDS = ("reset_ge", "reset_ef", "reset_herald")
assert MMAveragerProgram.active_reset_read_num(**RESET_PARAMS) == len(RESET_LANE_KINDS)


def ladders(expt):
    """-> the photon number of each M1 ladder the prep plays, in order: storages with n > 0, then M1."""
    numbers = [int(n) for n in expt.occupations[1:] if n > 0]
    return numbers + ([int(expt.occupations[0])] if expt.occupations[0] > 0 else [])


def prep_lanes(expt):
    """-> [(ladder, step, kind), ...]: the prep readout lanes in the order they are played.

    kind: "herald" (bare readout), or "reset_ge", "reset_ef", "reset_herald" (the three readouts
    of one qubit reset). The readout right after a step is "herald" or "reset_ge".
    """
    mode = expt.get("prep_herald", "none")
    if mode not in HERALD_MODES:
        raise ValueError(f"prep_herald={mode!r}; expected one of {HERALD_MODES}")
    if mode == "none":  # also for jobs without occupations (e.g. the multiparity confusion matrix)
        return []
    result = []
    for k, n in enumerate(ladders(expt)):
        for i in range(n):
            if mode == "each" or (mode == "end" and i == n - 1):
                result.append((k, i, "herald"))
            elif mode == "each_reset":
                result += [(k, i, kind) for kind in RESET_LANE_KINDS]
    return result


def shot_labels(cfg, data, centers=None):
    """-> dict of per-shot arrays of one job, shape (points, shots), from its cfg and data:

    n: n mod 4 of the multiparity readout (the last two lanes).
    step_g: (points, shots, steps), True where the readout right after a step reads g.
    herald: all readouts right after a step read g (None without heralds).
    after_reset: all final heralds of the resets read g (None without resets).
    no_f: neither parity readout reads f (None without ``centers``, the g/e/f cloud centers of a
        reference at the same readout phase).
    """
    cfg = AttrDict(cfg)
    read_num = int(cfg.get("read_num") or FockPrepHeraldProgram.readouts_per_shot(cfg))
    threshold = np.ravel(cfg.device.readout.threshold)[0]
    shots = f_postselect.lanes(data["idata"], data["qdata"], read_num)
    layout = prep_lanes(cfg.expt)
    start = _herald_readouts(cfg)
    reads_g = np.real(shots[..., start:start + len(layout)]) < threshold
    after_step = [j for j, (_, _, kind) in enumerate(layout) if kind in ("herald", "reset_ge")]
    final = [j for j, (_, _, kind) in enumerate(layout) if kind == "reset_herald"]
    no_f = None
    if centers is not None:
        no_f = ~(f_postselect.reads_f(shots[..., -2], centers)
                 | f_postselect.reads_f(shots[..., -1], centers))
    return dict(
        n=f_postselect.decode_n_mod4(shots[..., -2], shots[..., -1], threshold),
        step_g=reads_g[..., after_step],
        herald=reads_g[..., after_step].all(axis=-1) if after_step else None,
        after_reset=reads_g[..., final].all(axis=-1) if final else None,
        no_f=no_f,
    )


def select(labels, herald=False, after_reset=False, no_f=False):
    """-> the per-shot keep mask (points, shots) of ``shot_labels`` output for the chosen cuts.

    A cut that does not apply to the job (no heralds, no resets) keeps every shot; ``no_f`` needs
    labels made with ``centers``.
    """
    keep = np.ones(labels["n"].shape, bool)
    for use, key in ((herald, "herald"), (after_reset, "after_reset"), (no_f, "no_f")):
        if use and labels[key] is not None:
            keep &= labels[key]
        elif use and key == "no_f":
            raise ValueError("no_f cut: make the labels with the g/e/f centers")
    return keep


def labels_from_h5(path, centers=None):
    """-> (cfg, shot_labels) of a multiparity job file: ``FockPrepHeraldExperiment``,
    ``FockPopulationExperiment`` (with or without ``FockPrepHeraldProgram``), or
    ``MultiParityConfusionMatrixExperiment``. cfg is the file's config."""
    with h5py.File(path, "r") as f:
        cfg = AttrDict(json.loads(f.attrs["config"]))
        data = dict(idata=f["idata"][()], qdata=f["qdata"][()])
    return cfg, shot_labels(cfg, data, centers)


class FockPrepHeraldProgram(FockPopulationProgram):
    """``FockPopulationProgram`` with heralds (and resets) in the M1 ladders; see the module docstring."""

    @classmethod
    def readouts_per_shot(cls, cfg):
        return super().readouts_per_shot(cfg) + len(prep_lanes(cfg.expt))

    def _play_ladder(self, n, tag):
        mode = self.cfg.expt.get("prep_herald", "none")
        pulses = self.prep_man_fock_state(1, str(n))
        for i in range(n):
            self._play_gate_list(pulses[3 * i:3 * i + 3], prefix=f"{tag}_step{i}_")
            if mode == "each" or (mode == "end" and i == n - 1):
                self.post_selection_measure(parity=False, prefix=f"{tag}_herald{i}")
            elif mode == "each_reset":
                self.active_reset(prefix=f"{tag}_reset{i}", **RESET_PARAMS)

    def _prepare_occupation(self):
        expt = self.cfg.expt
        prep_lanes(expt)  # checks the mode
        if expt.get("prep_herald", "none") == "none":
            return super()._prepare_occupation()
        for storage, number in zip(expt.swap_stors, expt.occupations[1:]):
            if number == 0:
                continue
            self._play_ladder(int(number), f"population_prep_S{storage}")
            self.man_stor_swap(man_idx=1, stor_idx=storage)
        if expt.occupations[0] > 0:
            self._play_ladder(int(expt.occupations[0]), "population_prep_M1")


class FockPrepHeraldExperiment(QsimExperiment):
    """One prep and one herald mode, the read mode swept over ``ro_stors``; see the module docstring."""

    default_program = FockPrepHeraldProgram

    def sweep_axes(self):
        self.cfg.expt.swept_params = ["ro_stor"]
        return [("ro_stor", [int(r) for r in self.cfg.expt.ro_stors])]

    def analyze(self, data=None, **kwargs):
        """p(n mod 4) per read mode into ``data``: p_all; p_herald (all readouts right after a step
        read g); p_after_reset (all final heralds of the resets read g; each_reset only);
        kept_herald, kept_after_reset; step_g (points, steps): the fraction that reads g right
        after each step. NaN where it does not apply.

        go() calls this before it saves, so an error here is printed and the shots are saved.
        """
        data = self.data if data is None else data
        try:
            self._analyze(data)
        except Exception:
            print("[FockPrepHeraldExperiment] analysis failed; the shots are saved without it:")
            traceback.print_exc()
        return data

    def shot_labels(self, data=None, centers=None):
        """-> the module's ``shot_labels`` for this job."""
        return shot_labels(self.cfg, self.data if data is None else data, centers)

    def display(self, data=None, **kwargs):
        """Raw p(n mod 4) per read mode: all shots, and the herald cuts (kept fraction in the legend)."""
        data = self.data if data is None else data
        if "p_all" not in data:
            self.analyze(data)
        if "p_all" not in data:
            return None
        expt = self.cfg.expt
        series = [("all shots", data["p_all"], None)]
        if np.isfinite(data["kept_herald"]).any():
            series.append(("herald", data["p_herald"], data["kept_herald"]))
        if np.isfinite(data["kept_after_reset"]).any():
            series.append(("after reset", data["p_after_reset"], data["kept_after_reset"]))
        reads = [int(r) for r in expt.ro_stors]
        width = 0.8 / len(series)
        with plot_style.context():
            fig, axs = plt.subplots(1, len(reads), figsize=(3.2 * len(reads), 3.6), sharey=True, squeeze=False)
            for k, ro in enumerate(reads):
                ax = axs[0, k]
                for j, (label, p, kept) in enumerate(series):  # colors: the categorical slots in order
                    ax.bar(np.arange(4) + (j - (len(series) - 1) / 2) * width, p[k], width=width * 0.9,
                           label=label if kept is None else f"{label} (kept {kept[k]:.2f})")
                means = ", ".join(f"{f_postselect.mean_n(p[k]):.2f}" for _, p, _ in series)
                ax.set(title=f"read {'M1' if ro == 0 else f'S{ro}'}: <n> {means}", xlabel="n mod 4 (raw)",
                       xticks=range(4))
                ax.grid(False, axis="x")
                ax.legend(fontsize=7)
            axs[0, 0].set_ylabel("probability")
            name = os.path.basename(str(getattr(self, "fname", "")))
            plot_style.suptitle(fig, f"prep {list(expt.occupations)}, "
                                     f"prep_herald={expt.get('prep_herald', 'none')}  {name}")
            fig.tight_layout()
        return fig

    def _analyze(self, data):
        labels = self.shot_labels(data)
        n = labels["n"]
        points = n.shape[0]
        nan4, nan1 = np.full((points, 4), np.nan), np.full(points, np.nan)
        herald, after_reset = labels["herald"], labels["after_reset"]
        data.update(
            p_all=f_postselect.n_distribution(n),
            p_herald=nan4 if herald is None else f_postselect.n_distribution(n, herald),
            p_after_reset=nan4 if after_reset is None else f_postselect.n_distribution(n, after_reset),
            kept_herald=nan1 if herald is None else herald.mean(axis=-1),
            kept_after_reset=nan1 if after_reset is None else after_reset.mean(axis=-1),
            step_g=(labels["step_g"].mean(axis=1) if labels["step_g"].shape[-1]
                    else np.full((points, 1), np.nan)),
        )


class HeraldedCalibrationProgram(FockPrepHeraldProgram):
    """Fock n in M1 (n = 0..3) with exactly one herald readout per shot.

    Fock 1..3: the population ladder with the "end" herald, as the population jobs play it.
    Fock 0 has no ladder, so it gets one bare herald readout: all points of one calibration
    job then have the same readouts.
    """

    @classmethod
    def readouts_per_shot(cls, cfg):
        return FockPopulationProgram.readouts_per_shot(cfg) + 1

    def _prepare_occupation(self):
        expt = self.cfg.expt
        if expt.get("prep_herald", "none") != "end":
            raise ValueError("HeraldedCalibrationProgram plays the 'end' herald: set prep_herald='end'")
        if any(int(n) for n in expt.occupations[1:]):
            raise ValueError("the heralded calibration prepares M1 only")
        number = int(expt.occupations[0])
        if number == 0:
            self.post_selection_measure(parity=False, prefix="population_prep_M1_herald0")
        else:
            self._play_ladder(number, "population_prep_M1")


class HeraldedConfusionMatrixExperiment(MultiParityConfusionMatrixExperiment):
    """The multiparity confusion matrix with the prep herald cut, as one job.

    Points: Fock 0, 1, 2, 3 in M1 (``occupations`` [n, 0, ...] over ``swap_stors``), read M1,
    no Floquet cycles, one herald readout per shot (``HeraldedCalibrationProgram``). Needs
    ``prep_herald="end"``, the herald of the population jobs it corrects. analyze() gives
    ``multiparity_confusion_matrix`` from the kept shots (``multiparity.herald_keep``) and
    ``multiparity_confusion_matrix_shot_counts``, the kept shots of each row (read by
    ``agp.calibration_shots``).
    Give its file to the population jobs as ``multiparity_confusion_matrix``, as for
    ``MultiParityConfusionMatrixExperiment``.
    """

    default_program = HeraldedCalibrationProgram

    def acquire(self, progress=False, debug=False):
        expt = self.cfg.expt
        if saved_readout_mode(expt) != "multiparity":
            raise ValueError("HeraldedConfusionMatrixExperiment requires readout='multiparity'")
        if expt.get("cond_sec_phase", 90) != -90 or expt.get("phase_second_pulse", 180) != 180:
            raise ValueError("HeraldedConfusionMatrixExperiment requires cond_sec_phase=-90 "
                             "and phase_second_pulse=180 for n = b0 + 2*b1")
        if expt.get("prep_herald", "none") != "end":
            raise ValueError("HeraldedConfusionMatrixExperiment requires prep_herald='end'")
        expt.prepulse = False
        expt.postpulse = True
        expt.ro_stor = 0
        expt.floquet_cycle = 0
        expt.multiparity_confusion_matrix = None
        # Not the parent's acquire: that one sweeps the prepulse Fock states.
        return QsimExperiment.acquire(self, progress=progress, debug=debug)

    def sweep_axes(self):
        """One axis: ``occupations`` [n, 0, ...] for n = 0, 1, 2, 3."""
        storages = list(self.cfg.expt.swap_stors)
        self.cfg.expt.swept_params = ["occupations"]
        return [("occupations", [[n] + [0] * len(storages) for n in range(4)])]

    def shape_data(self, axes, points):
        data = QsimExperiment.shape_data(self, axes, points)
        data["xpts"] = np.arange(4)  # the prepared number, as the parent's analyze() needs
        return data

    def analyze(self, data=None, **kwargs):
        """The parent's matrix (from the kept shots), and the kept shots of each row."""
        super().analyze(data=data, **kwargs)
        prepared = np.asarray(self.data["xpts"]).reshape(-1).astype(int)
        counts = np.empty(4)
        counts[prepared] = np.asarray(self.data["shot_counts"]).reshape(-1)
        self.data["multiparity_confusion_matrix_shot_counts"] = counts
        return self.data

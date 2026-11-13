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
from experiments.qsim.qsim_base import QsimExperiment, _herald_readouts
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
        colors = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3
        reads = [int(r) for r in expt.ro_stors]
        width = 0.8 / len(series)
        fig, axs = plt.subplots(1, len(reads), figsize=(3.2 * len(reads), 3.6), sharey=True,
                                squeeze=False, facecolor="#fcfcfb")
        for k, ro in enumerate(reads):
            ax = axs[0, k]
            ax.set_facecolor("#fcfcfb")
            for j, (label, p, kept) in enumerate(series):
                ax.bar(np.arange(4) + (j - (len(series) - 1) / 2) * width, p[k], width=width * 0.9,
                       color=colors[j], label=label if kept is None else f"{label} (kept {kept[k]:.2f})")
            means = ", ".join(f"{f_postselect.mean_n(p[k]):.2f}" for _, p, _ in series)
            ax.set_title(f"read {'M1' if ro == 0 else f'S{ro}'}: <n> {means}", fontsize=9)
            ax.set_xticks(range(4))
            ax.set_xlabel("n mod 4 (raw)", fontsize=9)
            ax.tick_params(labelsize=8, colors="#52514e")
            ax.grid(True, axis="y", color="#e1e0d9", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False, fontsize=7)
        axs[0, 0].set_ylabel("probability", fontsize=9)
        name = os.path.basename(str(getattr(self, "fname", "")))
        fig.suptitle(f"prep {list(expt.occupations)}, prep_herald={expt.get('prep_herald', 'none')}  {name}",
                     x=0.01, ha="left", fontsize=10)
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

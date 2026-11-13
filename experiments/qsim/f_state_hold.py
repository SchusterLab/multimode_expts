"""Hold |f,n> (qubit in f, n photons in M1) for a time t, then read g/e/f in one shot.

This checks the |f,n> -> |g,n-1> decay that limits M1 Fock N >= 2 preparation
(suspected TLS near 11.96 GHz; vault report
``Lab/guan/AGP/2026/10/2026-10-29_TLS_12GHz_report``). On 2026-10-28/29 each hold
point was its own ``HistogramExperiment`` job
(``measurement_notebooks/guan/yoko450ua_calibration.py``, cells 8, 10, 11). Here one
job has the g/e/f reference clouds and all (n, wait) points, so the assignment
matrix and the holds are measured together.

- ``FStateHoldProgram``: ``HistogramProgram`` with one sweep point as its prepulse.
  It is the same program as on 10-28/29, so the pulse timing is the same.
- ``FStateHoldExperiment``: the sweep (``QsimExperiment``: one program per point,
  raw shots kept), the analysis (``fitting/qsim/f_hold.py``) and the plot.

``cfg.expt`` keys:
    hold_ns: photon numbers n, e.g. [0, 1, 2]. n = 0 is the control: |f,0> decays
        only to e (T1 of e-f).
    hold_waits: waits after the e_n-f_n pulse, in us, e.g. [0, 2, 5, 10, 20, 40, 60].
    reps, relax_delay (us; with photons in M1 and no reset, 2500 as on 10-28),
    qubits=[0], active_reset (default False; with True the reset plays before the
    prepulse and its readouts are skipped in the analysis).

Point order (``hold_points``): the references g, e, f, then n-major, wait-minor.
The references use the standard qubit ge and ef pi pulses, as the
``HistogramExperiment`` references of 10-28; the holds use the multiphoton table.
"""
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.MM_base import MM_base, MMAveragerProgram
from experiments.qsim.qsim_base import QsimExperiment
from experiments.single_qubit.single_shot import HistogramProgram
from fitting.qsim import f_hold

REFERENCES = ("g", "e", "f")


def hold_points(hold_ns, hold_waits):
    """-> the sweep points in order: [state, 0.0] for g, e, f, then [n, wait] per n and wait."""
    return ([[state, 0.0] for state in REFERENCES]
            + [[int(n), float(t)] for n in hold_ns for t in hold_waits])


class FStateHoldProgram(HistogramProgram):
    """One point of the hold scan: ``hold_points(hold_ns, hold_waits)[hold_index]``.

    A reference (g, e or f with the standard qubit pi pulses), or |f,n> held for a
    wait: the Fock n ladder (``MM_base.prep_man_fock_state``), g_n-e_n, e_n-f_n,
    then the wait.
    """

    def __init__(self, soccfg, cfg):
        # The point keys set in body() go into this copy of cfg.expt only, not into
        # the Experiment's config (AttrDict copies only the top level).
        cfg = AttrDict(cfg)
        cfg.expt = AttrDict(cfg.expt)
        super().__init__(soccfg, cfg)

    @classmethod
    def readouts_per_shot(cls, cfg):
        """-> readouts per shot: the active-reset readouts (if any), then the final one."""
        count = 1
        if cfg.expt.get("active_reset", False):
            params = MM_base.get_active_reset_params(cfg)
            count += MMAveragerProgram.active_reset_read_num(**params)
        return count

    def hold_pulses(self, n, wait):
        """-> the gate list that prepares |f,n> and holds it for ``wait`` us."""
        pulses = self.prep_man_fock_state(1, str(n))
        pulses += [["multiphoton", f"g{n}-e{n}", "pi", 0],
                   ["multiphoton", f"e{n}-f{n}", "pi", 0]]
        if wait > 0:
            pulses.append(["wait", wait])
        return pulses

    def body(self):
        expt = self.cfg.expt
        state, wait = hold_points(expt.hold_ns, expt.hold_waits)[expt.hold_index]
        is_reference = state in REFERENCES
        expt.pulse_e = state in ("e", "f")
        expt.pulse_f = state == "f"
        expt.prepulse = not is_reference
        expt.gate_based = True
        expt.pre_sweep_pulse = [] if is_reference else self.hold_pulses(int(state), float(wait))
        super().body()


class FStateHoldExperiment(QsimExperiment):
    """g/e/f references and |f,n> holds in one job; see the module docstring."""

    default_program = FStateHoldProgram

    def sweep_axes(self):
        points = hold_points(self.cfg.expt.hold_ns, self.cfg.expt.hold_waits)
        return [("hold_index", list(range(len(points))))]

    def analyze(self, data=None, **kwargs):
        """Populations of every hold point and one decay fit per n, into ``data``.

        Keys added: hold_ns, hold_waits, blob_centers_i/_q (3,), assignment_matrix
        (3, 3), raw_populations and populations (n, wait, g/e/f), the decay fit per n
        (decay_rate, decay_rate_err, decay_amplitude, decay_offset, share_to_g,
        rate_to_g; 1/us), omega_m1 and nu (MHz, nu_n = E(f,n) - E(g,n-1)).

        go() calls this before it saves, so an unexpected error here is printed
        and the shots are saved without the analysis keys.
        """
        data = self.data if data is None else data
        try:
            self._analyze(data)
        except Exception:
            print("[FStateHoldExperiment] analysis failed; the shots are saved without it:")
            traceback.print_exc()
        return data

    def _analyze(self, data):
        expt = self.cfg.expt
        ns = [int(n) for n in expt.hold_ns]
        waits = [float(t) for t in expt.hold_waits]
        read_num = int(self.cfg.get("read_num") or FStateHoldProgram.readouts_per_shot(self.cfg))

        # The final readout is the last lane of each shot.
        shots = np.asarray(data["idata"]) + 1j * np.asarray(data["qdata"])
        shots = shots[:, read_num - 1::read_num]
        refs, holds = shots[:len(REFERENCES)], shots[len(REFERENCES):]

        centers = f_hold.blob_centers(refs)
        A = f_hold.assignment_matrix(refs, centers)
        raw = np.array([f_hold.assigned_fractions(s, centers) for s in holds])
        raw = raw.reshape(len(ns), len(waits), 3)
        pop = f_hold.corrected_populations(raw, A)
        fits = [f_hold.decay_fit(waits, pop[i]) for i in range(len(ns))]

        qubit = self.cfg.device.qubit
        omega, nu = f_hold.transition_frequencies(
            np.ravel(qubit.f_ge)[0], np.ravel(qubit.f_ef)[0],
            self.cfg.device.multiphoton.pi["fn-gn+1"].frequency, ns)

        data.update(
            hold_ns=np.array(ns), hold_waits=np.array(waits),
            blob_centers_i=centers.real, blob_centers_q=centers.imag,
            assignment_matrix=A, raw_populations=raw, populations=pop,
            decay_rate=np.array([f["rate"] for f in fits]),
            decay_rate_err=np.array([f["rate_err"] for f in fits]),
            decay_amplitude=np.array([f["amplitude"] for f in fits]),
            decay_offset=np.array([f["offset"] for f in fits]),
            share_to_g=np.array([f["share_to_g"] for f in fits]),
            rate_to_g=np.array([f["rate_to_g"] for f in fits]),
            omega_m1=omega, nu=nu,
        )

    def display(self, data=None, **kwargs):
        """P(g), P(e), P(f) against the wait, one panel per n, with the P(f) fit."""
        data = self.data if data is None else data
        if "populations" not in data:
            self.analyze(data)
        if "populations" not in data:
            return None

        ns, waits, pop = data["hold_ns"], np.asarray(data["hold_waits"]), data["populations"]
        colors = {"g": "#2a78d6", "e": "#eb6834", "f": "#1baf7a"}  # categorical slots 1-3
        markers = {"g": "o", "e": "s", "f": "^"}
        ink, muted, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

        fig, axs = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.0), sharey=True,
                                squeeze=False, facecolor="#fcfcfb")
        for i, n in enumerate(ns):
            ax = axs[0, i]
            ax.set_facecolor("#fcfcfb")
            for j, state in enumerate("gef"):
                # P(f) gets markers only: its fit line is drawn below.
                ax.plot(waits, pop[i, :, j], marker=markers[state], ms=6, lw=0 if state == "f" else 1,
                        color=colors[state], mec="#fcfcfb", mew=1, label=f"P({state})")
            title = f"hold |f,{n}>"
            if np.isfinite(data["nu"][i]):
                title += f", nu_{n} = {data['nu'][i]:.1f} MHz"
            rate = data["decay_rate"][i]
            if np.isfinite(rate):
                tt = np.linspace(0, waits.max(), 200)
                ax.plot(tt, data["decay_amplitude"][i] * np.exp(-rate * tt) + data["decay_offset"][i],
                        "-", lw=1.5, color=colors["f"], label="P(f) fit")
                title += f"\ntau {1 / rate:.1f} us, share to g {data['share_to_g'][i]:.2f}"
                if data["share_to_g"][i] > 0.05:
                    title += f", tau to g {1 / data['rate_to_g'][i]:.1f} us"
            ax.set_title(title, fontsize=9, color=ink)
            ax.set_xlabel("wait (us)", fontsize=9, color=ink)
            ax.grid(True, color=grid, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(axis)
            ax.tick_params(colors=muted, labelsize=8)
        axs[0, 0].set_ylabel("population (assignment-corrected)", fontsize=9, color=ink)
        name = os.path.basename(str(getattr(self, "fname", "")))
        fig.suptitle(f"|f,n> hold  {name}", x=0.01, ha="left", fontsize=10, color=ink)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig

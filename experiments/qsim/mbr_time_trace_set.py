# -*- coding: utf-8 -*-
"""A plain set of MBR time traces, diagonal or off-diagonal, kept together.

An assembled class (docs/qsim/mbr_redesign.md, section 5): one
:class:`MBRTimeTraceExperiment` per (initial, final) pair. It gives the
returns as one array and nothing more: no spectrum, no phase rebuilding.
Its use is the off-diagonal traces of the old notebook section 7-2 ("D72")
campaigns, whose diagonal traces form an ``MBRDisorderEnsembleExperiment``
(docs/qsim/mbr_step7_plan.md, decision 2: only diagonal traces are
canonical). It keeps those traces loadable without a dedicated analysis.

    traces = MBRTimeTraceSetExperiment.from_manifest(path)
    traces.analyze()                       # data.A: (n_pairs, n_cycles), as acquired
    traces.display()
"""
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment


class MBRTimeTraceSetExperiment(AssembledExperiment):
    """TimeTrace jobs of any (initial, final) pairs on one cycle grid."""

    child_class = MBRTimeTraceExperiment

    def __init__(self, pairs, cycles, calibration=None, realization_record=None, notes=""):
        super().__init__(notes=notes)
        self.pairs = [(tuple(int(n) for n in initial), tuple(int(n) for n in final))
                      for initial, final in pairs]
        self.cycles = [int(n) for n in cycles]
        self.calibration = calibration
        self.realization_record = (None if realization_record is None
                                   else dict(realization_record))

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", calibration=None,
                      realization_record=None):
        """Assemble loaded TimeTrace jobs; all must share one cycle grid."""
        children = list(children)
        if not children:
            raise ValueError("a time-trace set needs at least one TimeTrace job")
        cycles = list(children[0].cfg.expt.floquet_cycles)
        for child in children:
            if list(child.cfg.expt.floquet_cycles) != cycles:
                raise ValueError(f"{child.initial_occupation}: different Floquet cycles")
        pairs = [(child.initial_occupation, child.final_occupation) for child in children]
        if len(set(pairs)) != len(pairs):
            raise ValueError("each (initial, final) pair must appear once")
        traces = cls(pairs, cycles, calibration=calibration,
                     realization_record=realization_record, notes=notes)
        traces.children = children
        traces.job_ids = list(job_ids)
        traces._check_children()
        return traces

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment

        kwargs = dict(realization_record=manifest["parameters"].get("realization_record"))
        if manifest.get("calibration_manifest"):
            kwargs["calibration"] = MBRCalibrationSetExperiment.from_manifest(
                manifest["calibration_manifest"])
        return kwargs

    def analyze(self, **kwargs):
        """Stack the returns. ``data.A`` is complex, ``(n_pairs, n_cycles)``, as acquired."""
        self._check_children()
        for child in self.children:
            if "complex_return" not in child.data:
                child.analyze()
        self.data = AttrDict(dict(
            initial=[initial for initial, _ in self.pairs],
            final=[final for _, final in self.pairs],
            cycles=np.asarray(self.cycles),
            A=np.asarray([child.data["complex_return"] for child in self.children],
                         dtype=complex),
        ))
        return self.data

    def display(self, **kwargs):
        """|A| of every trace vs cycles, one line each."""
        if "A" not in self.data:
            self.analyze()
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        for (initial, final), row in zip(self.pairs, self.data.A):
            ax.plot(self.data.cycles, np.abs(row), lw=1,
                    label=f"{initial}" if initial == final else f"{final}<-{initial}")
        ax.set(xlabel="Floquet cycles", ylabel="|A|", title=f"{len(self.pairs)} traces")
        if len(self.pairs) <= 16:
            ax.legend(fontsize=7, ncol=2)
        return fig

    # -- persistence ------------------------------------------------------

    def calibration_manifest(self):
        return None if self.calibration is None else self.calibration.manifest_path

    def manifest_parameters(self):
        return dict(pairs=[[list(initial), list(final)] for initial, final in self.pairs],
                    cycles=self.cycles,
                    realization_record=self.realization_record)

    def assembled_arrays(self):
        return dict(initial=np.asarray(self.data.initial, dtype=int),
                    final=np.asarray(self.data.final, dtype=int),
                    cycles=self.data.cycles,
                    A=self.data.A)

"""Combine existing heralded preparation, sequential readout and ensemble loops.

The ordinary heralded classes also provide the zero-detuning walk check. The
ensemble subclasses reuse the DMEM upload, pointer and Floquet implementation;
only readout layout and full realization-preserving labels are added here.
"""

import numpy as np

from experiments.qsim.fock_population_ensemble import FockPopulationEnsembleProgram
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
    SequentialMultiparityProgramMixin,
    readout_layout,
)
from experiments.qsim.fock_prep_herald import FockPrepHeraldProgram, prep_lanes


def heralded_readout_layout(cfg, minimum_prep_heralds=0):
    """Insert preparation readouts before every sequential science pair."""
    layout = readout_layout(cfg)
    preparation_kinds = [kind for _, _, kind in prep_lanes(cfg.expt)]
    preparation_kinds += ["herald"] * max(0, minimum_prep_heralds - len(preparation_kinds))
    preparation_readout_count = len(preparation_kinds)
    science_pairs = np.asarray(layout["readout_lane_pairs"], dtype=int)
    preparation_readout_start = int(science_pairs[0, 0])
    layout.update(read_num=layout["read_num"] + preparation_readout_count,
                  readout_lane_pairs=(science_pairs + preparation_readout_count).tolist(),
                  preselection_lanes=[*layout["preselection_lanes"],
                                      *(preparation_readout_start + index 
                                        for index, kind in enumerate(preparation_kinds)
                                        if kind in ("herald", "reset_herald"))],)
    return layout


class HeraldedSequentialMultiparityProgramMixin(SequentialMultiparityProgramMixin):
    """Add herald lane counts without a Program parent, avoiding diamond inheritance."""

    @classmethod
    def readouts_per_shot(cls, cfg):
        return heralded_readout_layout(cfg)["read_num"]


class FockPopulationSequentialHeraldProgram(HeraldedSequentialMultiparityProgramMixin,
                                           FockPrepHeraldProgram):
    """Use one preparation Program to avoid diamond inheritance.

    The shared readout mixin has no Program parent.
    """


class FockPopulationSequentialHeraldExperiment(FockPopulationSequentialReadoutExperiment):
    """Ordinary population sweep with a preparation-aware sequential decoder."""

    default_program = FockPopulationSequentialHeraldProgram

    def readout_layout(self):
        return heralded_readout_layout(self.cfg)


class FockPopulationSequentialEnsembleProgram(HeraldedSequentialMultiparityProgramMixin,
                                            FockPopulationEnsembleProgram):
    """Use one ensemble Program to avoid diamond inheritance.

    The shared readout mixin has no Program parent.
    """


class FockPopulationSequentialEnsembleExperiment(FockPopulationSequentialHeraldExperiment):
    """Keep one Experiment parent to avoid diamond inheritance.

    acquire() and shape_data() adapt the small acquisition/metadata methods from
    FockPopulationEnsembleExperiment. They are copied here deliberately to avoid
    inheriting two Experiment classes with the same FockPopulationExperiment base.
    DMEM upload, Floquet pulses and sequential decoding remain shared implementations.
    """

    default_program = FockPopulationSequentialEnsembleProgram

    def acquire(self, progress=False, debug=False):
        """Adapt FockPopulationEnsembleExperiment.acquire to avoid diamond inheritance."""
        self.cfg.expt.reps = len(self.cfg.expt.disorder_detunings_mhz)
        return super().acquire(progress=progress, debug=debug)

    def shape_data(self, axes, points):
        """Copy ensemble metadata shaping here to avoid diamond inheritance.

        Adapted from FockPopulationEnsembleExperiment.shape_data; the single
        sequential parent creates the lane metadata before full labels are stored.
        """
        data = super().shape_data(axes, points)
        detunings = np.asarray(self.cfg.expt.disorder_detunings_mhz, dtype=float)
        realization_indices = np.arange(len(detunings), dtype=int)
        data.update(occupations=np.asarray(self.cfg.expt.occupations, dtype=int),
                    swap_stors=np.asarray(self.cfg.expt.swap_stors, dtype=int),
                    cycles=np.asarray(data["xpts"], dtype=int),
                    time_us=np.asarray(data["xpts"]) * self.prog.calculate_floquet_cycle_us(),
                    disorder_detunings_mhz=detunings,
                    disorder_frequency_words=self.prog.disorder_frequency_words.copy(),
                    realization_indices=realization_indices,
                    shot_realization_indices=np.broadcast_to(
                        realization_indices, (len(data["xpts"]), len(detunings))).copy(),
                    disorder_dmem_address=np.asarray(self.prog.disorder_dmem_address))
        if self.cfg.expt.get("disorder_seed") is not None:
            data["disorder_seed"] = np.asarray(self.cfg.expt.disorder_seed)
        self._store_labels(data)
        return data

    def _store_labels(self, data):
        self.data = data
        decoded_points = [self.decode_readouts(point, accepted_only=False)
                          for point in range(len(data["idata"]))]
        data["readout_labels"] = np.asarray(
            [decoded_point["labels"] for decoded_point in decoded_points], dtype=np.uint8)
        data["herald_keep"] = np.asarray(
            [decoded_point["keep"] for decoded_point in decoded_points], dtype=bool)
        if self.cfg.expt.get("disorder_patterns") is not None:
            data["disorder_patterns"] = np.asarray(self.cfg.expt.disorder_patterns, dtype=float)

    def analyze(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        if "idata" in self.data:
            super().analyze(**kwargs)
            self._store_labels(self.data)
        return self.data

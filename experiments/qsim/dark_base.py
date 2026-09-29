# -*- coding: utf-8 -*-
"""Old names kept until the renames of step 10F, and the RAverager template.

Since steps 10B-10E the code these names stood for lives elsewhere
(``docs/qsim/program_tree_plan.md``): the sweep driver and
``analyze_multiparity`` in ``QsimBaseExperiment``, the template in
``QsimBaseProgram``, the Floquet and dark-mode methods in ``FloquetProgram``
and ``DarkModeProgram``.

``DarkBaseExperiment``
    The old name of the sweep driver, with no methods of its own.

``DarkBaseProgram`` / ``DarkBaseRProgram``
    ``DarkBaseProgram`` is the old name of ``DarkModeProgram``
    (``dark_mode_encoding``), kept until step 10F; no live Program uses it.
    The template (reset, prepulse, the ``core_pulses`` hook each measurement
    overrides, postpulse, readout) is ``QsimBaseProgram``'s. The R variant
    is the RAverager counterpart for hardware depth sweeps; it shares ``body`` and takes its pulse methods by explicit
    assignment, and the asymmetry in *which* it takes is deliberate: it keeps
    MM_base's ``man_reset`` (see ``DarkBaseRProgram``).

Provenance note
---------------
``DarkBaseExperiment`` is a recorded name: saved HDF5 files are called
``JOB-<id>_DarkBaseExperiment.h5`` and the acquisition notebooks submit with
``ExptClass=...DarkBaseExperiment``. The class name is therefore fixed, and
this move deliberately keeps it -- only the module changed, and
``floquet_dark_mode_readout`` still imports it so its old address resolves.
"""
from slab import AttrDict

from experiments.MM_base import MMRAveragerProgram
from experiments.qsim.dark_mode_encoding import DarkModeProgram
from experiments.qsim.qsim_base import (
    QsimBaseExperiment,
    QsimBaseProgram,
    classify_two_parity_readouts,  # re-exported: its old address
    readout_lane_count,  # re-exported: deprecated/legacy_mbr imports it from here
    readout_mode,
)


class DarkBaseExperiment(QsimBaseExperiment):
    """The old name of the qsim sweep driver; it has no methods of its own.

    Its ``acquire`` (the same loop as ``QsimBaseExperiment``'s, with the
    readout count that knows multiparity) and ``analyze_multiparity`` moved
    into ``QsimBaseExperiment`` in step 10E. Saved files and the notebooks
    still carry the name; step 10F renames the driver and migrates them.
    """

class DarkBaseProgram(DarkModeProgram):
    """The old name of ``DarkModeProgram``; no live Program uses it.

    Its template (``initialize``, ``body``) and the manipulate-mode pulses
    moved into ``QsimBaseProgram`` in step 10B, and its two mixins became the
    classes ``FloquetProgram`` and ``DarkModeProgram`` in step 10C
    (``docs/qsim/program_tree_plan.md``). It stays until the renames of step
    10F, for the deprecated modules and the old address in
    ``floquet_dark_mode_readout``.
    """


class DarkBaseRProgram(MMRAveragerProgram):
    """RAverager counterpart of ``DarkBaseProgram``.

    The pulse-building methods below do not depend on the AveragerProgram
    software loop, so both program types use the same implementations.
    Concrete RAverager programs only need to define ``core_pulses`` and
    ``update``.
    """

    _pre_selection_filtering = True

    retrieve_swap_parameters = QsimBaseProgram.retrieve_swap_parameters #borrowing methods
    _initialize_floquet_pulses = QsimBaseProgram._initialize_floquet_pulses
    # Two of the three manipulate-mode methods, by assignment rather than
    # inheritance: inheriting from QsimBaseProgram would also shadow MM_base's
    # ``man_reset``, which is the one ``active_reset`` plays here.
    prep_man_fock_state = QsimBaseProgram.prep_man_fock_state
    multi_parity_readout = QsimBaseProgram.multi_parity_readout
    body = QsimBaseProgram.body #borrowing methods

    def __init__(self, soccfg, cfg):
        readout_mode(cfg.expt)  # as QsimBaseProgram: refuse what it cannot play
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg)

    readouts_per_shot = QsimBaseProgram.readouts_per_shot

    def initialize(self):
        self.MM_base_initialize()

        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        man_mode_no = self.cfg.expt.get("man_mode_no", 1)
        self.man_mode_idx = man_mode_no - 1

        self._initialize_floquet_pulses()

        self.sync_all(200)



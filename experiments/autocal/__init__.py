"""Automated calibration: readout -> qubit -> manipulate photon ladder -> M1 properties -> storage swap.

The steps reproduce measurement_notebooks/QEC/single_qubit_autocalibrate_v2.ipynb and
multiphoton_calibration_v2.ipynb (settings and order), with a fit-quality check before every config change and a
stop at the first failed check. Design and usage: docs/autocal.md.

Usage (on the measurement PC, main checkout)::

    from experiments.autocal import AutocalRun, default_phases
    run = AutocalRun(station, client, default_phases(station), out_dir=station.autocalib_path, coupler_A=0.0003)
    run.budget()                       # time estimate, no job
    run.run()                          # or run.run(phases=['qubit']), run.run(start_at='ladder.f0-g1.qspec')

Module names start with '_' on purpose: ``import experiments`` imports every module of every subpackage and
copies its classes into the ``experiments`` namespace; it skips names that start with '_'.
"""
import yaml

from experiments.autocal import _ladder, _manipulate, _qubit, _storage
from experiments.autocal._core import AutocalRun, AutocalStop, Change, Outcome, Phase, Step


def default_phases(station, ladder_top=2, storage_mode=4, ladder_bare_frame=False):
    """readout, qubit, ladder (up to |ladder_top>), manipulate, storage (M1-S<storage_mode> swap)."""
    return (_qubit.phases() + _ladder.phases(station, top=ladder_top, bare_frame=ladder_bare_frame) + _manipulate.phases()
            + _storage.phases(mode=storage_mode))


def seed_sections(station, hardware_config_path, sections=('manipulate', 'multiphoton')):
    """Copy device.<section> (in memory) from another hardware config file, e.g. the last calibration at the same
    coupler current, as starting values for the ladder and M1 steps. Returns {section: (old, new)} for the record."""
    with open(hardware_config_path) as f:
        other = yaml.safe_load(f)
    out = {}
    for s in sections:
        out[s] = (dict(station.hardware_cfg.device[s]), other['device'][s])
        station.hardware_cfg.device[s] = type(station.hardware_cfg)(other['device'][s])
    return out


__all__ = ['AutocalRun', 'AutocalStop', 'Change', 'Outcome', 'Phase', 'Step', 'default_phases', 'seed_sections']

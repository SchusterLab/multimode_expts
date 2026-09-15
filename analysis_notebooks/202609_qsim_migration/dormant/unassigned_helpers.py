"""Helpers with no identified owner, parked by the stage-2 notebook split.

Both came from `measurement_notebooks/jonginn/data_postprocess.ipynb` cell 4,
the 1563-line cell that defined every helper for every section. Neither has a
caller anywhere in that notebook, and neither is a duplicate: the body of each
differs from every same-named definition in `qsim_experiments.ipynb`. So they
could be placed neither by use nor by an identified surviving copy.

They are parked rather than deleted because lack of a known caller does not
establish duplication.

- `error_amp_floquet_postproc` -- 11 code lines here; the two identical copies
  in `qsim_experiments.ipynb` cells 78 and 107 are 19 lines (similarity 0.73).
  The active copy lives in
  `experiments/qsim/notebook_helpers/floquet_calibration_hooks.py`.
- `flatten_exp_lists` -- 19 code lines here; cell 152 has a 6-line version
  (similarity 0.48) and cell 179 a 9-line version (similarity 0.43). The
  cell-152 version is the one the active floquet bare-readout section uses;
  the cell-179 version went to the dormant dark-mode notebook.

Nothing imports this module. Deleting a function here means deciding it is
genuinely dead, which this pass did not do.
"""

import numpy as np


def error_amp_floquet_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    opt_val = expt.data['fit_avgi'][2]
    stor_name = 'M1-S' + str(expt.cfg.expt.stor_mode_no)
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_floquet.update_gain(stor_name, opt_val)
        print(f'Updated gain for {stor_name} to {opt_val}')
    elif expt.cfg.expt.parameter_to_test == 'frequency':
        station.ds_floquet.update_freq(stor_name, opt_val)
        print(f'Updated frequency for {stor_name} to {opt_val}')
    station.snapshot_floquet_storage_swap(update_main=False)


def flatten_exp_lists(items, container_types=(list, tuple, set)):
    for x in items:
        if isinstance(x, container_types):
            yield from flatten_exp_lists(x, container_types)
        else:
            yield x


# ---- module-level constants used by the helpers above ----
IDENTITY_PARITY_CONFUSION = [1.0, 0.0, 0.0, 1.0]
confusion_matrix_manual = np.array([
    [0.854, 0.056, 0.042, 0.049],
    [0.047, 0.800, 0.044, 0.110],
    [0.072, 0.085, 0.759, 0.084],
    [0.095, 0.104, 0.050, 0.750],
])
_mod_vals = np.array([0, 1, 2, 3])
_parity_first_signs  = 1 - 2 * (_mod_vals % 2)    # [+1, -1, +1, -1]
_parity_second_signs = 1 - 2 * (_mod_vals // 2)   # [+1, +1, -1, -1]
DarkParams = namedtuple("DarkParams", ["swap_stors", "swap_man_dark", "dark_swap_order"])
DARK_PARAMS = None


# ---- one-time setup ----
set_confusion_matrix(confusion_matrix_manual)

"""One helper with no identified owner, parked by the stage-2 notebook split.

`flatten_exp_lists` came from `measurement_notebooks/jonginn/data_postprocess.ipynb`
cell 4, the 1563-line cell that defined every helper for every section.

It survived the placement pass with no caller: nothing in that notebook
references it, even after closing each destination's helper set over its
internal dependencies. Nor is it a duplicate. Its body is 19 code lines;
`qsim_experiments.ipynb` cell 152 has a 6-line version (similarity 0.48) and
cell 179 a 9-line version (0.43). The cell-152 version is what the active
floquet bare-readout section uses, and the cell-179 version went to the
dormant dark-mode notebook. So this third variant could be placed neither by
use nor by an identified surviving copy.

It is parked rather than deleted because lack of a known caller does not
establish duplication -- and because the first pass at this split got exactly
that judgement wrong. It searched for call syntax `name(` and concluded that
`error_amp_floquet_preproc`, `error_amp_floquet_postproc`,
`sideband_scramble_preproc` and `get_floquet_parameters` had no callers. They
did: the dormant pulse-scratch section passes two of them as
`preprocessor=` / `postprocessor=` arguments, by bare name. All four now live
in `pulse_scratch.py`, and this file is what is genuinely left over.

Nothing imports this module. Deleting the function below means deciding it is
dead, which this pass did not do.
"""


def flatten_exp_lists(items, container_types=(list, tuple, set)):
    for x in items:
        if isinstance(x, container_types):
            yield from flatten_exp_lists(x, container_types)
        else:
            yield x

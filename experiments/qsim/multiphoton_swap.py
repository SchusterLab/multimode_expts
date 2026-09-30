# -*- coding: utf-8 -*-
"""Pulse sequences of the N-photon M1-storage swap calibration.

Used by `measurement_notebooks/202609_qsim_migration/multiphoton_calibration.py`.
Moved out of `experiments/qsim/notebook_helpers/multiphoton_calibration.py` in
MBR redesign step 9C.
"""
from experiments.MM_dual_rail_base import MM_dual_rail_base


def swap_pulse_sequences(station, photon_number):
    """The preparation of |g,N> and the endpoint decoder of one swap row.

    The decoder uses only the top two transitions, f_(N-1) <-> g_N and
    e_(N-1) <-> f_(N-1), so only an exact return to |g,N> is bright; lower M1
    occupations stay dark rather than read as a successful return.

    -> dict: ``prep_descriptions``; ``extra_prep`` (what
    ErrorAmplificationProgram does not supply itself: all but the first three);
    ``endpoint_decoder_descriptions``; and ``prep_pulse``, ``endpoint_decoder``
    as compiled pulse tables.
    """
    prep_descriptions = [
        ['qubit', 'ge', 'pi', 0.0],
        ['qubit', 'ef', 'pi', 0.0],
        ['man', 'M1', 'pi', 0.0],
    ]
    for n in range(1, photon_number):
        prep_descriptions += [
            ['multiphoton', f'g{n}-e{n}', 'pi', 0.0],
            ['multiphoton', f'e{n}-f{n}', 'pi', 0.0],
            ['multiphoton', f'f{n}-g{n + 1}', 'pi', 0.0],
        ]
    n_top = photon_number - 1
    endpoint_decoder_descriptions = [
        ['multiphoton', f'f{n_top}-g{n_top + 1}', 'pi', 0.0],
        ['multiphoton', f'e{n_top}-f{n_top}', 'pi', 0.0],
    ]
    builder = MM_dual_rail_base(station.hardware_cfg, soccfg=station.soccfg)
    return {
        'prep_descriptions': prep_descriptions,
        'extra_prep': prep_descriptions[3:],
        'endpoint_decoder_descriptions': endpoint_decoder_descriptions,
        'prep_pulse': builder.get_prepulse_creator(prep_descriptions).pulse.tolist(),
        'endpoint_decoder': builder.get_prepulse_creator(endpoint_decoder_descriptions).pulse.tolist(),
    }

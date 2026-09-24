"""Default settings shared by the qsim migration measurement notebooks.

Cell 6 of `measurement_notebooks/jonginn/qsim_experiments.ipynb`, verbatim.
"""

MEASUREMENT_CONFIG_DEFAULTS = {
    "avoid_yoko": False,
    "use_multiphoton_swap": False,
}

ACTIVE_RESET_DEFAULTS = {
    "reset_dump_mode": 2,
    "dump_reset_iter_num": 1,
}

FLOQUET_DEFAULTS = {
    # For phase accumulation:
    "include_10cycles_buffer": True,
    "include_10cycles_buffer_in_pi_half": True,
    # flat_top: legacy 3-segment pulse; preload_flattop: one preloaded arb envelope
    "floquet_waveform": "preload_flattop",
    "floquet_hardware_loop": True,
    "scramble_sync_cycles": 1,
    "palindrome_scramble": False,
}

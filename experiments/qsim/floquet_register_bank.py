# -*- coding: utf-8 -*-
"""Preloaded Floquet pulse settings in one tProc register page.

Why this is its own module
--------------------------
A Floquet cycle replays the same handful of swap pulses hundreds of times and
only the phase changes between repeats. Writing all six pulse settings before
every pulse costs tProc instructions the cycle cannot spare, so the settings
are written once into a spare register page and each repeat emits one raw
``set`` plus a phase write.

That is a self-contained piece of tProc bookkeeping -- page allocation,
register layout, mode word, timestamp accounting -- and nothing about it
depends on dark modes, encodings or the MBR campaign. It is also where the
sharpest hardware constraints live (one page, six registers per pulse,
full-speed generators only, raw ``set`` must follow a ``sync_all``), which are
much easier to see, and to verify, in 150 lines than buried in a 1,800-line
program base.

Both functions take the program as their first argument rather than being
methods: they are addressed by two program bases (Averager and RAverager) and
by ``mbr_sff``, and none of them owns this machinery.
"""
import numpy as np


def _prepare_preloaded_floquet_register_bank(program,
                                             pulse_args_list,
                                             first_cycle_phases_deg,
                                             phase_steps_deg,
                                             reserved_registers=0):
    """
    Load complete arb Floquet pulse parameters into one unused register page.

    Caution:
        - Raw set uses an explicitly selected page, not ch_page(channel).
        - QICK v1 has eight pages (0..7); this helper searches pages 7..1,
          leaving page 0 for global counters.
        - All pulse settings currently share one page, with six registers per pulse.
          Pulse registers plus reserved_registers must fit in registers 1..31.

    Args:
        - ``pulse_args_list``: one pulse-settings entry per pulse in a Floquet cycle.
        - ``first_cycle_phases_deg``: each pulse's phase on its first occurrence, in degrees.
        - ``phase_steps_deg``: each pulse's phase increment between Floquet cycles, in degrees.
        - ``reserved_registers``: capacity reserved for caller registers, such as a loop counter.
    """
    pulse_count = len(pulse_args_list)
    if len(first_cycle_phases_deg) != pulse_count or len(phase_steps_deg) != pulse_count:
        raise ValueError("Floquet pulse, phase, and phase-step counts differ")

    # TODO: When the next pulse no longer fits, allocate another unused page.
    # Keep each pulse's six registers on one page, preserve the caller's
    # reserved counter location, and track bank reservations to prevent reuse.
    registers_per_pulse = 6
    required_registers = registers_per_pulse * pulse_count + int(reserved_registers)
    if required_registers > 31:
        raise RuntimeError("preload_flattop needs more than one tProc register page")

    register_maps = list(program._gen_regmap.values())
    register_maps += list(program._ro_regmap.values())
    register_maps += list(getattr(program, "_sff_register_map", {}).values())
    occupied_pages = {int(page) for page, register in register_maps if int(register) > 0}
    
    
    # Raw ``set`` operands must share a page. Use a page with no generator,
    # readout, or SFF custom registers, and leave page 0 to global counters.
    register_page = next((page for page in range(7, 0, -1) if page not in occupied_pages),None)
    if register_page is None:
        raise RuntimeError("preload_flattop could not find an unused tProc register page")

    bank = []
    next_register = 1
    pulse_phase_rows = zip(pulse_args_list, first_cycle_phases_deg, phase_steps_deg)
    
    for pulse_args, phase_deg, phase_step_deg in pulse_phase_rows:
        channel = int(pulse_args["ch"])
        generator_manager = program._gen_mgrs[channel]
        if generator_manager.__class__.__name__ != "FullSpeedGenManager":
            raise RuntimeError("preload_flattop register banks require a full-speed "
                               f"generator; channel {channel} uses "
                               f"{generator_manager.__class__.__name__}")

        parameters = dict(generator_manager.defaults)
        parameters.update(pulse_args)
        if parameters.get("style") != "arb":
            raise RuntimeError("preload_flattop register-bank pulses must use arb style")

        waveform_name = parameters["waveform"]
        waveform = generator_manager.envelopes[waveform_name]
        samples_per_clock = int(program.soccfg["gens"][channel]["samps_per_clk"])
        waveform_samples = int(waveform["data"].shape[0])
        if waveform_samples % samples_per_clock:
            raise RuntimeError(f"waveform {waveform_name!r} is not aligned to generator fabric clocks")
        
        
        waveform_length = waveform_samples // samples_per_clock
        waveform_address = int(waveform["addr"]) // samples_per_clock
        mode_word = generator_manager.get_mode_code(
            phrst=parameters.get("phrst"),
            stdysel=parameters.get("stdysel"),
            mode=parameters.get("mode"),
            outsel=parameters.get("outsel"),
            length=waveform_length,
        )

        frequency_register = next_register
        phase_register = next_register + 1
        address_register = next_register + 2
        gain_register = next_register + 3
        mode_register = next_register + 4
        phase_step_register = next_register + 5
        next_register += registers_per_pulse

        first_phase_register_value = int(program.deg2reg(phase_deg, gen_ch=channel))
        phase_step_register_value = int(program.deg2reg(phase_step_deg, gen_ch=channel))
        
        
        register_values = (
            (frequency_register, int(parameters["freq"])),
            (phase_register, first_phase_register_value),
            (address_register, waveform_address),
            (gain_register, int(parameters["gain"])),
            (mode_register, int(mode_word)),
            (phase_step_register, phase_step_register_value),
        )
        
        
        for register, value in register_values:
            program.safe_regwi(register_page, register, value)

        pulse_duration_tproc_cycles = waveform_length * (float(program.tproccfg["f_time"]) / float(program.soccfg["gens"][channel]["f_fabric"]))
        
        bank.append({
            "channel": channel,
            "tproc_channel": int(generator_manager.tproc_ch),
            "register_page": register_page,
            "frequency_register": frequency_register,
            "phase_register": phase_register,
            "address_register": address_register,
            "gain_register": gain_register,
            "mode_register": mode_register,
            "phase_step_register": phase_step_register,
            "first_phase_register_value": first_phase_register_value,
            "pulse_duration_tproc_cycles": pulse_duration_tproc_cycles,
            "waveform_name": waveform_name,
        })

    return bank, register_page, next_register


def _play_preloaded_floquet_register_bank_entry(program, entry):
    """
    Emit one raw arb ``set`` while preserving QICK pulse timestamps.
    """
    channel = entry["channel"]
    current_timestamp = float(program.get_timestamp(gen_ch=channel))
    if not np.isclose(current_timestamp, 0.0):
        raise RuntimeError(
            "preload_flattop raw pulse must start after sync_all()"
        )

    program.set(entry["tproc_channel"],
                entry["register_page"],
                entry["frequency_register"],
                entry["phase_register"],
                entry["address_register"],
                entry["gain_register"],
                entry["mode_register"],
                0,
                f"preloaded arb {entry['waveform_name']} on channel {channel}")
    program.set_timestamp(entry["pulse_duration_tproc_cycles"],
                          gen_ch=channel)

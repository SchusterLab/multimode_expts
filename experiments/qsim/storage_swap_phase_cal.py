# -*- coding: utf-8 -*-
"""Phase matrix of the ds_storage swap pulses.

Split out of ``floquet_dark_mode_readout.py`` unchanged.
"""
from slab import AttrDict

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.floquet_dark_mode_readout import DarkBaseProgram

class StorageSwapPhaseAccumulationProgram(DarkBaseProgram):
    """Measure the phase matrix of the ds_storage swap pulses.

    ``stor_A`` is the affected Ramsey mode and ``stor_B`` is the pulsed mode.
    The first and last A pulses use exactly half of the calibrated full-swap
    plateau time.  Every B full/inverse-full pair closes its population action
    while retaining the phase accumulated by A.  ``advance_phase`` is the
    compensation in degrees per physical B full-swap pulse.
    """

    def _play_storage_pulse(self, stor, length_us, phase_deg):
        storage_ds = self.cfg.device.storage._ds_storage
        stor_name = f"M1-S{stor}"
        freq = storage_ds.get_freq(stor_name)

        if freq < 1800:
            ch = self.flux_low_ch[0]
            waveform = "pi_m1si_low"
        else:
            ch = self.flux_high_ch[0]
            waveform = "pi_m1si_high"

        self.setup_and_pulse(
            ch=ch,
            style="flat_top",
            freq=self.freq2reg(freq, gen_ch=ch),
            phase=self.deg2reg(self._mod360(phase_deg), gen_ch=ch),
            gain=storage_ds.get_gain(stor_name),
            length=self.us2cycles(length_us, gen_ch=ch),
            waveform=waveform,
        )
        self.sync_all(self.us2cycles(0.01))

    def core_pulses(self):
        stor_A = int(self.cfg.expt.stor_A)
        stor_B = int(self.cfg.expt.stor_B)
        n_pulse_B = int(self.cfg.expt.n_pulse)
        advance_phase = float(self.cfg.expt.advance_phase)

        storage_ds = self.cfg.device.storage._ds_storage
        pi_A = float(storage_ds.get_pi(f"M1-S{stor_A}"))
        pi_B = float(storage_ds.get_pi(f"M1-S{stor_B}"))

        # Ramsey preparation on A.  This intentionally uses pi_A / 2 rather
        # than the separately calibrated ds_storage h_pi entry.
        self._play_storage_pulse(
            stor=stor_A,
            length_us=pi_A / 2.0,
            phase_deg=0.0,
        )

        # A 0/180 pair is a full swap followed by its physical inverse.
        # Both pulses produce the same off-target Stark phase on A.
        for _ in range(n_pulse_B):
            self._play_storage_pulse(
                stor=stor_B,
                length_us=pi_B,
                phase_deg=0.0,
            )
            self._play_storage_pulse(
                stor=stor_B,
                length_us=pi_B,
                phase_deg=180.0,
            )

        self._play_storage_pulse(
            stor=stor_A,
            length_us=pi_A / 2.0,
            phase_deg=180.0 + 2.0 * n_pulse_B * advance_phase,
        )
        self.sync_all()

    def body(self):
        cfg = AttrDict(self.cfg)

        self.reset_and_sync()
        if cfg.expt.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = cfg.expt.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        # Load one photon into M1, close the storage Ramsey sequence, and
        # unload the photon.  The measured state is therefore g at n_pulse=0.
        prepulse = self.get_prepulse_creator([
            ["qubit", "ge", "pi", 0.0],
            ["qubit", "ef", "pi", 0.0],
            ["man", "M1", "pi", 0.0],
        ])
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="storage_phase_pre_")
        self.sync_all()

        self.core_pulses()

        postpulse = self.get_prepulse_creator([
            ["man", "M1", "pi", 180.0],
            ["qubit", "ef", "pi", 180.0],
            ["qubit", "ge", "pi", 180.0],
        ])
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="storage_phase_post_")
        self.sync_all()
        self.measure_wrapper()

# -*- coding: utf-8 -*-
"""Validate the broadband ge pi pulse on |g,n> and |e,n>.

Split out of ``floquet_dark_mode_readout.py`` unchanged. This program shares
nothing with the dark-mode pulse layer beyond ``QsimBaseProgram``.
"""
from slab import AttrDict

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.qsim_base import QsimBaseProgram

class BroadbandGeValidationProgram(QsimBaseProgram):
    """Measure the configured broadband ge pi on |g,n> and |e,n>.

    ``validation_case`` selects one of six direct-readout experiments:

      0: |g,n> reference, no broadband pulse
      1: |e,n> reference, no broadband pulse
      2: |g,n> followed by pi_ge_broadband
      3: |e,n> followed by pi_ge_broadband
      4: |g,n> followed by repeated B(0) B(180) inverse pairs
      5: |e,n> followed by repeated B(0) B(180) inverse pairs

    Sweeping photon number and these six cases gives a separate IQ axis for
    every n, so photon-number-dependent readout shifts are not mistaken for a
    broadband-pulse error.
    """

    def initialize(self):
        # No storage-swap pulse is played here.  The runner still supplies the
        # storage/Floquet dataset handles required by the generic pulse creator.
        self.MM_base_initialize()
        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        photon_number = int(cfg.expt.validation_photon_number)
        validation_case = int(cfg.expt.validation_case)

        if photon_number not in (0, 1, 2, 3):
            raise ValueError(
                "validation_photon_number must be 0, 1, 2, or 3; "
                f"got {photon_number}"
            )
        if validation_case not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                "validation_case must be 0 (g ref), 1 (e ref), "
                "2 (g->e), 3 (e->g), 4 (g inverse pairs), or "
                "5 (e inverse pairs)"
            )

        self.reset_and_sync()
        if cfg.expt.get('active_reset', False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = cfg.expt.get('pre_relax_delay', 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        pulse_seq = self.prep_man_photon(photon_number)
        if validation_case in (1, 3, 5):
            pulse_seq.append([
                'multiphoton',
                f'g{photon_number}-e{photon_number}',
                'pi',
                0.0,
            ])
        if validation_case in (2, 3):
            pulse_seq.append(['qubit', 'ge_broadband', 'pi', 0.0])
        elif validation_case in (4, 5):
            inverse_pairs = int(
                cfg.expt.get('validation_inverse_pairs', 4))
            if inverse_pairs < 1:
                raise ValueError(
                    "validation_inverse_pairs must be at least 1")
            for _ in range(inverse_pairs):
                pulse_seq.append(
                    ['qubit', 'ge_broadband', 'pi', 0.0])
                pulse_seq.append(
                    ['qubit', 'ge_broadband', 'pi', 180.0])

        if pulse_seq:
            pulse = self.get_prepulse_creator(pulse_seq)
            self.sync_all()
            self.custom_pulse(
                cfg, pulse.pulse, prefix='ge_broadband_validation_')
            self.sync_all()

        self.measure_wrapper()

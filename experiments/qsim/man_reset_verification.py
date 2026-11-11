"""Manipulate-mode active-reset verification after one-photon preparation."""

from slab import AttrDict

from experiments.MM_base import MM_base, MMAveragerProgram
from experiments.qsim.qsim_base import QsimProgram 



class ManActiveResetVerificationProgram(QsimProgram):
    """
    Prepare manipulate photons before optional active reset.

    Apply configured pre/post pulses after reset, then measure the result.

    cfg.expt options:
        prep_man_photons: photons prepared in the manipulate mode before the reset
            (default 1; 0 prepares none, for the readout floor).
        pre_reset_pulse: optional pulse list (``get_prepulse_creator(...).pulse.tolist()``)
            played after the photon preparation and before the reset, e.g. a qubit
            ge pi to test the reset with the qubit in e.
    """

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)
        
    def initialize(self):
        """
        MM_base_init to pull basic info 
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulses
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1  

        self.sync_all(200)
        
    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.cfg.expt.qubits[0]

        # phase reset
        self.reset_and_sync()
        photons = int(cfg.expt.get('prep_man_photons', 1))
        if photons > 0:
            pulse_cfg = self.prep_man_photon(photons)
            pulse = self.get_prepulse_creator(pulse_cfg)
            self.custom_pulse(AttrDict(self.cfg), pulse.pulse, prefix = 'prep_man_1_initializer')
        if cfg.expt.get('pre_reset_pulse'):
            self.custom_pulse(cfg, cfg.expt.pre_reset_pulse, prefix='pre_reset')
        self.sync_all()
        #do the active reset
        if cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        #  prepulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        self.sync_all()  # align channels

        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post')
        self.sync_all(self.us2cycles(0.05))
        self.measure_wrapper()
    


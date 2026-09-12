"""f0/g1 spectroscopy for calibrating flux-drive amplitude from resonance shifts.

The pulse program supports spectroscopy with or without flux modulation.
The notebook sweeps frequency and flux-drive gain; resonance-boundary fitting
and conversion from gain to current remain in the notebook.
"""

from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseProgram


class FluxDriveF0g1SpectroscopyProgram(QsimBaseProgram):
    """
    Probe the f0/g1 transition with optional flux modulation.

    Prepare the qubit in f and apply a spectroscopy pulse at expt.freq.
    expt.gain controls the probe; expt.flux_drive_gain controls the flux drive.
    Modulation can run periodically from initialization or during the core sequence.
    """

    def initialize(self):
        super().initialize()
        # flux line modulation
        if self.cfg.expt.get("modulate_flux_at_init", False):
            qTest = self.qubits[0]
            _flux_ch = self.flux_low_ch[qTest] #from the parse_config method of MM base
            self.setup_and_pulse(ch=_flux_ch,
                                style="const",
                                freq=self.freq2reg(self.cfg.expt.flux_freq, gen_ch = _flux_ch), 
                                phase=0,
                                gain=self.cfg.expt.flux_drive_gain,
                                length=self.us2cycles(self.cfg.expt.length),
                                mode = "periodic")
        self.sync_all(10)

    def core_pulses(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]
        self.sync_all()
        _f_load_pulse = self.prep_man_photon(1)[0:2:1]
        _load_f_state = self.get_prepulse_creator(_f_load_pulse).pulse.tolist()
        # _man_unload_pulse = _man_load_pulse[-1:-3:-1] #reverse until ge
        # _unload_manipulate = self.get_prepulse_creator(_man_unload_pulse).pulse.tolist()
        self.custom_pulse(cfg, _load_f_state, prefix = "Load_Manipulate")
        self.sync_all()
        if self.cfg.expt.get("modulate_flux_at_core", False):
            if self.cfg.expt.get("debug", False):
                print(f"modulate at core, phase deg = {self.cfg.expt.get("flux_deg", 0)}")
            qTest = self.qubits[0]
            _flux_ch = self.flux_low_ch[qTest] #from the parse_config method of MM base
            self.setup_and_pulse(ch=_flux_ch,
                                style="const",
                                freq=self.freq2reg(self.cfg.expt.flux_freq, gen_ch = _flux_ch), 
                                phase=self.deg2reg(self.cfg.expt.get("flux_deg", 0), gen_ch = _flux_ch),
                                gain=self.cfg.expt.flux_drive_gain,
                                length=self.us2cycles(self.cfg.expt.length))
        # f0g1 spectroscopy
        self.setup_and_pulse(
            ch=self.f0g1_ch[qTest],
            style="const",
            freq= self.freq2reg(self.cfg.expt.freq, gen_ch = self.f0g1_ch[qTest]), #sweep_params should be 'freqs'
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.f0g1_ch[qTest]))
        
        self.sync_all()  # align channels

        # post pulse
        # self.custom_pulse(cfg, _unload_manipulate, prefix = 'Unload_Manipulate')
        self.sync_all(self.us2cycles(0.05))

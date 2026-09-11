from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseProgram


class ExcursionTransitionDebuggingProgram(QsimBaseProgram):
    def initialize(self):
        super().initialize()

        qTest = self.qubits[0]
        ch = self.cfg.hw.soc.dacs.flux_low.ch[qTest]
        _default_nqz = self.cfg.hw.soc.dacs.flux_low.nyquist[qTest]

        self.declare_gen(
            ch=ch,
            nqz=self.cfg.expt.get("kerr_nqz", _default_nqz)
        )
    
    def core_pulses(self):
        cfg=AttrDict(self.cfg)
        ecfg = self.cfg.expt
        qTest = self.qubits[0]
        flat_top_sigma = ecfg.get("flat_top_sigma", 0.005)
        if ecfg.get("debug", True):
            print(f"kerr pulse length = {ecfg.kerr_length}")
        kerr_pulse = [
            [ecfg.kerr_freq],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.cfg.hw.soc.dacs.flux_low.ch[0]],
            ['flat_top'],
            [flat_top_sigma],
        ]
        _ch_kerr = kerr_pulse[4][0]
        
        pulse_cfg = self.prep_man_photon(1)
        prep_man_pulse = self.get_prepulse_creator(pulse_cfg)
        
        if not ecfg.get("skip_all", False) and not ecfg.get("only_flux_excursion", False):
            if not ecfg.get("prepulse", False) and not ecfg.get("skip_prep_man", False):
                if ecfg.get("debug", False):
                    print("performing prep manipulate photon pulse")
                self.sync_all()
                self.custom_pulse(AttrDict(self.cfg), prep_man_pulse.pulse.tolist(), prefix = 'prep_man_1')
            self.sync_all()
            
            if not ecfg.get("skip_kerr_pulse", False):
                if ecfg.get("debug", False):
                    print("performing kerr pulse")
                if ecfg.get("use_flat_top", True):
                    self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
                else:
                    if ecfg.get("debug", False):
                        print("pulsing const")
                    self.setup_and_pulse(ch = _ch_kerr,
                                    style = "const",
                                    freq = self.freq2reg(kerr_pulse[0][0], gen_ch = _ch_kerr),
                                    phase = self.deg2reg(kerr_pulse[3][0], gen_ch = _ch_kerr),
                                    gain = kerr_pulse[1][0],
                                    length = self.us2cycles(kerr_pulse[2][0]),
                                    phrst = 1
                                    )
                self.sync_all()
            if not ecfg.get("perform_wigner", False) and not ecfg.get("skip_man_conv", False):
                if ecfg.get("debug", False):
                    print("performing man pi and qubit ef")
                pulse_cfg2 = [
                    ['man', 'M1', 'pi', 0],
                    ['qubit', 'ef', 'pi', 0]
                ]
                if ecfg.get("skip_ef", False):
                    pulse_cfg2 = [pulse_cfg2[0]]
                pulse_f0g1 = self.get_prepulse_creator(pulse_cfg2)
                self.custom_pulse(AttrDict(self.cfg), pulse_f0g1.pulse.tolist(), prefix = 'man_to_f0g1')
                self.sync_all()
            if ecfg.get("skip_man_conv", False) and not ecfg.get("skip_gef", False):
                if ecfg.get("debug", False):
                    print("performing qubit gef while skipping f0g1")
                pulse_cfg2 = [
                    ['qubit', 'ef', 'pi', 0],
                    ['qubit', 'ge', 'pi', 0]
                ]
                pulse_f0g1 = self.get_prepulse_creator(pulse_cfg2)
                self.custom_pulse(AttrDict(self.cfg), pulse_f0g1.pulse.tolist(), prefix = 'debugging_pulse')
                self.sync_all()
        elif ecfg.get("only_flux_excursion", False):
            self.sync_all()
            if ecfg.get("debug", False):
                print("performing ONLY kerr pulse for the debugging purpose")
            if ecfg.get("excite_qubit", False):
                if ecfg.get("debug", False):
                    print("exciting qubit to e")
                pulse_cfg_qubite = [['qubit', 'ge', 'pi', 0]]
                pulse_qubit_ex = self.get_prepulse_creator(pulse_cfg_qubite)
                self.custom_pulse(AttrDict(self.cfg), pulse_qubit_ex.pulse.tolist(), prefix = 'qubit_exciting_pulse_ge')
                self.sync_all()
            if ecfg.get("use_flat_top", True):
                self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
            else:
                if ecfg.get("debug", False):
                    print("pulsing const")
                self.setup_and_pulse(ch = _ch_kerr,
                                style = "const",
                                freq = self.freq2reg(kerr_pulse[0][0], gen_ch = _ch_kerr),
                                phase = self.deg2reg(kerr_pulse[3][0], gen_ch = _ch_kerr),
                                gain = kerr_pulse[1][0],
                                length = self.us2cycles(kerr_pulse[2][0]),
                                phrst = 1
                                )
            self.sync_all()
            
        
        else:
            if ecfg.get("debug", False):
                print("skipping all pulses, only measuring")
            self.sync_all()

        
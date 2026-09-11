from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseProgram


class Manf0g1RamseyProgram(QsimBaseProgram):

    def core_pulses(self):
        cfg = AttrDict(self.cfg)
        ecfg = self.cfg.expt
        
        qTest = self.qubits[0]
        
        self.sync_all()
        
        #----configuring pulse sequence
        prep_and_man_hpi_seq = [
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ef', 'pi', 0],
            ['man', 'M1', 'hpi', 0]
        ]
        man_hpi_seq = [
            ['man', 'M1', 'hpi', 0],
            ['qubit', 'ef', 'pi', 0],
        ]
        if ecfg.get("do_virtual_ramsey", False):
            virtual_ramsey_freq = ecfg.virtual_ramsey_freq
            if not ecfg.get("use_clock", False):
                kerr_length = ecfg.kerr_length
            else:
                kerr_length = self.cycles2us(ecfg.kerr_length) # convert us to clock cycle
                if ecfg.get("debug", False):
                    print(f"using clock for kerr length. the time is {kerr_length}")
            virtual_ramsey_phase = (virtual_ramsey_freq * kerr_length * 360) % 360
            man_hpi_seq[0][3] = virtual_ramsey_phase
            if ecfg.get("debug", False):
                print(f"Applying virtual ramsey phase shift of {virtual_ramsey_phase} degrees to compensate for frequency detuning during kerr pulse")
                print(man_hpi_seq)
        kerr_pulse = [
            [ecfg.kerr_freq],
            [ecfg.kerr_gain],
            [ecfg.kerr_length],
            [0],
            [self.cfg.hw.soc.dacs.flux_low.ch[0]],
            ['flat_top'],
            [0.005],
        ]
        
        prep_and_man_hpi_pulse = self.get_prepulse_creator(prep_and_man_hpi_seq)
        self.custom_pulse(cfg, prep_and_man_hpi_pulse.pulse.tolist(), prefix = "Prep_and_Manipulate_half_pi")
        self.sync_all()
        
        _ch_kerr = kerr_pulse[4][0]
        self.sync_all()
        #--------- Ramsey wait is now controlled by kerr excursion length.
        if ecfg.get("use_flat_top", True):
            self.custom_pulse(cfg, kerr_pulse, prefix='KerrEng_')
        else:
            if not ecfg.get("use_clock", False):
                _length_of_kerr = self.us2cycles(kerr_pulse[2][0])
            else:
                _length_of_kerr = max(kerr_pulse[2][0], 3) # minimal clock cycles required for RFSoC
                if ecfg.get("debug", False):
                    print("using clock for kerr length. the time is ", self.cycles2us(_length_of_kerr))
            self.setup_and_pulse(ch = _ch_kerr,
                                style = "const",
                                freq = self.freq2reg(kerr_pulse[0][0], gen_ch = _ch_kerr),
                                phase = self.deg2reg(kerr_pulse[3][0], gen_ch = _ch_kerr),
                                gain = kerr_pulse[1][0],
                                length = _length_of_kerr,
                                phrst = 1
                                )
        self.sync_all()
        if not self.cfg.expt.get("postpulse", True):
            if self.cfg.expt.get("debug", False):
                print("playing man_hpi_seq")
            man_hpi_pulse = self.get_prepulse_creator(man_hpi_seq).pulse.tolist()
            self.custom_pulse(cfg, man_hpi_pulse, prefix = "Manipulate_half_pi")
            self.sync_all()




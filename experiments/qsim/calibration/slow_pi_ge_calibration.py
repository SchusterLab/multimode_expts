"""Length Rabi and Ramsey sequences for calibrating the slow qubit ge pulse."""

from slab import AttrDict

from experiments.qsim.qsim_base import QsimBaseProgram


class SlowPiGeLengthRabiProgram(QsimBaseProgram):
    """Apply the slow qubit ge pulse with length_to_sweep as its duration."""

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def core_pulses(self):
        cfg=AttrDict(self.cfg)

        qTest = self.qubits[0]

        pulse = cfg.device.qubit.pulses.slow_pi_ge
        gain = pulse.gain[qTest]
        length = self.cfg.length_to_sweep
        style = pulse.type[qTest]
        sigma = pulse.sigma[qTest]
        print(f"slow ge pulse params: gain {gain}, length {length} us, sigma {sigma} us")

        pulse_data = [
            [cfg.device.qubit.f_ge[qTest]],     # frequency
            [gain],                             # gain
            [length],                           # length (us)
            [0],                                # phase
            [self.qubit_chs[qTest]],            # drive channel
            [style],                            # shape
            [sigma],                            # ramp sigma
        ]
        self.custom_pulse(cfg, pulse_data, prefix='slow_ge_rabi')
        self.sync_all()


class SlowPiGeRamseyProgram(QsimBaseProgram):
    """
    Apply two slow ge half-pi pulses separated by expt.wait_time.

    Use expt.hpi_length for each pulse and advance the second pulse phase
    according to expt.ramsey_freq and the wait time.
    """

    def initialize(self):
        super().initialize()

        qTest = self.qubits[0]
        pulse = self.cfg.device.qubit.pulses.slow_pi_ge
        if pulse.type[qTest] != 'flat_top':
            raise ValueError(
                f"slow_pi_ge must be flat_top, got {pulse.type[qTest]!r}"
            )

        self.slow_pi_ge_gain = pulse.gain[qTest]
        self.slow_pi_ge_length = self.us2cycles(
            self.cfg.expt.hpi_length,
            gen_ch=self.qubit_chs[qTest],
        )
        self.slow_pi_ge_sigma = self.us2cycles(
            pulse.sigma[qTest],
            gen_ch=self.qubit_chs[qTest],
        )
        self.add_gauss(
            ch=self.qubit_chs[qTest],
            name='slow_pi_ge_ramsey',
            sigma=self.slow_pi_ge_sigma,
            length=4 * self.slow_pi_ge_sigma,
        )

    def core_pulses(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]
        qubit_ch = self.qubit_chs[qTest]
        second_phase = (
            360.0 * cfg.expt.ramsey_freq * cfg.expt.wait_time
        ) % 360.0

        self.setup_and_pulse(
            ch=qubit_ch,
            style='flat_top',
            freq=self.f_ge_reg[0],
            phase=0,
            gain=self.slow_pi_ge_gain,
            length=self.slow_pi_ge_length,
            waveform='slow_pi_ge_ramsey',
        )
        self.sync_all(self.us2cycles(cfg.expt.wait_time))
        self.setup_and_pulse(
            ch=qubit_ch,
            style='flat_top',
            freq=self.f_ge_reg[0],
            phase=self.deg2reg(second_phase, gen_ch=qubit_ch),
            gain=self.slow_pi_ge_gain,
            length=self.slow_pi_ge_length,
            waveform='slow_pi_ge_ramsey',
        )
        self.sync_all()

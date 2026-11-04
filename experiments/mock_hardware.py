"""Mock hardware stubs for running experiments without an FPGA.

The point: let the real qick library run unchanged through program
construction, ASM compile, and the acquire() control loop, intercepting only
at the leaf soc.xxx() / instrument-driver calls. This catches the python-side
qick validation errors (bad params, malformed pulses, channel mis-declared,
sweep-step quantization, etc.) without any FPGA contact or job-server
overhead.

A real `qick.QickConfig` is still used in mock mode -- only the `QickSoc`
proxy and the Yokogawa drivers are stubbed. See docs/mock_mode_architecture_plan.md.

With a `soccfg`, `MockQickSoc` also does the resource checks that the board
drivers do at upload (envelope memory, mux tones, readout buffers, tProc
program memory), so a program that does not fit fails in the mock too.
"""

import sys

import numpy as np

from experiments.board_resources import BoardResourceError, envelope_report


class MockQickSoc:
    """Stub QickSoc that satisfies the surface qick's AcquireMixin calls.

    Three qick acquisition entry points (acquire, acquire_decimated, run_rounds)
    call slightly different soc methods; this class covers all three. All
    methods either no-op or return correctly-shaped zeros. Methods accept
    *args/**kwargs so future qick library additions don't break us silently.

    With ``soccfg``, the calls where the board drivers check a resource limit
    raise ``BoardResourceError`` on the same condition. Without it, they do not
    check.
    """

    def __init__(self, soccfg=None):
        self.soccfg = soccfg
        # Recorded by start_readout so poll_data can return the right shape.
        self._total_count = 0
        self._reads_per_shot = []

    def _readout(self, ch):
        return self.soccfg["readouts"][ch]

    # --- config_all / config_gens / config_readouts / load_pulses ---

    def start_src(self, *args, **kwargs):
        pass

    def stop_tproc(self, *args, **kwargs):
        pass

    def load_pulse_data(self, ch, data, addr):
        # Same check as qick/drivers/generator.py AbsSignalGen.load.
        if self.soccfg is None:
            return
        gen = self.soccfg["gens"][ch]
        if len(data) + addr <= gen["maxlen"]:
            return
        message = (
            f"gen ch {ch} ({gen['type']}): envelope of {len(data)} samples at address {addr} "
            f"ends past maxlen {gen['maxlen']}. On the board: 'buffer length must be "
            f"{gen['maxlen']} samples or less'."
        )
        # The caller is QickProgram.load_pulses; its `self` is the program,
        # which has the envelope names.
        prog = sys._getframe(1).f_locals.get("self")
        envelopes = getattr(prog, "envelopes", None)
        if envelopes is not None and ch < len(envelopes):
            message += (
                f"\n{type(prog).__name__} has {envelopes[ch]['next_addr']} samples of envelopes "
                f"on ch {ch}:\n" + envelope_report(prog, ch)
            )
        raise BoardResourceError(message)

    def set_nyquist(self, *args, **kwargs):
        pass

    def set_mixer_freq(self, *args, **kwargs):
        pass

    def config_mux_gen(self, ch, tones):
        # Same check as qick/drivers/generator.py AxisSgMux4V2.set_tones_int.
        if self.soccfg is None:
            return
        n_tones = self.soccfg["gens"][ch].get("n_tones")
        if n_tones is not None and len(tones) > n_tones:
            raise BoardResourceError(
                f"gen ch {ch}: {len(tones)} mux tones, the generator has {n_tones}."
            )

    def configure_readout(self, *args, **kwargs):
        pass

    def config_mux_readout(self, *args, **kwargs):
        pass

    def load_bin_program(self, binprog, load_mem=True):
        # The board fails when the program does not fit the program memory.
        if self.soccfg is None:
            return
        tproc = self.soccfg["tprocs"][0]
        pmem = binprog["pmem"] if isinstance(binprog, dict) else binprog
        if len(pmem) > tproc["pmem_size"]:
            raise BoardResourceError(
                f"tProc: program is {len(pmem)} words, pmem_size is {tproc['pmem_size']}."
            )

    def config_avg(self, *args, **kwargs):
        pass

    def config_buf(self, ch, address=0, length=1, enable=True):
        # Same check as qick/drivers/readout.py AxisAvgBuffer.config_buf.
        if self.soccfg is not None and length >= self._readout(ch)["buf_maxlen"]:
            raise BoardResourceError(
                f"readout ch {ch}: decimated length {length} >= buf_maxlen "
                f"{self._readout(ch)['buf_maxlen']}."
            )

    def reload_mem(self, *args, **kwargs):
        pass

    # --- averaged-acquire path ---

    def start_readout(self, total_count, counter_addr=None, ch_list=None,
                      reads_per_shot=None):
        self._total_count = int(total_count)
        self._reads_per_shot = list(reads_per_shot or [1])

    def poll_data(self):
        # qick acquire() loop expects: [(new_points, (data_per_ch, stats))]
        # where data_per_ch[i].shape == (new_points * reads_per_shot[i], 2)
        # and dtype int64. Returning everything in one chunk so count==total_count
        # on the first poll and the loop exits.
        n = self._total_count
        data_per_ch = [
            np.zeros((n * nr, 2), dtype=np.int64)
            for nr in self._reads_per_shot
        ]
        return [(n, (data_per_ch, {}))]

    # --- decimated-acquire and run_rounds paths ---

    def start_tproc(self, *args, **kwargs):
        pass

    def set_tproc_counter(self, *args, **kwargs):
        pass

    def get_tproc_counter(self, addr=None):
        # Polling loops in acquire_decimated / run_rounds exit when this
        # returns >= total_count.
        return self._total_count

    def _check_transfer(self, ch, length, key):
        # Same check as qick/drivers/readout.py transfer_avg / transfer_buf,
        # with the transfer length that QickSoc.get_accumulated / get_decimated pass.
        if self.soccfg is None:
            return
        transfer = int(length) + int(length) % 2 + 2
        maxlen = self._readout(ch)[key]
        if transfer >= maxlen:
            raise BoardResourceError(
                f"readout ch {ch}: transfer of {length} samples needs {transfer} >= {key} {maxlen}."
            )

    def get_decimated(self, ch=None, address=0, length=0):
        self._check_transfer(ch, length, "buf_maxlen")
        return np.zeros((int(length), 2))

    def get_accumulated(self, ch=None, address=0, length=0):
        self._check_transfer(ch, length, "avg_maxlen")
        # Caller does .reshape((*loop_dims, trigs, 2)), needs length*2 elements.
        return np.zeros((int(length), 2))

    def __repr__(self):
        checks = "resource checks on" if self.soccfg is not None else "no resource checks"
        return f"<MockQickSoc: stub, zero data, {checks}>"


class MockInstrumentManager(dict):
    """Dict-of-instruments stand-in matching the real InstrumentManager surface.

    Holds a MockQickSoc at the configured qick alias. Mid-session swap code in
    MultimodeStation populates this with the right alias key.
    """

    def __init__(self, qick_alias="Qick101", soccfg=None):
        super().__init__()
        self[qick_alias] = MockQickSoc(soccfg=soccfg)

    def __repr__(self):
        return f"<MockInstrumentManager: {list(self.keys())}>"


class MockYokogawa:
    """Stub Yokogawa GS200 voltage source. All ops are no-ops."""

    def __init__(self, name, address):
        self.name = name
        self.address = address
        self._voltage = 0.0
        self._current = 0.0
        self._output_enabled = False

    def set_voltage(self, v):
        print(f"[MOCK] {self.name}.set_voltage({v})")
        self._voltage = float(v)

    def get_voltage(self):
        return self._voltage

    def output_on(self):
        print(f"[MOCK] {self.name}.output_on()")
        self._output_enabled = True

    def output_off(self):
        print(f"[MOCK] {self.name}.output_off()")
        self._output_enabled = False

    def ramp_current(self, current, sweeprate=0.001, channel=1):
        print(f"[MOCK] {self.name}.ramp_current({current}, sweeprate={sweeprate}, channel={channel})")
        self._current = float(current)

    def __repr__(self):
        return f"<MockYokogawa {self.name} @ {self.address}>"

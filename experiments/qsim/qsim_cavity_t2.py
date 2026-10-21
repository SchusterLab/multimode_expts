import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.qsim_base import QsimProgram, QsimExperiment
from scipy.optimize import curve_fit
from fitting.fitting import decaysin1
from fitting.fit_utils import guess_decaysin_params

class QsimCavityT2Program(QsimProgram):
    def core_pulses(self):
        expt = self.cfg.expt

        wait_cycles = self.us2cycles(expt.wait)
        wait_us = self.cycles2us(wait_cycles)
        self.sync_all(wait_cycles)

        pulses = []
        if expt.ro_stor > 0:
            pulses.append(
                ["storage", f"M1-S{expt.ro_stor}", "pi", 0]
            )

        phase = (360 * expt.ramsey_freq * wait_us) % 360
        pulses += [
            ["multiphoton", "f0-g1", "pi", 0],
            ["multiphoton", "e0-f0", "pi", 0],
            ["multiphoton", "g0-e0", "hpi", phase],
        ]

        creator = self.get_prepulse_creator(pulses)
        self.custom_pulse(
            self.cfg, creator.pulse, prefix="t2_readout_"
        )
        self.sync_all()


class QsimCavityT2Experiment(QsimExperiment):
    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        time = np.asarray(data["xpts"])
        max_freq = 0.5 / (time[1] - time[0])

        for channel in ["avgi", "avgq"]:
            guess = guess_decaysin_params(time, data[channel])
            fit, covariance = curve_fit(
                decaysin1,
                time,
                data[channel],
                p0=guess,
                bounds=(
                    [-np.inf, 0, -np.inf, 0, -np.inf],
                    [np.inf, max_freq, np.inf, np.inf, np.inf],
                ),
                maxfev=20000,
            )
            data[f"fit_{channel}"] = fit
            data[f"fit_err_{channel}"] = covariance

        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        _default_size = kwargs.get("figsize", (8, 6))
        time = np.asarray(data["xpts"])
        _mode = self.cfg.expt.init_stor
        _name = "Manipulate" if _mode == 0 else f"Storage {_mode}"

        fig, axes = plt.subplots(2, 1, figsize= _default_size, sharex=True)
        
        for ax, channel in zip(axes, ["avgi", "avgq"]):
            ax.plot(time, data[channel], ".", label="Measured")
            if fit:
                params = data[f"fit_{channel}"]
                ax.plot(
                    time,
                    decaysin1(time, *params),
                    label=f"T2* = {params[3]:.1f} us",
                )
            ax.set_ylabel(channel)
            ax.legend()

        axes[0].set_title(f"{_name} Ramsey")
        axes[1].set_xlabel("Wait time (us)")
        fig.tight_layout()
        plt.show()
        return fig, axes
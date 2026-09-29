# -*- coding: utf-8 -*-
"""The base a dark-mode or MBR measurement is built on: one sweep, one program.

Two classes and one analysis helper, in the repo's usual pairing of an
Experiment that owns acquisition with the Program that owns the pulses.

``DarkBaseExperiment``
    The sweep driver. It works out how many readouts a shot contains --
    active reset, parity check and multiparity readout each add their own --
    then runs the 1D or 2D sweep declared in ``cfg.expt.swept_params``,
    rebuilding the Program per point and keeping the per-shot I/Q.

    The plural-to-singular expansion lives here and nowhere else: the
    Experiment sets ``cfg.expt[param]`` from ``cfg.expt[param + 's']`` before
    each build, which is why a Program constructed directly from a stage
    config raises on the singular key its body reads.

``DarkBaseProgram`` / ``DarkBaseRProgram``
    ``DarkBaseProgram`` is the old name of ``DarkModeProgram``
    (``dark_mode_encoding``), kept until step 10F; no live Program uses it.
    The template (reset, prepulse, the ``core_pulses`` hook each measurement
    overrides, postpulse, readout) is ``QsimBaseProgram``'s. The R variant is the RAverager counterpart for hardware depth
    sweeps; it shares ``body`` and takes its pulse methods by explicit
    assignment, and the asymmetry in *which* it takes is deliberate: it keeps
    MM_base's ``man_reset`` (see ``DarkBaseRProgram``).

Provenance note
---------------
``DarkBaseExperiment`` is a recorded name: saved HDF5 files are called
``JOB-<id>_DarkBaseExperiment.h5`` and the acquisition notebooks submit with
``ExptClass=...DarkBaseExperiment``. The class name is therefore fixed, and
this move deliberately keeps it -- only the module changed, and
``floquet_dark_mode_readout`` still imports it so its old address resolves.
"""
import numpy as np
from slab import AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import MMAveragerProgram, MMRAveragerProgram
from experiments.qsim.dark_mode_encoding import DarkModeProgram
from experiments.qsim.qsim_base import (
    QsimBaseExperiment,
    QsimBaseProgram,
    readout_lane_count,
)
from experiments.qsim.utils import ensure_list_in_cfg
from fitting.fit_display_classes import GeneralFitting


def classify_two_parity_readouts(expt, point_idx=0, threshold=None, e_is_high_I=True):
    rn = expt.cfg.read_num
    qTest = expt.cfg.expt.qubits[0]

    if threshold is None:
        threshold = expt.cfg.device.readout.threshold[qTest]

    idata = np.asarray(expt.data['idata'][point_idx])
    qdata = np.asarray(expt.data['qdata'][point_idx])

    i_first  = idata[rn-2::rn]
    q_first  = qdata[rn-2::rn]
    i_second = idata[rn-1::rn]
    q_second = qdata[rn-1::rn]

    if e_is_high_I:
        first_e = i_first > threshold
        second_e = i_second > threshold
    else:
        first_e = i_first < threshold
        second_e = i_second < threshold

    b0 = first_e.astype(int)
    b1 = second_e.astype(int)

    # cond_sec_phase = -90 convention:
    # (g,g)->0, (e,g)->1, (g,e)->2, (e,e)->3
    n_mod4 = b0 + 2*b1

    out = {
        'i_first': i_first,
        'q_first': q_first,
        'i_second': i_second,
        'q_second': q_second,

        'first_e': first_e,
        'second_e': second_e,

        # parity expectation values:
        # +1 means bit=0, -1 means bit=1.
        # first parity = (-1)^n
        # second parity = +1 for n=0,1 mod 4 and -1 for n=2,3 mod 4.
        'parity_first': 1 - 2*b0,
        'parity_second': 1 - 2*b1,

        'n_mod4': n_mod4,

        'p_first_e': np.mean(first_e),
        'p_second_e': np.mean(second_e),

        'p_gg': np.mean((~first_e) & (~second_e)),
        'p_eg': np.mean(( first_e) & (~second_e)),
        'p_ge': np.mean((~first_e) & ( second_e)),
        'p_ee': np.mean(( first_e) & ( second_e)),
    }

    out['p_mod0'] = np.mean(n_mod4 == 0)
    out['p_mod1'] = np.mean(n_mod4 == 1)
    out['p_mod2'] = np.mean(n_mod4 == 2)
    out['p_mod3'] = np.mean(n_mod4 == 3)

    out['mean_parity_first'] = np.mean(out['parity_first'])
    out['mean_parity_second'] = np.mean(out['parity_second'])
    out['mean_n_mod4'] = np.mean(n_mod4)

    return out


class DarkBaseExperiment(QsimBaseExperiment):
    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = readout_lane_count(self.cfg)
        self.cfg.read_num = read_num
        assert len(self.cfg.expt.swept_params) in {1,2}, "can only handle 1D and 2D sweeps for now"
        sweep_dim = 2 if len(self.cfg.expt.swept_params) == 2 else 1

        if 'perform_wigner' not in self.cfg.expt:
            self.cfg.expt.perform_wigner = False

        outer_param = self.cfg.expt.swept_params[0]
        outer_params = self.cfg.expt[outer_param+'s']
        if sweep_dim == 2:
            inner_param = self.cfg.expt.swept_params[1]
            inner_params = self.cfg.expt[inner_param+'s']
        else:
            inner_param = 'dummy'
            inner_params = [None]  # Dummy value for single parameter sweep
        self.outer_param, self.inner_param = outer_param, inner_param

        data = {
            'avgi': [], 'avgq': [],
            'amps': [], 'phases': [],
            'idata': [], 'qdata': [],
        }
        if sweep_dim == 2:
            data['xpts'] = inner_params
            data['ypts'] = outer_params
        else:
            data['xpts'] = outer_params

        for self.cfg.expt[outer_param] in tqdm(outer_params, disable=not progress):
            for self.cfg.expt[inner_param] in inner_params:
                self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)

                avgi, avgq = self.prog.acquire(self.im[self.cfg.aliases.soc],
                                                threshold=None,
                                                load_pulses=True,
                                                progress=False,
                                                debug=debug,
                                                readouts_per_experiment=read_num)

                idata, qdata = self.prog.collect_shots()
                data['idata'].append(idata)
                data['qdata'].append(qdata)

                if self.cfg.expt.active_reset and self.cfg.expt.get('pre_selection_reset', False):
                    avgi_val, avgq_val = GeneralFitting.filter_shots_per_point(
                        idata, qdata, read_num,
                        threshold=self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]],
                        pre_selection=True)
                else:
                    avgi_val = avgi[0][-1]
                    avgq_val = avgq[0][-1]

                avgi, avgq = avgi_val, avgq_val
                data['avgi'].append(avgi)
                data['avgq'].append(avgq)
                data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
                data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

        for key in 'avgi avgq amps phases'.split():
            data[key] = np.array(data[key])
            if sweep_dim == 2:
                data[key] = np.reshape(data[key], (len(outer_params), len(inner_params)))

        if self.cfg.expt.get('parity_check', False):
            idata_all = np.array(data['idata'])
            qdata_all = np.array(data['qdata'])
            _parity_start_idx= 0  
            if self.cfg.expt.get('active_reset', False):
                params = MMAveragerProgram.get_active_reset_params(self.cfg)
                _parity_start_idx += MMAveragerProgram.active_reset_read_num(**params)
            
            data['parity_idata'] = idata_all[..., _parity_start_idx::read_num]
            data['parity_qdata'] = qdata_all[..., _parity_start_idx::read_num]

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data=data
        return data

    def analyze_multiparity(self):
        keys = [
            'mean_parity_first', 'mean_parity_second',
            'p_first_e', 'p_second_e',
            'p_mod0', 'p_mod1', 'p_mod2', 'p_mod3',
            'p_gg', 'p_eg', 'p_ge', 'p_ee',
            'mean_n_mod4',
        ]

        xpts = np.asarray(self.data['xpts']).reshape(-1)
        ypts = np.asarray(self.data.get('ypts', [])).reshape(-1)
        is_2d = len(ypts) > 0

        if is_2d:
            data_shape = (len(ypts), len(xpts))
            point_count = len(ypts) * len(xpts)
        else:
            data_shape = (len(xpts),)
            point_count = len(xpts)

        if len(self.data['idata']) != point_count:
            raise ValueError(
                'multiparity point count does not match the sweep axes: '
                f"{len(self.data['idata'])} shots rows for shape {data_shape}"
            )

        out = {'xpts': xpts}
        if is_2d:
            out['ypts'] = ypts
        for key in keys:
            out[key] = []

        for j in range(point_count):
            r = classify_two_parity_readouts(self, point_idx=j)

            for key in keys:
                out[key].append(r[key])

        for key in keys:
            out[key] = np.asarray(out[key]).reshape(data_shape)

        # Same quantity as p1 + 2*p2 + 3*p3.
        # Kept as a convenient explicit name.
        out['nmod4_mean'] = out['mean_n_mod4']

        self.data['multiparity'] = out
        return out
        

class DarkBaseProgram(DarkModeProgram):
    """The old name of ``DarkModeProgram``; no live Program uses it.

    Its template (``initialize``, ``body``) and the manipulate-mode pulses
    moved into ``QsimBaseProgram`` in step 10B, and its two mixins became the
    classes ``FloquetProgram`` and ``DarkModeProgram`` in step 10C
    (``docs/qsim/program_tree_plan.md``). It stays until the renames of step
    10F, for the deprecated modules and the old address in
    ``floquet_dark_mode_readout``.
    """


class DarkBaseRProgram(MMRAveragerProgram):
    """RAverager counterpart of ``DarkBaseProgram``.

    The pulse-building methods below do not depend on the AveragerProgram
    software loop, so both program types use the same implementations.
    Concrete RAverager programs only need to define ``core_pulses`` and
    ``update``.
    """

    _pre_selection_filtering = True

    retrieve_swap_parameters = QsimBaseProgram.retrieve_swap_parameters #borrowing methods
    _initialize_floquet_pulses = QsimBaseProgram._initialize_floquet_pulses
    # Two of the three manipulate-mode methods, by assignment rather than
    # inheritance: inheriting from QsimBaseProgram would also shadow MM_base's
    # ``man_reset``, which is the one ``active_reset`` plays here.
    prep_man_fock_state = QsimBaseProgram.prep_man_fock_state
    multi_parity_readout = QsimBaseProgram.multi_parity_readout
    body = QsimBaseProgram.body #borrowing methods

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()

        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        man_mode_no = self.cfg.expt.get("man_mode_no", 1)
        self.man_mode_idx = man_mode_no - 1

        self._initialize_floquet_pulses()

        self.sync_all(200)



"""Opt-in multiparity analysis for individual Experiment classes.

Place MultiparityMixin before the Experiment base in the class declaration.
The owning Experiment decides when its analyze() calls analyze_multiparity().
Results are numeric arrays in self.data, ready for the normal save lifecycle.
"""

import numpy as np
from experiments.qsim.qsim_base import QsimExperiment

class MultiparityMixin:
    """Manual multiparity analysis explicitly added to an Experiment class.

    ``cfg.expt.multiparity_confusion_matrix``, when supplied, is a 4x4
    matrix C[prepared number, measured label]. Each row sums to one.
    This mixin owns shot classification, lane order and the label convention.
    """

    def classify_two_parity_readouts(self, point_idx=0, threshold=None, e_is_high_I=True):
        rn = self.cfg.read_num
        qTest = self.cfg.expt.qubits[0]

        if threshold is None:
            threshold = self.cfg.device.readout.threshold[qTest]

        idata = np.asarray(self.data['idata'][point_idx])
        qdata = np.asarray(self.data['qdata'][point_idx])

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

    def analyze_multiparity(self):
        """Classify each point and optionally correct its mean number modulo 4.

        Raw probabilities and their existing names are preserved. Correction
        solves ``C.T @ p_true = p_raw`` without clipping. Both raw and
        corrected means retain the shape of the 1D or 2D sweep. Numeric
        results are added directly to ``self.data`` and also returned.
        """
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
        data_shape = (len(ypts), len(xpts)) if is_2d else (len(xpts),)
        point_count = int(np.prod(data_shape))

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
            result = self.classify_two_parity_readouts(point_idx=j)
            for key in keys:
                out[key].append(result[key])
        for key in keys:
            out[key] = np.asarray(out[key]).reshape(data_shape)

        out['nmod4_mean'] = out['mean_n_mod4']
        confusion = self.cfg.expt.get('multiparity_confusion_matrix')
        if confusion is not None:
            confusion = np.asarray(confusion, dtype=float)
            if (confusion.shape != (4, 4)
                    or not np.isfinite(confusion).all()
                    or np.any(confusion < 0)
                    or not np.allclose(confusion.sum(axis=1), 1.0,
                                       rtol=0, atol=1e-8)):
                raise ValueError(
                    'multiparity_confusion_matrix must be a finite, '
                    'nonnegative 4x4 matrix C[prepared, measured] '
                    'whose rows sum to one'
                )
            raw_probability = np.stack(
                [out[f'p_mod{n}'] for n in range(4)], axis=-1
            ).reshape(-1, 4)
            try:
                corrected = np.linalg.solve(confusion.T, raw_probability.T).T
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    'multiparity_confusion_matrix is singular; '
                    'cannot correct multiparity probabilities'
                ) from exc
            out['nmod4_mean_corrected'] = (corrected @ np.arange(4)).reshape(data_shape)
            out['confusion_matrix'] = confusion.copy()

        # Reanalysis after removing a calibration must not expose an old
        # corrected result, including fields loaded from an analyzed HDF5.
        self.data.pop('nmod4_mean_corrected', None)
        self.data.pop('confusion_matrix', None)
        self.data.update(out)
        return out


class MultiparityQsimExperiment(MultiparityMixin, QsimExperiment):
    pass
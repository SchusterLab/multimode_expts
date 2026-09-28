"""Fitter B: one matrix pencil for all rows together (spec 4.1).

All rows share the poles, so their Hankel matrices share the right factor
``z_lambda^k`` and can be stacked into one pencil: one SVD, one rank, one
eigenproblem.

Pure numerics.
"""
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fitting.qsim.poles.pole_fit import (check_shape, normalize_to_initial_return, pole_fit,
                                         sample_time)
from fitting.qsim.poles.rank import RankRule, choose_rank


class JointPencilSettings(BaseModel):
    """The free parameters of the joint pencil."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: How the number of poles is chosen (spec 4.6).
    rank_rule: RankRule = RankRule()
    #: Pencil length L as a fraction of the samples N. At most L poles can be
    #: found; a longer pencil resolves closer poles but has fewer rows per trace,
    #: so each singular value averages less noise.
    pencil_fraction: Annotated[float, Field(gt=0., lt=1.)] = 2 / 3


def fit(A, time_us, settings=JointPencilSettings(), row_groups=None):
    """-> the PoleFit of rows ``A`` (row x time) on ``time_us``. ``row_groups`` unused."""
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(check_shape(A, time_us))
    pencil_length = round(settings.pencil_fraction * a.shape[1])
    H0, H1 = stacked_hankel_pair(a, pencil_length)
    U, s, Vh = np.linalg.svd(H0, full_matrices=False)
    rank = choose_rank(s, H0.shape[0], settings.rank_rule)
    poles = shift_invariance_poles(U, s, Vh, H1, rank)
    return pole_fit(poles, a, dt_us, rank)


def stacked_hankel_pair(a, pencil_length):
    """-> (H0, H1) = (H[:, :-1], H[:, 1:]) of H = [H_1; ...; H_R], where

        (H_b)_{ij} = a_b[i + j],   i = 0 .. N-L-1,   j = 0 .. L.

    Each H_b = X_b diag(c_b) Z^T with Z_{j,lambda} = z_lambda^j shared, so H1 = H0
    shifted by one column keeps the invariance H1 = X C diag(z) Z^T.
    """
    rows, samples = a.shape
    i = np.arange(samples - pencil_length)[:, None]
    j = np.arange(pencil_length + 1)[None, :]
    H = a[:, i + j].reshape(rows * (samples - pencil_length), pencil_length + 1)
    return H[:, :-1], H[:, 1:]


def shift_invariance_poles(U, s, Vh, H1, rank):
    """-> z = eig(S_M^-1 U_M^h H1 V_M), with H0 = U S V^h truncated to rank M."""
    U_M, s_M, V_M = U[:, :rank], s[:rank], Vh[:rank].conj().T
    return np.linalg.eigvals((U_M.conj().T @ H1 @ V_M) / s_M[:, None])

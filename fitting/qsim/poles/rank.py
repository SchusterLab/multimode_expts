"""The rank rule shared by the pencil fitters (spec 4.6).

The rank sets the number of poles, so it is the main free parameter of a pencil.
Two candidate rules; benchmark 2 chooses between them.

Pure numerics.
"""
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_Positive = Annotated[float, Field(gt=0., allow_inf_nan=False)]


class RankRule(BaseModel):
    """How many singular values of the pencil are signal."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: "threshold": count s_k > factor x median(s). Cheap; undercounts when the
    #: signal fills more than half the singular values (the median is then signal).
    #: "mdl": minimum description length of the tail. Needs no noise level;
    #: assumes the noise singular values are about equal.
    rule: Literal["threshold", "mdl"] = "mdl"
    #: Threshold rule factor. 2.858 is the Gavish-Donoho value for a square matrix.
    threshold_factor: _Positive = 2.858
    #: Singular values below this fraction of the largest are zero for every rule.
    #: Lower finds poles closer to degenerate on clean data; higher rejects noise.
    numerical_floor: _Positive = 1e-12


def threshold_rank(singular_values, factor):
    """-> #{k : s_k > factor x median(s)}."""
    return int(np.sum(singular_values > factor * np.median(singular_values)))


def mdl_rank(singular_values, sample_count):
    """-> argmin_k MDL(k), with lambda = s^2, p values and n samples:

        MDL(k) = -n (p - k) ln(G_k / M_k) + k (2p - k) ln(n) / 2,

    where G_k and M_k are the geometric and arithmetic means of lambda_{k+1..p}.
    """
    eigenvalues = np.maximum(singular_values ** 2, np.finfo(float).tiny)
    p = len(eigenvalues)
    ks = np.arange(p)
    log_ratio = np.array([np.mean(np.log(eigenvalues[k:])) - np.log(np.mean(eigenvalues[k:]))
                          for k in ks])
    mdl = -sample_count * (p - ks) * log_ratio + ks * (2 * p - ks) * np.log(sample_count) / 2
    return int(np.argmin(mdl))


def choose_rank(singular_values, sample_count, rule):
    """-> the signal rank: the rule's count, capped by the numerical floor.

    ``singular_values`` descending; ``sample_count`` is the number of rows of
    the pencil (the snapshots of the MDL rule).
    """
    above_floor = int(np.sum(singular_values > rule.numerical_floor * singular_values[0]))
    if rule.rule == "threshold":
        rank = threshold_rank(singular_values, rule.threshold_factor)
    else:
        rank = mdl_rank(singular_values, sample_count)
    return max(1, min(rank, above_floor))


def explain_rank_choice(singular_values, sample_count, rule, ax=None):
    """Plot the singular values, the floor and the chosen rank. -> the rank."""
    rank = choose_rank(singular_values, sample_count, rule)
    ax = ax or plt.subplots(figsize=(5, 3))[1]
    ax.semilogy(np.arange(1, len(singular_values) + 1), singular_values / singular_values[0], ".")
    ax.axvline(rank + 0.5, color="k", lw=0.8, label=f"rank {rank} ({rule.rule})")
    ax.axhline(rule.numerical_floor, color="gray", ls=":", label="numerical floor")
    if rule.rule == "threshold":
        ax.axhline(rule.threshold_factor * np.median(singular_values) / singular_values[0],
                   color="gray", ls="--", label="threshold")
    ax.set(xlabel="k", ylabel="s_k / s_1")
    ax.legend(fontsize=8)
    return rank

from dataclasses import dataclass
from fearless_ep import BaseFamily
from expfam.distributions.mvn import *
from plum import dispatch
from typing import Any, Tuple

@dataclass
class MvnFamily(BaseFamily):
    D: int

@dispatch
def natural_from_mean(F: MvnFamily, mean_params: MvnMeanParams):
    return mvn_natural_from_mean(mvn_symmetrize(mean_params))

@dispatch
def mean_from_natural(F: MvnFamily, natural_params: MvnNaturalParams):
    return mvn_mean_from_natural(mvn_symmetrize(natural_params))

@dispatch
def estimate_natural_params(F: MvnFamily, average_stats: MvnMeanParams, n: int):
    x, xx = average_stats
    assert(F.D == x.shape[-1])
    S = n * (xx - outer(x, x))
    Q = (n - F.D - 2) * jnp.linalg.inv(S)
    h = mvp(Q, x)
    J = -.5 * Q
    return h, J

@dispatch
def stats(F: MvnFamily, x: Any):
    return mvn_stats(x)

@dispatch
def log_prob(F: MvnFamily, natural_params: Any, x: Any):
    return mvn_log_prob(natural_params, x)

@dispatch
def kl(F: MvnFamily, natural_params_from: Any, natural_params_to: Any):
    return mvn_kl(natural_params_from, natural_params_to)

@dispatch
def sample(rng: Array, F: MvnFamily, natural_params: Any, shape_prefix: Tuple):
    return mvn_sample(rng, natural_params, shape_prefix)


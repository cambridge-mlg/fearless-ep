from base_family import BaseFamily
from dataclasses import dataclass
from expfam.distributions.diagonal_mvn import *
from plum import dispatch
from typing import Any, Tuple

@dataclass
class DiagonalMvnFamily(BaseFamily):
    D: int

@dispatch
def natural_from_mean(F: DiagonalMvnFamily, mean_params: DiagonalMvnMeanParams):
    return diagonal_mvn_natural_from_mean(mean_params)

@dispatch
def mean_from_natural(F: DiagonalMvnFamily, natural_params: DiagonalMvnNaturalParams):
    return diagonal_mvn_mean_from_natural(natural_params)

@dispatch
def estimate_natural_params(F: DiagonalMvnFamily, average_stats: DiagonalMvnMeanParams, n: int):
    x, xx = average_stats
    d = 1
    s = n*(xx - jnp.square(x))
    q = (n-d-2)/s
    h = q*x
    j = -.5*q
    return h, j

@dispatch
def stats(F: DiagonalMvnFamily, x: Any):
    return diagonal_mvn_stats(x)
@dispatch
def log_prob(F: DiagonalMvnFamily, natural_params: Any, x: Any):
    return diagonal_mvn_log_prob(natural_params, x)

@dispatch
def kl(F: DiagonalMvnFamily, natural_params_from: Any, natural_params_to: Any):
    return diagonal_mvn_kl(natural_params_from, natural_params_to)

@dispatch
def sample(rng: Array, F: DiagonalMvnFamily, natural_params: Any, shape_prefix: Tuple):
    return diagonal_mvn_sample(rng, natural_params, shape_prefix)


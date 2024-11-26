from dataclasses import dataclass
from fearless_ep import BaseFamily
from expfam.distributions.niw import *
from plum import dispatch
from typing import Any

@dataclass
class NiwFamily(BaseFamily):
    D: int

@dispatch
def natural_from_mean(F: NiwFamily, mean_params: NiwMeanParams):
    return niw_natural_from_mean(niw_symmetrize(mean_params))

@dispatch
def mean_from_natural(F: NiwFamily, natural_params: NiwNaturalParams):
    return niw_mean_from_natural(niw_symmetrize(natural_params))

@dispatch
def estimate_natural_params(F: NiwFamily, average_stats: NiwMeanParams, n: int):
    return natural_from_mean(F, average_stats)

@dispatch
def stats(F: NiwFamily, x: Any):
    return niw_stats(x)

@dispatch
def log_prob(F: NiwFamily, natural_params: Any, x: Any):
    return niw_log_prob(natural_params, x)

@dispatch
def kl(F: NiwFamily, natural_params_from: Any, natural_params_to: Any):
    return niw_kl(natural_params_from, natural_params_to)

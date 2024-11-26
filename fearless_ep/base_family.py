from abc import ABC
from jax import Array
from plum import dispatch
from typing import Any, Tuple

class BaseFamily(ABC):
    pass

@dispatch.abstract
def natural_from_mean(F: BaseFamily, mean_params: Any):
    pass

@dispatch.abstract
def mean_from_natural(F: BaseFamily, natural_params: Any):
    pass

@dispatch.abstract
def estimate_natural_params(F: BaseFamily, average_stats: Any, n: int):
    pass

@dispatch.abstract
def stats(F: BaseFamily, x: Any):
    pass

@dispatch.abstract
def log_prob(F: BaseFamily, natural_params: Any, x: Any):
    pass

@dispatch.abstract
def sample(rng: Array, F: BaseFamily, natural_params: Any, shape_prefix: Tuple):
    pass

@dispatch.abstract
def kl(F: BaseFamily, natural_params: Any):
    pass

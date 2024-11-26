import jax
import jax.numpy as jnp
from fearless_ep.base_family import *
from jaxutil.functional import umap
from jaxutil.tree import *
from jax import vmap, vjp
from jax.lax import scan
from jax.random import split
from functools import partial
from enum import Enum, auto
from typing import Callable

class InferenceMethod(Enum):
    EP = auto()
    EP_ETA = auto()
    EP_MU = auto()
    SNEP = auto()

def _update_theta(eta_0, lam):
    return tree_add(eta_0, tree_map(partial(jnp.sum, axis=0), lam))

def _update_lam_ep(eta_0, lam, eta_tilted, lr):
    eta = tree_add(eta_0, tree_map(partial(jnp.sum, axis=0), lam))
    eta_prime = tree_interpolate(lr, eta, eta_tilted)
    lam_prime = tree_sub(eta_prime, tree_sub(eta, lam))
    return lam_prime

def _update_lam_ep_eta(F, eta_0, lam, mu_diff, lr):
    eta = tree_add(eta_0, tree_map(partial(jnp.sum, axis=0), lam))
    mu = mean_from_natural(F, eta)
    _vjpfun = vjp(lambda _: natural_from_mean(F, _), mu)[1]
    ng = vmap(_vjpfun)(mu_diff)[0]
    lam_prime = tree_add(lam, tree_scale(ng, lr))
    return lam_prime

def _update_lam_ep_mu(F, eta_0, lam, mu_diff, lr):
    eta = tree_add(eta_0, tree_map(partial(jnp.sum, axis=0), lam))
    mu = mean_from_natural(F, eta)
    mu_prime = tree_add(mu, tree_scale(mu_diff, lr))
    eta_prime = natural_from_mean(F, mu_prime)
    lam_prime = tree_sub(eta_prime, tree_sub(eta, lam))
    return lam_prime

def _update_lam_snep(F, lam, mu_diff, lr):
    gamma = mean_from_natural(F, lam)
    gamma_prime = tree_add(gamma, tree_scale(mu_diff, lr))
    lam_prime = natural_from_mean(gamma_prime)
    return lam_prime

def inference_init(
    F: BaseFamily,
    method: InferenceMethod,
    eta_0: Any, # prior natural parameters
    n_inner: int, # number of inner updates per outer update
    n_samp: int, # number of samples per inner update
    factors: Any,
    sampler_draw_fn: Callable,
):
    def _inner_step(theta, lam, samplers, lr):
        samples, samplers = umap(sampler_draw_fn, (0, None, 0, 0, None))(
            samplers, theta, lam, factors, n_samp)

        tilted_moments = vmap(vmap(lambda _: stats(F, _)))(samples) # per-sample stats
        tilted_moments = tree_map(partial(jnp.mean, axis=1), tilted_moments) # average stats

        if method == InferenceMethod.EP:
            eta_tilted = estimate_natural_params(F, tilted_moments, n_samp)
            lam = _update_lam_ep(eta_0, lam, eta_tilted, lr)
        else:
            eta = tree_add(eta_0, tree_map(partial(jnp.sum, axis=0), lam))
            mu_diff = tree_sub(tilted_moments, mean_from_natural(F, eta))

            _update_lam = {
                InferenceMethod.EP_ETA: _update_lam_ep_eta,
                InferenceMethod.EP_MU: _update_lam_ep_mu,
                InferenceMethod.SNEP: _update_lam_snep,
            }[method]

            lam = _update_lam(F, eta_0, lam, mu_diff, lr)

        return lam, samplers

    def _step(theta, lam, samplers, lr):
        # inner updates
        lam, samplers = scan(
                lambda c, _: (_inner_step(theta, c[0], c[1], lr), None),
            (lam, samplers), jnp.arange(n_inner))[0]
        # outer update
        theta = _update_theta(eta_0, lam)

        return theta, lam, samplers
       
    return _step
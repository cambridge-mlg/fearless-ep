import jax
import jax.numpy as jnp
from jaxutil.functional import umap
from jaxutil.tree import *
from numpyro.infer.hmc import hmc
from jax import vmap
from jax.lax import scan
from jax.random import split
from typing import NamedTuple

def make_sampling_functions(F, n_chains, n_burnin, tau, init_position_fn, tilted_potential_fn, sample_transform_fn):

    numpyro_init_fn, numpyro_draw_fn = hmc(potential_fn_gen=lambda theta, lam, factor: \
        lambda positions: tilted_potential_fn(F, theta, lam, factor, positions), algo='NUTS')

    def _init(rng, theta, lam, factor):
        rng_positions, rng_sampler = split(rng)
        positions = vmap(
                lambda _: init_position_fn(_, F, theta)
            )(split(rng_positions, n_chains))

        # model_args and num_warmup below are only used to initialise with the right structure
        sampler = vmap(
                lambda rng, position: numpyro_init_fn(
                    init_params=position,
                    model_args=(theta, lam, factor),
                    rng_key=rng,
                    num_warmup=1)
            )(split(rng_sampler, n_chains), positions)
        return sampler
    
    def _draw(sampler, theta, lam, factor, n_samp):
        # we need the number of samples to be divisible by the number of chains
        assert(n_samp % n_chains == 0)

        n_total = n_burnin + tau*(n_samp // n_chains)
    
        sampler, samples = umap(lambda chain_sampler_0:
                scan(lambda chain_sampler_t, _:
                        (lambda _: (_, _.z))(
                            numpyro_draw_fn(chain_sampler_t, model_args=(theta, lam, factor))),
                    chain_sampler_0, jnp.arange(n_total))
            )(sampler)
        
        # discard burn-in, thin samples, and concatenate chains
        samples = tree_map(lambda _: _[:, n_burnin:], samples)
        samples = tree_map(lambda _: _.reshape((n_chains, n_samp // n_chains, tau, *_.shape[2:]))[:,:,-1], samples)
        samples = tree_map(lambda _: _.reshape((n_samp, *_.shape[2:])), samples)

        samples = vmap(sample_transform_fn)(samples)

        return samples, sampler

    def _warmup(sampler, theta, lam, factor, n_warmup):
        # we have to re-initialize to start a new warmup phase
        sampler = vmap(lambda chain_sampler: numpyro_init_fn(
                init_params=chain_sampler.z,
                model_args=(theta, lam, factor),
                num_warmup=n_warmup,
                rng_key=chain_sampler.rng_key,
                step_size=chain_sampler.adapt_state.step_size,
                inverse_mass_matrix=chain_sampler.adapt_state.inverse_mass_matrix)
            )(sampler)

        sampler = umap(lambda chain_sampler_0:
                scan(lambda chain_sampler_t, _:
                        (lambda _: (_, None))(
                            numpyro_draw_fn(chain_sampler_t, model_args=(theta, lam, factor))),
                    chain_sampler_0, jnp.arange(n_warmup))
            )(sampler)[0]
 
        return sampler
    
    return _init, _warmup, _draw

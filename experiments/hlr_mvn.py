# hierarchical logistic regression model with multivariate normal prior
import argparse
import jax.numpy as jnp
import math
import pickle
from fearless_ep.base_families.mvn import *
from fearless_ep.inference import *
from fearless_ep.numpyro import *
from jax.random import PRNGKey, split
from jaxutil.functional import umap
from jaxutil.random import rngcall
from jaxutil.tree import *
from expfam.distributions.diagonal_mvn import *
from expfam.distributions.mvn import *

jax.config.update("jax_enable_x64", True)

def _generate_synthetic_data(rng, n_groups, n_covariates, n_observations):
    rng_mu, rng_logsigma, rng_w, rng_x, rng_y = split(rng, 5)
    mu = jnp.array([1.5, *jax.random.uniform(rng_mu, (n_covariates-1,), minval=-2., maxval=2.)])
    logsigma = jnp.array([0.4, *jax.random.uniform(rng_logsigma, (n_covariates-1,), minval=-.5, maxval=.5)])
    w = mu + jax.random.normal(rng_w, (n_groups,n_covariates))*jnp.exp(logsigma)
    x = jnp.concatenate([
            jnp.ones((n_groups,n_observations,1)),
            jax.random.normal(rng_x, (n_groups,n_observations,n_covariates-1))*.5],
        -1)
    f = jnp.sum(w[:,None]*x, -1)
    y = jax.random.uniform(rng_y, f.shape) < jax.nn.sigmoid(f)
    return x, y

def _tilted_potential(F, theta, lam, factor, position):
    x, y = factor
    n_groups, n_obs, n_covariates = x.shape
    assert(y.shape == (n_groups, n_obs))
    mu, logsigma, w = position
    assert(mu.shape == (n_covariates,))
    assert(logsigma.shape == (n_covariates,))
    assert(w.shape == (n_groups, n_covariates))

    cavity = tree_sub(theta, lam)
    w_prior_natural_params = mu/jnp.exp(2*logsigma)[None], -.5/jnp.exp(2*logsigma)[None]
    f = jnp.sum(w[:,None]*x, -1)

    return -((jnp.sum(jax.nn.log_sigmoid((2*y-1)*f), axis=(0,1)) +
        jnp.sum(diagonal_mvn_log_prob(w_prior_natural_params, w), 0)) +
        log_prob(F, cavity, jnp.concatenate([mu, logsigma])))

def _init_sampler_position(rng, F, theta, n_groups):
    rng_z, rng_w = split(rng)
    z = sample(rng_z, F, theta, ())
    mu, logsigma = jnp.split(z, 2, -1)
    w = mu + jax.random.normal(rng_w, (n_groups, mu.shape[-1]))*jnp.exp(logsigma)
    return mu, logsigma, w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['EP', 'EP_ETA', 'EP_MU', 'SNEP'], default='EP_ETA')
    parser.add_argument('--n-outer', type=int, default=100_000, help='total number of outer updates')
    parser.add_argument('--n-inner', type=int, default=1, help='number of inner updates per outer update')
    parser.add_argument('--n-chains', type=int, default=1, help='number of sampling chains')
    parser.add_argument('--n-burnin', type=int, default=0, help='number of burn-in samples per inner update')
    parser.add_argument('--n-samp', type=int, default=1, help='total number of samples per inner update')
    parser.add_argument('--n-warmup', type=int, default=200, help='number of warm-up samples used for adaptation')
    parser.add_argument('--warmup-interval', type=int, default=400, help='number of outer updates per warmup phase')
    parser.add_argument('--tau', type=int, default=1, help='thinning ratio')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate / step size')
    parser.add_argument('--n-factors', type=int, default=16, help='number of factors')
    parser.add_argument('--n-groups', type=int, default=16, help='number of groups')
    parser.add_argument('--n-covariates', type=int, default=4, help='number of regression covariates')
    parser.add_argument('--n-observations', type=int, default=20, help='number of observations per group')
    parser.add_argument('--data-seed', type=int, default=0, help='data generation random seed')
    parser.add_argument('--init-seed', type=int, default=0, help='sampler initialisation random seed')
    parser.add_argument('--load-reference-path', default=None, help='path to load reference posterior from')
    parser.add_argument('--save-reference-path', default=None, help='path to save reference posterior to')
    parser.add_argument('--load-checkpoint-path', default=None, help='path to load checkpoint from')
    parser.add_argument('--save-checkpoint-path', default=None, help='path to save checkpoint to')

    args = parser.parse_args()

    n_factors, n_groups, n_covariates, n_observations = (
        args.n_factors, args.n_groups, args.n_covariates, args.n_observations)
    n_groups_per_factor = n_groups//n_factors

    # generate data and combine groups into factors
    x, y = _generate_synthetic_data(PRNGKey(args.data_seed), n_groups, n_covariates, n_observations)
    x = x.reshape(n_factors, n_groups_per_factor, n_observations, n_covariates)
    y = y.reshape(n_factors, n_groups_per_factor, n_observations)
    factors = x, y

    rng = PRNGKey(args.init_seed)
    method = InferenceMethod[args.method]
    F = MvnFamily(n_covariates*2)

    # make sampling functions
    sampler_init_fn, sampler_warmup_fn, sampler_draw_fn = make_sampling_functions(
        F,
        args.n_chains,
        args.n_burnin,
        args.tau,
        partial(_init_sampler_position, n_groups=n_groups_per_factor),
        _tilted_potential,
        sample_transform_fn=lambda _: jnp.concatenate([_[0], _[1]], -1))

    # define prior
    eta_0_h = jnp.zeros(2*n_covariates)
    eta_0_J = -.5 * jnp.diag(
        jnp.array(
            [.25**2]*n_covariates +
            [.5**2]*n_covariates))
    eta_0 = eta_0_h, eta_0_J

    # initialize approximation parameters
    lam_0_scale = {
        InferenceMethod.EP: .0,
        InferenceMethod.EP_ETA: .0,
        InferenceMethod.EP_MU: .0,
        InferenceMethod.SNEP: .5/n_factors
    }[method]

    lam = vmap(lambda _: tree_scale(eta_0, lam_0_scale))(np.arange(n_factors))
    theta = tree_add(eta_0, tree_map(partial(jnp.sum, axis=0), lam))

    if args.load_checkpoint_path:
        theta, lam, samplers = pickle.load(open(args.load_checkpoint_path, "rb"))

    # initialize inference routine
    inference_step_fn = inference_init(
        F, method, eta_0, args.n_inner, args.n_samp, factors, sampler_draw_fn)

    # initialize samplers
    rng_samplers, rng = split(rng)
    samplers = vmap(sampler_init_fn, (0, None, 0, 0))(
        split(rng_samplers, n_factors), theta, lam, factors)

    if args.load_reference_path:
        reference = pickle.load(open(args.load_reference_path, "rb"))
    else:
        reference = None

    n_iter = int(math.ceil(args.n_outer / args.warmup_interval))

    for i in range(n_iter + 1):

        print({
            'steps': i*args.warmup_interval,
            'kl_p_q': kl(F, reference, theta).item() if reference else None,
            'kl_q_p': kl(F, theta, reference).item() if reference else None})

        if i < n_iter:
            # warmup samplers
            samplers = umap(sampler_warmup_fn, (0, None, 0, 0, None))(
                samplers, theta, lam, factors, args.n_warmup)

            # perform args.warmup_interval inference steps
            theta, lam, samplers = scan(lambda carry, _:
                    (inference_step_fn(*carry, args.lr), None),
                (theta, lam, samplers), jnp.arange(args.warmup_interval))[0]

            # enforce symmetry constraints
            theta = mvn_symmetrize(theta)
            lam = mvn_symmetrize(lam)

            if args.save_checkpoint_path:
                pickle.dump((theta, lam, samplers), open(args.save_checkpoint_path, "wb"))

        if args.save_reference_path:
            pickle.dump(theta, open(args.save_reference_path, "wb"))

if __name__ == "__main__":
    main()

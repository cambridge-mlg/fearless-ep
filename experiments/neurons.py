# neural response model
import argparse
import jax.numpy as jnp
import math
import pickle
from fearless_ep.base_families.niw import *
from fearless_ep.inference import *
from fearless_ep.numpyro import *
from jax import jacfwd
from jax.random import split
from jaxutil.functional import umap
from jaxutil.la import *
from jaxutil.random import rngcall
from jaxutil.tree import *
from expfam.distributions.diagonal_mvn import *
from expfam.distributions.mvn import *
from expfam.distributions.niw import *

jax.config.update("jax_enable_x64", True)

def _generate_synthetic_data(rng, n_trials, n_neurons):
    rng_mu, rng_V, rng_w, rng_x = split(rng, 4)
    mu = 2.5 + jax.random.normal(rng_mu, (n_neurons,))
    V = (lambda _: _.T @ _ + .1*jnp.eye(n_neurons))(
        jax.random.uniform(rng_V, (n_neurons,n_neurons), minval=-1., maxval=1.))
    V = V * n_neurons / jnp.trace(V)
    r = jax.random.multivariate_normal(rng_w, mu, V, (n_trials,))
    x = jax.random.poisson(rng_x, jnp.exp(r))
    return x

def _cholesky_log_diag_to_matrix(_):
    chol_V_tril, chol_V_log_diag = _
    chol_V_tril = trilm(chol_V_tril)
    chol_V_tril = jnp.pad(chol_V_tril, [(0,0)]*(chol_V_tril.ndim-2) + [(1,0),(0,1)])
    chol_V = chol_V_tril + diagm(jnp.exp(chol_V_log_diag))
    return mmp(chol_V, transpose(chol_V))

def _tilted_potential(F, theta, lam, factor, position):
    x = factor
    n_trials, n_neurons = x.shape
    mu, r = position[0], position[2]
    V = _cholesky_log_diag_to_matrix(position[1])
    assert(mu.shape == (n_neurons,))
    assert(V.shape == (n_neurons,n_neurons))
    assert(r.shape == (n_trials,n_neurons))

    # compute log Jacobian determinant for log-cholesky transformation
    unvec = tree_vec(position[1], True)[1]
    logabsJ = jnp.linalg.slogdet(
        jacfwd(
                lambda _: trilv(_cholesky_log_diag_to_matrix(unvec(_)))
            )(tree_vec(position[1])))[1]

    cavity = tree_sub(theta, lam)
    log_p_z = niw_log_prob(cavity, (mu, V))
    log_p_r = jax.scipy.stats.multivariate_normal.logpdf(r, mu, V)
    log_p_x = jax.scipy.stats.poisson.logpmf(x, jnp.exp(r))

    energy = log_p_z + jnp.sum(log_p_r, 0) + jnp.sum(log_p_x, axis=(0,1)) + logabsJ
    return -energy

# sample initial hmc positions from the prior
def _init_sampler_position(rng, F, theta, n_trials):
    rng_mu, rng_S, rng_w = split(rng, 3)
    Psi, delta, gamma, alpha = niw_standard_from_natural(theta)
    S = jax.random.multivariate_normal(rng_S, jnp.zeros(Psi.shape[-1]), jnp.linalg.inv(Psi), (int(round(alpha)),))
    V = jnp.linalg.inv(mmp(transpose(S), S))
    chol_V = jnp.linalg.cholesky(V)
    chol_V_tril = trilv(chol_V[...,:-1,:-1])
    chol_V_log_diag = jnp.log(diagv(chol_V))
    mu = jax.random.multivariate_normal(rng_mu, delta, V/gamma)
    w = jax.random.multivariate_normal(rng_w, mu, V, (n_trials,))    
    return mu, (chol_V_tril, chol_V_log_diag), w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['EP', 'EP_ETA', 'EP_MU'], default='EP_ETA')
    parser.add_argument('--n-outer', type=int, default=100_000, help='total number of outer updates')
    parser.add_argument('--n-inner', type=int, default=1, help='number of inner updates per outer update')
    parser.add_argument('--n-chains', type=int, default=1, help='number of sampling chains')
    parser.add_argument('--n-burnin', type=int, default=0, help='number of burn-in samples per inner update')
    parser.add_argument('--n-samp', type=int, default=1, help='total number of samples per inner update')
    parser.add_argument('--n-warmup', type=int, default=200, help='number of warm-up samples used for adaptation')
    parser.add_argument('--warmup-interval', type=int, default=400, help='number of outer updates per warmup phase')
    parser.add_argument('--tau', type=int, default=1, help='thinning ratio')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate / step size')
    parser.add_argument('--n-factors', type=int, default=8, help='number of factors')
    parser.add_argument('--n-trials', type=int, default=144, help='number of trials')
    parser.add_argument('--n-neurons', type=int, default=10, help='number of neurons')
    parser.add_argument('--data-seed', type=int, default=0, help='data generation random seed')
    parser.add_argument('--init-seed', type=int, default=0, help='sampler initialisation random seed')
    parser.add_argument('--load-reference-path', default=None, help='path to load reference posterior from')
    parser.add_argument('--save-reference-path', default=None, help='path to save reference posterior to')
    parser.add_argument('--load-checkpoint-path', default=None, help='path to load checkpoint from')
    parser.add_argument('--save-checkpoint-path', default=None, help='path to save checkpoint to')

    args = parser.parse_args()

    n_factors, n_trials, n_neurons = (
        args.n_factors, args.n_trials, args.n_neurons)
    n_trials_per_factor = n_trials//n_factors

    x = _generate_synthetic_data(jax.random.PRNGKey(args.data_seed), n_trials, n_neurons)
    x = x.reshape(n_factors, n_trials_per_factor, n_neurons)
    factors = x

    rng = jax.random.PRNGKey(args.init_seed)
    method = InferenceMethod[args.method]
    F = NiwFamily(args.n_neurons)

    # make sampling functions
    sampler_init_fn, sampler_warmup_fn, sampler_draw_fn = make_sampling_functions(
        F,
        args.n_chains,
        args.n_burnin,
        args.tau,
        partial(_init_sampler_position, n_trials=n_trials_per_factor),
        _tilted_potential,
        sample_transform_fn=lambda _: (_[0], _cholesky_log_diag_to_matrix(_[1])))

    # define prior
    eta_0_alpha = n_neurons + 2.
    eta_0_delta = 2.5
    eta_0_gamma = 1.
    eta_0 = niw_natural_from_standard((
        jnp.eye(n_neurons)*(eta_0_alpha - n_neurons - 1)*(1.25**2),
        jnp.ones(n_neurons)*eta_0_delta,
        jnp.array(eta_0_gamma),
        jnp.array(eta_0_alpha)))

    # initialize approximation parameters
    lam_0_scale = {
        InferenceMethod.EP: .0,
        InferenceMethod.EP_ETA: .0,
        InferenceMethod.EP_MU: .0,
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
            theta = niw_symmetrize(theta)
            lam = niw_symmetrize(lam)

            if args.save_checkpoint_path:
                pickle.dump((theta, lam, samplers), open(args.save_checkpoint_path, "wb"))

        if args.save_reference_path:
            pickle.dump(theta, open(args.save_reference_path, "wb"))

if __name__ == "__main__":
    main()

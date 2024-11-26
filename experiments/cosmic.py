# cosmic radiation data model -- see "expectation propagation as a way of life", vehtari et al. (2020), for details
import argparse
import jax.numpy as jnp
import math
import pickle
from fearless_ep.base_families.mvn import *
from fearless_ep.inference import *
from fearless_ep.numpyro import *
from jax.scipy.special import logit
from jax.nn import log_sigmoid, sigmoid
from jax.numpy import exp, log
from jax.random import split
from jaxutil.functional import umap
from jaxutil.random import rngcall
from jaxutil.tree import *
from expfam.distributions.diagonal_mvn import *
from expfam.distributions.mvn import *
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

class GlobalParams(NamedTuple):
    beta_0_mean: float
    beta_0_log_var: float
    beta_1_mean: float
    beta_1_log_var: float
    log_mu_1_mean: float
    log_mu_1_log_var: float
    log_sigma1_mean: float
    log_sigma1_log_var: float
    invsig_beta2_mean: float
    invsig_beta2_log_var: float
    log_mu_2_mean: float
    log_mu_2_log_var: float
    log_sigma2_mean: float
    log_sigma2_log_var: float
    invsig_pi_mean: float
    invsig_pi_log_var: float
    log_sigma_mean: float
    log_sigma_log_var: float

class LocalParams(NamedTuple):
    beta_0: float
    beta_1: float
    log_mu_1: float
    log_sigma1: float
    invsig_beta2: float
    log_mu_2: float
    log_sigma2: float
    invsig_pi: float
    log_sigma: float

# just because jax.random.normal doesn't let you pass loc or scale parameters
_sample_normal = lambda rng, mu, log_v: mu + jax.random.normal(rng)*exp(.5*log_v)

def _generate_synthetic_data(rng, n_sectors, n_observations):
    globals = GlobalParams(
        beta_0_mean = log(10.0),
        beta_0_log_var = log(.5**2),
        beta_1_mean = 2.0,
        beta_1_log_var = log(.1**2),
        log_mu_1_mean = log(1.0),
        log_mu_1_log_var = log(.5**2),
        log_sigma1_mean = -1.0,
        log_sigma1_log_var = log(.15**2),
        invsig_beta2_mean = logit(0.6),
        invsig_beta2_log_var = log((logit(0.1)-logit(0.125))**2),
        log_mu_2_mean = log(2.2),
        log_mu_2_log_var = log(.2**2),
        log_sigma2_mean = -1.,
        log_sigma2_log_var = log(.2**2),
        invsig_pi_mean = logit(0.5),
        invsig_pi_log_var = log((logit(0.5)-logit(0.6))**2),
        log_sigma_mean = log(0.25),
        log_sigma_log_var = log(.05**2))
    locals = []
    x, y = [], []
    switches = []
    for _ in range(n_sectors):
        beta_0, rng = rngcall(_sample_normal, rng, globals.beta_0_mean, globals.beta_0_log_var)
        beta_1, rng = rngcall(_sample_normal, rng, globals.beta_1_mean, globals.beta_1_log_var)
        log_mu_1, rng = rngcall(_sample_normal, rng, globals.log_mu_1_mean, globals.log_mu_1_log_var)
        log_sigma1, rng = rngcall(_sample_normal, rng, globals.log_sigma1_mean, globals.log_sigma1_log_var)
        invsig_beta2, rng = rngcall(_sample_normal, rng, globals.invsig_beta2_mean, globals.invsig_beta2_log_var)
        log_mu_2, rng = rngcall(_sample_normal, rng, globals.log_mu_2_mean, globals.log_mu_2_log_var)
        log_sigma2, rng = rngcall(_sample_normal, rng, globals.log_sigma2_mean, globals.log_sigma2_log_var)
        invsig_pi, rng = rngcall(_sample_normal, rng, globals.invsig_pi_mean, globals.invsig_pi_log_var)
        log_sigma, rng = rngcall(_sample_normal, rng, globals.log_sigma_mean, globals.log_sigma_log_var)
        locals.append(
            LocalParams(
                beta_0 = beta_0,
                beta_1 = beta_1,
                log_mu_1 = log_mu_1,
                log_sigma1 = log_sigma1,
                invsig_beta2 = invsig_beta2,
                log_mu_2 = log_mu_2,
                log_sigma2 = log_sigma2,
                invsig_pi = invsig_pi,
                log_sigma = log_sigma))
        logx, rng = rngcall(jax.random.uniform, rng, (n_observations,), minval=-1, maxval=3)
        f = beta_0 + beta_1*sigmoid((logx - exp(log_mu_1))/exp(log_sigma1))
        g = beta_0 + beta_1*sigmoid((logx - exp(log_mu_1))/exp(log_sigma1))*(
            1 - sigmoid(invsig_beta2)*exp(-.5*jnp.square((logx - exp(log_mu_2))/exp(log_sigma2))))
        c, rng = rngcall(jax.random.bernoulli, rng, sigmoid(invsig_pi), f.shape)
        logy, rng = rngcall(lambda _: jax.random.normal(_, f.shape)*jnp.exp(log_sigma) + c*f + (1-c)*g, rng)
        x.append(exp(logx))
        y.append(exp(logy))
        switches.append(c)
    x = jnp.stack(x)
    y = jnp.stack(y)
    z = jnp.array(globals)
    w = jnp.stack(list(map(jnp.array, locals)))
    switches = jnp.stack(switches)
    return x, y, z, w, switches

def _tilted_potential(F, theta, lam, factor, position):
    x, y = factor
    assert(x.shape == y.shape)
    n_sectors, _ = x.shape # (n_sectors, n_obs)
    z, w = position
    assert(z.shape == (len(GlobalParams._fields),))
    assert(w.shape == (n_sectors,len(LocalParams._fields)))
    globals = GlobalParams(*z)

    w_prior_natural_params = diagonal_mvn_natural_from_standard((
        jnp.array([
            globals.beta_0_mean,
            globals.beta_1_mean,
            globals.log_mu_1_mean,
            globals.log_sigma1_mean,
            globals.invsig_beta2_mean,
            globals.log_mu_2_mean,
            globals.log_sigma2_mean,
            globals.invsig_pi_mean,
            globals.log_sigma_mean]),
        exp(jnp.array([
            globals.beta_0_log_var,
            globals.beta_1_log_var,
            globals.log_mu_1_log_var,
            globals.log_sigma1_log_var,
            globals.invsig_beta2_log_var,
            globals.log_mu_2_log_var,
            globals.log_sigma2_log_var,
            globals.invsig_pi_log_var,
            globals.log_sigma_log_var]))))

    def _log_likelihood(w, logx, logy):
        locals = LocalParams(*w)
        beta_0 = locals.beta_0
        beta_1 = locals.beta_1
        mu_1 = exp(locals.log_mu_1)
        sigma1 = exp(locals.log_sigma1)
        beta2 = sigmoid(locals.invsig_beta2)
        mu_2 = exp(locals.log_mu_2)
        sigma2 = exp(locals.log_sigma2)
        invsig_pi = locals.invsig_pi
        sigma = exp(locals.log_sigma)
        f = beta_0 + beta_1*sigmoid((logx - mu_1)/sigma1)
        g = beta_0 + beta_1*sigmoid((logx - mu_1)/sigma1)*(
            1 - beta2*exp(-.5*jnp.square((logx - mu_2)/sigma2)))
        return jnp.sum(jnp.logaddexp(
            log_sigmoid(invsig_pi) + jax.scipy.stats.norm.logpdf(logy, f, sigma),
            log_sigmoid(-invsig_pi) + jax.scipy.stats.norm.logpdf(logy, g, sigma)), -1)

    energy = (log_prob(F, tree_sub(theta, lam), z)
        + jnp.sum(diagonal_mvn_log_prob(w_prior_natural_params, w), 0)
        + jnp.sum(vmap(_log_likelihood)(w, log(x), log(y)), 0))

    return -energy

def _init_sampler_position(rng, F, theta, n_sectors):
    rng_z, rng_w = split(rng)
    z = sample(rng_z, F, theta, ())
    globals = GlobalParams(*z)
    def _init_w(rng):
        beta_0, rng = rngcall(_sample_normal, rng, globals.beta_0_mean, globals.beta_0_log_var)
        beta_1, rng = rngcall(_sample_normal, rng, globals.beta_1_mean, globals.beta_1_log_var)
        log_mu_1, rng = rngcall(_sample_normal, rng, globals.log_mu_1_mean, globals.log_mu_1_log_var)
        log_sigma1, rng = rngcall(_sample_normal, rng, globals.log_sigma1_mean, globals.log_sigma1_log_var)
        invsig_beta2, rng = rngcall(_sample_normal, rng, globals.invsig_beta2_mean, globals.invsig_beta2_log_var)
        log_mu_2, rng = rngcall(_sample_normal, rng, globals.log_mu_2_mean, globals.log_mu_2_log_var)
        log_sigma2, rng = rngcall(_sample_normal, rng, globals.log_sigma2_mean, globals.log_sigma2_log_var)
        invsig_pi, rng = rngcall(_sample_normal, rng, globals.invsig_pi_mean, globals.invsig_pi_log_var)
        log_sigma, rng = rngcall(_sample_normal, rng, globals.log_sigma_mean, globals.log_sigma_log_var)
        return jnp.array(
            LocalParams(
                beta_0 = beta_0,
                beta_1 = beta_1,
                log_mu_1 = log_mu_1,
                log_sigma1 = log_sigma1,
                invsig_beta2 = invsig_beta2,
                log_mu_2 = log_mu_2,
                log_sigma2 = log_sigma2,
                invsig_pi = invsig_pi,
                log_sigma = log_sigma))
    w = vmap(_init_w)(split(rng_w, n_sectors))
    return z, w

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
    parser.add_argument('--n-factors', type=int, default=36, help='number of factors')
    parser.add_argument('--n-sectors', type=int, default=36, help='number of sectors')
    parser.add_argument('--n-observations', type=int, default=200, help='number of observations per group')
    parser.add_argument('--data-seed', type=int, default=0, help='data generation random seed')
    parser.add_argument('--init-seed', type=int, default=0, help='sampler initialisation random seed')
    parser.add_argument('--load-reference-path', default=None, help='path to load approximate posterior from')
    parser.add_argument('--save-reference-path', default=None, help='path to save approximate posterior to')
    parser.add_argument('--load-checkpoint-path', default=None, help='path to load checkpoint from')
    parser.add_argument('--save-checkpoint-path', default=None, help='path to save checkpoint to')

    args = parser.parse_args()

    n_factors, n_sectors, n_observations = (
        args.n_factors, args.n_sectors, args.n_observations)
    n_covariates = len(GlobalParams._fields)
    n_sectors_per_factor = n_sectors//n_factors

    x, y = _generate_synthetic_data(jax.random.PRNGKey(args.data_seed), n_sectors, n_observations)[:2]
    x = x.reshape(n_factors, n_sectors_per_factor, n_observations)
    y = y.reshape(n_factors, n_sectors_per_factor, n_observations)
    factors = x, y

    rng = jax.random.PRNGKey(args.init_seed)
    method = InferenceMethod[args.method]
    F = MvnFamily(n_covariates)

    # make sampling functions
    sampler_init_fn, sampler_warmup_fn, sampler_draw_fn = make_sampling_functions(
        F,
        args.n_chains,
        args.n_burnin,
        args.tau,
        partial(_init_sampler_position, n_sectors=n_sectors_per_factor),
        _tilted_potential,
        sample_transform_fn=lambda _: _[0])

    # define prior
    eta_0_h = jnp.zeros(n_covariates)
    eta_0_J = -.5 * jnp.diag(jnp.array([.1]*n_covariates))
    eta_0 = eta_0_h, eta_0_J

    # site parameter initialisation
    lam_0_scale = {
        InferenceMethod.EP: .5/n_factors,
        InferenceMethod.EP_ETA: .5/n_factors,
        InferenceMethod.EP_MU: .5/n_factors,
        InferenceMethod.SNEP: .5/n_factors,
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

    if args.load_checkpoint_path:
        theta, lam, samplers = pickle.load(open(args.load_checkpoint_path, "rb"))

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

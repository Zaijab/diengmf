"""
In this file, we provide a number of functions relating to the Epanechnikov RVs.
For some reason, very little stats packages provide such utilities.
Namely, we provide functions for the computation of `pdf`, `logpdf`, and `sampling`.
"""

import jax
import jax.numpy as jnp
from functools import partial

from jax.scipy.special import gamma
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker


#@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['normalized', 'debug'])
def pdf_epanechnikov(x: Float[Array, "state_dim"],
                     mu: Float[Array, "state_dim"],
                     sigma: Float[Array, "state_dim state_dim"],
                     normalized: bool = True,
                     debug: bool = False):
    n = mu.shape[0]
    standard_input = jnp.linalg.solve(jnp.linalg.cholesky(sigma), x - mu)
    normalizing_constant = 1 / (jnp.sqrt(jnp.linalg.det(sigma)))
    density = ((n + 2) / (2 * ((jnp.pi ** (n / 2)) / (gamma((n / 2) + 1))) * ((n + 4) ** ((n + 2) / 2)))) * ((n + 4) - (standard_input @ standard_input))

    density = jnp.where(
        standard_input @ standard_input <= (n + 4),
        normalizing_constant * density,
        0.0
    )
    return density


#@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['normalized', 'debug'])
def logpdf_epanechnikov(x: Float[Array, "state_dim"],
                        mu: Float[Array, "state_dim"],
                        sigma: Float[Array, "state_dim state_dim"],
                        normalized: bool = True,
                        debug: bool = False):
    return jnp.log(pdf_epanechnikov(x, mu, sigma, normalized, debug))


@partial(jax.jit, static_argnames=['num_samples', 'standard', 'debug'])
def sample_epanechnikov(key: Key[Array, "1"], mu, sigma, num_samples, standard=False, debug=False):
    n = mu.shape[0]
    key, subkey1, subkey2 = jax.random.split(key, 3)
    s = jax.random.multivariate_normal(subkey1, jnp.zeros(n), jnp.eye(n), (num_samples,))
    s_hat = jnp.sqrt(n + 4) * s / jnp.linalg.norm(s, axis=1, keepdims=True)
    kappa = jax.random.beta(subkey2, b=n / 2, a=2, shape=(num_samples,))[:, None]
    scaled_s = kappa * s_hat

    if standard:
        return scaled_s

    # Use symmetric sqrt (SVD)
    eigvals, eigvecs = jnp.linalg.eigh(sigma)
    sqrt_sigma = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T

    samples = mu + (sqrt_sigma @ scaled_s.T).T
    return samples

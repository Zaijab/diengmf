"""
In this file we provide multiple classes related to Gaussian Mixture Models.

Namely, if the user needs a simple GMM, they may use the GMM class wherein you specify the means, covariances, and weights.
It comes with a sample method once the user specifies all the necessary weights.

If a user wishes to sample from multiple GMMs, then there are two approaches.
Either they form a list of the GMMs and we can VMAP over the index.

gmm_funcs = [jax.tree_util.Partial(gmm.sample)] * 50 + [
    jax.tree_util.Partial(gmm_1.sample)
] * 50


index = jnp.arange(len(gmm_funcs))
subkeys = jax.random.split(subkey, len(gmm_funcs))


@eqx.filter_jit
@eqx.filter_vmap
def sample_parallel(i, key):
    return jax.lax.switch(i, gmm_funcs, key)


sample_parallel(index, subkeys).shape

The method we choose is to take the largest number of components (suppose gmm_1 has 3 components and gmm_2 has 5)
Then zero pad everything to be of the same size. The padding is done for you after initialization.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped, Int, Bool
from typing import List
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, jaxtyped, PyTree
from beartype import beartype as typechecker
from typing import List


@jaxtyped(typechecker=typechecker)
class GMM(eqx.Module):
    means: Float[Array, "num_components state_dim"]
    covs: Float[Array, "num_components state_dim state_dim"]
    weights: Float[Array, "num_components"]

    @jaxtyped(typechecker=typechecker)
    def __init__(self, means, covs, weights, max_components=1):
        max_components = max(means.shape[0], max_components)
        pad_width = max_components - means.shape[0]
        self.means = jnp.pad(means, ((0, pad_width), (0, 0)))
        self.covs = jnp.pad(covs, ((0, pad_width), (0, 0), (0, 0)))
        self.weights = jnp.pad(weights, (0, pad_width))

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample(self, key: Key[Array, ""]) -> Float[Array, "state_dim"]:
        sampling_weights = self.weights / jnp.sum(self.weights)
        gaussian_index_key, gaussian_state_key = jax.random.split(key)
        gaussian_index = jax.random.choice(
            key=gaussian_index_key,
            a=jnp.arange(self.weights.shape[0]),
            p=sampling_weights,
        )
        sample = jax.random.multivariate_normal(
            gaussian_state_key,
            mean=self.means[gaussian_index],
            cov=self.covs[gaussian_index],
        )
        return sample


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def merge_gmms(
    gmm1: GMM, gmm2: GMM, key: Key[Array, ""], target_components: int = 250
) -> GMM:
    target_components = min(
        target_components, gmm1.weights.shape[0] + gmm2.weights.shape[0]
    )
    # Compute total weights
    W1 = jnp.sum(gmm1.weights)
    W2 = jnp.sum(gmm2.weights)
    Z = W1 + W2

    # Sample target_components points using Algorithm 2 logic
    uniform_keys = jax.random.split(key, target_components)

    def sample_one(key_i):
        u = jax.random.uniform(key_i)
        return jax.lax.cond(
            u < W1 / Z, lambda: gmm1.sample(key_i), lambda: gmm2.sample(key_i)
        )

    samples = jax.vmap(sample_one)(uniform_keys)

    # Reconstruct GMM with uniform weights
    uniform_weights = jnp.full(
        target_components, jnp.array([W1 + W2]) / target_components
    )
    spatial_dimension = gmm1.means.shape[1]
    silverman_beta = (
        ((4) / (spatial_dimension + 2)) ** (2 / (spatial_dimension + 4))
    ) * ((target_components) ** (-(2) / (spatial_dimension + 4)))
    covariance = (silverman_beta / Z) * jnp.cov(samples.T)

    return GMM(
        samples, jnp.tile(covariance, (target_components, 1, 1)), uniform_weights
    )


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def merge_gmms(
    gmm1: GMM, gmm2: GMM, key: Key[Array, ""], target_components: int = 250
) -> GMM:

    # Compute total weights
    W1 = jnp.sum(gmm1.weights)
    W2 = jnp.sum(gmm2.weights)
    Z = W1 + W2

    # Sample target_components points using Algorithm 2 logic
    uniform_keys = jax.random.split(key, target_components)

    def sample_one(key_i):
        u = jax.random.uniform(key_i)
        return jax.lax.cond(
            u < W1 / Z, lambda: gmm1.sample(key_i), lambda: gmm2.sample(key_i)
        )

    samples = jax.vmap(sample_one)(uniform_keys)

    # Reconstruct GMM with uniform weights
    uniform_weights = jnp.full(
        target_components, jnp.array([W1 + W2]) / target_components
    )
    spatial_dimension = gmm1.means.shape[1]
    silverman_beta = (
        ((4) / (spatial_dimension + 2)) ** (2 / (spatial_dimension + 4))
    ) * ((target_components) ** (-(2) / (spatial_dimension + 4)))
    covariance = (silverman_beta / Z) * jnp.cov(samples.T)

    return GMM(
        samples, jnp.tile(covariance, (target_components, 1, 1)), uniform_weights
    )

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

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from distreqx.distributions import (AbstractProbDistribution,
                                    AbstractSampleLogProbDistribution)
from jaxtyping import Array, Float, Key, jaxtyped


@jaxtyped(typechecker=typechecker)
class GMM(AbstractSampleLogProbDistribution, AbstractProbDistribution, strict=True):
    ensemble: Float[Array, " n d"]
    cov: Float[Array, " n d d"]
    weights: Float[Array, " n"]

    def __init__(self, ensemble, cov=None, weights=None, bandwidth_scale: float = 1.0):
        self.ensemble = ensemble

        if cov is None or weights is None:
            n, d = ensemble.shape
            self.weights = jnp.ones(n) / n
            h = bandwidth_scale * ((4 / (d + 2)) ** (2 / (d + 4))) * (n ** (-2 / (d + 4)))
            cov = h * jnp.cov(ensemble.T)
            self.cov = jnp.tile(cov, (n, 1, 1))
        else:
            self.cov = cov
            self.weights = weights

    @classmethod
    def from_samples(cls, samples: Float[Array, " n d"], bandwidth_scale: float = 1.0):
        n, d = samples.shape
        weights = jnp.ones(n) / n
        h = bandwidth_scale * ((4 / (d + 2)) ** (2 / (d + 4))) * (n ** (-2 / (d + 4)))
        cov = h * jnp.cov(samples.T)
        cov = jnp.tile(cov, (n, 1, 1))
        return cls(samples, cov, weights)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.ensemble.shape[1],)

    def sample(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        component_key, sample_key = jax.random.split(key)
        idx = jax.random.choice(component_key, self.ensemble.shape[0], p=self.weights)
        mean = self.ensemble[idx]
        return jax.random.multivariate_normal(sample_key, self.ensemble[idx], self.cov[idx])

    def log_prob(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        component_logprobs = jax.vmap(
            lambda mean: jax.scipy.stats.multivariate_normal.logpdf(
                value, mean, self.cov
            )
        )(self.ensemble)
        return jax.scipy.special.logsumexp(component_logprobs + jnp.log(self.weights))

    def entropy(self) -> Float[Array, ""]:
        raise NotImplementedError

    def log_cdf(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def cdf(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def survival_function(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def log_survival_function(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def mean(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def median(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def variance(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def stddev(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def mode(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs) -> Float[Array, ""]:
        raise NotImplementedError

@jaxtyped(typechecker=typechecker)
class EnsembleGaussian(AbstractSampleLogProbDistribution, AbstractProbDistribution, strict=True):
    ensemble: Float[Array, " n d"]
    mean: Float[Array, " d"]
    cov: Float[Array, " d d"]

    def __init__(self, ensemble, mean=None, cov=None):
        self.ensemble = ensemble

        if mean is None:
            self.mean = jnp.mean(ensemble, axis=0)
        else:
            self.mean = mean
            
        if cov is None:
            self.cov = jnp.cov(ensemble.T)
        else:
            self.cov = cov

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.ensemble.shape[1],)

    def sample(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        return jax.random.multivariate_normal(key, self.mean, self.cov)

    def log_prob(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        return jax.scipy.stats.multivariate_normal.logpdf(value, mean=self.mean, cov=self.cov)

    def entropy(self) -> Float[Array, ""]:
        raise NotImplementedError

    def log_cdf(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def cdf(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def survival_function(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def log_survival_function(self, value: Float[Array, " d"]) -> Float[Array, ""]:
        raise NotImplementedError

    def mean(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def median(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def variance(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def stddev(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def mode(self) -> Float[Array, " d"]:
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs) -> Float[Array, ""]:
        raise NotImplementedError

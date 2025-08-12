from diengmf.losses.stochastic_filters import evaluate_filter
from diengmf.dynamical_systems import AbstractDynamicalSystem, Ikeda
from diengmf.measurement_systems import AbstractMeasurementSystem, RangeSensor
from diengmf.stochastic_filters import AbstractFilter, EnGMF
from jaxtyping import Array, Float, Key

import distrax


###

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from typing import Callable
from diengmf.stochastic_filters import AbstractFilter
from diengmf.measurement_systems import AbstractMeasurementSystem
from diengmf.dynamical_systems import AbstractDynamicalSystem



@jax.jit
@jax.vmap
def sample_gaussian_mixture(key: Key[Array, ""], point: Float[Array, "state_dim"], cov: Float[Array, "state_dim state_dim"]) -> Float[Array, "state_dim"]:
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


class EnGMF(AbstractFilter, strict=True):
    # State Space Model
    dynamical_system: AbstractDynamicalSystem
    measurement_system: AbstractMeasurementSystem

    # Belief Representation
    ensemble_size: int = 100
    
    # Hyperparameter
    silverman_bandwidth_scaling: float = 1.0

    sampling_function: Callable[[Key[Array, ""], Float[Array, "state_dim"], Float[Array, "state_dim state_dim"]], Float[Array, "state_dim"]] = jax.tree_util.Partial(sample_gaussian_mixture)
    
    debug: bool = False


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initialize(self, key: Key[Array, ""], belief: distrax.Distribution) -> distrax.Distribution:
        return silverman_kde_estimate(belief.sample(seed=key, sample_shape=self.ensemble_size), self.silverman_bandwidth_scaling)

    def predict(self, posterior_belief: distrax.Distribution) -> distrax.Distribution:
        posterior_belief
        pass
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_ensemble: Float[Array, "batch_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, " batch_size state_dim"]:
        # key: Key[Array, ""]
        # subkey: Key[Array, ""]
        # subkeys: Key[Array, "batch_dim"]

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, prior_ensemble.shape[0])

        # if self.debug:
        #     assert isinstance(subkeys, Key[Array, "batch_dim"])

        bandwidth = self.silverman_bandwidth_scaling * (
            (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
        ) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T) + 1e-8 * jnp.eye(6)

        state_dim = emperical_covariance.shape[0]
        i_indices = jnp.arange(state_dim)[:, None]
        j_indices = jnp.arange(state_dim)[None, :]
        distances = jnp.abs(i_indices - j_indices)

        # Gaussian localization with radius L
        L = 3.0  # or 4.0
        rho = jnp.exp(-(distances**2) / (2 * L**2))

        emperical_covariance = emperical_covariance * rho

        # if self.debug:
        #     jax.debug.callback(is_positive_definite, emperical_covariance)
        #     assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

        mixture_covariance = bandwidth * emperical_covariance
        # if self.debug:
        #     assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        posterior_ensemble, posterior_covariances, logposterior_weights = jax.vmap(
            self.update_point,
            in_axes=(0, None, None, None),
        )(
            prior_ensemble,
            mixture_covariance,
            measurement,
            measurement_system,
        )

        # if self.debug:
        #     assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
        #     assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
        #     assert isinstance(
        #         posterior_covariances, Float[Array, "batch_dim state_dim state_dim"]
        #     )
        #     jax.self.debug.callback(has_nan, posterior_covariances)

        # Scale Weights
        m = jnp.max(logposterior_weights)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        # if self.debug:
        #     assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        # Prevent Degenerate Particles
        variable = jax.random.choice(
            subkey,
            prior_ensemble.shape[0],
            shape=(self.ensemble_size,),
            p=posterior_weights,
        )
        posterior_ensemble = posterior_ensemble[variable, ...]
        posterior_covariances = posterior_covariances[variable, ...]

        # if self.debug:
        #     jax.self.debug.callback(has_nan, posterior_covariances)

        posterior_samples = self.sampling_function(
            subkeys, posterior_ensemble, posterior_covariances
        )
        # if self.debug:
        #     assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        return posterior_samples


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, " state_dim"], # x_{k|k-1}^(i)
        prior_mixture_covariance: Float[Array, "state_dim state_dim"], # \hat{P}_{k|k-1}^(i)
        measurement: Float[Array, "measurement_dim"], # z
        measurement_system: AbstractMeasurementSystem, # h
    ) -> tuple[Float[Array, "state_dim"], Float[Array, "state_dim state_dim"], Float[Array, ""]]:
        ### (eq. 21)
        # H_{k}^{(i)} = \frac{\partial h}{\partial x} (x_{k|k-1}^(i))
        measurement_jacobian = jax.jacfwd(measurement_system)(point)

        # if self.debug:
        #     assert isinstance(
        #         measurement_jacobian, Float[Array, "measurement_dim state_dim"]
        #     ), measurement_jacobian.shape
            
        ### (eq. 19)
        # S_k^(i) = H_k^(i) P_{k | k - 1}^(i) H_k^(i) + R

        innovation_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_system.covariance
        )
        innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize
        innovation_cov = innovation_cov + 1e-12 * jnp.eye(innovation_cov.shape[0])
        # if self.debug:
        #     assert isinstance(innovation_cov, Float[Array, "measurement_dim measurement_dim"])

        ### (eq. 18)

        # K_k^(i) = P H.T S^(-1)
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, measurement_jacobian @ prior_mixture_covariance).T

        # if self.debug:
        #     assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
        #     # jax.debug.print("Hello {}", jnp.allclose(kalman_gain_unstable, kalman_gain))

        ### (eq. 17)
        
        # \hat{P}_{k | k}^{(i)} = \hat{P}_{k | k - 1}^{(i)} - K_{k}^{(i)} H_{k}^{(i)} \hat{P}_{k | k - 1}^{(i)}
        # We may, of course, factor to the right
        # \hat{P}_{k | k}^{(i)} = ( I - K_{k}^{(i)} H_{k}^{(i)} ) \hat{P}_{k | k - 1}^{(i)}
        posterior_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance @ (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ).T + kalman_gain @ measurement_system.covariance @ kalman_gain.T
        

        # if self.debug:
        #     assert isinstance(
        #         posterior_covariance, Float[Array, "state_dim state_dim"]
        #     )

        ### (eq. 16)
        
        # \hat{x}_{k | k}^{(i)} = \hat{x}_{k | k - 1}^{(i)} + K_{k}^{(i)} ( z - h(\hat{x}_{k | k - 1}^{(i)}))
        posterior_point = point + kalman_gain @ (measurement - measurement_system(point))

        # if self.debug:
        #     assert isinstance(point, Float[Array, "state_dim"])
        #     assert measurement_system(point).shape == measurement.shape
        #     assert posterior_point.shape == point.shape

        ### (eq. 22)
        # \xi_{k}^{(i)} = N(z; \hat{x}_{k | k - 1}^{(i)}, S_{k}^{(i)})
        logposterior_weight = jsp.stats.multivariate_normal.logpdf(
            measurement,
            mean=measurement_system(point),
            cov=innovation_cov
        )

        # if self.debug:
        #     assert isinstance(logposterior_weight, Float[Array, ""])

        return posterior_point, posterior_covariance, logposterior_weight



###

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def filter_update(
    carry: tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    key: Key[Array, "..."],
    dynamical_system: AbstractDynamicalSystem,
    measurement_system: AbstractMeasurementSystem,
    update: Callable[
        [
            Key[Array, "..."],
            Float[Array, "batch_size state_dim"],
            Float[Array, "measurement_dim"],
            AbstractMeasurementSystem,
            bool,
        ],
        Float[Array, "batch_size state_dim"],
    ],
    debug: bool = False,
) -> tuple[
    tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    Float[Array, "state_dim"],
]:
    (prior_ensemble, true_state) = carry
    ensemble_updating_key, measurement_key = jax.random.split(key)
    updated_ensemble = update(
        prior_ensemble=prior_ensemble,
        measurement=measurement_system(true_state, measurement_key),
        measurement_system=measurement_system,
        key=ensemble_updating_key,
    )
    error = true_state - jnp.mean(updated_ensemble, axis=0)
    if debug:
        jax.debug.callback(plot_update, prior_ensemble, updated_ensemble, true_state)
    ensemble_next = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, updated_ensemble)
    true_state_next = dynamical_system.flow(0.0, 1.0, true_state)
    new_carry = (ensemble_next, true_state_next)
    return new_carry, error


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_filter(
    key: Key[Array, "..."],
    stochastic_filter: AbstractFilter,
    debug: bool = False,
) -> Float[Array, "..."]:
    ensemble_size = 100
    burn_in_time = 100
    measurement_time = 1000
    total_steps = burn_in_time + measurement_time

    key, subkey = jax.random.split(key)
    initial_ensemble = dynamical_system.generate(subkey, batch_size=ensemble_size)
    initial_true_state = dynamical_system.initial_state()

    keys = jax.random.split(key, num=(total_steps,))

    scan_step = jax.tree_util.Partial(
        filter_update,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        update=stochastic_filter.update,
    )

    (final_carry, errors_over_time) = jax.lax.scan(
        scan_step, (initial_ensemble, initial_true_state), keys
    )

    errors_past_burn_in = errors_over_time[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))

    return rmse

###

dynamical_system: AbstractDynamicalSystem = Ikeda()
measurement_system: AbstractMeasurementSystem = RangeSensor
stochastic_filter: AbstractFilter = EnGMF(dynamical_system=dynamical_system, measurement_system=measurement_system)

key = jax.random.key(10)
key, subkey = jax.random.split(key)


# evaluate_filter(key, stochastic_filter)

from distreqx.distributions import (
    MultivariateNormalTri,
    MixtureSameFamily,
    Independent,
    Categorical,
    AbstractDistribution,
    AbstractProbDistribution
)
initial_belief = MultivariateNormalTri(jnp.array([1.25, 0.0]), (1/2) * jnp.eye(2)) #.MultivariateNormalFullCovariance(loc=jnp.array([1.25, 0.0]), covariance_matrix=(1/4) * jnp.eye(2))
samples = eqx.filter_vmap(initial_belief.sample)(jax.random.split(subkey, 50))

@jaxtyped(typechecker=typechecker)
class SilvermanKDE(AbstractProbDistribution):
    means: Float[Array, "n d"]
    cov: Float[Array, "d d"]
    weights: Float[Array, "n"]
    
    @classmethod
    def from_samples(cls, samples: Float[Array, "n d"], bandwidth_scale: float = 1.0):
        n, d = samples.shape
        weights = jnp.ones(n) / n
        h = bandwidth_scale * ((4 / (d + 2)) ** (2 / (d + 4))) * (n ** (-2 / (d + 4)))
        cov = h * jnp.cov(samples.T)
        return cls(samples, cov, weights)
    
    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.means.shape[1],)
    
    def sample(self, key: Key[Array, ""]) -> Float[Array, "d"]:
        # Your custom sampling logic here
        component_key, sample_key = jax.random.split(key)
        idx = jax.random.choice(component_key, self.means.shape[0], p=self.weights)
        mean = self.means[idx]
        return jax.random.multivariate_normal(sample_key, mean, self.cov)
        
    def log_prob(self, x: Float[Array, "d"]) -> Float[Array, ""]:
        # Mixture log probability
        component_logprobs = jax.vmap(
            lambda mean: jax.scipy.stats.multivariate_normal.logpdf(x, mean, self.cov)
        )(self.means)
        return jax.scipy.special.logsumexp(component_logprobs + jnp.log(self.weights))

    def log_cdf(self):
        raise NotImplementedError

    def cdf(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError

    def log_survival_function(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def median(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError


from distreqx.distributions import (
    AbstractSampleLogProbDistribution, 
    AbstractProbDistribution
)

@jaxtyped(typechecker=typechecker)
class SilvermanKDE(AbstractSampleLogProbDistribution, AbstractProbDistribution, strict=True):
    means: Float[Array, "n d"]
    cov: Float[Array, "d d"] 
    weights: Float[Array, "n"]

    @classmethod
    def from_samples(cls, samples: Float[Array, "n d"], bandwidth_scale: float = 1.0):
        n, d = samples.shape
        weights = jnp.ones(n) / n
        h = bandwidth_scale * ((4 / (d + 2)) ** (2 / (d + 4))) * (n ** (-2 / (d + 4)))
        cov = h * jnp.cov(samples.T)
        return cls(samples, cov, weights)
    
    
    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.means.shape[1],)
    
    def sample(self, key: Key[Array, ""]) -> Float[Array, "d"]:
        component_key, sample_key = jax.random.split(key)
        idx = jax.random.choice(component_key, self.means.shape[0], p=self.weights)
        mean = self.means[idx]
        return jax.random.multivariate_normal(sample_key, mean, self.cov)
        
    def log_prob(self, value: Float[Array, "d"]) -> Float[Array, ""]:
        component_logprobs = jax.vmap(
            lambda mean: jax.scipy.stats.multivariate_normal.logpdf(value, mean, self.cov)
        )(self.means)
        return jax.scipy.special.logsumexp(component_logprobs + jnp.log(self.weights))
    
    def entropy(self) -> Float[Array, ""]:
        raise NotImplementedError
        
    def log_cdf(self, value: Float[Array, "d"]) -> Float[Array, ""]:
        raise NotImplementedError
        
    def cdf(self, value: Float[Array, "d"]) -> Float[Array, ""]:
        raise NotImplementedError
        
    def survival_function(self, value: Float[Array, "d"]) -> Float[Array, ""]:
        raise NotImplementedError
        
    def log_survival_function(self, value: Float[Array, "d"]) -> Float[Array, ""]:
        raise NotImplementedError
        
    def mean(self) -> Float[Array, "d"]:
        raise NotImplementedError
        
    def median(self) -> Float[Array, "d"]:
        raise NotImplementedError
        
    def variance(self) -> Float[Array, "d"]:
        raise NotImplementedError
        
    def stddev(self) -> Float[Array, "d"]:
        raise NotImplementedError
        
    def mode(self) -> Float[Array, "d"]:
        raise NotImplementedError
        
    def kl_divergence(self, other_dist, **kwargs) -> Float[Array, ""]:
        raise NotImplementedError
    
SilvermanKDE.from_samples(samples).prob(samples[0, :])

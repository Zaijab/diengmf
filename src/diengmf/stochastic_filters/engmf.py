import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from typing import Callable
from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.dynamical_systems import AbstractDynamicalSystem



@jax.jit
@jax.vmap
def sample_gaussian_mixture(key: Key[Array, ""], point: Float[Array, "state_dim"], cov: Float[Array, "state_dim state_dim"]) -> Float[Array, "state_dim"]:
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


class EnGMF(AbstractFilter, strict=True):
    ensemble_size: int = 50
    debug: bool = False
    sampling_function: Callable[[Key[Array, ""], Float[Array, "state_dim"], Float[Array, "state_dim state_dim"]], Float[Array, "state_dim"]] = jax.tree_util.Partial(sample_gaussian_mixture)
    silverman_bandwidth_scaling: float = 1.0
    
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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from diengmf.stochastic_filters import AbstractFilter
from diengmf.measurement_systems import AbstractMeasurementSystem


@jaxtyped(typechecker=typechecker)
class EnKF(AbstractFilter, strict=True):
    inflation_factor: float = 1.01
    debug: bool = False
    ensemble_size: int = 50

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, "state_dim"], # x_{k|k-1}^(i)
        prior_mixture_covariance: Float[Array, "state_dim state_dim"], # \hat{P}_{k|k-1}^(i)
        measurement: Float[Array, "measurement_dim"], # z
        measurement_system: AbstractMeasurementSystem, # h
    ):
        ### (eq. 21)
        # H_{k}^{(i)} = \frac{\partial h}{\partial x} (x_{k|k-1}^(i))
        measurement_jacobian = jax.jacfwd(measurement_system)(point)

        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            ), measurement_jacobian.shape
            
        ### (eq. 19)
        # S_k^(i) = H_k^(i) P_{k | k - 1}^(i) H_k^(i) + R

        innovation_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_system.covariance
        )
        innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize
        if self.debug:
            assert isinstance(innovation_cov, Float[Array, "measurement_dim measurement_dim"])

        ### (eq. 18)

        # K_k^(i) = P H.T S^(-1)
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, measurement_jacobian @ prior_mixture_covariance).T

        if self.debug:
            assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
            # jax.debug.print("Hello {}", jnp.allclose(kalman_gain_unstable, kalman_gain))

        ### (eq. 17)
        
        # \hat{P}_{k | k}^{(i)} = \hat{P}_{k | k - 1}^{(i)} - K_{k}^{(i)} H_{k}^{(i)} \hat{P}_{k | k - 1}^{(i)}
        # We may, of course, factor to the right
        # \hat{P}_{k | k}^{(i)} = ( I - K_{k}^{(i)} H_{k}^{(i)} ) \hat{P}_{k | k - 1}^{(i)}
        posterior_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance

        if self.debug:
            assert isinstance(
                posterior_covariance, Float[Array, "state_dim state_dim"]
            )

        ### (eq. 16)
        
        # \hat{x}_{k | k}^{(i)} = \hat{x}_{k | k - 1}^{(i)} + K_{k}^{(i)} ( z - h(\hat{x}_{k | k - 1}^{(i)}))
        posterior_point = point + kalman_gain @ (measurement - measurement_system(point))

        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])
            assert measurement_system(point).shape == measurement.shape
            assert posterior_point.shape == point.shape

        ### (eq. 22)
        # \xi_{k}^{(i)} = N(z; \hat{x}_{k | k - 1}^{(i)}, S_{k}^{(i)})
        logposterior_weight = jsp.stats.multivariate_normal.logpdf(
            measurement,
            mean=measurement_system(point),
            cov=innovation_cov
        )

        if self.debug:
            assert isinstance(logposterior_weight, Float[Array, ""])

        return posterior_point, posterior_covariance, logposterior_weight



    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        mean = jnp.mean(prior_ensemble, axis=0)

        if self.debug:
            jax.debug.print("{shape}", mean.shape)

        inflated = mean + self.inflation_factor * (prior_ensemble - mean)
        ensemble_covariance = jnp.cov(inflated.T)


        keys = jax.random.split(key, prior_ensemble.shape[0])
        # updated_ensemble = jax.vmap(update_ensemble_point)(inflated, keys)
        updated_ensemble, posterior_covariances, logposterior_weights = jax.vmap(
            self.update_point,
            in_axes=(0, None, None, None),
        )(
            inflated,
            ensemble_covariance,
            measurement,
            measurement_system,
        )

        return updated_ensemble

@jaxtyped(typechecker=typechecker)
class UKF(AbstractFilter, strict=True):
    inflation_factor: float = 1.01
    debug: bool = False
    ensemble_size: int = 13

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, "state_dim"], # x_{k|k-1}^(i)
        prior_mixture_covariance: Float[Array, "state_dim state_dim"], # \hat{P}_{k|k-1}^(i)
        measurement: Float[Array, "measurement_dim"], # z
        measurement_system: AbstractMeasurementSystem, # h
    ):
        ### (eq. 21)
        # H_{k}^{(i)} = \frac{\partial h}{\partial x} (x_{k|k-1}^(i))
        measurement_jacobian = jax.jacfwd(measurement_system)(point)

        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            ), measurement_jacobian.shape
            
        ### (eq. 19)
        # S_k^(i) = H_k^(i) P_{k | k - 1}^(i) H_k^(i) + R

        innovation_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_system.covariance
        )
        innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize
        if self.debug:
            assert isinstance(innovation_cov, Float[Array, "measurement_dim measurement_dim"])

        ### (eq. 18)

        # K_k^(i) = P H.T S^(-1)
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, measurement_jacobian @ prior_mixture_covariance).T

        if self.debug:
            assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
            # jax.debug.print("Hello {}", jnp.allclose(kalman_gain_unstable, kalman_gain))

        ### (eq. 17)
        
        # \hat{P}_{k | k}^{(i)} = \hat{P}_{k | k - 1}^{(i)} - K_{k}^{(i)} H_{k}^{(i)} \hat{P}_{k | k - 1}^{(i)}
        # We may, of course, factor to the right
        # \hat{P}_{k | k}^{(i)} = ( I - K_{k}^{(i)} H_{k}^{(i)} ) \hat{P}_{k | k - 1}^{(i)}
        posterior_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance

        if self.debug:
            assert isinstance(
                posterior_covariance, Float[Array, "state_dim state_dim"]
            )

        ### (eq. 16)
        
        # \hat{x}_{k | k}^{(i)} = \hat{x}_{k | k - 1}^{(i)} + K_{k}^{(i)} ( z - h(\hat{x}_{k | k - 1}^{(i)}))
        posterior_point = point + kalman_gain @ (measurement - measurement_system(point))

        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])
            assert measurement_system(point).shape == measurement.shape
            assert posterior_point.shape == point.shape

        ### (eq. 22)
        # \xi_{k}^{(i)} = N(z; \hat{x}_{k | k - 1}^{(i)}, S_{k}^{(i)})
        logposterior_weight = jsp.stats.multivariate_normal.logpdf(
            measurement,
            mean=measurement_system(point),
            cov=innovation_cov
        )

        if self.debug:
            assert isinstance(logposterior_weight, Float[Array, ""])

        return posterior_point, posterior_covariance, logposterior_weight



    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "13 state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        mean = jnp.mean(prior_ensemble, axis=0)

        if self.debug:
            jax.debug.print("{shape}", mean.shape)

        inflated = mean + self.inflation_factor * (prior_ensemble - mean)
        ensemble_covariance = jnp.cov(inflated.T)


        keys = jax.random.split(key, prior_ensemble.shape[0])
        # updated_ensemble = jax.vmap(update_ensemble_point)(inflated, keys)
        updated_ensemble, posterior_covariances, logposterior_weights = jax.vmap(
            self.update_point,
            in_axes=(0, None, None, None),
        )(
            inflated,
            ensemble_covariance,
            measurement,
            measurement_system,
        )

        # posterior_sigma_points = self.generate_sigma_points(posterior_mean, posterior_cov)

        return updated_ensemble

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from diengmf.measurement_systems.measurement_system_abc import AbstractMeasurementSystem


@jaxtyped(typechecker=typechecker)
class AdjacentPairMeasurement(AbstractMeasurementSystem):
    covariance: Float[Array, "20 20"]

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def likelihood(
        self,
        state: Float[Array, "state_dim"],
        measurement: Float[Array, "measurement_dim"],
        **kwargs,
    ) -> Float[Array, ""]:
        """
        Returns the likelihood of a point given a measurement.
        """
        return jax.scipy.stats.multivariate_normal.pdf(
            self(state), mean=measurement, cov=self.covariance
        )

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self, state: Float[Array, "state_dim"], key: Key[Array, ""] | None = None
    ) -> Float[Array, "20"]:
        # [h(x)]_i = sqrt(x_{2i + 1}^2 + x_{2i + 2}^2) for i=1:20
        # Note: adjusting for 0-based indexing, so i goes from 0 to 19
        measurements = []
        for i in range(20):
            x_2i_plus_1 = state[2 * i]  # 0-based: x_{2i}
            x_2i_plus_2 = state[2 * i + 1]  # 0-based: x_{2i+1}
            measurement_i = jnp.sqrt(x_2i_plus_1**2 + x_2i_plus_2**2)
            measurements.append(measurement_i)

        perfect_measurements = jnp.array(measurements)

        if key is None:
            noise = jnp.zeros_like(perfect_measurements)
        else:
            noise = jax.random.multivariate_normal(key, jnp.zeros(20), self.covariance)

        return perfect_measurements + noise

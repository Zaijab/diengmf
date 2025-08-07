import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from diengmf.measurement_systems import AbstractMeasurementSystem


@jaxtyped(typechecker=typechecker)
@jax.jit
def norm_measurement(
    state: Float[Array, "state_dim"],
    key: Key[Array, ""] | None = None,
    covariance: Float[Array, "1 1"] = jnp.array([[1.0]]),
) -> Float[Array, "1"]:
    perfect_measurement = jnp.linalg.norm(state)
    noise = 0 if key is None else jnp.sqrt(covariance) * jax.random.normal(key)
    return (perfect_measurement + noise).reshape(-1)


@jaxtyped(typechecker=typechecker)
class RangeSensor(AbstractMeasurementSystem):
    covariance: Float[Array, "1 1"] = eqx.field(
        default_factory=lambda: jnp.array([[0.25]])
    )


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
    ) -> Float[Array, "1"]:
        perfect_measurement = jnp.linalg.norm(state[:3])
        noise = 0 if key is None else jnp.sqrt(self.covariance) * jax.random.normal(key)
        return (perfect_measurement + noise).reshape(-1)

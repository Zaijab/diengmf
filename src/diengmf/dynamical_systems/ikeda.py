import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped
import equinox as eqx
from diengmf.dynamical_systems import (
    AbstractInvertibleDiscreteDynamicalSystem,
)


@jaxtyped(typechecker=typechecker)
class Ikeda(AbstractInvertibleDiscreteDynamicalSystem, strict=True):
    u: float = 0.9
    batch_size: int = 10**3

    @property
    def dimension(self):
        return 2

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        state = jnp.array([1.25, 0])

        if key is None:
            noise = 0
        else:
            noise = jax.random.multivariate_normal(
                key,
                mean=jnp.zeros(self.dimension),
                cov=(1 / 128) * jnp.eye(self.dimension),
            )

        return state + noise

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        x1, x2 = x[0], x[1]
        t = 0.4 - (6 / (1 + x1**2 + x2**2))
        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        x1_new = 1 + self.u * (x1 * cos_t - x2 * sin_t)
        x2_new = self.u * (x1 * sin_t + x2 * cos_t)
        return jnp.stack((x1_new, x2_new), axis=-1)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def backward(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        x1_unscaled = (x[0] - 1) / self.u
        x2_unscaled = x[1] / self.u
        t = 0.4 - (6 / (1 + x1_unscaled**2 + x2_unscaled**2))
        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        x1_prev = x1_unscaled * cos_t + x2_unscaled * sin_t
        x2_prev = -x1_unscaled * sin_t + x2_unscaled * cos_t
        return jnp.stack((x1_prev, x2_prev), axis=-1)

    @eqx.filter_jit
    def ikeda_attractor_discriminator(
        self, x: Float[Array, "*batch 2"], ninverses: int = 10, u: float = 0.9
    ) -> Bool[Array, "*batch"]:

        threshold_squared = 1.0 / (1.0 - u)

        def scan_fn(state, _):
            x_curr, is_outside = state
            x_inv = self.backward(x_curr, u)
            norm_squared = jnp.sum(x_inv**2, axis=-1)
            new_is_outside = is_outside | (norm_squared > threshold_squared)
            return (x_inv, new_is_outside), None

        init_state = (x, jnp.zeros(x.shape[:-1], dtype=bool))

        (x_final, is_outside), _ = jax.lax.scan(
            scan_fn,
            init_state,
            None,
            length=ninverses,
        )

        return ~is_outside

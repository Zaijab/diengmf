import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Key, jaxtyped
from beartype import beartype as beartype


class CustomLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    use_bias: bool

    def __init__(self, in_features, out_features, use_bias=True, *, key) -> None:
        weight_key, bias_key = jax.random.split(key)
        scale = 1.0 / (in_features**0.5)  # Xavier initialization
        self.weight = jax.random.normal(weight_key, (out_features, in_features)) * scale
        self.bias = (
            jax.random.normal(bias_key, (out_features,)) * scale if use_bias else None
        )
        self.use_bias = use_bias

    def __call__(self, x, inverse=False):
        return self.forward(x) if not inverse else self.inverse(x)

    def forward(self, x):
        # Forward transformation: x → y
        y = x @ self.weight.T
        if self.use_bias:
            y = y + self.bias
        # Log determinant of Jacobian for forward transform
        log_det = jnp.linalg.slogdet(self.weight)[1]
        return y, log_det

    def inverse(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        # Reverse transformation: y → x
        if self.use_bias:
            x = x - self.bias
        y = x @ jnp.linalg.inv(self.weight).T
        # Negative log determinant for reverse transform
        log_det = -jnp.linalg.slogdet(self.weight)[1]
        return y, log_det

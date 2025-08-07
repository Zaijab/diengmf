import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors


class TFPBijectorWrapper(eqx.Module):
    """Simple equinox wrapper for TFP bijectors."""
    
    bijector: tfb.Bijector
    
    def __init__(self, tfp_bijector):
        self.bijector = tfp_bijector
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.bijector.forward(x)
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.bijector.inverse(y)
    
    @jaxtyped(typechecker=typechecker)
    def forward_and_log_det(self, x: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        y = self.bijector.forward(x)
        log_det = self.bijector.forward_log_det_jacobian(x, event_ndims=0)
        return y, log_det
    
    @jaxtyped(typechecker=typechecker)
    def inverse_and_log_det(self, y: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        x = self.bijector.inverse(y)
        log_det = self.bijector.inverse_log_det_jacobian(y, event_ndims=0)
        return x, log_det


def test_float64_tfp():
    """Test TFP bijectors with float64."""
    
    print(f"JAX default dtype: {jnp.array(1.0).dtype}")
    
    # Create TFP bijector with float64 parameters
    # Key insight: make sure ALL parameters are float64
    concentration = jnp.array(2.0, dtype=jnp.float64)
    scale = jnp.array(1.0, dtype=jnp.float64)
    
    # Create the TFP bijector
    tfp_bijector = tfb.WeibullCDF(concentration=concentration, scale=scale)
    
    # Wrap it
    wrapped_bijector = TFPBijectorWrapper(tfp_bijector)
    
    # Test data
    key = jax.random.PRNGKey(42)
    x_test = jax.random.uniform(key, (3,), minval=0.1, maxval=5.0)
    
    print(f"Input: {x_test}")
    print(f"Input dtype: {x_test.dtype}")
    
    # Test forward
    y = wrapped_bijector.forward(x_test)
    print(f"Forward: {y}")
    print(f"Forward dtype: {y.dtype}")
    
    # Test inverse
    x_reconstructed = wrapped_bijector.inverse(y)
    print(f"Reconstructed: {x_reconstructed}")
    print(f"Reconstructed dtype: {x_reconstructed.dtype}")
    
    # Check error
    error = jnp.max(jnp.abs(x_test - x_reconstructed))
    print(f"Forward-inverse error: {error:.2e}")
    
    # Test with log det
    y_logdet, log_det_fwd = wrapped_bijector.forward_and_log_det(x_test)
    x_logdet, log_det_inv = wrapped_bijector.inverse_and_log_det(y_logdet)
    
    print(f"Log det forward: {log_det_fwd}")
    print(f"Log det forward dtype: {log_det_fwd.dtype}")
    
    logdet_error = jnp.max(jnp.abs(x_test - x_logdet))
    consistency = jnp.max(jnp.abs(log_det_fwd + log_det_inv))
    
    print(f"Log det error: {logdet_error:.2e}")
    print(f"Log det consistency: {consistency:.2e}")
    
    # Check all dtypes are float64
    assert x_test.dtype == jnp.float64
    assert y.dtype == jnp.float64
    assert x_reconstructed.dtype == jnp.float64
    assert log_det_fwd.dtype == jnp.float64
    
    print("✓ All dtypes are float64!")
    print(f"✓ Forward-inverse error: {error:.2e}")
    
    return wrapped_bijector


if __name__ == "__main__":
    bijector = test_float64_tfp()
    print("✓ Simple TFP float64 wrapper works!")

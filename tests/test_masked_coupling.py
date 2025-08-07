import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
import distrax


class MaskedCouplingLayer(eqx.Module):
    mask: Float[Array, "input_dim"]
    conditioner: eqx.nn.MLP
    scale_network: eqx.nn.MLP
    input_dim: int
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, hidden_dim: int = 64, *, key: Array):
        key_cond, key_scale = jax.random.split(key)
        
        self.input_dim = input_dim
        split_dim = input_dim // 2
        self.mask = jnp.array([1.0] * split_dim + [0.0] * (input_dim - split_dim))
        
        self.conditioner = eqx.nn.MLP(
            split_dim, input_dim - split_dim, 
            hidden_dim, depth=3, key=key_cond
        )
        self.scale_network = eqx.nn.MLP(
            split_dim, input_dim - split_dim,
            hidden_dim, depth=3, key=key_scale
        )

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        print(f"forward: x shape = {x.shape}, expected input_dim = {self.input_dim}")
        assert x.shape[-1] == self.input_dim

        # Save original shape for debugging
        original_shape = x.shape

        # Reshape to ensure we're working with a batch
        if x.ndim == 1:
            print("  forward: Input is a single example (ndim=1), reshaping to (1, input_dim)")
            x_batch = x[None, :]  # Add batch dimension
        else:
            print("  forward: Input is already a batch")
            x_batch = x

        print(f"  forward: x_batch shape = {x_batch.shape}")

        split_dim = self.input_dim // 2
        print(f"  forward: split_dim = {split_dim}")

        # Extract the masked part
        x1 = x_batch[..., :split_dim]
        print(f"  forward: x1 shape = {x1.shape} (should be batch_shape + ({split_dim},))")

        # Check what we're passing to the conditioner
        print(f"  forward: First element of x1 = {x1[0]}")
        print(f"  forward: Shape of first element of x1 = {x1[0].shape}")

        # Get transformations for the non-masked part
        shift = eqx.filter_vmap(self.conditioner)(x1)
        print(f"  forward: shift shape = {shift.shape} (should be batch_shape + ({self.input_dim - split_dim},))")

        log_scale = eqx.filter_vmap(self.scale_network)(x1)
        print(f"  forward: log_scale shape = {log_scale.shape}")

        scale = jnp.exp(jnp.clip(log_scale, -5, 3))
        print(f"  forward: scale shape = {scale.shape}")

        # Transform only the non-masked part
        x2 = x_batch[..., split_dim:]
        print(f"  forward: x2 shape = {x2.shape}")

        y2 = x2 * scale + shift
        print(f"  forward: y2 shape = {y2.shape}")

        # Combine results
        y = jnp.concatenate([x1, y2], axis=-1)
        print(f"  forward: y shape = {y.shape} (should match x_batch shape)")

        # Log determinant is sum of log(scale) for transformed dimensions
        logdet = jnp.sum(log_scale, axis=-1)
        print(f"  forward: logdet shape = {logdet.shape} (should be batch_shape)")

        # Reshape back if needed
        if original_shape != y.shape:
            y = y.reshape(original_shape)
            logdet = logdet.reshape(original_shape[:-1])
            print(f"  forward: Reshaped y to {y.shape}, logdet to {logdet.shape}")

        assert y.shape == original_shape
        assert logdet.shape == original_shape[:-1]
        print("  forward: ✓ Assertions passed\n")
        return y, logdet

    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        print(f"inverse: y shape = {y.shape}, expected input_dim = {self.input_dim}")
        assert y.shape[-1] == self.input_dim

        # Save original shape for debugging
        original_shape = y.shape

        # Reshape to ensure we're working with a batch
        if y.ndim == 1:
            print("  inverse: Input is a single example (ndim=1), reshaping to (1, input_dim)")
            y_batch = y[None, :]  # Add batch dimension
        else:
            print("  inverse: Input is already a batch")
            y_batch = y

        print(f"  inverse: y_batch shape = {y_batch.shape}")

        split_dim = self.input_dim // 2
        print(f"  inverse: split_dim = {split_dim}")

        # Extract the masked part
        y1 = y_batch[..., :split_dim]
        print(f"  inverse: y1 shape = {y1.shape} (should be batch_shape + ({split_dim},))")

        # Check what we're passing to the conditioner
        print(f"  inverse: First element of y1 = {y1[0]}")
        print(f"  inverse: Shape of first element of y1 = {y1[0].shape}")

        # Get transformations for the non-masked part
        shift = eqx.filter_vmap(self.conditioner)(y1)
        print(f"  inverse: shift shape = {shift.shape} (should be batch_shape + ({self.input_dim - split_dim},))")

        log_scale = eqx.filter_vmap(self.scale_network)(y1)
        print(f"  inverse: log_scale shape = {log_scale.shape}")

        scale = jnp.exp(jnp.clip(log_scale, -5, 3))
        print(f"  inverse: scale shape = {scale.shape}")

        # Inverse transform only the non-masked part
        y2 = y_batch[..., split_dim:]
        print(f"  inverse: y2 shape = {y2.shape}")

        x2 = (y2 - shift) / scale
        print(f"  inverse: x2 shape = {x2.shape}")

        # Combine results
        x = jnp.concatenate([y1, x2], axis=-1)
        print(f"  inverse: x shape = {x.shape} (should match y_batch shape)")

        # Log determinant for inverse is negative sum of log(scale)
        logdet = -jnp.sum(log_scale, axis=-1)
        print(f"  inverse: logdet shape = {logdet.shape} (should be batch_shape)")

        # Reshape back if needed
        if original_shape != x.shape:
            x = x.reshape(original_shape)
            logdet = logdet.reshape(original_shape[:-1])
            print(f"  inverse: Reshaped x to {x.shape}, logdet to {logdet.shape}")

        assert x.shape == original_shape
        assert logdet.shape == original_shape[:-1]
        print("  inverse: ✓ Assertions passed\n")
        return x, logdet

def test_invertibility():
    key = jax.random.key(42)
    
    for input_dim in [2, 3, 40]:
        key, subkey = jax.random.split(key)
        layer = MaskedCouplingLayer(input_dim, key=subkey)
        
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (10, input_dim))
        
        y, fwd_logdet = layer.forward(x)
        x_rec, inv_logdet = layer.inverse(y)
        
        assert jnp.allclose(x_rec, x, atol=1e-5)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-5)


def test_distrax_affine():
    key = jax.random.key(42)
    key_x, key_mask, key_params = jax.random.split(key, 3)
    
    input_dim = 4
    x = jax.random.normal(key_x, (5, input_dim))
    mask = jnp.array([True, True, False, False])
    
    shift_params = jax.random.normal(key_params, (input_dim,))
    scale_params = jnp.exp(jax.random.normal(key_params, (input_dim,)) * 0.1)
    
    distrax_layer = distrax.MaskedCoupling(
        mask=mask,
        conditioner=lambda _: (shift_params * ~mask, jnp.log(scale_params * ~mask)),
        bijector=lambda p: distrax.ScalarAffine(shift=p[0], log_scale=p[1])
    )
    
    y_distrax, logdet_distrax = distrax_layer.forward_and_log_det(x)
    
    y_manual = x * mask + (x * scale_params + shift_params) * ~mask
    logdet_manual = jnp.sum(jnp.log(scale_params) * ~mask, axis=-1)
    
    assert jnp.allclose(y_distrax, y_manual, atol=1e-6)
    assert jnp.allclose(logdet_distrax, logdet_manual, atol=1e-6)


def test_symbolic_check():
    input_dim = 2
    key = jax.random.key(0)
    layer = MaskedCouplingLayer(input_dim, hidden_dim=8, key=key)
    
    epsilon = 1e-7
    x = jnp.array([[0.5, 0.5]])
    
    y, analytical_logdet = layer.forward(x)
    
    def compute_numerical_jacobian(x_point):
        jacobian = jnp.zeros((input_dim, input_dim))
        for i in range(input_dim):
            x_plus = x_point.at[0, i].set(x_point[0, i] + epsilon)
            x_minus = x_point.at[0, i].set(x_point[0, i] - epsilon)
            y_plus = layer.forward(x_plus)[0]
            y_minus = layer.forward(x_minus)[0]
            jacobian = jacobian.at[:, i].set((y_plus[0] - y_minus[0]) / (2 * epsilon))
        return jacobian
    
    numerical_jac = compute_numerical_jacobian(x)
    numerical_logdet = jnp.log(jnp.abs(jnp.linalg.det(numerical_jac)))
    
    assert jnp.allclose(analytical_logdet[0], numerical_logdet, atol=1e-4)


def test_training_loop_compatibility():
    from diengmf.dynamical_systems import Ikeda
    from diengmf.losses import make_step
    import optax
    
    key = jax.random.key(0)
    model = MaskedCouplingLayer(input_dim=2, hidden_dim=32, key=key)
    
    system = Ikeda(batch_size=25)
    batch = system.generate(jax.random.key(0), batch_size=100)
    
    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    for _ in range(10_000):
        batch = eqx.filter_vmap(system.forward)(batch)
        loss, model, opt_state = make_step(model, batch, optim, opt_state)

        if (_ % 500) == 0:
            print(loss)
    
    assert isinstance(loss, jnp.ndarray)


def test_mask_preservation():
    key = jax.random.key(42)
    layer = MaskedCouplingLayer(input_dim=4, key=key)
    
    x = jax.random.normal(key, (10, 4))
    y, _ = layer.forward(x)
    
    mask_indices = jnp.where(layer.mask == 1.0)[0]
    assert jnp.allclose(x[:, mask_indices], y[:, mask_indices])
    
    x_rec, _ = layer.inverse(y)
    assert jnp.allclose(x[:, mask_indices], x_rec[:, mask_indices])


def test_batched_operations():
    key = jax.random.key(42)
    
    for batch_size in [1, 10, 100]:
        for input_dim in [2, 3, 40]:
            key, subkey = jax.random.split(key)
            layer = MaskedCouplingLayer(input_dim, key=subkey)
            
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (batch_size, input_dim))
            
            y, fwd_logdet = layer.forward(x)
            x_rec, inv_logdet = layer.inverse(y)
            
            assert y.shape == (batch_size, input_dim)
            assert fwd_logdet.shape == (batch_size,)
            assert jnp.allclose(x_rec, x, atol=1e-5)


test_invertibility()
test_distrax_affine()
test_symbolic_check()
test_mask_preservation()
test_batched_operations()
test_training_loop_compatibility()

print("✓ All masked coupling layer tests pass!")

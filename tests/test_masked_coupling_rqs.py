import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Literal


class RQSTransform(eqx.Module):
    """Single-use RQS transform with given parameters."""
    params: Float[Array, "spline_param_dim"]
    range_min: float
    range_max: float
    min_bin_size: float
    min_slope: float
    
    def __init__(self, params, range_min=-5.0, range_max=5.0):
        self.params = params
        self.range_min = range_min
        self.range_max = range_max
        self.min_bin_size = 1e-4
        self.min_slope = 1e-4
    
    def _get_spline_params(self):
        K = (self.params.shape[-1] - 1) // 3
        
        from test_rqs import _normalize_bins, _normalize_slopes
        
        widths = _normalize_bins(self.params[:K], 
                                self.range_max - self.range_min, self.min_bin_size)
        heights = _normalize_bins(self.params[K:2*K],
                                self.range_max - self.range_min, self.min_bin_size)
        slopes = _normalize_slopes(self.params[2*K:], self.min_slope)
        
        x_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(widths)]) + self.range_min
        y_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(heights)]) + self.range_min
        
        return x_pos, y_pos, slopes
    
    def forward(self, x):
        from test_rqs import _spline_forward
        x_pos, y_pos, slopes = self._get_spline_params()
        return _spline_forward(x[0], x_pos, y_pos, slopes, self.range_min, self.range_max)
    
    def inverse(self, y):
        from test_rqs import _spline_inverse
        x_pos, y_pos, slopes = self._get_spline_params()
        return _spline_inverse(y[0], x_pos, y_pos, slopes, self.range_min, self.range_max)


class MaskedCouplingRQS(eqx.Module):
    """Masked coupling layer using Rational Quadratic Splines."""
    mask: Float[Array, "input_dim"]
    conditioner: eqx.nn.MLP
    num_bins: int
    input_dim: int
    range_min: float
    range_max: float
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, 
                 input_dim: int,
                 num_bins: int = 8,
                 hidden_dim: int = 128,
                 mask_type: Literal["half", "alternating", "random"] = "half",
                 *, key: Array,
                 range_min: float = -5.0,
                 range_max: float = 5.0):
        key_mask, key_mlp = jax.random.split(key)
        
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        
        if mask_type == "half":
            split_dim = input_dim // 2
            self.mask = jnp.array([1.0] * split_dim + [0.0] * (input_dim - split_dim))
        elif mask_type == "alternating":
            self.mask = jnp.array([float(i % 2) for i in range(input_dim)])
        else:
            self.mask = jax.random.bernoulli(key_mask, 0.5, (input_dim,)).astype(float)
        
        mask_dim = int(self.mask.sum())
        transform_dim = input_dim - mask_dim
        spline_params = 3 * num_bins + 1
        
        print(f"MaskedCouplingRQS.__init__: input_dim={input_dim}, mask_dim={mask_dim}, transform_dim={transform_dim}")
        print(f"  Conditioner: {mask_dim} -> {transform_dim * spline_params}")
        
        self.conditioner = eqx.nn.MLP(
            in_size=mask_dim,
            out_size=transform_dim * spline_params,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.relu,
            key=key_mlp
        )
    
    @jaxtyped(typechecker=typechecker)
    def _apply_spline(self, x_transform, params):
        """Apply spline transformation to each dimension."""
        batch_size = x_transform.shape[0]
        transform_dim = x_transform.shape[1]
        
        print(f"  _apply_spline: x_transform.shape={x_transform.shape}, params.shape={params.shape}")
        params = params.reshape(batch_size, transform_dim, -1)
        print(f"    Reshaped params: {params.shape} (batch_size={batch_size}, transform_dim={transform_dim}, spline_params)")
        
        y_list = []
        logdet_list = []
        
        for i in range(transform_dim):
            def apply_to_batch(x_val, param_val):
                spline = RQSTransform(param_val, self.range_min, self.range_max)
                return spline.forward(x_val[None])
            
            y_vals, logdets = jax.vmap(apply_to_batch)(
                x_transform[:, i], params[:, i, :]
            )
            
            y_list.append(y_vals)
            logdet_list.append(logdets)
        
        y_transform = jnp.stack(y_list, axis=-1)
        logdet = jnp.sum(jnp.stack(logdet_list, axis=-1), axis=-1)
        
        print(f"    Output: y_transform.shape={y_transform.shape}, logdet.shape={logdet.shape}")
        return y_transform, logdet
    
    @jaxtyped(typechecker=typechecker)
    def _apply_inverse_spline(self, y_transform, params):
        """Apply inverse spline transformation."""
        batch_size = y_transform.shape[0]
        transform_dim = y_transform.shape[1]
        
        params = params.reshape(batch_size, transform_dim, -1)
        
        x_list = []
        logdet_list = []
        
        for i in range(transform_dim):
            def apply_to_batch(y_val, param_val):
                spline = RQSTransform(param_val, self.range_min, self.range_max)
                return spline.inverse(y_val[None])
            
            x_vals, logdets = jax.vmap(apply_to_batch)(
                y_transform[:, i], params[:, i, :]
            )
            
            x_list.append(x_vals)
            logdet_list.append(logdets)
        
        x_transform = jnp.stack(x_list, axis=-1)
        logdet = jnp.sum(jnp.stack(logdet_list, axis=-1), axis=-1)
        
        return x_transform, logdet
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        print(f"\nMaskedCouplingRQS.forward: x.shape={x.shape}")
        assert x.shape[-1] == self.input_dim
        batch_shape = x.shape[:-1]
        original_shape = x.shape
        
        if x.ndim == 1:
            x = x[None, :]
            print(f"  Added batch dim: x.shape={x.shape}")
        
        x = x.reshape(-1, self.input_dim)
        batch_size = x.shape[0]
        print(f"  Reshaped to batch: x.shape={x.shape}, batch_size={batch_size}")
        
        mask_idx = jnp.where(self.mask)[0]
        transform_idx = jnp.where(1 - self.mask)[0]
        print(f"  mask_idx={mask_idx}, transform_idx={transform_idx}")
        
        x_masked = x[:, mask_idx]
        x_transform = x[:, transform_idx]
        print(f"  x_masked.shape={x_masked.shape} (extracting indices {mask_idx})")
        print(f"  x_transform.shape={x_transform.shape} (extracting indices {transform_idx})")
        
        print(f"  Applying conditioner: {x_masked.shape} -> MLP -> params")
        params = eqx.filter_vmap(self.conditioner)(x_masked)
        print(f"  params.shape={params.shape} (from conditioner)")
        
        y_transform, logdet = self._apply_spline(x_transform, params)
        
        y = jnp.zeros_like(x)
        print(f"  Creating output: y.shape={y.shape}")
        y = y.at[:, mask_idx].set(x_masked)
        print(f"  Set masked dims: y[:, {mask_idx}] = x_masked")
        y = y.at[:, transform_idx].set(y_transform)
        print(f"  Set transform dims: y[:, {transform_idx}] = y_transform")
        
        y = y.reshape(original_shape)
        logdet = logdet.reshape(batch_shape)
        print(f"  Final: y.shape={y.shape}, logdet.shape={logdet.shape}")
        
        assert y.shape == original_shape
        assert logdet.shape == batch_shape
        return y, logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        assert y.shape[-1] == self.input_dim
        batch_shape = y.shape[:-1]
        original_shape = y.shape
        
        if y.ndim == 1:
            y = y[None, :]
        
        y = y.reshape(-1, self.input_dim)
        
        mask_idx = jnp.where(self.mask)[0]
        transform_idx = jnp.where(1 - self.mask)[0]
        
        y_masked = y[:, mask_idx]
        y_transform = y[:, transform_idx]
        
        params = eqx.filter_vmap(self.conditioner)(y_masked)
        x_transform, logdet = self._apply_inverse_spline(y_transform, params)
        
        x = jnp.zeros_like(y)
        x = x.at[:, mask_idx].set(y_masked)
        x = x.at[:, transform_idx].set(x_transform)
        
        x = x.reshape(original_shape)
        logdet = logdet.reshape(batch_shape)
        
        assert x.shape == original_shape
        assert logdet.shape == batch_shape
        return x, logdet


class MaskedCouplingAffine(eqx.Module):
    """Simple affine coupling for comparison."""
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
        
        mask_dim = int(self.mask.sum())
        transform_dim = input_dim - mask_dim
        
        print(f"MaskedCouplingAffine.__init__: input_dim={input_dim}, mask_dim={mask_dim}, transform_dim={transform_dim}")
        print(f"  Conditioner MLP: {mask_dim} -> {transform_dim}")
        print(f"  Scale MLP: {mask_dim} -> {transform_dim}")
        
        self.conditioner = eqx.nn.MLP(
            mask_dim, transform_dim, 
            hidden_dim, depth=3, key=key_cond
        )
        self.scale_network = eqx.nn.MLP(
            mask_dim, transform_dim,
            hidden_dim, depth=3, key=key_scale
        )
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        print(f"\nMaskedCouplingAffine.forward: x.shape={x.shape}")
        assert x.shape[-1] == self.input_dim
        batch_shape = x.shape[:-1]
        original_shape = x.shape
        
        if x.ndim == 1:
            x = x[None, :]
            print(f"  Added batch dim: x.shape={x.shape}")
        
        print(f"  mask={self.mask}")
        mask_idx = jnp.where(self.mask)[0]
        transform_idx = jnp.where(1 - self.mask)[0]
        print(f"  mask_idx={mask_idx} (dims to keep fixed)")
        print(f"  transform_idx={transform_idx} (dims to transform)")
        
        # Extract only the masked dimensions for conditioning
        x_masked = x[..., mask_idx]
        x_transform = x[..., transform_idx]
        print(f"  x_masked = x[..., {mask_idx}], shape={x_masked.shape}")
        print(f"  x_transform = x[..., {transform_idx}], shape={x_transform.shape}")
        
        # Pass masked dims through MLPs to get params for transform dims
        print(f"  Applying conditioner MLP: {x_masked.shape} -> MLP")
        shift = eqx.filter_vmap(self.conditioner)(x_masked)
        print(f"  shift.shape={shift.shape} (parameters for {len(transform_idx)} transform dims)")
        
        print(f"  Applying scale MLP: {x_masked.shape} -> MLP")
        log_scale = eqx.filter_vmap(self.scale_network)(x_masked)
        print(f"  log_scale.shape={log_scale.shape}")
        
        scale = jnp.exp(jnp.clip(log_scale, -5, 3))
        print(f"  scale = exp(clip(log_scale, -5, 3)), shape={scale.shape}")
        
        # Transform only the non-masked dimensions
        y_transform = x_transform * scale + shift
        print(f"  y_transform = x_transform * scale + shift")
        print(f"  y_transform.shape={y_transform.shape}")
        
        # Combine: keep masked dims, replace transform dims
        y = jnp.zeros_like(x)
        y = y.at[..., mask_idx].set(x_masked)
        y = y.at[..., transform_idx].set(y_transform)
        print(f"  y[..., {mask_idx}] = x_masked (unchanged)")
        print(f"  y[..., {transform_idx}] = y_transform (transformed)")
        print(f"  y.shape={y.shape}")
        
        # Log determinant is sum of log scales for transformed dimensions
        logdet = jnp.sum(log_scale, axis=-1)
        print(f"  logdet = sum(log_scale) over {len(transform_idx)} dims")
        print(f"  logdet.shape={logdet.shape}")
        
        # Reshape back if we added batch dim
        if original_shape != y.shape:
            y = y.reshape(original_shape)
            logdet = logdet.reshape(batch_shape)
            print(f"  Reshaped to original: y.shape={y.shape}, logdet.shape={logdet.shape}")
        
        assert y.shape == original_shape
        assert logdet.shape == batch_shape
        return y, logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        print(f"\nMaskedCouplingAffine.inverse: y.shape={y.shape}")
        assert y.shape[-1] == self.input_dim
        batch_shape = y.shape[:-1]
        original_shape = y.shape
        
        if y.ndim == 1:
            y = y[None, :]
            print(f"  Added batch dim: y.shape={y.shape}")
        
        mask_idx = jnp.where(self.mask)[0]
        transform_idx = jnp.where(1 - self.mask)[0]
        print(f"  mask_idx={mask_idx}, transform_idx={transform_idx}")
        
        # Extract dimensions
        y_masked = y[..., mask_idx]
        y_transform = y[..., transform_idx]
        print(f"  y_masked.shape={y_masked.shape}, y_transform.shape={y_transform.shape}")
        
        # Get transformation parameters from masked dims
        print(f"  Applying conditioner: {y_masked.shape} -> MLP")
        shift = eqx.filter_vmap(self.conditioner)(y_masked)
        print(f"  shift.shape={shift.shape}")
        
        print(f"  Applying scale network: {y_masked.shape} -> MLP")
        log_scale = eqx.filter_vmap(self.scale_network)(y_masked)
        print(f"  log_scale.shape={log_scale.shape}")
        
        scale = jnp.exp(jnp.clip(log_scale, -5, 3))
        print(f"  scale.shape={scale.shape}")
        
        # Inverse transform
        x_transform = (y_transform - shift) / scale
        print(f"  x_transform = (y_transform - shift) / scale")
        print(f"  x_transform.shape={x_transform.shape}")
        
        # Combine
        x = jnp.zeros_like(y)
        x = x.at[..., mask_idx].set(y_masked)
        x = x.at[..., transform_idx].set(x_transform)
        print(f"  x.shape={x.shape}")
        
        # Negative log det for inverse
        logdet = -jnp.sum(log_scale, axis=-1)
        print(f"  logdet = -sum(log_scale), shape={logdet.shape}")
        
        if original_shape != x.shape:
            x = x.reshape(original_shape)
            logdet = logdet.reshape(batch_shape)
            print(f"  Reshaped: x.shape={x.shape}, logdet.shape={logdet.shape}")
        
        assert x.shape == original_shape
        assert logdet.shape == batch_shape
        return x, logdet


def test_rqs_coupling():
    key = jax.random.key(42)
    
    for input_dim in [2, 3, 40]:
        print(f"\n{'='*50}")
        print(f"Testing RQS coupling with input_dim={input_dim}")
        print(f"{'='*50}")
        
        key, subkey = jax.random.split(key)
        layer = MaskedCouplingRQS(input_dim, num_bins=4, key=subkey)
        
        x = jax.random.normal(subkey, (5, input_dim))
        y, fwd_logdet = layer.forward(x)
        x_rec, inv_logdet = layer.inverse(y)
        
        assert jnp.allclose(x_rec, x, atol=1e-4)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-4)
        print(f"✓ RQS test passed for dim={input_dim}")


def test_affine_coupling():
    key = jax.random.key(42)
    
    for input_dim in [2, 3, 40]:
        print(f"\n{'='*50}")
        print(f"Testing Affine coupling with input_dim={input_dim}")
        print(f"{'='*50}")
        
        key, subkey = jax.random.split(key)
        layer = MaskedCouplingAffine(input_dim, key=subkey)
        
        x = jax.random.normal(subkey, (10, input_dim))
        y, fwd_logdet = layer.forward(x)
        x_rec, inv_logdet = layer.inverse(y)
        
        assert jnp.allclose(x_rec, x, atol=1e-5)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-5)
        print(f"✓ Affine test passed for dim={input_dim}")


test_rqs_coupling()
test_affine_coupling()
print("\n✓ Both RQS and Affine coupling tests pass!")

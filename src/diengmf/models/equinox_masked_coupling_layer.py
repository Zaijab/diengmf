import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Literal

def _normalize_bins(unnormalized_sizes: Float[Array, "K"], 
                   total_size: float, min_size: float) -> Float[Array, "K"]:
    normalized = jax.nn.softmax(unnormalized_sizes, axis=-1)
    num_bins = unnormalized_sizes.shape[-1]
    return normalized * (total_size - num_bins * min_size) + min_size


def _normalize_slopes(unnormalized_slopes: Float[Array, "K1"], 
                     min_slope: float) -> Float[Array, "K1"]:
    offset = jnp.log(jnp.exp(1.0 - min_slope) - 1.0)
    return jax.nn.softplus(unnormalized_slopes + offset) + min_slope


# @jaxtyped(typechecker=typechecker)
def _quadratic_solve(a: Array, b: Array, c: Array) -> Array:
    discriminant = b*b - 4*a*c
    safe_discriminant = jnp.maximum(discriminant, 0.0)
    sqrt_discriminant = jnp.sqrt(safe_discriminant)
    
    return jnp.where(
        b >= 0, 
        -2*c / (b + sqrt_discriminant + 1e-8),
        (-b + sqrt_discriminant) / (2*a + 1e-8)
    )


# @jaxtyped(typechecker=typechecker)
def _spline_forward(x: Array, x_pos: Array, y_pos: Array, 
                   slopes: Array, range_min: float, range_max: float) -> tuple[Array, Array]:
    num_bins = x_pos.shape[-1] - 1
    
    # Spline computation for inside range
    bin_idx = jnp.searchsorted(x_pos[1:-1], x, side='right')
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
    
    x_k = x_pos[bin_idx]
    x_k1 = x_pos[bin_idx + 1]
    y_k = y_pos[bin_idx]
    y_k1 = y_pos[bin_idx + 1]
    slope_k = slopes[bin_idx]
    slope_k1 = slopes[bin_idx + 1]
    
    width = x_k1 - x_k
    xi = jnp.clip((x - x_k) / (width + 1e-8), 0.0, 1.0)
    height = y_k1 - y_k
    s = height / (width + 1e-8)
    
    numerator = s * xi**2 + slope_k * xi * (1 - xi)
    denominator = s + (slope_k1 + slope_k - 2*s) * xi * (1 - xi)
    denominator = jnp.maximum(jnp.abs(denominator), 1e-8) * jnp.sign(denominator)
    
    y_spline = y_k + height * (numerator / denominator)
    
    xi_1mxi = xi * (1 - xi)
    derivative_num = s**2 * (slope_k1 * xi**2 + 2*s * xi_1mxi + slope_k * (1-xi)**2)
    derivative = derivative_num / (denominator**2)
    derivative = jnp.maximum(derivative, 1e-8)
    logdet_spline = jnp.log(derivative)
    
    # Linear extrapolation for outside range
    slope_left = slopes[0]  
    slope_right = slopes[-1]
    y_linear_left = (x - range_min) * slope_left + range_min
    y_linear_right = (x - range_max) * slope_right + range_max
    logdet_linear_left = jnp.log(slope_left)
    logdet_linear_right = jnp.log(slope_right)
    
    below_range = x < range_min
    above_range = x > range_max
    
    y = jnp.where(below_range, y_linear_left,
                 jnp.where(above_range, y_linear_right, y_spline))
    logdet = jnp.where(below_range, logdet_linear_left,
                      jnp.where(above_range, logdet_linear_right, logdet_spline))
    
    return y, logdet


# @jaxtyped(typechecker=typechecker)
def _spline_inverse(y: Array, x_pos: Array, y_pos: Array, 
                   slopes: Array, range_min: float, range_max: float) -> tuple[Array, Array]:
    num_bins = y_pos.shape[-1] - 1
    
    # Spline computation for inside range
    bin_idx = jnp.searchsorted(y_pos[1:-1], y, side='right')
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
    
    x_k = x_pos[bin_idx]
    x_k1 = x_pos[bin_idx + 1]
    y_k = y_pos[bin_idx]
    y_k1 = y_pos[bin_idx + 1]
    slope_k = slopes[bin_idx]
    slope_k1 = slopes[bin_idx + 1]
    
    width = x_k1 - x_k
    height = y_k1 - y_k
    s = height / (width + 1e-8)
    y_rel = y - y_k
    
    a = height * (s - slope_k) + y_rel * (slope_k1 + slope_k - 2*s)
    b = height * slope_k - y_rel * (slope_k1 + slope_k - 2*s)
    c = -s * y_rel
    
    xi = _quadratic_solve(a, b, c)
    xi = jnp.clip(xi, 0.0, 1.0)
    x_spline = x_k + xi * width
    
    denominator = s + (slope_k1 + slope_k - 2*s) * xi * (1 - xi)
    denominator = jnp.maximum(jnp.abs(denominator), 1e-8) * jnp.sign(denominator)
    
    xi_1mxi = xi * (1 - xi)
    derivative_num = s**2 * (slope_k1 * xi**2 + 2*s * xi_1mxi + slope_k * (1-xi)**2)
    derivative = derivative_num / (denominator**2)
    derivative = jnp.maximum(derivative, 1e-8)
    
    logdet_spline = -jnp.log(derivative)
    
    # Linear extrapolation for outside range
    slope_left = slopes[0]
    slope_right = slopes[-1]
    x_linear_left = (y - range_min) / slope_left + range_min
    x_linear_right = (y - range_max) / slope_right + range_max
    logdet_linear_left = -jnp.log(slope_left)
    logdet_linear_right = -jnp.log(slope_right)
    
    below_range = y < range_min
    above_range = y > range_max
    
    x = jnp.where(below_range, x_linear_left,
                 jnp.where(above_range, x_linear_right, x_spline))
    logdet = jnp.where(below_range, logdet_linear_left,
                      jnp.where(above_range, logdet_linear_right, logdet_spline))
    
    return x, logdet


class RationalQuadraticSpline(eqx.Module):
    params: Float[Array, "input_dim spline_param_dim"] 
    input_dim: int
    range_min: float
    range_max: float
    min_bin_size: float
    min_slope: float
    
    # @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, num_bins: int = 8, *, key: Array,
                 range_min: float = -3.0, range_max: float = 3.0,
                 min_bin_size: float = 1e-4, min_slope: float = 1e-4):
        spline_param_dim = 3 * num_bins + 1
        assert spline_param_dim >= 4
        
        self.params = jax.random.normal(key, (input_dim, spline_param_dim)) * 0.1
        self.input_dim = input_dim
        self.range_min = range_min
        self.range_max = range_max
        self.min_bin_size = min_bin_size  
        self.min_slope = min_slope
    
    # @jaxtyped(typechecker=typechecker)
    def _get_spline_params_per_dim(self, dim_idx: int) -> tuple[Array, Array, Array]:
        """Extract spline parameters for a specific dimension."""
        K = (self.params.shape[-1] - 1) // 3
        dim_params = self.params[dim_idx]
        
        widths = _normalize_bins(dim_params[:K], 
                               self.range_max - self.range_min, self.min_bin_size)
        heights = _normalize_bins(dim_params[K:2*K],
                                self.range_max - self.range_min, self.min_bin_size)
        slopes = _normalize_slopes(dim_params[2*K:], self.min_slope)
        
        x_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(widths)]) + self.range_min
        y_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(heights)]) + self.range_min
        
        return x_pos, y_pos, slopes
    
    # @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        """Forward transformation matching training script interface.""" 
        assert x.shape[-1] == self.input_dim
        batch_shape = x.shape[:-1]
        
        y_dims = []
        logdet_dims = []
        
        for i in range(self.input_dim):
            x_pos, y_pos, slopes = self._get_spline_params_per_dim(i)
            x_i = x[..., i]
            
            fn = jnp.vectorize(_spline_forward, 
                              signature='(),(n),(n),(n),(),()->(),()')
            y_i, logdet_i = fn(x_i, x_pos, y_pos, slopes, 
                              self.range_min, self.range_max)
            
            y_dims.append(y_i)
            logdet_dims.append(logdet_i)
        
        y = jnp.stack(y_dims, axis=-1)
        total_logdet = jnp.sum(jnp.stack(logdet_dims, axis=-1), axis=-1)
        
        assert y.shape == x.shape
        assert total_logdet.shape == batch_shape
        
        return y, total_logdet
    
    # @jaxtyped(typechecker=typechecker)  
    def inverse(self, y: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        """Inverse transformation matching training script interface."""
        assert y.shape[-1] == self.input_dim
        batch_shape = y.shape[:-1]
        
        x_dims = []
        logdet_dims = []
        
        for i in range(self.input_dim):
            x_pos, y_pos, slopes = self._get_spline_params_per_dim(i)
            y_i = y[..., i]
            
            fn = jnp.vectorize(_spline_inverse, 
                              signature='(),(n),(n),(n),(),()->(),()')
            x_i, logdet_i = fn(y_i, x_pos, y_pos, slopes,
                              self.range_min, self.range_max)
            
            x_dims.append(x_i)
            logdet_dims.append(logdet_i)
        
        x = jnp.stack(x_dims, axis=-1)
        total_logdet = jnp.sum(jnp.stack(logdet_dims, axis=-1), axis=-1)
        
        assert x.shape == y.shape
        assert total_logdet.shape == batch_shape
        
        return x, total_logdet


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
        # Add to __init__ method after mask creation:
        mask_indices = jnp.arange(self.input_dim)
        self.mask_idx = mask_indices[self.mask.astype(bool)]
        self.transform_idx = mask_indices[~self.mask.astype(bool)]

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
    

    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        assert y.shape[-1] == self.input_dim
        batch_shape = y.shape[:-1]
        original_shape = y.shape

        if y.ndim == 1:
            y = y[None, :]

        y = y.reshape(-1, self.input_dim)

        y_masked = y[:, self.mask_idx]
        y_transform = y[:, self.transform_idx]

        params = eqx.filter_vmap(self.conditioner)(y_masked)
        x_transform, logdet = self._apply_inverse_spline(y_transform, params)

        x = jnp.zeros_like(y)
        x = x.at[:, self.mask_idx].set(y_masked)
        x = x.at[:, self.transform_idx].set(x_transform)

        assert x.shape == y.shape
        return x.reshape(original_shape), logdet.reshape(batch_shape)

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


import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker


def _normalize_bins(unnormalized_sizes: Float[Array, "K"], 
                   total_size: float, min_size: float) -> Float[Array, "K"]:
    normalized = jax.nn.softmax(unnormalized_sizes, axis=-1)
    num_bins = unnormalized_sizes.shape[-1]
    return normalized * (total_size - num_bins * min_size) + min_size


def _normalize_slopes(unnormalized_slopes: Float[Array, "K1"], 
                     min_slope: float) -> Float[Array, "K1"]:
    offset = jnp.log(jnp.exp(1.0 - min_slope) - 1.0)
    return jax.nn.softplus(unnormalized_slopes + offset) + min_slope


# @jaxtyped(typechecker=typechecker)
def _quadratic_solve(a: Array, b: Array, c: Array) -> Array:
    discriminant = b*b - 4*a*c
    safe_discriminant = jnp.maximum(discriminant, 0.0)
    sqrt_discriminant = jnp.sqrt(safe_discriminant)
    
    return jnp.where(
        b >= 0, 
        -2*c / (b + sqrt_discriminant + 1e-8),
        (-b + sqrt_discriminant) / (2*a + 1e-8)
    )


# @jaxtyped(typechecker=typechecker)
def _spline_forward(x: Array, x_pos: Array, y_pos: Array, 
                   slopes: Array, range_min: float, range_max: float) -> tuple[Array, Array]:
    num_bins = x_pos.shape[-1] - 1
    
    # Spline computation for inside range
    bin_idx = jnp.searchsorted(x_pos[1:-1], x, side='right')
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
    
    x_k = x_pos[bin_idx]
    x_k1 = x_pos[bin_idx + 1]
    y_k = y_pos[bin_idx]
    y_k1 = y_pos[bin_idx + 1]
    slope_k = slopes[bin_idx]
    slope_k1 = slopes[bin_idx + 1]
    
    width = x_k1 - x_k
    xi = jnp.clip((x - x_k) / (width + 1e-8), 0.0, 1.0)
    height = y_k1 - y_k
    s = height / (width + 1e-8)
    
    numerator = s * xi**2 + slope_k * xi * (1 - xi)
    denominator = s + (slope_k1 + slope_k - 2*s) * xi * (1 - xi)
    denominator = jnp.maximum(jnp.abs(denominator), 1e-8) * jnp.sign(denominator)
    
    y_spline = y_k + height * (numerator / denominator)
    
    xi_1mxi = xi * (1 - xi)
    derivative_num = s**2 * (slope_k1 * xi**2 + 2*s * xi_1mxi + slope_k * (1-xi)**2)
    derivative = derivative_num / (denominator**2)
    derivative = jnp.maximum(derivative, 1e-8)
    logdet_spline = jnp.log(derivative)
    
    # Linear extrapolation for outside range
    slope_left = slopes[0]  
    slope_right = slopes[-1]
    y_linear_left = (x - range_min) * slope_left + range_min
    y_linear_right = (x - range_max) * slope_right + range_max
    logdet_linear_left = jnp.log(slope_left)
    logdet_linear_right = jnp.log(slope_right)
    
    below_range = x < range_min
    above_range = x > range_max
    
    y = jnp.where(below_range, y_linear_left,
                 jnp.where(above_range, y_linear_right, y_spline))
    logdet = jnp.where(below_range, logdet_linear_left,
                      jnp.where(above_range, logdet_linear_right, logdet_spline))
    
    return y, logdet


# @jaxtyped(typechecker=typechecker)
def _spline_inverse(y: Array, x_pos: Array, y_pos: Array, 
                   slopes: Array, range_min: float, range_max: float) -> tuple[Array, Array]:
    num_bins = y_pos.shape[-1] - 1
    
    # Spline computation for inside range
    bin_idx = jnp.searchsorted(y_pos[1:-1], y, side='right')
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
    
    x_k = x_pos[bin_idx]
    x_k1 = x_pos[bin_idx + 1]
    y_k = y_pos[bin_idx]
    y_k1 = y_pos[bin_idx + 1]
    slope_k = slopes[bin_idx]
    slope_k1 = slopes[bin_idx + 1]
    
    width = x_k1 - x_k
    height = y_k1 - y_k
    s = height / (width + 1e-8)
    y_rel = y - y_k
    
    a = height * (s - slope_k) + y_rel * (slope_k1 + slope_k - 2*s)
    b = height * slope_k - y_rel * (slope_k1 + slope_k - 2*s)
    c = -s * y_rel
    
    xi = _quadratic_solve(a, b, c)
    xi = jnp.clip(xi, 0.0, 1.0)
    x_spline = x_k + xi * width
    
    denominator = s + (slope_k1 + slope_k - 2*s) * xi * (1 - xi)
    denominator = jnp.maximum(jnp.abs(denominator), 1e-8) * jnp.sign(denominator)
    
    xi_1mxi = xi * (1 - xi)
    derivative_num = s**2 * (slope_k1 * xi**2 + 2*s * xi_1mxi + slope_k * (1-xi)**2)
    derivative = derivative_num / (denominator**2)
    derivative = jnp.maximum(derivative, 1e-8)
    
    logdet_spline = -jnp.log(derivative)
    
    # Linear extrapolation for outside range
    slope_left = slopes[0]
    slope_right = slopes[-1]
    x_linear_left = (y - range_min) / slope_left + range_min
    x_linear_right = (y - range_max) / slope_right + range_max
    logdet_linear_left = -jnp.log(slope_left)
    logdet_linear_right = -jnp.log(slope_right)
    
    below_range = y < range_min
    above_range = y > range_max
    
    x = jnp.where(below_range, x_linear_left,
                 jnp.where(above_range, x_linear_right, x_spline))
    logdet = jnp.where(below_range, logdet_linear_left,
                      jnp.where(above_range, logdet_linear_right, logdet_spline))
    
    return x, logdet


class RationalQuadraticSpline(eqx.Module):
    params: Float[Array, "input_dim spline_param_dim"] 
    input_dim: int
    range_min: float
    range_max: float
    min_bin_size: float
    min_slope: float
    
    # @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, num_bins: int = 8, *, key: Array,
                 range_min: float = -3.0, range_max: float = 3.0,
                 min_bin_size: float = 1e-4, min_slope: float = 1e-4):
        spline_param_dim = 3 * num_bins + 1
        assert spline_param_dim >= 4
        
        self.params = jax.random.normal(key, (input_dim, spline_param_dim)) * 0.1
        self.input_dim = input_dim
        self.range_min = range_min
        self.range_max = range_max
        self.min_bin_size = min_bin_size  
        self.min_slope = min_slope
    
    # @jaxtyped(typechecker=typechecker)
    def _get_spline_params_per_dim(self, dim_idx: int) -> tuple[Array, Array, Array]:
        """Extract spline parameters for a specific dimension."""
        K = (self.params.shape[-1] - 1) // 3
        dim_params = self.params[dim_idx]
        
        widths = _normalize_bins(dim_params[:K], 
                               self.range_max - self.range_min, self.min_bin_size)
        heights = _normalize_bins(dim_params[K:2*K],
                                self.range_max - self.range_min, self.min_bin_size)
        slopes = _normalize_slopes(dim_params[2*K:], self.min_slope)
        
        x_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(widths)]) + self.range_min
        y_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(heights)]) + self.range_min
        
        return x_pos, y_pos, slopes
    
    # @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        """Forward transformation matching training script interface.""" 
        assert x.shape[-1] == self.input_dim
        batch_shape = x.shape[:-1]
        
        y_dims = []
        logdet_dims = []
        
        for i in range(self.input_dim):
            x_pos, y_pos, slopes = self._get_spline_params_per_dim(i)
            x_i = x[..., i]
            
            fn = jnp.vectorize(_spline_forward, 
                              signature='(),(n),(n),(n),(),()->(),()')
            y_i, logdet_i = fn(x_i, x_pos, y_pos, slopes, 
                              self.range_min, self.range_max)
            
            y_dims.append(y_i)
            logdet_dims.append(logdet_i)
        
        y = jnp.stack(y_dims, axis=-1)
        total_logdet = jnp.sum(jnp.stack(logdet_dims, axis=-1), axis=-1)
        
        assert y.shape == x.shape
        assert total_logdet.shape == batch_shape
        
        return y, total_logdet
    
    # @jaxtyped(typechecker=typechecker)  
    def inverse(self, y: Float[Array, "... input_dim"]) -> tuple[Array, Array]:
        """Inverse transformation matching training script interface."""
        assert y.shape[-1] == self.input_dim
        batch_shape = y.shape[:-1]
        
        x_dims = []
        logdet_dims = []
        
        for i in range(self.input_dim):
            x_pos, y_pos, slopes = self._get_spline_params_per_dim(i)
            y_i = y[..., i]
            
            fn = jnp.vectorize(_spline_inverse, 
                              signature='(),(n),(n),(n),(),()->(),()')
            x_i, logdet_i = fn(y_i, x_pos, y_pos, slopes,
                              self.range_min, self.range_max)
            
            x_dims.append(x_i)
            logdet_dims.append(logdet_i)
        
        x = jnp.stack(x_dims, axis=-1)
        total_logdet = jnp.sum(jnp.stack(logdet_dims, axis=-1), axis=-1)
        
        assert x.shape == y.shape
        assert total_logdet.shape == batch_shape
        
        return x, total_logdet


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
        
        
        
        widths = _normalize_bins(self.params[:K], 
                                self.range_max - self.range_min, self.min_bin_size)
        heights = _normalize_bins(self.params[K:2*K],
                                self.range_max - self.range_min, self.min_bin_size)
        slopes = _normalize_slopes(self.params[2*K:], self.min_slope)
        
        x_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(widths)]) + self.range_min
        y_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(heights)]) + self.range_min
        
        return x_pos, y_pos, slopes
    
    def forward(self, x):
        x_pos, y_pos, slopes = self._get_spline_params()
        return _spline_forward(x[0], x_pos, y_pos, slopes, self.range_min, self.range_max)
    
    def inverse(self, y):
        x_pos, y_pos, slopes = self._get_spline_params()
        return _spline_inverse(y[0], x_pos, y_pos, slopes, self.range_min, self.range_max)


class MaskedCouplingRQS(eqx.Module):
    mask: Float[Array, "input_dim"]
    mask_idx: Array
    transform_idx: Array
    conditioner: eqx.nn.MLP
    num_bins: int
    input_dim: int
    range_min: float
    range_max: float
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, num_bins: int = 8, *, key: Array):
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.range_min = -5.0
        self.range_max = 5.0
        
        split_dim = input_dim // 2
        self.mask = jnp.array([1.0] * split_dim + [0.0] * (input_dim - split_dim))
        
        mask_indices = jnp.arange(input_dim)
        self.mask_idx = mask_indices[self.mask.astype(bool)]
        self.transform_idx = mask_indices[~self.mask.astype(bool)]
        
        mask_dim = int(self.mask.sum())
        transform_dim = input_dim - mask_dim
        spline_params = 3 * num_bins + 1
        
        self.conditioner = eqx.nn.MLP(
            in_size=mask_dim, out_size=transform_dim * spline_params,
            width_size=128, depth=3, key=key
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
        
        y_masked = y[:, self.mask_idx]
        y_transform = y[:, self.transform_idx]
        
        params = eqx.filter_vmap(self.conditioner)(y_masked)
        x_transform, logdet = self._apply_inverse_spline(y_transform, params)
        
        x = jnp.zeros_like(y)
        x = x.at[:, self.mask_idx].set(y_masked)
        x = x.at[:, self.transform_idx].set(x_transform)
        
        assert x.shape == y.shape
        return x.reshape(original_shape), logdet.reshape(batch_shape)

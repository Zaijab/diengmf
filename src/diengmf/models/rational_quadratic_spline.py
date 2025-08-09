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

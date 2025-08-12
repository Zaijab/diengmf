import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# class RQSBijector(eqx.Module):
#     params: Array
#     input_dim: int
#     num_bins: int
#     range_min: float
#     range_max: float
#     min_bin_size: float
#     min_slope: float
#     eps: float
#     debug: bool
    
#     @jaxtyped(typechecker=typechecker)
#     def __init__(
#         self, 
#         input_dim: int, 
#         num_bins: int = 8, 
#         range_min: float = -5.0, 
#         range_max: float = 5.0,
#         min_bin_size: float = 1e-3,
#         min_slope: float = 1e-3,
#         eps: float = 1e-6,
#         debug: bool = False,
#         *, 
#         key: Array
#     ):
#         self.input_dim = input_dim
#         self.num_bins = num_bins
#         self.range_min = range_min
#         self.range_max = range_max
#         self.min_bin_size = min_bin_size
#         self.min_slope = min_slope
#         self.eps = eps
#         self.debug = debug
        
#         # Correct parameterization: K widths + K heights + (K+1) slopes = 3K+1
#         param_size = 3 * num_bins + 1
#         self.params = jax.random.normal(key, (input_dim, param_size)) * 0.01
    
#     @jaxtyped(typechecker=typechecker)
#     def _normalize_params(self, params: Array) -> tuple[Array, Array, Array]:
#         """Transform raw parameters using stable sigmoid parameterization."""
#         K = self.num_bins
#         params = jnp.clip(params, -5.0, 5.0)  # Prevent extreme values
        
#         if self.debug:
#             jax.debug.print("Raw params max: {}, min: {}, has_nan: {}", 
#                            jnp.max(jnp.abs(params)), jnp.min(params), jnp.any(jnp.isnan(params)))
        
#         # Split parameters: K + K + (K+1) = 3K+1
#         widths_raw = params[:K]
#         heights_raw = params[K:2*K]
#         slopes_raw = params[2*K:3*K+1]
#         # widths_raw = jnp.clip(params[:K], -10.0, 10.0)
#         # heights_raw = jnp.clip(params[K:2*K], -10.0, 10.0) 
#         # slopes_raw = jnp.clip(params[2*K:3*K+1], -10.0, 10.0)
        
#         total_range = self.range_max - self.range_min
        
#         # Sigmoid parameterization: more stable than softmax
#         # Each bin gets (min_size + sigmoid(raw) * remaining_size)
#         widths_sigmoid = jax.nn.sigmoid(widths_raw)
#         remaining_width = total_range - K * self.min_bin_size
#         widths = self.min_bin_size + widths_sigmoid * remaining_width / jnp.sum(widths_sigmoid)
        
#         heights_sigmoid = jax.nn.sigmoid(heights_raw) 
#         remaining_height = total_range - K * self.min_bin_size
#         heights = self.min_bin_size + heights_sigmoid * remaining_height / jnp.sum(heights_sigmoid)
        
#         # Bounded slope parameterization: sigmoid maps to [min_slope, max_slope]
#         max_slope = 10.0  # Reasonable upper bound
#         slopes_sigmoid = jax.nn.sigmoid(slopes_raw)
#         slopes = self.min_slope + slopes_sigmoid * (max_slope - self.min_slope)
        
#         if self.debug:
#             jax.debug.print("Widths: sum={}, min={}, max={}, has_nan={}", 
#                            jnp.sum(widths), jnp.min(widths), jnp.max(widths), jnp.any(jnp.isnan(widths)))
#             jax.debug.print("Heights: sum={}, min={}, max={}, has_nan={}", 
#                            jnp.sum(heights), jnp.min(heights), jnp.max(heights), jnp.any(jnp.isnan(heights)))
#             jax.debug.print("Slopes: min={}, max={}, has_nan={}", 
#                            jnp.min(slopes), jnp.max(slopes), jnp.any(jnp.isnan(slopes)))
        
#         # Build knot positions
#         x_knots = jnp.concatenate([jnp.array([self.range_min]), 
#                                   self.range_min + jnp.cumsum(widths)])
#         y_knots = jnp.concatenate([jnp.array([self.range_min]), 
#                                   self.range_min + jnp.cumsum(heights)])
        
#         if self.debug:
#             jax.debug.print("x_knots increasing: {}, y_knots increasing: {}", 
#                            jnp.all(jnp.diff(x_knots) > 0), jnp.all(jnp.diff(y_knots) > 0))
#             jax.debug.print("x_knots has_nan: {}, y_knots has_nan: {}", 
#                            jnp.any(jnp.isnan(x_knots)), jnp.any(jnp.isnan(y_knots)))
        
#         return x_knots, y_knots, slopes

    
#     @jaxtyped(typechecker=typechecker)
#     def _forward_scalar(self, x: Array, params: Array) -> tuple[Array, Array]:
#         """Forward RQS transformation."""
#         x_knots, y_knots, slopes = self._normalize_params(params)
        
#         # Boundary cases: linear extrapolation
#         below_range = x <= self.range_min
#         above_range = x >= self.range_max
        
#         y_below = self.range_min + (x - self.range_min) * slopes[0]
#         logdet_below = jnp.log(slopes[0])
        
#         y_above = self.range_max + (x - self.range_max) * slopes[-1]
#         logdet_above = jnp.log(slopes[-1])
        
#         # Find bin for spline region
#         bin_idx = jnp.searchsorted(x_knots[1:-1], x, side='right')
#         bin_idx = jnp.clip(bin_idx, 0, self.num_bins - 1)
        
#         # Extract bin parameters
#         x_k, x_k1 = x_knots[bin_idx], x_knots[bin_idx + 1]
#         y_k, y_k1 = y_knots[bin_idx], y_knots[bin_idx + 1]
#         d_k, d_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
#         # Bin metrics
#         width = x_k1 - x_k
#         height = y_k1 - y_k
#         xi = (x - x_k) / width
#         s = height / width
        
#         # Rational quadratic spline
#         xi_1 = 1.0 - xi
#         numerator = s * xi * xi + d_k * xi * xi_1
#         denominator = s + (d_k1 + d_k - 2.0 * s) * xi * xi_1
        
#         if self.debug:
#             jax.debug.print("Spline calc: xi={}, s={}, d_k={}, d_k1={}", xi, s, d_k, d_k1)
#             jax.debug.print("Spline calc: numerator={}, denominator={}", numerator, denominator)
#             jax.debug.print("Denominator near zero: {}", jnp.abs(denominator) < self.eps)
        
#         y_spline = y_k + height * numerator / denominator
        
#         # Derivative for log-det
#         derivative_num = s * s * (d_k1 * xi * xi + 2.0 * s * xi * xi_1 + d_k * xi_1 * xi_1)
#         derivative = derivative_num / (denominator * denominator)
        
#         if self.debug:
#             jax.debug.print("Derivative: num={}, denom^2={}, result={}", 
#                            derivative_num, denominator * denominator, derivative)
#             jax.debug.print("Derivative positive: {}, log(derivative) finite: {}", 
#                            derivative > 0, jnp.isfinite(jnp.log(derivative)))
        
#         logdet_spline = jnp.log(derivative)
#         # derivative = jnp.clip(derivative, 1e-6, 1e6)  # Prevent extreme derivatives
        
#         # Select based on range
#         y = jnp.where(below_range, y_below, jnp.where(above_range, y_above, y_spline))
#         logdet = jnp.where(below_range, logdet_below, jnp.where(above_range, logdet_above, logdet_spline))
        
#         return y, logdet
    
#     @jaxtyped(typechecker=typechecker)
#     def _inverse_scalar(self, y: Array, params: Array) -> tuple[Array, Array]:
#         """Inverse RQS transformation."""
#         x_knots, y_knots, slopes = self._normalize_params(params)
        
#         # Boundary cases
#         below_range = y <= self.range_min
#         above_range = y >= self.range_max
        
#         x_below = self.range_min + (y - self.range_min) / slopes[0]
#         logdet_below = -jnp.log(slopes[0])
        
#         x_above = self.range_max + (y - self.range_max) / slopes[-1]
#         logdet_above = -jnp.log(slopes[-1])
        
#         # Find bin
#         bin_idx = jnp.searchsorted(y_knots[1:-1], y, side='right')
#         bin_idx = jnp.clip(bin_idx, 0, self.num_bins - 1)
        
#         # Extract bin parameters
#         x_k, x_k1 = x_knots[bin_idx], x_knots[bin_idx + 1]
#         y_k, y_k1 = y_knots[bin_idx], y_knots[bin_idx + 1]
#         d_k, d_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
#         width = x_k1 - x_k
#         height = y_k1 - y_k
#         s = height / width
        
#         # Solve quadratic for xi: a*xi^2 + b*xi + c = 0
#         y_rel = y - y_k
#         a = height * (s - d_k) + y_rel * (d_k1 + d_k - 2.0 * s)
#         b = height * d_k - y_rel * (d_k1 + d_k - 2.0 * s)
#         c = -s * y_rel
        
#         if self.debug:
#             jax.debug.print("Quadratic: a={}, b={}, c={}", a, b, c)
#             jax.debug.print("Discriminant before clamp: {}", b * b - 4.0 * a * c)
        
#         # Stable quadratic formula
#         discriminant = b * b - 4.0 * a * c
#         sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        
#         # Choose stable root
#         xi = jnp.where(b >= 0.0,
#                       (-2.0 * c) / (b + sqrt_disc),
#                       (-b + sqrt_disc) / (2.0 * a))
#         xi = jnp.clip(xi, 0.0, 1.0)
        
#         if self.debug:
#             jax.debug.print("Inverse xi: before_clip={}, after_clip={}", 
#                            jnp.where(b >= 0.0, (-2.0 * c) / (b + sqrt_disc), (-b + sqrt_disc) / (2.0 * a)), xi)
        
#         x_spline = x_k + xi * width
        
#         # Derivative for log-det
#         xi_1 = 1.0 - xi
#         denominator = s + (d_k1 + d_k - 2.0 * s) * xi * xi_1
#         derivative_num = s * s * (d_k1 * xi * xi + 2.0 * s * xi * xi_1 + d_k * xi_1 * xi_1)
#         derivative = derivative_num / (denominator * denominator)
        
#         if self.debug:
#             jax.debug.print("Inverse derivative: {}, log={}", derivative, -jnp.log(derivative))
        
#         logdet_spline = -jnp.log(derivative)
        
#         # Select based on range
#         x = jnp.where(below_range, x_below, jnp.where(above_range, x_above, x_spline))
#         logdet = jnp.where(below_range, logdet_below, jnp.where(above_range, logdet_above, logdet_spline))
        
#         return x, logdet
    
#     @jaxtyped(typechecker=typechecker)
#     def forward(self, x: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Float[Array, ""]]:
#         """Apply RQS transformation element-wise."""
#         assert x.shape[-1] == self.input_dim
        
#         if self.debug:
#             jax.debug.print("Forward input: shape={}, range=[{}, {}], has_nan={}", 
#                            x.shape, jnp.min(x), jnp.max(x), jnp.any(jnp.isnan(x)))
        
#         y_vals, logdet_vals = jax.vmap(self._forward_scalar)(x, self.params)
#         total_logdet = jnp.sum(logdet_vals)
        
#         if self.debug:
#             jax.debug.print("Forward output: y_range=[{}, {}], logdet_range=[{}, {}]", 
#                            jnp.min(y_vals), jnp.max(y_vals), jnp.min(logdet_vals), jnp.max(logdet_vals))
#             jax.debug.print("Forward has_nan: y={}, logdet={}", 
#                            jnp.any(jnp.isnan(y_vals)), jnp.isnan(total_logdet))
        
#         return y_vals, total_logdet
    
#     @jaxtyped(typechecker=typechecker)
#     def inverse(self, y: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Float[Array, ""]]:
#         """Apply inverse RQS transformation element-wise."""
#         assert y.shape[-1] == self.input_dim
        
#         if self.debug:
#             jax.debug.print("Inverse input: shape={}, range=[{}, {}], has_nan={}", 
#                            y.shape, jnp.min(y), jnp.max(y), jnp.any(jnp.isnan(y)))
        
#         x_vals, logdet_vals = jax.vmap(self._inverse_scalar)(y, self.params)
#         total_logdet = jnp.sum(logdet_vals)
        
#         if self.debug:
#             jax.debug.print("Inverse output: x_range=[{}, {}], logdet_range=[{}, {}]", 
#                            jnp.min(x_vals), jnp.max(x_vals), jnp.min(logdet_vals), jnp.max(logdet_vals))
#             jax.debug.print("Inverse has_nan: x={}, logdet={}", 
#                            jnp.any(jnp.isnan(x_vals)), jnp.isnan(total_logdet))
        
#         return x_vals, total_logdet


###

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# MAJOR CHANGES FROM ORIGINAL IMPLEMENTATION:
# 1. Parameter normalization: Sigmoid -> Softmax (distrax uses softmax for proper probability simplex)
# 2. Slope normalization: Sigmoid -> Softplus+offset (distrax ensures slope=1 when input=0)  
# 3. Bin finding: searchsorted -> vectorized dot product (distrax avoids indexing for TPU)
# 4. Quadratic solver: Custom -> Safe quadratic root (distrax handles discriminant=0 and b sign)
# 5. Matrix shapes: Exact match to distrax (num_bins+1) for positions and slopes

# Distrax line 27-35: Parameter normalization functions
def _normalize_bin_sizes(unnormalized_bin_sizes: Array, total_size: float, min_bin_size: float) -> Array:
    """Make bin sizes sum to `total_size` and be no less than `min_bin_size`."""
    # Distrax line 28-34: Exact distrax implementation
    num_bins = unnormalized_bin_sizes.shape[-1]
    assert num_bins * min_bin_size <= total_size, f"num_bins * min_bin_size > total_size"
    bin_sizes = jax.nn.softmax(unnormalized_bin_sizes, axis=-1)  # Distrax line 33
    return bin_sizes * (total_size - num_bins * min_bin_size) + min_bin_size  # Distrax line 34

def _normalize_knot_slopes(unnormalized_knot_slopes: Array, min_knot_slope: float) -> Array:
    """Make knot slopes be no less than `min_knot_slope`."""
    # MAJOR CHANGE FROM OUR VERSION: Softplus+offset instead of sigmoid
    # NUMERICAL STABILITY: Distrax offset ensures slope=1 when input=0 (line 45)
    # COMMENTED OUT (our version): sigmoid-based parameterization, manual bounds
    # Distrax line 37-47: Exact distrax implementation with offset trick
    assert min_knot_slope < 1.0, f"min_knot_slope must be < 1, got {min_knot_slope}"
    min_knot_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1. - min_knot_slope) - 1.)  # Distrax line 45: offset for slope=1 when input=0
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope  # Distrax line 46

def _safe_quadratic_root(a: Array, b: Array, c: Array) -> Array:
    """Numerically stable quadratic formula from distrax line 124-140."""
    # NUMERICAL STABILITY IMPROVEMENTS FROM DISTRAX:
    # 1. Use finfo().tiny instead of manual epsilon (line 129)
    # 2. Conditional sqrt=0 when discriminant<=0 (line 131) 
    # 3. Choose stable root based on sign of b (lines 132-140)
    # COMMENTED OUT (our version): Manual discriminant handling, jnp.maximum
    # Distrax line 124-140: Exact distrax safe quadratic implementation
    sqrt_diff = b ** 2 - 4. * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))  # Distrax line 129
    safe_sqrt = jnp.where(sqrt_diff > 0., safe_sqrt, 0.)  # Distrax line 131
    
    # Distrax line 132-140: Choose stable root based on sign of b  
    numerator_1 = 2. * c
    denominator_1 = -b - safe_sqrt
    numerator_2 = -b + safe_sqrt  
    denominator_2 = 2 * a
    numerator = jnp.where(b >= 0, numerator_1, numerator_2)
    denominator = jnp.where(b >= 0, denominator_1, denominator_2)
    return numerator / denominator

def _rational_quadratic_spline_fwd(x: Array, x_pos: Array, y_pos: Array, knot_slopes: Array) -> tuple[Array, Array]:
    """Forward RQS from distrax line 49-94."""
    # MAJOR CHANGE: Bin finding via vectorized dot product (distrax line 66-69)
    # COMMENTED OUT (our version): jnp.searchsorted approach 
    # NUMERICAL STABILITY: Distrax ensures correct_bin fallback to first_bin
    # Distrax line 58-65: Find bin using vectorized approach (not searchsorted!)
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1] 
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([jnp.array([1]), jnp.zeros(len(correct_bin)-1)]).astype(bool)
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    
    # Distrax line 66-69: Vectorized parameter extraction via dot product
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)
    
    # Distrax line 71-74: Extract bin parameters  
    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])
    
    # Distrax line 76-89: Spline computation
    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width
    
    z = (x - x_pos_bin[0]) / bin_width
    z = jnp.clip(z, 0., 1.)  # Distrax line 81: Ensure z in [0,1]
    sq_z = z * z
    z1mz = z - sq_z  # Distrax line 83: z(1-z)
    sq_1mz = (1. - z) ** 2
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2. * bin_slope
    numerator = bin_height * (bin_slope * sq_z + knot_slopes_bin[0] * z1mz)
    denominator = bin_slope + slopes_term * z1mz
    y = y_pos_bin[0] + numerator / denominator
    
    # Distrax line 97-100: Log determinant computation  
    logdet = 2. * jnp.log(bin_slope) + jnp.log(
        knot_slopes_bin[1] * sq_z + 2. * bin_slope * z1mz + knot_slopes_bin[0] * sq_1mz
    ) - 2. * jnp.log(denominator)
    
    # Distrax line 103-106: Boundary handling
    y = jnp.where(below_range, (x - x_pos[0]) * knot_slopes[0] + y_pos[0], y)
    y = jnp.where(above_range, (x - x_pos[-1]) * knot_slopes[-1] + y_pos[-1], y) 
    logdet = jnp.where(below_range, jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, jnp.log(knot_slopes[-1]), logdet)
    return y, logdet

def _rational_quadratic_spline_inv(y: Array, x_pos: Array, y_pos: Array, knot_slopes: Array) -> tuple[Array, Array]:
    """Inverse RQS from distrax line 142-190."""
    # Distrax line 151-158: Find bin using vectorized approach  
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([jnp.array([1]), jnp.zeros(len(correct_bin)-1)]).astype(bool)
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    
    # Distrax line 159-162: Vectorized parameter extraction
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)
    
    # Distrax line 164-167: Extract bin parameters
    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1]) 
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])
    
    # Distrax line 169-179: Solve quadratic for z
    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width
    w = (y - y_pos_bin[0]) / bin_height
    w = jnp.clip(w, 0., 1.)  # Distrax line 173: Ensure w in [0,1]
    
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2. * bin_slope
    c = -bin_slope * w
    b = knot_slopes_bin[0] - slopes_term * w  
    a = bin_slope - b
    
    # Distrax line 180-181: Use safe quadratic root
    z = _safe_quadratic_root(a, b, c)
    z = jnp.clip(z, 0., 1.)  # Distrax line 182: Ensure z in [0,1]
    x = bin_width * z + x_pos_bin[0]
    
    # Distrax line 184-190: Log determinant (same as forward but negative)
    sq_z = z * z
    z1mz = z - sq_z
    sq_1mz = (1. - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet = -2. * jnp.log(bin_slope) - jnp.log(
        knot_slopes_bin[1] * sq_z + 2. * bin_slope * z1mz + knot_slopes_bin[0] * sq_1mz  
    ) + 2. * jnp.log(denominator)
    
    # Distrax line 192-195: Boundary handling
    x = jnp.where(below_range, (y - y_pos[0]) / knot_slopes[0] + x_pos[0], x)
    x = jnp.where(above_range, (y - y_pos[-1]) / knot_slopes[-1] + x_pos[-1], x)
    logdet = jnp.where(below_range, -jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, -jnp.log(knot_slopes[-1]), logdet)
    return x, logdet

class RQSBijector(eqx.Module):
    params: Array
    input_dim: int  
    num_bins: int
    range_min: float
    range_max: float
    min_bin_size: float
    min_knot_slope: float
    
    # Following distrax constructor line 206-238: Shape checks and parameter extraction
    @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, num_bins: int = 8, range_min: float = -5.0, range_max: float = 5.0,
                 min_bin_size: float = 1e-3, min_knot_slope: float = 1e-3, *, key: Array):
        self.input_dim = input_dim
        self.num_bins = num_bins  
        self.range_min = range_min
        self.range_max = range_max
        self.min_bin_size = min_bin_size
        self.min_knot_slope = min_knot_slope
        
        # Distrax line 206-209: Validate params shape (3 * num_bins + 1)
        param_size = 3 * num_bins + 1
        assert param_size >= 4, f"param_size must be >= 4, got {param_size}"
        assert range_min < range_max, f"range_min >= range_max"
        assert min_bin_size > 0., f"min_bin_size <= 0"
        assert min_knot_slope > 0., f"min_knot_slope <= 0"
        
        self.params = jax.random.normal(key, (input_dim, param_size)) * 0.01
    
    @jaxtyped(typechecker=typechecker) 
    def _get_spline_params(self, params: Array) -> tuple[Array, Array, Array]:
        """Extract and normalize spline parameters following distrax line 216-244."""
    # COMMENTED OUT (not in distrax): Our debug prints, manual clipping, sigmoid parameterization
    # COMMENTED OUT (not in distrax): self.eps, self.debug flags  
    # DISTRAX ADDITION: Shape validation and proper dtype handling
    # Distrax line 216-218: Extract unnormalized parameters
        unnormalized_bin_widths = params[:self.num_bins]
        unnormalized_bin_heights = params[self.num_bins:2*self.num_bins]
        unnormalized_knot_slopes = params[2*self.num_bins:]
        
        # Distrax line 219-228: Normalize and compute positions
        range_size = self.range_max - self.range_min
        bin_widths = _normalize_bin_sizes(unnormalized_bin_widths, range_size, self.min_bin_size)
        bin_heights = _normalize_bin_sizes(unnormalized_bin_heights, range_size, self.min_bin_size)
        
        x_pos_inner = self.range_min + jnp.cumsum(bin_widths[:-1])  # Distrax line 224
        y_pos_inner = self.range_min + jnp.cumsum(bin_heights[:-1])  # Distrax line 225
        
        # Distrax line 226-230: Add boundary positions
        pad_below = jnp.array([self.range_min], dtype=params.dtype)
        pad_above = jnp.array([self.range_max], dtype=params.dtype)
        x_pos = jnp.concatenate([pad_below, x_pos_inner, pad_above])
        y_pos = jnp.concatenate([pad_below, y_pos_inner, pad_above])
        
        # Distrax line 231-244: Normalize slopes (unconstrained boundary_slopes)
        knot_slopes = _normalize_knot_slopes(unnormalized_knot_slopes, self.min_knot_slope)
        
        assert x_pos.shape == (self.num_bins + 1,), f"x_pos shape {x_pos.shape}"
        assert y_pos.shape == (self.num_bins + 1,), f"y_pos shape {y_pos.shape}"  
        assert knot_slopes.shape == (self.num_bins + 1,), f"knot_slopes shape {knot_slopes.shape}"
        
        return x_pos, y_pos, knot_slopes
    
    @jaxtyped(typechecker=typechecker)
    def _forward_scalar(self, x: Array, params: Array) -> tuple[Array, Array]:
        """Forward scalar transformation for compatibility with MaskedCoupling."""
        x_pos, y_pos, knot_slopes = self._get_spline_params(params)
        return _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)
    
    @jaxtyped(typechecker=typechecker)
    def _inverse_scalar(self, y: Array, params: Array) -> tuple[Array, Array]:
        """Inverse scalar transformation for compatibility with MaskedCoupling."""
        x_pos, y_pos, knot_slopes = self._get_spline_params(params)
        return _rational_quadratic_spline_inv(y, x_pos, y_pos, knot_slopes)
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Forward transformation using distrax vectorized approach."""
        assert x.shape[-1] == self.input_dim
        
        def forward_scalar(x_scalar: Array, params_scalar: Array) -> tuple[Array, Array]:
            x_pos, y_pos, knot_slopes = self._get_spline_params(params_scalar)
            return _rational_quadratic_spline_fwd(x_scalar, x_pos, y_pos, knot_slopes)
        
        # Distrax uses jnp.vectorize with signature for broadcasting  
        y_vals, logdet_vals = jax.vmap(forward_scalar)(x, self.params)
        total_logdet = jnp.sum(logdet_vals)
        return y_vals, total_logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Inverse transformation using distrax vectorized approach."""
        assert y.shape[-1] == self.input_dim
        
        def inverse_scalar(y_scalar: Array, params_scalar: Array) -> tuple[Array, Array]:
            x_pos, y_pos, knot_slopes = self._get_spline_params(params_scalar) 
            return _rational_quadratic_spline_inv(y_scalar, x_pos, y_pos, knot_slopes)
        
        x_vals, logdet_vals = jax.vmap(inverse_scalar)(y, self.params)
        total_logdet = jnp.sum(logdet_vals)
        return x_vals, total_logdet

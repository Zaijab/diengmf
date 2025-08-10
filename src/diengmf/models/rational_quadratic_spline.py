import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

class RQSBijector(eqx.Module):
    params: Array
    input_dim: int
    num_bins: int
    range_min: float
    range_max: float
    min_bin_size: float
    min_slope: float
    eps: float
    debug: bool
    
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        input_dim: int, 
        num_bins: int = 8, 
        range_min: float = -5.0, 
        range_max: float = 5.0,
        min_bin_size: float = 1e-3,
        min_slope: float = 1e-3,
        eps: float = 1e-6,
        debug: bool = False,
        *, 
        key: Array
    ):
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.min_bin_size = min_bin_size
        self.min_slope = min_slope
        self.eps = eps
        self.debug = debug
        
        # Correct parameterization: K widths + K heights + (K+1) slopes = 3K+1
        param_size = 3 * num_bins + 1
        self.params = jax.random.normal(key, (input_dim, param_size)) * 0.01
    
    @jaxtyped(typechecker=typechecker)
    def _normalize_params(self, params: Array) -> tuple[Array, Array, Array]:
        """Transform raw parameters using stable sigmoid parameterization."""
        K = self.num_bins
        
        if self.debug:
            jax.debug.print("Raw params max: {}, min: {}, has_nan: {}", 
                           jnp.max(jnp.abs(params)), jnp.min(params), jnp.any(jnp.isnan(params)))
        
        # Split parameters: K + K + (K+1) = 3K+1
        widths_raw = params[:K]
        heights_raw = params[K:2*K]
        slopes_raw = params[2*K:3*K+1]
        
        total_range = self.range_max - self.range_min
        
        # Sigmoid parameterization: more stable than softmax
        # Each bin gets (min_size + sigmoid(raw) * remaining_size)
        widths_sigmoid = jax.nn.sigmoid(widths_raw)
        remaining_width = total_range - K * self.min_bin_size
        widths = self.min_bin_size + widths_sigmoid * remaining_width / jnp.sum(widths_sigmoid)
        
        heights_sigmoid = jax.nn.sigmoid(heights_raw) 
        remaining_height = total_range - K * self.min_bin_size
        heights = self.min_bin_size + heights_sigmoid * remaining_height / jnp.sum(heights_sigmoid)
        
        # Bounded slope parameterization: sigmoid maps to [min_slope, max_slope]
        max_slope = 10.0  # Reasonable upper bound
        slopes_sigmoid = jax.nn.sigmoid(slopes_raw)
        slopes = self.min_slope + slopes_sigmoid * (max_slope - self.min_slope)
        
        if self.debug:
            jax.debug.print("Widths: sum={}, min={}, max={}, has_nan={}", 
                           jnp.sum(widths), jnp.min(widths), jnp.max(widths), jnp.any(jnp.isnan(widths)))
            jax.debug.print("Heights: sum={}, min={}, max={}, has_nan={}", 
                           jnp.sum(heights), jnp.min(heights), jnp.max(heights), jnp.any(jnp.isnan(heights)))
            jax.debug.print("Slopes: min={}, max={}, has_nan={}", 
                           jnp.min(slopes), jnp.max(slopes), jnp.any(jnp.isnan(slopes)))
        
        # Build knot positions
        x_knots = jnp.concatenate([jnp.array([self.range_min]), 
                                  self.range_min + jnp.cumsum(widths)])
        y_knots = jnp.concatenate([jnp.array([self.range_min]), 
                                  self.range_min + jnp.cumsum(heights)])
        
        if self.debug:
            jax.debug.print("x_knots increasing: {}, y_knots increasing: {}", 
                           jnp.all(jnp.diff(x_knots) > 0), jnp.all(jnp.diff(y_knots) > 0))
            jax.debug.print("x_knots has_nan: {}, y_knots has_nan: {}", 
                           jnp.any(jnp.isnan(x_knots)), jnp.any(jnp.isnan(y_knots)))
        
        return x_knots, y_knots, slopes
    
    @jaxtyped(typechecker=typechecker)
    def _forward_scalar(self, x: Array, params: Array) -> tuple[Array, Array]:
        """Forward RQS transformation."""
        x_knots, y_knots, slopes = self._normalize_params(params)
        
        # Boundary cases: linear extrapolation
        below_range = x <= self.range_min
        above_range = x >= self.range_max
        
        y_below = self.range_min + (x - self.range_min) * slopes[0]
        logdet_below = jnp.log(slopes[0])
        
        y_above = self.range_max + (x - self.range_max) * slopes[-1]
        logdet_above = jnp.log(slopes[-1])
        
        # Find bin for spline region
        bin_idx = jnp.searchsorted(x_knots[1:-1], x, side='right')
        bin_idx = jnp.clip(bin_idx, 0, self.num_bins - 1)
        
        # Extract bin parameters
        x_k, x_k1 = x_knots[bin_idx], x_knots[bin_idx + 1]
        y_k, y_k1 = y_knots[bin_idx], y_knots[bin_idx + 1]
        d_k, d_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
        # Bin metrics
        width = x_k1 - x_k
        height = y_k1 - y_k
        xi = (x - x_k) / width
        s = height / width
        
        # Rational quadratic spline
        xi_1 = 1.0 - xi
        numerator = s * xi * xi + d_k * xi * xi_1
        denominator = s + (d_k1 + d_k - 2.0 * s) * xi * xi_1
        
        if self.debug:
            jax.debug.print("Spline calc: xi={}, s={}, d_k={}, d_k1={}", xi, s, d_k, d_k1)
            jax.debug.print("Spline calc: numerator={}, denominator={}", numerator, denominator)
            jax.debug.print("Denominator near zero: {}", jnp.abs(denominator) < self.eps)
        
        y_spline = y_k + height * numerator / denominator
        
        # Derivative for log-det
        derivative_num = s * s * (d_k1 * xi * xi + 2.0 * s * xi * xi_1 + d_k * xi_1 * xi_1)
        derivative = derivative_num / (denominator * denominator)
        
        if self.debug:
            jax.debug.print("Derivative: num={}, denom^2={}, result={}", 
                           derivative_num, denominator * denominator, derivative)
            jax.debug.print("Derivative positive: {}, log(derivative) finite: {}", 
                           derivative > 0, jnp.isfinite(jnp.log(derivative)))
        
        logdet_spline = jnp.log(derivative)
        
        # Select based on range
        y = jnp.where(below_range, y_below, jnp.where(above_range, y_above, y_spline))
        logdet = jnp.where(below_range, logdet_below, jnp.where(above_range, logdet_above, logdet_spline))
        
        return y, logdet
    
    @jaxtyped(typechecker=typechecker)
    def _inverse_scalar(self, y: Array, params: Array) -> tuple[Array, Array]:
        """Inverse RQS transformation."""
        x_knots, y_knots, slopes = self._normalize_params(params)
        
        # Boundary cases
        below_range = y <= self.range_min
        above_range = y >= self.range_max
        
        x_below = self.range_min + (y - self.range_min) / slopes[0]
        logdet_below = -jnp.log(slopes[0])
        
        x_above = self.range_max + (y - self.range_max) / slopes[-1]
        logdet_above = -jnp.log(slopes[-1])
        
        # Find bin
        bin_idx = jnp.searchsorted(y_knots[1:-1], y, side='right')
        bin_idx = jnp.clip(bin_idx, 0, self.num_bins - 1)
        
        # Extract bin parameters
        x_k, x_k1 = x_knots[bin_idx], x_knots[bin_idx + 1]
        y_k, y_k1 = y_knots[bin_idx], y_knots[bin_idx + 1]
        d_k, d_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
        width = x_k1 - x_k
        height = y_k1 - y_k
        s = height / width
        
        # Solve quadratic for xi: a*xi^2 + b*xi + c = 0
        y_rel = y - y_k
        a = height * (s - d_k) + y_rel * (d_k1 + d_k - 2.0 * s)
        b = height * d_k - y_rel * (d_k1 + d_k - 2.0 * s)
        c = -s * y_rel
        
        if self.debug:
            jax.debug.print("Quadratic: a={}, b={}, c={}", a, b, c)
            jax.debug.print("Discriminant before clamp: {}", b * b - 4.0 * a * c)
        
        # Stable quadratic formula
        discriminant = b * b - 4.0 * a * c
        sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        
        # Choose stable root
        xi = jnp.where(b >= 0.0,
                      (-2.0 * c) / (b + sqrt_disc),
                      (-b + sqrt_disc) / (2.0 * a))
        xi = jnp.clip(xi, 0.0, 1.0)
        
        if self.debug:
            jax.debug.print("Inverse xi: before_clip={}, after_clip={}", 
                           jnp.where(b >= 0.0, (-2.0 * c) / (b + sqrt_disc), (-b + sqrt_disc) / (2.0 * a)), xi)
        
        x_spline = x_k + xi * width
        
        # Derivative for log-det
        xi_1 = 1.0 - xi
        denominator = s + (d_k1 + d_k - 2.0 * s) * xi * xi_1
        derivative_num = s * s * (d_k1 * xi * xi + 2.0 * s * xi * xi_1 + d_k * xi_1 * xi_1)
        derivative = derivative_num / (denominator * denominator)
        
        if self.debug:
            jax.debug.print("Inverse derivative: {}, log={}", derivative, -jnp.log(derivative))
        
        logdet_spline = -jnp.log(derivative)
        
        # Select based on range
        x = jnp.where(below_range, x_below, jnp.where(above_range, x_above, x_spline))
        logdet = jnp.where(below_range, logdet_below, jnp.where(above_range, logdet_above, logdet_spline))
        
        return x, logdet
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Apply RQS transformation element-wise."""
        assert x.shape[-1] == self.input_dim
        
        if self.debug:
            jax.debug.print("Forward input: shape={}, range=[{}, {}], has_nan={}", 
                           x.shape, jnp.min(x), jnp.max(x), jnp.any(jnp.isnan(x)))
        
        y_vals, logdet_vals = jax.vmap(self._forward_scalar)(x, self.params)
        total_logdet = jnp.sum(logdet_vals)
        
        if self.debug:
            jax.debug.print("Forward output: y_range=[{}, {}], logdet_range=[{}, {}]", 
                           jnp.min(y_vals), jnp.max(y_vals), jnp.min(logdet_vals), jnp.max(logdet_vals))
            jax.debug.print("Forward has_nan: y={}, logdet={}", 
                           jnp.any(jnp.isnan(y_vals)), jnp.isnan(total_logdet))
        
        return y_vals, total_logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Apply inverse RQS transformation element-wise."""
        assert y.shape[-1] == self.input_dim
        
        if self.debug:
            jax.debug.print("Inverse input: shape={}, range=[{}, {}], has_nan={}", 
                           y.shape, jnp.min(y), jnp.max(y), jnp.any(jnp.isnan(y)))
        
        x_vals, logdet_vals = jax.vmap(self._inverse_scalar)(y, self.params)
        total_logdet = jnp.sum(logdet_vals)
        
        if self.debug:
            jax.debug.print("Inverse output: x_range=[{}, {}], logdet_range=[{}, {}]", 
                           jnp.min(x_vals), jnp.max(x_vals), jnp.min(logdet_vals), jnp.max(logdet_vals))
            jax.debug.print("Inverse has_nan: x={}, logdet={}", 
                           jnp.any(jnp.isnan(x_vals)), jnp.isnan(total_logdet))
        
        return x_vals, total_logdet

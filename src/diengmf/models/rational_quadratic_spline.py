import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

class RQSBijector(eqx.Module):
    range_min: float = -5.0
    range_max: float = 5.0
    min_bin_size: float = 1e-4
    min_slope: float = 1e-4
    
    def _normalize_params(self, params: Array):
        """Shared parameter normalization to ensure exact consistency"""
        K = (params.shape[-1] - 1) // 3
        widths_unnorm, heights_unnorm, slopes_unnorm = params[:K], params[K:2*K], params[2*K:]
        total_size = self.range_max - self.range_min
        
        widths_norm = jax.nn.softmax(widths_unnorm, axis=-1)
        widths = widths_norm * (total_size - K * self.min_bin_size) + self.min_bin_size
        
        heights_norm = jax.nn.softmax(heights_unnorm, axis=-1)
        heights = heights_norm * (total_size - K * self.min_bin_size) + self.min_bin_size
        
        slopes_offset = jnp.log(jnp.exp(1.0 - self.min_slope) - 1.0)
        slopes = jax.nn.softplus(slopes_unnorm + slopes_offset) + self.min_slope
        
        x_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(widths)]) + self.range_min
        y_pos = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(heights)]) + self.range_min
        
        return x_pos, y_pos, slopes
    
    def forward_with_params(self, x: Array, params: Array) -> tuple[Array, Array]:
        x_pos, y_pos, slopes = self._normalize_params(params)
        
        # Determine which region x is in
        below_range = x < self.range_min
        above_range = x > self.range_max
        
        # Linear extrapolation regions
        slope_left, slope_right = slopes[0], slopes[-1]
        y_linear_left = (x - self.range_min) * slope_left + self.range_min
        y_linear_right = (x - self.range_max) * slope_right + self.range_max
        logdet_linear_left = jnp.log(slope_left)
        logdet_linear_right = jnp.log(slope_right)
        
        # Spline computation for interior region
        num_bins = x_pos.shape[-1] - 1
        bin_idx = jnp.searchsorted(x_pos[1:-1], x, side='right')
        bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
        
        x_k, x_k1 = x_pos[bin_idx], x_pos[bin_idx + 1]
        y_k, y_k1 = y_pos[bin_idx], y_pos[bin_idx + 1]
        slope_k, slope_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
        width = x_k1 - x_k
        height = y_k1 - y_k
        xi = (x - x_k) / width
        xi = jnp.clip(xi, 0.0, 1.0)
        s = height / width
        
        # Rational quadratic transformation
        numerator = s * xi * xi + slope_k * xi * (1.0 - xi)
        denominator = s + (slope_k1 + slope_k - 2.0 * s) * xi * (1.0 - xi)
        
        y_spline = y_k + height * numerator / denominator
        
        # Derivative calculation for logdet
        xi_1m = 1.0 - xi
        xi_xi_1m = xi * xi_1m
        
        num_derivative = s * s * (slope_k1 * xi * xi + 2.0 * s * xi_xi_1m + slope_k * xi_1m * xi_1m)
        den_derivative = denominator * denominator
        
        derivative = num_derivative / den_derivative
        logdet_spline = jnp.log(derivative)
        
        # Combine results based on region
        y = jnp.where(below_range, y_linear_left, 
                     jnp.where(above_range, y_linear_right, y_spline))
        logdet = jnp.where(below_range, logdet_linear_left,
                          jnp.where(above_range, logdet_linear_right, logdet_spline))
        
        return y, logdet
    
    def inverse_with_params(self, y: Array, params: Array) -> tuple[Array, Array]:
        x_pos, y_pos, slopes = self._normalize_params(params)
        
        # Determine which region y is in
        below_range = y < self.range_min
        above_range = y > self.range_max
        
        # Linear extrapolation regions
        slope_left, slope_right = slopes[0], slopes[-1]
        x_linear_left = (y - self.range_min) / slope_left + self.range_min
        x_linear_right = (y - self.range_max) / slope_right + self.range_max
        logdet_linear_left = -jnp.log(slope_left)
        logdet_linear_right = -jnp.log(slope_right)
        
        # Spline inverse computation for interior region
        num_bins = y_pos.shape[-1] - 1
        bin_idx = jnp.searchsorted(y_pos[1:-1], y, side='right')
        bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
        
        x_k, x_k1 = x_pos[bin_idx], x_pos[bin_idx + 1]
        y_k, y_k1 = y_pos[bin_idx], y_pos[bin_idx + 1]
        slope_k, slope_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
        width = x_k1 - x_k
        height = y_k1 - y_k
        s = height / width
        y_rel = y - y_k
        
        # Solve the quadratic equation: a*xi^2 + b*xi + c = 0
        slope_sum = slope_k1 + slope_k
        a = height * (s - slope_k) + y_rel * (slope_sum - 2.0 * s)
        b = height * slope_k - y_rel * (slope_sum - 2.0 * s)
        c = -s * y_rel
        
        # Discriminant
        discriminant = b * b - 4.0 * a * c
        
        # Numerically stable quadratic formula
        sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        
        # Choose the stable root
        xi = jnp.where(
            b >= 0.0,
            (-2.0 * c) / (b + sqrt_discriminant),
            (-b + sqrt_discriminant) / (2.0 * a)
        )
        
        xi = jnp.clip(xi, 0.0, 1.0)
        x_spline = x_k + xi * width
        
        # Compute logdet (negative of forward derivative)
        xi_1m = 1.0 - xi
        xi_xi_1m = xi * xi_1m
        
        denominator = s + (slope_k1 + slope_k - 2.0 * s) * xi_xi_1m
        num_derivative = s * s * (slope_k1 * xi * xi + 2.0 * s * xi_xi_1m + slope_k * xi_1m * xi_1m)
        den_derivative = denominator * denominator
        
        derivative = num_derivative / den_derivative
        logdet_spline = -jnp.log(derivative)
        
        # Combine results based on region
        x = jnp.where(below_range, x_linear_left,
                     jnp.where(above_range, x_linear_right, x_spline))
        logdet = jnp.where(below_range, logdet_linear_left,
                          jnp.where(above_range, logdet_linear_right, logdet_spline))
        
        return x, logdet

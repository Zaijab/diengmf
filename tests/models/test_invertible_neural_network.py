import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped, Key
from beartype import beartype as typechecker
from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass

class RQSBijector(eqx.Module):
    range_min: float = -5.0
    range_max: float = 5.0
    min_bin_size: float = 1e-4
    min_slope: float = 1e-4
    
    def _normalize_params(self, params: Array):
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
        below_range, above_range = x < self.range_min, x > self.range_max
        slope_left, slope_right = slopes[0], slopes[-1]
        
        # Linear regions
        y_linear_left = (x - self.range_min) * slope_left + self.range_min
        y_linear_right = (x - self.range_max) * slope_right + self.range_max
        logdet_linear_left, logdet_linear_right = jnp.log(slope_left), jnp.log(slope_right)
        
        # Spline region
        num_bins = x_pos.shape[-1] - 1
        bin_idx = jnp.searchsorted(x_pos[1:-1], x, side='right')
        bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
        
        x_k, x_k1 = x_pos[bin_idx], x_pos[bin_idx + 1]
        y_k, y_k1 = y_pos[bin_idx], y_pos[bin_idx + 1]
        slope_k, slope_k1 = slopes[bin_idx], slopes[bin_idx + 1]
        
        width, height = x_k1 - x_k, y_k1 - y_k
        xi = (x - x_k) / width
        xi = jnp.clip(xi, 0.0, 1.0)
        s = height / width
        
        numerator = s * xi * xi + slope_k * xi * (1.0 - xi)
        denominator = s + (slope_k1 + slope_k - 2.0 * s) * xi * (1.0 - xi)
        y_spline = y_k + height * numerator / denominator
        
        xi_1m, xi_xi_1m = 1.0 - xi, xi * (1.0 - xi)
        num_derivative = s * s * (slope_k1 * xi * xi + 2.0 * s * xi_xi_1m + slope_k * xi_1m * xi_1m)
        derivative = num_derivative / (denominator * denominator)
        logdet_spline = jnp.log(derivative)
        
        # Combine regions
        y = jnp.where(below_range, y_linear_left, jnp.where(above_range, y_linear_right, y_spline))
        logdet = jnp.where(below_range, logdet_linear_left, jnp.where(above_range, logdet_linear_right, logdet_spline))
        
        return y, logdet

class MaskedCoupling(eqx.Module):
    mask: Float[Array, "input_dim"]
    conditioner: eqx.nn.MLP
    bijector: RQSBijector
    input_dim: int
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, bijector: RQSBijector, mask_type: Literal["half", "alternating"] = "half", 
                 conditioner_hidden_dim: int = 128, conditioner_depth: int = 3, num_bins: int = 8, *, key: Array):
        self.input_dim, self.bijector = input_dim, bijector
        split_dim = input_dim // 2
        transform_dim = input_dim - split_dim
        
        if mask_type == "half":
            self.mask = jnp.array([1.0] * split_dim + [0.0] * transform_dim)
        elif mask_type == "alternating":
            self.mask = jnp.array([float(i % 2) for i in range(input_dim)])
        
        spline_params = 3 * num_bins + 1
        self.conditioner = eqx.nn.MLP(split_dim, transform_dim * spline_params, conditioner_hidden_dim, conditioner_depth, jax.nn.relu, key=key)
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Array) -> tuple[Array, Array]:
        assert x.shape[-1] == self.input_dim
        mask_idx, transform_idx = jnp.where(self.mask)[0], jnp.where(1 - self.mask)[0]
        x_masked, x_transform = x[..., mask_idx], x[..., transform_idx]
        
        params = eqx.filter_vmap(self.conditioner)(x_masked)
        batch_size, transform_dim = x_transform.shape
        params_reshaped = params.reshape(batch_size, transform_dim, -1)
        
        y_list, logdet_list = [], []
        for i in range(transform_dim):
            def apply_bijector_to_batch(x_val, param_val):
                return self.bijector.forward_with_params(x_val, param_val)
            
            y_vals, logdets = eqx.filter_vmap(apply_bijector_to_batch)(x_transform[:, i], params_reshaped[:, i, :])
            y_list.append(y_vals)
            logdet_list.append(logdets)
        
        y_transform = jnp.stack(y_list, axis=-1)
        logdet = jnp.sum(jnp.stack(logdet_list, axis=-1), axis=-1)
        
        y = jnp.zeros_like(x)
        y = y.at[..., mask_idx].set(x_masked).at[..., transform_idx].set(y_transform)
        return y, logdet

class PLULinear(eqx.Module):
    n: int
    P: Array
    L_params: Array
    U_diag: Array
    U_upper: Array
    bias: Optional[Array]
    use_bias: bool
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, n: int, use_bias: bool = True, *, key: Array) -> None:
        p_key, l_key, u_diag_key, u_upper_key, bias_key = jax.random.split(key, 5)
        
        self.P = jnp.arange(n)
        l_size = (n * (n - 1)) // 2
        self.L_params = jax.random.normal(l_key, (l_size,)) * 0.01
        
        u_diag_init = jnp.ones(n) + jax.random.normal(u_diag_key, (n,)) * 0.01
        self.U_diag = jnp.log(jnp.exp(u_diag_init) - 1.0)
        
        u_upper_size = (n * (n - 1)) // 2
        self.U_upper = jax.random.normal(u_upper_key, (u_upper_size,)) * 0.01
        
        self.bias = jax.random.normal(bias_key, (n,)) * 0.01 if use_bias else None
        self.use_bias, self.n = use_bias, n
    
    def _construct_matrices(self):
        L = jnp.eye(self.n)
        indices = jnp.tril_indices(self.n, -1)
        L = L.at[indices].set(self.L_params)
        
        U = jnp.zeros((self.n, self.n))
        U = U.at[jnp.diag_indices(self.n)].set(jax.nn.softplus(self.U_diag) + 1e-6)
        
        upper_indices = jnp.triu_indices(self.n, 1)
        U = U.at[upper_indices].set(self.U_upper)
        
        P_matrix = jnp.eye(self.n)[self.P]
        return P_matrix, L, U
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Array) -> tuple[Array, Array]:
        P, L, U = self._construct_matrices()
        A = P @ L @ U
        y = x @ A.T
        if self.use_bias:
            y = y + self.bias
        
        logdet = jnp.sum(jnp.log(jnp.diag(U)))
        logdet = jnp.broadcast_to(logdet, x.shape[:-1])
        return y, logdet

@dataclass
class RQSINNConfig:
    """Comprehensive hyperparameter configuration for RQS Invertible Neural Networks"""
    
    # Core Architecture
    input_dim: int = 2
    num_coupling_layers: int = 6
    num_plu_layers: int = 2  # Number of PLU layers to intersperse
    layer_pattern: Literal["coupling_only", "sandwich", "interleaved"] = "interleaved"
    
    # RQS Parameters
    num_bins: int = 8
    spline_range_min: float = -5.0
    spline_range_max: float = 5.0
    
    # Conditioner Network
    conditioner_hidden_dim: int = 128
    conditioner_depth: int = 3
    conditioner_activation: Literal["relu", "gelu", "swish", "elu"] = "relu"
    
    # Masking Strategy
    mask_type_sequence: Optional[List[Literal["half", "alternating"]]] = None
    auto_reverse_masks: bool = True  # Automatically alternate mask types
    
    # PLU Configuration
    plu_use_bias: bool = True
    plu_initialization_scale: float = 0.01
    
    # Training Stability
    gradient_clip_norm: Optional[float] = 1.0
    weight_decay: float = 1e-5
    
    def get_mask_sequence(self) -> List[Literal["half", "alternating"]]:
        if self.mask_type_sequence is not None:
            return self.mask_type_sequence
        
        if self.auto_reverse_masks:
            return ["half" if i % 2 == 0 else "alternating" for i in range(self.num_coupling_layers)]
        else:
            return ["half"] * self.num_coupling_layers

class RQSINN(eqx.Module):
    """Rational Quadratic Spline Invertible Neural Network"""
    
    coupling_layers: List[MaskedCoupling]
    plu_layers: List[PLULinear]
    input_dim: int
    config: RQSINNConfig
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, config: RQSINNConfig, *, key: Array):
        self.input_dim = config.input_dim
        self.config = config
        
        keys = jax.random.split(key, config.num_coupling_layers + config.num_plu_layers)
        bijector = RQSBijector(range_min=config.spline_range_min, range_max=config.spline_range_max)
        mask_sequence = config.get_mask_sequence()
        
        # Create coupling layers
        self.coupling_layers = []
        for i in range(config.num_coupling_layers):
            layer = MaskedCoupling(
                input_dim=config.input_dim,
                bijector=bijector,
                mask_type=mask_sequence[i],
                conditioner_hidden_dim=config.conditioner_hidden_dim,
                conditioner_depth=config.conditioner_depth,
                num_bins=config.num_bins,
                key=keys[i]
            )
            self.coupling_layers.append(layer)
        
        # Create PLU layers based on pattern
        self.plu_layers = []
        if config.layer_pattern in ["sandwich", "interleaved"]:
            for i in range(config.num_plu_layers):
                plu_layer = PLULinear(
                    n=config.input_dim,
                    use_bias=config.plu_use_bias,
                    key=keys[config.num_coupling_layers + i]
                )
                self.plu_layers.append(plu_layer)
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Array) -> tuple[Array, Array]:
        """Forward: data space → latent space"""
        total_logdet = jnp.zeros(x.shape[:-1])
        
        if self.config.layer_pattern == "coupling_only":
            # Just apply coupling layers sequentially
            for coupling in self.coupling_layers:
                x, ldj = coupling.forward(x)
                total_logdet += ldj
                
        elif self.config.layer_pattern == "sandwich":
            # PLU -> Couplings -> PLU
            if self.plu_layers:
                x, ldj = self.plu_layers[0].forward(x)
                total_logdet += ldj
            
            for coupling in self.coupling_layers:
                x, ldj = coupling.forward(x)
                total_logdet += ldj
            
            if len(self.plu_layers) > 1:
                x, ldj = self.plu_layers[1].forward(x)
                total_logdet += ldj
                
        elif self.config.layer_pattern == "interleaved":
            # Coupling -> PLU -> Coupling -> PLU -> ...
            for i, coupling in enumerate(self.coupling_layers):
                x, ldj = coupling.forward(x)
                total_logdet += ldj
                
                # Add PLU layer if available and not the last coupling
                if i < len(self.plu_layers):
                    x, ldj = self.plu_layers[i].forward(x)
                    total_logdet += ldj
        
        return x, total_logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, z: Array) -> tuple[Array, Array]:
        """Inverse: latent space → data space"""
        total_logdet = jnp.zeros(z.shape[:-1])
        
        if self.config.layer_pattern == "coupling_only":
            # Apply coupling layers in reverse
            for coupling in reversed(self.coupling_layers):
                z, ldj = coupling.inverse(z)
                total_logdet += ldj
                
        elif self.config.layer_pattern == "sandwich":
            # Reverse: PLU -> Couplings -> PLU
            if len(self.plu_layers) > 1:
                z, ldj = self.plu_layers[1].inverse(z)
                total_logdet += ldj
            
            for coupling in reversed(self.coupling_layers):
                z, ldj = coupling.inverse(z)
                total_logdet += ldj
            
            if self.plu_layers:
                z, ldj = self.plu_layers[0].inverse(z)
                total_logdet += ldj
                
        elif self.config.layer_pattern == "interleaved":
            # Reverse: PLU -> Coupling -> PLU -> Coupling -> ...
            for i in range(len(self.coupling_layers) - 1, -1, -1):
                # Apply PLU in reverse if it exists
                if i < len(self.plu_layers):
                    z, ldj = self.plu_layers[i].inverse(z)
                    total_logdet += ldj
                
                # Apply coupling in reverse
                z, ldj = self.coupling_layers[i].inverse(z)
                total_logdet += ldj
        
        return z, total_logdet

def test_rqs_inn():
    """Test the RQS INN implementation"""
    key = jax.random.key(42)
    
    # Test configuration
    config = RQSINNConfig(
        input_dim=2,
        num_coupling_layers=4,
        num_plu_layers=2,
        layer_pattern="interleaved",
        num_bins=8,
        conditioner_hidden_dim=64,
        conditioner_depth=3
    )
    
    # Create model
    model = RQSINN(config, key=key)
    
    # Test forward and inverse
    x_test = jax.random.normal(jax.random.key(123), (10, 2))
    
    # Forward pass
    z, fwd_logdet = model.forward(x_test)
    print(f"Forward: x.shape={x_test.shape} -> z.shape={z.shape}, logdet.shape={fwd_logdet.shape}")
    
    # Inverse pass  
    x_recon, inv_logdet = model.inverse(z)
    print(f"Inverse: z.shape={z.shape} -> x_recon.shape={x_recon.shape}, logdet.shape={inv_logdet.shape}")
    
    # Test invertibility
    reconstruction_error = jnp.max(jnp.abs(x_test - x_recon))
    logdet_consistency = jnp.max(jnp.abs(fwd_logdet + inv_logdet))
    
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    print(f"Logdet consistency: {logdet_consistency:.2e}")
    
    assert reconstruction_error < 1e-8, f"Poor reconstruction: {reconstruction_error:.2e}"
    assert logdet_consistency < 1e-8, f"Poor logdet consistency: {logdet_consistency:.2e}"
    
    print("✅ RQS INN test passed!")
    
    # Count parameters
    param_count = sum(jnp.size(leaf) for leaf in jax.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"Total parameters: {param_count:,}")

test_rqs_inn()

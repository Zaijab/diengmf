import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker


# class CouplingLayer(eqx.Module):
#     s_net: eqx.nn.MLP
#     t_net: eqx.nn.MLP
#     input_dim: int
#     swap: bool

#     def __init__(self, input_dim, hidden_dim, num_hidden_layers, swap, *, key):
#         s_key, t_key = jax.random.split(key)

#         self.input_dim = input_dim
#         self.swap = swap

#         # Calculate dimensions correctly for odd inputs
#         split_dim1 = input_dim // 2
#         split_dim2 = input_dim - split_dim1

#         # Determine conditioning and output dimensions based on swap
#         if swap:
#             condition_dim = split_dim2  # Larger part when odd
#             output_dim = split_dim1  # Smaller part when odd
#         else:
#             condition_dim = split_dim1  # Smaller part when odd
#             output_dim = split_dim2  # Larger part when odd

#         self.s_net = eqx.nn.MLP(
#             in_size=condition_dim,
#             out_size=output_dim,
#             width_size=hidden_dim,
#             depth=num_hidden_layers,
#             activation=jax.nn.gelu,
#             key=s_key,
#             dtype=jnp.float64,
#         )

#         self.t_net = eqx.nn.MLP(
#             in_size=condition_dim,
#             out_size=output_dim,
#             width_size=hidden_dim,
#             depth=num_hidden_layers,
#             activation=jax.nn.gelu,
#             key=t_key,
#             dtype=jnp.float64,
#         )

#     def _safe_split(self, x):
#         """Safely split input handling odd dimensions."""
#         input_dim = x.shape[-1]
#         split_point = input_dim // 2

#         if self.swap:
#             # For swap layers, take the larger part first when odd
#             split_point = input_dim - split_point
#             x1 = x[..., :split_point]
#             x2 = x[..., split_point:]
#             return x2, x1  # Return swapped
#         else:
#             x1 = x[..., :split_point]
#             x2 = x[..., split_point:]
#             return x1, x2

#     @jaxtyped(typechecker=typechecker)
#     def forward(
#         self, x: Float[Array, "... d_in"]
#     ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
#         """
#         Forward transformation through coupling layer.

#         Args:
#             x: Input tensor with shape (..., input_dim)

#         Returns:
#             Tuple of (transformed tensor, log determinant of Jacobian)
#         """
#         if self.swap:
#             x1, x2 = self._safe_split(x)
#             x1, x2 = x2, x1
#         else:
#             x1, x2 = self._safe_split(x)

#         s = self.s_net(x1)
#         s = 15 * jnp.tanh(s)
#         t = self.t_net(x1)

#         y2 = x2 * jnp.exp(s) + t
#         log_det_jacobian = jnp.sum(s, axis=-1)

#         y = jnp.concatenate([x1, y2], axis=-1)

#         return y, log_det_jacobian

#     @jaxtyped(typechecker=typechecker)
#     def inverse(
#         self, y: Float[Array, "... d_in"]
#     ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
#         """
#         Inverse transformation through coupling layer.

#         Args:
#             y: Input tensor with shape (..., input_dim)

#         Returns:
#             Tuple of (transformed tensor, log determinant of Jacobian)
#         """
#         if self.swap:
#             y1, y2 = self._safe_split(y)
#             y1, y2 = y2, y1
#         else:
#             y1, y2 = self._safe_split(y)

#         s = self.s_net(y1)
#         s = 15 * jnp.tanh(s)
#         t = self.t_net(y1)

#         x2 = (y2 - t) * jnp.exp(-s)
#         log_det_jacobian = -jnp.sum(s, axis=-1)

#         x = jnp.concatenate([y1, x2], axis=-1)

#         return x, log_det_jacobian

#     @jaxtyped(typechecker=typechecker)
#     def __call__(
#         self, x: Float[Array, "... d_in"], inverse: bool = False
#     ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
#         """
#         Apply forward or inverse transformation.

#         Args:
#             x: Input tensor
#             inverse: Whether to apply inverse transformation

#         Returns:
#             Tuple of (transformed tensor, log determinant of Jacobian)
#         """
#         return self.inverse(x) if inverse else self.forward(x)


class MaskedCoupling(eqx.Module):
    mask: Float[Array, "input_dim"]
    conditioner: eqx.nn.MLP
    bijector: eqx.Module
    input_dim: int
    debug: bool
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, bijector: eqx.Module, debug: bool = False, *, key: Array):
        self.input_dim, self.debug, self.bijector = input_dim, debug, bijector
        split_dim = input_dim // 2
        transform_dim = input_dim - split_dim
        self.mask = jnp.array([1.0] * split_dim + [0.0] * (input_dim - split_dim))
        spline_params = 3 * 8 + 1  # num_bins=8
        self.conditioner = eqx.nn.MLP(split_dim, transform_dim * spline_params, 128, 3, jax.nn.relu, key=key)
        if self.debug: print(f"MaskedCoupling init: {split_dim} -> {transform_dim * spline_params}")
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Array) -> tuple[Array, Array]:
        assert x.shape[-1] == self.input_dim
        if self.debug: print(f"forward: x.shape={x.shape}")
        
        mask_idx = jnp.where(self.mask)[0] 
        transform_idx = jnp.where(1 - self.mask)[0]
        if self.debug: print(f"mask_idx={mask_idx}, transform_idx={transform_idx}")
        
        x_masked = x[..., mask_idx]
        x_transform = x[..., transform_idx]
        if self.debug: print(f"x_masked.shape={x_masked.shape}, x_transform.shape={x_transform.shape}")
        
        params = eqx.filter_vmap(self.conditioner)(x_masked)
        if self.debug: print(f"params.shape={params.shape}")
        
        batch_size, transform_dim = x_transform.shape
        params_reshaped = params.reshape(batch_size, transform_dim, -1)
        if self.debug: print(f"params_reshaped.shape={params_reshaped.shape}")
        
        y_list, logdet_list = [], []
        for i in range(transform_dim):
            if self.debug: print(f"Processing dim {i}")
            def apply_bijector_to_batch(x_val, param_val):
                if self.debug: print(f"  x_val={x_val}, param_val.shape={param_val.shape}")
                return self.bijector.forward_with_params(x_val, param_val)
            
            y_vals, logdets = eqx.filter_vmap(apply_bijector_to_batch)(x_transform[:, i], params_reshaped[:, i, :])
            if self.debug: print(f"  y_vals.shape={y_vals.shape}, logdets.shape={logdets.shape}")
            
            y_list.append(y_vals)
            logdet_list.append(logdets)
        
        y_transform = jnp.stack(y_list, axis=-1)
        logdet = jnp.sum(jnp.stack(logdet_list, axis=-1), axis=-1)
        if self.debug: print(f"y_transform.shape={y_transform.shape}, logdet.shape={logdet.shape}")
        
        y = jnp.zeros_like(x)
        y = y.at[..., mask_idx].set(x_masked)
        y = y.at[..., transform_idx].set(y_transform)
        
        return y, logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Array) -> tuple[Array, Array]:
        assert y.shape[-1] == self.input_dim
        if self.debug: print(f"inverse: y.shape={y.shape}")
        
        mask_idx = jnp.where(self.mask)[0] 
        transform_idx = jnp.where(1 - self.mask)[0]
        
        y_masked = y[..., mask_idx]
        y_transform = y[..., transform_idx]
        if self.debug: print(f"y_masked.shape={y_masked.shape}, y_transform.shape={y_transform.shape}")
        
        params = eqx.filter_vmap(self.conditioner)(y_masked)
        batch_size, transform_dim = y_transform.shape
        params_reshaped = params.reshape(batch_size, transform_dim, -1)
        
        x_list, logdet_list = [], []
        for i in range(transform_dim):
            def apply_inverse_bijector_to_batch(y_val, param_val):
                return self.bijector.inverse_with_params(y_val, param_val)
            
            x_vals, logdets = eqx.filter_vmap(apply_inverse_bijector_to_batch)(y_transform[:, i], params_reshaped[:, i, :])
            x_list.append(x_vals)
            logdet_list.append(logdets)
        
        x_transform = jnp.stack(x_list, axis=-1)
        logdet = jnp.sum(jnp.stack(logdet_list, axis=-1), axis=-1)
        
        x = jnp.zeros_like(y)
        x = x.at[..., mask_idx].set(y_masked)
        x = x.at[..., transform_idx].set(x_transform)
        
        return x, logdet

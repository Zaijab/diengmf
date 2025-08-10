import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

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

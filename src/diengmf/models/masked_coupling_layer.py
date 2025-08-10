import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Callable

class MaskedCoupling(eqx.Module):
    mask: Float[Array, "input_dim"]
    mask_idx: Array
    transform_idx: Array
    conditioner: eqx.nn.MLP
    bijector: eqx.Module
    input_dim: int
    num_bins: int
    debug: bool
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, input_dim: int, bijector: eqx.Module, num_bins: int = 8, 
                 conditioner_hidden_dim: int = 128, conditioner_depth: int = 6,
                 mask_strategy: str = "half", activation_function: Callable = jax.nn.gelu,
                 debug: bool = False, *, key: Array):
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.debug = debug
        self.bijector = bijector
        
        split_dim = input_dim // 2
        transform_dim = input_dim - split_dim
        
        if mask_strategy == "half":
            self.mask = jnp.array([1.0] * split_dim + [0.0] * transform_dim)
        elif mask_strategy == "alternating":
            self.mask = jnp.array([float(i % 2) for i in range(input_dim)])
        else:
            raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
        
        # Precompute indices for compile-time shape determination
        indices = jnp.arange(input_dim)
        self.mask_idx = indices[self.mask.astype(bool)]
        self.transform_idx = indices[~self.mask.astype(bool)]
        
        split_dim = self.mask_idx.shape[0]
        transform_dim = self.transform_idx.shape[0]
        spline_params = 3 * num_bins + 1
        self.conditioner = eqx.nn.MLP(
            split_dim, transform_dim * spline_params, 
            conditioner_hidden_dim, conditioner_depth, 
            activation_function, key=key
        )
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Array]:
        assert x.shape == (self.input_dim,)
        
        x_masked = x[self.mask_idx]
        x_transform = x[self.transform_idx]
        
        params = self.conditioner(x_masked)
        transform_dim = self.transform_idx.shape[0]
        spline_params = 3 * self.num_bins + 1
        params_reshaped = params.reshape(transform_dim, spline_params)
        
        y_transform_list = []
        logdet_list = []
        
        for i in range(transform_dim):
            y_val, logdet_val = self.bijector._forward_scalar(x_transform[i], params_reshaped[i])
            y_transform_list.append(y_val)
            logdet_list.append(logdet_val)
        
        y_transform = jnp.array(y_transform_list)
        logdet = jnp.sum(jnp.array(logdet_list))
        
        y = jnp.zeros_like(x)
        y = y.at[self.mask_idx].set(x_masked)
        y = y.at[self.transform_idx].set(y_transform)
        
        return y, logdet
    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, y: Float[Array, "input_dim"]) -> tuple[Float[Array, "input_dim"], Array]:
        assert y.shape == (self.input_dim,)
        
        y_masked = y[self.mask_idx]
        y_transform = y[self.transform_idx]
        
        params = self.conditioner(y_masked)
        transform_dim = self.transform_idx.shape[0]
        spline_params = 3 * self.num_bins + 1
        params_reshaped = params.reshape(transform_dim, spline_params)
        
        x_transform_list = []
        logdet_list = []
        
        for i in range(transform_dim):
            x_val, logdet_val = self.bijector._inverse_scalar(y_transform[i], params_reshaped[i])
            x_transform_list.append(x_val)
            logdet_list.append(logdet_val)
        
        x_transform = jnp.array(x_transform_list)
        logdet = jnp.sum(jnp.array(logdet_list))
        
        x = jnp.zeros_like(y)
        x = x.at[self.mask_idx].set(y_masked)
        x = x.at[self.transform_idx].set(x_transform)
        
        return x, logdet

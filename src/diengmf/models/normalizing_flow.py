"""
This file contains the definition for the composite invertible neural network, the normalizing flow.
This builds on the other invertible layers by chaining together compositions and summing the log-determinant-jacobians.
"""
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diengmf.models.invertible_linear_layer import PLULinear
from diengmf.models.masked_coupling_layer import MaskedCoupling
from diengmf.models.rational_quadratic_spline import RQSBijector
from jaxtyping import Array, Float, jaxtyped


class NormalizingFlow(eqx.Module):
    coupling_layers: eqx.Module
    plu_layers: eqx.Module
    masks: Float[Array, "num_layers input_dim"]
    input_dim: int
    
    @jaxtyped(typechecker=typechecker) 
    def __init__(self, input_dim: int, num_layers: int,
                 # RQSBijector
                 num_bins: int = 8, rqs_range_min: float = -5.0, rqs_range_max: float = 5.0,
                 min_bin_size: float = 1e-3, min_knot_slope: float = 1e-3,
                 # PLU
                 #plu_use_bias: bool = True,
                 initialization_scale: float = 0.2,
                 # Masked Coupling Conditioner
                 conditioner_hidden_dim: int = 128, conditioner_depth: int = 3,
                 #mask_strategy: str = "alternating",
                 activation_function: Callable = jax.nn.gelu, *, key: Array):
        
        self.input_dim = input_dim
        coupling_key, plu_key, bijector_key = jax.random.split(key, 3)
        coupling_keys = jax.random.split(coupling_key, num_layers)
        plu_keys = jax.random.split(plu_key, num_layers)
        
        def make_coupling(k):
            bijector_key, conditioner_key = jax.random.split(k, 2)
            bijector = RQSBijector(input_dim=input_dim, num_bins=num_bins,
                                   range_min=rqs_range_min, range_max=rqs_range_max,
                                   min_bin_size=min_bin_size, min_knot_slope=min_knot_slope,
                                   key=bijector_key)
            return MaskedCoupling(input_dim, bijector, num_bins=num_bins,
                                conditioner_hidden_dim=conditioner_hidden_dim,
                                conditioner_depth=conditioner_depth,
                                mask_strategy="half", activation_function=activation_function,
                                key=conditioner_key)
        
        def make_plu(k):
            return PLULinear(input_dim, True, initialization_scale, key=k) # Hard coding use Bias
        
        self.coupling_layers = eqx.filter_vmap(make_coupling)(coupling_keys)
        self.plu_layers = eqx.filter_vmap(make_plu)(plu_keys)
        
        if mask_strategy == "alternating":
            mask_pattern = jnp.array([float(i % 2) for i in range(input_dim)])
            self.masks = jnp.array([mask_pattern if i % 2 == 0 else 1.0 - mask_pattern 
                                   for i in range(num_layers)])
        else:
            split_dim = input_dim // 2
            base_mask = jnp.array([1.0] * split_dim + [0.0] * (input_dim - split_dim))
            self.masks = jnp.tile(base_mask, (num_layers, 1))
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(self, x: Array) -> tuple[Array, Array]:
        assert x.shape[-1] == self.input_dim
        dynamic_coupling, static_coupling = eqx.partition(self.coupling_layers, eqx.is_array)
        dynamic_plu, static_plu = eqx.partition(self.plu_layers, eqx.is_array)
        
        def forward_step(carry, layer_data):
            x_curr, logdet_curr = carry
            dynamic_c, dynamic_p, mask = layer_data
            
            coupling = eqx.combine(dynamic_c, static_coupling)
            coupling = eqx.tree_at(lambda c: c.mask, coupling, mask)
            x_new, ldj1 = coupling.forward(x_curr)
            
            plu = eqx.combine(dynamic_p, static_plu)
            x_final, ldj2 = plu.forward(x_new)
            return (x_final, logdet_curr + ldj1 + ldj2), None
        
        layer_data = (dynamic_coupling, dynamic_plu, self.masks)
        (x_final, total_logdet), _ = jax.lax.scan(forward_step, (x, jnp.zeros(())), layer_data)
        return x_final, total_logdet
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def inverse(self, z: Array) -> tuple[Array, Array]:
        assert z.shape[-1] == self.input_dim
        dynamic_coupling, static_coupling = eqx.partition(self.coupling_layers, eqx.is_array)
        dynamic_plu, static_plu = eqx.partition(self.plu_layers, eqx.is_array)
        
        def inverse_step(carry, layer_data):
            z_curr, logdet_curr = carry
            dynamic_c, dynamic_p, mask = layer_data
            
            plu = eqx.combine(dynamic_p, static_plu)
            z_new, ldj1 = plu.inverse(z_curr)
            
            coupling = eqx.combine(dynamic_c, static_coupling)
            coupling = eqx.tree_at(lambda c: c.mask, coupling, mask)
            z_final, ldj2 = coupling.inverse(z_new)
            return (z_final, logdet_curr + ldj1 + ldj2), None
        
        reversed_dynamic_coupling = jax.tree.map(lambda x: x[::-1], dynamic_coupling)
        reversed_dynamic_plu = jax.tree.map(lambda x: x[::-1], dynamic_plu)
        reversed_masks = self.masks[::-1]
        
        layer_data = (reversed_dynamic_coupling, reversed_dynamic_plu, reversed_masks)
        (z_final, total_logdet), _ = jax.lax.scan(inverse_step, (z, jnp.zeros(())), layer_data)
        return z_final, total_logdet

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
from jaxtyping import Array, jaxtyped


class NormalizingFlow(eqx.Module):
    coupling_layers: list
    plu_layers: list
    input_dim: int
    
    @jaxtyped(typechecker=typechecker) 
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 num_bins: int = 8,
                 rqs_range_min: float = -5.0,
                 rqs_range_max: float = 5.0,
                 plu_use_bias: bool = True,
                 conditioner_hidden_dim: int = 128,
                 conditioner_depth: int = 3,
                 mask_strategy: str = "alternating",
                 activation_function: Callable = jax.nn.gelu,
                 *,
                 key: Array):
        self.input_dim, keys = input_dim, jax.random.split(key, 2 * num_layers)
        bijector = RQSBijector(range_min=rqs_range_min, range_max=rqs_range_max)
        self.coupling_layers = [MaskedCoupling(input_dim, bijector, key=keys[i],
                                               conditioner_hidden_dim=conditioner_hidden_dim,
                                               conditioner_depth=conditioner_depth,
                                               mask_strategy=mask_strategy,
                                               activation_function=activation_function) for i in range(num_layers)]
        self.plu_layers = [PLULinear(input_dim, plu_use_bias, key=keys[num_layers + i]) for i in range(num_layers)]
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Array) -> tuple[Array, Array]:
        assert x.shape[-1] == self.input_dim
        total_logdet = jnp.zeros(x.shape[:-1])
        for coupling, plu in zip(self.coupling_layers, self.plu_layers):
            x, ldj = coupling.forward(x); total_logdet += ldj
            x, ldj = plu.forward(x); total_logdet += ldj
        return x, total_logdet

    
    @jaxtyped(typechecker=typechecker)
    def inverse(self, z: Array) -> tuple[Array, Array]:
        assert z.shape[-1] == self.input_dim
        total_logdet = jnp.zeros(z.shape[:-1])
        for plu, coupling in zip(reversed(self.plu_layers), reversed(self.coupling_layers)):
            z, ldj = plu.inverse(z); total_logdet += ldj
            z, ldj = coupling.inverse(z); total_logdet += ldj
        return z, total_logdet
    
NormalizingFlow(2, 10, key=jax.random.key(0)).forward(jax.numpy.array([0.0, 0.0]))

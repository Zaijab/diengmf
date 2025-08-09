from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class MaskedCoupling(eqx.Module):
    transform: eqx.Module  # Standalone RQS
    mask: Array
    mask_idx: Array        # Precomputed at init
    transform_idx: Array   # Precomputed at init
    
    def __init__(self, transform, mask):
        self.transform = transform
        self.mask = mask
        # These are compile-time constants
        indices = jnp.arange(len(mask))
        self.mask_idx = indices[mask.astype(bool)]
        self.transform_idx = indices[~mask.astype(bool)]

    @eqx.filter_jit
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        x_masked = x[self.mask_idx]
        x_transform = x[self.transform_idx]  # Fixed shape: (n_transform,)
        
        y_transform, log_det = self.transform(x_transform)
        
        y = jnp.zeros_like(x)
        y = y.at[self.mask_idx].set(x_masked)
        y = y.at[self.transform_idx].set(y_transform)
        
        return y, log_det

key = jax.random.key(10)

@eqx.filter_jit
def f(key):
    key, subkey = jax.random.split(key)
    mask = jax.random.bernoulli(subkey, shape=(10,))
    model = MaskedCoupling(transform=lambda x: (x ** 2, jnp.sum(x)), mask=mask)
    x = jax.random.normal(subkey, shape=(10,))
    return x, model(x)[0]

f(key)

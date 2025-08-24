"""
Flow discriminator resampling for EnGMF.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key

from diengmf.models.normalizing_flow import NormalizingFlow


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, 0, None, None, None))
def flow_discriminator_resampler(
    key: Key[Array, ""],
    mean: Float[Array, " state_dim"],
    cov: Float[Array, " state_dim state_dim"],
    model: NormalizingFlow,
    threshold: float = 0.0,
    max_steps: int = 1000,
) -> Float[Array, " state_dim"]:
    """
    Discriminator resampling function for EnGMF sampling_function.
    
    Samples from multivariate normal until flow log-probability exceeds threshold.
    Returns the first sample that passes.
    """
    
    def cond_fn(carry):
        key_state, sample, accepted = carry
        return ~accepted
    
    def body_fn(carry):
        key_state, sample, accepted = carry
        key_state, subkey = jax.random.split(key_state)
        
        candidate = jax.random.multivariate_normal(subkey, mean, cov)
        _, log_prob = model.forward(candidate)
        pass_threshold = log_prob > threshold
        
        new_sample = jnp.where(pass_threshold, candidate, sample)
        new_accepted = accepted | pass_threshold
        
        return (key_state, new_sample, new_accepted)
    
    # Initialize with first random sample as fallback
    key, fallback_key = jax.random.split(key)
    fallback_sample = jax.random.multivariate_normal(fallback_key, mean, cov)
    
    init_carry = (key, fallback_sample, False)
    _, final_sample, accepted = eqx.internal.while_loop(
        cond_fn, body_fn, init_carry, max_steps=max_steps, kind="bounded"
    )
    
    # If no sample was accepted, return the fallback
    return jnp.where(accepted, final_sample, fallback_sample)

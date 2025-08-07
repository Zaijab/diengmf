import equinox as eqx
from jaxtyping import Float, Array, Key, jaxtyped
from collections.abc import Callable
from beartype import beartype as typechecker
import jax.numpy as jnp
import jax
import optax
from diengmf.statistics import logpdf_epanechnikov

@eqx.filter_value_and_grad
def kl_divergence(
    model: Callable, batch: Float[Array, "batch_dim state_dim"]
) -> Float[Array, "1"]:
    z, log_det_jacobian = eqx.filter_vmap(model.inverse)(batch)
    base_log_prob = eqx.filter_vmap(logpdf_epanechnikov, in_axes=(0, None, None))(
        z, jnp.zeros(batch.shape[-1]), jnp.eye(batch.shape[-1])
    )
    total_log_prob = base_log_prob + log_det_jacobian
    total_log_prob = jnp.where(jnp.isfinite(total_log_prob), total_log_prob, -1e6)
    return -jnp.mean(total_log_prob)


@eqx.filter_jit
def make_step(
    model: Callable, x, optim, opt_state
) -> tuple[Float[Array, "..."], eqx.Module, Float[Array, "..."]]:
    loss, grad = kl_divergence(model, x)
    updates, opt_state = optim.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

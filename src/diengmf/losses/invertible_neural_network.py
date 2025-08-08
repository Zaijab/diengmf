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


"""
The purpose of this test is to run an arbitrary Equinox model through a training loop.
"""

import uuid
from tqdm import tqdm
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from diengmf.dynamical_systems import Ikeda
from diengmf.losses import make_step
from diengmf.models import InvertibleNN
from diengmf.statistics import sample_epanechnikov
from jaxtyping import Array, Float, Key, jaxtyped

print(uuid.uuid4())


def training_loop(model, system, optim):
    """
    Less flexible loop structure for testing JIT compatibility of models easily.
    """
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    model = RationalQuadraticSpline(input_dim=2, num_bins=8, key=subkey, range_min=-5.0, range_max=5.0)
    system = Ikeda(batch_size=25)
    batch = system.generate(jax.random.key(0), batch_size=1000)

    optim = optax.chain(
        optax.adam(
            learning_rate=1e-4,
            eps=1e-4,
        ),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    loss, model, opt_state = make_step(model, batch, optim, opt_state)

    x = jax.random.multivariate_normal(subkey, mean=jnp.zeros(2), cov=jnp.eye(2))

    # batch = eqx.filter_vmap(system.forward)(batch)

    # loss, model, opt_state = make_step(model, batch, optim, opt_state)
    # model


    # def plot_learning(model: InvertibleNN) -> None:
    #     samples = sample_epanechnikov(
    #         jax.random.key(0), jnp.zeros(2), jnp.eye(2), batch.shape[0]
    #     )

    #     generated_data = eqx.filter_vmap(model)(samples)[0]

    #     plt.scatter(generated_data[:, 0], generated_data[:, 1], c="red", alpha=0.15)
    #     plt.xlim(-1, 2)
    #     plt.ylim(-3, 1.5)
    #     plt.show()


    for i in range(1_000):
        # print(
        #     x,
        #     model.inverse(model.forward(x)[0])[0],
        #     jnp.allclose(x, model.inverse(model.forward(x)[0])[0]),
        # )
        batch = eqx.filter_vmap(system.forward)(batch)
        loss, model, opt_state = make_step(model, batch, optim, opt_state)

        if (i % 100) == 0:
            print(loss)
            # plot_learning(model)

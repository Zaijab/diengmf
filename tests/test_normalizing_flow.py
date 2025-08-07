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


debug = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)

###

# model = InvertibleNN(hidden_dim=32, num_coupling_layers=6,  num_hidden_layers=2, key=subkey)
model = RationalQuadraticSpline(input_dim=2, num_bins=8, key=subkey, range_min=-5.0, range_max=5.0)

###
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

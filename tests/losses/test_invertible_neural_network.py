from diengmf.losses import training_loop, make_step
from diengmf.models import InvertibleNN
from diengmf.dynamical_systems import Ikeda
import optax
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import jaxtyped, Array
from beartype import beartype as typechecker

# def plot_learning(model: InvertibleNN) -> None:
#     samples = sample_epanechnikov(
#         jax.random.key(0), jnp.zeros(2), jnp.eye(2), batch.shape[0]
#     )

#     generated_data = eqx.filter_vmap(model)(samples)[0]

#     plt.scatter(generated_data[:, 0], generated_data[:, 1], c="red", alpha=0.15)
#     plt.xlim(-1, 2)
#     plt.ylim(-3, 1.5)
#     plt.show()

@eqx.filter_jit
def training_loop(key: Array, model: eqx.Module, system: eqx.Module, optim: optax.Schedule):
    """
    # batch = dynamical_system.generate(subkey, batch_size=500, final_time=50)
    # opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # for _ in range(401):
    #     batch = eqx.filter_vmap(dynamical_system.forward)(batch)
    #     loss, model, opt_state = make_step(model, batch, optim, opt_state)
    #     if (i % 100) == 0:
    #         print(loss)
    # loss
    """
    key, subkey = jax.random.split(key)
    batch = system.generate(subkey, batch_size=500, final_time=50)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Partition model and opt_state to separate arrays from static elements
    model_arrays, model_static = eqx.partition(model, eqx.is_array)
    opt_arrays, opt_static = eqx.partition(opt_state, eqx.is_array)

    def scan_step(carry, _):
        batch, model_arrays, opt_arrays, i = carry
        
        # Reconstruct full objects from arrays + static parts
        model = eqx.combine(model_arrays, model_static)
        opt_state = eqx.combine(opt_arrays, opt_static)
        
        # Perform training step
        batch = eqx.filter_vmap(system.forward)(batch)
        loss, model, opt_state = make_step(model, batch, optim, opt_state)
        
        # Debug print - only every 100 iterations
        jax.lax.cond(
            (i % 100) == 0,
            lambda _: jax.debug.print('Step {}: loss = {}', i, loss),
            lambda _: None,
            None
        )
        
        # Partition updated objects back to arrays for next iteration
        model_arrays, _ = eqx.partition(model, eqx.is_array)
        opt_arrays, _ = eqx.partition(opt_state, eqx.is_array)
        
        return (batch, model_arrays, opt_arrays, i + 1), loss

    initial_carry = (batch, model_arrays, opt_arrays, 0)
    (final_batch, final_model_arrays, final_opt_arrays, _), losses = jax.lax.scan(
        scan_step, 
        initial_carry, 
        xs=jnp.zeros(401)
    )
    
    # Reconstruct final objects
    final_model = eqx.combine(final_model_arrays, model_static)
    final_opt_state = eqx.combine(final_opt_arrays, opt_static)
    
    return final_model, final_opt_state

key = jax.random.key(10)
key, subkey = jax.random.split(key)
###
##
# from diengmf.models.equinox_rational_quadratic_spline import RationalQuadraticSpline
# model = RationalQuadraticSpline(input_dim=2, key=key)
##
# model = InvertibleNN(key=key)
##
# from diengmf.models.equinox_masked_coupling_layer import MaskedCouplingLayer
# model = MaskedCouplingLayer(input_dim=2, key=key)
##
# from diengmf.models.equinox_masked_coupling_layer import MaskedCouplingAffine
# model = MaskedCouplingAffine(input_dim=2, key=key)
##


####



####
from diengmf.models.equinox_masked_coupling_layer import MaskedCouplingRQS
model = MaskedCouplingRQS(input_dim=2, key=key)
###

from diengmf.dynamical_systems
dynamical_system = Ikeda()
optim = optax.chain(
    optax.adam(
        learning_rate=1e-4,
        eps=1e-4,
    ),
)

# dynamical_systems = [Ikeda(), Lorenz63(), Lorenz96()]

final_model, final_opt_state = training_loop(key, model, dynamical_system, optim)

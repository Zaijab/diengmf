import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from diengmf.models import PLULinear
from diengmf.losses import make_step, kl_divergence
import optax

# @eqx.filter_jit
def training_loop(key: Array, model: eqx.Module, system: eqx.Module, optim: optax.Schedule):
    key, subkey = jax.random.split(key)
    batch = system.generate(subkey, batch_size=500, final_time=50.0)
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
        batch = eqx.filter_vmap(system.flow)(0.0, 1.0, batch)
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
        xs=jnp.zeros(10_0001)
    )
    
    # Reconstruct final objects
    final_model = eqx.combine(final_model_arrays, model_static)
    final_opt_state = eqx.combine(final_opt_arrays, opt_static)
    
    return final_model, final_opt_state


def test_normalizing_flow_suite(model_class):
    key = jax.random.key(0)
    dynamical_systems = [Lorenz63()]
    for dynamical_system in dynamical_systems:

        dimension = dynamical_system.dimension
        model = model_class(input_dim=dimension, key=key, debug=False)
        optim = optax.chain(
            optax.adam(
                learning_rate=1e-4,
                eps=1e-4,
            ),
            # optax.clip_by_global_norm(1e-10),
        )
        print(dynamical_system)

        single_point = jax.random.normal(key, (dimension,))
        batch_data = jax.random.normal(key, (100, dimension))

        x_forward, _ = model.forward(single_point)
        x_reconstructed, _ = model.inverse(x_forward)
        assert jnp.max(jnp.abs(single_point - x_reconstructed)) < 1e-12

        print(f"{jnp.max(jnp.abs(single_point - x_reconstructed))=}")
        
        z_inverse, _ = model.inverse(single_point)  
        z_reconstructed, _ = model.forward(z_inverse)
        assert jnp.max(jnp.abs(single_point - z_reconstructed)) < 1e-12
        print(f"{jnp.max(jnp.abs(single_point - z_reconstructed))=}")
        
        try: model.forward(batch_data); assert False
        except: pass
        eqx.filter_vmap(model.forward)(batch_data)

        final_model, final_opt_state = training_loop(key, model, dynamical_system, optim)

        
        x_forward, _ = final_model.forward(single_point)
        x_reconstructed, _ = final_model.inverse(x_forward)
        assert jnp.max(jnp.abs(single_point - x_reconstructed)) < 1e-12

        print(f"{jnp.max(jnp.abs(single_point - x_reconstructed))=}")
        
        z_inverse, _ = final_model.inverse(single_point)  
        z_reconstructed, _ = final_model.forward(z_inverse)
        assert jnp.max(jnp.abs(single_point - z_reconstructed)) < 1e-12
        print(f"{jnp.max(jnp.abs(single_point - z_reconstructed))=}")
        
        try: final_model.forward(batch_data); assert False
        except: pass
        eqx.filter_vmap(final_model.forward)(batch_data)




###



###

test_normalizing_flow_suite(RQSBijector)

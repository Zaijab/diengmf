# from diengmf.losses import training_loop, make_step

# from diengmf.dynamical_systems import Ikeda
# import optax
# import jax
# import jax.numpy as jnp
# import equinox as eqx
# from jaxtyping import jaxtyped, Array
# from beartype import beartype as typechecker


# @eqx.filter_jit
# def training_loop(key: Array, model: eqx.Module, system: eqx.Module, optim: optax.Schedule):
#     key, subkey = jax.random.split(key)
#     batch = system.generate(subkey, batch_size=500, final_time=50.0)
#     opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

#     # Partition model and opt_state to separate arrays from static elements
#     model_arrays, model_static = eqx.partition(model, eqx.is_array)
#     opt_arrays, opt_static = eqx.partition(opt_state, eqx.is_array)

#     def scan_step(carry, _):
#         batch, model_arrays, opt_arrays, i = carry
        
#         # Reconstruct full objects from arrays + static parts
#         model = eqx.combine(model_arrays, model_static)
#         opt_state = eqx.combine(opt_arrays, opt_static)
        
#         # Perform training step
#         batch = eqx.filter_vmap(system.flow)(0.0, 1.0, batch)
#         loss, model, opt_state = make_step(model, batch, optim, opt_state)
        
#         # Debug print - only every 100 iterations
#         jax.lax.cond(
#             (i % 100) == 0,
#             lambda _: jax.debug.print('Step {}: loss = {}', i, loss),
#             lambda _: None,
#             None
#         )
        
#         # Partition updated objects back to arrays for next iteration
#         model_arrays, _ = eqx.partition(model, eqx.is_array)
#         opt_arrays, _ = eqx.partition(opt_state, eqx.is_array)
        
#         return (batch, model_arrays, opt_arrays, i + 1), loss

#     initial_carry = (batch, model_arrays, opt_arrays, 0)
#     (final_batch, final_model_arrays, final_opt_arrays, _), losses = jax.lax.scan(
#         scan_step, 
#         initial_carry, 
#         xs=jnp.zeros(401)
#     )
    
#     # Reconstruct final objects
#     final_model = eqx.combine(final_model_arrays, model_static)
#     final_opt_state = eqx.combine(final_opt_arrays, opt_static)
    
#     return final_model, final_opt_state

# key = jax.random.key(10)
# key, subkey = jax.random.split(key)

# ###
# ##
# # from diengmf.models.equinox_rational_quadratic_spline import RationalQuadraticSpline
# # model = RationalQuadraticSpline(input_dim=2, key=key)
# ##
# # model = InvertibleNN(key=key)
# ##
# # from diengmf.models.equinox_masked_coupling_layer import MaskedCouplingLayer
# # model = MaskedCouplingLayer(input_dim=2, key=key)
# ##
# # from diengmf.models.equinox_masked_coupling_layer import MaskedCouplingAffine
# # model = MaskedCouplingAffine(input_dim=2, key=key)
# ##
# ####



# ####
# from diengmf.models import PLULinear
# ###

# from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96


# dynamical_systems = [Ikeda(), Lorenz63(), Lorenz96()]
# for dynamical_system in dynamical_systems:
#     model = PLULinear(input_dim=dynamical_system.dimension, key=key)
#     print(model)
#     optim = optax.chain(
#         optax.adam(
#             learning_rate=1e-4,
#             eps=1e-4,
#         ),
#     )

#     final_model, final_opt_state = training_loop(key, model, dynamical_system, optim)

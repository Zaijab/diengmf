import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from diengmf.models import PLULinear, RQSBijector, MaskedCoupling, NormalizingFlow
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


def test_normalizing_flow_suite():
    key = jax.random.key(0)
    dynamical_systems = [Ikeda()]
    for dynamical_system in dynamical_systems:

        dimension = dynamical_system.dimension

        # model = MaskedCoupling(input_dim=dimension, bijector=RQSBijector(input_dim=dimension, key=key),
                               # conditioner_depth=5, conditioner_hidden_dim=128, key=key, debug=False)

        model = NormalizingFlow(input_dim=dimension,num_layers=10, key=key)
        optim = optax.chain(
            optax.adam(
                learning_rate=1e-4,
                # eps=1e-4,
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

# #Classifier
# @jaxtyped(typechecker=typechecker)
# @eqx.filter_jit
# def attractor_distance_classifier(
#     point: Float[Array, "dim"], 
#     reference_attractor: Float[Array, "n_ref dim"],
#     tolerance: float = 0.1
# ) -> Float[Array, ""]:
#     assert point.shape[-1] == reference_attractor.shape[-1]
#     assert reference_attractor.ndim == 2
#     assert point.ndim == 1
    
#     distances = jnp.linalg.norm(reference_attractor - point[None, :], axis=1)
#     min_distance = jnp.min(distances)
#     assert min_distance.shape == ()
    
#     probability_on_attractor = jnp.exp(-min_distance**2 / (2 * tolerance**2))
#     assert probability_on_attractor.shape == ()
    
#     return probability_on_attractor

# @jaxtyped(typechecker=typechecker) 
# @eqx.filter_jit
# def classification_metrics(
#     probabilities: Float[Array, "batch"],
#     threshold: float = 0.5
# ) -> dict[str, Float[Array, ""]]:
#     assert probabilities.ndim == 1
#     predictions = probabilities > threshold
#     precision = jnp.mean(probabilities[predictions]) if jnp.sum(predictions) > 0 else 0.0
#     recall = jnp.mean(predictions.astype(float))
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#     return {"precision": precision, "recall": recall, "f1_score": f1_score}

# @jaxtyped(typechecker=typechecker)
# def generate_reference_attractor(system: eqx.Module, key: Key[Array, ""], n_points: int = 5000) -> Float[Array, "{n_points} {system.dimension}"]:
#     return system.generate(key, batch_size=n_points, final_time=100.0)

# # Usage in training loop: 
# # reference_attractor = generate_reference_attractor(system, reference_key)
# # classifier_fn = lambda x: attractor_distance_classifier(x, reference_attractor, tolerance=0.1) 
# # probabilities = eqx.filter_vmap(classifier_fn)(generated_batch)
# # metrics = classification_metrics(probabilities)

# ###

# @jaxtyped(typechecker=typechecker)
# @eqx.filter_jit
# def nf_rejection_sample(
#     key: Key[Array, ""], 
#     mean: Float[Array, "dim"], 
#     cov: Float[Array, "dim dim"],
#     nf_model: eqx.Module,
#     threshold: float = -20.0
# ) -> Float[Array, "dim"]:
#     def cond_fn(carry):
#         _, _, log_prob, attempt = carry
#         return (log_prob < threshold) & (attempt < 10)
    
#     def body_fn(carry):
#         key, sample, log_prob, attempt = carry
#         key, subkey = jax.random.split(key)
#         new_sample = jax.random.multivariate_normal(subkey, mean, cov)
#         new_log_prob = nf_model.log_prob(new_sample)
#         return key, new_sample, new_log_prob, attempt + 1
    
#     key, subkey = jax.random.split(key)
#     initial_sample = jax.random.multivariate_normal(subkey, mean, cov)
#     initial_log_prob = nf_model.log_prob(initial_sample)
    
#     _, final_sample, _, _ = jax.lax.while_loop(
#         cond_fn, body_fn, (key, initial_sample, initial_log_prob, 0)
#     )
    
#     return final_sample

# # Usage: sampling_function = jax.tree_util.Partial(nf_rejection_sample, nf_model=trained_nf, threshold=-20.0)

###

test_normalizing_flow_suite()

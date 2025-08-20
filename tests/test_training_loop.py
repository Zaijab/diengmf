import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from diengmf.models import PLULinear, RQSBijector, MaskedCoupling, NormalizingFlow
from diengmf.losses import make_step, kl_divergence
import optax
import matplotlib.pyplot as plt

# checkpoint_dir = os.path.abspath("model_checkpoints")
# os.makedirs(checkpoint_dir, exist_ok=True)


###

# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit 
def compute_debug_metrics(model: eqx.Module, system: eqx.Module, debug_key: Array) -> tuple[Float[Array, ""], Float[Array, "1000 3"]]:
    gen_key, attractor_key, sample_key = jax.random.split(debug_key, 3)
    base_plot_samples = jax.random.multivariate_normal(sample_key, jnp.zeros(system.dimension), jnp.eye(system.dimension), (1000,))
    plot_samples, _ = eqx.filter_vmap(model.forward)(base_plot_samples)
    plot_data = jnp.concatenate([plot_samples[:, :min(3, system.dimension)], jnp.zeros((1000, max(0, 3-system.dimension)))], axis=1)
    return plot_data

def plot_callback(step, score, samples, system_name, dim, dynamical_system):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    if dim == 2:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
        plt.xlabel('x1'); plt.ylabel('x2')
        ax = plt.gca()
    else:
        ax = plt.axes(projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=1)
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
    ax.set_xlim(dynamical_system.plot_limits[0])
    ax.set_ylim(dynamical_system.plot_limits[1])
    if hasattr(ax, 'set_zlim'): ax.set_zlim(dynamical_system.plot_limits[2])
    plt.title(f'Step {step}: {system_name} - Score: {score:.3f}'); plt.show()

def debug_with_plots(carry_data):
    step, loss, model, system, debug_key = carry_data
    samples = compute_debug_metrics(model, system, debug_key)
    jax.debug.print('Step {}: loss={:.4f}', step, loss)
    jax.debug.callback(plot_callback, step, 0, samples, type(system).__name__, system.dimension, system)
    return None

def regular_debug_print(carry_data):
    step, loss = carry_data
    jax.debug.print('Step {}: loss = {}', step, loss)
    return None

def io_operations_callback(step, loss, samples, system_name, dim, dynamical_system):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    if dim == 2:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
        plt.xlabel('x1'); plt.ylabel('x2')
        ax = plt.gca()
    else:
        ax = plt.axes(projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=1)
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
    ax.set_xlim(dynamical_system.plot_limits[0])
    ax.set_ylim(dynamical_system.plot_limits[1])
    if hasattr(ax, 'set_zlim'): ax.set_zlim(dynamical_system.plot_limits[2])
    plt.title(f'Step {step}: {system_name} - Loss: {loss:.3f}'); plt.show()
    # Add model saving logic here

    # Need model, opt_state

    model_path = os.path.join(checkpoint_dir, f"model_step_{step}.eqx")
    opt_path = os.path.join(checkpoint_dir, f"optimizer_step_{step}.eqx")

    eqx.tree_serialise_leaves(model_path, model)
    eqx.tree_serialise_leaves(opt_path, opt_state)


###

@eqx.filter_jit
def training_loop(key: Array, model: eqx.Module, system: eqx.Module, optim: optax.Schedule):
    key, subkey = jax.random.split(key)
    batch = system.generate(subkey, batch_size=1_000, final_time=50.0)
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
        
        debug_key = jax.random.fold_in(key, i)

        jax.lax.cond(
            ((i % 100) == 0),
            lambda _: jax.debug.print('Step {}: loss = {}', i, loss),
            lambda _: None, None
        )

        gen_key, attractor_key, sample_key = jax.random.split(debug_key, 3)
        base_plot_samples = jax.random.multivariate_normal(sample_key, jnp.zeros(system.dimension), jnp.eye(system.dimension), (1000,))
        plot_samples, _ = eqx.filter_vmap(model.forward)(base_plot_samples)
    
        
        jax.lax.cond(
            (i % 500) == 0,
            lambda _: jax.debug.callback(io_operations_callback, i, loss, 
                                        plot_samples, 
                                        type(system).__name__, system.dimension, system),
            lambda _: None, None
        )
        
        # Partition updated objects back to arrays for next iteration
        model_arrays, _ = eqx.partition(model, eqx.is_array)
        opt_arrays, _ = eqx.partition(opt_state, eqx.is_array)
        
        return (batch, model_arrays, opt_arrays, i + 1), loss

    initial_carry = (batch, model_arrays, opt_arrays, 0)
    (final_batch, final_model_arrays, final_opt_arrays, _), losses = jax.lax.scan(
        scan_step, 
        initial_carry, 
        xs=jnp.zeros(20_001)
    )
    
    # Reconstruct final objects
    final_model = eqx.combine(final_model_arrays, model_static)
    final_opt_state = eqx.combine(final_opt_arrays, opt_static)
    
    return final_model, final_opt_state



def test_normalizing_flow_suite():
    key = jax.random.key(0)
    dynamical_systems = [Lorenz63()]
    for dynamical_system in dynamical_systems:
        print(dynamical_system)
        
        dimension = dynamical_system.dimension
        model = NormalizingFlow(input_dim=dimension,
                                num_layers=3, conditioner_hidden_dim=128, conditioner_depth=3,
                                key=key)
        optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=optax.cosine_decay_schedule(
                    init_value=3e-4,
                    decay_steps=5000,
                    alpha=0.03
                ),
                weight_decay=1e-5,
            ),
        )

        final_model, final_opt_state = training_loop(key, model, dynamical_system, optim)

final_model = test_normalizing_flow_suite()

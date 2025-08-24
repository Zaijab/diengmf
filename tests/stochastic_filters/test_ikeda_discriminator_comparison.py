"""
Focused test comparing EnGMF, Normalizing Flow EnGMF, EnKF, and Ikeda's natural discriminator on the Ikeda map.
"""

import jax
import jax.numpy as jnp
import pytest
import optax
import equinox as eqx
from distreqx.distributions import MultivariateNormalDiag

from diengmf.dynamical_systems import Ikeda
from diengmf.measurement_systems import RangeSensor
from diengmf.stochastic_filters import EnGMF, EnKF
from diengmf.stochastic_filters.evaluate import FilterExperiment, evaluate_filter
from diengmf.stochastic_filters.flow_discriminator import flow_discriminator_resampler
from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.losses.invertible_neural_network import kl_divergence, make_step


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, 0, None, None))
def discriminator_resampler(
    key,
    mean,
    cov,
    discriminator_fn,
    max_steps: int = 1000,
):
    """
    Discriminator-based rejection sampling function for EnGMF sampling_function.
    
    Args:
        key: Random key
        mean: Mean of the distribution to sample from
        cov: Covariance matrix
        discriminator_fn: Function that takes x and returns True/False for acceptance
        max_steps: Maximum number of rejection sampling steps
    
    Returns:
        Sample that passes the discriminator
    """
    
    def cond_fn(carry):
        key_state, sample, accepted = carry
        return ~accepted
    
    def body_fn(carry):
        key_state, sample, accepted = carry
        key_state, subkey = jax.random.split(key_state)
        
        candidate = jax.random.multivariate_normal(subkey, mean, cov)
        pass_discriminator = discriminator_fn(candidate)
        
        new_sample = jnp.where(pass_discriminator, candidate, sample)
        new_accepted = accepted | pass_discriminator
        
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


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, 0, None, None))
def batched_discriminator_resampler(
    key,
    mean,
    cov,
    discriminator_fn,
    max_candidates: int = 1000,
):
    """
    Batch-first discriminator-based sampling function for EnGMF sampling_function.
    
    Args:
        key: Random key
        mean: Mean of the distribution to sample from
        cov: Covariance matrix
        discriminator_fn: Function that takes x and returns True/False for acceptance
        max_candidates: Maximum number of candidates to generate upfront
    
    Returns:
        First sample that passes the discriminator (or fallback if none pass)
    """
    # Generate all candidates at once
    candidates = jax.random.multivariate_normal(key, mean, cov, (max_candidates,))
    
    # Apply discriminator to entire batch (single vmap call)
    acceptances = jax.vmap(discriminator_fn)(candidates)
    
    # Find first accepted sample (or use fallback)
    first_accepted_idx = jnp.argmax(acceptances)  # gives 0 if none accepted
    
    return candidates[first_accepted_idx]  # Always valid since argmax gives 0 if none accepted


def normalizing_flow_to_discriminator(model, threshold):
    """
    Convert a normalizing flow model to a boolean discriminator function.
    
    Args:
        model: NormalizingFlow that has a forward method returning (z, log_det_jacobian)
        threshold: Threshold for log probability - return True if log_prob > threshold
    
    Returns:
        Pre-compiled function that takes x and returns True/False based on flow log probability
    """
    @eqx.filter_jit
    def discriminator_fn(x):
        z, log_det_jacobian = model.inverse(x)
        # Compute log probability: log_p(x) = log_p_base(z) + log_det_jacobian
        # Assuming base distribution is standard normal: log_p_base(z) = -0.5 * (z^2 + log(2π))
        log_p_base = -0.5 * (jnp.sum(z**2, axis=-1) + z.shape[-1] * jnp.log(2 * jnp.pi))
        log_prob = log_p_base + log_det_jacobian
        return log_prob > threshold
    
    return discriminator_fn




def train_simple_flow(key, flow, batch, n_batches=500):
    """Train a flow using the existing KL divergence loss."""
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))
    
    for batch_idx in range(n_batches):
        key, batch_key = jax.random.split(key)
        loss, flow, opt_state = make_step(flow, batch, optimizer, opt_state)
        
        if batch_idx % 25 == 0:
            print(f"    Batch {batch_idx}: Loss = {loss:.4f}")
    
    return flow

def test_ikeda_discriminator_comparison(use_normalizing_flow=False):
    """Test filters on the Ikeda map: EnGMF, EnKF, DI-EnGMF (classical), and NF-DI-EnGMF (normalizing flow)."""
    print(f"\n=== Testing Ikeda Discriminator Comparison ===")
    
    key = jax.random.key(42)
    system = Ikeda()
    
    measurement_system = RangeSensor(
        covariance=jnp.array([[4.0]]), 
        center=jnp.array([0.0, 0.0])
    )
    initial_state = jnp.array([0.1, 0.2])
    
    # Create initial belief
    initial_belief = MultivariateNormalDiag(
        loc=initial_state,
        scale_diag=jnp.ones(2) * 0.5
    )
    
    # Create filters with 10 particles as requested
    standard_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=10
    )
    
    enkf = EnKF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=10
    )
    
    
    # Pre-compile the ikeda discriminator
    print("Pre-compiling Ikeda attractor discriminator...")
    dummy_sample = jnp.array([0.1, 0.2])  # Single sample
    _ = system.ikeda_attractor_discriminator(dummy_sample)
    print("Ikeda discriminator pre-compilation complete.")
    
    ikeda_discriminator_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=10,
        sampling_function=jax.tree_util.Partial(
            discriminator_resampler, 
            discriminator_fn=system.ikeda_attractor_discriminator,
            max_steps=5000
        )
    )
    
    filters = {
        "EnGMF": standard_engmf,
        "EnKF": enkf,
        "DI-EnGMF": ikeda_discriminator_engmf,  # Classical Discriminator
    }
    
    # Add normalizing flow discriminator if requested
    if use_normalizing_flow:
        print("Generating training data for normalizing flow...")
        train_key, key = jax.random.split(key)
        training_data = system.generate(key=train_key, final_time=jnp.array(50.0))
        
        print("Training normalizing flow...")
        flow_key, key = jax.random.split(key)
        flow = NormalizingFlow(
            input_dim=2,
            num_layers=10,
            num_bins=8,
            conditioner_hidden_dim=256,
            conditioner_depth=5,
            key=flow_key
        )
        trained_flow = train_simple_flow(key, flow, training_data, n_batches=1_000)
        
        # Create normalizing flow discriminator using our new function
        nf_discriminator_fn = normalizing_flow_to_discriminator(trained_flow, threshold=-5.0)
        
        # Pre-warm the discriminator by calling it once (triggers compilation)
        print("Pre-compiling normalizing flow discriminator...")
        dummy_x = jnp.array([0.1, 0.2])
        _ = nf_discriminator_fn(dummy_x)  # This triggers JIT compilation
        print("Discriminator pre-compilation complete.")
        
        nf_discriminator_engmf = EnGMF(
            dynamical_system=system,
            measurement_system=measurement_system,
            ensemble_size=10,
            sampling_function=jax.tree_util.Partial(
                discriminator_resampler, 
                discriminator_fn=nf_discriminator_fn,
                max_steps=5000
            )
        )
        
        filters["NF-DI-EnGMF"] = nf_discriminator_engmf  # Normalizing Flow Discriminator
    
    # Run filtering experiments
    results = {}
    n_mc_runs = 10
    
    for filter_name, filter_obj in filters.items():
        print(f"\nTesting {filter_name}...")
        
        experiment = FilterExperiment(
            initial_belief=initial_belief,
            dynamical_system=system,
            measurement_system=measurement_system,
            stochastic_filter=filter_obj,
            measurement_times=jnp.array(1.0),  # Use dt=1.0 for discrete system
            burn_in_time=10,
            total_experiment_time=100,
            initial_true_state=initial_state
        )
        
        # Run Monte Carlo evaluations
        eval_key, key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, n_mc_runs)
        
        evaluate_vmap = jax.vmap(evaluate_filter, in_axes=(0, None))
        rmse_values, snees_values = evaluate_vmap(eval_keys, experiment)
        
        results[filter_name] = {
            "avg_rmse": float(jnp.mean(rmse_values)),
            "std_rmse": float(jnp.std(rmse_values)),
            "avg_snees": float(jnp.mean(snees_values)),
            "std_snees": float(jnp.std(snees_values)),
        }
        
        print(f"  RMSE: {results[filter_name]['avg_rmse']:.4f} ± {results[filter_name]['std_rmse']:.4f}")
        print(f"  SNEES: {results[filter_name]['avg_snees']:.4f} ± {results[filter_name]['std_snees']:.4f}")
    
    # Print comparison
    print(f"\n=== Ikeda Discriminator Results ===")
    print("Filter Name               : RMSE          : SNEES")
    print("-" * 55)
    for name, metrics in results.items():
        print(f"{name:25}: {metrics['avg_rmse']:.4f} ± {metrics['std_rmse']:.4f} : {metrics['avg_snees']:.4f} ± {metrics['std_snees']:.4f}")
    
    # Check if DI-EnGMF (classical discriminator) outperforms standard EnGMF
    if "DI-EnGMF" in results and "EnGMF" in results:
        di_rmse = results["DI-EnGMF"]["avg_rmse"]
        standard_rmse = results["EnGMF"]["avg_rmse"]
        improvement = (standard_rmse - di_rmse) / standard_rmse * 100
        print(f"\nDI-EnGMF (Classical Discriminator) RMSE improvement over standard EnGMF: {improvement:.2f}%")
    
    # Check if NF-DI-EnGMF (normalizing flow discriminator) outperforms standard EnGMF
    if "NF-DI-EnGMF" in results and "EnGMF" in results:
        nf_di_rmse = results["NF-DI-EnGMF"]["avg_rmse"]
        standard_rmse = results["EnGMF"]["avg_rmse"]
        improvement = (standard_rmse - nf_di_rmse) / standard_rmse * 100
        print(f"NF-DI-EnGMF (Normalizing Flow Discriminator) RMSE improvement over standard EnGMF: {improvement:.2f}%")
    
    # Compare classical vs normalizing flow discriminators
    if "DI-EnGMF" in results and "NF-DI-EnGMF" in results:
        di_rmse = results["DI-EnGMF"]["avg_rmse"]
        nf_di_rmse = results["NF-DI-EnGMF"]["avg_rmse"]
        if di_rmse < nf_di_rmse:
            improvement = (nf_di_rmse - di_rmse) / nf_di_rmse * 100
            print(f"Classical discriminator outperforms NF discriminator by {improvement:.2f}%")
        else:
            improvement = (di_rmse - nf_di_rmse) / di_rmse * 100
            print(f"NF discriminator outperforms classical discriminator by {improvement:.2f}%")
    
    # Basic assertions
    for name, metrics in results.items():
        assert metrics["avg_rmse"] > 0, f"{name} RMSE should be positive"
        assert jnp.isfinite(metrics["avg_rmse"]), f"{name} RMSE should be finite"
        assert metrics["avg_snees"] > 0, f"{name} SNEES should be positive"
        assert jnp.isfinite(metrics["avg_snees"]), f"{name} SNEES should be finite"


def test_ikeda_discriminator_without_normalizing_flow():
    """Test without the normalizing flow to focus on the core comparison."""
    test_ikeda_discriminator_comparison(use_normalizing_flow=False)


def test_ikeda_batched_discriminator_comparison(use_normalizing_flow=True):
    """Test filters on the Ikeda map using batched discriminator approach."""
    print(f"\n=== Testing Ikeda Batched Discriminator Comparison ===")
    
    key = jax.random.key(42)
    system = Ikeda()
    
    measurement_system = RangeSensor(
        covariance=jnp.array([[4.0]]), 
        center=jnp.array([0.0, 0.0])
    )
    initial_state = jnp.array([0.1, 0.2])
    
    # Create initial belief
    initial_belief = MultivariateNormalDiag(
        loc=initial_state,
        scale_diag=jnp.ones(2) * 0.5
    )
    
    # Create filters
    standard_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=10
    )
    
    enkf = EnKF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=10
    )
    
    # Pre-compile the ikeda discriminator
    print("Pre-compiling Ikeda attractor discriminator...")
    dummy_sample = jnp.array([0.1, 0.2])
    _ = system.ikeda_attractor_discriminator(dummy_sample)
    print("Ikeda discriminator pre-compilation complete.")
    
    # Use batched discriminator resampler
    batched_ikeda_discriminator_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=10,
        sampling_function=jax.tree_util.Partial(
            batched_discriminator_resampler, 
            discriminator_fn=system.ikeda_attractor_discriminator,
            max_candidates=5000
        )
    )
    
    filters = {
        "EnGMF": standard_engmf,
        "EnKF": enkf,
        "Batched-DI-EnGMF": batched_ikeda_discriminator_engmf,
    }
    
    # Add normalizing flow discriminator if requested
    if use_normalizing_flow:
        print("Generating training data for normalizing flow...")
        train_key, key = jax.random.split(key)
        training_data = system.generate(key=train_key, final_time=jnp.array(50.0))
        
        print("Training normalizing flow...")
        flow_key, key = jax.random.split(key)
        flow = NormalizingFlow(
            input_dim=2,
            num_layers=8,
            num_bins=8,
            conditioner_hidden_dim=64,
            conditioner_depth=3,
            key=flow_key
        )
        trained_flow = train_simple_flow(key, flow, training_data, n_batches=1_000)
        
        # Create normalizing flow discriminator
        nf_discriminator_fn = normalizing_flow_to_discriminator(trained_flow, threshold=0.01)
        
        # Pre-warm the discriminator
        print("Pre-compiling normalizing flow discriminator...")
        dummy_x = jnp.array([0.1, 0.2])
        _ = nf_discriminator_fn(dummy_x)
        print("Discriminator pre-compilation complete.")
        
        batched_nf_discriminator_engmf = EnGMF(
            dynamical_system=system,
            measurement_system=measurement_system,
            ensemble_size=10,
            sampling_function=jax.tree_util.Partial(
                batched_discriminator_resampler, 
                discriminator_fn=nf_discriminator_fn,
                max_candidates=5000
            )
        )
        
        filters["Batched-NF-DI-EnGMF"] = batched_nf_discriminator_engmf
    
    # Run filtering experiments
    results = {}
    n_mc_runs = 5  # Reduced for faster testing
    
    for filter_name, filter_obj in filters.items():
        print(f"\nTesting {filter_name}...")
        
        experiment = FilterExperiment(
            initial_belief=initial_belief,
            dynamical_system=system,
            measurement_system=measurement_system,
            stochastic_filter=filter_obj,
            measurement_times=jnp.array(1.0),
            burn_in_time=10,
            total_experiment_time=50,  # Reduced for faster testing
            initial_true_state=initial_state
        )
        
        # Run Monte Carlo evaluations
        eval_key, key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, n_mc_runs)
        
        evaluate_vmap = jax.vmap(evaluate_filter, in_axes=(0, None))
        rmse_values, snees_values = evaluate_vmap(eval_keys, experiment)
        
        results[filter_name] = {
            "avg_rmse": float(jnp.mean(rmse_values)),
            "std_rmse": float(jnp.std(rmse_values)),
            "avg_snees": float(jnp.mean(snees_values)),
            "std_snees": float(jnp.std(snees_values)),
        }
        
        print(f"  RMSE: {results[filter_name]['avg_rmse']:.4f} ± {results[filter_name]['std_rmse']:.4f}")
        print(f"  SNEES: {results[filter_name]['avg_snees']:.4f} ± {results[filter_name]['std_snees']:.4f}")
    
    # Print comparison
    print(f"\n=== Batched Ikeda Discriminator Results ===")
    print("Filter Name               : RMSE          : SNEES")
    print("-" * 55)
    for name, metrics in results.items():
        print(f"{name:25}: {metrics['avg_rmse']:.4f} ± {metrics['std_rmse']:.4f} : {metrics['avg_snees']:.4f} ± {metrics['std_snees']:.4f}")
    
    # Basic assertions
    for name, metrics in results.items():
        assert metrics["avg_rmse"] > 0, f"{name} RMSE should be positive"
        assert jnp.isfinite(metrics["avg_rmse"]), f"{name} RMSE should be finite"
        assert metrics["avg_snees"] > 0, f"{name} SNEES should be positive"
        assert jnp.isfinite(metrics["avg_snees"]), f"{name} SNEES should be finite"


def test_ikeda_batched_discriminator_without_normalizing_flow():
    """Test batched discriminator without the normalizing flow to focus on performance."""
    test_ikeda_batched_discriminator_comparison(use_normalizing_flow=False)


def test_batched_discriminator_resampler_functionality():
    """Test that the batched discriminator resampler works correctly."""
    key = jax.random.key(123)
    system = Ikeda()
    
    # Test with a batch of samples (since batched_discriminator_resampler has vmap)
    batch_size = 5
    keys = jax.random.split(key, batch_size)
    means = jnp.tile(jnp.array([0.1, 0.2]), (batch_size, 1))
    covs = jnp.tile(0.5 * jnp.eye(2), (batch_size, 1, 1))
    
    samples = batched_discriminator_resampler(
        keys, means, covs, system.ikeda_attractor_discriminator, max_candidates=20
    )
    
    # Check that samples pass the discriminator (vmap over each sample)
    on_attractor = jax.vmap(system.ikeda_attractor_discriminator)(samples)
    
    print(f"Samples shape: {samples.shape}")
    print(f"Samples: {samples}")
    print(f"On attractor: {on_attractor}")
    
    assert samples.shape == (batch_size, 2), f"Samples should be ({batch_size}, 2)"
    assert jnp.isfinite(samples).all(), "Samples should be finite"


if __name__ == "__main__":
    test_discriminator_resampler_functionality()
    test_ikeda_discriminator_comparison()

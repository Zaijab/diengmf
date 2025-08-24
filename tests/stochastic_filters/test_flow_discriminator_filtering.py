"""
Test flow discriminator resampling in filtering scenarios.
"""

import jax
import jax.numpy as jnp
import pytest
import optax
import equinox as eqx
from distreqx.distributions import MultivariateNormalDiag

from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from diengmf.measurement_systems import RangeSensor, AdjacentPairMeasurement
from diengmf.stochastic_filters import EnGMF, EnKF
from diengmf.stochastic_filters.evaluate import FilterExperiment, evaluate_filter
from diengmf.stochastic_filters.flow_discriminator import flow_discriminator_resampler
from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.losses.invertible_neural_network import kl_divergence, make_step


def train_simple_flow(key, flow, training_data, n_batches=100):
    """Train a flow using the existing KL divergence loss."""
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))
    
    n_data = training_data.shape[0]
    batch_size = 2 ** 9
    
    for batch_idx in range(n_batches):
        key, batch_key = jax.random.split(key)
        batch_indices = jax.random.choice(batch_key, n_data, (batch_size,), replace=False)
        batch = training_data[batch_indices]
        
        loss, flow, opt_state = make_step(flow, batch, optimizer, opt_state)
        
        if batch_idx % 25 == 0:
            print(f"    Batch {batch_idx}: Loss = {loss:.4f}")
    
    return flow


def generate_trajectory_data(key, system, initial_state, n_trajectories=200, seq_length=50, dt=0.1):
    """Generate trajectory data for training."""
    keys = jax.random.split(key, n_trajectories)
    
    def single_trajectory(traj_key):
        current_state = initial_state
        trajectory = []
        for _ in range(seq_length):
            next_state = system.flow(jnp.array(0.0), jnp.array(dt), current_state)
            trajectory.append(next_state)
            current_state = next_state
        return jnp.array(trajectory)
    
    trajectories = jax.vmap(single_trajectory)(keys)
    return trajectories.reshape(-1, trajectories.shape[-1])


@pytest.mark.parametrize("system_name,system,measurement_system,initial_state,measurement_times", [
    (
        "Ikeda", 
        Ikeda(), 
        RangeSensor(covariance=jnp.array([[0.1]]), center=jnp.array([0.0, 0.0])),
        jnp.array([0.1, 0.2]),
        jnp.array(0.1)
    ),
    (
        "Lorenz63", 
        Lorenz63(), 
        RangeSensor(covariance=jnp.array([[0.1]]), center=jnp.array([6 * jnp.sqrt(2), 6 * jnp.sqrt(2), 27.0])),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.array(0.01)
    ),
])
def test_flow_discriminator_comparison(system_name, system, measurement_system, initial_state, measurement_times):
    """Test flow discriminator enhanced filtering vs standard methods."""
    print(f"\n=== Testing {system_name} ===")
    
    key = jax.random.key(42)
    state_dim = initial_state.shape[0]
    
    # Generate training data
    print("Generating training data...")
    train_key, key = jax.random.split(key)
    training_data = generate_trajectory_data(train_key, system, initial_state)
    
    # Initialize and train flow
    print("Training flow discriminator...")
    flow_key, key = jax.random.split(key)
    flow = NormalizingFlow(
        input_dim=state_dim,
        num_layers=10,
        num_bins=8,
        conditioner_hidden_dim=128,
        conditioner_depth=5,
        key=flow_key
    )
    
    trained_flow = train_simple_flow(key, flow, training_data, n_batches=1000)
    
    # Create initial belief
    initial_belief = MultivariateNormalDiag(
        loc=initial_state,
        scale_diag=jnp.ones(state_dim)
    )
    
    # Create filters
    standard_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=5
    )
    
    discriminator_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=5,
        sampling_function=jax.tree_util.Partial(
            flow_discriminator_resampler, 
            model=trained_flow,
            threshold=0.05,
            max_steps=1000
        )
    )
    
    enkf = EnKF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=5
    )
    
    filters = {
        "EnGMF_Standard": standard_engmf,
        "EnGMF_Discriminator": discriminator_engmf,
        "EnKF": enkf
    }
    
    # Run filtering experiments
    results = {}
    n_mc_runs = 5
    
    for filter_name, filter_obj in filters.items():
        print(f"\nTesting {filter_name}...")
        
        experiment = FilterExperiment(
            initial_belief=initial_belief,
            dynamical_system=system,
            measurement_system=measurement_system,
            stochastic_filter=filter_obj,
            measurement_times=measurement_times,
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
    print(f"\n=== {system_name} Results ===")
    for name, metrics in results.items():
        print(f"{name:20}: RMSE = {metrics['avg_rmse']:.4f} ± {metrics['std_rmse']:.4f}")
    
    # Basic assertions - all methods should produce finite, positive RMSE
    for name, metrics in results.items():
        assert metrics["avg_rmse"] > 0, f"{name} RMSE should be positive"
        assert jnp.isfinite(metrics["avg_rmse"]), f"{name} RMSE should be finite"
        assert metrics["avg_snees"] > 0, f"{name} SNEES should be positive"
        assert jnp.isfinite(metrics["avg_snees"]), f"{name} SNEES should be finite"


def test_lorenz96_flow_discriminator():
    """Test flow discriminator on Lorenz96 with adjacent pair measurement."""
    print(f"\n=== Testing Lorenz96 ===")
    
    key = jax.random.key(42)
    system = Lorenz96(dim=40)
    measurement_system = AdjacentPairMeasurement(covariance=0.1 * jnp.eye(20))
    initial_state = jnp.ones(40) + 0.1 * jax.random.normal(jax.random.key(123), (40,))
    
    # Generate training data
    print("Generating training data...")
    train_key, key = jax.random.split(key)
    training_data = generate_trajectory_data(train_key, system, initial_state, n_trajectories=100)
    
    # Initialize and train flow
    print("Training flow discriminator...")
    flow_key, key = jax.random.split(key)
    flow = NormalizingFlow(
        input_dim=40,
        num_layers=6,
        num_bins=8,
        conditioner_hidden_dim=128,
        conditioner_depth=3,
        key=flow_key
    )
    
    trained_flow = train_simple_flow(key, flow, training_data, n_batches=50)
    
    # Create discriminator filter
    discriminator_engmf = EnGMF(
        dynamical_system=system,
        measurement_system=measurement_system,
        ensemble_size=100,
        sampling_function=jax.tree_util.Partial(
            flow_discriminator_resampler, 
            model=trained_flow,
            threshold=-5.0,  # Lower threshold for high-dim case
            max_steps=100
        )
    )
    
    # Test single run
    initial_belief = MultivariateNormalDiag(
        loc=jnp.ones(40),
        scale_diag=jnp.ones(40)
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=system,
        measurement_system=measurement_system,
        stochastic_filter=discriminator_engmf,
        measurement_times=jnp.array(0.05),
        burn_in_time=10,
        total_experiment_time=50,  # Shorter for computational efficiency
        initial_true_state=initial_state
    )
    
    eval_key, key = jax.random.split(key)
    rmse, snees = evaluate_filter(eval_key, experiment)
    
    print(f"Lorenz96 Discriminator RMSE: {rmse:.4f}, SNEES: {snees:.4f}")
    
    # Assertions
    assert rmse > 0 and jnp.isfinite(rmse), "RMSE should be positive and finite"
    assert snees > 0 and jnp.isfinite(snees), "SNEES should be positive and finite"

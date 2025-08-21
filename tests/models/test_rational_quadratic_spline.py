import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# Assume imports from your project
from diengmf.models.rational_quadratic_spline import RQSBijector
from diengmf.losses import make_step, kl_divergence
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96

# Comprehensive hyperparameter grid for optimization
RQS_HYPERPARAMETER_GRID = [
    # Standard configurations
    {
        "num_bins": 4,
        "range_min": -3.0,
        "range_max": 3.0,
        "min_bin_size": 1e-3,
        "min_knot_slope": 1e-3,
        "param_init": jax.nn.initializers.normal(stddev=0.01)
    },
    {
        "num_bins": 8,
        "range_min": -5.0,
        "range_max": 5.0,
        "min_bin_size": 1e-3,
        "min_knot_slope": 1e-3,
        "param_init": jax.nn.initializers.normal(stddev=0.1)
    },
    {
        "num_bins": 16,
        "range_min": -10.0,
        "range_max": 10.0,
        "min_bin_size": 1e-4,
        "min_knot_slope": 1e-4,
        "param_init": jax.nn.initializers.xavier_normal()
    },
    # Narrow range configurations
    {
        "num_bins": 8,
        "range_min": -2.0,
        "range_max": 2.0,
        "min_bin_size": 1e-3,
        "min_knot_slope": 1e-3,
        "param_init": jax.nn.initializers.uniform(scale=0.02)
    },
    # Wide range configurations
    {
        "num_bins": 12,
        "range_min": -15.0,
        "range_max": 15.0,
        "min_bin_size": 1e-2,
        "min_knot_slope": 1e-2,
        "param_init": jax.nn.initializers.he_normal()
    },
    # Different initialization strategies
    {
        "num_bins": 8,
        "range_min": -5.0,
        "range_max": 5.0,
        "min_bin_size": 1e-3,
        "min_knot_slope": 1e-3,
        "param_init": jax.nn.initializers.truncated_normal(stddev=0.05)
    },
    {
        "num_bins": 6,
        "range_min": -4.0,
        "range_max": 4.0,
        "min_bin_size": 5e-4,
        "min_knot_slope": 5e-4,
        "param_init": jax.nn.initializers.xavier_uniform()
    },
    # High precision configurations
    {
        "num_bins": 10,
        "range_min": -8.0,
        "range_max": 8.0,
        "min_bin_size": 1e-5,
        "min_knot_slope": 1e-5,
        "param_init": jax.nn.initializers.normal(stddev=0.001)
    },
    # Conservative configurations
    {
        "num_bins": 4,
        "range_min": -1.0,
        "range_max": 1.0,
        "min_bin_size": 1e-2,
        "min_knot_slope": 1e-2,
        "param_init": jax.nn.initializers.zeros
    },
    # Extreme configurations (for robustness testing)
    {
        "num_bins": 20,
        "range_min": -20.0,
        "range_max": 20.0,
        "min_bin_size": 1e-6,
        "min_knot_slope": 1e-6,
        "param_init": jax.nn.initializers.normal(stddev=0.5)
    }
]

DYNAMICAL_SYSTEMS = [
    ("Ikeda", lambda: Ikeda()),
    ("Lorenz63", lambda: Lorenz63()),
    ("Lorenz96", lambda: Lorenz96())
]

@pytest.fixture
def test_keys():
    return jax.random.split(jax.random.key(42), 10)

@pytest.mark.parametrize("config", RQS_HYPERPARAMETER_GRID)
@pytest.mark.parametrize("system_name,system_factory", DYNAMICAL_SYSTEMS)
def test_rqs_hyperparameter_invertibility(config, system_name, system_factory, test_keys):
    """Test invertibility across different hyperparameter configurations and dynamical systems."""
    system = system_factory()
    key = test_keys[0]
    
    # Create RQS with configuration
    rqs = RQSBijector(
        input_dim=system.dimension,
        key=key,
        **config
    )
    
    # Test single point invertibility
    x_single = jax.random.normal(test_keys[1], (system.dimension,))
    y, fwd_logdet = rqs.forward(x_single)
    x_recon, inv_logdet = rqs.inverse(y)
    
    assert jnp.allclose(x_single, x_recon, atol=1e-5), f"Failed for {system_name} with config {config}"
    assert jnp.allclose(fwd_logdet + inv_logdet, 0.0, atol=1e-5), f"Logdet inconsistent for {system_name}"

@pytest.mark.parametrize("config", RQS_HYPERPARAMETER_GRID[:5])  # Test subset for training
@pytest.mark.parametrize("system_name,system_factory", DYNAMICAL_SYSTEMS[:2])  # Faster systems
def test_rqs_hyperparameter_training_stability(config, system_name, system_factory, test_keys):
    """Test training stability across different configurations."""
    system = system_factory()
    key = test_keys[2]
    
    rqs = RQSBijector(
        input_dim=system.dimension,
        key=key,
        **config
    )
    
    # Generate training data
    batch = system.generate(test_keys[3], batch_size=50, final_time=jnp.asarray(5.0))
    
    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(eqx.filter(rqs, eqx.is_inexact_array))
    
    initial_loss = None
    # Train for several steps
    for i in range(20):
        batch = eqx.filter_vmap(system.flow, in_axes=(None, None, 0))(jnp.asarray(0.0), jnp.asarray(1.0), batch)
        loss, rqs, opt_state = make_step(rqs, batch, optim, opt_state)
        
        if i == 0:
            initial_loss = loss
        
        # Check for NaN/Inf
        assert jnp.isfinite(loss), f"Loss became non-finite at step {i} for {system_name}"
        assert not jnp.isnan(loss), f"Loss became NaN at step {i} for {system_name}"
    
    # Test invertibility is maintained after training
    x_test = jax.random.normal(test_keys[4], (system.dimension,))
    y, fwd_logdet = rqs.forward(x_test)
    x_recon, inv_logdet = rqs.inverse(y)
    
    assert jnp.allclose(x_test, x_recon, atol=1e-4), f"Training broke invertibility for {system_name}"

def test_rqs_parameter_shapes_across_configs():
    """Test parameter shapes are correct across all configurations."""
    key = jax.random.key(123)
    
    for i, config in enumerate(RQS_HYPERPARAMETER_GRID):
        test_key = jax.random.fold_in(key, i)
        
        rqs = RQSBijector(
            input_dim=2,  # Fixed for this test
            key=test_key,
            **config
        )
        
        expected_param_shape = (2, 3 * config["num_bins"] + 1)
        assert rqs.params.shape == expected_param_shape
        assert jnp.isfinite(rqs.params).all()
        assert not jnp.isnan(rqs.params).any()

# Generate Optuna-style configuration helper
def suggest_rqs_config(trial):
    """Helper function for Optuna hyperparameter optimization."""
    return {
        "num_bins": trial.suggest_categorical("num_bins", [4, 6, 8, 10, 12, 16, 20]),
        "range_min": -trial.suggest_float("range_magnitude", 1.0, 20.0, log=True),
        "range_max": trial.suggest_float("range_magnitude", 1.0, 20.0, log=True),
        "min_bin_size": trial.suggest_float("min_bin_size", 1e-6, 1e-2, log=True),
        "min_knot_slope": trial.suggest_float("min_knot_slope", 1e-6, 1e-2, log=True),
        "param_init": trial.suggest_categorical("param_init", [
            jax.nn.initializers.normal(stddev=0.01),
            jax.nn.initializers.normal(stddev=0.1),
            jax.nn.initializers.xavier_normal(),
            jax.nn.initializers.he_normal(),
            jax.nn.initializers.uniform(scale=0.02)
        ])
    }

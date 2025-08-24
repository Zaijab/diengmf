"""Test suite for evaluating different stochastic filters on various dynamical systems."""

import jax
import jax.numpy as jnp
import pytest
from distreqx.distributions import MultivariateNormalDiag

from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from diengmf.measurement_systems import RangeSensor, AdjacentPairMeasurement
from diengmf.stochastic_filters import EnGMF, EnKF
from diengmf.stochastic_filters.evaluate import FilterExperiment, evaluate_filter


@pytest.fixture
def base_key():
    """Base random key for reproducible tests."""
    return jax.random.key(42)


@pytest.fixture
def n_mc_iterations():
    """Number of Monte Carlo iterations for testing."""
    return 5


@pytest.fixture
def covariances():
    """Test covariance scales."""
    return [0.25, 1.0, 4.0]


@pytest.fixture
def ikeda_system():
    """Ikeda dynamical system."""
    return Ikeda()


@pytest.fixture
def lorenz63_system():
    """Lorenz63 dynamical system."""
    return Lorenz63()


@pytest.fixture
def lorenz96_system():
    """Lorenz96 dynamical system with 40 oscillators."""
    return Lorenz96(dim=40)


@pytest.fixture
def range_sensor_origin():
    """Range sensor at origin."""
    return RangeSensor(
        covariance=jnp.array([[0.1]]), 
        center=jnp.array([0.0, 0.0])
    )


@pytest.fixture
def range_sensor_lorenz():
    """Range sensor at Lorenz63 fixed point."""
    return RangeSensor(
        covariance=jnp.array([[0.1]]),
        center=jnp.array([6 * jnp.sqrt(2), 6 * jnp.sqrt(2), 27.0])
    )


@pytest.fixture
def adjacent_measurement():
    """Adjacent pair measurement for Lorenz96."""
    return AdjacentPairMeasurement(
        covariance=0.1 * jnp.eye(20)  # 20 measurement pairs
    )


def run_monte_carlo_evaluation(key, experiment, n_iterations):
    """Run Monte Carlo evaluation using vmap over keys."""
    # Generate keys for all iterations
    keys = jax.random.split(key, n_iterations)
    
    # Vectorize evaluate_filter over keys
    evaluate_filter_vmap = jax.vmap(evaluate_filter, in_axes=(0, None))
    
    # Run all evaluations in parallel
    rmse_values, snees_values = evaluate_filter_vmap(keys, experiment)
    
    return rmse_values, snees_values


@pytest.mark.parametrize("filter_class", [EnGMF, EnKF])
@pytest.mark.parametrize("cov_scale", [0.25, 1.0, 4.0])
def test_ikeda_filter_evaluation(base_key, n_mc_iterations, filter_class, cov_scale, ikeda_system, range_sensor_origin):
    """Test filter evaluation on Ikeda map with range sensor at origin."""
    # Create initial belief with scaled covariance
    initial_cov = cov_scale * jnp.eye(2)
    initial_belief = MultivariateNormalDiag(
        loc=jnp.array([0.5, 0.5]), 
        scale_diag=jnp.sqrt(jnp.diag(initial_cov))
    )
    
    # Create filter
    stochastic_filter = filter_class(
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        ensemble_size=50
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        stochastic_filter=stochastic_filter,
        measurement_times=jnp.array(0.1),
        burn_in_time=10,
        total_experiment_time=100,
        initial_true_state=jnp.array([0.1, 0.2])
    )
    
    # Run Monte Carlo iterations using vmap
    rmse_values, snees_values = run_monte_carlo_evaluation(base_key, experiment, n_mc_iterations)
    
    # Compute statistics
    avg_rmse = jnp.mean(rmse_values)
    avg_snees = jnp.mean(snees_values)
    
    # Verify results are reasonable
    assert avg_rmse > 0, "RMSE should be positive"
    assert avg_snees > 0, "SNEES should be positive"
    assert jnp.isfinite(avg_rmse), "RMSE should be finite"
    assert jnp.isfinite(avg_snees), "SNEES should be finite"


@pytest.mark.parametrize("filter_class", [EnGMF, EnKF])
@pytest.mark.parametrize("cov_scale", [0.25, 1.0, 4.0])
def test_lorenz63_filter_evaluation(base_key, n_mc_iterations, filter_class, cov_scale, lorenz63_system, range_sensor_lorenz):
    """Test filter evaluation on Lorenz63 with range sensor."""
    # Create initial belief with scaled covariance
    initial_cov = cov_scale * jnp.eye(3)
    initial_belief = MultivariateNormalDiag(
        loc=jnp.array([1.0, 1.0, 1.0]), 
        scale_diag=jnp.sqrt(jnp.diag(initial_cov))
    )
    
    # Create filter
    stochastic_filter = filter_class(
        dynamical_system=lorenz63_system,
        measurement_system=range_sensor_lorenz,
        ensemble_size=50
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=lorenz63_system,
        measurement_system=range_sensor_lorenz,
        stochastic_filter=stochastic_filter,
        measurement_times=jnp.array(0.01),
        burn_in_time=10,
        total_experiment_time=100,
        initial_true_state=jnp.array([1.0, 1.0, 1.0])
    )
    
    # Run Monte Carlo iterations using vmap
    rmse_values, snees_values = run_monte_carlo_evaluation(base_key, experiment, n_mc_iterations)
    
    # Compute statistics
    avg_rmse = jnp.mean(rmse_values)
    avg_snees = jnp.mean(snees_values)
    
    # Verify results are reasonable
    assert avg_rmse > 0, "RMSE should be positive"
    assert avg_snees > 0, "SNEES should be positive"
    assert jnp.isfinite(avg_rmse), "RMSE should be finite"
    assert jnp.isfinite(avg_snees), "SNEES should be finite"


@pytest.mark.parametrize("filter_class", [EnGMF, EnKF])
@pytest.mark.parametrize("cov_scale", [0.25, 1.0, 4.0])
def test_lorenz96_filter_evaluation(base_key, n_mc_iterations, filter_class, cov_scale, lorenz96_system, adjacent_measurement):
    """Test filter evaluation on Lorenz96 with adjacent pair measurement."""
    # Create initial belief with scaled covariance
    initial_cov = cov_scale * jnp.eye(40)  # 40-dimensional state
    initial_belief = MultivariateNormalDiag(
        loc=jnp.ones(40), 
        scale_diag=jnp.sqrt(jnp.diag(initial_cov))
    )
    
    # Create filter
    stochastic_filter = filter_class(
        dynamical_system=lorenz96_system,
        measurement_system=adjacent_measurement,
        ensemble_size=100  # Larger ensemble for higher dimensional system
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=lorenz96_system,
        measurement_system=adjacent_measurement,
        stochastic_filter=stochastic_filter,
        measurement_times=jnp.array(0.05),
        burn_in_time=10,
        total_experiment_time=100,
        initial_true_state=jnp.ones(40) + 0.1 * jax.random.normal(jax.random.key(123), (40,))
    )
    
    # Run Monte Carlo iterations using vmap
    rmse_values, snees_values = run_monte_carlo_evaluation(base_key, experiment, n_mc_iterations)
    
    # Compute statistics
    avg_rmse = jnp.mean(rmse_values)
    avg_snees = jnp.mean(snees_values)
    
    # Verify results are reasonable
    assert avg_rmse > 0, "RMSE should be positive"
    assert avg_snees > 0, "SNEES should be positive"
    assert jnp.isfinite(avg_rmse), "RMSE should be finite"
    assert jnp.isfinite(avg_snees), "SNEES should be finite"


@pytest.mark.parametrize("filter_class", [EnGMF, EnKF])
def test_filter_experiment_initialization(filter_class, ikeda_system, range_sensor_origin):
    """Test that FilterExperiment can be properly initialized."""
    initial_belief = MultivariateNormalDiag(
        loc=jnp.array([0.5, 0.5]), 
        scale_diag=jnp.array([1.0, 1.0])
    )
    
    stochastic_filter = filter_class(
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        ensemble_size=50
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        stochastic_filter=stochastic_filter,
        measurement_times=jnp.array(0.1),
        burn_in_time=10,
        total_experiment_time=100,
        initial_true_state=jnp.array([0.1, 0.2])
    )
    
    assert experiment.initial_belief is not None
    assert experiment.dynamical_system is not None
    assert experiment.measurement_system is not None
    assert experiment.stochastic_filter is not None


@pytest.mark.parametrize("filter_class", [EnGMF, EnKF])
def test_evaluate_filter_single_run(base_key, filter_class, ikeda_system, range_sensor_origin):
    """Test a single evaluation run returns reasonable values."""
    initial_belief = MultivariateNormalDiag(
        loc=jnp.array([0.5, 0.5]), 
        scale_diag=jnp.array([1.0, 1.0])
    )
    
    stochastic_filter = filter_class(
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        ensemble_size=50
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        stochastic_filter=stochastic_filter,
        measurement_times=jnp.array(0.1),
        burn_in_time=10,
        total_experiment_time=100,
        initial_true_state=jnp.array([0.1, 0.2])
    )
    
    rmse, snees = evaluate_filter(base_key, experiment)
    
    assert rmse > 0, "RMSE should be positive"
    assert snees > 0, "SNEES should be positive"
    assert jnp.isfinite(rmse), "RMSE should be finite"
    assert jnp.isfinite(snees), "SNEES should be finite"


@pytest.mark.parametrize("filter_class", [EnGMF, EnKF])
def test_monte_carlo_evaluation_vmap(base_key, filter_class, ikeda_system, range_sensor_origin):
    """Test that vmap-based Monte Carlo evaluation works correctly."""
    initial_belief = MultivariateNormalDiag(
        loc=jnp.array([0.5, 0.5]), 
        scale_diag=jnp.array([1.0, 1.0])
    )
    
    stochastic_filter = filter_class(
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        ensemble_size=50
    )
    
    experiment = FilterExperiment(
        initial_belief=initial_belief,
        dynamical_system=ikeda_system,
        measurement_system=range_sensor_origin,
        stochastic_filter=stochastic_filter,
        measurement_times=jnp.array(0.1),
        burn_in_time=10,
        total_experiment_time=100,
        initial_true_state=jnp.array([0.1, 0.2])
    )
    
    n_iterations = 3
    rmse_values, snees_values = run_monte_carlo_evaluation(base_key, experiment, n_iterations)
    
    # Check shapes
    assert rmse_values.shape == (n_iterations,)
    assert snees_values.shape == (n_iterations,)
    
    # Check all values are reasonable
    assert jnp.all(rmse_values > 0), "All RMSE values should be positive"
    assert jnp.all(snees_values > 0), "All SNEES values should be positive"
    assert jnp.all(jnp.isfinite(rmse_values)), "All RMSE values should be finite"
    assert jnp.all(jnp.isfinite(snees_values)), "All SNEES values should be finite"
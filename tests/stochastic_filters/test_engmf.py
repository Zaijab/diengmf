import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.distributions import AbstractDistribution, MultivariateNormalTri
from jaxtyping import Array, Float, Int, Key

from diengmf.dynamical_systems import (AbstractDynamicalSystem, Ikeda,
                                       Lorenz63, Lorenz96)
from diengmf.measurement_systems import AbstractMeasurementSystem, RangeSensor
from diengmf.stochastic_filters import AbstractFilter, EnGMF
from diengmf.stochastic_filters.evaluate import FilterExperiment, evaluate_filter

print(uuid.uuid4())


# FilterExperiment is now imported from diengmf.stochastic_filters.evaluate


### Ikeda Experiments

initial_state = jnp.array([1.25, 0.0])
initial_covariance = 1 / 128 * jnp.eye(initial_state.shape[0])
measurement_covariance = jnp.array([[0.25]])
dynamical_system = Ikeda()
measurement_system = RangeSensor(
    covariance=measurement_covariance, center=jnp.array([0.0, 0.0])
)
burn_in_time = 10
total_experiment_time = 100 * burn_in_time

experiment_description = FilterExperiment(
    initial_belief=MultivariateNormalTri(initial_state, initial_covariance),
    dynamical_system=dynamical_system,
    measurement_system=measurement_system,
    stochastic_filter=EnGMF(
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        ensemble_size=1000,
    ),
    measurement_times=jnp.asarray(1.0),
    burn_in_time=burn_in_time,
    total_experiment_time=total_experiment_time,
    initial_true_state=initial_state,
)


# evaluate_filter is now imported from diengmf.stochastic_filters.evaluate
# It works with single keys and returns (rmse, snees) tuple


def test_engmf_filter():
    """Test the EnGMF filter using the new evaluation framework."""
    key = jax.random.key(10)
    key, subkey = jax.random.split(key)

    # Test single evaluation
    rmse, snees = evaluate_filter(subkey, experiment_description)
    print(f"Single run - RMSE: {rmse}, SNEES: {snees}")
    
    # Test multiple Monte Carlo runs using vmap
    num_mc_iterations = 5
    vmapped_evaluate = eqx.filter_vmap(evaluate_filter, in_axes=(0, None))
    
    keys = jax.random.split(subkey, num_mc_iterations)
    rmse_array, snees_array = vmapped_evaluate(keys, experiment_description)
    
    mean_rmse = jnp.mean(rmse_array)
    mean_snees = jnp.mean(snees_array)
    
    print(f"Monte Carlo ({num_mc_iterations} runs) - Mean RMSE: {mean_rmse}, Mean SNEES: {mean_snees}")
    
    # Basic assertions to ensure reasonable results
    assert rmse > 0.0, "RMSE should be positive"
    assert snees > 0.0, "SNEES should be positive"
    assert mean_rmse > 0.0, "Mean RMSE should be positive"
    assert mean_snees > 0.0, "Mean SNEES should be positive"


if __name__ == "__main__":
    test_engmf_filter()

###

# covariances = [jnp.array([[0.25]]), jnp.array([[1.0]]), jnp.array([[4.0]])]
# experiments_list = [eqx.tree_at(lambda x: x.measurement_system.covariance,
#                                 experiment_description, cov) for cov in covariances]
# experiment_descriptions = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *experiments_list)


# This function is no longer needed - using evaluate_filter from main package



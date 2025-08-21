import equinox as eqx
import jax
import jax.numpy as jnp
from diengmf.dynamical_systems import AbstractDynamicalSystem, Ikeda, Lorenz63, Lorenz96
from diengmf.measurement_systems import AbstractMeasurementSystem, RangeSensor
from diengmf.stochastic_filters import AbstractFilter, EnGMF
from distreqx.distributions import AbstractDistribution, MultivariateNormalTri
from jaxtyping import Array, Float, Int, Key
from dataclasses import dataclass

import uuid
print(uuid.uuid4())

@dataclass
class FilterExperiment(eqx.Module):
    initial_belief: AbstractDistribution
    dynamical_system: AbstractDynamicalSystem
    measurement_system: AbstractMeasurementSystem
    stochastic_filter: AbstractFilter
    measurement_times: Float[Array, ""]
    burn_in_time: int
    total_experiment_time: int
    initial_true_state: Float[Array, " state_dim"]

### Ikeda Experiments

initial_state = jnp.array([1.25, 0.0])
initial_covariance = 1/128 * jnp.eye(initial_state.shape[0])
measurement_covariance = jnp.array([[0.25]])
dynamical_system = Ikeda()
measurement_system=RangeSensor(measurement_covariance)
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

@eqx.filter_jit(donate="all-except-first")
@eqx.filter_vmap(in_axes=(0, None))
def run_filter_experiment(key: Key[Array, " mc_iterations"], experiment_description: FilterExperiment):
    key, initial_belief_key, initial_key = jax.random.split(key, 3)

    burn_in_time = experiment_description.burn_in_time
    measurement_time = experiment_description.total_experiment_time

    initial_state: Float[Array, "2"] = experiment_description.initial_true_state
    stochastic_filter: AbstractFilter = experiment_description.stochastic_filter
    
    initial_belief: AbstractDistribution = experiment_description.initial_belief
    initial_belief = stochastic_filter.initialize(initial_belief_key, initial_belief)
    initial_carry = (initial_key, initial_belief, initial_state)

    @eqx.filter_jit
    def filtering_step(carry, _):
        key, posterior_belief, true_state = carry
        key, measurement_key = jax.random.split(key)

        ### Predict: Advance in time (system.dt)
        prior_belief = stochastic_filter.predict(posterior_belief, jnp.asarray(0.0), experiment_description.measurement_times)
        true_state = stochastic_filter.dynamical_system.flow(jnp.asarray(0.0), experiment_description.measurement_times, true_state)

        ### Update: Bayes' Rule
        measurement = stochastic_filter.measurement_system(true_state, measurement_key)
        posterior_belief = stochastic_filter.update(key, prior_belief, measurement)

        # Compute metrics
        error = true_state - jnp.mean(posterior_belief.means, axis=0)
        cfac = jax.scipy.linalg.cho_factor(posterior_belief.cov[0, ...])
        snees_error = error.T @ jax.scipy.linalg.cho_solve(cfac, error)
        return (key, posterior_belief, true_state), (error, snees_error)
    
    (final_key, final_belief, final_state), (errors, snees_errors) = jax.lax.scan(filtering_step, initial_carry, length=measurement_time)

    errors_past_burn_in = errors[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))
    snees = jnp.mean(snees_errors[burn_in_time:])
    
    return jnp.array([rmse, snees])


key = jax.random.key(10)
key, subkey = jax.random.split(key)

num_mc_iterations = 5
rmse, snees = jnp.mean(run_filter_experiment(jax.random.split(subkey, num_mc_iterations), experiment_description), axis=0)


print(rmse, snees)

###

# covariances = [jnp.array([[0.25]]), jnp.array([[1.0]]), jnp.array([[4.0]])]
# experiments_list = [eqx.tree_at(lambda x: x.measurement_system.covariance, 
#                                 experiment_description, cov) for cov in covariances]
# experiment_descriptions = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *experiments_list)

# @eqx.filter_vmap(in_axes=(0, 0))
@eqx.filter_vmap(in_axes=(0, None))
def run_filter_experiment(key: Key[Array, ""], experiment_description: FilterExperiment):
    jax.debug.print("{} {}", key, experiment_description.measurement_system.covariance)
    pass

# keys = jax.random.split(subkey, 3 * 2).reshape(-1, 3)
# num_mc_iterations = 5
# keys = jax.random.split(subkey, num_mc_iterations)
# run_filter_experiment(keys, experiment_descriptions)

###

###

# def test_methods():
#     key = jax.random.key(10)
#     key, subkey, initial_key = jax.random.split(key, 3)

#     burn_in_time = 100
#     measurement_time = 10 * burn_in_time

#     dynamical_system: AbstractDynamicalSystem = Ikeda()
#     initial_state: Float[Array, "2"] = dynamical_system.initial_state()
#     measurement_system: AbstractMeasurementSystem = RangeSensor(jnp.array([[0.25]]))
#     initial_belief: AbstractDistribution = MultivariateNormalTri(initial_state, 1/128 * jnp.eye(initial_state.shape[0]))
    
#     stochastic_filter: AbstractFilter = EnGMF(
#         dynamical_system=dynamical_system,
#         measurement_system=measurement_system,
#         ensemble_size=1000,
#         # silverman_bandwidth_scaling=22.0,
#     )

#     initial_belief = stochastic_filter.initialize(subkey, initial_belief)
#     initial_carry = (initial_key, initial_belief, initial_state)

#     ###
#     # Filtering Loop Logic Goes Here
#     # WIP

#     @eqx.filter_jit
#     def filtering_step(carry, _):
#         key, posterior_belief, true_state = carry
#         key, measurement_key = jax.random.split(key)

#         ### Predict: Advance in time (system.dt)
#         prior_belief = stochastic_filter.predict(posterior_belief, 0.0, dynamical_system.dt)
#         true_state = stochastic_filter.dynamical_system.flow(0.0, dynamical_system.dt, true_state)

#         ### Update: Bayes' Rule
#         measurement = stochastic_filter.measurement_system(true_state, measurement_key)
#         posterior_belief = stochastic_filter.update(key, prior_belief, measurement)

#         # Compute metrics
#         error = true_state - jnp.mean(posterior_belief.means, axis=0)
#         cfac = jax.scipy.linalg.cho_factor(posterior_belief.cov[0, ...])
#         snees_error = error.T @ jax.scipy.linalg.cho_solve(cfac, error)
#         return (key, posterior_belief, true_state), (error, snees_error)
    
#     (final_key, final_belief, final_state), (errors, snees_errors) = jax.lax.scan(filtering_step, initial_carry, length=measurement_time)

#     errors_past_burn_in = errors[burn_in_time:]
#     rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))
#     snees = jnp.mean(snees_errors[burn_in_time:])
    
#     return rmse, snees


# test_methods()

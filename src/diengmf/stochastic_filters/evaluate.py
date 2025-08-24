from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key
from distreqx.distributions import AbstractDistribution

from diengmf.dynamical_systems import AbstractDynamicalSystem
from diengmf.measurement_systems import AbstractMeasurementSystem
from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.stochastic_filters.stochastic_filter_abc import AbstractFilter


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


@eqx.filter_jit(donate="all-except-first")
def evaluate_filter(
    key: Key[Array, ""], experiment_description: FilterExperiment
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    key, initial_belief_key, initial_key = jax.random.split(key, 3)

    burn_in_time = experiment_description.burn_in_time
    measurement_time = experiment_description.total_experiment_time

    initial_state: Float[Array, "state_dim"] = experiment_description.initial_true_state
    stochastic_filter: AbstractFilter = experiment_description.stochastic_filter

    initial_belief: AbstractDistribution = experiment_description.initial_belief
    initial_belief = stochastic_filter.initialize(initial_belief_key, initial_belief)
    initial_carry = (initial_key, initial_belief, initial_state)

    @eqx.filter_jit
    def filtering_step(carry, _):
        key, posterior_belief, true_state = carry
        key, measurement_key = jax.random.split(key)

        # Predict: Advance in time
        prior_belief = stochastic_filter.predict(
            posterior_belief, jnp.asarray(0.0), experiment_description.measurement_times
        )
        true_state = stochastic_filter.dynamical_system.flow(
            jnp.asarray(0.0), experiment_description.measurement_times, true_state
        )

        # Update: Bayes' Rule
        measurement = stochastic_filter.measurement_system(true_state, measurement_key)
        posterior_belief = stochastic_filter.update(key, prior_belief, measurement)

        # Compute metrics
        error = true_state - jnp.mean(posterior_belief.ensemble, axis=0)
        
        # Handle different covariance structures: GMM has per-component cov, EnsembleGaussian has single cov
        if hasattr(posterior_belief, 'weights'):  # GMM case
            cfac = jax.scipy.linalg.cho_factor(posterior_belief.cov[0, ...])
        else:  # EnsembleGaussian case
            cfac = jax.scipy.linalg.cho_factor(posterior_belief.cov)
        snees_error = error.T @ jax.scipy.linalg.cho_solve(cfac, error)
        return (key, posterior_belief, true_state), (error, snees_error)

    (final_key, final_belief, final_state), (errors, snees_errors) = jax.lax.scan(
        filtering_step, initial_carry, length=measurement_time
    )

    errors_past_burn_in = errors[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))
    snees = jnp.mean(snees_errors[burn_in_time:])

    return rmse, snees


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, 0, None, None, None))
def flow_discriminator_sampler(
    key: Key[Array, ""],
    mean: Float[Array, " state_dim"],
    cov: Float[Array, " state_dim state_dim"],
    flow: NormalizingFlow,
    rejection_threshold: float = 0.0,
    max_attempts: int = 5,
) -> Float[Array, " state_dim"]:
    """Sample with normalizing flow discriminator for rejection sampling."""

    def sample_and_score(sample_key):
        sample = jax.random.multivariate_normal(sample_key, mean, cov)
        _, log_prob = flow.forward(sample)
        return sample, log_prob

    def rejection_step(carry, _):
        key_state, best_sample, best_score = carry
        key_state, subkey = jax.random.split(key_state)

        sample, score = sample_and_score(subkey)
        is_better = score > best_score

        new_best_sample = jnp.where(is_better, sample, best_sample)
        new_best_score = jnp.where(is_better, score, best_score)

        return (key_state, new_best_sample, new_best_score), None

    # Initialize with first sample
    key, init_key = jax.random.split(key)
    init_sample, init_score = sample_and_score(init_key)

    # Run rejection sampling
    (final_key, final_sample, final_score), _ = jax.lax.scan(
        rejection_step, (key, init_sample, init_score), jnp.arange(max_attempts)
    )

    return final_sample

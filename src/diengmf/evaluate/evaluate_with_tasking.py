from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Int, Key
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import (AbstractMeasurementSystem,
                                           simulate_thrust)
from xradar_uq.statistics import generate_random_impulse_velocity
from xradar_uq.stochastic_filters import AbstractFilter

@eqx.filter_jit
def maneuver_aware(thrust_key, ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system):
    thrust_ensemble = simulate_thrust(thrust_key, ensemble, num_simulations, delta_v_magnitude)
    thrust_ensemble = eqx.filter_vmap(dynamical_system.flow, in_axes=(None, None, 0))(0.0, time_horrizon, thrust_ensemble)
    return thrust_ensemble


@eqx.filter_jit
def tracking_scan_step(
    carry: tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"], Float[Array, ""], int | Int[Array, ""]],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: AbstractFilter,
    tracking_fn: Callable,
    time_range: float | Float[Array, ""],
    delta_v_magnitude: float | Float[Array, ""],
    maneuver_proportion: float | Float[Array, ""],
    random_impulse_velocity: Float[Array, "3"],
) -> tuple[tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"], Float[Array, ""], int | Int[Array, ""]], Bool[Array, ""]]:
    posterior_ensemble, true_state, total_fuel, times_found, measurement_id = carry
    update_key, measurement_key, thrust_key, tracking_key = jax.random.split(key, 4)
    
    current_time = measurement_id * time_range
    previous_time = (measurement_id - 1) * time_range
    maneuver_interval = 86400/382981
    max_days_per_step = 20
    
    def scan_day_step(day_carry, day_idx):
        state, t_current, fuel_remaining = day_carry
        maneuver_time = day_idx * maneuver_interval
        should_flow_to_maneuver = (maneuver_time >= previous_time) & (maneuver_time < current_time)
        should_apply_maneuver = should_flow_to_maneuver & (fuel_remaining > 0)
        
        flow_time = jnp.where(should_flow_to_maneuver, maneuver_time - t_current, 0.0)
        state = jnp.where(should_flow_to_maneuver, dynamical_system.flow(0.0, flow_time, state), state)
        
        maneuver_key = jax.random.fold_in(thrust_key, day_idx)
        impulse = jax.random.normal(maneuver_key, (3,)) * delta_v_magnitude
        state = jnp.where(should_apply_maneuver, state.at[3:].add(impulse), state)
        fuel_remaining = jnp.where(should_apply_maneuver, fuel_remaining - delta_v_magnitude, fuel_remaining)
        t_current = jnp.where(should_flow_to_maneuver, maneuver_time, t_current)
        
        return (state, t_current, fuel_remaining), None
    
    def apply_scheduled_maneuvers_true_state(state):
        initial_carry = (state, previous_time, total_fuel)
        day_indices = jnp.arange(max_days_per_step)
        final_carry, _ = jax.lax.scan(scan_day_step, initial_carry, day_indices)
        final_state, final_time, final_fuel = final_carry
        
        remaining_flow_time = current_time - final_time
        final_state = dynamical_system.flow(0.0, remaining_flow_time, final_state)
        
        return final_state, final_fuel
    
    def flow_ensemble_no_maneuvers(state):
        return dynamical_system.flow(0.0, time_range, state)

    @eqx.filter_jit
    def reachable_set(thrust_key, ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system, n_directions=100):
        thrust_ensemble = simulate_thrust(thrust_key, ensemble, num_simulations, delta_v_magnitude)
        thrust_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_horrizon, thrust_ensemble)
        return thrust_ensemble

    
    
    true_state_next, total_fuel_next = apply_scheduled_maneuvers_true_state(true_state)
    prior_ensemble = eqx.filter_vmap(flow_ensemble_no_maneuvers)(posterior_ensemble)
    # num_simulations = 1
    # prior_ensemble = maneuver_aware(thrust_key, posterior_ensemble, time_range, delta_v_magnitude, num_simulations, dynamical_system)
    
    is_measurable = tracking_fn(true_state_next, prior_ensemble, tracking_key, posterior_ensemble, dynamical_system, time_range, delta_v_magnitude)
    posterior_ensemble_next = jnp.where(is_measurable, stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state_next, measurement_key), measurement_system), prior_ensemble)
    
    return (posterior_ensemble_next, true_state_next, total_fuel_next, times_found + jnp.where(is_measurable, 1, 0), measurement_id + 1), is_measurable

###

# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_tracking_single_case(
    delta_v_magnitude: float | Float[Array, ""],
    maneuver_proportion: float | Float[Array, ""],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: AbstractFilter,
    tracking_fn: Callable[[Float[Array, "state_dim"], Float[Array, "batch_size state_dim"], Key[Array, ""]], Bool[Array, ""]],
    time_range: float | Float[Array, ""] = 0.242,
    measurement_time: int = 200,
    initial_fuel: float = 1.25,
) -> Float[Array, ""]:
    key, state_key, impulse_key = jax.random.split(key, 3)
    
    true_state = dynamical_system.initial_state()
    posterior_ensemble = dynamical_system.generate(state_key, batch_size=stochastic_filter.ensemble_size)
    random_impulse_velocity = generate_random_impulse_velocity(impulse_key, delta_v_magnitude)
    
    measurement_keys = jax.random.split(key, measurement_time)
    initial_carry = (posterior_ensemble, true_state, initial_fuel, 0, 0)
    
    def scan_fn(carry, key):
        return tracking_scan_step(
            carry, key, dynamical_system, measurement_system, stochastic_filter,
            tracking_fn, time_range, delta_v_magnitude, maneuver_proportion, random_impulse_velocity
        )
    
    final_carry, _ = jax.lax.scan(scan_fn, initial_carry, measurement_keys)
    _, _, _, times_found, _ = final_carry
    
    found_proportion = times_found / measurement_time
    return found_proportion


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_tracking_grid(
    delta_v_range: Float[Array, "n_dv"],
    maneuver_proportion_range: Float[Array, "n_mp"],
    time_horrizon,
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: AbstractFilter,
    tracking_fn: Callable[[Float[Array, "state_dim"], Float[Array, "batch_size state_dim"], Key[Array, ""]], Bool[Array, ""]],
    measurement_count,
    mc_iterations: int = 1,
) -> Float[Array, "n_dv n_mp mc_iterations"]:
    n_dv, n_mp = len(delta_v_range), len(maneuver_proportion_range)
    
    mc_keys = jax.random.split(key, mc_iterations)
    
    def evaluate_mc_iteration(mc_key):
        dv_keys = jax.random.split(mc_key, n_dv)
        
        def evaluate_dv(dv_key, dv_val):
            mp_keys = jax.random.split(dv_key, n_mp)
            
            def evaluate_mp(mp_key, mp_val):
                return evaluate_tracking_single_case(
                    dv_val, mp_val, mp_key,
                    dynamical_system, measurement_system, stochastic_filter, tracking_fn, time_horrizon, measurement_time=measurement_count
                )
            
            return eqx.filter_vmap(evaluate_mp)(mp_keys, maneuver_proportion_range)
        
        return eqx.filter_vmap(evaluate_dv)(dv_keys, delta_v_range)
    
    results = eqx.filter_vmap(evaluate_mc_iteration)(mc_keys)
    return jnp.transpose(results, (1, 2, 0))

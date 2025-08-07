
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from diengmf.measurement_systems import tracking_measurability
from diengmf.statistics import (generate_random_impulse_velocity,
                                  silverman_kde_estimate)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def find_random_second_sensor_state(
    prior_ensemble: Float[Array, "batch_size state_dim"], 
    key: Key[Array, ""],
) -> Float[Array, "state_dim"]:
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    gmm = silverman_kde_estimate(prior_ensemble[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    optimal_position = points_3d[jax.random.choice(key, pdf_values.shape[0])]
    
    return jnp.concatenate([optimal_position, jnp.zeros(3)])


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def find_optimal_second_sensor_state(
    prior_ensemble: Float[Array, "batch_size state_dim"], 
) -> Float[Array, "state_dim"]:
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    gmm = silverman_kde_estimate(prior_ensemble[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    optimal_position = points_3d[jnp.argmax(pdf_values)]
    
    return jnp.concatenate([optimal_position, jnp.zeros(3)])

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def simulate_thrust(
    key: Key[Array, ""],
    posterior_ensemble: Float[Array, "ensemble_size state_dim"], 
    num_particles: int,
    delta_v_magnitude: float | Float[Array, ""],
) -> Float[Array, "ensemble_size*{num_particles} state_dim"]:
    ensemble_size, state_dim = posterior_ensemble.shape
    keys = jax.random.split(key, ensemble_size * num_particles)
    
    velocity_impulses = eqx.filter_vmap(lambda k: generate_random_impulse_velocity(k, delta_v_magnitude))(keys)
    state_impulses = jnp.concatenate([jnp.zeros((ensemble_size * num_particles, 3)), velocity_impulses], axis=1)
    state_impulses = state_impulses.reshape(ensemble_size, num_particles, state_dim)
    
    expanded_ensemble = jnp.repeat(posterior_ensemble[:, None, :], num_particles, axis=1)
    result = (expanded_ensemble + state_impulses).reshape(-1, state_dim)
    
    assert result.shape == (ensemble_size * num_particles, state_dim)
    return result


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def mc_deltav_dual_sensor_tracking(
    true_state: Float[Array, "state_dim"],
    prior_ensemble,
    key: Key[Array, ""],
    posterior_ensemble: Float[Array, "batch_size state_dim"],
    dynamical_system,
) -> Bool[Array, ""]:
    mc_key, sensor_key = jax.random.split(key)
    mc_ensemble = simulate_thrust(mc_key, posterior_ensemble, 100, 1e-2)
    mc_propagated = eqx.filter_vmap(dynamical_system.flow)(0.0, 0.242, mc_ensemble)
    
    primary_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 0.242, posterior_ensemble)
    primary_state = jnp.mean(primary_ensemble, axis=0)
    
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    gmm = silverman_kde_estimate(mc_propagated[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    secondary_state = jnp.concatenate([points_3d[jnp.argmax(pdf_values)], jnp.zeros(3)])
    
    return tracking_measurability(true_state, primary_state) | tracking_measurability(true_state, secondary_state)

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def dual_sensor_tracking_random(
    true_state: Float[Array, "state_dim"],
    prior_ensemble: Float[Array, "batch_size state_dim"],
    key: Key[Array, ""],
    posterior_ensemble,
) -> Bool[Array, ""]:
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    second_predicted_state = find_random_second_sensor_state(prior_ensemble, key)
    return tracking_measurability(true_state, predicted_state) | tracking_measurability(true_state, second_predicted_state)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def dual_sensor_tracking_optimal(
    true_state: Float[Array, "state_dim"],
    prior_ensemble: Float[Array, "batch_size state_dim"],
    key: Key[Array, ""],
    posterior_ensemble,
) -> Bool[Array, ""]:
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    second_predicted_state = find_optimal_second_sensor_state(prior_ensemble)
    return tracking_measurability(true_state, predicted_state) | tracking_measurability(true_state, second_predicted_state)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def single_sensor_tracking(
    true_state: Float[Array, "state_dim"],
    prior_ensemble: Float[Array, "batch_size state_dim"],
    key: Key[Array, ""],
    posterior_ensemble,
) -> Bool[Array, ""]:
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    return tracking_measurability(true_state, predicted_state)

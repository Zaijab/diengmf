"""

"""
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from xradar_uq.statistics import silverman_kde_estimate


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def find_random_second_sensor_state(
    prior_ensemble: Float[Array, "batch_size state_dim"], key,
) -> Float[Array, "state_dim"]:
    # Generate constraint boundary points (from your code)
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    
    # Convert to 3D coordinates (degrees to radians, then spherical to Cartesian)
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    # Build GMM from ensemble positions and evaluate at boundary points
    gmm = silverman_kde_estimate(prior_ensemble[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    optimal_position = points_3d[jax.random.choice(key, pdf_values.shape[0])]
    
    return jnp.concatenate([optimal_position, jnp.zeros(3)])

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def find_optimal_second_sensor_state(
    prior_ensemble: Float[Array, "batch_size state_dim"], 
) -> Float[Array, "state_dim"]:
    # Generate constraint boundary points (from your code)
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    
    # Convert to 3D coordinates (degrees to radians, then spherical to Cartesian)
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    # Build GMM from ensemble positions and evaluate at boundary points
    gmm = silverman_kde_estimate(prior_ensemble[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    optimal_position = points_3d[jnp.argmax(pdf_values)]
    
    return jnp.concatenate([optimal_position, jnp.zeros(3)])

@eqx.filter_jit
def place_sensors(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"],
    n_sensors: int, exclusion_radius: float, selection: Callable
) -> Float[Array, "n_sensors 2"]:
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, subkey = jax.random.split(key_state)
        
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_distances, axis=1)
        valid_candidate_mask = min_distances > exclusion_radius
        
        valid_candidates = jnp.where(valid_candidate_mask[:, None], candidates, jnp.inf)
        finite_mask = jnp.isfinite(valid_candidates).all(axis=1)
        filtered_candidates = jnp.where(finite_mask[:, None], valid_candidates, 0.0)
        
        new_position = selection(subkey, filtered_candidates)
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        
        return (updated_positions, updated_mask, key_state)
    
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, (sensor_positions, placement_mask, key))
    assert final_positions.shape == (n_sensors, 2)
    return final_positions

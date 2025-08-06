"""
This is a funciton
RangeSensor r < rthresh
AnglesOnly uses r_0 < r < r_1
Radar uses r_0 < r < r_1
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, jaxtyped

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def tracking_measurability(
    state: Float[Array, "state_dim"],
    predicted_state: Float[Array, "state_dim"], 
    observer_position: Float[Array, "3"] = jnp.array([-0.01215058560962404, 0.0, 0.0]),
    elevation_fov: float  | Float[Array, ""] = jnp.deg2rad(5),
    azimuth_fov: float | Float[Array, ""] = jnp.deg2rad(5),
    range_fov: float  | Float[Array, ""] = jnp.inf
) -> Bool[Array, ""]:
    
    # Calculate relative positions from observer
    rel_pos = state[:3] - observer_position
    rel_pos_pred = predicted_state[:3] - observer_position
    
    # Convert to spherical coordinates
    rho = jnp.linalg.norm(rel_pos)
    rho_pred = jnp.linalg.norm(rel_pos_pred)
    
    elevation = jnp.arcsin(rel_pos[2] / rho)
    elevation_pred = jnp.arcsin(rel_pos_pred[2] / rho_pred)
    
    azimuth = jnp.arctan2(rel_pos[1], rel_pos[0])
    azimuth_pred = jnp.arctan2(rel_pos_pred[1], rel_pos_pred[0])
    
    # Handle angle wrapping for azimuth difference
    azimuth_diff = azimuth - azimuth_pred
    azimuth_diff = jnp.arctan2(jnp.sin(azimuth_diff), jnp.cos(azimuth_diff))
    
    # Check if within field of view
    elevation_check = jnp.abs(elevation - elevation_pred) <= elevation_fov / 2
    azimuth_check = jnp.abs(azimuth_diff) <= azimuth_fov / 2
    range_check = jnp.abs(rho - rho_pred) <= range_fov
    
    return elevation_check & azimuth_check & range_check


import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from xradar_uq.dynamical_systems import CR3BP


@jaxtyped(typechecker=typechecker)
def compute_cr3bp_reachable_set(
    key,
    initial_condition: Float[Array, "6"],
    delta_v_magnitude: float,
    seeds: int = 900,
    outer: int = 400, 
    delta_t: float = 0.1,
    N: float = 10.0,
    weight_exponent: float = 2.0,
    fraction: float = 0.9,
    V: int = 10,
    sigma: float = 1.0,
    R: int = 3,
) -> Dict[str, jax.Array]:
    dynamical_system = CR3BP()
    
    # Step 1: Generate seed vertices in ΔV sphere
    key, subkey = jax.random.split(key)
    interior_vertices = jax.random.multivariate_normal(
        subkey, shape=(seeds,), mean=jnp.zeros(3), cov=jnp.eye(3)
    )
    interior_vertices /= jnp.linalg.norm(interior_vertices, axis=1, keepdims=True)
    key, subkey = jax.random.split(key)
    radii = jax.random.uniform(subkey, shape=(seeds, 1)) ** (1/3)
    interior_vertices = radii * interior_vertices
    
    key, subkey = jax.random.split(key)
    boundary_vertices = jax.random.multivariate_normal(
        subkey, shape=(outer,), mean=jnp.zeros(3), cov=jnp.eye(3)
    )
    boundary_vertices /= jnp.linalg.norm(boundary_vertices, axis=1, keepdims=True)
    
    vertices = jnp.concatenate([interior_vertices, boundary_vertices], axis=0)
    
    # Step 2-4: Initial trajectories and stability filtering
    states_6d = vertices * delta_v_magnitude + initial_condition
    times = jnp.arange(0.0, 10.0, delta_t)
    trajectories = eqx.filter_vmap(
        lambda state: dynamical_system.integrate(state, times)
    )(states_6d)
    
    # Remove unstable trajectories
    trajectory_positions = trajectories[:, :, :3]
    distances_from_origin = jnp.linalg.norm(trajectory_positions, axis=2)
    max_distances = jnp.max(distances_from_origin, axis=1)
    stable_mask = max_distances <= N
    
    stable_vertices = vertices[stable_mask]
    stable_trajectories = trajectories[stable_mask]
    
    # Iterative refinement (Steps 5-8)
    current_vertices = stable_vertices
    current_trajectories = stable_trajectories
    
    for iteration in range(R):
        # Step 5: Compute edge weights for refinement
        n_vertices = current_vertices.shape[0]
        final_states = current_trajectories[:, -1, :]
        final_positions = final_states[:, :3]
        final_distances = jnp.linalg.norm(final_positions, axis=1)
        
        # Classify outcomes
        impact_threshold = 0.1
        escape_threshold = 2.0
        impacted = final_distances < impact_threshold
        escaped = final_distances > escape_threshold
        in_system = ~(impacted | escaped)
        
        outcomes = jnp.where(impacted, 0, jnp.where(escaped, 1, 2))
        
        # Generate edges and compute weights
        key, subkey = jax.random.split(key)
        edge_indices = jax.random.choice(
            subkey, n_vertices, shape=(V, 2), replace=True
        )
        
        edge_lengths = jnp.linalg.norm(
            current_vertices[edge_indices[:, 0]] - current_vertices[edge_indices[:, 1]], 
            axis=1
        )
        
        outcome_diff = outcomes[edge_indices[:, 0]] != outcomes[edge_indices[:, 1]]
        weights = edge_lengths ** weight_exponent
        boundary_crossing_weights = jnp.where(outcome_diff, weights * 2.0, weights)
        
        # Step 6: Add new vertices
        selected_edges = edge_indices[:V]
        edge_centers = 0.5 * (
            current_vertices[selected_edges[:, 0]] + current_vertices[selected_edges[:, 1]]
        )
        
        key, subkey = jax.random.split(key)
        perturbations = jax.random.multivariate_normal(
            subkey, shape=(V,), mean=jnp.zeros(3), cov=sigma**2 * jnp.eye(3)
        )
        new_vertices = edge_centers + perturbations
        
        # Step 7: Compute trajectories for new vertices
        new_states_6d = new_vertices * delta_v_magnitude + initial_condition
        new_trajectories = eqx.filter_vmap(
            lambda state: dynamical_system.integrate(state, times)
        )(new_states_6d)
        
        # Filter new stable trajectories
        new_positions = new_trajectories[:, :, :3]
        new_distances = jnp.linalg.norm(new_positions, axis=2)
        new_max_distances = jnp.max(new_distances, axis=1)
        new_stable_mask = new_max_distances <= N
        
        # Step 8: Combine vertices and trajectories
        if jnp.sum(new_stable_mask) > 0:
            stable_new_vertices = new_vertices[new_stable_mask]
            stable_new_trajectories = new_trajectories[new_stable_mask]
            
            current_vertices = jnp.concatenate([current_vertices, stable_new_vertices])
            current_trajectories = jnp.concatenate([current_trajectories, stable_new_trajectories])
    
    # Return reachable set
    final_positions = current_trajectories[:, -1, :3]
    final_velocities = current_trajectories[:, -1, 3:]
    
    return {
        'vertices': current_vertices,
        'trajectories': current_trajectories,
        'final_positions': final_positions,
        'final_velocities': final_velocities,
        'times': times
    }


compute_cr3bp_reachable_set()

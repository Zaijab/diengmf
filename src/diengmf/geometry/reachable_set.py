"""
This file contains two main functions:
- `angular_convex_hull` and `thrust_convex_hull`, these will monte carlo simulate a trajectory and return reachable sets (convex hulls)
  in either 6D (all allowable states) or in 2D (all allowable sensor tasking locations)

- We also provide utility functions to triangulate a convex hull and to do barycentric subdivision

"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polytopax as ptx
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import AnglesOnly, simulate_thrust


class ZConvexHull(eqx.Module, ptx.ConvexHull):
    pass

@eqx.filter_jit
def reachable_set(thrust_key, ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system, n_directions=100):
    thrust_ensemble = simulate_thrust(thrust_key, ensemble, num_simulations, delta_v_magnitude)
    thrust_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_horrizon, thrust_ensemble)
    hull = ZConvexHull.from_points(thrust_ensemble, n_directions=n_directions)
    return hull

@eqx.filter_jit
def angular_reachable_set(thrust_key, ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system, n_directions=100):
    thrust_ensemble = simulate_thrust(thrust_key, ensemble, num_simulations, delta_v_magnitude)
    thrust_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_horrizon, thrust_ensemble)

    angles = AnglesOnly()
    ensemble_angles = eqx.filter_vmap(angles)(thrust_ensemble)
    angle_hull = angular_convex_hull(ensemble_angles, n_directions)
    
    return angle_hull


@eqx.filter_jit
def angular_convex_hull(ensemble_angles: Float[Array, "n_samples 2"], n_directions=100) -> Float[Array, "n_hull_vertices 2"]:
    azimuth_range = jnp.max(ensemble_angles[:, 0]) - jnp.min(ensemble_angles[:, 0])
    
    # Handle azimuth wraparound
    wrapped_angles = jnp.where(azimuth_range > jnp.pi,
                               jnp.where(ensemble_angles[:, 0] < 0, 
                                       ensemble_angles[:, 0] + 2*jnp.pi, 
                                       ensemble_angles[:, 0]),
                               ensemble_angles[:, 0])
    points_2d = jnp.column_stack([wrapped_angles, ensemble_angles[:, 1]])

    return ZConvexHull.from_points(points_2d, n_directions=n_directions)

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def fan_triangulate(hull: ptx.ConvexHull) -> Float[Array, "n_triangles 3 2"]:
    hull_vertices = hull.vertices_array()
    centroid = jnp.mean(hull_vertices, axis=-2) #hull.centroid()
    n_vertices = hull_vertices.shape[0]
    
    # Create triangles: (centroid, vertex_i, vertex_{i+1})
    triangles = jnp.zeros((n_vertices, 3, 2))
    triangles = triangles.at[:, 0, :].set(centroid)  # All triangles share centroid
    triangles = triangles.at[:, 1, :].set(hull_vertices)  # Current vertex
    triangles = triangles.at[:, 2, :].set(jnp.roll(hull_vertices, -1, axis=0))  # Next vertex
    
    return triangles


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def barycentric_subdivision(triangles: Float[Array, "n_triangles 3 2"]) -> Float[Array, "4*n_triangles 3 2"]:
    n_triangles = triangles.shape[0]
    
    # Extract vertices A, B, C for all triangles
    A = triangles[:, 0, :]  # shape (n_triangles, 2)
    B = triangles[:, 1, :]  # shape (n_triangles, 2)
    C = triangles[:, 2, :]  # shape (n_triangles, 2)
    
    # Compute edge midpoints
    M_AB = (A + B) / 2
    M_BC = (B + C) / 2
    M_AC = (A + C) / 2
    
    # Create 4 sub-triangles per original triangle
    new_triangles = jnp.zeros((n_triangles, 4, 3, 2))
    
    # Triangle 1: (A, M_AB, M_AC)
    new_triangles = new_triangles.at[:, 0, 0, :].set(A)
    new_triangles = new_triangles.at[:, 0, 1, :].set(M_AB)
    new_triangles = new_triangles.at[:, 0, 2, :].set(M_AC)
    
    # Triangle 2: (M_AB, B, M_BC)
    new_triangles = new_triangles.at[:, 1, 0, :].set(M_AB)
    new_triangles = new_triangles.at[:, 1, 1, :].set(B)
    new_triangles = new_triangles.at[:, 1, 2, :].set(M_BC)
    
    # Triangle 3: (M_AC, M_BC, C)  
    new_triangles = new_triangles.at[:, 2, 0, :].set(M_AC)
    new_triangles = new_triangles.at[:, 2, 1, :].set(M_BC)
    new_triangles = new_triangles.at[:, 2, 2, :].set(C)
    
    # Triangle 4: (M_AB, M_BC, M_AC) - center triangle
    new_triangles = new_triangles.at[:, 3, 0, :].set(M_AB)
    new_triangles = new_triangles.at[:, 3, 1, :].set(M_BC)
    new_triangles = new_triangles.at[:, 3, 2, :].set(M_AC)
    
    return new_triangles.reshape(4 * n_triangles, 3, 2)

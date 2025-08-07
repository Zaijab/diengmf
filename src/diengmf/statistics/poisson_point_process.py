import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from diengmf.statistics.random_finite_sets import RFS


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def poisson_point_process_hyperrectangle(
    key: Key[Array, "..."],
    intensity: float,
    bounds: Float[Array, "n_dims 2"],
    max_points: int = 1000,
) -> tuple[Float[Array, "max_points n_dims"], Bool[Array, "max_points"]]:
    """
    Simulate a homogeneous Poisson point process on an n-dimensional hyperrectangle.

    Returns:
        Tuple of (points, mask) where mask indicates valid points
    """
    n_dims = bounds.shape[0]

    # Compute hypervolume
    widths = bounds[:, 1] - bounds[:, 0]
    hypervolume = jnp.prod(widths)

    # Step 1: Sample number of points
    count_key, points_key = jax.random.split(key)
    n_points = jax.random.poisson(count_key, lam=intensity)
    n_points = jnp.minimum(n_points, max_points)

    # Step 2: Sample points uniformly in hyperrectangle
    uniform_samples = jax.random.uniform(
        points_key, shape=(max_points, n_dims), minval=0.0, maxval=1.0
    )
    points = bounds[:, 0] + uniform_samples * widths[None, :]

    # Create mask for valid points
    mask = jnp.arange(max_points) < n_points

    return points, mask


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def poisson_point_process_rectangular_region(
    key: Key[Array, "..."],
    intensity: float,
    bounds: Float[Array, "n_dims 2"],
    max_points: int = 1000,
) -> RFS:
    """
    Simulate a homogeneous Poisson point process on an n-dimensional hyperrectangle.

    Returns:
        Tuple of (points, mask) where mask indicates valid points
    """
    n_dims = bounds.shape[0]

    # Compute hypervolume
    widths = bounds[:, 1] - bounds[:, 0]
    hypervolume = jnp.prod(widths)
    # Step 1: Sample number of points
    count_key, points_key = jax.random.split(key)
    n_points = jax.random.poisson(count_key, lam=intensity)
    n_points = jnp.minimum(n_points, max_points)

    # Step 2: Sample points uniformly in hyperrectangle
    uniform_samples = jax.random.uniform(
        points_key, shape=(max_points, n_dims), minval=0.0, maxval=1.0
    )
    points = bounds[:, 0] + uniform_samples * widths[None, :]

    # Create mask for valid points
    mask = jnp.arange(max_points) < n_points

    return RFS(points, mask)

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def poisson_point_process_rectangular_region(
    key: Key[Array, "..."],
    intensity: float,
    bounds: Float[Array, "n_dims 2"],
    max_points: int = 1000,
) -> RFS:
    """
    Simulate a homogeneous Poisson point process on an n-dimensional hyperrectangle.

    Returns:
        Tuple of (points, mask) where mask indicates valid points
    """
    n_dims = bounds.shape[0]

    # Compute hypervolume
    widths = bounds[:, 1] - bounds[:, 0]
    hypervolume = jnp.prod(widths)
    # Step 1: Sample number of points
    count_key, points_key = jax.random.split(key)
    n_points = jax.random.poisson(count_key, lam=intensity)
    n_points = jnp.minimum(n_points, max_points)

    # Step 2: Sample points uniformly in hyperrectangle
    uniform_samples = jax.random.uniform(
        points_key, shape=(max_points, n_dims), minval=0.0, maxval=1.0
    )
    points = bounds[:, 0] + uniform_samples * widths[None, :]

    # Create mask for valid points
    mask = jnp.arange(max_points) < n_points

    return RFS(points, mask)

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key


@eqx.filter_jit
def generate_random_impulse_velocity(key: Key[Array, ""], delta_v_magnitude: Float[Array, ""] | float) -> Float[Array, "3"]:
    azimuth_key, elevation_key = jax.random.split(key)
    random_impulse_azimuth = jax.random.uniform(azimuth_key, minval=0, maxval=2 * jnp.pi)
    random_impulse_elevation = jax.random.uniform(elevation_key, minval=- jnp.pi / 2, maxval=jnp.pi / 2)
    
    vx = delta_v_magnitude * jnp.cos(random_impulse_elevation) * jnp.cos(random_impulse_azimuth)
    vy = delta_v_magnitude * jnp.cos(random_impulse_elevation) * jnp.sin(random_impulse_azimuth)
    vz = delta_v_magnitude * jnp.sin(random_impulse_elevation)
    
    random_impulse_velocity = jnp.array([vx, vy, vz])
    random_impulse_velocity = (delta_v_magnitude / jnp.linalg.norm(random_impulse_velocity)) * random_impulse_velocity
    return random_impulse_velocity

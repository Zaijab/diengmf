import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from diengmf.measurement_systems import AbstractMeasurementSystem

# Look at orbital stability
# Experiment with prograde orbit
# Chase work with Spencer
# Look at different NRHO

class DeepSpaceNetwork(AbstractMeasurementSystem, strict=True):
    """
    This measurement system takes in an array whose first three indices are the spatial dimensions.
    Then returns the distance, elevation and azimuth.
    """

    mu: float = 0.01215058560962404
    covariance: Float[Array, "4 4"] = eqx.field(
        default_factory=lambda: jnp.diag(jnp.array([
            # https://science.nasa.gov/learn/basics-of-space-flight/chapter18-1/
            # 1 meter range precision → DU^2
            (1.0/(1000 * 389703))**2,
            #
            # 0.1mm/s range-rate → DU/TU^2
            (1e-7/389703*382981)**2,
            # https://descanso.jpl.nasa.gov/performmetrics/DSN_NavSysAccuracy.html
            # 5 nanorad azimuth → rad^2
            (5e-9)**2,             
            (5e-9)**2
        ]))
    )
    


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        state: Float[Array, "state_dim"],
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "4"]:
        
        # Satellite position in barycentric coordinates
        satellite_pos = state[:3]
        satellite_vel = state[3:]
        
        # earth position in barycentric coordinates
        earth_pos = jnp.array([- self.mu, 0.0, 0.0])
        
        # Satellite position relative to Moon
        relative_pos = satellite_pos - earth_pos
        x, y, z = relative_pos[0], relative_pos[1], relative_pos[2]
        
        # Range, azimuth, elevation from Earth
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        rho_dot = jnp.dot(relative_pos, satellite_vel) / rho
        alpha = jnp.arctan2(y, x)          # Azimuth
        epsilon = jnp.arcsin(z / rho)      # Elevation

        
        measurements = jnp.array([rho, rho_dot, alpha, epsilon])

        if key is not None:
            measurements = jax.random.multivariate_normal(
                key, mean=measurements, cov=self.covariance
            )

        return measurements

import equinox as eqx
import jax
import jax.numpy as jnp

from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.dynamical_systems import AbstractDynamicalSystem


class UKF(AbstractFilter, strict=True):
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    regularization: float = 1e-6
    debug: bool = False
    ensemble_size: int = 13
    
    @eqx.filter_jit  
    def generate_sigma_points(self, mean, covariance):
        n = mean.shape[0]
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        # Regularize covariance
        reg_cov = covariance + self.regularization * jnp.eye(n)
        reg_cov = (reg_cov + reg_cov.T) / 2
        
        # Robust matrix square root
        eigenvals, eigenvecs = jnp.linalg.eigh(reg_cov)
        eigenvals = jnp.maximum(eigenvals, 1e-10)
        L = eigenvecs @ jnp.diag(jnp.sqrt((n + lambda_param) * eigenvals))
        
        sigma_points = jnp.zeros((2*n + 1, n))
        sigma_points = sigma_points.at[0].set(mean)
        sigma_points = sigma_points.at[1:n+1].set(mean[None, :] + L)
        sigma_points = sigma_points.at[n+1:2*n+1].set(mean[None, :] - L)
        
        return sigma_points

    @eqx.filter_jit
    def get_sigma_weights(self, n):
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        weights_m = jnp.zeros(2*n + 1)
        weights_c = jnp.zeros(2*n + 1)
        
        weights_m = weights_m.at[0].set(lambda_param / (n + lambda_param))
        weights_c = weights_c.at[0].set(lambda_param / (n + lambda_param) + (1 - self.alpha**2 + self.beta))
        
        weight_other = 1.0 / (2 * (n + lambda_param))
        weights_m = weights_m.at[1:].set(weight_other)
        weights_c = weights_c.at[1:].set(weight_other)
        
        return weights_m, weights_c

    @eqx.filter_jit
    def ensemble_to_sigma_points(self, ensemble):
        """Convert ensemble to sigma points (for initialization)"""
        mean = jnp.mean(ensemble, axis=0)
        deviations = ensemble - mean[None, :]
        cov = (deviations.T @ deviations) / (ensemble.shape[0] - 1)
        cov = cov + self.regularization * jnp.eye(cov.shape[0])
        return self.generate_sigma_points(mean, cov)

    @eqx.filter_jit
    def sigma_points_to_statistics(self, sigma_points):
        """Convert sigma points to mean and covariance"""
        n = sigma_points.shape[1]
        weights_m, weights_c = self.get_sigma_weights(n)
        
        mean = jnp.sum(weights_m[:, None] * sigma_points, axis=0)
        residuals = sigma_points - mean[None, :]
        cov = jnp.sum(
            weights_c[:, None, None] * residuals[:, :, None] * residuals[:, None, :], 
            axis=0
        )
        
        # Ensure positive definiteness
        cov = (cov + cov.T) / 2
        eigenvals, eigenvecs = jnp.linalg.eigh(cov)
        eigenvals = jnp.maximum(eigenvals, self.regularization)
        cov = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        return mean, cov

    @eqx.filter_jit
    def predict(self, posterior_ensemble, dynamical_system, initial_time, final_time):
        """
        UKF Predict Step: Returns sigma points as 'ensemble'
        
        For compatibility: if input is not sigma points, convert ensemble to sigma points first
        """
        n = posterior_ensemble.shape[1]
        expected_sigma_points = 2 * n + 1
        
        # Check if input is already sigma points or needs conversion
        if posterior_ensemble.shape[0] == expected_sigma_points:
            # Input is already sigma points
            sigma_points = posterior_ensemble
        else:
            # Convert ensemble to sigma points
            sigma_points = self.ensemble_to_sigma_points(posterior_ensemble)
        
        # Propagate sigma points through dynamics
        propagated_sigma_points = eqx.filter_vmap(dynamical_system.flow)(
            initial_time, final_time, sigma_points
        )
        
        return propagated_sigma_points

    @eqx.filter_jit
    def update(self, key, prior_ensemble, measurement, measurement_system):
        """
        UKF Update Step: Takes sigma points as 'ensemble', returns new sigma points
        
        prior_ensemble should be sigma points from predict step
        """
        # Treat prior_ensemble as sigma points
        sigma_points = prior_ensemble
        n = sigma_points.shape[1]
        weights_m, weights_c = self.get_sigma_weights(n)
        
        # Propagate through measurement function  
        predicted_measurements = eqx.filter_vmap(measurement_system)(sigma_points)
        
        # Compute measurement statistics
        measurement_mean = jnp.sum(weights_m[:, None] * predicted_measurements, axis=0)
        
        innovation_residuals = predicted_measurements - measurement_mean[None, :]
        innovation_cov = jnp.sum(
            weights_c[:, None, None] * innovation_residuals[:, :, None] * innovation_residuals[:, None, :], 
            axis=0
        ) + measurement_system.covariance
        
        # Cross-covariance
        prior_mean = jnp.sum(weights_m[:, None] * sigma_points, axis=0)
        state_residuals = sigma_points - prior_mean[None, :]
        cross_cov = jnp.sum(
            weights_c[:, None, None] * state_residuals[:, :, None] * innovation_residuals[:, None, :],
            axis=0
        )
        
        # Kalman update
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, cross_cov.T).T
        innovation = measurement - measurement_mean
        
        posterior_mean = prior_mean + kalman_gain @ innovation
        
        # Get prior covariance for update
        prior_cov = jnp.sum(
            weights_c[:, None, None] * state_residuals[:, :, None] * state_residuals[:, None, :], 
            axis=0
        )
        prior_cov = prior_cov + self.regularization * jnp.eye(prior_cov.shape[0])
        
        # Covariance update
        posterior_cov = prior_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        
        # Ensure positive definiteness
        # posterior_cov = (posterior_cov + posterior_cov.T) / 2
        # eigenvals, eigenvecs = jnp.linalg.eigh(posterior_cov)
        # eigenvals = jnp.maximum(eigenvals, self.regularization)
        # posterior_cov = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        # Generate new sigma points from posterior
        posterior_sigma_points = self.generate_sigma_points(posterior_mean, posterior_cov)
        
        return posterior_sigma_points

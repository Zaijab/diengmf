"""
Example usage of flow discriminator training for EnGMF enhancement.
"""
import jax
import jax.numpy as jnp
from diengmf.dynamical_systems import Lorenz63
from diengmf.measurement_systems import RangeSensor
from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.stochastic_filters.engmf import EnGMF
from diengmf.training.flow_discriminator_training import train_flow_discriminator


def main():
    # Set up random key
    key = jax.random.key(42)
    
    # Create dynamical system (Lorenz63)
    dynamical_system = Lorenz63()
    
    # Create measurement system
    measurement_covariance = jnp.array([[1.0]])  # Range measurement noise
    measurement_system = RangeSensor(
        covariance=measurement_covariance, 
        center=jnp.array([6 * jnp.sqrt(2), 6 * jnp.sqrt(2), 27])
    )
    
    # Create normalizing flow
    state_dim = 3  # Lorenz63 is 3D
    flow = NormalizingFlow(
        input_dim=state_dim,
        num_layers=4,
        num_bins=8,
        conditioner_hidden_dim=64,
        conditioner_depth=2,
        key=key
    )
    
    # Create base EnGMF
    base_engmf = EnGMF(
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        ensemble_size=100,
        silverman_bandwidth_scaling=1.0
    )
    
    # Train the flow discriminator
    print("Starting flow discriminator training...")
    trained_flow, rmse_history = train_flow_discriminator(
        flow=flow,
        base_engmf=base_engmf,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        key=key,
        num_steps=20,  # Smaller number for demo
        learning_rate=1e-3,
        rejection_threshold=0.0,
        max_attempts=3,
        experiment_name="lorenz63_flow_discriminator"
    )
    
    print(f"Training completed!")
    print(f"Initial RMSE: {rmse_history[0]:.6f}")
    print(f"Final RMSE: {rmse_history[-1]:.6f}")
    print(f"Improvement: {rmse_history[0] - rmse_history[-1]:.6f}")
    print("Check MLFlow for detailed logs and visualizations.")


if __name__ == "__main__":
    main()
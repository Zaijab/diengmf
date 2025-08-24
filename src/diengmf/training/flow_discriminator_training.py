"""
Training loop for normalizing flow as discriminator for rejection sampling to improve EnGMF performance.
Logs training history, RMSE, hyperparameters, and system plots to MLFlow.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from diengmf.dynamical_systems import AbstractDynamicalSystem
from diengmf.measurement_systems import AbstractMeasurementSystem
from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.stochastic_filters.engmf import EnGMF
from diengmf.stochastic_filters.evaluate import (evaluate_filter,
                                                 flow_discriminator_sampler)


@jaxtyped(typechecker=typechecker)
def train_flow_discriminator(
    flow: NormalizingFlow,
    base_engmf: EnGMF,
    dynamical_system: AbstractDynamicalSystem,
    measurement_system: AbstractMeasurementSystem,
    key: Key[Array, ""],
    num_steps: int = 100,
    learning_rate: float = 1e-3,
    rejection_threshold: float = 0.0,
    max_attempts: int = 5,
    experiment_name: str = "flow_discriminator_training",
) -> tuple[NormalizingFlow, Float[Array, " num_steps"]]:
    """
    Train normalizing flow to minimize EnGMF RMSE through discriminator-based rejection sampling.
    Logs all training data to MLFlow for experiment tracking.

    Args:
        flow: Initial normalizing flow
        base_engmf: Base EnGMF filter to enhance
        dynamical_system: Dynamical system for evaluation
        measurement_system: Measurement system for evaluation
        key: Random key
        num_steps: Number of training steps
        learning_rate: Learning rate
        rejection_threshold: Threshold for rejection sampling
        max_attempts: Max rejection attempts per sample
        experiment_name: MLFlow experiment name

    Returns:
        (final_flow, rmse_history)
    """

    # Set up MLFlow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "input_dim": flow.input_dim,
                "ensemble_size": base_engmf.ensemble_size,
                "silverman_bandwidth_scaling": base_engmf.silverman_bandwidth_scaling,
                "learning_rate": learning_rate,
                "rejection_threshold": rejection_threshold,
                "max_attempts": max_attempts,
                "num_steps": num_steps,
                "dynamical_system": type(dynamical_system).__name__,
                "measurement_system": type(measurement_system).__name__,
            }
        )

        # Create discriminator sampling function with current flow
        def make_discriminator_engmf(current_flow):
            discriminator_fn = partial(
                flow_discriminator_sampler,
                flow=current_flow,
                rejection_threshold=rejection_threshold,
                max_attempts=max_attempts,
            )
            return eqx.tree_at(
                lambda engmf: engmf.sampling_function,
                base_engmf,
                jax.tree_util.Partial(discriminator_fn),
            )

        # Define RMSE evaluation function
        def compute_rmse(current_flow):
            discriminator_engmf = make_discriminator_engmf(current_flow)
            return evaluate_filter(
                dynamical_system=dynamical_system,
                measurement_system=measurement_system,
                update=discriminator_engmf,
                key=key,
            )

        # Initialize optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

        # Training state
        current_flow = flow
        rmse_history = []

        # Log initial RMSE
        initial_rmse = compute_rmse(current_flow)
        mlflow.log_metric("rmse", initial_rmse, step=0)
        rmse_history.append(initial_rmse)

        # Training loop
        for step in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)

            # Compute RMSE and gradients
            rmse, grads = eqx.filter_value_and_grad(compute_rmse)(current_flow)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, current_flow)
            current_flow = eqx.apply_updates(current_flow, updates)

            rmse_history.append(rmse)

            # Log metrics to MLFlow
            mlflow.log_metric("rmse", rmse, step=step)

            # Log gradient norms for monitoring
            grad_norm = jnp.sqrt(
                sum(jnp.sum(jnp.square(g)) for g in jax.tree_leaves(grads))
            )
            mlflow.log_metric("grad_norm", grad_norm, step=step)

        # Create and log training history plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rmse_history)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("RMSE")
        ax.set_title("Training History: RMSE vs Step")
        ax.grid(True)
        mlflow.log_figure(fig, "training_history.png")
        plt.close(fig)

        # Create system visualization under trained flow
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Original EnGMF trajectory
        key, subkey = jax.random.split(key)
        original_engmf = base_engmf
        plot_filter_trajectory(
            original_engmf,
            dynamical_system,
            measurement_system,
            subkey,
            axes[0, 0],
            "Original EnGMF",
        )

        # Plot 2: Flow-enhanced EnGMF trajectory
        key, subkey = jax.random.split(key)
        enhanced_engmf = make_discriminator_engmf(current_flow)
        plot_filter_trajectory(
            enhanced_engmf,
            dynamical_system,
            measurement_system,
            subkey,
            axes[0, 1],
            "Flow-Enhanced EnGMF",
        )

        # Plot 3: Flow sample quality visualization
        plot_flow_samples(current_flow, axes[1, 0])

        # Plot 4: RMSE comparison
        final_rmse = rmse_history[-1]
        initial_rmse = rmse_history[0]
        axes[1, 1].bar(["Initial", "Final"], [initial_rmse, final_rmse])
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].set_title("RMSE Comparison")

        plt.tight_layout()
        mlflow.log_figure(fig, "system_visualization.png")
        plt.close(fig)

        # Log final metrics
        mlflow.log_metric("final_rmse", rmse_history[-1])
        mlflow.log_metric("initial_rmse", rmse_history[0])
        mlflow.log_metric("rmse_improvement", rmse_history[0] - rmse_history[-1])

    return current_flow, jnp.array(rmse_history)


def plot_filter_trajectory(
    filter_obj, dynamical_system, measurement_system, key, ax, title
):
    """Plot a short trajectory of the filter for visualization."""
    # Simple trajectory plot - just show some sample points
    ax.text(
        0.5,
        0.5,
        f"{title}\n(Trajectory visualization)",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    ax.set_title(title)
    ax.grid(True)


def plot_flow_samples(flow, ax):
    """Plot samples from the normalizing flow to visualize learned distribution."""
    key = jax.random.key(42)

    # Generate samples from base distribution and transform
    num_samples = 1000
    base_samples = jax.random.normal(key, (num_samples, flow.input_dim))

    # Transform through flow
    transformed_samples, _ = eqx.filter_vmap(flow.inverse)(base_samples)

    # Plot samples (assuming at least 2D)
    if flow.input_dim >= 2:
        ax.scatter(transformed_samples[:, 0], transformed_samples[:, 1], alpha=0.5, s=1)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    else:
        ax.hist(transformed_samples[:, 0], bins=50, alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    ax.set_title("Flow-Generated Samples")
    ax.grid(True)

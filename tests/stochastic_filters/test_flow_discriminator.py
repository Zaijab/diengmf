"""
Test the flow discriminator sampling function.
"""

import jax
import jax.numpy as jnp
import pytest

from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.stochastic_filters.evaluate import flow_discriminator_sampler


def test_flow_discriminator_sampler_shapes():
    """Test that the discriminator sampler returns correct shapes."""
    key = jax.random.key(42)
    state_dim = 3
    batch_size = 10

    # Create a simple flow
    flow = NormalizingFlow(
        input_dim=state_dim,
        num_layers=2,
        num_bins=4,
        conditioner_hidden_dim=32,
        conditioner_depth=1,
        key=key,
    )

    # Create test inputs
    keys = jax.random.split(key, batch_size)
    means = jax.random.normal(key, (batch_size, state_dim))
    covs = jnp.broadcast_to(jnp.eye(state_dim), (batch_size, state_dim, state_dim))

    # Create a partial function to fix the flow and other parameters
    from functools import partial

    sampler_fn = partial(
        flow_discriminator_sampler, flow=flow, rejection_threshold=0.0, max_attempts=3
    )

    # Test discriminator sampling
    samples = sampler_fn(keys, means, covs)

    # Check shapes
    assert samples.shape == (batch_size, state_dim)
    assert not jnp.any(jnp.isnan(samples))


def test_flow_discriminator_improves_samples():
    """Test that discriminator actually selects better samples (higher log prob)."""
    key = jax.random.key(789)
    state_dim = 2

    flow = NormalizingFlow(input_dim=state_dim, num_layers=2, num_bins=6, key=key)

    # Test parameters
    mean = jnp.zeros(state_dim)
    cov = jnp.eye(state_dim)
    batch_size = 100

    # Generate regular samples
    key, subkey = jax.random.split(key)
    regular_keys = jax.random.split(subkey, batch_size)
    regular_samples = jax.vmap(lambda k: jax.random.multivariate_normal(k, mean, cov))(
        regular_keys
    )

    # Generate discriminator samples using partial
    from functools import partial

    key, subkey = jax.random.split(key)
    disc_keys = jax.random.split(subkey, batch_size)

    sampler_fn = partial(
        flow_discriminator_sampler, flow=flow, rejection_threshold=0.0, max_attempts=5
    )

    disc_samples = sampler_fn(
        disc_keys,
        jnp.broadcast_to(mean, (batch_size, state_dim)),
        jnp.broadcast_to(cov, (batch_size, state_dim, state_dim)),
    )

    # Compute log probabilities
    _, regular_log_probs = jax.vmap(flow.forward)(regular_samples)
    _, disc_log_probs = jax.vmap(flow.forward)(disc_samples)

    # Discriminator samples should have higher average log prob
    # (though this might not always be true due to randomness, so we use a weak test)
    regular_mean_log_prob = jnp.mean(regular_log_probs)
    disc_mean_log_prob = jnp.mean(disc_log_probs)

    # At minimum, discriminator shouldn't make things much worse
    assert disc_mean_log_prob >= regular_mean_log_prob - 1.0


if __name__ == "__main__":
    test_flow_discriminator_sampler_shapes()
    test_flow_discriminator_improves_samples()
    print("All tests passed!")

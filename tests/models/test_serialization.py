import json
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.models.serialization import load_model, save_model


@pytest.fixture
def test_key():
    return jax.random.key(42)


@pytest.fixture
def sample_hyperparams():
    return {
        "input_dim": 3,
        "num_layers": 2,
        "num_bins": 4,
        "conditioner_hidden_dim": 32,
        "conditioner_depth": 2,
        "activation_function": jax.nn.gelu,  # Keep as callable
    }


def test_save_and_load_model(test_key, sample_hyperparams):
    """Test basic save and load functionality."""
    # Create model
    original_model = NormalizingFlow(key=test_key, **sample_hyperparams)

    # Test forward pass to get some reference values
    test_input = jax.random.normal(test_key, (3,))
    original_output, original_logdet = original_model.forward(test_input)

    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
        temp_path = f.name

    try:
        # Save model (activation function will be filtered out automatically)
        save_model(temp_path, sample_hyperparams, original_model)

        # Load model
        loaded_model = load_model(temp_path)

        # Test that loaded model produces same output
        loaded_output, loaded_logdet = loaded_model.forward(test_input)

        assert jnp.allclose(original_output, loaded_output, atol=1e-6)
        assert jnp.allclose(original_logdet, loaded_logdet, atol=1e-6)

    finally:
        Path(temp_path).unlink()


def test_hyperparameter_preservation(test_key, sample_hyperparams):
    """Test that non-callable hyperparameters are correctly preserved in the file."""
    original_model = NormalizingFlow(key=test_key, **sample_hyperparams)

    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
        temp_path = f.name

    try:
        # Save model
        save_model(temp_path, sample_hyperparams, original_model)

        # Read hyperparameters directly from file
        with open(temp_path, "rb") as f:
            saved_hyperparams = json.loads(f.readline().decode())

        # Check that scalar hyperparameters match (callable excluded)
        assert saved_hyperparams["input_dim"] == sample_hyperparams["input_dim"]
        assert saved_hyperparams["num_layers"] == sample_hyperparams["num_layers"]
        assert saved_hyperparams["num_bins"] == sample_hyperparams["num_bins"]

        # activation_function should not be in saved hyperparams
        assert "activation_function" not in saved_hyperparams

    finally:
        Path(temp_path).unlink()


def test_model_invertibility_preserved(test_key, sample_hyperparams):
    """Test that invertibility is preserved after save/load."""
    original_model = NormalizingFlow(key=test_key, **sample_hyperparams)

    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
        temp_path = f.name

    try:
        # Save model
        save_model(temp_path, sample_hyperparams, original_model)

        # Load model
        loaded_model = load_model(temp_path)

        # Test invertibility
        test_input = jax.random.normal(test_key, (3,))

        # Forward then inverse
        y, fwd_logdet = loaded_model.forward(test_input)
        x_recon, inv_logdet = loaded_model.inverse(y)

        assert jnp.allclose(test_input, x_recon, atol=1e-5)
        assert jnp.allclose(fwd_logdet + inv_logdet, 0.0, atol=1e-5)

    finally:
        Path(temp_path).unlink()


def test_batch_processing_preserved(test_key, sample_hyperparams):
    """Test that batch processing works after save/load."""
    original_model = NormalizingFlow(key=test_key, **sample_hyperparams)

    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
        temp_path = f.name

    try:
        # Save model
        save_model(temp_path, sample_hyperparams, original_model)

        # Load model
        loaded_model = load_model(temp_path)

        # Test batch processing
        batch_input = jax.random.normal(test_key, (10, 3))

        # Original model batch forward
        original_batch_outputs = eqx.filter_vmap(original_model.forward)(batch_input)

        # Loaded model batch forward
        loaded_batch_outputs = eqx.filter_vmap(loaded_model.forward)(batch_input)

        assert jnp.allclose(
            original_batch_outputs[0], loaded_batch_outputs[0], atol=1e-6
        )
        assert jnp.allclose(
            original_batch_outputs[1], loaded_batch_outputs[1], atol=1e-6
        )

    finally:
        Path(temp_path).unlink()


def test_jit_compatibility_preserved(test_key, sample_hyperparams):
    """Test that JIT compilation works after save/load."""
    original_model = NormalizingFlow(key=test_key, **sample_hyperparams)

    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
        temp_path = f.name

    try:
        # Save model
        save_model(temp_path, sample_hyperparams, original_model)

        # Load model
        loaded_model = load_model(temp_path)

        # JIT compile loaded model
        jit_forward = eqx.filter_jit(loaded_model.forward)
        jit_inverse = eqx.filter_jit(loaded_model.inverse)

        test_input = jax.random.normal(test_key, (3,))

        # Test JIT forward
        y, fwd_logdet = jit_forward(test_input)
        x_recon, inv_logdet = jit_inverse(y)

        assert jnp.allclose(test_input, x_recon, atol=1e-5)
        assert jnp.allclose(fwd_logdet + inv_logdet, 0.0, atol=1e-5)

    finally:
        Path(temp_path).unlink()

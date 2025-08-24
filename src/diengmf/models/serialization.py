"""
Model serialization utilities for normalizing flows.

Following the Equinox serialization pattern with hyperparameters.
"""

import json

import equinox as eqx
import jax
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from diengmf.models.normalizing_flow import NormalizingFlow


@jaxtyped(typechecker=typechecker)
def save_model(filename: str, hyperparams: dict, model: NormalizingFlow) -> None:
    """
    Save normalizing flow model with hyperparameters.

    Args:
        filename: Path to save the model
        hyperparams: Dictionary of hyperparameters used to create the model
        model: Trained model to serialize
    """
    with open(filename, "wb") as f:
        # Only save JSON-serializable hyperparameters
        serializable_params = {k: v for k, v in hyperparams.items() if not callable(v)}
        hyperparam_str = json.dumps(serializable_params)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


@jaxtyped(typechecker=typechecker)
def load_model(filename: str) -> NormalizingFlow:
    """
    Load normalizing flow model with hyperparameters.

    Args:
        filename: Path to the saved model

    Returns:
        Loaded model
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = NormalizingFlow(key=jax.random.key(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from diengmf.losses import kl_divergence, make_step
from diengmf.models.invertible_linear_layer import PLULinear
from diengmf.models.normalizing_flow import NormalizingFlow

# Comprehensive hyperparameter grid for PLU layer optimization
PLU_HYPERPARAMETER_GRID = [
    # Standard configurations
    {"use_bias": True, "initialization_scale": 0.2, "permutation_type": "random"},
    {"use_bias": False, "initialization_scale": 0.1, "permutation_type": "random"},
    {"use_bias": True, "initialization_scale": 0.5, "permutation_type": "identity"},
    {"use_bias": True, "initialization_scale": 0.05, "permutation_type": "reverse"},
    # Different initialization scales
    {"use_bias": True, "initialization_scale": 1.0, "permutation_type": "random"},
    {"use_bias": False, "initialization_scale": 0.01, "permutation_type": "random"},
    # Different permutation strategies
    {"use_bias": True, "initialization_scale": 0.2, "permutation_type": "identity"},
    {"use_bias": True, "initialization_scale": 0.2, "permutation_type": "reverse"},
    # Conservative configurations
    {"use_bias": False, "initialization_scale": 0.02, "permutation_type": "identity"},
    # Extreme configurations (for robustness testing)
    {"use_bias": True, "initialization_scale": 2.0, "permutation_type": "random"},
]

DYNAMICAL_SYSTEMS = [
    ("Ikeda", lambda: Ikeda()),
    ("Lorenz63", lambda: Lorenz63()),
    ("Lorenz96", lambda: Lorenz96()),
]


@pytest.fixture
def test_keys():
    return jax.random.split(jax.random.key(42), 10)


@pytest.mark.parametrize("config", PLU_HYPERPARAMETER_GRID)
@pytest.mark.parametrize("system_name,system_factory", DYNAMICAL_SYSTEMS)
def test_plu_hyperparameter_invertibility(
    config, system_name, system_factory, test_keys
):
    """Test invertibility across different hyperparameter configurations and dynamical systems."""
    system = system_factory()
    key = test_keys[0]

    # Create PLU with configuration
    plu = PLULinear(input_dim=system.dimension, key=key, **config)

    # Test single point invertibility
    x_single = jax.random.normal(test_keys[1], (system.dimension,))
    y, fwd_logdet = plu.forward(x_single)
    x_recon, inv_logdet = plu.inverse(y)

    assert jnp.allclose(
        x_single, x_recon, atol=1e-5
    ), f"Failed for {system_name} with config {config}"
    assert jnp.allclose(
        fwd_logdet + inv_logdet, 0.0, atol=1e-5
    ), f"Logdet inconsistent for {system_name}"


@pytest.mark.parametrize("config", PLU_HYPERPARAMETER_GRID)
def test_plu_batch_invertibility(config, test_keys):
    """Test invertibility with batched inputs."""
    key = test_keys[0]
    input_dim = 3
    batch_size = 10

    plu = PLULinear(input_dim=input_dim, key=key, **config)

    # Test batch invertibility
    x_batch = jax.random.normal(test_keys[1], (batch_size, input_dim))
    y_batch, fwd_logdet_batch = eqx.filter_vmap(plu.forward)(x_batch)
    x_recon_batch, inv_logdet_batch = eqx.filter_vmap(plu.inverse)(y_batch)

    assert jnp.allclose(
        x_batch, x_recon_batch, atol=1e-5
    ), f"Batch invertibility failed with config {config}"
    assert jnp.allclose(
        fwd_logdet_batch + inv_logdet_batch, 0.0, atol=1e-5
    ), f"Batch logdet inconsistent"


@pytest.mark.parametrize(
    "config", PLU_HYPERPARAMETER_GRID[:5]
)  # Test subset for training
@pytest.mark.parametrize(
    "system_name,system_factory", DYNAMICAL_SYSTEMS[:2]
)  # Faster systems
def test_plu_in_normalizing_flow_training(
    config, system_name, system_factory, test_keys
):
    """Test PLU layer works within normalizing flow training loop."""
    system = system_factory()
    key = test_keys[2]

    # Create a simple normalizing flow with PLU layers
    model = NormalizingFlow(
        input_dim=system.dimension,
        num_layers=2,
        conditioner_hidden_dim=32,
        conditioner_depth=2,
        **{f"plu_{k}": v for k, v in config.items()},  # Prefix PLU params
        key=key,
    )

    # Generate training data
    batch = system.generate(test_keys[3], batch_size=50, final_time=jnp.asarray(5.0))

    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Train for several steps
    for i in range(10):
        batch = eqx.filter_vmap(system.flow, in_axes=(None, None, 0))(
            jnp.asarray(0.0), jnp.asarray(1.0), batch
        )
        loss, model, opt_state = make_step(model, batch, optim, opt_state)

        # Check for NaN/Inf
        assert jnp.isfinite(
            loss
        ), f"Loss became non-finite at step {i} for {system_name}"
        assert not jnp.isnan(loss), f"Loss became NaN at step {i} for {system_name}"

    # Test overall flow invertibility is maintained after training
    x_test = jax.random.normal(test_keys[4], (system.dimension,))
    y, fwd_logdet = model.forward(x_test)
    x_recon, inv_logdet = model.inverse(y)

    assert jnp.allclose(
        x_test, x_recon, atol=1e-4
    ), f"Training broke flow invertibility for {system_name}"


def test_plu_parameter_shapes_across_configs():
    """Test parameter shapes are correct across all configurations."""
    key = jax.random.key(123)

    for i, config in enumerate(PLU_HYPERPARAMETER_GRID):
        test_key = jax.random.fold_in(key, i)
        input_dim = 4  # Fixed for this test

        plu = PLULinear(input_dim=input_dim, key=test_key, **config)

        # Check shapes
        l_size = (input_dim * (input_dim - 1)) // 2
        u_upper_size = (input_dim * (input_dim - 1)) // 2

        assert plu.L_params.shape == (l_size,), f"L_params wrong shape for config {i}"
        assert plu.U_diag.shape == (input_dim,), f"U_diag wrong shape for config {i}"
        assert plu.U_upper.shape == (
            u_upper_size,
        ), f"U_upper wrong shape for config {i}"
        assert plu.P.shape == (input_dim,), f"P wrong shape for config {i}"

        if config["use_bias"]:
            assert plu.bias is not None and plu.bias.shape == (
                input_dim,
            ), f"Bias wrong shape for config {i}"
        else:
            assert plu.bias is None, f"Bias should be None for config {i}"

        # Check values are finite
        assert jnp.isfinite(plu.L_params).all(), f"L_params not finite for config {i}"
        assert jnp.isfinite(plu.U_diag).all(), f"U_diag not finite for config {i}"
        assert jnp.isfinite(plu.U_upper).all(), f"U_upper not finite for config {i}"

        if plu.bias is not None:
            assert jnp.isfinite(plu.bias).all(), f"Bias not finite for config {i}"


def test_plu_permutation_types():
    """Test different permutation types work correctly."""
    key = jax.random.key(456)
    input_dim = 5

    # Test identity permutation
    plu_identity = PLULinear(input_dim=input_dim, permutation_type="identity", key=key)
    expected_identity = jnp.arange(input_dim)
    assert jnp.array_equal(plu_identity.P, expected_identity)

    # Test reverse permutation
    plu_reverse = PLULinear(input_dim=input_dim, permutation_type="reverse", key=key)
    expected_reverse = jnp.arange(input_dim)[::-1]
    assert jnp.array_equal(plu_reverse.P, expected_reverse)

    # Test random permutation (should be valid permutation)
    plu_random = PLULinear(input_dim=input_dim, permutation_type="random", key=key)
    assert jnp.sort(plu_random.P).shape == (input_dim,)
    assert jnp.array_equal(jnp.sort(plu_random.P), jnp.arange(input_dim))


def test_plu_jit_compatibility():
    """Test that PLU layer works with JIT compilation."""
    key = jax.random.key(789)
    input_dim = 3

    plu = PLULinear(input_dim=input_dim, key=key)

    # JIT the forward and inverse operations
    jit_forward = eqx.filter_jit(plu.forward)
    jit_inverse = eqx.filter_jit(plu.inverse)

    x = jax.random.normal(key, (input_dim,))

    # Test JIT forward
    y, fwd_logdet = jit_forward(x)
    assert y.shape == x.shape
    assert fwd_logdet.shape == ()

    # Test JIT inverse
    x_recon, inv_logdet = jit_inverse(y)
    assert x_recon.shape == x.shape
    assert inv_logdet.shape == ()

    # Test invertibility with JIT
    assert jnp.allclose(x, x_recon, atol=1e-5)
    assert jnp.allclose(fwd_logdet + inv_logdet, 0.0, atol=1e-5)


def test_plu_glorot_initialization():
    """Test that PLU uses Glorot uniform initialization correctly."""
    key = jax.random.key(999)
    input_dim = 6

    # Create multiple PLU layers to test initialization variance
    plus = [
        PLULinear(input_dim=input_dim, key=jax.random.fold_in(key, i))
        for i in range(10)
    ]

    # Check that L_params and U_upper use reasonable initialization ranges
    l_params_all = jnp.concatenate([plu.L_params for plu in plus])
    u_upper_all = jnp.concatenate([plu.U_upper for plu in plus])

    # Glorot uniform should have reasonable variance
    l_std = jnp.std(l_params_all)
    u_std = jnp.std(u_upper_all)

    # Rough check that initialization is not too small or too large
    assert 0.1 < l_std < 2.0, f"L parameter std {l_std} seems unreasonable"
    assert 0.1 < u_std < 2.0, f"U parameter std {u_std} seems unreasonable"


# Generate Optuna-style configuration helper
def suggest_plu_config(trial):
    """Helper function for Optuna hyperparameter optimization."""
    return {
        "use_bias": trial.suggest_categorical("plu_use_bias", [True, False]),
        "initialization_scale": trial.suggest_float(
            "plu_init_scale", 0.01, 1.0, log=True
        ),
        "permutation_type": trial.suggest_categorical(
            "plu_permutation_type", ["random", "identity", "reverse"]
        ),
    }

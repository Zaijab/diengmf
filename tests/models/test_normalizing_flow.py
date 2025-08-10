import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, jaxtyped
from beartype import beartype as typechecker

from diengmf.models.normalizing_flow import NormalizingFlow
from diengmf.models.invertible_linear_layer import PLULinear
from diengmf.models.rational_quadratic_spline import RQSBijector
from diengmf.models.masked_coupling_layer import MaskedCoupling

@pytest.fixture
def flow_2d():
    key = jax.random.key(42)
    return NormalizingFlow(input_dim=2, num_layers=3, num_bins=8, key=key)

@pytest.fixture  
def flow_4d():
    key = jax.random.key(123)
    return NormalizingFlow(input_dim=4, num_layers=2, num_bins=6, key=key)

@pytest.fixture
def test_data_2d():
    key = jax.random.key(999)
    return jax.random.normal(key, (100, 2)) * 2.0

@pytest.fixture
def test_data_4d():
    key = jax.random.key(888)
    return jax.random.normal(key, (50, 4)) * 1.5

def test_forward_inverse_invertibility_2d(flow_2d, test_data_2d):
    """Test forward(inverse(x)) == x"""
    x = test_data_2d
    z, logdet_fwd = flow_2d.forward(x)
    x_recon, logdet_inv = flow_2d.inverse(z)
    
    max_error = jnp.max(jnp.abs(x - x_recon))
    assert max_error < 1e-14, f"Forward-inverse error: {max_error}"
    assert x.shape == x_recon.shape
    assert z.shape == x.shape

def test_inverse_forward_invertibility_2d(flow_2d, test_data_2d):
    """Test inverse(forward(x)) == x"""
    x = test_data_2d
    z, logdet_fwd = flow_2d.forward(x)
    x_recon, logdet_inv = flow_2d.inverse(z)
    
    max_error = jnp.max(jnp.abs(x - x_recon))
    assert max_error < 1e-14, f"Inverse-forward error: {max_error}"

def test_logdet_consistency_2d(flow_2d, test_data_2d):
    """Test logdet_forward + logdet_inverse ≈ 0"""
    x = test_data_2d
    z, logdet_fwd = flow_2d.forward(x)
    x_recon, logdet_inv = flow_2d.inverse(z)
    
    logdet_sum = logdet_fwd + logdet_inv
    max_logdet_error = jnp.max(jnp.abs(logdet_sum))
    assert max_logdet_error < 1e-12, f"Logdet consistency error: {max_logdet_error}"

def test_forward_inverse_invertibility_4d(flow_4d, test_data_4d):
    """Test forward(inverse(x)) == x for 4D"""
    x = test_data_4d
    z, logdet_fwd = flow_4d.forward(x)
    x_recon, logdet_inv = flow_4d.inverse(z)
    
    max_error = jnp.max(jnp.abs(x - x_recon))
    assert max_error < 1e-14, f"4D Forward-inverse error: {max_error}"

def test_logdet_consistency_4d(flow_4d, test_data_4d):
    """Test logdet consistency for 4D"""
    x = test_data_4d
    z, logdet_fwd = flow_4d.forward(x)
    x_recon, logdet_inv = flow_4d.inverse(z)
    
    logdet_sum = logdet_fwd + logdet_inv
    max_logdet_error = jnp.max(jnp.abs(logdet_sum))
    assert max_logdet_error < 1e-12, f"4D Logdet consistency error: {max_logdet_error}"

def test_manual_composition_equivalence():
    """Test NormalizingFlow matches manual layer composition"""
    key = jax.random.key(555)
    input_dim, num_layers = 3, 2
    
    flow = NormalizingFlow(input_dim=input_dim, num_layers=num_layers, key=key)
    
    keys = jax.random.split(key, 2 * num_layers)
    bijector = RQSBijector(range_min=-5.0, range_max=5.0)
    manual_couplings = [MaskedCoupling(input_dim, bijector, key=keys[i]) for i in range(num_layers)]
    manual_plus = [PLULinear(input_dim, True, key=keys[num_layers + i]) for i in range(num_layers)]
    
    x = jax.random.normal(jax.random.key(777), (20, input_dim))
    
    flow_z, flow_logdet = flow.forward(x)
    
    manual_x, manual_logdet = x, jnp.zeros(x.shape[:-1])
    for coupling, plu in zip(manual_couplings, manual_plus):
        manual_x, ldj = coupling.forward(manual_x); manual_logdet += ldj
        manual_x, ldj = plu.forward(manual_x); manual_logdet += ldj
    
    assert jnp.allclose(flow_z, manual_x, atol=1e-15)
    assert jnp.allclose(flow_logdet, manual_logdet, atol=1e-15)

def test_distrax_comparison():
    """Compare with distrax if available"""
    try:
        import distrax
        
        key = jax.random.key(321)
        input_dim = 2
        
        flow = NormalizingFlow(input_dim=input_dim, num_layers=1, key=key)
        
        def simple_conditioner(x):
            return jnp.concatenate([jnp.sum(x, axis=-1, keepdims=True)] * 2, axis=-1)
        
        def simple_bijector(params):
            scale, shift = params[..., 0], params[..., 1]
            return distrax.ScalarAffine(shift=shift, scale=jnp.exp(scale))
        
        distrax_coupling = distrax.MaskedCoupling(
            mask=jnp.array([True, False]),
            conditioner=simple_conditioner,
            bijector=simple_bijector
        )
        
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        
        our_z, our_logdet = flow.forward(x)
        our_x_recon, our_inv_logdet = flow.inverse(our_z)
        
        distrax_z, distrax_logdet = distrax_coupling.forward_and_log_det(x)
        distrax_x_recon, distrax_inv_logdet = distrax_coupling.inverse_and_log_det(distrax_z)
        
        assert jnp.allclose(our_x_recon, x, atol=1e-14)
        assert jnp.allclose(distrax_x_recon, x, atol=1e-14)
        assert abs(our_logdet[0] + our_inv_logdet[0]) < 1e-12
        assert abs(distrax_logdet[0] + distrax_inv_logdet[0]) < 1e-12
        
    except ImportError:
        pytest.skip("distrax not available")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    key = jax.random.key(111)
    flow = NormalizingFlow(input_dim=2, num_layers=1, key=key)
    
    zero_input = jnp.zeros((1, 2))
    z_zero, logdet_zero = flow.forward(zero_input)
    x_zero_recon, logdet_zero_inv = flow.inverse(z_zero)
    assert jnp.allclose(zero_input, x_zero_recon, atol=1e-14)
    
    large_input = jnp.array([[100.0, -100.0]])
    z_large, logdet_large = flow.forward(large_input)
    x_large_recon, logdet_large_inv = flow.inverse(z_large)
    assert jnp.allclose(large_input, x_large_recon, atol=1e-12)

@pytest.mark.parametrize("input_dim,num_layers", [(2, 1), (3, 2), (4, 3), (8, 1)])
def test_parametric_invertibility(input_dim, num_layers):
    """Parametric test across dimensions and layer counts"""
    key = jax.random.key(input_dim * num_layers)
    flow = NormalizingFlow(input_dim=input_dim, num_layers=num_layers, key=key)
    
    x = jax.random.normal(jax.random.key(input_dim + num_layers), (25, input_dim))
    z, logdet_fwd = flow.forward(x)
    x_recon, logdet_inv = flow.inverse(z)
    
    max_error = jnp.max(jnp.abs(x - x_recon))
    logdet_error = jnp.max(jnp.abs(logdet_fwd + logdet_inv))
    
    assert max_error < 1e-14, f"Dim {input_dim}, Layers {num_layers}: error {max_error}"
    assert logdet_error < 1e-12, f"Dim {input_dim}, Layers {num_layers}: logdet error {logdet_error}"

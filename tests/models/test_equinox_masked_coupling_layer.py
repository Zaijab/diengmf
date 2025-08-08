def test_rqs_coupling():
    key = jax.random.key(42)
    
    for input_dim in [2, 3, 40]:
        print(f"\n{'='*50}")
        print(f"Testing RQS coupling with input_dim={input_dim}")
        print(f"{'='*50}")
        
        key, subkey = jax.random.split(key)
        layer = MaskedCouplingRQS(input_dim, num_bins=4, key=subkey)
        
        x = jax.random.normal(subkey, (5, input_dim))
        y, fwd_logdet = layer.forward(x)
        x_rec, inv_logdet = layer.inverse(y)
        
        assert jnp.allclose(x_rec, x, atol=1e-4)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-4)
        print(f"✓ RQS test passed for dim={input_dim}")


def test_affine_coupling():
    key = jax.random.key(42)
    
    for input_dim in [2, 3, 40]:
        print(f"\n{'='*50}")
        print(f"Testing Affine coupling with input_dim={input_dim}")
        print(f"{'='*50}")
        
        key, subkey = jax.random.split(key)
        layer = MaskedCouplingAffine(input_dim, key=subkey)
        
        x = jax.random.normal(subkey, (10, input_dim))
        y, fwd_logdet = layer.forward(x)
        x_rec, inv_logdet = layer.inverse(y)
        
        assert jnp.allclose(x_rec, x, atol=1e-5)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-5)
        print(f"✓ Affine test passed for dim={input_dim}")


test_rqs_coupling()
test_affine_coupling()
print("\n✓ Both RQS and Affine coupling tests pass!")


def test_invertibility():
    key = jax.random.key(42)
    
    for input_dim in [2, 3, 40]:
        key, subkey = jax.random.split(key)
        layer = MaskedCouplingLayer(input_dim, key=subkey)
        
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (10, input_dim))
        
        y, fwd_logdet = layer.forward(x)
        x_rec, inv_logdet = layer.inverse(y)
        
        assert jnp.allclose(x_rec, x, atol=1e-5)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-5)


def test_distrax_affine():
    key = jax.random.key(42)
    key_x, key_mask, key_params = jax.random.split(key, 3)
    
    input_dim = 4
    x = jax.random.normal(key_x, (5, input_dim))
    mask = jnp.array([True, True, False, False])
    
    shift_params = jax.random.normal(key_params, (input_dim,))
    scale_params = jnp.exp(jax.random.normal(key_params, (input_dim,)) * 0.1)
    
    distrax_layer = distrax.MaskedCoupling(
        mask=mask,
        conditioner=lambda _: (shift_params * ~mask, jnp.log(scale_params * ~mask)),
        bijector=lambda p: distrax.ScalarAffine(shift=p[0], log_scale=p[1])
    )
    
    y_distrax, logdet_distrax = distrax_layer.forward_and_log_det(x)
    
    y_manual = x * mask + (x * scale_params + shift_params) * ~mask
    logdet_manual = jnp.sum(jnp.log(scale_params) * ~mask, axis=-1)
    
    assert jnp.allclose(y_distrax, y_manual, atol=1e-6)
    assert jnp.allclose(logdet_distrax, logdet_manual, atol=1e-6)


def test_symbolic_check():
    input_dim = 2
    key = jax.random.key(0)
    layer = MaskedCouplingLayer(input_dim, hidden_dim=8, key=key)
    
    epsilon = 1e-7
    x = jnp.array([[0.5, 0.5]])
    
    y, analytical_logdet = layer.forward(x)
    
    def compute_numerical_jacobian(x_point):
        jacobian = jnp.zeros((input_dim, input_dim))
        for i in range(input_dim):
            x_plus = x_point.at[0, i].set(x_point[0, i] + epsilon)
            x_minus = x_point.at[0, i].set(x_point[0, i] - epsilon)
            y_plus = layer.forward(x_plus)[0]
            y_minus = layer.forward(x_minus)[0]
            jacobian = jacobian.at[:, i].set((y_plus[0] - y_minus[0]) / (2 * epsilon))
        return jacobian
    
    numerical_jac = compute_numerical_jacobian(x)
    numerical_logdet = jnp.log(jnp.abs(jnp.linalg.det(numerical_jac)))
    
    assert jnp.allclose(analytical_logdet[0], numerical_logdet, atol=1e-4)


def test_training_loop_compatibility():
    from diengmf.dynamical_systems import Ikeda
    from diengmf.losses import make_step
    import optax
    
    key = jax.random.key(0)
    model = MaskedCouplingLayer(input_dim=2, hidden_dim=32, key=key)
    
    system = Ikeda(batch_size=25)
    batch = system.generate(jax.random.key(0), batch_size=100)
    
    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    for _ in range(10_000):
        batch = eqx.filter_vmap(system.forward)(batch)
        loss, model, opt_state = make_step(model, batch, optim, opt_state)

        if (_ % 500) == 0:
            print(loss)
    
    assert isinstance(loss, jnp.ndarray)


def test_mask_preservation():
    key = jax.random.key(42)
    layer = MaskedCouplingLayer(input_dim=4, key=key)
    
    x = jax.random.normal(key, (10, 4))
    y, _ = layer.forward(x)
    
    mask_indices = jnp.where(layer.mask == 1.0)[0]
    assert jnp.allclose(x[:, mask_indices], y[:, mask_indices])
    
    x_rec, _ = layer.inverse(y)
    assert jnp.allclose(x[:, mask_indices], x_rec[:, mask_indices])


def test_batched_operations():
    key = jax.random.key(42)
    
    for batch_size in [1, 10, 100]:
        for input_dim in [2, 3, 40]:
            key, subkey = jax.random.split(key)
            layer = MaskedCouplingLayer(input_dim, key=subkey)
            
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (batch_size, input_dim))
            
            y, fwd_logdet = layer.forward(x)
            x_rec, inv_logdet = layer.inverse(y)
            
            assert y.shape == (batch_size, input_dim)
            assert fwd_logdet.shape == (batch_size,)
            assert jnp.allclose(x_rec, x, atol=1e-5)


test_invertibility()
test_distrax_affine()
test_symbolic_check()
test_mask_preservation()
test_batched_operations()
test_training_loop_compatibility()

print("✓ All masked coupling layer tests pass!")

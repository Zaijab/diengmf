def test_rational_quadratic_spline():
    """Test multi-dimensional RQS for training loop compatibility."""
    key = jax.random.key(42)
    
    # Test different input dimensions
    for input_dim in [2, 3, 40]:
        key, subkey = jax.random.split(key)
        spline = RationalQuadraticSpline(input_dim, num_bins=4, key=subkey)
        
        key, subkey = jax.random.split(key)
        x_test = jax.random.normal(subkey, (10, input_dim))
        
        y, fwd_logdet = spline.forward(x_test)
        x_reconstructed, inv_logdet = spline.inverse(y)
        
        assert jnp.allclose(x_reconstructed, x_test, atol=1e-4)
        assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-4)
        assert y.shape == x_test.shape
        assert fwd_logdet.shape == (10,)


test_rational_quadratic_spline()

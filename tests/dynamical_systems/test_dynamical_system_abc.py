import jax
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96

def test_dynamical_system_abc():
    dynamical_systems = [Ikeda(), Lorenz63(), Lorenz96()]
    
    for dynamical_system in dynamical_systems:
        initial_state = dynamical_system.initial_state()
        initial_state = dynamical_system.flow(0.0, 10.0, initial_state)
        batch = dynamical_system.generate(jax.random.key(0))

    assert True

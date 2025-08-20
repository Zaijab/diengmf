from diengmf.losses.stochastic_filters import evaluate_filter
from diengmf.dynamical_systems import AbstractDynamicalSystem, Ikeda
from diengmf.measurement_systems import AbstractMeasurementSystem, RangeSensor
from diengmf.stochastic_filters import AbstractFilter
from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import equinox as eqx
import jax
import distrax

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from typing import Callable
from diengmf.stochastic_filters import AbstractFilter
from diengmf.measurement_systems import AbstractMeasurementSystem
from diengmf.dynamical_systems import AbstractDynamicalSystem
from diengmf.statistics import GMM

###

###

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def filter_update(
    carry: tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    key: Key[Array, "..."],
    dynamical_system: AbstractDynamicalSystem,
    measurement_system: AbstractMeasurementSystem,
    update: Callable[
        [
            Key[Array, "..."],
            Float[Array, "batch_size state_dim"],
            Float[Array, "measurement_dim"],
            AbstractMeasurementSystem,
            bool,
        ],
        Float[Array, "batch_size state_dim"],
    ],
    debug: bool = False,
) -> tuple[
    tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    Float[Array, "state_dim"],
]:
    (prior_ensemble, true_state) = carry
    ensemble_updating_key, measurement_key = jax.random.split(key)
    updated_ensemble = update(
        prior_ensemble=prior_ensemble,
        measurement=measurement_system(true_state, measurement_key),
        measurement_system=measurement_system,
        key=ensemble_updating_key,
    )
    error = true_state - jnp.mean(updated_ensemble, axis=0)
    if debug:
        jax.debug.callback(plot_update, prior_ensemble, updated_ensemble, true_state)
    ensemble_next = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, updated_ensemble)
    true_state_next = dynamical_system.flow(0.0, 1.0, true_state)
    new_carry = (ensemble_next, true_state_next)
    return new_carry, error


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_filter(
    key: Key[Array, "..."],
    stochastic_filter: AbstractFilter,
    debug: bool = False,
) -> Float[Array, "..."]:
    ensemble_size = 100
    burn_in_time = 100
    measurement_time = 1000
    total_steps = burn_in_time + measurement_time

    key, subkey = jax.random.split(key)
    initial_ensemble = dynamical_system.generate(subkey, batch_size=ensemble_size)
    initial_true_state = dynamical_system.initial_state()

    keys = jax.random.split(key, num=(total_steps,))

    scan_step = jax.tree_util.Partial(
        filter_update,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        update=stochastic_filter.update,
    )

    (final_carry, errors_over_time) = jax.lax.scan(
        scan_step, (initial_ensemble, initial_true_state), keys
    )

    errors_past_burn_in = errors_over_time[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))

    return rmse

###

# Possibly use generics

class FilterExperiment(eqx.Module):
    dynamical_system: AbstractDynamicalSystem
    measurement_system: AbstractMeasurementSystem
    stochastic_filter: AbstractFilter

    

###
dynamical_system: AbstractDynamicalSystem = Ikeda()
measurement_system: AbstractMeasurementSystem = RangeSensor()
stochastic_filter: AbstractFilter = EnGMF(dynamical_system=dynamical_system, measurement_system=measurement_system)

@eqx.filter_jit
def evaluate_filter_function(stochastic_filter: AbstractFilter):
    """
    Initialize Filtering Loop
    """
    true_state = stochastic_filter.dynamical_system.initial_state()
    belief = stochastic_filter.initialize(key, initial_belief)

    """
    Actual Filtering Body
    """
    return belief

true_state = dynamical_system.initial_state()
key = jax.random.key(10)
key, subkey = jax.random.split(key)

@eqx.filter_jit
def f(carry, _):
    key, belief = carry
    true_state = stochastic_filter.dynamical_system.flow()
    belief = stochastic_filter.predict(belief, 0.0, 1.0)
    belief = stochastic_filter.update(key, belief, jnp.array([0.0]))
    return (key, belief), None

key = jax.random.key(10)
key, subkey = jax.random.split(key)

initial_belief = MultivariateNormalTri(jnp.zeros(2), 1/2 * jnp.eye(2))
(final_key, final_belief), _ = jax.lax.scan(f, (key, stochastic_filter.initialize(key, initial_belief)), length=10)

final_belief


###


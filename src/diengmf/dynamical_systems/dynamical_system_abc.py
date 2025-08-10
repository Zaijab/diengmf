"""
This module describes dynamical systems for the express purpose of evaluating stochastic filtering algorithms.
The ABC structure allows the user to define their choice of dynamical system to reduce code duplication.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import abc
from jaxtyping import Array, Float, Key, jaxtyped, Shaped
from beartype import beartype as typechecker
from diffrax import (
    SaveAt,
    ODETerm,
    diffeqsolve,
    AbstractSolver,
    AbstractStepSizeController,
)


class AbstractDynamicalSystem(eqx.Module, strict=True):
    """
    Abstract base class for dynamical systems in stochastic filtering.
    """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the state space."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        """
        Return a default initial state.
        Many dynamical systems have a cannonical / useful state that they start from.
        We have `None` act at this singular state and a `jax.random.key` will initialize the point in a random manner.
        This will be useful for generating points in an attractor if need be.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def trajectory(
        self,
        initial_time: float | Shaped[Array, ""] | int,
        final_time: float | Shaped[Array, ""] | int,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
    ) -> tuple[Float[Array, "..."], Float[Array, "... state_dim"]]:
        """
        Solve for the trajectory given boundary times (and how many points to save).
        Return the times and corresponding solutions.
        """
        raise NotImplementedError

    def flow(
        self,
        initial_time: float | Shaped[Array, ""] | int,
        final_time: float | Shaped[Array, ""] | int,
        state: Float[Array, "state_dim"],
    ) -> Float[Array, "state_dim"]:
        """
        Trajectory with SaveAt = t1.
        Returns the y value at t1
        """
        _, states = self.trajectory(
            initial_time=initial_time,
            final_time=final_time,
            state=state,
            saveat=SaveAt(t1=True),
        )
        return states[-1]

    def orbit(
        self,
        initial_time: float | Shaped[Array, ""] | int,
        final_time: float | Shaped[Array, ""] | int,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ) -> Shaped[Array, "state_dim"]:
        """
        Trajectory but just return ys.
        """
        _, states = self.trajectory(initial_time, final_time, state, saveat)
        return states

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def generate(
        self,
        key: Key[Array, "..."],
        batch_size: int = 50,
        final_time: float | int = 0.0,
    ) -> Shaped[Array, "{batch_size} state_dim"]:
        keys = jax.random.split(key, batch_size)
        initial_states = eqx.filter_vmap(self.initial_state)(keys)
        final_states = eqx.filter_vmap(self.flow)(0.0, final_time, initial_states)
        return final_states


class AbstractContinuousDynamicalSystem(AbstractDynamicalSystem, strict=True):
    """ """

    @abc.abstractmethod
    def vector_field():
        raise NotImplementedError

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float | Shaped[Array, ""] | int,
        final_time: float | Shaped[Array, ""] | int,
        state: Shaped[Array, "{self.dimension}"],
        saveat: SaveAt,
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "... {self.dimension}"]]:
        """Integrate a single point forward in time."""

        sol = diffeqsolve(
            terms=ODETerm(self.vector_field),
            solver=self.solver,
            t0=initial_time,
            t1=final_time,
            dt0=self.dt,
            y0=state,
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
            max_steps=100_000_000,
        )
        return sol.ts, sol.ys


import jax
import jax.numpy as jnp
import equinox as eqx
import abc
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from diffrax import (
    SaveAt,
    ODETerm,
    diffeqsolve,
    AbstractSolver,
    AbstractStepSizeController,
)


class AbstractDynamicalSystem(eqx.Module, strict=True):
    """
    Abstract base class for dynamical systems in stochastic filtering.
    """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the state space."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Shaped[Array, "state_dim"]:
        """
        Return a default initial state.
        Many dynamical systems have a cannonical / useful state that they start from.
        We have `None` act at this singular state and a `jax.random.key` will initialize the point in a random manner.
        This will be useful for generating points in an attractor if need be.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "... state_dim"]]:
        """
        Solve for the trajectory given boundary times (and how many points to save).
        Return the times and corresponding solutions.
        """
        raise NotImplementedError

    def flow(
        self,
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
    ) -> Shaped[Array, "state_dim"]:
        """
        Trajectory with SaveAt = t1.
        Returns the y value at t1
        """
        _, states = self.trajectory(
            initial_time=initial_time,
            final_time=final_time,
            state=state,
            saveat=SaveAt(t1=True),
        )
        return states[-1]

    def orbit(
        self,
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ) -> Shaped[Array, "state_dim"]:
        """
        Trajectory but just return ys.
        """
        _, states = self.trajectory(initial_time, final_time, state, saveat)
        return states

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def generate(
        self,
        key: Key[Array, "..."],
        batch_size: int = 1000,
        final_time: float | int = 100.0,
    ) -> Shaped[Array, "{batch_size} state_dim"]:
        keys = jax.random.split(key, batch_size)
        initial_states = eqx.filter_vmap(self.initial_state)(keys)
        final_states = eqx.filter_vmap(self.flow)(0.0, final_time, initial_states)
        return final_states


class AbstractContinuousDynamicalSystem(AbstractDynamicalSystem, strict=True):
    """ """

    @abc.abstractmethod
    def vector_field():
        raise NotImplementedError

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "{self.dimension}"],
        saveat: SaveAt,
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "... {self.dimension}"]]:
        """Integrate a single point forward in time."""

        sol = diffeqsolve(
            terms=ODETerm(self.vector_field),
            solver=self.solver,
            t0=initial_time,
            t1=final_time,
            dt0=self.dt,
            y0=state,
            stepsize_controller=self.stepsize_contoller,
            saveat=saveat,
            max_steps=10_000,
        )
        return sol.ts, sol.ys


class AbstractDiscreteDynamicalSystem(AbstractDynamicalSystem, strict=True):

    @abc.abstractmethod
    def forward():
        raise NotImplementedError

    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ):
        """
        This function computes the trajectory for a discrete system.
        It returns the tuple of the times and
        """
        assert initial_time <= final_time, "This is a discrete system without inverse."

        if saveat.subs.steps:
            safe_initial_time = (
                jnp.atleast_1d(initial_time) if saveat.subs.t0 else jnp.array([])
            )
            safe_dense = jnp.arange(initial_time, final_time) + 1
            xs = jnp.concatenate([safe_initial_time, safe_dense])
        else:
            safe_initial_time = (
                jnp.atleast_1d(initial_time) if saveat.subs.t0 else jnp.array([])
            )
            safe_final_time = (
                jnp.atleast_1d(final_time) if saveat.subs.t1 else jnp.array([])
            )
            safe_array = jnp.array([]) if saveat.subs.ts is None else saveat.subs.ts
            xs = jnp.concatenate([safe_initial_time, safe_array, safe_final_time])

        def body_fn(carry, x):
            """
            state = carry
            time = x
            """
            current_state, current_time = carry

            def sub_while_cond_fun(sub_carry):
                sub_state, sub_time = sub_carry
                return sub_time < x

            def sub_while_body_fun(sub_carry):
                sub_state, sub_time = sub_carry
                return (self.forward(sub_state), sub_time + 1)

            final_state, final_time = jax.lax.while_loop(
                sub_while_cond_fun, sub_while_body_fun, carry
            )

            return (final_state, final_time), final_state

        initial_carry = (state, 0)
        (final_state, final_time), states = jax.lax.scan(body_fn, initial_carry, xs)

        return xs, states


class AbstractInvertibleDiscreteDynamicalSystem(AbstractDynamicalSystem, strict=True):

    @abc.abstractmethod
    def forward():
        raise NotImplementedError

    @abc.abstractmethod
    def backward():
        raise NotImplementedError

    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ):
        """
        This function computes the trajectory for a discrete system.
        It returns the tuple of the times and
        """
        is_forward = final_time >= initial_time

        safe_initial_time = (
            jnp.atleast_1d(initial_time) if saveat.subs.t0 else jnp.array([])
        )
        safe_final_time = (
            jnp.atleast_1d(final_time) if saveat.subs.t1 else jnp.array([])
        )
        safe_array = jnp.array([]) if saveat.subs.ts is None else saveat.subs.ts
        xs = jnp.concatenate([safe_initial_time, safe_array, safe_final_time])
        xs = jnp.sort(xs) if is_forward else jnp.sort(xs)[::-1]

        def body_fn(carry, x):
            """
            state = carry
            time = x
            """
            current_state, current_time = carry

            def sub_while_cond_fun(sub_carry):
                sub_state, sub_time = sub_carry
                return sub_time < x if is_forward else sub_time > x

            def sub_while_body_fun(sub_carry):
                sub_state, sub_time = sub_carry
                if is_forward:
                    return (self.forward(sub_state), sub_time + 1)
                else:
                    return (self.backward(sub_state), sub_time - 1)

            final_state, final_time = jax.lax.while_loop(
                sub_while_cond_fun, sub_while_body_fun, carry
            )

            return (final_state, final_time), final_state

        initial_carry = (state, initial_time)
        (final_state, final_time), states = jax.lax.scan(body_fn, initial_carry, xs)

        return xs, states


class AbstractStochasticDynamicalSystem(eqx.Module, strict=True):
    """Abstract base class for stochastic dynamical systems."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def initial_state(
        self,
        key: Key[Array, "..."],
        **kwargs,
    ) -> Shaped[Array, "state_dim"]:
        raise NotImplementedError

    @abc.abstractmethod
    def trajectory(
        self,
        key: Key[Array, "..."],
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "... state_dim"]]:
        raise NotImplementedError

    @eqx.filter_jit
    def flow(
        self,
        key: Key[Array, "..."],
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
    ) -> Shaped[Array, "state_dim"]:
        _, states = self.trajectory(
            key=key,
            initial_time=initial_time,
            final_time=final_time,
            state=state,
            saveat=SaveAt(t1=True),
        )
        return states[-1]

    def orbit(
        self,
        key: Key[Array, "..."],
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ) -> Shaped[Array, "state_dim"]:
        """
        Trajectory but just return ys.
        """
        _, states = self.trajectory(key, initial_time, final_time, state, saveat)
        return states

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def generate(
        self,
        key: Key[Array, "..."],
        batch_size: int = 1000,
        final_time: float | int = 100.0,
    ) -> Shaped[Array, "{batch_size} state_dim"]:
        init_key, flow_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, batch_size)
        flow_keys = jax.random.split(flow_key, batch_size)

        initial_states = eqx.filter_vmap(self.initial_state)(init_keys)
        final_states = eqx.filter_vmap(self.flow)(
            flow_keys, 0.0, final_time, initial_states
        )
        return final_states


class AbstractStochasticContinuousDynamicalSystem(
    AbstractStochasticDynamicalSystem, strict=True
):

    @abc.abstractmethod
    def vector_field(
        self, t: float, y: Shaped[Array, "state_dim"], args
    ) -> Shaped[Array, "state_dim"]:
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion(
        self, t: float, y: Shaped[Array, "state_dim"], args
    ) -> Shaped[Array, "state_dim state_dim"]:
        """Diffusion matrix for SDE: dx = f(x)dt + g(x)dW"""
        raise NotImplementedError

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def trajectory(
        self,
        key: Key[Array, "..."],
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ) -> tuple[Shaped[Array, "..."], Shaped[Array, "... state_dim"]]:
        # Implementation depends on SDE solver choice
        # Could use diffrax with SDETerm
        raise NotImplementedError


class AbstractStochasticDiscreteDynamicalSystem(
    AbstractStochasticDynamicalSystem, strict=True
):

    @abc.abstractmethod
    def forward(
        self, key: Key[Array, "..."], state: Shaped[Array, "state_dim"]
    ) -> Shaped[Array, "state_dim"]:
        raise NotImplementedError

    @eqx.filter_jit
    def trajectory(
        self,
        key: Key[Array, "..."],
        initial_time: float,
        final_time: float,
        state: Shaped[Array, "state_dim"],
        saveat: SaveAt,
    ):
        assert initial_time <= final_time

        safe_initial_time = (
            jnp.atleast_1d(initial_time) if saveat.subs.t0 else jnp.array([])
        )
        safe_final_time = (
            jnp.atleast_1d(final_time) if saveat.subs.t1 else jnp.array([])
        )
        safe_array = jnp.array([]) if saveat.subs.ts is None else saveat.subs.ts
        xs = jnp.concatenate([safe_initial_time, safe_array, safe_final_time])

        n_steps = int(final_time - initial_time)
        step_keys = jax.random.split(key, n_steps)

        def body_fn(carry, x):
            current_state, current_time, key_idx = carry

            def sub_while_cond_fun(sub_carry):
                sub_state, sub_time, sub_key_idx = sub_carry
                return sub_time < x

            def sub_while_body_fun(sub_carry):
                sub_state, sub_time, sub_key_idx = sub_carry
                next_state = self.forward(sub_state, step_keys[sub_key_idx])
                return (next_state, sub_time + 1, sub_key_idx + 1)

            final_state, final_time, final_key_idx = jax.lax.while_loop(
                sub_while_cond_fun, sub_while_body_fun, carry
            )

            return (final_state, final_time, final_key_idx), final_state

        initial_carry = (state, initial_time, 0)
        (final_state, final_time, _), states = jax.lax.scan(body_fn, initial_carry, xs)

        return xs, states

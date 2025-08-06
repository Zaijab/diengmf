"""
This module describes dynamical systems for the express purpose of evaluating stochastic filtering algorithms.
The ABC structure allows the user to define their choice of dynamical system to reduce code duplication.
"""

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
        initial_time: float | Float[Array, ""] | int,
        final_time: float | Float[Array, ""] | int,
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
        initial_time: float | Float[Array, ""] | int,
        final_time: float | Float[Array, ""] | int,
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
        initial_time: float | Float[Array, ""] | int,
        final_time: float | Float[Array, ""] | int,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
    ) -> Float[Array, "state_dim"]:
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
    ) -> Float[Array, "{batch_size} state_dim"]:
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
        initial_time: float | Float[Array, ""] | int,
        final_time: float | Float[Array, ""] | int,
        state: Float[Array, "{self.dimension}"],
        saveat: SaveAt,
    ) -> tuple[Float[Array, "..."], Float[Array, "... {self.dimension}"]]:
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

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped


@jaxtyped(typechecker=typechecker)
class RFS(eqx.Module, strict=True):
    """
    A class implementing some of the behavior of a random finite set.
    A random finite set is a random variable whose realization is a finite set of elements.
    Both the cardinality of the set and the individual states are random.

    A basic but useful example is a Poisson point process where:

    Cardinality is Poisson distributed, states are uniform in a region.
    These conditions make it somewhat challenging to construct these in JAX,
    wherein XLA-compatible code requires array shapes to be determined at compile time.
    A naive data structure would have an array (or set) which changes size depending on the specified cardinality distribution.
    But this makes the size of the array depend on the output of `jax.random.poisson(key)` which cannot be determined while tracing.

    The solution this class uses is a maximum array size and arrays representing state and active components.
    If the Poisson RV spawns N elements, we make `max_compnents` larger than most reasonable values and insert the N states into state and flip N indices to true in mask.
    This approach isn't perfect, you need to be confident in the max_component size to work.
    It is not completely true to the mathematical theory, Poisson point processes don't have an arbitrary cap on their size.
    Most irritatingly, `jnp.where` currently executes both branches of conditionals.
    This means that functions of this RFS class have their speed determined by the max_components.
    Said another way, if `mask` has 10% of the indices as `True`, you still perform 100% of the computation, not 10%.

    But whatever, we cut our losses, and on a computer we rarely have access to infinite datasets and perhaps its a blessing in disguise to cap the amount of memory we use.
    This makes programs more consistent (no OOM error).
    """

    state: Float[Array, "max_components state_dim"]
    mask: Bool[Array, "max_components"]

    @jaxtyped(typechecker=typechecker)
    def insert(self, new_states: Float[Array, "n_birth state_dim"]):

        assert new_states.shape[0] <= self.mask.shape[0] - jnp.sum(
            self.mask
        ), "Array is full"

        # Find first n_birth empty slots
        empty_slots = jnp.where(~self.mask, size=new_states.shape[0], fill_value=-1)[0]
        valid_births = empty_slots >= 0

        # Update states and mask
        new_state = self.state.at[empty_slots].set(
            jnp.where(valid_births[:, None], new_states, 0)
        )
        new_mask = self.mask.at[empty_slots].set(valid_births)

        return RFS(new_state, new_mask)

    def remove(self, state: Float[Array, "n_death state_dim"]):
        pass

    @jaxtyped(typechecker=typechecker)
    def death(self, key: Key[Array, ""], p_S: float = 0.9) -> "RFS":
        survival = jax.random.bernoulli(key, p_S, shape=self.mask.shape)
        new_mask = self.mask & survival
        return RFS(self.state, new_mask)

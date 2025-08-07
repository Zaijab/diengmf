import jax
import jax.numpy as jnp
import equinox as eqx


class DenseNetwork(eqx.Module):
    input_dim: int
    hidden_dim: int
    num_hidden_layers: int
    output_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

        self.layers = [
            nnx.Linear(
                self.input_dim,
                self.hidden_dim,
                kernel_init=nnx.initializers.glorot_uniform(),
                rngs=rngs,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
            )
        ]
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                nnx.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_init=nnx.initializers.glorot_uniform(),
                    rngs=rngs,
                    dtype=jnp.float64,
                    param_dtype=jnp.float64,
                )
            )
            self.output_layer = nnx.Linear(
                self.hidden_dim,
                self.output_dim,
                kernel_init=nnx.initializers.glorot_uniform(),
                rngs=rngs,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)
        x = self.output_layer(x)
        return x

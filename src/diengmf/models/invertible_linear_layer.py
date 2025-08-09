import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Key, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Optional, List


class PLULinear(eqx.Module):
    """
    Invertible linear layer using PLU decomposition.

    Represents a linear transformation y = Ax + b where A = PLU.
    """

    n: int
    P: jax.Array  # Permutation indices
    L_params: jax.Array  # Strictly lower triangular elements
    U_diag: jax.Array  # Diagonal elements of U
    U_upper: jax.Array  # Strictly upper triangular elements
    bias: Optional[jax.Array]
    use_bias: bool

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, n: int, use_bias: bool = True, *, key: Key[Array, "..."]
    ) -> None:
        """
        Initialize PLU layer with random values.

        Args:
            n: Dimension of input/output
            use_bias: Whether to use bias term
            key: JAX PRNG key
        """
        p_key, l_key, u_diag_key, u_upper_key, bias_key = jax.random.split(key, 5)

        # Initialize permutation as identity (no permutation initially)
        self.P = jnp.arange(n)

        # Initialize L as identity + random strictly lower triangular part
        # We need n * (n-1) // 2 parameters for the strictly lower triangular part
        l_size = (n * (n - 1)) // 2
        self.L_params = jax.random.normal(l_key, (l_size,)) * 0.01

        # Initialize U diagonal with non-zero values for invertibility
        # We use softplus to ensure positivity and add small constant for stability
        u_diag_init = jnp.ones(n) + jax.random.normal(u_diag_key, (n,)) * 0.01
        self.U_diag = jnp.log(
            jnp.exp(u_diag_init) - 1.0
        )  # Inverse softplus for parameterization

        # Initialize strictly upper triangular part of U
        u_upper_size = (n * (n - 1)) // 2
        self.U_upper = jax.random.normal(u_upper_key, (u_upper_size,)) * 0.01

        # Initialize bias
        self.bias = jax.random.normal(bias_key, (n,)) * 0.01 if use_bias else None
        self.use_bias = use_bias
        self.n = n

    def _construct_L(self) -> jax.Array:
        """Construct lower triangular matrix L with ones on diagonal."""
        L = jnp.eye(self.n)
        indices = jnp.tril_indices(self.n, -1)  # Strictly lower triangular indices
        L = L.at[indices].set(self.L_params)
        return L

    def _construct_U(self) -> jax.Array:
        """Construct upper triangular matrix U."""
        # Apply softplus to ensure diagonal elements are positive
        u_diag = jax.nn.softplus(self.U_diag)  # + 1e-5

        # Create diagonal matrix
        U = jnp.diag(u_diag)

        # Fill in strictly upper triangular part
        indices = jnp.triu_indices(self.n, 1)  # Strictly upper triangular indices
        U = U.at[indices].set(self.U_upper)

        return U

    def _construct_P_matrix(self) -> jax.Array:
        """Construct permutation matrix from permutation indices."""
        P = jnp.zeros((self.n, self.n))
        P = P.at[jnp.arange(self.n), self.P].set(1.0)
        return P

    def _log_det(self) -> Float[Array, "..."]:
        """Compute log determinant of the transformation matrix."""
        # Log det of a permutation matrix is 0 if even permutation, log(-1) if odd
        # But we'll ignore this for now since we're not updating P during training
        # log_det_P = jnp.log(jnp.linalg.det(self._construct_P_matrix()))

        # Log det of L is 0 since diagonal elements are 1
        # log_det_L = 0.0

        # Log det of U is sum of log of diagonal elements
        u_diag = jax.nn.softplus(self.U_diag)  # + 1e-5
        log_det_U = jnp.sum(jnp.log(u_diag))

        # Total log det: P contributes sign only
        # parity = jnp.array(0.0, jnp.float32)  # Even permutation initially

        return log_det_U

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Array, "... n"]
    ) -> Tuple[Float[Array, "... n"], Float[Array, "..."]]:
        """
        Forward transformation: x → PLUx + b

        Args:
            x: Input tensor with last dimension n

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        # Construct matrices
        L = self._construct_L()
        U = self._construct_U()
        P_matrix = self._construct_P_matrix()

        # Apply PLU transformation
        y = x @ (P_matrix @ L @ U).T

        # Add bias if specified
        if self.use_bias and self.bias is not None:
            y = y + self.bias

        # Compute log determinant of Jacobian
        log_det = self._log_det()

        return y, log_det

    def inverse(
        self, y: Float[Array, "... n"]
    ) -> Tuple[Float[Array, "... n"], Float[Array, "..."]]:
        """
        Inverse transformation: y → U⁻¹L⁻¹P⁻¹(y - b)
        """
        # Remove bias if specified
        if self.use_bias and self.bias is not None:
            y = y - self.bias

        # Construct matrices
        L = self._construct_L()
        U = self._construct_U()
        P_matrix = self._construct_P_matrix()

        # Step 1: Apply P inverse (P^T)^(-1) = P
        y_p = y @ P_matrix

        # Step 2: Solve y_p = w @ L^T for w
        # Since L is lower triangular, L^T is upper triangular
        # We solve L @ w^T = y_p^T with lower=True
        w_t = jax.scipy.linalg.solve_triangular(L, y_p.T, lower=True)

        # Step 3: Solve w = x @ U^T for x
        # Since U is upper triangular, U^T is lower triangular
        # We solve U @ x^T = w^T with lower=False
        x_t = jax.scipy.linalg.solve_triangular(U, w_t, lower=False)

        # Transpose to get x
        x = x_t.T

        # Compute negative log determinant
        log_det = -self._log_det()

        return x, log_det

    def __call__(self, x, inverse=False):
        """
        Apply forward or inverse transformation.

        Args:
            x: Input tensor
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        return self.inverse(x) if inverse else self.forward(x)


class CouplingLayer(eqx.Module):
    s_net: eqx.nn.MLP
    t_net: eqx.nn.MLP
    input_dim: int
    swap: bool

    def __init__(self, input_dim, hidden_dim, num_hidden_layers, swap, *, key):
        s_key, t_key = jax.random.split(key)

        self.input_dim = input_dim
        self.swap = swap

        # Calculate dimensions correctly for odd inputs
        split_dim1 = input_dim // 2
        split_dim2 = input_dim - split_dim1

        # Determine conditioning and output dimensions based on swap
        if swap:
            condition_dim = split_dim2  # Larger part when odd
            output_dim = split_dim1  # Smaller part when odd
        else:
            condition_dim = split_dim1  # Smaller part when odd
            output_dim = split_dim2  # Larger part when odd

        self.s_net = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.gelu,
            key=s_key,
            dtype=jnp.float64,
        )

        self.t_net = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.gelu,
            key=t_key,
            dtype=jnp.float64,
        )

    def _safe_split(self, x):
        """Safely split input handling odd dimensions."""
        input_dim = x.shape[-1]
        split_point = input_dim // 2

        if self.swap:
            # For swap layers, take the larger part first when odd
            split_point = input_dim - split_point
            x1 = x[..., :split_point]
            x2 = x[..., split_point:]
            return x2, x1  # Return swapped
        else:
            x1 = x[..., :split_point]
            x2 = x[..., split_point:]
            return x1, x2

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Array, "... d_in"]
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Forward transformation through coupling layer.

        Args:
            x: Input tensor with shape (..., input_dim)

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        if self.swap:
            x1, x2 = self._safe_split(x)
            x1, x2 = x2, x1
        else:
            x1, x2 = self._safe_split(x)

        s = self.s_net(x1)
        s = 15 * jnp.tanh(s)
        t = self.t_net(x1)

        y2 = x2 * jnp.exp(s) + t
        log_det_jacobian = jnp.sum(s, axis=-1)

        y = jnp.concatenate([x1, y2], axis=-1)

        return y, log_det_jacobian

    @jaxtyped(typechecker=typechecker)
    def inverse(
        self, y: Float[Array, "... d_in"]
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Inverse transformation through coupling layer.

        Args:
            y: Input tensor with shape (..., input_dim)

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        if self.swap:
            y1, y2 = self._safe_split(y)
            y1, y2 = y2, y1
        else:
            y1, y2 = self._safe_split(y)

        s = self.s_net(y1)
        s = 15 * jnp.tanh(s)
        t = self.t_net(y1)

        x2 = (y2 - t) * jnp.exp(-s)
        log_det_jacobian = -jnp.sum(s, axis=-1)

        x = jnp.concatenate([y1, x2], axis=-1)

        return x, log_det_jacobian

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, "... d_in"], inverse: bool = False
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Apply forward or inverse transformation.

        Args:
            x: Input tensor
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        return self.inverse(x) if inverse else self.forward(x)


class InvertibleNN(eqx.Module):
    coupling_layers: List[CouplingLayer]
    plu_layers: List[PLULinear]
    input_dim: int

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        input_dim: int = 2,
        num_coupling_layers: int = 6,
        hidden_dim: int = 64,
        num_hidden_layers: int = 4,
        *,
        key: Key[Array, "..."],
    ) -> None:
        self.input_dim = input_dim

        # Split keys for coupling layers and PLU layers
        coupling_keys = jax.random.split(key, num_coupling_layers)
        plu_keys = jax.random.split(key, num_coupling_layers - 1)

        # Initialize coupling layers
        self.coupling_layers = []
        for i in range(num_coupling_layers):
            swap = i % 2 == 1
            layer = CouplingLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                swap=swap,
                key=coupling_keys[i],
            )
            self.coupling_layers.append(layer)

        # Initialize PLU layers between coupling layers
        self.plu_layers = []
        for i in range(num_coupling_layers - 1):
            plu_layer = PLULinear(n=input_dim, key=plu_keys[i])
            self.plu_layers.append(plu_layer)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Array, "... d_in"]
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Forward transformation: map from data space to latent space.

        Args:
            x: Input tensor in data space

        Returns:
            Tuple of (transformed tensor in latent space, log determinant of Jacobian)
        """
        log_det_jacobian = 0

        # Apply first coupling layer
        x, ldj = self.coupling_layers[0].forward(x)
        log_det_jacobian += ldj

        # Apply alternating PLU and coupling layers
        for i in range(len(self.plu_layers)):
            # Apply PLU layer
            x, ldj = self.plu_layers[i].forward(x)
            log_det_jacobian += ldj

            # Apply next coupling layer
            x, ldj = self.coupling_layers[i + 1].forward(x)
            log_det_jacobian += ldj

        return x, log_det_jacobian

    @jaxtyped(typechecker=typechecker)
    def inverse(
        self, z: Float[Array, "... d_in"]
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Inverse transformation: map from latent space to data space.

        Args:
            z: Input tensor in latent space

        Returns:
            Tuple of (transformed tensor in data space, log determinant of Jacobian)
        """
        log_det_jacobian = 0

        # Apply layers in reverse order
        # Start with the last coupling layer
        z, ldj = self.coupling_layers[-1].inverse(z)
        log_det_jacobian += ldj

        # Apply alternating PLU and coupling layers in reverse
        for i in range(len(self.plu_layers) - 1, -1, -1):
            # Apply PLU layer in reverse
            z, ldj = self.plu_layers[i].inverse(z)
            log_det_jacobian += ldj

            # Apply coupling layer in reverse
            z, ldj = self.coupling_layers[i].inverse(z)
            log_det_jacobian += ldj

        return z, log_det_jacobian

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, "... d_in"], inverse: bool = False
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Apply forward or inverse transformation.

        Args:
            x: Input tensor
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        return self.inverse(x) if inverse else self.forward(x)

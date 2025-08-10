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
    ) -> tuple[Float[Array, "... n"], Float[Array, "..."]]:
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
    ) -> tuple[Float[Array, "... n"], Float[Array, "..."]]:
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

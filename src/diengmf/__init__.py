import jax
from pathlib import Path

jax.config.update("jax_enable_x64", True)
ROOT_DIR = Path(__file__).parent

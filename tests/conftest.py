import pytest
from pathlib import Path

@pytest.fixture
def ROOT_DIR():
    return Path(__file__).parent.parent

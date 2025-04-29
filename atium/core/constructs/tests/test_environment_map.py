import numpy as np
import pytest

from atium.core.constructs.environment_map import EnvironmentMap2D


@pytest.fixture(scope="function")
def emap2d() -> EnvironmentMap2D:
    return EnvironmentMap2D(
        array=np.zeros((100, 100), dtype=np.uint8),
        resolution=0.1,
    )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])

import numpy as np
import pytest

from atium.core.constructs.environment_map import EnvironmentMap2D


@pytest.fixture(scope="function")
def emap2d() -> EnvironmentMap2D:
    return EnvironmentMap2D(
        array=np.zeros((100, 100), dtype=np.uint8),
        resolution=0.1,
    )


def test_emap2d_invalid() -> None:
    # Non positive resolution
    with pytest.raises(ValueError):
        EnvironmentMap2D(
            array=np.zeros((100, 100), dtype=np.uint8),
            resolution=-0.1,
        )
    with pytest.raises(ValueError):
        EnvironmentMap2D(
            array=np.zeros((100, 100), dtype=np.uint8),
            resolution=0.0,
        )

    # Invalid array
    with pytest.raises(ValueError):
        EnvironmentMap2D(
            array=np.zeros((100, 100, 3), dtype=np.uint8),
            resolution=0.1,
        )

    with pytest.raises(ValueError):
        EnvironmentMap2D(
            array=np.zeros((100, 100), dtype=np.int32),
            resolution=0.1,
        )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])

import cv2
import numpy as np
import pytest

from atium.core.constructs.environment_map import EnvironmentLabels, EnvironmentMap2D


@pytest.fixture(scope="function")
def emap2d() -> EnvironmentMap2D:
    return EnvironmentMap2D(
        # Asymmetric array to catch bugs.
        array=np.zeros((100, 200), dtype=np.uint8),
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


def test_emap2d_from_empty() -> None:
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(20.0, 10.0),
        resolution=0.1,
    )
    np.testing.assert_allclose(
        emap2d.array,
        np.zeros((100, 200), dtype=np.uint8),
    )


def test_emap2d_size_px(emap2d: EnvironmentMap2D) -> None:
    assert emap2d.size_px == (100, 200)


def test_emap2d_size_xy(emap2d: EnvironmentMap2D) -> None:
    assert emap2d.size_xy == (20.0, 10.0)


def test_emap2d_xy_to_px(emap2d: EnvironmentMap2D) -> None:
    # Test the conversion from world coordinates to pixel coordinates.
    x, y = 1.0, 0.5
    px_x, px_y = emap2d.xy_to_px((x, y))
    assert px_x == 10
    assert px_y == 5


def test_emap2d_px_to_xy(emap2d: EnvironmentMap2D) -> None:
    # Test the conversion from pixel coordinates to world coordinates.
    px_x, px_y = 10, 5
    x, y = emap2d.px_to_xy((px_x, px_y))
    np.testing.assert_allclose(x, 1.0, atol=1e-6)
    np.testing.assert_allclose(y, 0.5, atol=1e-6)


def test_emap2d_get_cv2_coordinates(emap2d: EnvironmentMap2D) -> None:
    # Test the conversion from world coordinates to OpenCV coordinates.
    px_x, px_y = 24, 7
    cv2_x, cv2_y = emap2d.get_cv2_coordinates((px_x, px_y))
    assert cv2_x == 24
    assert cv2_y == 93


@pytest.mark.parametrize("thickness", [1, cv2.FILLED])
def test_emap2d_add_rectangular_obstacle(
    emap2d: EnvironmentMap2D,
    thickness: int,
    debug: bool = True,
) -> None:
    # Test adding a rectangular obstacle to the environment map.
    emap2d.add_rectangular_obstacle(
        center_xy=(10.0, 5.0),
        size_xy=(1.0, 3.0),
        label=EnvironmentLabels.STATIC_OBSTACLE,
        thickness=thickness,
    )
    if debug:
        cv2.imshow(
            "EnvironmentMap2D",
            emap2d.array,
        )
        cv2.waitKey(0)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])

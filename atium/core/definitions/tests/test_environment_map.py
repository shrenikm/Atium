import cv2
import numpy as np
import pytest

from atium.core.definitions.environment_map import EnvironmentLabels, EnvironmentMap2D
from atium.core.utils.color_utils import ColorType


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
    # Int output (also the default)
    x, y = 1.0, 0.5
    px_x, px_y = emap2d.xy_to_px((x, y))
    assert px_x == 10
    assert px_y == 5

    # Float output
    x, y = 1.11, 0.55
    px_x, px_y = emap2d.xy_to_px((x, y), output_as_float=True)
    np.testing.assert_allclose(px_x, 11.1, atol=1e-6)
    np.testing.assert_allclose(px_y, 5.5, atol=1e-6)


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
    debug: bool = False,
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


def test_emap2d_compute_signed_distance_transform() -> None:
    array = np.zeros((10, 10), dtype=np.uint8)
    array[3:8, 3:8] = EnvironmentLabels.STATIC_OBSTACLE
    emap2d = EnvironmentMap2D(
        array=array,
        resolution=0.1,
    )
    signed_dtf = emap2d.compute_signed_distance_transform()
    expected_signed_dtf = np.zeros_like(signed_dtf)
    for i in range(3):
        # Closest obstacle is the top left corner at (3, 3).
        for j in range(3):
            expected_signed_dtf[i, j] = np.hypot(3 - i, 3 - j)

        # Closest obstacle is vertically below at (3, j).
        for j in range(3, 8):
            expected_signed_dtf[i, j] = 3 - i

        # Closest obstacle is the top right corner (3, 7)
        for j in range(8, 10):
            expected_signed_dtf[i, j] = np.hypot(3 - i, j - 7)

    for i in range(3, 8):
        # Closest obstacle is horizontally to the right at (i, 3).
        for j in range(3):
            expected_signed_dtf[i, j] = 3 - j

        # Closest obstacle is in the middle at (5, 5).
        for j in range(3, 8):
            expected_signed_dtf[i, j] = max(abs(5 - i), abs(5 - j)) - 3

        # Closest obstacle is horizontally to the left at (i, 7).
        for j in range(8, 10):
            expected_signed_dtf[i, j] = j - 7

    for i in range(8, 10):
        # Closest obstacle is the bottom left corner at (7, 3).
        for j in range(3):
            expected_signed_dtf[i, j] = np.hypot(7 - i, 3 - j)

        # Closest obstacle is vertically above at (7, j).
        for j in range(3, 8):
            expected_signed_dtf[i, j] = i - 7

        # Closest obstacle is the bottom right corner (7, 7)
        for j in range(8, 10):
            expected_signed_dtf[i, j] = np.hypot(7 - i, j - 7)

    expected_signed_dtf = emap2d.resolution * expected_signed_dtf
    np.testing.assert_allclose(
        signed_dtf,
        expected_signed_dtf,
        atol=1e-6,
    )


@pytest.mark.parametrize("color_type", [ColorType.RGB, ColorType.BGR])
def test_create_rgb_viz(
    color_type: ColorType,
    debug: bool = False,
) -> None:
    # Test the creation of an RGB visualization of the environment map.
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(20.0, 10.0),
        resolution=0.1,
    )
    emap2d.add_rectangular_obstacle(
        center_xy=(10.0, 5.0),
        size_xy=(1.0, 3.0),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )
    rgb_viz = emap2d.create_rgb_viz(color_type=color_type)
    if debug:
        cv2.imshow("RGB Visualization", rgb_viz)
        cv2.waitKey(0)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])

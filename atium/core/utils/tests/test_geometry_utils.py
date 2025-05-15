import numpy as np
import pytest

from atium.core.utils.geometry_utils import (
    construct_rectangle_polygon,
    densify_polygon,
    normalize_angle,
    normalize_angle_differentiable,
)


def test_normalize_angle() -> None:
    input_and_expected_angles = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (np.pi - 0.1, np.pi - 0.1),
        (-np.pi + 0.1, -np.pi + 0.1),
        (2 * np.pi, 0.0),
        (-2 * np.pi, 0.0),
        (5 * np.pi / 2.0, np.pi / 2.0),
        (-5 * np.pi / 2.0, -np.pi / 2.0),
    ]
    for input_angle, expected_angle in input_and_expected_angles:
        normalized_angle = normalize_angle(input_angle)
        np.testing.assert_allclose(
            normalized_angle,
            expected_angle,
            atol=1e-12,
        )


def test_normalize_angle_differentiable() -> None:
    input_and_expected_angles = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (np.pi - 0.1, np.pi - 0.1),
        (-np.pi + 0.1, -np.pi + 0.1),
        (2 * np.pi, 0.0),
        (-2 * np.pi, 0.0),
        (5 * np.pi / 2.0, np.pi / 2.0),
        (-5 * np.pi / 2.0, -np.pi / 2.0),
    ]
    for input_angle, expected_angle in input_and_expected_angles:
        normalized_angle = normalize_angle_differentiable(input_angle)
        np.testing.assert_allclose(
            normalized_angle,
            expected_angle,
            atol=1e-12,
        )


def test_construct_rectangle_polygon() -> None:
    polygon = construct_rectangle_polygon(
        center_xy=(0.0, 0.0),
        size_xy=(1.0, 2.0),
    )
    expected_polygon = np.array(
        [
            [-0.5, -1.0],
            [0.5, -1.0],
            [0.5, 1.0],
            [-0.5, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        polygon,
        expected_polygon,
        atol=1e-12,
    )

    polygon = construct_rectangle_polygon(
        center_xy=(1.0, 2.0),
        size_xy=(1.0, 2.0),
    )
    expected_polygon = np.array(
        [
            [0.5, 1.0],
            [1.5, 1.0],
            [1.5, 3.0],
            [0.5, 3.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        polygon,
        expected_polygon,
        atol=1e-12,
    )


def test_densify_polygon() -> None:
    polygon = construct_rectangle_polygon(
        center_xy=(0.0, 0.0),
        size_xy=(8.0, 6.0),
    )
    dense_polygon = densify_polygon(
        polygon=polygon,
        spacing=1.0,
    )
    expected_polygon = np.array(
        [
            [-4.0, -3.0],
            [-3.0, -3.0],
            [-2.0, -3.0],
            [-1.0, -3.0],
            [0.0, -3.0],
            [1.0, -3.0],
            [2.0, -3.0],
            [3.0, -3.0],
            [4.0, -3.0],
            [4.0, -2.0],
            [4.0, -1.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [4.0, 2.0],
            [4.0, 3.0],
            [3.0, 3.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [0.0, 3.0],
            [-1.0, 3.0],
            [-2.0, 3.0],
            [-3.0, 3.0],
            [-4.0, 3.0],
            [-4.0, 2.0],
            [-4.0, 1.0],
            [-4.0, 0.0],
            [-4.0, -1.0],
            [-4.0, -2.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        dense_polygon,
        expected_polygon,
        atol=1e-12,
    )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])

import numpy as np
import pytest
from pydrake.math import RotationMatrix

from atium.core.utils.transformation_utils import rotation_matrix_2d, transform_points_2d, transformation_matrix_2d


@pytest.fixture(scope="module")
def rng() -> np.random.RandomState:
    return np.random.RandomState(7)


def test_rotation_matrix_2d(rng: np.random.RandomState) -> None:
    rotmat2d = rotation_matrix_2d(angle=0.0)
    np.testing.assert_allclose(rotmat2d, np.eye(2), atol=1e-12)

    angle = rng.randn()
    rotmat2d = rotation_matrix_2d(angle=angle)
    expected_rotmat2d = RotationMatrix.MakeZRotation(angle).matrix()[:2, :2]
    np.testing.assert_allclose(
        rotmat2d,
        RotationMatrix.MakeZRotation(angle).matrix()[:2, :2],
        atol=1e-12,
    )


def test_transformation_matrix_2d(rng: np.random.RandomState) -> None:
    rotation = rng.randn()
    translation = rng.randn(2)

    tfmat2d = transformation_matrix_2d()
    np.testing.assert_allclose(tfmat2d, np.eye(3), atol=1e-12)

    tfmat2d = transformation_matrix_2d(rotation=rotation)
    np.testing.assert_allclose(
        tfmat2d[:2, :2],
        RotationMatrix.MakeZRotation(rotation).matrix()[:2, :2],
        atol=1e-12,
    )
    np.testing.assert_allclose(tfmat2d[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-12)

    tfmat2d = transformation_matrix_2d(translation=translation)
    np.testing.assert_allclose(
        tfmat2d[:2, :2],
        np.eye(2),
        atol=1e-12,
    )
    np.testing.assert_allclose(tfmat2d[:, 2], np.hstack((translation, 1)), atol=1e-12)

    tfmat2d = transformation_matrix_2d(translation=translation, rotation=rotation)
    np.testing.assert_allclose(
        tfmat2d[:2, :2],
        RotationMatrix.MakeZRotation(rotation).matrix()[:2, :2],
        atol=1e-12,
    )
    np.testing.assert_allclose(tfmat2d[:, 2], np.hstack((translation, 1)), atol=1e-12)


def test_transform_points_2d(rng: np.random.RandomState) -> None:
    points = rng.randn(10, 2)
    translation = rng.randn(2)
    rotation = rng.randn()

    transformed_points = transform_points_2d(
        points=points,
        translation=translation,
        rotation=rotation,
    )
    rotmat2d = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    expected_transformed_points = (rotmat2d @ points.T).T + translation
    np.testing.assert_allclose(
        transformed_points,
        expected_transformed_points,
        atol=1e-12,
    )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])

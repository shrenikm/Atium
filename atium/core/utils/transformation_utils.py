import numpy as np
from pydrake.math import RotationMatrix

from atium.core.utils.custom_types import PointXYArray, PointXYVector, RotationMatrix2D, TransformationMatrix2D


def rotation_matrix_2d(angle: float) -> RotationMatrix2D:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )


def transformation_matrix_2d(
    translation: PointXYVector | None = None,
    rotation: float | None = None,
) -> TransformationMatrix2D:
    transformation_matrix = np.eye(3, dtype=np.float64)
    if rotation is not None:
        transformation_matrix[0:2, 0:2] = rotation_matrix_2d(rotation)
    if translation is not None:
        transformation_matrix[0:2, 2] = translation
    return transformation_matrix


def transform_points_2d(
    points: PointXYArray,
    translation: PointXYVector | None = None,
    rotation: float | None = None,
) -> PointXYArray:
    transformation_matrix = transformation_matrix_2d(translation, rotation)
    return (np.hstack((points, np.ones((points.shape[0], 1)))) @ transformation_matrix.T)[:, :2]

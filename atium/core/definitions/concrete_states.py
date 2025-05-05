from typing import Self

import attr
import numpy as np

from atium.core.utils.custom_types import Pose2DVector, StateVector, Velocity2DVector


@attr.frozen
class Pose2D:
    """
    A 2D pose represented by an x/y position and orientation.
    """

    x: float
    y: float
    theta: float

    @classmethod
    def from_vector(cls, pose_vector: Pose2DVector) -> Self:
        """
        Create a Pose2D from a state vector.
        """
        if not isinstance(pose_vector, np.ndarray) or pose_vector.ndim != 1 or pose_vector.size != 3:
            raise ValueError("State vector must be of length 3.")
        return cls(
            x=pose_vector[0],
            y=pose_vector[1],
            theta=pose_vector[2],
        )

    def to_vector(self) -> Pose2DVector:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)


@attr.frozen
class Velocity2D:
    """
    A 2D velocity represented as linear/angular.
    """

    linear: float
    angular: float

    @classmethod
    def from_vector(cls, velocity_vector: Velocity2DVector) -> Self:
        """
        Create a Velocity2D from a state vector.
        """
        if not isinstance(velocity_vector, np.ndarray) or velocity_vector.ndim != 1 or velocity_vector.size != 2:
            raise ValueError("Velocity vector must be of length 3.")
        return cls(
            linear=velocity_vector[0],
            angular=velocity_vector[1],
        )

    def to_vector(self) -> Velocity2DVector:
        return np.array([self.linear, self.angular], dtype=np.float64)

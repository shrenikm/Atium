import numpy as np
import pytest

from atium.core.definitions.concrete_states import Pose2D, Velocity2D


@pytest.fixture(scope="module")
def rng() -> np.random.RandomState:
    return np.random.RandomState(7)


def test_pose2d_invalid() -> None:
    # Not a vector.
    with pytest.raises(ValueError):
        Pose2D.from_vector([1.0, 2.0, 3.0])

    # Not a 1D vector.
    with pytest.raises(ValueError):
        Pose2D.from_vector(np.array([[1.0, 2.0, 3.0]]))

    # Not of size 3.
    with pytest.raises(ValueError):
        Pose2D.from_vector(np.array([1.0, 2.0]))


def test_pose2d_from_vector(rng: np.random.RandomState) -> None:
    # Valid vector.
    pose_vector = rng.randn(3)
    pose = Pose2D.from_vector(pose_vector=pose_vector)
    np.testing.assert_equal(pose.x, pose_vector[0])
    np.testing.assert_equal(pose.y, pose_vector[1])
    np.testing.assert_equal(pose.theta, pose_vector[2])


def test_pose2d_to_vector(rng: np.random.RandomState) -> None:
    # Valid vector.
    pose_vector = rng.randn(3)
    pose = Pose2D.from_vector(pose_vector)
    vector = pose.to_vector()
    np.testing.assert_equal(vector, pose_vector)


def test_velocity2d_invalid() -> None:
    # Not a vector.
    with pytest.raises(ValueError):
        Velocity2D.from_vector([1.0, 2.0])

    # Not a 1D vector.
    with pytest.raises(ValueError):
        Velocity2D.from_vector(np.array([[1.0, 2.0]]))

    # Not of size 2.
    with pytest.raises(ValueError):
        Velocity2D.from_vector(np.array([1.0, 2.0, 3.0]))


def test_velocity2d_from_vector(rng: np.random.RandomState) -> None:
    # Valid vector.
    velocity_vector = rng.randn(2)
    velocity = Velocity2D.from_vector(velocity_vector=velocity_vector)
    np.testing.assert_equal(velocity.linear, velocity_vector[0])
    np.testing.assert_equal(velocity.angular, velocity_vector[1])


def test_velocity2d_to_vector(rng: np.random.RandomState) -> None:
    # Valid vector.
    velocity_vector = rng.randn(2)
    velocity = Velocity2D.from_vector(velocity_vector)
    vector = velocity.to_vector()
    np.testing.assert_equal(vector, velocity_vector)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

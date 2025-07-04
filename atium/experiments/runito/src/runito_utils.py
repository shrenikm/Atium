import attr

from atium.core.definitions.concrete_states import Pose2D, Velocity2D
from atium.core.definitions.environment_map import EnvironmentMap2D
from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.custom_types import (
    NpMatrix22f64,
    PolygonXYArray,
)

DEFAULT_CONSTRAINT_TOLERANCE = 1e-2


@attr.frozen
class RunitoParams:
    """
    Parameters for Runito.
    Includes problem formulation and optimization params.
    """

    # Basis beta will be of degree 2*h-1
    h: int = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=1))
    # Number of segments
    M: int = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=1))
    # Number of sampling intervals (Note: this is not the number of points in each segment)
    # Each segment has n intervals = n + 1 points (including the endpoints)
    n: int = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=2))

    # Costs
    epsilon_t: float = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=0))
    W: NpMatrix22f64

    # Tolerances.

    initial_state_equality_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE
    final_state_equality_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE
    continuity_equality_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE
    kinematic_equality_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE

    final_xy_equality_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE


@attr.frozen
class RunitoInitialStateInputs:
    initial_pose: Pose2D
    initial_velocity: Velocity2D


@attr.frozen
class RunitoFinalStateInputs:
    final_pose: Pose2D
    final_velocity: Velocity2D | None = None


@attr.frozen
class RunitoInputs:
    """
    Runito solve inputs.
    """

    footprint: PolygonXYArray
    emap2d: EnvironmentMap2D
    obstacle_clearance: float
    upper_velocity_limits: Velocity2D
    lower_velocity_limits: Velocity2D
    initial_state_inputs: RunitoInitialStateInputs
    final_state_inputs: RunitoFinalStateInputs

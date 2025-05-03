import attr

from atium.core.constructs.environment_map import EnvironmentMap2D
from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.custom_types import (
    NpMatrix22f64,
    NpVector2f64,
    PointXYArray,
    PolygonXYArray,
    PositionXYVector,
    StateDerivativeVector,
    StateVector,
)


@attr.frozen
class UnitoParams:
    """
    Parameters for Unito.
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
    # continuity_equality_tolerance: float = 1e-6
    # initial_state_equality_tolerance: float = 1e-6
    # final_state_equality_tolerance: float = 1e-6
    continuity_equality_tolerance: float = 0.
    initial_state_equality_tolerance: float = 0.
    final_state_equality_tolerance: float = 0.
    final_xy_equality_tolerance: float = 1e-2


@attr.frozen
class UnitoInitialStateInputs:
    initial_ms_map: dict[int, StateVector | StateDerivativeVector]
    initial_xy: PositionXYVector


@attr.frozen
class UnitoFinalStateInputs:
    """
    Note that this gets used a bit differently than the initial ms state.
    For example, we cannot have a constraint on the final s value (0 derivative) as this isn't really known until the end of the optimization.
    Final s constraints on non zero derivatives are fine as these correspond to velocities, accelerations, etc.
    But we can have a constraint on the final theta value.
    So any constraint on the final (0 derivative) s value is ignored.
    """

    final_ms_map: dict[int, StateVector | StateDerivativeVector]
    final_xy: PositionXYVector


@attr.frozen
class UnitoInputs:
    """
    Unito solve inputs.
    """

    footprint: PolygonXYArray
    emap2d: EnvironmentMap2D
    obstacle_clearance: float
    initial_state_inputs: UnitoInitialStateInputs
    final_state_inputs: UnitoFinalStateInputs

import attr

from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.custom_types import NpMatrix22f64, NpVector2f64, StateDerivativeVector, StateVector


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
    # Number of sampling intervals
    n: int = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=2))

    # Costs
    epsilon_t: float = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=0))
    W: NpMatrix22f64


@attr.frozen
class UnitoStartInputs:
    ms_state_map: dict[int, StateVector | StateDerivativeVector]


@attr.frozen
class UnitoEndInputs:
    """
    End inputs are split into theta and s separately as there are some differences in how they're handled.
    For example, we cannot have a constraint on the final s value as this isn't really known until the end of the optimization.
    But we can have a constraint on the final theta value.
    """

    theta_map: dict[int, float]
    s_map: dict[int, float]
    xy: NpVector2f64


@attr.frozen
class UnitoInputs:
    """
    Unito solve inputs.
    """

    start_inputs: UnitoStartInputs
    end_inputs: UnitoEndInputs

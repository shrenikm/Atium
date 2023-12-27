from enum import Enum, auto
from typing import Callable, Protocol

from common.custom_types import ControlInputVector, StateDerivativeFn, StateVector


class StateIntegratorCallable(Protocol):
    def __call__(
        self,
        state: StateVector,
        control_input: ControlInputVector,
        state_derivative_fn: StateDerivativeFn,
        dt: float,
    ) -> StateVector:
        raise NotImplemented


class StateIntegratorType(Enum):
    EXPLICIT_EULER = auto()
    IMPLICIT_EULER = auto()
    MID_POINT = auto()


def explicit_euler_state_integrator_fn(
    state: StateVector,
    control_input: ControlInputVector,
    state_derivative_fn: StateDerivativeFn,
    dt: float,
) -> StateVector:

    return state + dt * state_derivative_fn(state, control_input)


def implicit_euler_state_integrator_fn(
    state: StateVector,
    control_input: ControlInputVector,
    state_derivative_fn: StateDerivativeFn,
    dt: float,
) -> StateVector:
    raise NotImplementedError


def mid_point_state_integrator_fn(
    state: StateVector,
    control_input: ControlInputVector,
    state_derivative_fn: StateDerivativeFn,
    dt: float,
) -> StateVector:
    """
    Assuming constant control.
    """
    mid_point_state = state + 0.5 * dt * state_derivative_fn(state, control_input)
    return state + dt * state_derivative_fn(mid_point_state, control_input)


def state_integrator_fn_from_type(
    state_integrator_type: StateIntegratorType,
) -> StateIntegratorCallable:

    mapping = {
        StateIntegratorType.EXPLICIT_EULER: explicit_euler_state_integrator_fn,
        StateIntegratorType.IMPLICIT_EULER: implicit_euler_state_integrator_fn,
        StateIntegratorType.MID_POINT: mid_point_state_integrator_fn,
    }
    return mapping[state_integrator_type]

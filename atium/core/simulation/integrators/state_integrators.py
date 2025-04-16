from enum import Enum, auto
from typing import Protocol

from atium.core.utils.custom_types import ControlInputVector, StateDerivativeFn, StateVector


class StateIntegratorCallable(Protocol):
    def __call__(
        self,
        state: StateVector,
        control_input: ControlInputVector,
        state_derivative_fn: StateDerivativeFn,
        dt: float,
    ) -> StateVector:
        raise NotImplementedError


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


STATE_INTEGRATORS_FN_MAP = {
    StateIntegratorType.EXPLICIT_EULER: explicit_euler_state_integrator_fn,
    StateIntegratorType.IMPLICIT_EULER: implicit_euler_state_integrator_fn,
    StateIntegratorType.MID_POINT: mid_point_state_integrator_fn,
}

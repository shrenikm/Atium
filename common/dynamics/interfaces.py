from typing import Protocol
import attr

from common.custom_types import ControlInputVector, StateDerivativeVector, StateVector


@attr.frozen
class Dynamics(Protocol):
    def compute_state_derivative(
        self,
        state: StateVector,
        control_input: ControlInputVector,
    ) -> StateDerivativeVector:
        raise NotImplementedError

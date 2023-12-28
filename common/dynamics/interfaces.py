from typing import Protocol

import attr

from common.custom_types import ControlInputVector, StateDerivativeVector, StateVector


@attr.frozen
class Dynamics(Protocol):
    def normalize_state(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Perform any normalizations on the state (like wrapping angles) and return the normalized value.
        """
        raise NotImplemented

    def compute_state_derivative(
        self,
        state: StateVector,
        control_input: ControlInputVector,
    ) -> StateDerivativeVector:
        raise NotImplementedError

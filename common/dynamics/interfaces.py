from abc import ABCMeta, abstractmethod

import attr

from common.custom_types import (
    ControlInputVector,
    StateDerivativeVector,
    StateVector,
)
from common.dynamics.utils import StateVectorLimits, ControlInputVectorLimits


@attr.frozen
class IDynamics(metaclass=ABCMeta):
    state_limits: StateVectorLimits
    control_input_limits: ControlInputVectorLimits

    @abstractmethod
    def compute_state_derivative(
        self,
        state: StateVector,
        control_input: ControlInputVector,
    ) -> StateDerivativeVector:
        raise NotImplementedError

from abc import ABCMeta, abstractmethod

import attr
import numpy as np

from common.custom_types import ControlInputVector, StateDerivativeVector, StateVector
from common.dynamics.utils import ControlInputVectorLimits, StateVectorLimits


@attr.frozen
class IDynamics(metaclass=ABCMeta):
    state_limits: StateVectorLimits
    control_input_limits: ControlInputVectorLimits

    def normalize_state(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Normalize the state (Angle wrapping, etc).
        Ugly mixing impl in the interface but it's fine.
        """
        return state

    def bound_state(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Bound the state vector to be within limits.
        Ugly mixing impl in the interface but it's fine.
        """
        return np.clip(
            state,
            self.state_limits.lower,
            self.state_limits.upper,
        )

    def bound_control_input(
        self,
        control_input: ControlInputVector,
    ) -> ControlInputVector:
        """
        Bound the control input vector to be within limits.
        Ugly mixing impl in the interface but it's fine.
        """
        return np.clip(
            control_input,
            self.control_input_limits.lower,
            self.control_input_limits.upper,
        )

    @abstractmethod
    def compute_state_derivative(
        self,
        state: StateVector,
        control_input: ControlInputVector,
    ) -> StateDerivativeVector:
        raise NotImplementedError

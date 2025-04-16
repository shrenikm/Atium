from abc import ABCMeta, abstractmethod
from typing import Self

import attr
import numpy as np

from atium.core.utils.custom_types import ControlInputVector, StateDerivativeVector, StateVector


@attr.frozen
class StateVectorLimits:
    lower: StateVector
    upper: StateVector

    @classmethod
    def from_free(cls, size: int) -> Self:
        """
        Free limits <=> No limits.
        -inf to inf.
        """
        return cls(
            lower=np.full(size, -np.inf),
            upper=np.full(size, np.inf),
        )


@attr.frozen
class ControlInputVectorLimits:
    lower: ControlInputVector
    upper: ControlInputVector

    @classmethod
    def from_free(cls, size: int) -> Self:
        """
        Free limits <=> No limits.
        -inf to inf.
        """
        return cls(
            lower=np.full(size, -np.inf),
            upper=np.full(size, np.inf),
        )


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

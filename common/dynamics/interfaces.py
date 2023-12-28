from abc import ABCMeta, abstractmethod

import attr

from common.custom_types import ControlInputVector, StateDerivativeVector, StateVector


@attr.frozen
class IDynamics(metaclass=ABCMeta):
    @abstractmethod
    def normalize_state(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Perform any normalizations on the state (like wrapping angles) and return the normalized value.
        """
        raise NotImplemented

    @abstractmethod
    def compute_state_derivative(
        self,
        state: StateVector,
        control_input: ControlInputVector,
    ) -> StateDerivativeVector:
        raise NotImplementedError

"""
Simple controller interface
"""
from typing import Protocol
import attr

from common.custom_types import ControlInputVector, StateVector


@attr.define
class Controller(Protocol):
    def compute_control_input(
        self,
        state: StateVector,
    ) -> ControlInputVector:
        raise NotImplementedError

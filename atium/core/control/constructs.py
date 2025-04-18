"""
Simple controller interface
"""

from typing import Protocol

import attr

from atium.core.utils.custom_types import ControlInputVector, StateVector


@attr.define
class IController(Protocol):
    def compute_control_input(
        self,
        state: StateVector,
    ) -> ControlInputVector:
        raise NotImplementedError

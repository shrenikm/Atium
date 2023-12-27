"""
Light-weight simulator for kinematic and simple dynamic systems.
Useful for quick prototyping with custom visualization without having to set up a PyBullet/Drake env.
"""
import time
from typing import Protocol

import attr
import numpy as np

from common.control.controller import Controller
from common.custom_types import ControlInputVector, StateVector


@attr.frozen
class SiliconSimulatorParams:
    max_time_s: float


@attr.define
class SiliconSimulator(Protocol):
    state: StateVector
    timestamp_s: float

    def udpate_state(
        self,
        control_input: ControlInputVector,
        dt: float,
    ) -> None:
        raise NotImplementedError

    def step_simulate(
        self,
        controller: Controller,
        dt: float,
        max_time_s: float = np.inf,
    ) -> None:

        sim_time_s = 0.

        while sim_time_s <= max_time_s:
            control_input = controller.compute_control_input(
                state=self.state,
            )
            self.state = self.update_state(
                control

    def realtime_simulate(
        self,
        realtime_rate: float,
    ) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    ...

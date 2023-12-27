"""
Light-weight simulator for kinematic and simple dynamic systems.
Useful for quick prototyping with custom visualization without having to set up a PyBullet/Drake env.
"""

import time
from typing import Protocol

import attr

from common.control.controller import Controller
from common.custom_types import ControlVector, StateVector


@attr.frozen
class SiliconSimulatorParams:
    max_time_s: float


@attr.define
class SiliconSimulator(Protocol):
    state: StateVector
    timestamp_s: float

    def udpate_state(
        self,
        control: ControlVector,
        dt: float,
    ) -> None:
        raise NotImplementedError

    def step_simulate(
        self,
        controller: Controller,
        dt: float,
    ) -> None:
        raise NotImplementedError

    def realtime_simulate(
        self,
        realtime_rate: float,
    ) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    ...

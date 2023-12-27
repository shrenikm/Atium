"""
Light-weight simulator for kinematic and simple dynamic systems.
Useful for quick prototyping with custom visualization without having to set up a PyBullet/Drake env.
"""
import time
from typing import Protocol

import attr
import numpy as np

from common.control.interfaces import Controller
from common.custom_types import ControlInputVector, StateVector
from common.dynamics.interfaces import Dynamics
from common.simulation.integrators.state_integrators import StateIntegratorCallable


@attr.frozen
class SiliconSimulatorParams:
    max_time_s: float


@attr.define
class SiliconSimulator(Protocol):
    state: StateVector
    control_input: ControlInputVector
    dynamics: Dynamics
    state_integrator_fn: StateIntegratorCallable
    timestamp_s: float

    def visualize(self) -> None:
        raise NotImplementedError

    def step_simulate(
        self,
        controller: Controller,
        dt: float,
        max_time_s: float = np.inf,
    ) -> None:

        sim_time_s = 0.0

        while sim_time_s <= max_time_s:
            self.control_input = controller.compute_control_input(
                state=self.state,
            )
            self.state = self.state_integrator_fn(
                state=self.state,
                control_input=self.control_input,
                state_derivative_fn=self.dynamics.compute_state_derivative,
                dt=dt,
            )
            self.timestamp_s = sim_time_s

            self.visualize()

    def realtime_simulate(
        self,
        controller: Controller,
        realtime_rate: float,
        max_time_s: float = np.inf,
    ) -> None:
        """
        Runs the simulation in real-time.
        To do this accurately, we need to run them in separate processes and communicate the
        Not Implemented yet.
        """
        raise NotImplementedError


if __name__ == "__main__":
    ...

"""
Light-weight simulator for kinematic and simple dynamic systems.
Useful for quick prototyping with custom visualization without having to set up a PyBullet/Drake env.
"""

from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import attr
import numpy as np

from atium.core.control.constructs import IController
from atium.core.dynamics.constructs import IDynamics
from atium.core.simulation.integrators.state_integrators import STATE_INTEGRATORS_FN_MAP, StateIntegratorType
from atium.core.utils.custom_types import ControlInputVector, StateVector

TDynamics = TypeVar("TDynamics", bound=IDynamics)


@attr.define
class SiliconSimulator(Generic[TDynamics], metaclass=ABCMeta):
    state: StateVector
    control_input: ControlInputVector
    dynamics: TDynamics
    timestamp_s: float = 0.0
    _stop_sim: bool = False

    @abstractmethod
    def visualize(self) -> None:
        raise NotImplementedError

    def step_simulate(
        self,
        controller: IController,
        dt: float,
        max_time_s: float = np.inf,
        state_integrator_type: StateIntegratorType = StateIntegratorType.EXPLICIT_EULER,
    ) -> None:
        """
        Lots of ugly side-effects here but it keeps things simple.
        It isn't dirty if you avert your eyes.
        """
        self.timestamp_s = 0.0  # TODO: Maybe option to not reset?
        state_integrator_fn = STATE_INTEGRATORS_FN_MAP[state_integrator_type]

        while self.timestamp_s <= max_time_s and not self._stop_sim:
            control_input = controller.compute_control_input(
                state=self.state,
            )
            # Bound the control input.
            self.control_input = self.dynamics.bound_control_input(control_input=control_input)

            state = state_integrator_fn(
                state=self.state,
                control_input=self.control_input,
                state_derivative_fn=self.dynamics.compute_state_derivative,
                dt=dt,
            )
            # Normalize and bound the state.
            state = self.dynamics.normalize_state(state=state)
            self.state = self.dynamics.bound_state(state=state)

            self.timestamp_s += dt

            self.visualize()

    def realtime_simulate(
        self,
        controller: IController,
        realtime_rate: float,
        max_time_s: float = np.inf,
        state_integrator_type: StateIntegratorType = StateIntegratorType.EXPLICIT_EULER,
    ) -> None:
        """
        Runs the simulation in real-time.
        To do this accurately, we need to run them in separate processes and communicate the
        Not Implemented yet.
        """
        raise NotImplementedError


if __name__ == "__main__":
    ...

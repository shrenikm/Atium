"""
Light-weight simulator for kinematic and simple dynamic systems.
Useful for quick prototyping with custom visualization without having to set up a PyBullet/Drake env.
"""
from abc import ABCMeta, abstractmethod

import attr
import numpy as np

from common.control.interfaces import IController
from common.custom_types import (
    ControlInputVector,
    ControlInputVectorLimits,
    StateVector,
    StateVectorLimits,
)
from common.dynamics.interfaces import IDynamics
from common.simulation.integrators.state_integrators import (
    STATE_INTEGRATORS_FN_MAP,
    StateIntegratorType,
)


@attr.define
class SiliconSimulator(metaclass=ABCMeta):
    state: StateVector
    control_input: ControlInputVector
    state_limits: StateVectorLimits
    control_input_limits: ControlInputVectorLimits
    dynamics: IDynamics
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
            # Apply control limits
            self.control_input = np.clip(
                control_input,
                self.control_input_limits.lower,
                self.control_input_limits.upper,
            )
            state = state_integrator_fn(
                state=self.state,
                control_input=self.control_input,
                state_derivative_fn=self.dynamics.compute_state_derivative,
                dt=dt,
            )
            # Normalize and apply state limits.
            state = self.dynamics.normalize_state(state=state)
            self.state = np.clip(
                state, self.state_limits.lower, self.state_limits.upper
            )
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

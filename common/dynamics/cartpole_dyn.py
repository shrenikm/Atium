import attr
import numpy as np
from common.constants import ACC_GRAVITY

from common.custom_types import ControlInputVector, StateDerivativeVector, StateVector


@attr.frozen
class CartpoleDynamics:

    m_c: float
    m_p: float
    l: float
    g: float = ACC_GRAVITY

    def compute_state_derivative(
        self,
        state: StateVector,
        control_input: ControlInputVector,
    ) -> StateDerivativeVector:
        """
        State = [x, theta, x_dot, theta_dot]
        Control Input = [F_x]
        """
        assert state.size == 4
        assert control_input.size == 1

        _, theta, x_dot, theta_dot = state
        f_x = control_input[0]
        ct, st = np.cos(theta), np.sin(theta)

        k1 = self.m_c + self.m_p * st**2
        k2 = self.m_p * st * (self.l * theta_dot**2 + self.g * ct)
        k3 = (
            -self.m_p * self.l * theta_dot**2 * ct * st
            - (self.m_c + self.m_p) * self.g * st
        )

        x_ddot = (1.0 / k1) * (f_x + k2)
        theta_ddot = (1.0 / (self.l * k1)) * (-f_x * ct + k3)

        return np.array([x_dot, theta_dot, x_ddot, theta_ddot])

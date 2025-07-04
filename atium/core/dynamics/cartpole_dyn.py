import attr
import numpy as np

from atium.core.dynamics.constructs import IDynamics
from atium.core.utils.constants import ACC_GRAVITY
from atium.core.utils.custom_types import ControlInputVector, StateDerivativeVector, StateVector
from atium.core.utils.geometry_utils import normalize_angle


@attr.frozen
class CartpoleParams:
    m_c: float
    m_p: float
    l: float
    g: float = ACC_GRAVITY


@attr.frozen
class CartpoleDynamics(IDynamics):
    params: CartpoleParams

    def normalize_state(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Normalize the theta value.
        """
        normalized_state = np.copy(state)
        normalized_state[1] = normalize_angle(angle=normalized_state[1])
        return normalized_state

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

        x, theta, x_dot, theta_dot = state
        f_x = control_input[0]
        ct, st = np.cos(theta), np.sin(theta)

        # If the cart is at either x limit, we don't allow a force in that direction.
        # If we don't do this, the cart will remain stationary at the x limits due to clipping,
        # but the pole will continue swinging as the force will induce a change of state.
        if np.isclose(x, self.state_limits.lower[0]) and f_x < 0.0:
            f_x = 0.0
        if np.isclose(x, self.state_limits.upper[0]) and f_x > 0.0:
            f_x = 0.0

        k1 = self.params.m_c + self.params.m_p * st**2
        k2 = self.params.m_p * st * (self.params.l * theta_dot**2 + self.params.g * ct)
        k3 = (
            -self.params.m_p * self.params.l * theta_dot**2 * ct * st
            - (self.params.m_c + self.params.m_p) * self.params.g * st
        )

        x_ddot = (1.0 / k1) * (f_x + k2)
        theta_ddot = (1.0 / (self.params.l * k1)) * (-f_x * ct + k3)

        return np.array([x_dot, theta_dot, x_ddot, theta_ddot])

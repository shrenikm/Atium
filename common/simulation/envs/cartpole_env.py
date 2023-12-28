from __future__ import annotations

import attr
import cv2
import numpy as np
from typing_extensions import override

from common.colors import AtiumColorsBGR
from common.constants import ACC_GRAVITY
from common.custom_types import ControlInputVector, StateVector
from common.dynamics.cartpole_dyn import CartpoleDynamics, CartpoleParams
from common.dynamics.interfaces import IDynamics
from common.dynamics.utils import ControlInputVectorLimits, StateVectorLimits
from common.img_utils import (
    create_canvas,
    draw_circle_on_canvas,
    draw_line_on_canvas,
    draw_rectangle_on_canvas,
)
from common.simulation.silicon.silicon_simulator import SiliconSimulator

RESOLUTION = 0.01


@attr.define
class CartpoleSiliconEnv(SiliconSimulator[CartpoleDynamics]):
    @classmethod
    def from_params(
        cls,
        initial_state: StateVector,
        initial_control_input: ControlInputVector,
        state_limits: StateVectorLimits,
        control_input_limits: ControlInputVectorLimits,
        params: CartpoleParams,
    ) -> CartpoleSiliconEnv:
        dynamics = CartpoleDynamics(
            state_limits=state_limits,
            control_input_limits=control_input_limits,
            params=params,
        )
        return cls(
            state=initial_state,
            control_input=initial_control_input,
            dynamics=dynamics,
        )

    @override
    def visualize(self) -> None:

        window_name = f"{self.__class__.__name__}"
        x, theta, _, _ = self.state

        cv2.namedWindow(window_name)

        env_width = (
            self.dynamics.state_limits.upper[0] - self.dynamics.state_limits.lower[0]
        )
        env_height = max(env_width, 2 * self.dynamics.params.l)

        img_width = int(env_width // RESOLUTION)
        img_height = int(env_height // RESOLUTION)

        canvas = create_canvas(
            img_width=img_width,
            img_height=img_height,
            color=AtiumColorsBGR.WHITE,
        )
        # Draw horizon line.
        draw_line_on_canvas(
            canvas=canvas,
            start_xy=(0.0, env_height / 2.0),
            end_xy=(env_width, env_height / 2.0),
            color=AtiumColorsBGR.BLACK,
            thickness_px=2,
            resolution=RESOLUTION,
        )

        # Draw cart.
        cart_length_m, cart_width_m = 1.0, 0.2
        # Reframing the cart's x coordinate so that the canvas has the min state x limit at 0.
        cart_x = x - self.dynamics.state_limits.lower[0]
        cart_center_xy = (cart_x, env_height / 2.0)
        draw_rectangle_on_canvas(
            canvas=canvas,
            center_xy=cart_center_xy,
            length=cart_length_m,
            width=cart_width_m,
            color=AtiumColorsBGR.BLUE,
            thickness_px=cv2.FILLED,
            resolution=RESOLUTION,
        )

        # Draw pole
        # Find the end coordinate of the pole.
        # theta = 0 means that the pole is oriented straight down.
        pole_end_xy = (
            cart_center_xy[0] + np.cos(theta - np.pi / 2.0),
            cart_center_xy[1] + np.sin(theta - np.pi / 2.0),
        )
        draw_line_on_canvas(
            canvas=canvas,
            start_xy=cart_center_xy,
            end_xy=pole_end_xy,
            color=AtiumColorsBGR.LIGHT_RED,
            thickness_px=5,
            resolution=RESOLUTION,
        )

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1)
        if key & 0xFF in (ord("q"), 27):
            self._stop_sim = True

from typing import Self, override

import attr
import cv2
import numpy as np

from atium.core.dynamics.cartpole_dyn import CartpoleDynamics, CartpoleParams
from atium.core.dynamics.constructs import ControlInputVectorLimits, StateVectorLimits
from atium.core.simulation.silicon.silicon_simulator import SiliconSimulator
from atium.core.utils.colors import AtiumColorsBGR
from atium.core.utils.custom_types import ControlInputVector, StateVector
from atium.core.utils.img_utils import create_canvas, draw_line_on_canvas, draw_rectangle_on_canvas

RESOLUTION = 0.015
MIN_ENV_SIZE = 10.0


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
    ) -> Self:
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

        env_width = self.dynamics.state_limits.upper[0] - self.dynamics.state_limits.lower[0]
        env_height = max(env_width, 2 * self.dynamics.params.l)
        env_size = max(env_width, env_height, MIN_ENV_SIZE)

        img_width = int(env_size // RESOLUTION)
        img_height = int(env_size // RESOLUTION)

        canvas = create_canvas(
            img_width=img_width,
            img_height=img_height,
            color=AtiumColorsBGR.WHITE,
        )

        min_x, max_x = (
            self.dynamics.state_limits.lower[0],
            self.dynamics.state_limits.upper[0],
        )
        # To go from 'x' in the system frame to canvas frame (origin bottom left corner),
        # we need to x - min_x + padding
        # Where padding accounts for the fact that the canvas can be bigger than the system x limits.
        padding = (env_size - (max_x - min_x)) / 2.0

        def _system_x_to_canvas_x(x: float) -> float:
            """
            Changes frames from system fframe to the canvas frame.
            """
            return x - min_x + padding

        # Max and min x in the canvas frame (still in meters)
        min_x_canvas = _system_x_to_canvas_x(x=min_x)
        max_x_canvas = _system_x_to_canvas_x(x=max_x)

        # Draw horizon line for the track.
        draw_line_on_canvas(
            canvas=canvas,
            start_xy=(min_x_canvas, env_size / 2.0),
            end_xy=(max_x_canvas, env_size / 2.0),
            color=AtiumColorsBGR.BLACK,
            thickness_px=2,
            resolution=RESOLUTION,
        )
        # Draw vertical lines for the x bounds.
        draw_line_on_canvas(
            canvas=canvas,
            start_xy=(min_x_canvas, 0.0),
            end_xy=(min_x_canvas, env_size),
            color=AtiumColorsBGR.LIGHT_GRAY,
            thickness_px=2,
            resolution=RESOLUTION,
        )
        draw_line_on_canvas(
            canvas=canvas,
            start_xy=(max_x_canvas, 0.0),
            end_xy=(max_x_canvas, env_size),
            color=AtiumColorsBGR.LIGHT_GRAY,
            thickness_px=2,
            resolution=RESOLUTION,
        )

        # Draw cart.
        cart_length_m, cart_width_m = 1.0, 0.2
        # Reframing the cart's x coordinate so that the canvas has the min state x limit at 0.
        cart_x = _system_x_to_canvas_x(x=x)
        cart_center_xy = (cart_x, env_size / 2.0)
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

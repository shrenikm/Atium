import attr
import cv2
import numpy as np
from typing_extensions import override

from common.colors import AtiumColorsBGR
from common.constants import ACC_GRAVITY
from common.dynamics.cartpole_dyn import CartpoleDynamics
from common.dynamics.interfaces import IDynamics
from common.img_utils import (
    create_canvas,
    draw_circle_on_canvas,
    draw_line_on_canvas,
    draw_rectangle_on_canvas,
)
from common.simulation.silicon.silicon_simulator import SiliconSimulator

RESOLUTION = 0.01


@attr.define
class CartpoleSiliconEnv(SiliconSimulator):
    m_c: float = 1.0
    m_p: float = 0.5
    l: float = 1.0
    g: float = ACC_GRAVITY

    dynamics: IDynamics = attr.ib(init=False)

    @dynamics.default  # pyright: ignore (Doesn't play well with attr here.)
    def _initialize_dynamics(self) -> IDynamics:
        return CartpoleDynamics(
            m_c=self.m_c,
            m_p=self.m_p,
            l=self.l,
            g=self.g,
        )

    @override
    def visualize(self) -> None:

        window_name = f"{self.__class__.__name__}"
        x, theta, _, _ = self.state

        cv2.namedWindow(window_name)

        env_width = self.state_limits.upper[0] - self.state_limits.lower[0]
        env_height = max(env_width, 2 * self.l)

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
        cart_x = x - self.state_limits.lower[0]
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

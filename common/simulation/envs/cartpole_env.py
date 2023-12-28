import attr
import cv2
import numpy as np
from typing_extensions import override

from common.colors import AtiumColorsBGR
from common.constants import ACC_GRAVITY
from common.dynamics.cartpole_dyn import CartpoleDynamics
from common.dynamics.interfaces import IDynamics
from common.img_utils import create_canvas
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
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
        cv2.line(canvas, (0, 0), (0, 0), AtiumColorsBGR.BLACK, 1)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1)
        if key & 0xFF in (ord("q"), 27):
            self._stop_sim = True

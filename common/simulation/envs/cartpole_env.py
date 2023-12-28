import attr
import numpy as np
from typing_extensions import override

from common.constants import ACC_GRAVITY
from common.dynamics.cartpole_dyn import CartpoleDynamics
from common.dynamics.interfaces import Dynamics
from common.simulation.silicon.silicon_simulator import SiliconSimulator

RESOLUTION = 0.01


@attr.define
class CartpoleSiliconEnv(SiliconSimulator):
    m_c: float = 1.0
    m_p: float = 0.5
    l: float = 1.0
    g: float = ACC_GRAVITY

    dynamics: Dynamics = attr.ib(init=False)

    @dynamics.default  # pyright: ignore (Doesn't play well with attr here.)
    def _initialize_dynamics(self) -> Dynamics:
        return CartpoleDynamics(
            m_c=self.m_c,
            m_p=self.m_p,
            l=self.l,
            g=self.g,
        )

    @override
    def visualize(self) -> None:
        ...

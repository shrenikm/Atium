import attr
import numpy as np
from common.control.constructs import IController

from common.custom_types import (
    ControlInputVector,
    StateVector,
)
from common.dynamics.utils import ControlInputVectorLimits


@attr.define
class ZeroController(IController):
    size: int

    def compute_control_input(
        self,
        state: StateVector,
    ) -> ControlInputVector:
        return np.zeros(self.size)


@attr.define
class ConstantController(IController):
    control_input: ControlInputVector

    def compute_control_input(
        self,
        state: StateVector,
    ) -> ControlInputVector:
        return self.control_input


@attr.define
class UniformPDFController(IController):
    control_input_limits: ControlInputVectorLimits
    seed: int = 0

    _rng: np.random.RandomState = attr.ib(init=False)

    @_rng.default  # pyright: ignore
    def _initialize_rng(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def compute_control_input(
        self,
        state: StateVector,
    ) -> ControlInputVector:
        return self._rng.uniform(
            self.control_input_limits.lower, self.control_input_limits.upper
        )


@attr.define
class NormalPDFController(IController):
    size: int
    mean: float
    std: float
    seed: int = 0

    _rng: np.random.RandomState = attr.ib(init=False)

    @_rng.default  # pyright:ignore
    def _initialize_rng(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def compute_control_input(
        self,
        state: StateVector,
    ) -> ControlInputVector:
        return self._rng.normal(self.mean, self.std, self.size)

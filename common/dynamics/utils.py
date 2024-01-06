from __future__ import annotations

import attr
import numpy as np

from common.custom_types import ControlInputVector, StateVector


@attr.frozen
class StateVectorLimits:
    lower: StateVector
    upper: StateVector

    @classmethod
    def from_free(cls, size: int) -> StateVectorLimits:
        """
        Free limits <=> No limits.
        -inf to inf.
        """
        return cls(
            lower=np.full(size, -np.inf),
            upper=np.full(size, np.inf),
        )


@attr.frozen
class ControlInputVectorLimits:
    lower: ControlInputVector
    upper: ControlInputVector

    @classmethod
    def from_free(cls, size: int) -> ControlInputVectorLimits:
        """
        Free limits <=> No limits.
        -inf to inf.
        """
        return cls(
            lower=np.full(size, -np.inf),
            upper=np.full(size, np.inf),
        )

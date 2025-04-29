from functools import cached_property

import attr

from atium.core.utils.custom_types import EnvironmentArray2D, SizeXY


@attr.define
class EnvironmentMap2D:
    """
    A map of obstacles in the environment represented as a numpy array.
    """

    array: EnvironmentArray2D
    resolution: float

    @cached_property
    def size_xy(self) -> SizeXY:
        pass

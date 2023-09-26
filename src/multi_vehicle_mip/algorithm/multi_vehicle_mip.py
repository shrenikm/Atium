from typing import Sequence

import attr

from common.custom_types import (
    AMatrix,
    BMatrix,
    PointXYArray,
    Polygon2DArray,
    StateTrajectoryArray,
    StateVector,
    VelocityXYArray,
)


@attr.s(auto_attribs=True, frozen=True)
class MVMIPCosts:
    q_cost: float
    r_cost: float
    p_cost: float


@attr.s(auto_attribs=True, frozen=True)
class MVMIPVehicle:
    a_matrix: AMatrix
    b_matrix: BMatrix
    initial_state: StateVector
    final_state: StateVector


@attr.s(auto_attribs=True, frozen=True)
class MVMIPObstacle:
    polygon: Polygon2DArray
    start_xy: PointXYArray
    velocity_xy: VelocityXYArray


def solve_mv_mip(
    costs: MVMIPCosts,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Sequence[StateTrajectoryArray]:
    ...

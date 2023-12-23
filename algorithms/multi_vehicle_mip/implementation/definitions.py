import numpy as np
from functools import cached_property
from typing import Protocol, Sequence, Union
import attr
from algorithms.multi_vehicle_mip.implementation.custom_types import (
    VehicleControlTrajectoryMap,
    VehicleStateTrajectoryMap,
)

from common.custom_types import (
    AMatrix,
    BMatrix,
    ControlVector,
    CostVector,
    PointXYArray,
    PointXYVector,
    Polygon2DArray,
    SizeXYVector,
    StateVector,
    VelocityXYArray,
    VelocityXYVector,
)


@attr.frozen
class MVMIPOptimizationParams:
    num_time_steps: int
    dt: float
    M: float
    result_float_precision: int


@attr.frozen
class MVMIPVehicleDynamics:
    a_matrix: AMatrix
    b_matrix: BMatrix
    initial_state: StateVector
    final_state: StateVector
    # Clearance required to be maintained by other vehicles to this one.
    clearance_m: float


@attr.frozen
class MVMIPVehicleOptimizationParams:
    q_cost_vector: CostVector
    r_cost_vector: CostVector
    p_cost_vector: CostVector
    state_min: StateVector
    state_max: StateVector
    control_min: ControlVector
    control_max: ControlVector


@attr.frozen
class MVMIPVehicle:
    dynamics: MVMIPVehicleDynamics
    optimization_params: MVMIPVehicleOptimizationParams


@attr.frozen
class MVMIPObstacle(Protocol):
    def ordered_corner_points(self, time_step_id: int) -> Polygon2DArray:
        raise NotImplemented


@attr.frozen(slots=False)
class MVMIPRectangleObstacle(MVMIPObstacle):
    initial_center_xy: PointXYVector
    size_xy_m: SizeXYVector
    velocities_xy_mps: Union[VelocityXYVector, VelocityXYArray]
    clearance_m: float

    # TODO: Not a fan of having these attributes here (duplciated from optimization params)
    num_time_steps: int
    dt: float

    @cached_property
    def centers_xy(self) -> PointXYArray:
        centers_xy = np.empty((self.num_time_steps + 1, 2), dtype=np.float64)

        if self.velocities_xy_mps.size == 2:
            velocities_xy_mps = np.repeat(
                self.velocities_xy_mps.reshape(1, 2), self.num_time_steps, axis=0
            )
        else:
            assert self.velocities_xy_mps.ndim == 2
            assert self.velocities_xy_mps.shape[0] == self.num_time_steps
            velocities_xy_mps = self.velocities_xy_mps

        centers_xy[0] = self.initial_center_xy
        for i in range(1, self.num_time_steps + 1):
            centers_xy[i] = centers_xy[i - 1] + self.dt * velocities_xy_mps[i - 1]

        return centers_xy

    def compute_min_limits_xy(
        self,
        time_step_id: int,
    ) -> PointXYVector:
        """
        Min limit for MVMIP optimization.
        This corresponds to the bottom left coordinate (including clearance)
        """
        return self.centers_xy[time_step_id] - 0.5 * self.size_xy_m - self.clearance_m

    def compute_max_limits_xy(
        self,
        time_step_id: int,
    ) -> PointXYVector:
        """
        Max limit for MVMIP optimization.
        This corresponds to the bottom left coordinate (including clearance)
        """
        return self.centers_xy[time_step_id] + 0.5 * self.size_xy_m + self.clearance_m

    def ordered_corner_points(self, time_step_id: int) -> Polygon2DArray:
        xc, yc = self.centers_xy[time_step_id]
        xhs, yhs = 0.5 * self.size_xy_m  # Half sizes.

        return np.array(
            [
                [xc - xhs, yc - yhs],
                [xc - xhs, yc + yhs],
                [xc + xhs, yc + yhs],
                [xc + xhs, yc - yhs],
            ],
            dtype=np.float64,
        )


@attr.frozen
class MVMIPPolygonObstacle(MVMIPObstacle):
    polygon: Polygon2DArray
    initial_center_xy: PointXYArray
    velocities_xy_mps: Union[VelocityXYVector, VelocityXYArray]
    clearance_m: float

    def ordered_corner_points(self) -> Polygon2DArray:
        return self.polygon


@attr.frozen
class MVMIPResult:
    # Actual result attributes.
    objective_value: float
    vehicle_state_trajectory_map: VehicleStateTrajectoryMap
    vehicle_control_trajectory_map: VehicleControlTrajectoryMap
    # Attributes that make the result self-sufficient
    mvmip_params: MVMIPOptimizationParams
    vehicles: Sequence[MVMIPVehicle]
    obstacles: Sequence[MVMIPObstacle]
    # Performance attributes.
    solver_setup_time_s: float
    solver_solving_time_s: float

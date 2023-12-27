from functools import lru_cache
from typing import Protocol, Sequence, Union

import attr
import numpy as np

from algorithms.multi_vehicle_mip.implementation.custom_types import (
    VehicleControlTrajectoryMap,
    VehicleStateTrajectoryMap,
)
from common.attrs_utils import AttrsConverters
from common.custom_types import (
    AMatrix,
    BMatrix,
    ControlVector,
    CostVector,
    PointXYArray,
    PointXYVector,
    PolygonXYArray,
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
    a_matrix: AMatrix = attr.ib(converter=AttrsConverters.np_f64_converter())
    b_matrix: BMatrix = attr.ib(converter=AttrsConverters.np_f64_converter())
    initial_state: StateVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    final_state: StateVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    # Clearance required to be maintained by other vehicles to this one.
    clearance_m: float


@attr.frozen
class MVMIPVehicleOptimizationParams:
    q_cost_vector: CostVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    r_cost_vector: CostVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    p_cost_vector: CostVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    state_min: StateVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    state_max: StateVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    control_min: ControlVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    control_max: ControlVector = attr.ib(converter=AttrsConverters.np_f64_converter())


@attr.frozen
class MVMIPVehicle:
    dynamics: MVMIPVehicleDynamics
    optimization_params: MVMIPVehicleOptimizationParams


@attr.frozen(slots=False, eq=False)
class MVMIPObstacle(Protocol):
    initial_center_xy: PointXYVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    size_xy_m: SizeXYVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    velocities_xy_mps: Union[VelocityXYVector, VelocityXYArray] = attr.ib(
        converter=AttrsConverters.np_f64_converter()
    )
    clearance_m: float

    def ordered_corner_points_xy(
        self,
        time_step_id: int,
        num_time_steps: int,
        dt: float,
        add_clearance: bool,
    ) -> PolygonXYArray:
        raise NotImplemented


@attr.frozen(slots=False, eq=False)
class MVMIPRectangleObstacle:
    initial_center_xy: PointXYVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    size_xy_m: SizeXYVector = attr.ib(converter=AttrsConverters.np_f64_converter())
    velocities_xy_mps: Union[VelocityXYVector, VelocityXYArray] = attr.ib(
        converter=AttrsConverters.np_f64_converter()
    )
    clearance_m: float

    @lru_cache
    def get_centers_xy(
        self,
        num_time_steps: int,
        dt: float,
    ) -> PointXYArray:
        centers_xy = np.empty((num_time_steps + 1, 2), dtype=np.float64)

        if self.velocities_xy_mps.size == 2:
            velocities_xy_mps = np.repeat(
                self.velocities_xy_mps.reshape(1, 2), num_time_steps, axis=0
            )
        else:
            assert self.velocities_xy_mps.ndim == 2
            assert self.velocities_xy_mps.shape[0] == num_time_steps
            velocities_xy_mps = self.velocities_xy_mps

        centers_xy[0] = self.initial_center_xy
        for i in range(1, num_time_steps + 1):
            centers_xy[i] = centers_xy[i - 1] + dt * velocities_xy_mps[i - 1]

        return centers_xy

    def compute_min_limits_xy(
        self,
        time_step_id: int,
        num_time_steps: int,
        dt: float,
    ) -> PointXYVector:
        """
        Min limit for MVMIP optimization.
        This corresponds to the bottom left coordinate (including clearance)
        """
        centers_xy = self.get_centers_xy(num_time_steps, dt)
        return centers_xy[time_step_id] - 0.5 * self.size_xy_m - self.clearance_m

    def compute_max_limits_xy(
        self,
        time_step_id: int,
        num_time_steps: int,
        dt: float,
    ) -> PointXYVector:
        """
        Max limit for MVMIP optimization.
        This corresponds to the bottom left coordinate (including clearance)
        """
        centers_xy = self.get_centers_xy(num_time_steps, dt)
        return centers_xy[time_step_id] + 0.5 * self.size_xy_m + self.clearance_m

    def ordered_corner_points_xy(
        self,
        time_step_id: int,
        num_time_steps: int,
        dt: float,
        add_clearance: bool,
    ) -> PolygonXYArray:
        centers_xy = self.get_centers_xy(num_time_steps, dt)
        xc, yc = centers_xy[time_step_id]
        if add_clearance:
            # Half sizes with clearance.
            xhs, yhs = 0.5 * self.size_xy_m + self.clearance_m
        else:
            # Half sizes.
            xhs, yhs = 0.5 * self.size_xy_m

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
class MVMIPPolygonObstacle:
    polygon: PolygonXYArray = attr.ib(converter=AttrsConverters.np_f64_converter())
    initial_center_xy: PointXYArray = attr.ib(converter=AttrsConverters.np_f64_converter())
    velocities_xy_mps: Union[VelocityXYVector, VelocityXYArray] = attr.ib(
        converter=AttrsConverters.np_f64_converter()
    )
    clearance_m: float

    def ordered_corner_points_xy(
        self,
        time_step_id: float,
        num_time_steps: int,
        dt: float,
        add_clearance: bool,
    ) -> PolygonXYArray:
        raise NotImplementedError


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


if __name__ == "__main__":

    @attr.frozen(slots=False, eq=False)
    class A:
        a: int
        v: SizeXYVector

        @lru_cache
        def f(self, b) -> int:
            return self.a**2 + b

    a = A(3, np.array([2.0, 3.0]))
    print(a.f(2))

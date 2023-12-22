import attr

from common.custom_types import (
    AMatrix,
    BMatrix,
    ControlVector,
    CostVector,
    PointXYArray,
    PointXYVector,
    Polygon2DArray,
    StateVector,
    VelocityXYArray,
)


@attr.frozen
class MVMIPOptimizationParams:
    num_time_steps: int
    dt: float


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
class MVMIPObstacle:
    pass


@attr.frozen
class MVMIPRectangleObstacle(MVMIPObstacle):
    center: PointXYVector
    x_size_m: float
    y_size_m: float
    velocity_xy_mps: VelocityXYArray
    clearance_m: float


@attr.frozen
class MVMIPPolygonObstacle(MVMIPObstacle):
    polygon: Polygon2DArray
    start_xy: PointXYArray
    velocity_xy_mps: VelocityXYArray
    clearance_m: float

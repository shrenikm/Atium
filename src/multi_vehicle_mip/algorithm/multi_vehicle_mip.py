from typing import Sequence

import attr
from ortools.linear_solver import pywraplp

from common.custom_types import (
    AMatrix,
    BMatrix,
    ControlVector,
    CostVector,
    PointXYArray,
    PointXYVector,
    Polygon2DArray,
    StateTrajectoryArray,
    StateVector,
    VelocityXYArray,
)
from src.multi_vehicle_mip.algorithm.utils import \
    control_slack_variable_str_from_ids as csv
from src.multi_vehicle_mip.algorithm.utils import \
    control_variable_str_from_ids as cv
from src.multi_vehicle_mip.algorithm.utils import \
    state_slack_variable_str_from_ids as ssv
from src.multi_vehicle_mip.algorithm.utils import \
    state_variable_str_from_ids as sv
from src.multi_vehicle_mip.algorithm.utils import \
    vehicle_collision_binary_slack_variable_str_from_ids as vbsv
from src.multi_vehicle_mip.algorithm.utils import \
    obstacle_collision_binary_slack_variable_str_from_ids as obsv


@attr.frozen
class MVMIPOptimizationParams:
    num_time_steps: int

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

@attr.frozen
class MVMIPPolygonObstacle(MVMIPObstacle):
    polygon: Polygon2DArray
    start_xy: PointXYArray
    velocity_xy_mps: VelocityXYArray
    clearance_m: float


def solve_mv_mip(
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Sequence[StateTrajectoryArray]:

    # TODO: Check validity of vehicles and obstacles.

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("GLOP")
    assert solver is not None, "Solver could not be created."

    nt = mvmip_params.num_time_steps

    vars = dict()
    # Creating state and control trajectory variables for each vehicle.
    for vehicle_id, vehicle in enumerate(vehicles):
        nx = vehicle.dynamics.a_matrix.shape[0]
        nu = vehicle.dynamics.b_matrix.shape[1]
        for time_step_id in range(1, nt + 1):
            for state_id in range(nx):
                # State variable
                var_str = sv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                min_limit = vehicle.optimization_params.state_min[state_id]
                max_limit = vehicle.optimization_params.state_max[state_id]
                vars[var_str] = solver.NumVar(min_limit, max_limit, var_str)

                # State slack variable
                var_str = ssv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                vars[var_str] = solver.NumVar(-solver.infinity(), solver.infinity(), var_str)

        for time_step_id in range(nt):
            for control_id in range(nu):
                # Control variable
                var_str = cv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    control_id=control_id,
                )
                min_limit = vehicle.optimization_params.state_min[control_id]
                max_limit = vehicle.optimization_params.state_max[control_id]
                vars[var_str] = solver.NumVar(min_limit, max_limit, var_str)

                # Control slack variable
                var_str = csv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    control_id=control_id,
                )
                vars[var_str] = solver.NumVar(-solver.infinity(), solver.infinity(), var_str)

        # Binary variables for vehicle-vehicle collision constraint variables.
        for time_step_id in range(1, nt + 1):
            for other_vehicle_id in range(vehicle_id + 1, len(vehicles)):
                for var_id in range(4):
                    var_str = vbsv(
                            vehicle_id=vehicle_id,
                            other_vehicle_id=other_vehicle_id,
                            time_step_id=time_step_id,
                            var_id=var_id,
                    )
                    vars[var_str] = solver.IntVar(0, 1, var_str)


        # Vehicle-obstacle collision constraint variables.
        for time_step_id in range(1, nt + 1):
            for obstacle_id, obstacle in enumerate(obstacles):
                if isinstance(obstacle, MVMIPRectangleObstacle):
                    for var_id in range(4):
                        var_str = obsv(
                                vehicle_id=vehicle_id,
                                obstacle_id=obstacle_id,
                                time_step_id=time_step_id,
                                var_id=var_id,
                        )
                        vars[var_str] = solver.IntVar(0, 1, var_str)
                else:
                    raise NotImplemented("Only rectangular obstacles have been implemented.")





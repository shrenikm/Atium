from typing import Sequence

import attr
from ortools.linear_solver import pywraplp

from common.custom_types import (
    AMatrix,
    BMatrix,
    ControlVector,
    CostVector,
    PointXYArray,
    Polygon2DArray,
    StateTrajectoryArray,
    StateVector,
    VelocityXYArray,
)
from src.multi_vehicle_mip.algorithm.utils import (
    control_variable_str_from_ids,
    state_variable_str_from_ids,
)


@attr.frozen
class MVMIPVehicleDynamics:
    a_matrix: AMatrix
    b_matrix: BMatrix
    initial_state: StateVector
    final_state: StateVector
    # Clearance for other vehicles around this one.
    clearance_m: float


@attr.s(auto_attribs=True, frozen=True)
class MVMIPVehicleOptimizationParams:
    num_time_steps: int
    q_cost_vector: CostVector
    r_cost_vector: CostVector
    p_cost_vector: CostVector
    state_min: StateVector
    state_max: StateVector
    control_min: ControlVector
    control_max: ControlVector


@attr.s(auto_attribs=True, frozen=True)
class MVMIPVehicle:
    dynamics: MVMIPVehicleDynamics
    optimization_params: MVMIPVehicleOptimizationParams


@attr.s(auto_attribs=True, frozen=True)
class MVMIPObstacle:
    polygon: Polygon2DArray
    start_xy: PointXYArray
    velocity_xy: VelocityXYArray
    clearance_m: float


def solve_mv_mip(
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Sequence[StateTrajectoryArray]:

    # TODO: Check validity of vehicles and obstacles.

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("GLOP")
    assert solver is not None, "Solver could not be created."

    vars = dict()
    # Creating state and control trajectory variables for each vehicle.
    for vehicle_id, vehicle in enumerate(vehicles):
        nt = vehicle.optimization_params.num_time_steps
        nx = vehicle.dynamics.a_matrix.shape[0]
        nu = vehicle.dynamics.b_matrix.shape[1]
        for time_step_id in range(nt):
            for state_id in range(nx):
                var_str = state_variable_str_from_ids(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                min_limit = vehicle.optimization_params.state_min[state_id]
                max_limit = vehicle.optimization_params.state_max[state_id]
                vars[var_str] = solver.NumVar(min_limit, max_limit, var_str)

            for control_id in range(nu):
                var_str = control_variable_str_from_ids(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    control_id=control_id,
                )
                min_limit = vehicle.optimization_params.state_min[control_id]
                max_limit = vehicle.optimization_params.state_max[control_id]
                vars[var_str] = solver.NumVar(min_limit, max_limit, var_str)

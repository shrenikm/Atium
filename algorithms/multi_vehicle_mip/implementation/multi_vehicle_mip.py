from typing import Dict, Sequence
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
from algorithms.multi_vehicle_mip.implementation.utils import (
    control_slack_constraint_var_from_var_strs as csc,
    control_slack_variable_str_from_ids as csv,
    control_variable_str_from_ids as cv,
    state_slack_constraint_var_from_var_strs as ssc,
    state_slack_variable_str_from_ids as ssv,
    state_transition_constraint_var_from_var_strs as stc,
    state_variable_str_from_ids as sv,
    vehicle_obstacle_collision_binary_constraint_var_from_ids as ocbc,
    vehicle_obstacle_collision_binary_slack_variable_str_from_ids as obsv,
    vehicle_obstacle_collision_constraint_var_from_var_strs as occ,
    vehicle_vehicle_collision_binary_constraint_var_from_ids as vcbc,
    vehicle_vehicle_collision_binary_slack_variable_str_from_ids as vbsv,
    vehicle_vehicle_collision_constraint_var_from_var_strs as vcc,
)

# Types
SolverVariable = pywraplp.Variable
Solver = pywraplp.Solver


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


def create_variables_for_mvmip(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Dict[str, SolverVariable]:

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
                vars[var_str] = solver.NumVar(
                    -solver.infinity(), solver.infinity(), var_str
                )

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
                vars[var_str] = solver.NumVar(
                    -solver.infinity(), solver.infinity(), var_str
                )

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
                    raise NotImplemented(
                        "Only rectangular obstacles have been implemented so far."
                    )

    return vars


def construct_constraints_for_mvmip(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> None:

    nt = mvmip_params.num_time_steps

    constraints = dict()

    for vehicle_id, vehicle in enumerate(vehicles):

        # State slack constraints
        nx = vehicle.dynamics.a_matrix.shape[0]
        nu = vehicle.dynamics.b_matrix.shape[1]
        for time_step_id in range(1, nt + 1):
            for state_id in range(nx):
                # State and state slack variables
                s_var_str = sv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                w_var_str = ssv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                constraint = solver.Constraint(
                    -solver.infinity(), 0.0, f"c_{s_var_str}_{w_var_str}"
                )
                vars[var_str] = solver.NumVar(min_limit, max_limit, var_str)

                # State slack variable


def solve_mvmip(
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Sequence[StateTrajectoryArray]:

    # TODO: Check validity of vehicles and obstacles.

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    assert solver is not None, "Solver could not be created."

    vars = create_variables_for_mvmip(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )

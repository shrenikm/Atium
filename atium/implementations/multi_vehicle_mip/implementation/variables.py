from typing import Sequence

from atium.implementations.multi_vehicle_mip.implementation.custom_types import Solver, SolverVariableMap
from atium.implementations.multi_vehicle_mip.implementation.definitions import (
    MVMIPObstacle,
    MVMIPOptimizationParams,
    MVMIPRectangleObstacle,
    MVMIPVehicle,
)
from atium.implementations.multi_vehicle_mip.implementation.utils import assert_uniqueness_and_update_mvmip_map
from atium.implementations.multi_vehicle_mip.implementation.utils import control_slack_variable_str_from_ids as c_sv
from atium.implementations.multi_vehicle_mip.implementation.utils import control_variable_str_from_ids as c_v
from atium.implementations.multi_vehicle_mip.implementation.utils import state_slack_variable_str_from_ids as s_sv
from atium.implementations.multi_vehicle_mip.implementation.utils import state_variable_str_from_ids as s_v
from atium.implementations.multi_vehicle_mip.implementation.utils import (
    vehicle_obstacle_collision_binary_slack_variable_str_from_ids as voc_bsv,
)
from atium.implementations.multi_vehicle_mip.implementation.utils import (
    vehicle_vehicle_collision_binary_slack_variable_str_from_ids as vvc_bsv,
)


def construct_state_slack_variables(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicle: MVMIPVehicle,
) -> SolverVariableMap:
    vars_map = {}
    nt = mvmip_params.num_time_steps
    nx = vehicle.dynamics.a_matrix.shape[0]

    for time_step_id in range(1, nt + 1):
        for state_id in range(nx):
            # State variable
            var_str = s_v(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                state_id=state_id,
            )
            min_limit = vehicle.optimization_params.state_min[state_id]
            max_limit = vehicle.optimization_params.state_max[state_id]
            assert var_str not in vars_map
            vars_map[var_str] = solver.NumVar(min_limit, max_limit, var_str)

            # State slack variable
            var_str = s_sv(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                state_id=state_id,
            )
            assert var_str not in vars_map
            vars_map[var_str] = solver.NumVar(-solver.infinity(), solver.infinity(), var_str)

    return vars_map


def construct_control_slack_variables(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicle: MVMIPVehicle,
) -> SolverVariableMap:
    vars_map = {}
    nt = mvmip_params.num_time_steps
    nu = vehicle.dynamics.b_matrix.shape[1]
    for time_step_id in range(nt):
        for control_id in range(nu):
            # Control variable
            var_str = c_v(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                control_id=control_id,
            )
            min_limit = vehicle.optimization_params.control_min[control_id]
            max_limit = vehicle.optimization_params.control_max[control_id]
            assert var_str not in vars_map
            vars_map[var_str] = solver.NumVar(min_limit, max_limit, var_str)

            # Control slack variable
            var_str = c_sv(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                control_id=control_id,
            )
            assert var_str not in vars_map
            vars_map[var_str] = solver.NumVar(-solver.infinity(), solver.infinity(), var_str)

    return vars_map


def construct_vehicle_obstacle_collision_variables(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    obstacles: Sequence[MVMIPObstacle],
) -> SolverVariableMap:
    vars_map = {}
    nt = mvmip_params.num_time_steps
    # Vehicle-obstacle collision constraint variables.
    for obstacle_id, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, MVMIPRectangleObstacle):
            raise NotImplemented("Only rectangular obstacles have been implemented so far.")
        for time_step_id in range(1, nt + 1):
            for var_id in range(4):
                var_str = voc_bsv(
                    vehicle_id=vehicle_id,
                    obstacle_id=obstacle_id,
                    time_step_id=time_step_id,
                    var_id=var_id,
                )
                assert var_str not in vars_map
                vars_map[var_str] = solver.IntVar(0, 1, var_str)

    return vars_map


def construct_vehicle_vehicle_collision_variables(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicles: Sequence[MVMIPVehicle],
) -> SolverVariableMap:
    vars_map = {}
    nt = mvmip_params.num_time_steps
    # Binary variables for vehicle-vehicle collision constraint variables.
    for time_step_id in range(1, nt + 1):
        for other_vehicle_id in range(vehicle_id + 1, len(vehicles)):
            for var_id in range(4):
                var_str = vvc_bsv(
                    current_vehicle_id=vehicle_id,
                    other_vehicle_id=other_vehicle_id,
                    time_step_id=time_step_id,
                    var_id=var_id,
                )
                assert var_str not in vars_map
                vars_map[var_str] = solver.IntVar(0, 1, var_str)

    return vars_map


def construct_variables_for_mvmip(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> SolverVariableMap:
    vars_map = {}
    # Creating state and control trajectory variables for each vehicle.
    for vehicle_id, vehicle in enumerate(vehicles):
        ss_vars_map = construct_state_slack_variables(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        cs_vars_map = construct_control_slack_variables(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        voc_vars_map = construct_vehicle_obstacle_collision_variables(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            obstacles=obstacles,
        )
        vvc_vars_map = construct_vehicle_vehicle_collision_variables(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicles=vehicles,
        )

        # Add each individual variable map into the final variable map.
        for individual_vars_map in [
            ss_vars_map,
            cs_vars_map,
            voc_vars_map,
            vvc_vars_map,
        ]:
            assert_uniqueness_and_update_mvmip_map(
                mvmip_map_to_be_added=individual_vars_map,
                mvmip_map=vars_map,
            )

    return vars_map

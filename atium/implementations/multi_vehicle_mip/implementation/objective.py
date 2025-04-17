from typing import Sequence, Tuple

import numpy as np

from atium.implementations.multi_vehicle_mip.implementation.custom_types import (
    Solver,
    SolverObjective,
    VehicleControlTrajectoryMap,
    VehicleStateTrajectoryMap,
)
from atium.implementations.multi_vehicle_mip.implementation.definitions import (
    MVMIPObstacle,
    MVMIPOptimizationParams,
    MVMIPResult,
    MVMIPVehicle,
)
from atium.implementations.multi_vehicle_mip.implementation.utils import control_slack_variable_str_from_ids as c_sv
from atium.implementations.multi_vehicle_mip.implementation.utils import control_variable_str_from_ids as c_v
from atium.implementations.multi_vehicle_mip.implementation.utils import state_slack_variable_str_from_ids as s_sv
from atium.implementations.multi_vehicle_mip.implementation.utils import state_variable_str_from_ids as s_v


def construct_objective_for_mvmip(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
) -> SolverObjective:
    nt = mvmip_params.num_time_steps
    objective = solver.Objective()

    for vehicle_id, vehicle in enumerate(vehicles):
        nx = vehicle.dynamics.a_matrix.shape[0]
        nu = vehicle.dynamics.b_matrix.shape[1]

        for time_step_id in range(1, nt + 1):
            for state_id in range(nx):
                # State variable
                var_str = s_sv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                if time_step_id == nt:
                    # For the last step, we use the p cost vector.
                    coefficient = vehicle.optimization_params.p_cost_vector[state_id]
                else:
                    # For all other steps, we use the q cost vector
                    coefficient = vehicle.optimization_params.q_cost_vector[state_id]
                objective.SetCoefficient(
                    solver.LookupVariable(var_str),
                    coefficient,
                )

        for time_step_id in range(nt):
            for control_id in range(nu):
                # Control variable
                var_str = c_sv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    control_id=control_id,
                )
                coefficient = vehicle.optimization_params.r_cost_vector[control_id]
                objective.SetCoefficient(
                    solver.LookupVariable(var_str),
                    coefficient,
                )

    objective.SetMinimization()
    return objective


def vehicle_state_and_control_trajectory_map_from_solver(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
) -> Tuple[VehicleStateTrajectoryMap, VehicleControlTrajectoryMap]:
    nt = mvmip_params.num_time_steps
    vehicle_state_trajectory_map = {}
    vehicle_control_trajectory_map = {}

    for vehicle_id, vehicle in enumerate(vehicles):
        nx = vehicle.dynamics.a_matrix.shape[0]
        nu = vehicle.dynamics.b_matrix.shape[1]
        state_trajectory = np.empty((nt + 1, nx), dtype=np.float64)
        control_trajectory = np.empty((nt, nx), dtype=np.float64)

        state_trajectory[0] = vehicle.dynamics.initial_state

        for time_step_id in range(1, nt + 1):
            for state_id in range(nx):
                var_str = s_v(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                state_trajectory[time_step_id, state_id] = solver.LookupVariable(var_str).solution_value()

        for time_step_id in range(nt):
            for control_id in range(nu):
                var_str = c_v(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    control_id=control_id,
                )
                control_trajectory[time_step_id, control_id] = solver.LookupVariable(var_str).solution_value()

        vehicle_state_trajectory_map[vehicle_id] = state_trajectory.round(
            mvmip_params.result_float_precision,
        )
        vehicle_control_trajectory_map[vehicle_id] = control_trajectory.round(
            mvmip_params.result_float_precision,
        )

    return vehicle_state_trajectory_map, vehicle_control_trajectory_map


def mvmip_result_from_solver(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
    solver_setup_time_s: float,
    solver_solve_time_s: float,
) -> MVMIPResult:
    objective_value = np.round(solver.Objective().Value(), mvmip_params.result_float_precision)

    vst_map, vct_map = vehicle_state_and_control_trajectory_map_from_solver(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
    )

    return MVMIPResult(
        objective_value=objective_value,
        vehicle_state_trajectory_map=vst_map,
        vehicle_control_trajectory_map=vct_map,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
        solver_setup_time_s=solver_setup_time_s,
        solver_solving_time_s=solver_solve_time_s,
    )

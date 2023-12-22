from typing import Sequence

from algorithms.multi_vehicle_mip.implementation.definitions import (
    MVMIPOptimizationParams,
    MVMIPVehicle,
)

from algorithms.multi_vehicle_mip.implementation.utils import (
    state_slack_variable_str_from_ids as s_sv,
    control_slack_variable_str_from_ids as c_sv,
)
from algorithms.multi_vehicle_mip.implementation.custom_types import (
    Solver,
    SolverObjective,
    SolverVariableMap,
)


def construct_objective_for_mvmip(
    solver: Solver,
    vars_map: SolverVariableMap,
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
                assert var_str in vars_map
                if time_step_id == nt:
                    # For the last step, we use the p cost vector.
                    coefficient = vehicle.optimization_params.p_cost_vector[state_id]
                else:
                    # For all other steps, we use the q cost vector
                    coefficient = vehicle.optimization_params.q_cost_vector[state_id]
                objective.SetCoefficient(
                    vars_map[var_str],
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
                assert var_str in vars_map
                coefficient = vehicle.optimization_params.r_cost_vector[control_id]
                objective.SetCoefficient(
                    vars_map[var_str],
                    coefficient,
                )

    objective.SetMinimization()
    return objective

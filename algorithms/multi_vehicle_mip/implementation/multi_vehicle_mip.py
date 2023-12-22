import numpy as np
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
    assert_uniqueness_and_update_cons_map,
    state_variable_str_from_ids as sv,
    state_slack_variable_str_from_ids as ssv,
    control_variable_str_from_ids as cv,
    control_slack_variable_str_from_ids as csv,
    vehicle_obstacle_collision_binary_slack_variable_str_from_ids as obsv,
    vehicle_vehicle_collision_binary_slack_variable_str_from_ids as vbsv,
    state_slack_constraint_var_from_var_strs as ssc,
    control_slack_constraint_var_from_var_strs as csc,
    state_transition_constraint_var_from_var_strs as stc,
    vehicle_obstacle_collision_constraint_var_from_var_strs as occ,
    vehicle_obstacle_collision_binary_constraint_var_from_ids as ocbc,
    vehicle_vehicle_collision_constraint_var_from_var_strs as vcc,
    vehicle_vehicle_collision_binary_constraint_var_from_ids as vcbc,
)

# Types
Solver = pywraplp.Solver
SolverVariable = pywraplp.Variable
SolverVariableMap = Dict[str, SolverVariable]
SolverConstraint = pywraplp.Constraint
SolverConstraintMap = Dict[str, pywraplp.Constraint]


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
) -> SolverVariableMap:

    nt = mvmip_params.num_time_steps

    vars_map = {}
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
                assert var_str not in vars_map
                vars_map[var_str] = solver.NumVar(min_limit, max_limit, var_str)

                # State slack variable
                var_str = ssv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                assert var_str not in vars_map
                vars_map[var_str] = solver.NumVar(
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
                assert var_str not in vars_map
                vars_map[var_str] = solver.NumVar(min_limit, max_limit, var_str)

                # Control slack variable
                var_str = csv(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    control_id=control_id,
                )
                assert var_str not in vars_map
                vars_map[var_str] = solver.NumVar(
                    -solver.infinity(), solver.infinity(), var_str
                )

        # Binary variables for vehicle-vehicle collision constraint variables.
        for time_step_id in range(1, nt + 1):
            for other_vehicle_id in range(vehicle_id + 1, len(vehicles)):
                for var_id in range(4):
                    var_str = vbsv(
                        current_vehicle_id=vehicle_id,
                        other_vehicle_id=other_vehicle_id,
                        time_step_id=time_step_id,
                        var_id=var_id,
                    )
                    assert var_str not in vars_map
                    vars_map[var_str] = solver.IntVar(0, 1, var_str)

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
                        assert var_str not in vars_map
                        vars_map[var_str] = solver.IntVar(0, 1, var_str)
                else:
                    raise NotImplemented(
                        "Only rectangular obstacles have been implemented so far."
                    )

    return vars_map


def construct_state_slack_constraints(
    solver: Solver,
    vars_map: SolverVariableMap,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicle: MVMIPVehicle,
) -> SolverConstraintMap:

    cons_map = {}
    nt = mvmip_params.num_time_steps
    nx = vehicle.dynamics.a_matrix.shape[0]

    # Defining state slack constraints.
    for time_step_id in range(1, nt + 1):
        for state_id in range(nx):
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

            # s_pi - s_pf <= w_pi (For each state index)
            # => -inf <= s_pi - w_pi <= s_pf
            cons_var = ssc(
                state_var_str=s_var_str,
                state_slack_var_str=w_var_str,
                constraint_id=1,
            )
            constraint = solver.Constraint(
                -solver.infinity(),
                vehicle.dynamics.final_state[state_id],
                cons_var,
            )
            constraint.SetCoefficient(vars_map[s_var_str], 1.0)
            constraint.SetCoefficient(vars_map[w_var_str], -1.0)
            assert cons_var not in cons_map
            cons_map[cons_var] = constraint

            # -s_pi + s_pf <= w_pi (For each state index)
            # => -inf <= -s_pi - w_pi <= -s_pf
            cons_var = ssc(
                state_var_str=s_var_str,
                state_slack_var_str=w_var_str,
                constraint_id=2,
            )
            constraint = solver.Constraint(
                -solver.infinity(),
                -vehicle.dynamics.final_state[state_id],
                cons_var,
            )
            constraint.SetCoefficient(vars_map[s_var_str], -1.0)
            constraint.SetCoefficient(vars_map[w_var_str], -1.0)
            assert cons_var not in cons_map
            cons_map[cons_var] = constraint

    return cons_map


def construct_control_slack_constraints(
    solver: Solver,
    vars_map: SolverVariableMap,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicle: MVMIPVehicle,
) -> SolverConstraintMap:

    cons_map = {}
    nt = mvmip_params.num_time_steps
    nu = vehicle.dynamics.b_matrix.shape[1]

    for time_step_id in range(nt):
        for control_id in range(nu):
            u_var_str = cv(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                control_id=control_id,
            )
            v_var_str = csv(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                control_id=control_id,
            )

            # u_pi <= vpi (For each control index)
            # => u_pi - v_pi <= 0
            cons_var = csc(
                control_var_str=u_var_str,
                control_slack_var_str=v_var_str,
                constraint_id=1,
            )
            constraint = solver.Constraint(
                -solver.infinity(),
                0.0,
                cons_var,
            )
            constraint.SetCoefficient(vars_map[u_var_str], 1.0)
            constraint.SetCoefficient(vars_map[v_var_str], -1.0)
            assert cons_var not in cons_map
            cons_map[cons_var] = constraint

            # -u_pi <= vpi (For each control index)
            # => -u_pi - v_pi <= 0
            cons_var = csc(
                control_var_str=u_var_str,
                control_slack_var_str=v_var_str,
                constraint_id=2,
            )
            constraint = solver.Constraint(
                -solver.infinity(),
                0.0,
                cons_var,
            )
            constraint.SetCoefficient(vars_map[u_var_str], -1.0)
            constraint.SetCoefficient(vars_map[v_var_str], -1.0)
            assert cons_var not in cons_map
            cons_map[cons_var] = constraint

    return cons_map


def construct_state_transition_constraints(
    solver: Solver,
    vars_map: SolverVariableMap,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicle: MVMIPVehicle,
) -> SolverConstraintMap:

    cons_map = {}

    nt = mvmip_params.num_time_steps
    nx = vehicle.dynamics.a_matrix.shape[0]
    nu = vehicle.dynamics.b_matrix.shape[1]
    a_mat, b_mat = vehicle.dynamics.a_matrix, vehicle.dynamics.b_matrix
    initial_state = vehicle.dynamics.initial_state

    assert a_mat.shape[1] == nx
    assert initial_state.size == nx

    for time_step_id in range(nt - 1):

        current_time_step_id, next_time_step_id = time_step_id, time_step_id + 1

        # Constraint is of the form:
        # s_p(i+1) = A_p * s_pi + B_p * u_pi
        # s_p(i+1) - A_p * s_pi - B_p * u_pi = 0
        # And for the first step, as s_p0 is not a variable, it is:
        # s_p1 - B_p * u_p0 = A_p * s_p0  As s_p0 is not a variable.
        for next_state_id in range(nx):
            # LHS of the first equation (s_p(i+1))
            next_s_var_str = ssv(
                vehicle_id=vehicle_id,
                time_step_id=next_time_step_id,
                state_id=next_state_id,
            )
            cons_var = stc(
                current_time_step_id=current_time_step_id,
                next_time_step_id=next_time_step_id,
                constraint_id=next_state_id,
            )
            if current_time_step_id == 0:
                # For the first step, we have an equality constraint that is equal to A_p * s_pi
                cons_eq_value = np.dot(a_mat[next_state_id, :], initial_state)
                constraint = solver.Constraint(cons_eq_value, cons_eq_value, cons_var)
            else:
                # For all other steps, the constraint is equal to 0.
                constraint = solver.Constraint(0.0, 0.0, cons_var)
            # Either way the coefficient for s_p(i+1)j is 1.
            constraint.SetCoefficient(vars_map[next_s_var_str], 1.0)

            # In the constraint, each row of s_p(i+1) is a function of all of s_pi and u_pi, hence we need to iterate over nx and nu again.
            # Note that this is only true for time steps > 0 as other s_pi is not a variable.
            if current_time_step_id > 0:
                for current_state_id in range(nx):
                    current_s_var_str = ssv(
                        vehicle_id=vehicle_id,
                        time_step_id=current_time_step_id,
                        state_id=current_state_id,
                    )
                    # Coeffient of s_pij is A_p(row)j Where A_p's row is defined by next_state_id or the s_p(i+1) element under consideration.
                    constraint.SetCoefficient(
                        vars_map[current_s_var_str],
                        -1.0 * a_mat[next_state_id, current_state_id],
                    )

            for current_control_id in range(nu):

                current_u_var_str = cv(
                    vehicle_id=vehicle_id,
                    time_step_id=current_time_step_id,
                    control_id=current_control_id,
                )
                # Coeffient of u_pij is B_p(row)j Where B_p's row is defined by next_state_id or the s_p(i+1) element under consideration.
                constraint.SetCoefficient(
                    vars_map[current_u_var_str],
                    -1.0 * b_mat[next_state_id, current_control_id],
                )

            assert cons_var not in cons_map
            cons_map[cons_var] = constraint

    return cons_map


def construct_constraints_for_mvmip(
    solver: Solver,
    vars_map: SolverVariableMap,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> SolverConstraintMap:

    nt = mvmip_params.num_time_steps

    cons_map = {}

    for vehicle_id, vehicle in enumerate(vehicles):

        # State slack constraints
        ss_cons_map = construct_state_slack_constraints(
            solver=solver,
            vars_map=vars_map,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        assert_uniqueness_and_update_cons_map(
            cons_map_to_be_added=ss_cons_map,
            cons_map=cons_map,
        )

        # Control slack constraints
        cs_cons_map = construct_control_slack_constraints(
            solver=solver,
            vars_map=vars_map,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        assert_uniqueness_and_update_cons_map(
            cons_map_to_be_added=cs_cons_map,
            cons_map=cons_map,
        )

        # State transition constraints
        dyn_cons_map = construct_state_transition_constraints(
            solver=solver,
            vars_map=vars_map,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        assert_uniqueness_and_update_cons_map(
            cons_map_to_be_added=dyn_cons_map,
            cons_map=cons_map,
        )

    return cons_map


def solve_mvmip(
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Sequence[StateTrajectoryArray]:

    # TODO: Check validity of vehicles and obstacles.

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    assert solver is not None, "Solver could not be created."

    vars_map = create_variables_for_mvmip(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(vars_map) == solver.NumVariables()

    cons_map = construct_constraints_for_mvmip(
        solver=solver,
        vars_map=vars_map,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(cons_map) == solver.NumConstraints()

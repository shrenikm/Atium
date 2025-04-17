from typing import Sequence

import numpy as np

from atium.implementations.multi_vehicle_mip.src.custom_types import Solver, SolverConstraintMap
from atium.implementations.multi_vehicle_mip.src.definitions import (
    MVMIPObstacle,
    MVMIPOptimizationParams,
    MVMIPRectangleObstacle,
    MVMIPVehicle,
)
from atium.implementations.multi_vehicle_mip.src.utils import assert_uniqueness_and_update_mvmip_map
from atium.implementations.multi_vehicle_mip.src.utils import control_slack_constraint_var_from_var_strs as c_sc
from atium.implementations.multi_vehicle_mip.src.utils import control_slack_variable_str_from_ids as c_sv
from atium.implementations.multi_vehicle_mip.src.utils import control_variable_str_from_ids as c_v
from atium.implementations.multi_vehicle_mip.src.utils import state_slack_constraint_var_from_var_strs as s_sc
from atium.implementations.multi_vehicle_mip.src.utils import state_slack_variable_str_from_ids as s_sv
from atium.implementations.multi_vehicle_mip.src.utils import state_transition_constraint_var_from_var_strs as st_c
from atium.implementations.multi_vehicle_mip.src.utils import state_variable_str_from_ids as s_v
from atium.implementations.multi_vehicle_mip.src.utils import (
    vehicle_obstacle_collision_binary_constraint_var_from_ids as voc_bc,
)
from atium.implementations.multi_vehicle_mip.src.utils import (
    vehicle_obstacle_collision_binary_slack_variable_str_from_ids as voc_bsv,
)
from atium.implementations.multi_vehicle_mip.src.utils import (
    vehicle_obstacle_collision_constraint_var_from_var_strs as voc_c,
)
from atium.implementations.multi_vehicle_mip.src.utils import (
    vehicle_vehicle_collision_binary_constraint_var_from_ids as vvc_bc,
)
from atium.implementations.multi_vehicle_mip.src.utils import (
    vehicle_vehicle_collision_binary_slack_variable_str_from_ids as vvc_bsv,
)
from atium.implementations.multi_vehicle_mip.src.utils import (
    vehicle_vehicle_collision_constraint_var_from_var_strs as vvc_c,
)


def construct_state_slack_constraints(
    solver: Solver,
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
            s_var_str = s_v(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                state_id=state_id,
            )
            w_var_str = s_sv(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                state_id=state_id,
            )

            # s_pi - s_pf <= w_pi (For each state index)
            # => -inf <= s_pi - w_pi <= s_pf
            cons_var = s_sc(
                state_var_str=s_var_str,
                state_slack_var_str=w_var_str,
                constraint_id=1,
            )
            assert cons_var not in cons_map
            constraint = solver.Constraint(
                -solver.infinity(),
                vehicle.dynamics.final_state[state_id],
                cons_var,
            )
            constraint.SetCoefficient(solver.LookupVariable(s_var_str), 1.0)
            constraint.SetCoefficient(solver.LookupVariable(w_var_str), -1.0)
            cons_map[cons_var] = constraint

            # -s_pi + s_pf <= w_pi (For each state index)
            # => -inf <= -s_pi - w_pi <= -s_pf
            cons_var = s_sc(
                state_var_str=s_var_str,
                state_slack_var_str=w_var_str,
                constraint_id=2,
            )
            assert cons_var not in cons_map
            constraint = solver.Constraint(
                -solver.infinity(),
                -vehicle.dynamics.final_state[state_id],
                cons_var,
            )
            constraint.SetCoefficient(solver.LookupVariable(s_var_str), -1.0)
            constraint.SetCoefficient(solver.LookupVariable(w_var_str), -1.0)
            cons_map[cons_var] = constraint

    return cons_map


def construct_control_slack_constraints(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicle: MVMIPVehicle,
) -> SolverConstraintMap:
    cons_map = {}
    nt = mvmip_params.num_time_steps
    nu = vehicle.dynamics.b_matrix.shape[1]

    for time_step_id in range(nt):
        for control_id in range(nu):
            u_var_str = c_v(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                control_id=control_id,
            )
            v_var_str = c_sv(
                vehicle_id=vehicle_id,
                time_step_id=time_step_id,
                control_id=control_id,
            )

            # u_pi <= vpi (For each control index)
            # => u_pi - v_pi <= 0
            cons_var = c_sc(
                control_var_str=u_var_str,
                control_slack_var_str=v_var_str,
                constraint_id=1,
            )
            assert cons_var not in cons_map
            constraint = solver.Constraint(
                -solver.infinity(),
                0.0,
                cons_var,
            )
            constraint.SetCoefficient(solver.LookupVariable(u_var_str), 1.0)
            constraint.SetCoefficient(solver.LookupVariable(v_var_str), -1.0)
            cons_map[cons_var] = constraint

            # -u_pi <= vpi (For each control index)
            # => -u_pi - v_pi <= 0
            cons_var = c_sc(
                control_var_str=u_var_str,
                control_slack_var_str=v_var_str,
                constraint_id=2,
            )
            assert cons_var not in cons_map
            constraint = solver.Constraint(
                -solver.infinity(),
                0.0,
                cons_var,
            )
            constraint.SetCoefficient(solver.LookupVariable(u_var_str), -1.0)
            constraint.SetCoefficient(solver.LookupVariable(v_var_str), -1.0)
            cons_map[cons_var] = constraint

    return cons_map


def construct_state_transition_constraints(
    solver: Solver,
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

    for time_step_id in range(nt):
        current_time_step_id = time_step_id
        next_time_step_id = time_step_id + 1

        # Constraint is of the form:
        # s_p(i+1) = A_p * s_pi + B_p * u_pi
        # s_p(i+1) - A_p * s_pi - B_p * u_pi = 0
        # And for the first step, as s_p0 is not a variable, it is:
        # s_p1 - B_p * u_p0 = A_p * s_p0  As s_p0 is not a variable.
        for next_state_id in range(nx):
            # LHS of the first equation (s_p(i+1))
            next_s_var_str = s_v(
                vehicle_id=vehicle_id,
                time_step_id=next_time_step_id,
                state_id=next_state_id,
            )
            cons_var = st_c(
                vehicle_id=vehicle_id,
                current_time_step_id=current_time_step_id,
                next_time_step_id=next_time_step_id,
                constraint_id=next_state_id,
            )
            assert cons_var not in cons_map
            if current_time_step_id == 0:
                # For the first step, we have an equality constraint that is equal to A_p(row) * s_pi
                cons_eq_value = np.dot(a_mat[next_state_id, :], initial_state)
            else:
                # For all other steps, the constraint is equal to 0.
                cons_eq_value = 0.0

            constraint = solver.Constraint(cons_eq_value, cons_eq_value, cons_var)
            # Either way the coefficient for s_p(i+1)j is 1.
            constraint.SetCoefficient(solver.LookupVariable(next_s_var_str), 1.0)

            # In the constraint, each row of s_p(i+1) is a function of all of s_pi and u_pi, hence we need to iterate over nx and nu again.
            # Note that this is only true for time steps > 0 as other s_pi is not a variable otherwise (for the first step).
            if current_time_step_id > 0:
                for current_state_id in range(nx):
                    current_s_var_str = s_v(
                        vehicle_id=vehicle_id,
                        time_step_id=current_time_step_id,
                        state_id=current_state_id,
                    )
                    # Coeffient of s_pij is A_p(row)j Where A_p's row is defined by next_state_id or the s_p(i+1) element under consideration.
                    constraint.SetCoefficient(
                        solver.LookupVariable(current_s_var_str),
                        -1.0 * a_mat[next_state_id, current_state_id],
                    )

            for current_control_id in range(nu):
                current_u_var_str = c_v(
                    vehicle_id=vehicle_id,
                    time_step_id=current_time_step_id,
                    control_id=current_control_id,
                )
                # Coeffient of u_pij is B_p(row)j Where B_p's row is defined by next_state_id or the s_p(i+1) element under consideration.
                constraint.SetCoefficient(
                    solver.LookupVariable(current_u_var_str),
                    -1.0 * b_mat[next_state_id, current_control_id],
                )

            cons_map[cons_var] = constraint

    return cons_map


def construct_vehicle_obstacle_collision_constraints(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    obstacles: Sequence[MVMIPObstacle],
) -> SolverConstraintMap:
    cons_map = {}

    nt = mvmip_params.num_time_steps
    dt = mvmip_params.dt

    # Note that we assume the first two values in the state vector are x & y
    for obstacle_id, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, MVMIPRectangleObstacle):
            raise NotImplemented("Only rectangular obstacles have been implemented so far.")
        for time_step_id in range(1, nt + 1):
            # The constraints are of the form:
            # x_pi <= x_cmin + Mt_pci1
            # -x_pi <= -x_cmax + Mt_pci2
            # y_pi <= y_cmin + Mt_pci3
            # -y_pi <= -y_cmax + Mt_pci4

            # state_id to var_id map is as follows:
            # 0 -> 0
            # 0 -> 1
            # 1 -> 2
            # 1 -> 3
            # The first two constraints are on x and the next two are on y.
            # Again assuming that x and y are the first two terms in the state vector.
            for state_id in range(2):
                s_var_str = s_v(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                # First constraint
                # s_pi - Mt_* <= o_min
                t_var_str = voc_bsv(
                    vehicle_id=vehicle_id,
                    obstacle_id=obstacle_id,
                    time_step_id=time_step_id,
                    var_id=state_id * 2,
                )
                cons_var = voc_c(
                    state_var_str=s_var_str,
                    binary_var_str=t_var_str,
                )
                assert cons_var not in cons_map
                min_limits_xy = obstacle.compute_min_limits_xy(
                    time_step_id=time_step_id,
                    num_time_steps=nt,
                    dt=dt,
                )
                constraint = solver.Constraint(
                    -solver.infinity(),
                    min_limits_xy[state_id],
                    cons_var,
                )
                constraint.SetCoefficient(solver.LookupVariable(s_var_str), 1.0)
                constraint.SetCoefficient(solver.LookupVariable(t_var_str), -mvmip_params.M)
                cons_map[cons_var] = constraint

                # Second constraint
                # -s_pi - Mt_* <= -o_max
                t_var_str = voc_bsv(
                    vehicle_id=vehicle_id,
                    obstacle_id=obstacle_id,
                    time_step_id=time_step_id,
                    var_id=state_id * 2 + 1,
                )
                cons_var = voc_c(
                    state_var_str=s_var_str,
                    binary_var_str=t_var_str,
                )
                assert cons_var not in cons_map
                max_limits_xy = obstacle.compute_max_limits_xy(
                    time_step_id=time_step_id,
                    num_time_steps=nt,
                    dt=dt,
                )
                constraint = solver.Constraint(
                    -solver.infinity(),
                    -max_limits_xy[state_id],
                    cons_var,
                )
                constraint.SetCoefficient(solver.LookupVariable(s_var_str), -1.0)
                constraint.SetCoefficient(solver.LookupVariable(t_var_str), -mvmip_params.M)
                cons_map[cons_var] = constraint

            # Constraint for the sum of binary slack variables
            # Sum(t_*) <= 3
            cons_var = voc_bc(
                vehicle_id=vehicle_id,
                obstacle_id=obstacle_id,
                time_step_id=time_step_id,
            )
            assert cons_var not in cons_map
            constraint = solver.Constraint(
                -solver.infinity(),
                3.0,
                cons_var,
            )
            for var_id in range(4):
                t_var_str = voc_bsv(
                    vehicle_id=vehicle_id,
                    obstacle_id=obstacle_id,
                    time_step_id=time_step_id,
                    var_id=var_id,
                )
                constraint.SetCoefficient(solver.LookupVariable(t_var_str), 1.0)
            cons_map[cons_var] = constraint

    return cons_map


def construct_vehicle_vehicle_collision_constraints(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicle_id: int,
    vehicles: Sequence[MVMIPVehicle],
) -> SolverConstraintMap:
    cons_map = {}

    nt = mvmip_params.num_time_steps
    current_vehicle_id = vehicle_id

    for other_vehicle_id in range(vehicle_id + 1, len(vehicles)):
        # Computing dx=dy=d
        current_vehicle_d_m = vehicles[current_vehicle_id].dynamics.clearance_m
        other_vehicle_d_m = vehicles[other_vehicle_id].dynamics.clearance_m
        d_m = current_vehicle_d_m + other_vehicle_d_m

        for time_step_id in range(1, nt + 1):
            # Constraint is of the form
            # x_pi - x_qi >= d_x - Mb_pqi1
            # x_qi - x_pi >= d_x - Mb_pqi2
            # y_pi - y_qi >= d_x - Mb_pqi3
            # y_qi - y_pi >= d_x - Mb_pqi4
            for state_id in range(2):
                s_p_var_str = s_v(
                    vehicle_id=vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                s_q_var_str = s_v(
                    vehicle_id=other_vehicle_id,
                    time_step_id=time_step_id,
                    state_id=state_id,
                )
                # First constraint
                # x_pi - x_qi + Mb_piq1 >= dx
                b_var_str = vvc_bsv(
                    current_vehicle_id=current_vehicle_id,
                    other_vehicle_id=other_vehicle_id,
                    time_step_id=time_step_id,
                    var_id=state_id * 2,
                )
                cons_var = vvc_c(
                    current_state_var_str=s_p_var_str,
                    other_state_var_str=s_q_var_str,
                    binary_var_str=b_var_str,
                )
                assert cons_var not in cons_map
                constraint = solver.Constraint(
                    d_m,
                    solver.infinity(),
                    cons_var,
                )
                constraint.SetCoefficient(solver.LookupVariable(s_p_var_str), 1.0)
                constraint.SetCoefficient(solver.LookupVariable(s_q_var_str), -1.0)
                constraint.SetCoefficient(solver.LookupVariable(b_var_str), mvmip_params.M)
                cons_map[cons_var] = constraint

                # Second constraint
                b_var_str = vvc_bsv(
                    current_vehicle_id=current_vehicle_id,
                    other_vehicle_id=other_vehicle_id,
                    time_step_id=time_step_id,
                    var_id=state_id * 2 + 1,
                )
                cons_var = vvc_c(
                    current_state_var_str=s_p_var_str,
                    other_state_var_str=s_q_var_str,
                    binary_var_str=b_var_str,
                )
                assert cons_var not in cons_map
                constraint = solver.Constraint(
                    d_m,
                    solver.infinity(),
                    cons_var,
                )
                constraint.SetCoefficient(solver.LookupVariable(s_p_var_str), -1.0)
                constraint.SetCoefficient(solver.LookupVariable(s_q_var_str), 1.0)
                constraint.SetCoefficient(solver.LookupVariable(b_var_str), mvmip_params.M)
                cons_map[cons_var] = constraint

            # Constraint for the sum of binary slack variables
            # Sum(b_*) <= 3
            cons_var = vvc_bc(
                current_vehicle_id=current_vehicle_id,
                other_vehicle_id=other_vehicle_id,
                time_step_id=time_step_id,
            )
            assert cons_var not in cons_map
            constraint = solver.Constraint(
                -solver.infinity(),
                3.0,
                cons_var,
            )
            for var_id in range(4):
                b_var_str = vvc_bsv(
                    current_vehicle_id=current_vehicle_id,
                    other_vehicle_id=other_vehicle_id,
                    time_step_id=time_step_id,
                    var_id=var_id,
                )
                constraint.SetCoefficient(solver.LookupVariable(b_var_str), 1.0)
            cons_map[cons_var] = constraint

    return cons_map


def construct_constraints_for_mvmip(
    solver: Solver,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> SolverConstraintMap:
    cons_map = {}

    for vehicle_id, vehicle in enumerate(vehicles):
        # State slack constraints
        ss_cons_map = construct_state_slack_constraints(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        # Control slack constraints
        cs_cons_map = construct_control_slack_constraints(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        # State transition constraints
        st_cons_map = construct_state_transition_constraints(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicle=vehicle,
        )
        voc_cons_map = construct_vehicle_obstacle_collision_constraints(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            obstacles=obstacles,
        )
        vvc_cons_map = construct_vehicle_vehicle_collision_constraints(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicle_id=vehicle_id,
            vehicles=vehicles,
        )

        for individual_cons_map in [
            ss_cons_map,
            cs_cons_map,
            st_cons_map,
            voc_cons_map,
            vvc_cons_map,
        ]:
            assert_uniqueness_and_update_mvmip_map(
                mvmip_map_to_be_added=individual_cons_map,
                mvmip_map=cons_map,
            )

    return cons_map

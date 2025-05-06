"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

from functools import cached_property, partial

import attr
import numpy as np
from pydrake.solvers import (
    CommonSolverOption,
    MathematicalProgram,
    Solve,
    SolverOptions,
)

from atium.core.utils.custom_types import NpVectorNf64
from atium.core.utils.logging_utils import AtiumLogger
from atium.experiments.runito.src.runito_constraints import (
    continuity_constraint_func,
    final_pose_constraint_func,
    final_velocity_constraint_func,
    initial_pose_constraint_func,
    initial_velocity_constraint_func,
    obstacle_constraint_func,
)
from atium.experiments.runito.src.runito_costs import control_cost_func, time_regularization_cost_func
from atium.experiments.runito.src.runito_utils import RunitoInputs, RunitoParams
from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager
from atium.experiments.runito.src.runito_visualization import visualize_runito_result


@attr.define
class Runito:
    manager: RunitoVariableManager

    # Optimization variables
    _prog: MathematicalProgram = attr.ib(init=False)
    _logger: AtiumLogger = attr.ib(init=False)

    @_prog.default
    def _init_prog(self):
        return MathematicalProgram()

    @_logger.default
    def _init_logger(self):
        return AtiumLogger(self.__class__.__name__)

    @cached_property
    def params(self) -> RunitoParams:
        """
        Get the parameters for the Unito problem.
        """
        return self.manager.params

    def setup_optimization_program(self, inputs: RunitoInputs) -> None:
        """
        Initialize the optimization problem.
        """
        self.manager.create_decision_variables(self._prog)

        all_vars = self._prog.decision_variables()
        c_x_vars = self.manager.get_c_x_vars(all_vars)
        c_y_vars = self.manager.get_c_y_vars(all_vars)
        c_theta_vars = self.manager.get_c_theta_vars(all_vars)
        t_vars = self.manager.get_t_vars(all_vars)

        # Costs.
        self._prog.AddCost(
            func=partial(control_cost_func, manager=self.manager),
            vars=all_vars,
            description="Control cost",
        )
        self._prog.AddCost(
            func=partial(time_regularization_cost_func, manager=self.manager),
            vars=t_vars,
            description="Time regularization cost",
        )

        # Constraints.
        c_x_0_vars = self.manager.get_c_x_i_vars(all_vars=all_vars, i=0)
        c_y_0_vars = self.manager.get_c_y_i_vars(all_vars=all_vars, i=0)
        c_theta_0_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=0)
        t_0_var = t_vars[0]
        c_x_f_vars = self.manager.get_c_x_i_vars(all_vars=all_vars, i=self.params.M - 1)
        c_y_f_vars = self.manager.get_c_y_i_vars(all_vars=all_vars, i=self.params.M - 1)
        c_theta_f_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=self.params.M - 1)
        t_f_var = t_vars[-1]

        # Initial pose constraint.
        self._prog.AddConstraint(
            func=partial(
                initial_pose_constraint_func,
                initial_pose=inputs.initial_state_inputs.initial_pose,
                manager=self.manager,
            ),
            lb=np.full(3, -self.params.initial_state_equality_tolerance),
            ub=np.full(3, self.params.initial_state_equality_tolerance),
            vars=np.hstack((c_x_0_vars, c_y_0_vars, c_theta_0_vars)),
            description="Initial pose constraint",
        )

        # Initial velocity constraint.
        self._prog.AddConstraint(
            func=partial(
                initial_velocity_constraint_func,
                initial_velocity=inputs.initial_state_inputs.initial_velocity,
                manager=self.manager,
            ),
            lb=np.full(2, -self.params.initial_state_equality_tolerance),
            ub=np.full(2, self.params.initial_state_equality_tolerance),
            vars=np.hstack((c_x_0_vars, c_y_0_vars, c_theta_0_vars)),
            description="Initial velocity constraint",
        )

        # Final pose constraint.
        self._prog.AddConstraint(
            func=partial(
                final_pose_constraint_func,
                final_pose=inputs.final_state_inputs.final_pose,
                manager=self.manager,
            ),
            lb=np.full(3, -self.params.initial_state_equality_tolerance),
            ub=np.full(3, self.params.initial_state_equality_tolerance),
            vars=np.hstack((c_x_f_vars, c_y_f_vars, c_theta_f_vars, t_f_var)),
            description="Final pose constraint",
        )

        # Final velocity constraint.
        if inputs.final_state_inputs.final_velocity is not None:
            self._prog.AddConstraint(
                func=partial(
                    final_velocity_constraint_func,
                    final_velocity=inputs.final_state_inputs.final_velocity,
                    manager=self.manager,
                ),
                lb=np.full(2, -self.params.initial_state_equality_tolerance),
                ub=np.full(2, self.params.initial_state_equality_tolerance),
                vars=np.hstack((c_x_f_vars, c_y_f_vars, c_theta_f_vars, t_f_var)),
                description="Final velocity constraint",
            )

        # Continuity constraints.
        for derivative in range(self.params.h):
            for i in range(self.params.M - 1):
                prev_c_x_vars = self.manager.get_c_x_i_vars(all_vars=all_vars, i=i)
                prev_c_y_vars = self.manager.get_c_y_i_vars(all_vars=all_vars, i=i)
                prev_c_theta_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=i)
                next_c_x_vars = self.manager.get_c_x_i_vars(all_vars=all_vars, i=i + 1)
                next_c_y_vars = self.manager.get_c_y_i_vars(all_vars=all_vars, i=i + 1)
                next_c_theta_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=i + 1)
                prev_t_var = t_vars[i]
                self._prog.AddConstraint(
                    func=partial(
                        continuity_constraint_func,
                        derivative=derivative,
                        manager=self.manager,
                    ),
                    lb=np.full(3, -self.params.continuity_equality_tolerance),
                    ub=np.full(3, self.params.continuity_equality_tolerance),
                    vars=np.hstack(
                        (
                            prev_c_x_vars,
                            prev_c_y_vars,
                            prev_c_theta_vars,
                            next_c_x_vars,
                            next_c_y_vars,
                            next_c_theta_vars,
                            prev_t_var,
                        )
                    ),
                    description=f"Continuity constraint between segments {i} and {i + 1}, and derivative {derivative}",
                )

        # Obstacle avoidance constraints.
        signed_distance_map = inputs.emap2d.compute_signed_distance_transform()
        # self._prog.AddConstraint(
        #     func=partial(
        #         obstacle_constraint_func,
        #         footprint=inputs.footprint,
        #         emap2d=inputs.emap2d,
        #         signed_distance_map=signed_distance_map,
        #         obstacle_clearance=inputs.obstacle_clearance,
        #         manager=self.manager,
        #     ),
        #     lb=np.full(self.params.M * self.params.n * inputs.footprint.shape[0], 0.0),
        #     ub=np.full(self.params.M * self.params.n * inputs.footprint.shape[0], np.inf),
        #     vars=all_vars,
        #     description="Obstacle avoidance constraint",
        # )

        self._prog.AddBoundingBoxConstraint(
            0.0,
            np.inf,
            t_vars,
        )

    def compute_initial_guess(self, inputs: RunitoInputs) -> NpVectorNf64:
        """
        Compute the initial guess for the optimization problem.
        """
        initial_pose_vector = inputs.initial_state_inputs.initial_pose.to_vector()
        final_pose_vector = inputs.final_state_inputs.final_pose.to_vector()

        distance = np.linalg.norm(final_pose_vector - initial_pose_vector)
        distance_per_segment = distance / self.params.M
        x_delta_per_segment = (final_pose_vector[0] - initial_pose_vector[0]) / self.params.M
        y_delta_per_segment = (final_pose_vector[1] - initial_pose_vector[1]) / self.params.M
        nominal_v = 10.0
        nominal_t = distance_per_segment / nominal_v
        if np.isclose(nominal_t, 0.0):
            nominal_t = 0.1

        # We initialize x, y, theta as a straight line between the start and end points.
        # Note that the polynomial coefficients (first two values) need to be initialized to reflect this linear inteprolation.
        c_x_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        c_y_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        c_theta_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        # For t, we initialize them by a constant value.
        t_initial_guess = nominal_t * np.ones(self.params.M)
        theta_delta_per_segment = (final_pose_vector[2] - initial_pose_vector[2]) / self.params.M

        for i in range(self.params.M):
            c_x_initial_guess[i * 2 * self.params.h] = initial_pose_vector[0] + i * x_delta_per_segment
            c_x_initial_guess[i * 2 * self.params.h + 1] = (
                (i + 1) * distance_per_segment - c_x_initial_guess[i * 2 * self.params.h]
            ) / nominal_t

            c_y_initial_guess[i * 2 * self.params.h] = initial_pose_vector[1] + i * y_delta_per_segment
            c_y_initial_guess[i * 2 * self.params.h + 1] = (
                (i + 1) * distance_per_segment - c_y_initial_guess[i * 2 * self.params.h]
            ) / nominal_t

            c_theta_initial_guess[i * 2 * self.params.h] = initial_pose_vector[0] + i * theta_delta_per_segment
            c_theta_initial_guess[i * 2 * self.params.h + 1] = (
                (i + 1) * theta_delta_per_segment - c_theta_initial_guess[i * 2 * self.params.h]
            ) / nominal_t

        initial_guess = np.hstack((c_x_initial_guess, c_y_initial_guess, c_theta_initial_guess, t_initial_guess))
        print(f"Nominal t: {nominal_t}")
        print(f"X initial guess: {c_x_initial_guess}")
        print(f"Y initial guess: {c_y_initial_guess}")
        print(f"Theta initial guess: {c_theta_initial_guess}")
        print(f"T initial guess: {t_initial_guess}")

        return initial_guess

    def solve(
        self,
        inputs: RunitoInputs,
        initial_guess: NpVectorNf64,
        debug_solver: bool = False,
        visualize_solution: bool = False,
    ) -> None:
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, True)
        res = Solve(
            self._prog,
            initial_guess=initial_guess,
            solver_options=solver_options,
        )

        if debug_solver:
            self._logger.info("=" * 20)
            self._logger.info(f"Solver used: {res.get_solver_id().name()}")
            self._logger.info(f"Success: {res.is_success()}")
            self._logger.info(f"Status: {res.get_solution_result()}")
            self._logger.info(f"Solver solve time: {res.get_solver_details().solve_time}")
            self._logger.info(f"Infeasible constraints: {res.GetInfeasibleConstraintNames(self._prog)}")
            self._logger.info("=" * 20)
            self._logger.info("Solution:")
            self._logger.info("=" * 20)
            self._logger.info(f"c_x: {res.GetSolution(self.manager.get_c_x_vars(self._prog.decision_variables()))}")
            self._logger.info(f"c_y: {res.GetSolution(self.manager.get_c_y_vars(self._prog.decision_variables()))}")
            self._logger.info(
                f"c_theta: {res.GetSolution(self.manager.get_c_theta_vars(self._prog.decision_variables()))}"
            )
            self._logger.info(f"t: {res.GetSolution(self.manager.get_t_vars(self._prog.decision_variables()))}")

        if visualize_solution:
            visualize_runito_result(
                manager=self.manager,
                unito_inputs=inputs,
                all_vars_solution=res.GetSolution(self._prog.decision_variables()),
            )

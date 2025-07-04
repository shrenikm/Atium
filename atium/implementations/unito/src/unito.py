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
from atium.implementations.unito.src.unito_constraints import (
    continuity_constraint_func,
    final_ms_constraint_func,
    final_xy_constraint_func,
    initial_ms_constraint_func,
    obstacle_constraint_func,
)
from atium.implementations.unito.src.unito_costs import control_cost_func, time_regularization_cost_func
from atium.implementations.unito.src.unito_utils import (
    UnitoInputs,
    UnitoParams,
)
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager
from atium.implementations.unito.src.unito_visualization import visualize_unito_result


@attr.define
class Unito:
    manager: UnitoVariableManager

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
    def params(self) -> UnitoParams:
        """
        Get the parameters for the Unito problem.
        """
        return self.manager.params

    @staticmethod
    def start_constraints_callable():
        pass

    def setup_optimization_program(self, inputs: UnitoInputs) -> None:
        """
        Initialize the optimization problem.
        """
        self.manager.create_decision_variables(self._prog)

        all_vars = self._prog.decision_variables()
        c_theta_vars = self.manager.get_c_theta_vars(all_vars)
        c_s_vars = self.manager.get_c_s_vars(all_vars)
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
        c_theta_0_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=0)
        c_s_0_vars = self.manager.get_c_s_i_vars(all_vars=all_vars, i=0)
        t_0_var = t_vars[0]
        c_theta_f_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=self.params.M - 1)
        c_s_f_vars = self.manager.get_c_s_i_vars(all_vars=all_vars, i=self.params.M - 1)
        t_f_var = t_vars[-1]

        for derivative, initial_ms in inputs.initial_state_inputs.initial_ms_map.items():
            # Get the initial state.
            assert derivative <= self.params.h - 1

            # Add the constraints.
            self._prog.AddConstraint(
                func=partial(
                    initial_ms_constraint_func,
                    initial_ms_vector=initial_ms.to_vector(),
                    derivative=derivative,
                    manager=self.manager,
                ),
                lb=np.full(2, -self.params.initial_state_equality_tolerance),
                ub=np.full(2, self.params.initial_state_equality_tolerance),
                vars=np.hstack((c_theta_0_vars, c_s_0_vars)),
                description=f"Initial MS constraint for derivative: {derivative}",
            )

        for derivative, final_ms in inputs.final_state_inputs.final_ms_map.items():
            # Get the final state.
            assert derivative <= self.params.h - 1

            if derivative == 0:
                # For the 0th derivative, Only theta constraints can be added.
                self._prog.AddConstraint(
                    self.manager.compute_sigma_i_exp(
                        c_theta_i_vars=c_theta_f_vars,
                        c_s_i_vars=c_s_f_vars,
                        t_exp=t_f_var,
                    )[0]
                    - final_ms.theta,
                    -self.params.final_state_equality_tolerance,
                    self.params.final_state_equality_tolerance,
                )
            else:
                # Add the constraints.
                self._prog.AddConstraint(
                    func=partial(
                        final_ms_constraint_func,
                        final_ms_vector=final_ms.to_vector(),
                        derivative=derivative,
                        manager=self.manager,
                    ),
                    lb=np.full(2, -self.params.final_state_equality_tolerance),
                    ub=np.full(2, self.params.final_state_equality_tolerance),
                    vars=np.hstack((c_theta_f_vars, c_s_f_vars, t_f_var)),
                    description=f"Final MS constraint for derivative: {derivative}",
                )

        for derivative in range(self.params.h):
            for i in range(self.params.M - 1):
                prev_c_theta_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=i)
                prev_c_s_vars = self.manager.get_c_s_i_vars(all_vars=all_vars, i=i)
                next_c_theta_vars = self.manager.get_c_theta_i_vars(all_vars=all_vars, i=i + 1)
                next_c_s_vars = self.manager.get_c_s_i_vars(all_vars=all_vars, i=i + 1)
                prev_t_var = t_vars[i]
                self._prog.AddConstraint(
                    func=partial(
                        continuity_constraint_func,
                        derivative=derivative,
                        manager=self.manager,
                    ),
                    lb=np.full(2, -self.params.continuity_equality_tolerance),
                    ub=np.full(2, self.params.continuity_equality_tolerance),
                    vars=np.hstack((prev_c_theta_vars, prev_c_s_vars, next_c_theta_vars, next_c_s_vars, prev_t_var)),
                    description=f"Continuity constraint between segments {i} and {i + 1}, and derivative {derivative}",
                )

        # Add the final position constraint.
        self._prog.AddConstraint(
            func=partial(
                final_xy_constraint_func,
                final_xy=inputs.final_state_inputs.final_xy,
                initial_xy=inputs.initial_state_inputs.initial_xy,
                manager=self.manager,
            ),
            lb=np.full(2, -self.params.final_xy_equality_tolerance),
            ub=np.full(2, self.params.final_xy_equality_tolerance),
            vars=all_vars,
            description="Final xy constraint",
        )

        # Obstacle avoidance constraints.
        signed_distance_map = inputs.emap2d.compute_signed_distance_transform()
        self._prog.AddConstraint(
            func=partial(
                obstacle_constraint_func,
                footprint=inputs.footprint,
                emap2d=inputs.emap2d,
                signed_distance_map=signed_distance_map,
                obstacle_clearance=inputs.obstacle_clearance,
                initial_xy=inputs.initial_state_inputs.initial_xy,
                manager=self.manager,
            ),
            lb=np.full(self.params.M * self.params.n * inputs.footprint.shape[0], 0.0),
            ub=np.full(self.params.M * self.params.n * inputs.footprint.shape[0], np.inf),
            vars=all_vars,
            description="Obstacle avoidance constraint",
        )

        self._prog.AddBoundingBoxConstraint(
            0.0,
            np.inf,
            t_vars,
        )

    def compute_initial_guess_old(self, inputs: UnitoInputs) -> NpVectorNf64:
        """
        Compute the initial guess for the optimization problem.
        """

        # If the start theta value is given, we set c_i_theta[0] to that value.
        c_theta_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        if 0 in inputs.initial_state_inputs.initial_ms_map:
            c_theta_initial_guess[0] = inputs.initial_state_inputs.initial_ms_map[0].theta

        # If the start s value is given, we set c_i_s[0] to that value.
        # c_s_initial_guess = np.zeros(2 * params.h * params.M)
        distance = np.linalg.norm(inputs.final_state_inputs.final_xy - inputs.initial_state_inputs.initial_xy)
        c_s_initial_guess = np.linspace(
            0.0,
            distance,
            num=2 * self.params.h * self.params.M,
        )
        if 0 in inputs.initial_state_inputs.initial_ms_map:
            c_s_initial_guess[0] = inputs.initial_state_inputs.initial_ms_map[0].s

        # For t, we initialize them by a constant value.
        t_initial_guess = 1 * np.ones(self.params.M)

        initial_guess = np.hstack((c_theta_initial_guess, c_s_initial_guess, t_initial_guess))

        return initial_guess

    def compute_initial_guess(self, inputs: UnitoInputs) -> NpVectorNf64:
        """
        Compute the initial guess for the optimization problem.
        """

        distance = np.linalg.norm(inputs.final_state_inputs.final_xy - inputs.initial_state_inputs.initial_xy)
        distance_per_segment = distance / self.params.M
        nominal_v = 10.0
        nominal_t = distance_per_segment / nominal_v
        if np.isclose(nominal_t, 0.0):
            nominal_t = 0.1
        print(f"Nominal t: {nominal_t}")

        c_theta_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        c_s_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        # For t, we initialize them by a constant value.
        t_initial_guess = nominal_t * np.ones(self.params.M)

        # If the start theta value and end value is given, we set c_i_theta[0] and c_i_theta[1] such that
        # it makes the interpolated theta value from (start to end) at that segment.
        initial_theta, final_theta = 0.0, 0.0
        # Note that we don't handle the case where the initial theta is given but the final theta well as it becomes hard to define the guess.
        if 0 in inputs.initial_state_inputs.initial_ms_map:
            initial_theta = inputs.initial_state_inputs.initial_ms_map[0].theta
        if 0 in inputs.final_state_inputs.final_ms_map:
            final_theta = inputs.final_state_inputs.final_ms_map[0].theta
        theta_per_segment = (final_theta - initial_theta) / self.params.M

        for i in range(self.params.M):
            c_theta_initial_guess[i * 2 * self.params.h] = initial_theta + i * theta_per_segment
            c_theta_initial_guess[i * 2 * self.params.h + 1] = (
                (i + 1) * theta_per_segment - c_theta_initial_guess[i * 2 * self.params.h]
            ) / nominal_t

        # We initialize s such that it forms a straight line between the start and end points.
        # To do this, we divide the segments into equal lengths and set s_i[0] and s_i[1] so that it forms a straight line
        #  within that segment.
        initial_s = 0.0
        if 0 in inputs.initial_state_inputs.initial_ms_map:
            initial_s = inputs.initial_state_inputs.initial_ms_map[0].s

        for i in range(self.params.M):
            c_s_initial_guess[i * 2 * self.params.h] = initial_s + i * distance_per_segment
            # The first two values correspond to the coefficients of 1 and t
            # So c_s_i[0] + t * c_s_i[1] = distance up until the ith segment = (i + 1) * distance_per_segment
            # => c_s_i[1] = ((i + 1) * distance_per_segment - c_s_i[0]) / t
            c_s_initial_guess[i * 2 * self.params.h + 1] = (
                (i + 1) * distance_per_segment - c_s_initial_guess[i * 2 * self.params.h]
            ) / nominal_t

        initial_guess = np.hstack((c_theta_initial_guess, c_s_initial_guess, t_initial_guess))
        print(f"Theta initial guess: {c_theta_initial_guess}")
        print(f"S initial guess: {c_s_initial_guess}")
        print(f"T initial guess: {t_initial_guess}")

        return initial_guess

    def solve(
        self,
        inputs: UnitoInputs,
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
            self._logger.info(
                f"c_theta: {res.GetSolution(self.manager.get_c_theta_vars(self._prog.decision_variables()))}"
            )
            self._logger.info(f"c_s: {res.GetSolution(self.manager.get_c_s_vars(self._prog.decision_variables()))}")
            self._logger.info(f"t: {res.GetSolution(self.manager.get_t_vars(self._prog.decision_variables()))}")

        if visualize_solution:
            visualize_unito_result(
                manager=self.manager,
                unito_inputs=inputs,
                all_vars_solution=res.GetSolution(self._prog.decision_variables()),
            )

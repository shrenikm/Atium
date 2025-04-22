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
from pydrake.symbolic import Expression

from atium.implementations.unito.src.constraints import (
    continuity_constraint_func,
    final_ms_constraint_func,
    initial_ms_constraint_func,
)
from atium.implementations.unito.src.costs import control_cost_func, time_regularization_cost_func
from atium.implementations.unito.src.unito_utils import (
    UnitoFinalStateInputs,
    UnitoInitialStateInputs,
    UnitoInputs,
    UnitoParams,
)
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


@attr.define
class Unito:
    manager: UnitoVariableManager

    # Optimization variables
    _prog: MathematicalProgram = attr.ib(init=False)

    @_prog.default
    def _init_prog(self):
        return MathematicalProgram()

    @cached_property
    def params(self) -> UnitoParams:
        """
        Get the parameters for the Unito problem.
        """
        return self.manager.params

    def time_regularization_cost_expression(self) -> Expression:
        return self.params.epsilon_t * np.sum(self._var_t)

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
        c_theta_0_vars = manager.get_c_theta_i_vars(all_vars=all_vars, i=0)
        c_s_0_vars = manager.get_c_s_i_vars(all_vars=all_vars, i=0)
        c_theta_f_vars = manager.get_c_theta_i_vars(all_vars=all_vars, i=self.params.M - 1)
        c_s_f_vars = manager.get_c_s_i_vars(all_vars=all_vars, i=self.params.M - 1)

        for derivative, initial_ms_state in inputs.initial_state_inputs.initial_ms_map.items():
            # Get the initial state.
            assert derivative <= self.params.h - 1
            assert initial_ms_state.shape == (2,)

            # Add the constraints.
            self._prog.AddConstraint(
                func=partial(
                    initial_ms_constraint_func,
                    initial_ms_state=initial_ms_state,
                    derivative=derivative,
                    manager=self.manager,
                ),
                lb=np.full(2, -self.params.initial_state_equality_tolerance),
                ub=np.full(2, self.params.initial_state_equality_tolerance),
                vars=np.hstack((c_theta_0_vars, c_s_0_vars)),
                description=f"Initial MS constraint for derivative: {derivative}",
            )

        for derivative, final_ms_state in inputs.final_state_inputs.final_ms_map.items():
            # Get the final state.
            assert derivative <= self.params.h - 1
            assert final_ms_state.shape == (2,)

            lb = final_ms_state - self.params.final_state_equality_tolerance
            ub = final_ms_state + self.params.final_state_equality_tolerance
            if derivative == 0:
                # For the 0th derivative, s constraints cannot be added.
                lb[1] = -np.inf
                ub[1] = np.inf

            # Add the constraints.
            self._prog.AddConstraint(
                func=partial(
                    final_ms_constraint_func,
                    final_ms_state=final_ms_state,
                    derivative=derivative,
                    manager=self.manager,
                ),
                lb=np.full(2, -self.params.final_state_equality_tolerance),
                ub=np.full(2, self.params.final_state_equality_tolerance),
                vars=np.hstack((c_theta_f_vars, c_s_f_vars, t_vars[-1])),
                description=f"Final MS constraint for derivative: {derivative}",
            )

        for derivative in range(self.params.h):
            for i in range(self.params.M - 1):
                prev_c_theta_vars = manager.get_c_theta_i_vars(all_vars=all_vars, i=i)
                prev_c_s_vars = manager.get_c_s_i_vars(all_vars=all_vars, i=i)
                next_c_theta_vars = manager.get_c_theta_i_vars(all_vars=all_vars, i=i + 1)
                next_c_s_vars = manager.get_c_s_i_vars(all_vars=all_vars, i=i + 1)
                prev_t_var = t_vars[i]
                self._prog.AddConstraint(
                    func=partial(
                        continuity_constraint_func,
                        derivative=derivative,
                        manager=self.manager,
                    ),
                    lb=np.zeros(2) - self.params.continuity_equality_tolerance,
                    ub=np.zeros(2) + self.params.continuity_equality_tolerance,
                    vars=np.hstack((prev_c_theta_vars, prev_c_s_vars, next_c_theta_vars, next_c_s_vars, prev_t_var)),
                    description=f"Continuity constraint between segments {i} and {i + 1}, and derivative {derivative}",
                )

        self._prog.AddBoundingBoxConstraint(
            0.0,
            np.inf,
            t_vars,
        )

        print(self._prog)

    def solve(self, inputs: UnitoInputs) -> None:
        c_theta_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        # c_s_initial_guess = np.linspace(0., inputs.final_state_inputs.final_xy[0], 2 * self.params.h * self.params.M)
        c_s_initial_guess = np.zeros(2 * self.params.h * self.params.M)
        t_initial_guess = np.zeros(self.params.M)

        initial_guess = np.hstack((c_theta_initial_guess, c_s_initial_guess, t_initial_guess))
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, True)
        res = Solve(
            self._prog,
            initial_guess=initial_guess,
            solver_options=solver_options,
        )

        print("Solver used:", res.get_solver_id().name())
        print("Success:", res.is_success())
        print("Status:", res.get_solution_result())
        print("SNOPT info:", res.get_solver_details().info)
        print("SNOPT solve time:", res.get_solver_details().solve_time)
        print("Infeasible constraints:", res.GetInfeasibleConstraintNames(self._prog))
        print("c_theta:")
        print(res.GetSolution(self.manager.get_c_theta_vars(self._prog.decision_variables())))
        print("c_s:")
        print(res.GetSolution(self.manager.get_c_s_vars(self._prog.decision_variables())))
        print("t:")
        print(res.GetSolution(self.manager.get_t_vars(self._prog.decision_variables())))


if __name__ == "__main__":
    params = UnitoParams(
        h=3,
        M=3,
        n=4,
        epsilon_t=1e-2,
        W=1e-2 * np.ones((2, 2), dtype=np.float64),
    )
    manager = UnitoVariableManager(params=params)
    unito = Unito(manager=manager)
    initial_state_inputs = UnitoInitialStateInputs(
        initial_ms_map={
            0: np.array([0.0, 0.0]),
            1: np.array([-0.7, 1.1]),
        },
    )
    final_state_inputs = UnitoFinalStateInputs(
        final_ms_map={
            # 0: np.array([np.pi / 4.0, 0.0]),
            # 1: np.array([0.0, 0.0]),
        },
        final_xy=np.array([5.0, 5.0]),
    )
    inputs = UnitoInputs(
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )
    unito.setup_optimization_program(inputs=inputs)
    unito.solve(inputs=inputs)

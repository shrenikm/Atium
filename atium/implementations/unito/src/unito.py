"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

from functools import cached_property, partial

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.symbolic import Expression

from atium.implementations.unito.src.constraints import final_ms_constraint_func, initial_ms_constraint_func
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
                func=partial(initial_ms_constraint_func, derivative=derivative, manager=self.manager),
                lb=initial_ms_state - self.params.initial_state_tolerance,
                ub=initial_ms_state + self.params.initial_state_tolerance,
                vars=np.hstack((c_theta_0_vars, c_s_0_vars)),
                description=f"Initial MS constraint for derivative: {derivative}",
            )

        for derivative, final_ms_state in inputs.final_state_inputs.final_ms_map.items():
            # Get the final state.
            assert derivative <= self.params.h - 1
            assert final_ms_state.shape == (2,)

            lb = final_ms_state - self.params.final_state_tolerance
            ub = final_ms_state + self.params.final_state_tolerance
            if derivative == 0:
                # For the 0th derivative, s constraints cannot be added.
                lb[1] = -np.inf
                ub[1] = np.inf

            # Add the constraints.
            self._prog.AddConstraint(
                func=partial(final_ms_constraint_func, derivative=derivative, manager=self.manager),
                lb=lb,
                ub=ub,
                vars=np.hstack((c_theta_f_vars, c_s_f_vars, t_vars[-1])),
                description=f"Final MS constraint for derivative: {derivative}",
            )

        print(self._prog)

    def solve(self, inputs: UnitoInputs) -> None:
        initial_guess = np.zeros(self._prog.num_vars())
        res = Solve(
            self._prog,
            initial_guess=initial_guess,
        )

        print("Solver used: ", res.get_solver_id().name())
        print("Success: ", res.is_success())
        print("Status: ", res.get_solution_result())


if __name__ == "__main__":
    params = UnitoParams(
        h=3,
        M=3,
        n=4,
        epsilon_t=0.1,
        W=np.ones((2, 2), dtype=np.float64) * 0.1,
    )
    manager = UnitoVariableManager(params=params)
    unito = Unito(manager=manager)
    initial_state_inputs = UnitoInitialStateInputs(
        initial_ms_map={
            0: np.array([0.0, 0.0]),
            # 1: np.array([0.0, 0.0]),
        },
    )
    final_state_inputs = UnitoFinalStateInputs(
        final_ms_map={
            0: np.array([np.pi / 2.0, 1.0]),
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

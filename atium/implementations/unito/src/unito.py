"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

from functools import cached_property, partial

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression

from atium.implementations.unito.src.costs import control_cost_func, time_regularization_cost_func
from atium.implementations.unito.src.unito_utils import UnitoInputs, UnitoParams
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
        for derivative, initial_ms_state in inputs.start_inputs.ms_state_map.items():
            # Get the initial state.
            assert derivative <= 2 * self.params.h - 1
            assert initial_ms_state.shape == (2,)

            # Add the constraints.
            self._prog.AddConstraint(
                func=initial_ms_constraint_func,
                vars=c_0,
                description=f"Initial state constraint for {derivative}",
            )

        # End constraints.

        print(self._prog)


if __name__ == "__main__":
    params = UnitoParams(
        h=5,
        M=3,
        n=4,
        epsilon_t=0.1,
        W=np.ones((2, 2), dtype=np.float64) * 0.1,
    )
    manager = UnitoVariableManager(params=params)
    unito = Unito(manager=manager)
    unito.setup_optimization_program(None)

"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

from functools import cached_property

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression

from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.custom_types import NpMatrix22f64, NpVector2f64, StateDerivativeVector, StateVector
from atium.implementations.unito.src.costs import control_cost_func
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

    def var_time(self, i: int, j: int) -> Expression:
        """
        Get the time value (as an expression) for the ith segment and jth sample point.
        For n sample points, the time value is given by:
        0, T_i/(n-1), 2T_i/(n-1), ..., (n-1)T_i/(n-1)
        where T_i is the time value for the ith segment.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n
        return self._var_t[i] * (j / (self.params.n - 1))

    def basis_vector(self, i: int, j: int, derivative: int = 0) -> np.ndarray:
        """
        Compute the derivative of the basis vector for the time corresponding to the ith segment and jth sample point.
        The derivative is a polynomial of degree 2*h-2.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n
        assert 0 <= derivative < 2 * self.params.h - 1
        t = self.var_time(i, j)
        return np.array(
            [
                np.prod(range(k - derivative + 1, k + 1)) * t ** (k - derivative) if k >= derivative else 0
                for k in range(2 * self.params.h)
            ]
        )

    def c_theta(self, i: int) -> np.ndarray:
        """
        Returns the c_theta vector of variables for the ith segment.
        Each segment has 2*h variables. The ith segment starts at index i*2*h and ends at i*2*h + 2*h = (i+1)*2*h.
        """
        assert 0 <= i < self.params.M
        return self._var_theta[i * 2 * self.params.h : (i + 1) * 2 * self.params.h]

    def c_s(self, i: int) -> np.ndarray:
        """
        Returns the c_s vector of variables for the ith segment.
        Each segment has 2*h variables. The ith segment starts at index i*2*h and ends at i*2*h + 2*h = (i+1)*2*h.
        """
        assert 0 <= i < self.params.M
        return self._var_s[i * 2 * self.params.h : (i + 1) * 2 * self.params.h]

    def c(self, i: int) -> np.ndarray:
        """
        Returns the c vector of variables (c_theta and c_s) for the ith segment.
        """
        assert 0 <= i < self.params.M
        return np.vstack((self.c_theta(i), self.c_s(i)))

    def sigma(self, i: int, j: int, derivative: int = 0) -> np.ndarray:
        """
        Compute the MS trajectory value for the time corresponding to the ith segment and jth sample point.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n

        beta = self.basis_vector(i, j, derivative=derivative)
        theta_i = beta @ self.c_theta(i)
        s_i = beta @ self.c_s(i)
        return np.array([theta_i, s_i])

    def control_cost_expression(self) -> Expression:
        cost = 0.0
        # sigma(i, n-1) = sigma(i + 1, 0)
        # So we only go up to n-2  so that we don't double count.
        for i in range(self.params.M):
            for j in range(self.params.n - 1):
                sigma_i = self.sigma(i, j, derivative=self.params.h)
                cost += sigma_i @ self.params.W @ sigma_i.T

        return cost

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
            func=control_cost_func,
            vars=all_vars,
            description="Control cost",
        )
        self._prog.AddCost(
            func=time_regularization_cost_func,
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
                func=initial_state_constraint_func,
                vars=c_0,
                description=f"Initial state constraint for {derivative}",
            )

        # End constraints.

        print(self._prog)


# print(unito._var_theta)
# print(unito.basis_vector(1, 3))
# print(unito.basis_vector(1, 3, 1))
# for i in range(params.M):
#     print("===")
#     print(unito.c(i))
#     print("===")
#
# print(unito.sigma(1, 3))
# print(unito.sigma(1, 3, 1))

unito.setup_optimization_program()

"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression

from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.custom_types import NpMatrix22f64, NpVector2f64, StateDerivativeVector, StateVector


@attr.define
class Unito:
    params: UnitoParams

    # Optimization variables
    _prog: MathematicalProgram = attr.ib(init=False)
    _var_theta: np.ndarray = attr.ib(init=False)
    _var_s: np.ndarray = attr.ib(init=False)
    _var_t: np.ndarray = attr.ib(init=False)

    @_prog.default
    def _init_prog(self):
        return MathematicalProgram()

    @_var_theta.default
    def _init_var_theta(self):
        """
        Initializes the c_theta variables for the MS trajectory as a 2D array of sze 2*h*M.
        """
        return self._prog.NewContinuousVariables(2 * self.params.h * self.params.M, "c_theta")

    @_var_s.default
    def _init_var_s(self):
        """
        Initializes the c_s variables for the MS trajectory as a 2D array of sze 2*h*M.
        """
        return self._prog.NewContinuousVariables(2 * self.params.h * self.params.M, "c_s")

    @_var_t.default
    def _init_var_t(self):
        """
        Initializes the time variables Ti as a 1D array of size M.
        """
        return self._prog.NewContinuousVariables(self.params.M, "t")

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

    def setup_optimization_program(self):
        """
        Initialize the optimization problem.
        """

        # Costs.
        control_cost = self._prog.AddCost(self.control_cost_expression())
        control_cost.evaluator().set_description("Control cost")

        time_cost = self._prog.AddCost(self.time_regularization_cost_expression())
        time_cost.evaluator().set_description("Time regularization cost")

        # Constraints.

        # Start constraints.
        self._prog.AddConstraint

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

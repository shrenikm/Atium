"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression

from atium.core.utils.custom_types import NpMatrix22f64


@attr.frozen
class UnitoParams:
    """
    Parameters for Unito.
    Includes problem formulation and optimization params.
    """

    # Basis beta will be of degree 2*h-1
    h: int
    # Number of segments
    M: int
    # Number of sampling intervals
    n: int

    # Costs
    epsilon_t: float
    W: NpMatrix22f64


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

    def basis_vector(self, i: int, j: int, degree: int = 0) -> np.ndarray:
        """
        Compute the derivative of the basis vector for the time corresponding to the ith segment and jth sample point.
        The derivative is a polynomial of degree 2*h-2.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n
        assert 0 <= degree < 2 * self.params.h - 1
        t = self.var_time(i, j)
        return np.array(
            [
                np.prod(range(k - degree + 1, k + 1)) * t ** (k - degree) if k >= degree else 0
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

    def sigma(self, i: int, j: int, degree: int = 0) -> np.ndarray:
        """
        Compute the MS trajectory value for the time corresponding to the ith segment and jth sample point.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n

        beta = self.basis_vector(i, j, degree=degree)
        theta_i = beta @ self.c_theta(i)
        s_i = beta @ self.c_s(i)
        return np.array([theta_i, s_i])

    def f(self, v):
        """
        Cost function.
        """
        return 0

    def setup_optimization_program(self):
        """
        Initialize the optimization problem.
        """

        def f(x):
            """
            Cost function.
            """
            return 0

        print(self._var_theta[0])
        self._prog.AddCost(self.f, np.array([self._var_theta[0]]))


params = UnitoParams(
    h=2,
    M=3,
    n=4,
    epsilon_t=0.1,
    W=np.ones((2, 2), dtype=np.float64) * 0.1,
)
unito = Unito(params)
print(unito._var_theta)
print(unito.basis_vector(1, 3))
print(unito.basis_vector(1, 3, 1))
for i in range(params.M):
    print("===")
    print(unito.c(i))
    print("===")

print(unito.sigma(1, 3))
print(unito.sigma(1, 3, 1))

unito.setup_optimization_program()

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
    _var_c: np.ndarray = attr.ib(init=False)
    _var_t: np.ndarray = attr.ib(init=False)

    @_prog.default
    def _init_prog(self):
        return MathematicalProgram()

    @_var_c.default
    def _init_var_c(self):
        """
        Initializes the c_theta and c_s variables for the MS trajectory as a 2D array of sze (2*h*M, 2).
        """
        return self._prog.NewContinuousVariables(2 * self.params.h * self.params.M, 2, "c")

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

    def basis_vector(self, i: int, j: int) -> np.ndarray:
        """
        Compute the basis vector for the time corresponding to the ith segment and jth sample point.
        The basis vector is a polynomial of degree 2*h-1.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n
        t = self.var_time(i, j)
        return np.array([t**k for k in range(2 * self.params.h)])

    def c_theta(self, i: int) -> np.ndarray:
        """
        Returns the c_theta vector of variables for the ith segment.
        Each segment has 2*h variables. The ith segment starts at index i*2*h and ends at i*2*h + 2*h = (i+1)*2*h.
        """
        assert 0 <= i < self.params.M
        return self._var_c[i * 2 * self.params.h : (i + 1) * 2 * self.params.h, 0]

    def c_s(self, i: int) -> np.ndarray:
        """
        Returns the c_s vector of variables for the ith segment.
        Each segment has 2*h variables. The ith segment starts at index i*2*h and ends at i*2*h + 2*h = (i+1)*2*h.
        """
        assert 0 <= i < self.params.M
        return self._var_c[i * 2 * self.params.h : (i + 1) * 2 * self.params.h, 1]

    def c(self, i: int) -> np.ndarray:
        """
        Returns the c vector of variables (c_theta and c_s) for the ith segment.
        """
        assert 0 <= i < self.params.M
        return self._var_c[i * 2 * self.params.h : (i + 1) * 2 * self.params.h, :]

    def sigma(self, i: int, j: int) -> np.ndarray:
        """
        Compute the MS trajectory value for the time corresponding to the ith segment and jth sample point.
        """
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n

        beta = self.basis_vector(i, j)
        theta_i = beta @ self.c_theta(i)
        s_i = beta @ self.c_s(i)
        return np.array([theta_i, s_i])

    def setup_optimization_program(self):
        """
        Initialize the optimization problem.
        """


params = UnitoParams(
    h=2,
    M=3,
    n=4,
    epsilon_t=0.1,
    W=np.ones((2, 2), dtype=np.float64) * 0.1,
)
unito = Unito(params)
print(unito._var_c)
for i in range(params.M):
    print("===")
    print(unito.c(i))
    print("===")

print(unito.sigma(1, 3))

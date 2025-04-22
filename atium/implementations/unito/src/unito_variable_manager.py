import math

import attr
import numpy as np
import pydrake.math as dmath
from pydrake.autodiffutils import AutoDiffXd
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression, Variable

from atium.implementations.unito.src.unito_utils import UnitoParams


@attr.define
class UnitoVariableManager:
    VARS_C_THETA_NAME = "c_theta"
    VARS_C_S_NAME = "c_s"
    VARS_T_NAME = "t"

    params: UnitoParams

    def create_decision_variables(
        self,
        prog: MathematicalProgram,
    ) -> None:
        prog.NewContinuousVariables(2 * self.params.h * self.params.M, self.VARS_C_THETA_NAME)
        prog.NewContinuousVariables(2 * self.params.h * self.params.M, self.VARS_C_S_NAME)
        prog.NewContinuousVariables(self.params.M, self.VARS_T_NAME)

    def get_c_theta_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_theta variables from the decision variables.
        """
        return all_vars[: 2 * self.params.h * self.params.M]

    def get_c_s_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_s variables from the decision variables.
        """
        return all_vars[2 * self.params.h * self.params.M : 4 * self.params.h * self.params.M]

    def get_t_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the t variables from the decision variables.
        """
        return all_vars[4 * self.params.h * self.params.M :]

    def get_c_theta_i_vars(self, all_vars: np.ndarray, i: int) -> np.ndarray:
        """
        Get the c_theta variables corresponding to the ith segment.
        Each segment has 2*h variables. The ith segment starts at index i*2*h and ends at i*2*h + 2*h = (i+1)*2*h.
        """
        assert 0 <= i < self.params.M
        return all_vars[i * 2 * self.params.h : (i + 1) * 2 * self.params.h]

    def get_c_s_i_vars(self, all_vars: np.ndarray, i: int) -> np.ndarray:
        """
        Get the c_s variables corresponding to the ith segment.
        Each segment has 2*h variables. The ith segment starts at index i*2*h and ends at i*2*h + 2*h = (i+1)*2*h.
        """
        assert 0 <= i < self.params.M
        offset = 2 * self.params.h * self.params.M
        return all_vars[offset + i * 2 * self.params.h : offset + (i + 1) * 2 * self.params.h]

    def get_t_i_var(self, all_vars: np.ndarray, i: int) -> Variable:
        """
        Get the value of t for the ith segment.
        """
        assert 0 <= i < self.params.M
        offset = 4 * self.params.h * self.params.M
        return all_vars[offset + i]

    def get_t_ij_exp(self, t_vars: np.ndarray, i: int, j: int) -> Expression | float:
        assert 0 <= i < self.params.M
        assert 0 <= j < self.params.n
        return j * t_vars[i] / (self.params.n - 1)

    def get_basis_vector_ij_exp(self, t_ij_exp: Expression | float, derivative: int = 0) -> np.ndarray:
        """
        Compute the derivative of the basis vector for the time corresponding to the ith segment and jth sample point.
        The derivative is a polynomial of degree 2*h-2.
        """
        assert 0 <= derivative < 2 * self.params.h - 1
        basis = np.zeros((2 * self.params.h,), dtype=type(t_ij_exp))
        for k in range(2 * self.params.h):
            scalar = np.prod(range(k - derivative + 1, k + 1))
            if k == derivative:
                basis[k] = scalar
            elif k > derivative:
                basis[k] = scalar * t_ij_exp ** (k - derivative)
            else:
                basis[k] = 0.0
        return basis

    def get_sigma_ij_exp(
        self,
        c_theta_i_vars: np.ndarray,
        c_s_i_vars: np.ndarray,
        t_ij_exp: Expression | float,
        derivative: int = 0,
    ) -> np.ndarray:
        beta = self.get_basis_vector_ij_exp(t_ij_exp=t_ij_exp, derivative=derivative)
        theta_i = beta @ c_theta_i_vars
        s_i = beta @ c_s_i_vars
        return np.array([theta_i, s_i])

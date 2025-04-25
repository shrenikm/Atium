import math

import attr
import numpy as np
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

    def compute_t_ijl_exp(self, t_i_var: Variable, j: int, l: int) -> float | Expression:  # noqa: E741
        assert 0 <= j < self.params.n
        # Currently we only allow l to be 0, 1, or 2 for the 3 Simpson points.
        assert l in (0, 1, 2)

        if l == 0:
            return j * t_i_var / (self.params.n)
        elif l == 1:
            return (j + 0.5) * t_i_var / (self.params.n)
        else:
            return (j + 1) * t_i_var / (self.params.n)

    def compute_basis_vector_exp(self, t_exp: float | Expression, derivative: int = 0) -> np.ndarray:
        """
        Compute the derivative of the basis vector for the time corresponding to the ith segment and jth sample point.
        The derivative is a polynomial of degree 2*h-2.
        """
        assert 0 <= derivative < 2 * self.params.h - 1
        basis = np.zeros((2 * self.params.h,), dtype=type(t_exp))
        for k in range(2 * self.params.h):
            scalar = np.prod(range(k - derivative + 1, k + 1))
            if k == derivative:
                basis[k] = scalar
            elif k > derivative:
                basis[k] = scalar * t_exp ** (k - derivative)
            else:
                basis[k] = 0.0
        return basis

    def compute_sigma_i_exp(
        self,
        c_theta_i_vars: np.ndarray,
        c_s_i_vars: np.ndarray,
        t_exp: float | Variable | Expression,
        derivative: int = 0,
    ) -> np.ndarray:
        beta = self.compute_basis_vector_exp(t_exp=t_exp, derivative=derivative)
        theta_i = beta @ c_theta_i_vars
        s_i = beta @ c_s_i_vars
        return np.array([theta_i, s_i])

    def compute_x_ijl_exp(
        self,
        c_theta_i_vars: np.ndarray,
        c_s_i_vars: np.ndarray,
        t_ijl_exp: float | Expression,
    ) -> float | Expression:
        """
        Computes x_ijl as s_dot(T_ijl) * cos(theta_ijl)
        """
        beta = self.compute_basis_vector_exp(
            t_exp=t_ijl_exp,
        )
        beta_dot = self.compute_basis_vector_exp(
            t_exp=t_ijl_exp,
            derivative=1,
        )
        s_dot_exp = beta_dot @ c_s_i_vars
        theta_exp = beta @ c_theta_i_vars
        return s_dot_exp * np.cos(theta_exp)

    def compute_y_ijl_exp(
        self,
        c_theta_i_vars: np.ndarray,
        c_s_i_vars: np.ndarray,
        t_ijl_exp: float | Expression,
    ) -> float | Expression:
        """
        Computes y_ijl as s_dot(T_ijl) * sin(theta_ijl)
        """
        beta = self.compute_basis_vector_exp(
            t_exp=t_ijl_exp,
        )
        beta_dot = self.compute_basis_vector_exp(
            t_exp=t_ijl_exp,
            derivative=1,
        )
        s_dot_exp = beta_dot @ c_s_i_vars
        theta_exp = beta @ c_theta_i_vars
        return s_dot_exp * np.sin(theta_exp)

    def compute_x_i_bar_exp(
        self,
        c_theta_i_vars: np.ndarray,
        c_s_i_vars: np.ndarray,
        t_i_var: float | Variable,
    ) -> float | Expression:
        """
        Computes
        x_i_bar = (Ti / 6n) * sum_j(x_ij0 + 4x_ij1 + x_ij2)
        where x_ij0, x_ij1, x_ij2 are the three sampling points of the nth interval of the
        ith sampling point.
        """
        simpsons_sum = 0.0
        for j in range(self.params.n):
            x_ij0 = self.compute_x_ijl_exp(
                c_theta_i_vars=c_theta_i_vars,
                c_s_i_vars=c_s_i_vars,
                t_ijl_exp=self.compute_t_ijl_exp(t_i_var=t_i_var, j=j, l=0),
            )
            x_ij1 = self.compute_x_ijl_exp(
                c_theta_i_vars=c_theta_i_vars,
                c_s_i_vars=c_s_i_vars,
                t_ijl_exp=self.compute_t_ijl_exp(t_i_var=t_i_var, j=j, l=1),
            )
            x_ij2 = self.compute_x_ijl_exp(
                c_theta_i_vars=c_theta_i_vars,
                c_s_i_vars=c_s_i_vars,
                t_ijl_exp=self.compute_t_ijl_exp(t_i_var=t_i_var, j=2, l=2),
            )
            simpsons_sum += x_ij0 + 4 * x_ij1 + x_ij2
        return (t_i_var / (6 * self.params.n)) * simpsons_sum

    def compute_y_i_bar_exp(
        self,
        c_theta_i_vars: np.ndarray,
        c_s_i_vars: np.ndarray,
        t_i_var: float | Variable,
    ) -> float | Expression:
        """
        Computes
        y_i_bar = (Ti / 6n) * sum_j(y_ij0 + 4y_ij1 + y_ij2)
        where y_ij0, y_ij1, y_ij2 are the three sampling points of the nth interval of the
        ith sampling point.
        """
        simpsons_sum = 0.0
        for j in range(self.params.n):
            y_ij0 = self.compute_y_ijl_exp(
                c_theta_i_vars=c_theta_i_vars,
                c_s_i_vars=c_s_i_vars,
                t_ijl_exp=self.compute_t_ijl_exp(t_i_var=t_i_var, j=j, l=0),
            )
            y_ij1 = self.compute_y_ijl_exp(
                c_theta_i_vars=c_theta_i_vars,
                c_s_i_vars=c_s_i_vars,
                t_ijl_exp=self.compute_t_ijl_exp(t_i_var=t_i_var, j=j, l=1),
            )
            y_ij2 = self.compute_y_ijl_exp(
                c_theta_i_vars=c_theta_i_vars,
                c_s_i_vars=c_s_i_vars,
                t_ijl_exp=self.compute_t_ijl_exp(t_i_var=t_i_var, j=j, l=2),
            )
            simpsons_sum += y_ij0 + 4 * y_ij1 + y_ij2
        return (t_i_var / (6 * self.params.n)) * simpsons_sum

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression, Variable

from atium.experiments.runito.src.runito_utils import RunitoParams


@attr.define
class RunitoVariableManager:
    VARS_C_X_NAME = "c_x"
    VARS_C_Y_NAME = "c_y"
    VARS_C_THETA_NAME = "c_theta"
    VARS_T_NAME = "t"

    params: RunitoParams

    def create_decision_variables(
        self,
        prog: MathematicalProgram,
    ) -> None:
        prog.NewContinuousVariables(2 * self.params.h * self.params.M, self.VARS_C_X_NAME)
        prog.NewContinuousVariables(2 * self.params.h * self.params.M, self.VARS_C_Y_NAME)
        prog.NewContinuousVariables(2 * self.params.h * self.params.M, self.VARS_C_THETA_NAME)
        prog.NewContinuousVariables(self.params.M, self.VARS_T_NAME)

    def get_c_x_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_x variables from the decision variables.
        """
        return all_vars[: 2 * self.params.h * self.params.M]

    def get_c_y_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_y variables from the decision variables.
        """
        return all_vars[2 * self.params.h * self.params.M : 4 * self.params.h * self.params.M]

    def get_c_theta_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_theta variables from the decision variables.
        """
        return all_vars[4 * self.params.h * self.params.M : 6 * self.params.h * self.params.M]

    def get_t_vars(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the t variables from the decision variables.
        """
        return all_vars[6 * self.params.h * self.params.M :]

    def get_c_x_i_vars(self, all_vars: np.ndarray, i: int) -> np.ndarray:
        """
        Get the c_x variables corresponding to the ith segment.
        Each segment has 2*h variables. The ith segment starts at index offset + i*2*h and ends at offset + i*2*h + 2*h = offset + (i+1)*2*h.
        The offset is 0 in this case.
        """
        assert 0 <= i < self.params.M
        offset = 0
        return all_vars[offset + i * 2 * self.params.h : offset + (i + 1) * 2 * self.params.h]

    def get_c_y_i_vars(self, all_vars: np.ndarray, i: int) -> np.ndarray:
        """
        Get the c_y variables corresponding to the ith segment.
        Each segment has 2*h variables. The ith segment starts at index offset + i*2*h and ends at offset + i*2*h + 2*h = offset + (i+1)*2*h.
        The offset is 2*h*M in this case as c_x appears before c_y.
        """
        assert 0 <= i < self.params.M
        offset = 2 * self.params.h * self.params.M
        return all_vars[offset + i * 2 * self.params.h : offset + (i + 1) * 2 * self.params.h]

    def get_c_theta_i_vars(self, all_vars: np.ndarray, i: int) -> np.ndarray:
        """
        Get the c_theta variables corresponding to the ith segment.
        Each segment has 2*h variables. The ith segment starts at index offset + i*2*h and ends at offset + i*2*h + 2*h = offset + (i+1)*2*h.
        The offset is 4*h*M in this case as c_x and c_y appear before c_theta.
        """
        assert 0 <= i < self.params.M
        offset = 4 * self.params.h * self.params.M
        return all_vars[offset + i * 2 * self.params.h : offset + (i + 1) * 2 * self.params.h]

    def get_t_i_var(self, all_vars: np.ndarray, i: int) -> Variable:
        """
        Get the value of t for the ith segment.
        """
        assert 0 <= i < self.params.M
        offset = 6 * self.params.h * self.params.M
        return all_vars[offset + i]

    def compute_t_ijl_exp(self, t_i_var: Variable, j: int, l: int) -> float | Expression:  # noqa: E741
        """
        Expression for t at the ith segment, jth sampling interval and lth sampling point.
        """
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
        c_x_i_vars: np.ndarray,
        c_y_i_vars: np.ndarray,
        c_theta_i_vars: np.ndarray,
        t_exp: float | Variable | Expression,
        derivative: int = 0,
    ) -> np.ndarray:
        """
        Sigma_i = [ x_i, y_i, theta_i ]
        """
        beta = self.compute_basis_vector_exp(t_exp=t_exp, derivative=derivative)
        x_i = beta @ c_x_i_vars
        y_i = beta @ c_y_i_vars
        theta_i = beta @ c_theta_i_vars
        return np.array([x_i, y_i, theta_i])

    def compute_gamma_i_exp(
        self,
        c_x_i_vars: np.ndarray,
        c_y_i_vars: np.ndarray,
        c_theta_i_vars: np.ndarray,
        t_exp: float | Variable | Expression,
    ) -> np.ndarray:
        """
        Sigma_i = [ linear_velocity_i, angular_velocity_i ]
        """
        sigma_i_dot = self.compute_sigma_i_exp(
            c_x_i_vars=c_x_i_vars,
            c_y_i_vars=c_y_i_vars,
            c_theta_i_vars=c_theta_i_vars,
            t_exp=t_exp,
            derivative=1,
        )

        # TODO: We get a divide by zero error here if both xdot and ydot end up being 0.
        # So we add an epsilon.
        epsilon = 0
        if sigma_i_dot[0] == 0 and sigma_i_dot[1] == 0:
            epsilon = 1e-6
        v = np.sqrt(sigma_i_dot[0] ** 2 + sigma_i_dot[1] ** 2 + epsilon)
        return np.array(
            [
                v,
                sigma_i_dot[2],
            ]
        )

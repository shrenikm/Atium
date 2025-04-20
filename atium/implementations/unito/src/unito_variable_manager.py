import attr
import numpy as np
from pydrake.solvers import MathematicalProgram

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

    def get_vars_c_theta(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_theta variables from the decision variables.
        """
        return all_vars[: 2 * self.params.h * self.params.M]

    def get_vars_c_s(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the c_s variables from the decision variables.
        """
        return all_vars[2 * self.params.h * self.params.M : 4 * self.params.h * self.params.M]

    def get_vars_t(self, all_vars: np.ndarray) -> np.ndarray:
        """
        Get the t variables from the decision variables.
        """
        return all_vars[4 * self.params.h * self.params.M :]

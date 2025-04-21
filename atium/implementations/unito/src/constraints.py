import numpy as np

from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def initial_ms_constraint_func(
    c_0_vars: np.ndarray,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert c_0_vars.shape == (4 * manager.params.h,)
    c_theta_0_vars = c_0_vars[: 2 * manager.params.h]
    c_s_0_vars = c_0_vars[2 * manager.params.h :]

    return manager.get_sigma_ij_exp(
        c_theta_i_vars=c_theta_0_vars,
        c_s_i_vars=c_s_0_vars,
        t_ij_exp=0.0,
        derivative=derivative,
    )

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


def final_ms_constraint_func(
    c_f_t_f_vars: np.ndarray,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert c_f_t_f_vars.shape == (4 * manager.params.h + 1,)

    c_f_vars = c_f_t_f_vars[:-1]
    t_f_var = c_f_t_f_vars[-1]

    c_theta_f_vars = c_f_vars[: 2 * manager.params.h]
    c_s_f_vars = c_f_vars[2 * manager.params.h :]

    return manager.get_sigma_ij_exp(
        c_theta_i_vars=c_theta_f_vars,
        c_s_i_vars=c_s_f_vars,
        t_ij_exp=t_f_var,
        derivative=derivative,
    )

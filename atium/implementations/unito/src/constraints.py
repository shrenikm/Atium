import numpy as np

from atium.core.utils.custom_types import StateVector
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def initial_ms_constraint_func(
    func_vars: np.ndarray,
    initial_ms_state: StateVector,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * manager.params.h,)

    c_theta_0_vars = func_vars[: 2 * manager.params.h]
    c_s_0_vars = func_vars[2 * manager.params.h :]

    sigma_ij = manager.compute_sigma_ij_exp(
        c_theta_i_vars=c_theta_0_vars,
        c_s_i_vars=c_s_0_vars,
        t_ij_exp=0.0,
        derivative=derivative,
    )
    return sigma_ij - initial_ms_state


def final_ms_constraint_func(
    func_vars: np.ndarray,
    final_ms_state: StateVector,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * manager.params.h + 1,)

    c_f_vars = func_vars[:-1]
    t_f_var = func_vars[-1]

    assert c_f_vars.shape == (4 * manager.params.h,)

    c_theta_f_vars = c_f_vars[: 2 * manager.params.h]
    c_s_f_vars = c_f_vars[2 * manager.params.h :]

    sigma_ij = manager.compute_sigma_ij_exp(
        c_theta_i_vars=c_theta_f_vars,
        c_s_i_vars=c_s_f_vars,
        t_ij_exp=t_f_var,
        derivative=derivative,
    )
    return sigma_ij - final_ms_state


def continuity_constraint_func(
    func_vars: np.ndarray,
    derivative: int,
    manager: UnitoVariableManager,
) -> None:
    assert func_vars.shape == (4 * 2 * manager.params.h + 1,)

    prev_c_vars = func_vars[: 4 * manager.params.h]
    next_c_vars = func_vars[4 * manager.params.h : 4 * 2 * manager.params.h]
    prev_t_var = func_vars[-1]
    prev_c_theta_vars = prev_c_vars[: 2 * manager.params.h]
    prev_c_s_vars = prev_c_vars[2 * manager.params.h :]
    next_c_theta_vars = next_c_vars[: 2 * manager.params.h]
    next_c_s_vars = next_c_vars[2 * manager.params.h :]

    prev_sigma = manager.compute_sigma_ij_exp(
        c_theta_i_vars=prev_c_theta_vars,
        c_s_i_vars=prev_c_s_vars,
        t_ij_exp=prev_t_var,
        derivative=derivative,
    )
    next_sigma = manager.compute_sigma_ij_exp(
        c_theta_i_vars=next_c_theta_vars,
        c_s_i_vars=next_c_s_vars,
        t_ij_exp=0.0,
        derivative=derivative,
    )
    return prev_sigma - next_sigma


def final_xy_constraint_func(
    func_vars: np.ndarray,
    final_xy: StateVector,
    manager: UnitoVariableManager,
) -> None:
    assert func_vars.shape == (4 * manager.params.h * manager.params.M + manager.params.M,)

    for i in range(manager.params.M):
        x_bar = manager.compute
